"""
Train a diffusion model for recommendation
"""

import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import models.gaussian_diffusion as gd
from models.DNN import GDN
import evaluate_utils
import data_utils

import random
random_seed = 1997
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets_converted/', help='load data path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('--eval_interval', type=int, default=20, help='validate every N epochs')
parser.add_argument('--patience', type=int, default=5, help='early stop after N consecutive non-improving validations')
parser.add_argument('--topN', type=str, default='[1, 5, 10, 20]')
parser.add_argument('--disable_history_mask', action='store_true',default='true', help='disable masking historical interactions during evaluation')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for the model
parser.add_argument('--time_type', type=str, default='add', help='cat or add')
parser.add_argument('--graph_layers', type=int, default=1, help='the nums layer for the GNN')
parser.add_argument('--graph_views', type=int, default=1, help='the nums views for the GNN')
parser.add_argument('--mlp_hidden_dims', type=str, default='[128]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--sample_style', type=str, default='uniform', help='importance/uniform/fully')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=32, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='truncated-linear', help='the schedule for noise generating: truncated-linear/linear-var/linear/cosine/binomial')
parser.add_argument('--noise_scale', type=float, default=1.0, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
topN = eval(args.topN)
metric_keys = [f'HR@{k}' for k in topN] + [f'NDCG@{k}' for k in topN]
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')

train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item, g = data_utils.data_load(train_path, valid_path, test_path, args.w_min, args.w_max)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data_ori + valid_y_data

print('data ready.')

valid_pos_counts = np.asarray(valid_y_data.getnnz(axis=1)).reshape(-1)
is_leave_one_out = np.all((valid_pos_counts == 0) | (valid_pos_counts == 1))
print(f"Evaluation protocol: {'DiffuSR-like single-label (LOO)' if is_leave_one_out else 'G-Diff multi-label'}")


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale,
                                 args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
if eval(args.mlp_hidden_dims):
    mlp_dims = [n_item] + eval(args.mlp_hidden_dims) + [n_item]
else:
    mlp_dims = [n_item, n_item]
model = GDN(mlp_dims, args.emb_size, g, args.graph_layers, norm=args.norm, dropout=args.drop_out).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    if is_leave_one_out:
        metrics_dict = {f'HR@{k}': [] for k in topN}
        metrics_dict.update({f'NDCG@{k}': [] for k in topN})

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            user_indices = e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]
            his_data = mask_his[user_indices]
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
            if not args.disable_history_mask:
                prediction[his_data.nonzero()] = -np.inf

            if is_leave_one_out:
                valid_rows, labels = [], []
                for local_idx, user_idx in enumerate(user_indices):
                    gt = target_items[user_idx]
                    if len(gt) == 0:
                        continue
                    valid_rows.append(local_idx)
                    labels.append(gt[0])
                if len(labels) == 0:
                    continue
                labels = torch.tensor(labels, dtype=torch.long, device=prediction.device).view(-1, 1)
                batch_metrics = evaluate_utils.hrs_and_ndcgs_k(prediction[valid_rows], labels, topN)
                for key, value in batch_metrics.items():
                    metrics_dict[key].append(value)
            else:
                _, indices = torch.topk(prediction, topN[-1], dim=-1)
                predict_items.extend(indices.cpu().numpy().tolist())

    if is_leave_one_out:
        metrics = {}
        for key, values in metrics_dict.items():
            metrics[key] = round(float(np.mean(values)), 4) if len(values) > 0 else 0.0
        return metrics

    metrics = evaluate_utils.hrs_and_ndcgs_k_multi(target_items, predict_items, topN)
    return {k: round(v, 4) for k, v in metrics.items()}

best_epoch = 0
best_metrics = {k: -100.0 for k in metric_keys}
best_results, best_test_results = None, None
no_improve_evals = 0
print("Start training...")
lr_adjust_times = 0
all_lr = [args.lr*i for i in [1, 0.1, 0.01]]
for epoch in range(1, args.epochs + 1):
    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.sample_style, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch % args.eval_interval == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, topN)
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, topN)
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, topN)
        evaluate_utils.print_results(None, valid_results, test_results)

        has_improvement = False
        for key in metric_keys:
            if valid_results[key] > best_metrics[key]:
                best_metrics[key] = valid_results[key]
                has_improvement = True

        if has_improvement:
            best_epoch = epoch
            best_results = valid_results
            best_test_results = test_results
            no_improve_evals = 0
        else:
            no_improve_evals += 1
            if no_improve_evals >= args.patience:
                print('-'*18)
                print(f"Early stopping at epoch {epoch} after {args.patience} consecutive non-improving validations.")
                break

            # if not os.path.exists(args.save_path):
            #     os.makedirs(args.save_path)
            # torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth' \
            #     .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
            #     args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
if best_results is not None and best_test_results is not None:
    evaluate_utils.print_results(None, best_results, best_test_results)
else:
    print("No validation was run; increase --epochs or reduce --eval_interval.")
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
