import sys
sys.path.append('./')
import time
import functools
import torch
import numpy as np
from tqdm import tqdm, trange
from models.cadirec import CaDiRec 
from models.gaussian_diffusion import SpacedDiffusion,space_timesteps
from utils import get_full_sort_score, EarlyStopping
from models import gaussian_diffusion as gd
from .step_sample import UniformSampler

class Trainer:
    def __init__(self, args, device, generator):

        self.args = args
        self.device = device
        self.start_epoch = 0    # define the start epoch for keepon trainingzhonss

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.generator = generator
        self.train_dataloader = generator.train_dataloader
        self.valid_dataloader = generator.valid_dataloader
        self.test_dataloader = generator.test_dataloader
        self.item_size = generator.item_size
        self.args.item_size = generator.item_size
        self.generator = generator
        
        self._create_model()
        self._set_optimizer()
        # self._set_stopper()
        

    def _create_model(self):
        self.model = CaDiRec(self.device, self.args)
        self.model.to(self.device)
        
        betas = gd.get_named_beta_schedule(self.args.noise_schedule, self.args.diffusion_steps)
        timestep_respacing = [self.args.diffusion_steps]
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(self.args.diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=self.args.rescale_timesteps,
            predict_xstart=self.args.predict_xstart,
            learn_sigmas = self.args.learn_sigma,
            sigma_small = self.args.sigma_small,
            use_kl = self.args.use_kl,
            rescale_learned_sigmas=self.args.rescale_learned_sigmas
        )
        self.schedule_sampler = UniformSampler(self.diffusion)
    
    
    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(),  
                                        lr=self.args.learning_rate,
                                        # betas=(0.9, 0.999),
                                        weight_decay=self.args.weight_decay)


    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):

        tr_loss = 0
        tr_diff_loss = 0
        tr_sas_rec_loss = 0
        tr_sas_cl_loss = 0
        epoch_start = time.time()
        batch_times = []
      
        self.model.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')
  
        for batch in prog_iter:

            train_start = time.time()
       
            input_ids, target_pos, target_neg, attention_mask, masked_indices0 = \
                                                                                    batch["input_ids"].to(self.device), \
                                                                                    batch["target_pos"].to(self.device), \
                                                                                    batch["target_neg"].to(self.device), \
                                                                                    batch["attention_mask"].to(self.device), \
                                                                                    batch["masked_indices0"].to(self.device)
         
            self.optimizer.zero_grad()
            
            t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                t,
                input_ids,
                masked_indices0.long(), 
                attention_mask
            )
            
            diff_mse_loss, diff_nll_loss, aug_seq1, aug_seq2 = compute_losses()
            model_emb = torch.nn.Embedding(
                            num_embeddings=self.args.item_size, 
                            embedding_dim=self.args.hidden_size, 
                            _weight=self.model.item_embedding.weight.clone().cpu()
                            ).eval().requires_grad_(False)
            model_emb.to(self.device)
            aug_seq1_emb = model_emb(aug_seq1)
            aug_seq2_emb = model_emb(aug_seq2)
            

            sas_rec_loss = self.model.calculate_rec_loss(input_ids, target_pos, target_neg)
            
            
            if epoch <= self.args.warm_up_epochs: # warm_up_epochs=-1 denotes not using
                sas_cl_loss = 0.0
                loss = sas_rec_loss + self.args.gamma * diff_mse_loss + self.args.beta * diff_nll_loss
            else:
                sas_cl_loss = self.model.calculate_cl_loss(aug_seq1, aug_seq2, aug_seq1_emb, aug_seq2_emb) 
                loss = sas_rec_loss + self.args.gamma * diff_mse_loss + self.args.beta * diff_nll_loss + self.args.alpha * sas_cl_loss

        
            loss.backward()
            self.optimizer.step()
            
            tr_diff_loss += diff_nll_loss / len(self.train_dataloader)
         
            tr_sas_rec_loss += sas_rec_loss / len(self.train_dataloader)
            tr_sas_cl_loss += sas_cl_loss / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            batch_times.append(train_end - train_start)
        # if epoch %10==0:
        #     print("aug_seq1", aug_seq1[masked_indices0])
        #     print("aug_seq2", aug_seq2[masked_indices0])
        print(f' epoch {epoch}: diff_loss {tr_diff_loss:.4f}', end='   ')
        print(f'sas_rec_loss {tr_sas_rec_loss:.4f}', end='   ')
        print(f'sas_cl_loss {tr_sas_cl_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}', end='   ')
        epoch_time = time.time() - epoch_start
        avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
        print(f'epoch_time {epoch_time:.2f}s   avg_batch_time {avg_batch_time:.4f}s')
        return epoch_time

    @staticmethod
    def _format_metrics(metrics_percent, ks=(1, 5, 10, 20)):
        if not metrics_percent:
            return "N/A"
        ordered_keys = []
        for k in ks:
            ordered_keys.extend([f"HR@{k}", f"NDCG@{k}"])
        shown = [f"{key}: {metrics_percent[key]:.5f}%" for key in ordered_keys if key in metrics_percent]
        return ", ".join(shown) if shown else "N/A"

    @staticmethod
    def _pick_monitor_key(metrics_percent, objective_metric):
        if metrics_percent and objective_metric in metrics_percent:
            return objective_metric
        fallback_order = ("NDCG@10", "HR@10", "NDCG@5", "HR@5", "NDCG@20", "HR@20", "NDCG@1", "HR@1")
        for key in fallback_order:
            if metrics_percent and key in metrics_percent:
                return key
        return next(iter(metrics_percent.keys()), None)


    def train(self, objective_metric="NDCG@10", eval_interval=10, **kwargs):
        print("********** Running training **********")
        train_time = []
        total_start = time.time()
        eval_interval = int(eval_interval) if int(eval_interval) > 0 else 10
        if "eval_interval" in kwargs:
            kw_eval_interval = int(kwargs["eval_interval"])
            eval_interval = kw_eval_interval if kw_eval_interval > 0 else eval_interval
        early_stop_rounds = kwargs.get("early_stop_rounds", None)
        use_early_stop = True
        if early_stop_rounds is None:
            # Follow author's default in utils.EarlyStopping: patience=7
            early_stopping = EarlyStopping(self.args.checkpoint_path, verbose=True)
        else:
            early_stop_rounds = int(early_stop_rounds)
            if early_stop_rounds <= 0:
                use_early_stop = False
                early_stopping = None
            else:
                early_stopping = EarlyStopping(
                    self.args.checkpoint_path,
                    patience=early_stop_rounds,
                    verbose=True,
                )
        if use_early_stop:
            print(f"EarlyStopping enabled. monitor=[HR@10, NDCG@10], patience={early_stopping.patience} eval rounds")
        else:
            print("EarlyStopping disabled")

        trial = kwargs.get("trial", None)
        enable_pruning = bool(kwargs.get("enable_pruning", False))

        best_epoch = None
        best_valid_metrics = None
        best_test_metrics = None
        best_metric_key = "HR@10+NDCG@10"
        best_metric_value = None

        last_eval_epoch = None
        last_valid_metrics = None
        last_test_metrics = None
        eval_round_idx = 0
   
        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)
        
            train_time.append(t) 
            
            if (epoch - self.start_epoch) % eval_interval == 0:
                valid_metrics, valid_metrics_percent, _ = self.eval(epoch, test=False)  # valid
                test_metrics, test_metrics_percent, _ = self.eval(epoch, test=True)  # test

                last_eval_epoch = epoch
                last_valid_metrics = dict(valid_metrics_percent)
                last_test_metrics = dict(test_metrics_percent)

                # Follow author's comment in utils.EarlyStopping: score HIT@10 NDCG@10
                score_for_early_stop = [
                    float(valid_metrics.get("HR@10", 0.0)),
                    float(valid_metrics.get("NDCG@10", 0.0)),
                ]
                is_improved = (
                    early_stopping.best_score is None or not early_stopping.compare(score_for_early_stop)
                ) if use_early_stop else False

                if use_early_stop:
                    early_stopping(score_for_early_stop, self.model)
                    if is_improved:
                        best_epoch = epoch
                        best_valid_metrics = dict(valid_metrics_percent)
                        best_test_metrics = dict(test_metrics_percent)
                        best_metric_value = (
                            float(valid_metrics_percent.get("HR@10", 0.0)),
                            float(valid_metrics_percent.get("NDCG@10", 0.0)),
                        )
                    if early_stopping.early_stop:
                        print("Early stopping triggered by [HR@10, NDCG@10]")
                        break
                else:
                    # If early stop is manually disabled, still track best by author's default pair.
                    pair_value = (
                        float(valid_metrics_percent.get("HR@10", 0.0)),
                        float(valid_metrics_percent.get("NDCG@10", 0.0)),
                    )
                    if best_metric_value is None or pair_value > best_metric_value:
                        best_epoch = epoch
                        best_valid_metrics = dict(valid_metrics_percent)
                        best_test_metrics = dict(test_metrics_percent)
                        best_metric_value = pair_value

                if trial is not None:
                    metric_for_trial = float(valid_metrics.get(objective_metric, 0.0))
                    trial.report(metric_for_trial, step=eval_round_idx)
                    if enable_pruning and trial.should_prune():
                        import optuna
                        raise optuna.exceptions.TrialPruned()
                eval_round_idx += 1

        total_train_time = time.time() - total_start
        avg_epoch_time = float(np.mean(train_time)) if train_time else 0.0
        print(f"Training completed. total_train_time: {total_train_time:.2f}s, avg_epoch_time: {avg_epoch_time:.2f}s")
        print("============== Result Summary ==============")
        if best_epoch is not None:
            print(f"Recommended checkpoint: {self.args.checkpoint_path}")
            if best_metric_value is not None:
                print(
                    f"Best epoch by {best_metric_key}: {best_epoch} "
                    f"(HR@10={best_metric_value[0]:.5f}%, NDCG@10={best_metric_value[1]:.5f}%)"
                )
            else:
                print(f"Best epoch by {best_metric_key}: {best_epoch}")
            print(f"Best VALID: {self._format_metrics(best_valid_metrics)}")
            print(f"Best TEST : {self._format_metrics(best_test_metrics)}")
        print("============================================")
        return {
            "total_train_time_sec": total_train_time,
            "avg_epoch_time_sec": avg_epoch_time,
            "final_epoch": last_eval_epoch,
            "final_valid_metrics": last_valid_metrics,
            "final_test_metrics": last_test_metrics,
            "best_epoch": best_epoch,
            "best_metric": best_metric_key,
            "best_metric_percent": list(best_metric_value) if best_metric_value is not None else None,
            "best_valid_metrics": best_valid_metrics,
            "best_test_metrics": best_test_metrics,
        }
            
    
    def eval(self, epoch, test=False):
      
        self.model.eval()
        eval_start = time.time()
        if not test:
            print("********** Running eval **********")
            prog_iter = tqdm(self.valid_dataloader, leave=False, desc='eval')
        else:
            print("********** Running test **********")
            prog_iter = tqdm(self.test_dataloader, leave=False, desc='test')

        scores = []
        labels = []
        for batch in prog_iter:
            user_ids, input_ids, label_items = \
                                                batch["user_id"].to(self.device), \
                                                batch["input_ids"].to(self.device), \
                                                batch["answer"].to(self.device), \
                                                                      
            bs_scores = self.model.full_sort_predict(input_ids).detach().cpu()
            
            batch_user_index = user_ids.cpu().numpy()
            # print("bs_scores", bs_scores.shape)
            # print("valid_rating_matrix", (self.generator.test_rating_matrix[batch_user_index].toarray() > 0).shape)
            if not test:
                bs_scores[self.generator.valid_rating_matrix[batch_user_index].toarray() > 0] = -100
            else:
                bs_scores[self.generator.test_rating_matrix[batch_user_index].toarray() > 0] = -100
            bs_labels = label_items.reshape(-1,1).cpu()
            scores.append(bs_scores)
            labels.append(bs_labels)
            
        scores = torch.cat(scores, axis=0).numpy()
        eval_ks = (1, 5, 10, 20)
        top_k = min(max(eval_ks), scores.shape[1])
        partitioned_indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        pred_list = partitioned_indices[np.arange(scores.shape[0])[:, None], np.argsort(-scores[np.arange(scores.shape[0])[:, None], partitioned_indices], axis=1)].tolist()
        labels = torch.cat(labels, axis=0).numpy().tolist()
        metrics, metrics_percent = get_full_sort_score(epoch, labels, pred_list, ks=eval_ks)
        eval_time = time.time() - eval_start
        print(f"eval_time: {eval_time:.2f}s")
        return metrics, metrics_percent, eval_time
