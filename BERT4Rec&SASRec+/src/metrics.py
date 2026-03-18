"""
Metrics.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


def compute_hr_ndcg_metrics(ground_truth, preds, ks=(1, 5, 10, 20)):
    """
    Compute HR@K and NDCG@K for one-positive-item evaluation per user.
    The logic is aligned with DiffuSR: HR is hit ratio, NDCG gives
    1/log2(rank+1) when the target is within top-k.
    """

    ks = sorted(set(int(k) for k in ks))
    if len(ks) == 0:
        return {}

    gt = ground_truth.groupby("user_id", as_index=False).first()[["user_id", "item_id"]]
    gt_dict = dict(zip(gt["user_id"].values, gt["item_id"].values))

    # ensure ranking by model score
    preds_sorted = preds.sort_values(["user_id", "prediction"], ascending=[True, False])
    pred_lists = preds_sorted.groupby("user_id")["item_id"].apply(list).to_dict()

    hr_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    user_count = len(gt_dict)

    if user_count == 0:
        metrics = {}
        for k in ks:
            metrics[f"HR@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
        return metrics

    for user_id, target_item in gt_dict.items():
        ranked_items = pred_lists.get(user_id, [])
        rank = None
        # rank is 1-based
        for idx, item in enumerate(ranked_items):
            if item == target_item:
                rank = idx + 1
                break

        for k in ks:
            if rank is not None and rank <= k:
                hr_sums[k] += 1.0
                ndcg_sums[k] += 1.0 / np.log2(rank + 1)

    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = hr_sums[k] / user_count
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / user_count
    return metrics


def compute_sampled_metrics(seqrec_module, predict_dataset, test, item_counts,
                            popularity_sampling=True, num_negatives=100, k=10,
                            device='cuda'):

    test = test.set_index('user_id')['item_id'].to_dict()
    all_items = item_counts.index.values
    item_weights = item_counts.values
    # probabilities = item_weights/item_weights.sum()

    seqrec_module = seqrec_module.eval().to(device)

    ndcg, hit_rate, mrr = 0.0, 0.0, 0.0
    user_count = 0

    for user in tqdm(predict_dataset):

        if user['user_id'] not in test:
            continue

        positive = test[user['user_id']]
        indices = ~np.isin(all_items, user['full_history'])
        negatives = all_items[indices]
        if popularity_sampling:
            probabilities = item_weights[indices]
            probabilities = probabilities/probabilities.sum()
        else:
            probabilities = None
        negatives = np.random.choice(negatives, size=num_negatives,
                                     replace=False, p=probabilities)
        items = np.concatenate([np.array([positive]), negatives])

        # code from BERT4Rec original repo https://github.com/FeiSun/BERT4Rec/blob/master/run.py#L195
        # items = [test[user['user_id']]]
        # while len(items) < num_negatives + 1:
        #     sampled_ids = np.random.choice(all_items, num_negatives + 1, replace=False, p=probabilities)
        #     sampled_ids = [x for x in sampled_ids if x not in user['full_history'] and x not in items]
        #     items.extend(sampled_ids[:])
        # items = items[:num_negatives + 1]

        batch = {'input_ids': torch.tensor(user['input_ids']).unsqueeze(0).to(device),
                 'attention_mask': torch.tensor([1] * len(user['input_ids'])).unsqueeze(0).to(device)}
        pred = seqrec_module.prediction_output(batch)
        pred = pred[0, -1, items]

        rank = (-pred).argsort().argsort()[0].item() + 1
        if rank <= k:
            ndcg += 1 / np.log2(rank + 1)
            hit_rate += 1
            mrr += 1 / rank
        user_count += 1

    ndcg = ndcg / user_count
    hit_rate = hit_rate / user_count
    mrr = mrr / user_count

    return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}
