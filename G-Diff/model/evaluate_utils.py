import numpy as np
# import bottleneck as bn
import torch
import math

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR

def cal_hr(label, predict, ks):
    """
    DiffuSR-style HR@K for single-label evaluation.
    label: [B, 1], predict: [B, num_items]
    """
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :k].sum().item() / label.size()[0] for k in ks]
    return hr

def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1, device=hit.device) + 1).unsqueeze(0)
    rel = (hit.float() / log2).sum(dim=-1)
    return rel

def cal_ndcg(label, predict, ks):
    """
    DiffuSR-style NDCG@K for single-label evaluation.
    label: [B, 1], predict: [B, num_items]
    """
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1), device=hit.device))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg

def hrs_and_ndcgs_k(scores, labels, ks):
    """
    Return a dict like {'HR@1': x, 'NDCG@1': y, ...}, matching DiffuSR.
    """
    metrics = {}
    labels_cpu = labels.clone().detach().to('cpu')
    scores_cpu = scores.clone().detach().to('cpu')
    ndcg = cal_ndcg(labels_cpu, scores_cpu, ks)
    hr = cal_hr(labels_cpu, scores_cpu, ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics[f'HR@{k}'] = hr_temp
        metrics[f'NDCG@{k}'] = ndcg_temp
    return metrics

def hrs_and_ndcgs_k_multi(ground_truth, predicted_indices, ks):
    """
    G-Diff style multi-label HR/NDCG@K.
    ground_truth: List[List[int]]
    predicted_indices: List[List[int]]
    """
    metrics = {}
    valid_user_num = sum(1 for gt in ground_truth if len(gt) > 0)
    if valid_user_num == 0:
        for k in ks:
            metrics[f'HR@{k}'] = 0.0
            metrics[f'NDCG@{k}'] = 0.0
        return metrics

    for k in ks:
        hr_sum = 0.0
        ndcg_sum = 0.0
        for gt, pred in zip(ground_truth, predicted_indices):
            if len(gt) == 0:
                continue
            gt_set = set(gt)
            topk = pred[:k]
            hits = [1 if item in gt_set else 0 for item in topk]

            # HR@K: whether any positive item is hit in top-K.
            hr_sum += 1.0 if any(hits) else 0.0

            # NDCG@K with multi-positive normalization.
            dcg = 0.0
            for rank, h in enumerate(hits):
                if h:
                    dcg += 1.0 / math.log2(rank + 2)
            idcg_len = min(len(gt_set), k)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(idcg_len))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        metrics[f'HR@{k}'] = hr_sum / valid_user_num
        metrics[f'NDCG@{k}'] = ndcg_sum / valid_user_num
    return metrics

def _format_hr_ndcg(result):
    ks = sorted({int(k.split('@')[-1]) for k in result.keys() if '@' in k})
    parts = []
    for k in ks:
        hr_key = f'HR@{k}'
        ndcg_key = f'NDCG@{k}'
        if hr_key in result:
            parts.append(f"{hr_key}: {result[hr_key] * 100:.5f}")
        if ndcg_key in result:
            parts.append(f"{ndcg_key}: {result[ndcg_key] * 100:.5f}")
    return ' '.join(parts)

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        if isinstance(valid_result, dict):
            print("[Valid]: {}".format(_format_hr_ndcg(valid_result)))
        else:
            print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        if isinstance(test_result, dict):
            print("[Test]: {}".format(_format_hr_ndcg(test_result)))
        else:
            print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))
