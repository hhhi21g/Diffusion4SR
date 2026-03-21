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

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))


def computeHRNDCG(GroundTruth, predictedIndices, topN, scale=100.0):
    """
    Compute HR@K and NDCG@K (leave-one-out friendly).
    Returns values in percentage when scale=100.
    """
    hr = []
    ndcg = []
    user_num = max(len(predictedIndices), 1)

    for k in topN:
        hit_sum = 0.0
        ndcg_sum = 0.0
        for i in range(len(predictedIndices)):
            gt_items = GroundTruth[i]
            if len(gt_items) == 0:
                continue

            gt_set = set(gt_items)
            hit_rank = None
            for rank, item_id in enumerate(predictedIndices[i][:k]):
                if item_id in gt_set:
                    hit_rank = rank
                    break

            if hit_rank is not None:
                hit_sum += 1.0
                ndcg_sum += 1.0 / math.log2(hit_rank + 2)

        hr.append(round(hit_sum / user_num * scale, 5))
        ndcg.append(round(ndcg_sum / user_num * scale, 5))

    return hr, ndcg


def print_hr_ndcg_results(valid_result, test_result, topN):
    if valid_result is not None:
        valid_metrics = []
        for k, hr_value, ndcg_value in zip(topN, valid_result[0], valid_result[1]):
            valid_metrics.append("HR@{}: {:.5f} NDCG@{}: {:.5f}".format(k, hr_value, k, ndcg_value))
        print("[Valid]: {}".format(" | ".join(valid_metrics)))

    if test_result is not None:
        test_metrics = []
        for k, hr_value, ndcg_value in zip(topN, test_result[0], test_result[1]):
            test_metrics.append("HR@{}: {:.5f} NDCG@{}: {:.5f}".format(k, hr_value, k, ndcg_value))
        print("[Test]: {}".format(" | ".join(test_metrics)))
