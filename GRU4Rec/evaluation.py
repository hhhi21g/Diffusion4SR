from gru4rec_pytorch import SessionDataIterator
import torch

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time', eval_scope='last'):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    if eval_scope not in ('last', 'all'):
        raise ValueError('eval_scope must be "last" or "all", got: {}'.format(eval_scope))
    recall = dict()
    mrr = dict()
    hr = dict()
    ndcg = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
        hr[c] = 0
        ndcg[c] = 0
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    n = 0
    n_eval = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)
    for in_idxs, out_idxs, last_mask in data_iterator(enable_neg_samples=False, reset_hook=reset_hook, return_last_mask=True):
        for h in H: h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])
        if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).sum(dim=0) - 1) + 1
        else: raise NotImplementedError
        if eval_scope == 'last':
            eval_mask = last_mask
        else:
            eval_mask = torch.ones_like(last_mask, dtype=torch.bool)
        n_eval += int(eval_mask.sum().item())
        for c in cutoff:
            hits = (ranks <= c)
            selected_hits = hits & eval_mask
            recall[c] += selected_hits.sum().item()
            mrr[c] += (selected_hits.float() / ranks.float()).sum().item()
            # DiffuSR-aligned metrics for single-target evaluation.
            hr[c] += selected_hits.sum().item()
            ndcg[c] += (selected_hits.float() / torch.log2(ranks.float() + 1.0)).sum().item()
        n += O.shape[0]
    if n_eval == 0:
        raise RuntimeError('No valid evaluation targets were found. Check input data and eval_scope.')
    for c in cutoff:
        recall[c] /= n_eval
        mrr[c] /= n_eval
        hr[c] /= n_eval
        ndcg[c] /= n_eval
    return {
        'recall': recall,
        'mrr': mrr,
        'hr': hr,
        'ncg': ndcg,
        # Alias kept for compatibility with common naming.
        'ndcg': ndcg
    }
