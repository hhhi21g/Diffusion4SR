import argparse
import importlib.util
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gru4rec_pytorch import GRU4Rec, SessionDataIterator


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_params(param_path: Path):
    spec = importlib.util.spec_from_file_location(param_path.stem, str(param_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gru4rec_params


def load_data(path: Path):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["SessionId", "ItemId", "Time"],
        dtype={"SessionId": "int32", "ItemId": "str", "Time": "int32"},
    )


def iter_with_last(data_iterator: SessionDataIterator, reset_hook=None):
    batch_size = data_iterator.batch_size
    iters = np.arange(batch_size)
    maxiter = iters.max()
    start = data_iterator.offset_sessions[data_iterator.session_idx_arr[iters]]
    end = data_iterator.offset_sessions[data_iterator.session_idx_arr[iters] + 1]
    finished = False
    while not finished:
        minlen = (end - start).min()
        out_idx = torch.tensor(data_iterator.data_items[start], requires_grad=False, device=data_iterator.device)
        for i in range(minlen - 1):
            in_idx = out_idx
            out_idx = torch.tensor(data_iterator.data_items[start + i + 1], requires_grad=False, device=data_iterator.device)
            last_mask = torch.tensor((start + i + 2 == end), dtype=torch.bool, device=data_iterator.device)
            yield in_idx, out_idx, last_mask
        start = start + minlen - 1
        finished_mask = end - start <= 1
        n_finished = finished_mask.sum()
        iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
        maxiter += n_finished
        valid_mask = iters < len(data_iterator.offset_sessions) - 1
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = data_iterator.session_idx_arr[iters[mask]]
        start[mask] = data_iterator.offset_sessions[sessions]
        end[mask] = data_iterator.offset_sessions[sessions + 1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if reset_hook is not None:
            finished = reset_hook(n_valid, finished_mask, valid_mask)


@torch.no_grad()
def leave_one_out_eval(gru, test_data, cutoff=(1, 5, 10, 20), batch_size=512, mode="conservative"):
    hr = {c: 0.0 for c in cutoff}
    ndcg = {c: 0.0 for c in cutoff}

    n_sessions = int(test_data["SessionId"].nunique())
    eval_batch_size = min(batch_size, n_sessions)
    H = [
        torch.zeros((eval_batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32)
        for i in range(len(gru.layers))
    ]
    n_eval = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    data_iterator = SessionDataIterator(
        test_data,
        eval_batch_size,
        0,
        0,
        0,
        "ItemId",
        "SessionId",
        "Time",
        device=gru.device,
        itemidmap=gru.data_iterator.itemidmap,
    )

    for in_idxs, out_idxs, last_mask in iter_with_last(data_iterator, reset_hook=reset_hook):
        for h in H:
            h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])

        if mode == "standard":
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == "conservative":
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == "median":
            ranks = (oscores > tscores).sum(dim=0).float() + 0.5 * ((oscores == tscores).sum(dim=0).float() - 1.0) + 1.0
        else:
            raise NotImplementedError(f"Unknown eval mode: {mode}")

        selected_ranks = ranks[last_mask]
        if selected_ranks.numel() == 0:
            continue
        n_eval += int(selected_ranks.numel())

        for c in cutoff:
            hit = selected_ranks <= c
            hr[c] += hit.sum().item()
            ndcg[c] += (hit.float() / torch.log2(selected_ranks.float() + 1.0)).sum().item()

    if n_eval == 0:
        raise RuntimeError("No leave-one-out targets were evaluated. Check input data.")

    for c in cutoff:
        hr[c] /= n_eval
        ndcg[c] /= n_eval
    return hr, ndcg, n_eval


def main():
    parser = argparse.ArgumentParser(
        description="Train GRU4Rec and evaluate leave-one-out HR/NDCG using test_with_trainval as input sequence."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["amazon_beauty", "ml-1m", "sports", "video"],
        help="Dataset names under ./datasets",
    )
    parser.add_argument("--seed", type=int, default=1997, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--cutoff",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Evaluate HR/NDCG at these K values.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["standard", "conservative", "median"],
        default="conservative",
        help="Tiebreaking mode.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    set_global_seed(args.seed)

    total_t0 = time.time()
    all_results = []
    for ds in args.datasets:
        ds_t0 = time.time()
        ds_dir = repo_root / "datasets" / ds
        param_file = repo_root / "paramfiles" / f"{ds}_xe_shared_best.py"
        train_file = ds_dir / "train.tsv"
        test_file = ds_dir / "test_with_trainval.tsv"
        if not train_file.exists() or not test_file.exists() or not param_file.exists():
            raise FileNotFoundError(f"Missing files for dataset {ds}")

        print(f"\n=== Dataset: {ds} ===")
        print(f"Seed: {args.seed}")
        print("Split: leave-one-out (evaluate only the last target per session)")
        print(f"Test input sequence: {test_file.name} (train+val+test sequence)")

        params = load_params(param_file)
        gru = GRU4Rec(device=args.device)
        gru.set_params(**params)
        gru.set_params(random_seed=args.seed)

        print("Loading train data...")
        train_data = load_data(train_file)
        t0 = time.time()
        gru.fit(
            train_data,
            sample_cache_max_size=10000000,
            compatibility_mode=True,
            item_key="ItemId",
            session_key="SessionId",
            time_key="Time",
        )
        train_time = time.time() - t0

        print("Loading test_with_trainval data...")
        test_data = load_data(test_file)
        t0 = time.time()
        hr, ndcg, n_eval = leave_one_out_eval(
            gru,
            test_data,
            cutoff=tuple(args.cutoff),
            batch_size=512,
            mode=args.eval_type,
        )
        eval_time = time.time() - t0
        ds_time = time.time() - ds_t0

        print(f"Evaluated targets (sessions): {n_eval}")
        for k in args.cutoff:
            print(f"HR@{k}: {hr[k] * 100:.5f}%   NDCG@{k}: {ndcg[k] * 100:.5f}%")
        print(f"Training time: {train_time:.2f}s")
        print(f"Evaluation time: {eval_time:.2f}s")
        print(f"Dataset total time: {ds_time:.2f}s")

        all_results.append(
            {
                "dataset": ds,
                "hr": hr,
                "ndcg": ndcg,
                "train_time": train_time,
                "eval_time": eval_time,
                "total_time": ds_time,
                "n_eval": n_eval,
            }
        )

    total_time = time.time() - total_t0
    print("\n=== Summary ===")
    for r in all_results:
        print(f"\n{r['dataset']}:")
        for k in args.cutoff:
            print(f"HR@{k}: {r['hr'][k] * 100:.5f}%   NDCG@{k}: {r['ndcg'][k] * 100:.5f}%")
        print(f"Training: {r['train_time']:.2f}s  Eval: {r['eval_time']:.2f}s  Total: {r['total_time']:.2f}s")
    print(f"\nOverall runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()
