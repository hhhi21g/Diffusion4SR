#!/usr/bin/env python
"""
Convert DiffuSR-style dataset.pkl files to G-Diff .npy pair format.

Input (per dataset):
  my_data/<dataset>/dataset.pkl
  keys: train/val/test, each is Dict[user_id, List[item_id]]

Output (per dataset):
  <dst_root>/<dataset>/train_list.npy
  <dst_root>/<dataset>/valid_list.npy
  <dst_root>/<dataset>/test_list.npy
  each array shape: [N, 2], columns: [user_id, item_id]
"""

import argparse
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert my_data dataset.pkl to G-Diff npy format.")
    parser.add_argument(
        "--src_root",
        type=str,
        default="./my_data",
        help="Root folder containing dataset subfolders with dataset.pkl.",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="./datasets_converted",
        help="Output root folder for G-Diff formatted .npy files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=None,
        help="Dataset names to convert. If omitted, convert all subfolders under src_root that contain dataset.pkl.",
    )
    parser.add_argument(
        "--no_reindex_users",
        action="store_true",
        help="Keep original user ids (must be integer-like). Default reindexes users to 0..U-1.",
    )
    parser.add_argument(
        "--no_reindex_items",
        action="store_true",
        help="Keep original item ids (must be integer-like). Default reindexes items to 0..I-1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they exist.",
    )
    return parser.parse_args()


def _sorted_keys(d: Dict) -> List:
    # Stable sorting for mixed key types.
    return sorted(d.keys(), key=lambda x: str(x))


def _build_id_map(values: Iterable, reindex: bool) -> Dict:
    if reindex:
        uniq = sorted(set(values), key=lambda x: str(x))
        return {v: i for i, v in enumerate(uniq)}
    # Keep original ids, enforce integer-like.
    out = {}
    for v in set(values):
        try:
            out[v] = int(v)
        except Exception as exc:
            raise ValueError(f"ID {v!r} cannot be converted to int when reindex is disabled.") from exc
    return out


def _dict_to_pairs(
    user_dict: Dict,
    user_map: Dict,
    item_map: Dict,
) -> np.ndarray:
    pairs: List[Tuple[int, int]] = []
    for user in _sorted_keys(user_dict):
        items = user_dict[user]
        if items is None:
            continue
        for item in items:
            pairs.append((user_map[user], item_map[item]))
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def convert_one_dataset(
    src_root: str,
    dst_root: str,
    dataset_name: str,
    reindex_users: bool,
    reindex_items: bool,
    overwrite: bool,
) -> None:
    src_pkl = os.path.join(src_root, dataset_name, "dataset.pkl")
    if not os.path.exists(src_pkl):
        raise FileNotFoundError(f"Missing file: {src_pkl}")

    with open(src_pkl, "rb") as f:
        data = pickle.load(f)

    for key in ("train", "val", "test"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"{src_pkl} missing dict key '{key}'.")

    train_dict: Dict = data["train"]
    val_dict: Dict = data["val"]
    test_dict: Dict = data["test"]

    users = set(train_dict.keys()) | set(val_dict.keys()) | set(test_dict.keys())
    items = set()
    for d in (train_dict, val_dict, test_dict):
        for seq in d.values():
            if seq is None:
                continue
            items.update(seq)

    user_map = _build_id_map(users, reindex_users)
    item_map = _build_id_map(items, reindex_items)

    train_pairs = _dict_to_pairs(train_dict, user_map, item_map)
    valid_pairs = _dict_to_pairs(val_dict, user_map, item_map)
    test_pairs = _dict_to_pairs(test_dict, user_map, item_map)

    out_dir = os.path.join(dst_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    out_train = os.path.join(out_dir, "train_list.npy")
    out_valid = os.path.join(out_dir, "valid_list.npy")
    out_test = os.path.join(out_dir, "test_list.npy")

    if (not overwrite) and (os.path.exists(out_train) or os.path.exists(out_valid) or os.path.exists(out_test)):
        raise FileExistsError(
            f"Output exists for dataset '{dataset_name}' in {out_dir}. "
            "Use --overwrite to replace."
        )

    np.save(out_train, train_pairs)
    np.save(out_valid, valid_pairs)
    np.save(out_test, test_pairs)

    print(
        f"[OK] {dataset_name}: "
        f"users={len(user_map)} items={len(item_map)} "
        f"train={train_pairs.shape[0]} valid={valid_pairs.shape[0]} test={test_pairs.shape[0]}"
    )
    print(f"     saved to: {out_dir}")


def discover_datasets(src_root: str) -> List[str]:
    names: List[str] = []
    if not os.path.isdir(src_root):
        return names
    for name in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, name, "dataset.pkl")
        if os.path.exists(p):
            names.append(name)
    return names


def main() -> None:
    args = parse_args()
    reindex_users = not args.no_reindex_users
    reindex_items = not args.no_reindex_items

    datasets = args.dataset if args.dataset else discover_datasets(args.src_root)
    if not datasets:
        raise RuntimeError(f"No datasets found under {args.src_root}")

    print(f"src_root={args.src_root}")
    print(f"dst_root={args.dst_root}")
    print(f"datasets={datasets}")
    print(f"reindex_users={reindex_users} reindex_items={reindex_items} overwrite={args.overwrite}")

    for name in datasets:
        convert_one_dataset(
            src_root=args.src_root,
            dst_root=args.dst_root,
            dataset_name=name,
            reindex_users=reindex_users,
            reindex_items=reindex_items,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

