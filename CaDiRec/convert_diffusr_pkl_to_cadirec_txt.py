#!/usr/bin/env python3
"""
Convert DiffuSR processed dataset.pkl into CaDiRec txt format.

CaDiRec expected txt line format:
    <user_id> <item_1> <item_2> ... <item_n>

CaDiRec split logic (in data loader) is:
    train = items[:-2]
    valid target = items[-2]
    test target = items[-1]
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DiffuSR dataset.pkl to CaDiRec txt format."
    )
    parser.add_argument(
        "--input-pkl",
        type=Path,
        required=True,
        help="Path to DiffuSR dataset.pkl",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        required=True,
        help="Path to output txt file for CaDiRec",
    )
    parser.add_argument(
        "--train-key",
        type=str,
        default="train",
        help="Key name for train split in pkl (default: train)",
    )
    parser.add_argument(
        "--val-key",
        type=str,
        default="val",
        help="Key name for validation split in pkl (default: val)",
    )
    parser.add_argument(
        "--test-key",
        type=str,
        default="test",
        help="Key name for test split in pkl (default: test)",
    )
    parser.add_argument(
        "--user-id-offset",
        type=int,
        default=1,
        help=(
            "If --preserve-user-id is not set, output user ids are generated as "
            "1..N (offset=1 by default)."
        ),
    )
    parser.add_argument(
        "--preserve-user-id",
        action="store_true",
        help="Use original user ids from pkl as the first column.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=5,
        help=(
            "Drop sequences shorter than this value in output txt. "
            "Default 5 to match CaDiRec's effective user filtering."
        ),
    )
    parser.add_argument(
        "--strict-holdout",
        action="store_true",
        default=True,
        help=(
            "Require val/test length == 1 and exact consistency with "
            "train+val+test reconstruction (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-strict-holdout",
        dest="strict_holdout",
        action="store_false",
        help="Disable strict holdout checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists.",
    )
    return parser.parse_args()


def _ensure_dict(obj: Any, name: str) -> Dict[Any, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"'{name}' must be a dict, got {type(obj).__name__}")
    return obj


def _to_int_list(value: Any, name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        seq = value
    elif isinstance(value, tuple):
        seq = list(value)
    else:
        # Fallback for scalar values.
        seq = [value]
    out: List[int] = []
    for x in seq:
        try:
            out.append(int(x))
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"{name} contains non-integer value: {x!r}") from exc
    return out


def _sorted_user_ids(keys: Iterable[Any]) -> List[Any]:
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=str)


def main() -> None:
    args = parse_args()

    if not args.input_pkl.exists():
        raise FileNotFoundError(f"Input pkl not found: {args.input_pkl}")
    if args.output_txt.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_txt}. "
            f"Use --overwrite to replace it."
        )

    with args.input_pkl.open("rb") as f:
        data = pickle.load(f)

    data = _ensure_dict(data, "dataset.pkl root object")

    for required_key in (args.train_key, args.val_key, args.test_key):
        if required_key not in data:
            raise KeyError(
                f"Missing key '{required_key}' in pkl. "
                f"Available keys: {list(data.keys())}"
            )

    train = _ensure_dict(data[args.train_key], args.train_key)
    val = _ensure_dict(data[args.val_key], args.val_key)
    test = _ensure_dict(data[args.test_key], args.test_key)

    train_users = set(train.keys())
    val_users = set(val.keys())
    test_users = set(test.keys())
    common_users = train_users & val_users & test_users

    if not common_users:
        raise ValueError("No common users across train/val/test.")

    if train_users != val_users or train_users != test_users:
        missing_in_val = sorted(train_users - val_users, key=str)[:5]
        missing_in_test = sorted(train_users - test_users, key=str)[:5]
        raise ValueError(
            "User sets mismatch across train/val/test. "
            f"Example missing in val: {missing_in_val}, "
            f"missing in test: {missing_in_test}"
        )

    user_ids = _sorted_user_ids(common_users)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_short = 0
    val_len_violations = 0
    test_len_violations = 0

    with args.output_txt.open("w", encoding="utf-8") as wf:
        for idx, uid in enumerate(user_ids):
            train_seq = _to_int_list(train[uid], f"train[{uid}]")
            val_seq = _to_int_list(val[uid], f"val[{uid}]")
            test_seq = _to_int_list(test[uid], f"test[{uid}]")
            full_seq = train_seq + val_seq + test_seq

            if args.strict_holdout:
                if len(val_seq) != 1:
                    val_len_violations += 1
                if len(test_seq) != 1:
                    test_len_violations += 1

                if len(full_seq) < 2:
                    raise ValueError(
                        f"User {uid} has full sequence length < 2, "
                        f"cannot apply CaDiRec holdout."
                    )
                if train_seq != full_seq[:-2]:
                    raise ValueError(
                        f"Holdout mismatch for user {uid}: "
                        "train != full_seq[:-2]"
                    )
                if val_seq != [full_seq[-2]]:
                    raise ValueError(
                        f"Holdout mismatch for user {uid}: "
                        "val != [full_seq[-2]]"
                    )
                if test_seq != [full_seq[-1]]:
                    raise ValueError(
                        f"Holdout mismatch for user {uid}: "
                        "test != [full_seq[-1]]"
                    )

            if len(full_seq) < args.min_seq_len:
                dropped_short += 1
                continue

            out_uid = uid if args.preserve_user_id else (idx + args.user_id_offset)
            line = f"{int(out_uid)} " + " ".join(map(str, full_seq))
            wf.write(line + "\n")
            kept += 1

    if args.strict_holdout and (val_len_violations > 0 or test_len_violations > 0):
        raise ValueError(
            "Strict holdout check failed: "
            f"val length != 1 for {val_len_violations} users, "
            f"test length != 1 for {test_len_violations} users."
        )

    print(f"[DONE] Input pkl: {args.input_pkl}")
    print(f"[DONE] Output txt: {args.output_txt}")
    print(f"[STATS] users_in_common={len(user_ids)}, kept={kept}, dropped_short={dropped_short}")
    print(
        f"[CHECK] strict_holdout={args.strict_holdout}, "
        f"min_seq_len={args.min_seq_len}, preserve_user_id={args.preserve_user_id}"
    )


if __name__ == "__main__":
    main()
