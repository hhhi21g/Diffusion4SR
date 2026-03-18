"""
Convert DuoRec/ICLRec-style `dataset.pkl` files to this repo's txt format.

Output format (one interaction per line):
    user_id item_id
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert my_data/data/*/dataset.pkl to data/*.txt"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("my_data/data"),
        help="Root directory that contains per-dataset folders with dataset.pkl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Directory to write converted txt files",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder names to convert (e.g., ml-1m amazon_beauty)",
    )
    parser.add_argument(
        "--user-id-offset",
        type=int,
        default=1,
        help="Offset added to user_id (default 1 converts 0-based users to 1-based)",
    )
    parser.add_argument(
        "--item-id-offset",
        type=int,
        default=0,
        help="Offset added to item_id (default 0 keeps item ids unchanged)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output txt files",
    )
    return parser.parse_args()


def as_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def load_dataset(pkl_path: Path) -> Dict[str, Any]:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{pkl_path} is not a dict object.")
    for key in ("train", "val", "test"):
        if key not in obj:
            raise ValueError(f"{pkl_path} missing key: {key}")
        if not isinstance(obj[key], dict):
            raise ValueError(f"{pkl_path} key '{key}' is not a dict.")
    return obj


def build_user_sequences(dataset: Dict[str, Any]) -> Dict[int, List[int]]:
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]

    user_ids = set(train.keys()) | set(val.keys()) | set(test.keys())
    merged: Dict[int, List[int]] = {}
    for uid in sorted(user_ids):
        seq: List[int] = []
        seq.extend(as_list(train.get(uid, [])))
        seq.extend(as_list(val.get(uid, [])))
        seq.extend(as_list(test.get(uid, [])))
        if seq:
            merged[int(uid)] = seq
    return merged


def write_txt(
    user_sequences: Dict[int, List[int]],
    output_path: Path,
    user_id_offset: int,
    item_id_offset: int,
) -> int:
    interaction_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for uid in sorted(user_sequences.keys()):
            output_uid = uid + user_id_offset
            for item_id in user_sequences[uid]:
                output_item = int(item_id) + item_id_offset
                if output_item <= 0:
                    raise ValueError(
                        f"Invalid item id {output_item} for user {output_uid} "
                        f"(input item {item_id}, item_id_offset={item_id_offset})."
                    )
                f.write(f"{output_uid} {output_item}\n")
                interaction_count += 1
    return interaction_count


def discover_pkl_files(input_root: Path, datasets: Iterable[str] = None) -> List[Path]:
    if datasets:
        paths = [input_root / name / "dataset.pkl" for name in datasets]
    else:
        paths = sorted(input_root.glob("*/dataset.pkl"))
    return paths


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pkl_paths = discover_pkl_files(args.input_root, args.datasets)
    if not pkl_paths:
        raise FileNotFoundError(
            f"No dataset.pkl found under: {args.input_root.resolve()}"
        )

    for pkl_path in pkl_paths:
        if not pkl_path.exists():
            print(f"[SKIP] Not found: {pkl_path}")
            continue

        dataset_name = pkl_path.parent.name
        output_path = args.output_root / f"{dataset_name}.txt"
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {output_path} (use --overwrite to replace)")
            continue

        dataset = load_dataset(pkl_path)
        user_sequences = build_user_sequences(dataset)
        interactions = write_txt(
            user_sequences=user_sequences,
            output_path=output_path,
            user_id_offset=args.user_id_offset,
            item_id_offset=args.item_id_offset,
        )
        print(
            f"[OK] {dataset_name}: users={len(user_sequences)}, "
            f"interactions={interactions}, output={output_path}"
        )


if __name__ == "__main__":
    main()
