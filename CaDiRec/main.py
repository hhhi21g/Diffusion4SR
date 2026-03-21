import os
import sys
import time
import argparse
import importlib
import torch
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer 
from utils import set_seed


def _normalize_dataset_name(name):
    if name is None:
        return None
    n = str(name).strip().lower()
    alias = {
        "beauty": "amazon_beauty",
        "amazon-beauty": "amazon_beauty",
        "sports_and_outdoors": "sports",
        "sports-and-outdoors": "sports",
    }
    return alias.get(n, n)


def _replace_dataset_arg(argv, dataset_value):
    if dataset_value is None:
        return argv
    updated = list(argv)
    for i, token in enumerate(updated):
        if token.startswith("--dataset="):
            updated[i] = f"--dataset={dataset_value}"
            return updated
        if token == "--dataset" and i + 1 < len(updated):
            updated[i + 1] = dataset_value
            return updated
    return updated


def _select_config_module():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", type=str, default=None)
    known, _ = selector.parse_known_args(sys.argv[1:])

    dataset = _normalize_dataset_name(known.dataset)
    if dataset is not None:
        sys.argv = [sys.argv[0]] + _replace_dataset_arg(sys.argv[1:], dataset)

    dataset_to_module = {
        "ml-1m": "configs.cadirec_config_ml",
        "amazon_beauty": "configs.cadirec_config_beauty",
        "sports": "configs.cadirec_config_sports",
        "video": "configs.cadirec_config_video",
    }
    module_name = dataset_to_module.get(dataset, "configs.cadirec_config")
    config_module = importlib.import_module(module_name)
    return config_module.get_config, module_name


def main():
    run_start = time.time()
    get_config, config_module_name = _select_config_module()
    args = get_config()
    args.seed = 1997
    set_seed(args.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"config_script: {config_module_name}")
    print(f"dataset: {args.dataset}")
    print(f"random_seed: {args.seed}")
    print("evaluation protocol: leave-one-out (valid=last-2, test=last-1), test input uses train+val")
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    train_summary = trainer.train()
    total_time = time.time() - run_start
    print("==============================")
    print(f"Total running time: {total_time:.2f} seconds")
    if train_summary is not None:
        print(f"Total train time: {train_summary.get('total_train_time_sec', 0.0):.2f} seconds")
    print("==============================")
 

if __name__ == "__main__":
    main()



