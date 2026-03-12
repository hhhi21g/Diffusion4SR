import os
import sys
import argparse
import torch
from tqdm import tqdm
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer 
from utils import set_seed
import os


def get_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config_profile",
        type=str,
        default="default",
        choices=["default", "beauty", "ml","sports",'video'],
        help="Choose parameter preset from configs/",
    )
    known_args, remaining_argv = parser.parse_known_args()
    # Prevent selected config parser from seeing this helper arg.
    sys.argv = [sys.argv[0]] + remaining_argv

    if known_args.config_profile == "ml":
        from configs.cadirec_config_ml import get_config as _get_config
    elif known_args.config_profile == "beauty":
        from configs.cadirec_config_beauty import get_config as _get_config
    elif known_args.config_profile == "sports":
        from configs.cadirec_config_sports import get_config as _get_config
    elif known_args.config_profile == 'video':
        from configs.cadirec_config_video import get_config as _get_config
    else:
        from configs.cadirec_config import get_config as _get_config

    args = _get_config()
    args.config_profile = known_args.config_profile
    return args


def main():
    args = get_config()
    set_seed(args.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    trainer.train()
 

if __name__ == "__main__":
    main()





