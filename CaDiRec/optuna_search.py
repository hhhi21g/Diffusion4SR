import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer
from utils import set_seed


def load_profile_defaults(config_profile: str):
    if config_profile == "ml":
        from configs.cadirec_config_ml import get_config as _get_config
    elif config_profile == "beauty":
        from configs.cadirec_config_beauty import get_config as _get_config
    else:
        from configs.cadirec_config import get_config as _get_config

    argv_backup = sys.argv[:]
    try:
        # Load only defaults from profile file.
        sys.argv = [sys.argv[0]]
        args = _get_config()
    finally:
        sys.argv = argv_backup
    args.config_profile = config_profile
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna tuning for CaDiRec alpha/beta/rho in [0.1, 0.2]."
    )
    parser.add_argument(
        "--config_profile",
        type=str,
        default="default",
        choices=["default", "beauty", "ml"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data_new/")
    parser.add_argument("--output_dir", type=str, default="./saved_models/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--study_name", type=str, default="cadirec_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Optuna pruning strategy.",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of startup trials before pruning starts (median pruner).",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=1,
        help="Number of reported steps before pruning check (median pruner).",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--early_stop_rounds",
        type=int,
        default=3,
        help="Stop a trial if objective metric has no improvement for this many eval rounds.",
    )
    parser.add_argument(
        "--run_test_during_search",
        action="store_true",
        help="Also run test when a new best validation score appears (slower).",
    )
    parser.add_argument("--rho_low", type=float, default=0.1)
    parser.add_argument("--rho_high", type=float, default=0.2)
    parser.add_argument("--ab_low", type=float, default=0.1)
    parser.add_argument("--ab_high", type=float, default=0.2)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument(
        "--objective_metric",
        type=str,
        default="NDCG@10",
        choices=["NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20", "HR@1", "HR@5", "HR@10", "HR@20"],
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="./optuna_best.json",
        help="Path to save best trial summary json.",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna is not installed. Please run: pip install optuna"
        ) from exc

    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    def objective(trial):
        args = load_profile_defaults(cli.config_profile)
        args.dataset = cli.dataset
        args.data_path = cli.data_path
        args.output_dir = cli.output_dir
        args.config_profile = cli.config_profile
        args.seed = cli.seed + trial.number
        if cli.epochs is not None:
            args.epochs = cli.epochs

        args.alpha = trial.suggest_float("alpha", cli.ab_low, cli.ab_high, step=cli.step)
        args.beta = trial.suggest_float("beta", cli.ab_low, cli.ab_high, step=cli.step)
        rho = trial.suggest_float("rho", cli.rho_low, cli.rho_high, step=cli.step)
        args.mlm_probability_train = rho
        args.mlm_probability = rho
        args.model_idx = f"optuna_trial_{trial.number}"

        checkpoint = f"{args.model_name}-{args.dataset}-{args.model_idx}.pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        set_seed(args.seed)
        data_generator = DataGenerator(args)
        trainer = Trainer(args, device, data_generator)
        summary = trainer.train(
            trial=trial,
            objective_metric=cli.objective_metric,
            enable_pruning=(cli.pruner != "none"),
            early_stop_rounds=cli.early_stop_rounds,
            eval_interval=cli.eval_interval,
            run_test_during_tuning=cli.run_test_during_search,
        )

        valid_metrics = summary["best_valid_metrics"] or {}
        test_metrics = summary["best_test_metrics"] or {}
        metric_value = float(valid_metrics.get(cli.objective_metric, 0.0))

        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("best_valid_metrics", valid_metrics)
        trial.set_user_attr("best_test_metrics", test_metrics)
        trial.set_user_attr("alpha", args.alpha)
        trial.set_user_attr("beta", args.beta)
        trial.set_user_attr("rho", rho)

        del trainer
        del data_generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metric_value

    sampler = optuna.samplers.TPESampler(seed=cli.seed)
    pruner = None
    if cli.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cli.n_startup_trials,
            n_warmup_steps=cli.n_warmup_steps,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=cli.study_name,
        storage=cli.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=cli.n_trials,
        timeout=cli.timeout,
        n_jobs=cli.n_jobs,
        show_progress_bar=False,
    )

    best = study.best_trial
    print("[BEST] trial:", best.number)
    print("[BEST] params:", best.params)
    print(f"[BEST] {cli.objective_metric}: {best.value * 100:.5f}")
    print("[BEST] user_attrs:", best.user_attrs)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "study_name": cli.study_name,
        "objective_metric": cli.objective_metric,
        "best_trial": best.number,
        "best_params": best.params,
        "best_value": best.value,
        "best_value_percent": f"{best.value * 100:.5f}",
        "best_user_attrs": best.user_attrs,
    }
    with open(cli.results_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved best result to {cli.results_json}")


if __name__ == "__main__":
    main()
