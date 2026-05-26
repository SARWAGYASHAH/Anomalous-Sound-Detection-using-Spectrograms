"""Run preprocessing, training, evaluation, and optional prediction stages.

Full training is intended for Colab. Locally, use ``--train-dry-run`` for the
training stage or select only inference/preprocessing stages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the Keras audio anomaly pipeline")
    parser.add_argument("--config", default="config/default.yaml", help="Config or experiment override YAML.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["preprocess", "train", "evaluate", "predict"],
        default=["preprocess", "train", "evaluate"],
        help="Stages to run in sequence.",
    )
    parser.add_argument("--model-path", default=None, help="Saved model for evaluation/prediction.")
    parser.add_argument("--predict-file", default=None, help="WAV file to score in the optional prediction stage.")
    parser.add_argument("--save-png", action="store_true", help="Save PNG previews during preprocessing.")
    parser.add_argument("--train-dry-run", action="store_true", help="Use a one-batch no-training check for stage 2.")
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Permit model.fit() for stage 2. Use this in the Colab training workflow.",
    )
    return parser.parse_args()


def run_stage(name: str, arguments: list[str]) -> None:
    script = {
        "preprocess": "pipeline/01_preprocess.py",
        "train": "pipeline/02_train.py",
        "evaluate": "pipeline/03_evaluate.py",
        "predict": "pipeline/04_predict.py",
    }[name]
    command = [sys.executable, script, *arguments]
    print(f"\nRunning {name}: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    stages = list(args.stages)
    if args.predict_file and "predict" not in stages:
        stages.append("predict")

    if "train" in stages and not (args.allow_training or args.train_dry_run):
        raise SystemExit(
            "Stage 'train' uses model.fit(). Run this pipeline in Colab with --allow-training, "
            "or pass --train-dry-run for a local one-batch check."
        )
    if "predict" in stages and not args.predict_file:
        raise SystemExit("Stage 'predict' requires --predict-file path/to/audio.wav.")

    for stage in stages:
        common_args = ["--config", args.config]
        if stage == "preprocess":
            run_stage(stage, common_args + (["--save-png"] if args.save_png else []))
        elif stage == "train":
            train_args = common_args + (["--dry-run", "--no-mlflow"] if args.train_dry_run else [])
            run_stage(stage, train_args)
        elif stage == "evaluate":
            model_args = ["--model-path", args.model_path] if args.model_path else []
            run_stage(stage, common_args + model_args)
        elif stage == "predict":
            model_args = ["--model-path", args.model_path] if args.model_path else []
            run_stage(stage, common_args + ["--file", args.predict_file] + model_args)


if __name__ == "__main__":
    main()
