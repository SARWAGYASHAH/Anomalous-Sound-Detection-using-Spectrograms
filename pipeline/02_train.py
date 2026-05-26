"""
02_train.py - Train the Keras autoencoder.

Designed for Google Colab/full GPU training. Local use should stay limited to
--dry-run checks, which build the datasets/model and run one forward pass
without calling model.fit().

Usage:
    python pipeline/02_train.py --config config/default.yaml
    python pipeline/02_train.py --config config/experiment_01.yaml
    python pipeline/02_train.py --dry-run
    python pipeline/02_train.py --epochs 5 --no-mlflow
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

# Add project root to path for Colab and direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (  # noqa: E402
    create_autoencoder_train_val_datasets,
    discover_spectrogram_files,
    load_spectrogram_array,
)
from src.models import build_autoencoder  # noqa: E402
from src.training import KerasTrainer  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    base_path: str = "config/default.yaml",
    override_path: str | None = None,
) -> dict[str, Any]:
    """
    Load the base config and optionally merge an experiment override.

    Passing config/default.yaml as --config simply loads the base config.
    Passing another file, such as config/experiment_01.yaml, merges it on top.
    """
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if override_path and Path(override_path).resolve() != Path(base_path).resolve():
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)

    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply simple CLI overrides without mutating the caller's config."""
    updated = copy.deepcopy(config)

    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        updated["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        updated["training"]["learning_rate"] = args.learning_rate
    if args.no_mlflow:
        updated.setdefault("mlflow", {})["enabled"] = False

    return updated


def infer_input_shape(train_dir: Path) -> tuple[int, int, int]:
    """Infer channels-last input shape from the first normal spectrogram."""
    filepaths, _ = discover_spectrogram_files(train_dir, labels=["normal"])

    if not filepaths:
        raise FileNotFoundError(f"No normal spectrograms found in {train_dir / 'normal'}")

    sample = load_spectrogram_array(filepaths[0], add_channel=True)
    return tuple(sample.shape)


def describe_runtime(logger) -> None:
    """Log TensorFlow runtime information relevant for Colab."""
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("GPU devices: %s", [gpu.name for gpu in gpus] if gpus else "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="02_train: Train Keras autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Config YAML. default.yaml is loaded first; non-default files are merged as overrides.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="config/default.yaml",
        help="Base config YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow for this run.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache tf.data datasets in memory. Useful in Colab when RAM allows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data/model/trainer and run one forward pass without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(
        load_config(base_path=args.base_config, override_path=args.config),
        args,
    )

    set_seed(config["seed"])
    logger = get_logger("train", log_dir=config["artifacts"]["logs_dir"])

    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 2: Training (Keras Autoencoder)")
    logger.info("=" * 60)
    describe_runtime(logger)

    processed_dir = Path(config["data"]["processed_dir"])
    train_dir = processed_dir / "train"

    input_shape = infer_input_shape(train_dir)
    input_hw = input_shape[:2]
    batch_size = int(config["training"]["batch_size"])

    logger.info("Training data directory: %s", train_dir)
    logger.info("Input shape: %s", input_shape)
    logger.info("Batch size: %s", batch_size)

    train_dataset, val_dataset, counts = create_autoencoder_train_val_datasets(
        train_dir,
        batch_size=batch_size,
        train_split=float(config["data"]["train_split"]),
        seed=int(config["seed"]),
        input_shape=input_hw,
        cache=args.cache,
    )

    model = build_autoencoder(config["model"]["autoencoder"], input_shape=input_shape)
    temp_context = tempfile.TemporaryDirectory() if args.dry_run else None
    trainer_version_dir = Path(temp_context.name) / "dry_run_model" if temp_context else None

    trainer = KerasTrainer(model=model, config=config, version_dir=trainer_version_dir)
    trainer.compile()

    x_batch, y_batch = next(iter(train_dataset))
    output_batch = model(x_batch, training=False)

    logger.info("Train/val counts: %s", counts)
    logger.info("Dry batch input shape: %s", tuple(x_batch.shape))
    logger.info("Dry batch target shape: %s", tuple(y_batch.shape))
    logger.info("Dry batch output shape: %s", tuple(output_batch.shape))
    logger.info("Model parameters: %s", model.count_parameters())

    if args.dry_run:
        logger.info("Dry run complete. No training was executed.")
        if temp_context:
            temp_context.cleanup()
        return

    history, artifacts = trainer.fit(train_dataset, val_dataset)

    logger.info("Training history keys: %s", list(history.history.keys()))
    logger.info("Model: %s", artifacts.model_path)
    logger.info("Best model: %s", artifacts.best_model_path)
    logger.info("Final model: %s", artifacts.final_model_path)
    logger.info("Metadata: %s", artifacts.metadata_path)


if __name__ == "__main__":
    main()
