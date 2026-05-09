"""
03_fit_latent_scorer.py - Fit latent PCA/Mahalanobis anomaly scorer.

This stage uses a trained Keras autoencoder and normal training spectrograms.
It does not retrain the autoencoder. It fits lightweight scoring parameters
that can be used by 03_evaluate.py with --score-method latent.

Usage:
    python pipeline/03_fit_latent_scorer.py --model-path artifacts/models/v2/best_model.keras
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_autoencoder_dataset, discover_spectrogram_files, load_spectrogram_array  # noqa: E402
from src.inference import LatentMahalanobisScorer  # noqa: E402
from src.models import Conv2DAutoencoder  # noqa: E402
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
    """Load base config and optionally merge an override config."""
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if override_path and Path(override_path).resolve() != Path(base_path).resolve():
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)

    return config


def infer_input_shape(train_dir: Path) -> tuple[int, int, int]:
    """Infer channels-last input shape from the first normal spectrogram."""
    filepaths, _ = discover_spectrogram_files(train_dir, labels=["normal"])
    if not filepaths:
        raise FileNotFoundError(f"No normal spectrograms found in {train_dir / 'normal'}")

    sample = load_spectrogram_array(filepaths[0], add_channel=True)
    return tuple(sample.shape)


def default_output_path(model_path: Path) -> Path:
    """Resolve default params path beside the model version directory."""
    return model_path.parent / "scoring" / "latent_mahalanobis.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit latent PCA/Mahalanobis scorer")
    parser.add_argument("--model-path", required=True, help="Path to trained .keras model.")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override fitting batch size.")
    parser.add_argument("--pca-components", type=int, default=None, help="Override PCA component count.")
    parser.add_argument(
        "--covariance-regularization",
        type=float,
        default=None,
        help="Small diagonal value added before covariance inversion.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Destination .npz path. Defaults beside the model under scoring/.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache the normal training dataset in memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(base_path=args.base_config, override_path=args.config)
    set_seed(config["seed"])

    logger = get_logger("fit_latent_scorer", log_dir=config["artifacts"]["logs_dir"])
    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 3A: Fit Latent Mahalanobis Scorer")
    logger.info("=" * 60)
    logger.info("TensorFlow version: %s", tf.__version__)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    processed_dir = Path(config["data"]["processed_dir"])
    train_dir = processed_dir / "train"
    input_shape = infer_input_shape(train_dir)
    input_hw = input_shape[:2]

    latent_config = config.get("inference", {}).get("latent_scorer", {})
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    pca_components = args.pca_components or int(latent_config.get("pca_components", 32))
    covariance_regularization = (
        args.covariance_regularization
        if args.covariance_regularization is not None
        else float(latent_config.get("covariance_regularization", 1e-6))
    )
    output_path = Path(args.output_path) if args.output_path else default_output_path(model_path)

    logger.info("Model path: %s", model_path)
    logger.info("Training data directory: %s", train_dir)
    logger.info("Input shape: %s", input_shape)
    logger.info("PCA components: %s", pca_components)
    logger.info("Output path: %s", output_path)

    normal_dataset = create_autoencoder_dataset(
        train_dir,
        batch_size=batch_size,
        shuffle=False,
        input_shape=input_hw,
        cache=args.cache,
    )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Conv2DAutoencoder": Conv2DAutoencoder},
        compile=False,
    )

    scorer = LatentMahalanobisScorer.fit(
        model=model,
        normal_dataset=normal_dataset,
        pca_components=pca_components,
        covariance_regularization=covariance_regularization,
    )
    saved_path = scorer.save(output_path)

    logger.info("Latent scorer fitting complete.")
    logger.info("Saved scorer params: %s", saved_path)


if __name__ == "__main__":
    main()
