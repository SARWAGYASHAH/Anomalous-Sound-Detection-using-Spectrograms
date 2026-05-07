"""
03_evaluate.py - Evaluate a trained Keras autoencoder on labeled test data.

Usage:
    python pipeline/03_evaluate.py --model-path artifacts/models/v2/best_model.keras
    python pipeline/03_evaluate.py --model-path artifacts/models/v2/best_model.keras --split target_test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_labeled_dataset  # noqa: E402
from src.inference import (  # noqa: E402
    ReconstructionAnomalyScorer,
    SeverityClassifier,
    summarize_scores,
    threshold_from_config,
)
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import evaluate_all  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="03_evaluate: Evaluate Keras autoencoder")
    parser.add_argument("--model-path", required=True, help="Path to trained .keras model.")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML path.")
    parser.add_argument(
        "--split",
        choices=["source_test", "target_test"],
        default="source_test",
        help="Processed test split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation",
        help="Directory for metrics, score CSVs, and plots.",
    )
    parser.add_argument(
        "--assets-dir",
        default="docs/assets",
        help="Directory for README-friendly plot copies.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(normal_scores, bins=50, alpha=0.65, density=True, label=f"Normal (n={len(normal_scores)})")
    ax.hist(anomaly_scores, bins=50, alpha=0.65, density=True, label=f"Anomaly (n={len(anomaly_scores)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.6f}")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    auc_score: float,
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    args = parse_args()
    config = load_config(base_path=args.base_config, override_path=args.config)
    set_seed(config["seed"])

    logger = get_logger("evaluate", log_dir=config["artifacts"]["logs_dir"])
    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 3: Evaluation")
    logger.info("=" * 60)
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("Model path: %s", args.model_path)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    input_shape = (int(config["spectrogram"]["n_mels"]), None)
    split_dir = Path(config["data"]["processed_dir"]) / args.split

    dataset = create_labeled_dataset(
        split_dir,
        batch_size=batch_size,
        shuffle=False,
        input_shape=input_shape,
    )

    scorer = ReconstructionAnomalyScorer.from_model_path(str(model_path))
    scores, labels = scorer.score_dataset(dataset)

    normal_scores = scores[labels == 0]
    threshold = threshold_from_config(normal_scores, config)
    classifier = SeverityClassifier.from_config(threshold, config)
    binary_predictions = classifier.to_binary_predictions(scores)
    severity_labels = classifier.classify_scores(scores)

    metrics = evaluate_all(labels, scores, threshold=threshold)
    normal_summary = summarize_scores(normal_scores)
    anomaly_summary = summarize_scores(scores[labels == 1])

    output_dir = ensure_dir(Path(args.output_dir) / args.split)
    assets_dir = ensure_dir(args.assets_dir)

    score_table = pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "binary_prediction": binary_predictions,
            "severity": severity_labels,
        }
    )
    score_csv_path = output_dir / "scores.csv"
    score_table.to_csv(score_csv_path, index=False)

    metrics_payload = {
        "split": args.split,
        "model_path": str(model_path),
        "threshold": float(threshold),
        "metrics": metrics,
        "normal_score_summary": normal_summary.__dict__,
        "anomaly_score_summary": anomaly_summary.__dict__,
        "num_samples": int(len(scores)),
        "num_normal": int(np.sum(labels == 0)),
        "num_anomaly": int(np.sum(labels == 1)),
    }
    metrics_path = output_dir / "metrics.json"
    save_json(metrics_payload, metrics_path)

    score_plot_path = output_dir / "anomaly_score_distribution.png"
    roc_plot_path = output_dir / "roc_curve.png"
    plot_score_distribution(scores, labels, threshold, score_plot_path)
    plot_roc(labels, scores, metrics["auc_roc"], roc_plot_path)

    readme_score_plot = assets_dir / f"{args.split}_anomaly_scores.png"
    readme_roc_plot = assets_dir / f"{args.split}_roc_curve.png"
    plot_score_distribution(scores, labels, threshold, readme_score_plot)
    plot_roc(labels, scores, metrics["auc_roc"], readme_roc_plot)

    logger.info("Evaluation complete.")
    logger.info("Metrics: %s", metrics)
    logger.info("Score CSV: %s", score_csv_path)
    logger.info("Metrics JSON: %s", metrics_path)
    logger.info("Plots: %s, %s", score_plot_path, roc_plot_path)
    logger.info("README assets: %s, %s", readme_score_plot, readme_roc_plot)


if __name__ == "__main__":
    main()
