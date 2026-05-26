"""Evaluate a trained Keras autoencoder on labeled processed spectrograms.

Usage:
    python pipeline/03_evaluate.py --model-path artifacts/models/v2/best_model.keras
    python pipeline/03_evaluate.py --split target_test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import create_autoencoder_dataset, create_labeled_dataset  # noqa: E402
from src.inference import (  # noqa: E402
    ReconstructionAnomalyScorer,
    SeverityClassifier,
    classify_anomalies,
    score_to_probability,
    summarize_scores,
    threshold_from_config,
)
from src.inference.predictor import load_project_model, resolve_latest_model_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metadata_tracker import MetadataTracker  # noqa: E402
from src.utils.metrics import evaluate_all  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.visualization import plot_anomaly_scores, plot_roc_curve  # noqa: E402


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge an experiment override on top of base settings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(base_path: str, override_path: str | None) -> dict[str, Any]:
    """Load default configuration and an optional override file."""
    with open(base_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if override_path and Path(override_path).resolve() != Path(base_path).resolve():
        with open(override_path, "r", encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="03_evaluate: Evaluate Keras autoencoder")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML path.")
    parser.add_argument("--model-path", default=None, help="Saved .keras model. Defaults to latest artifact model.")
    parser.add_argument(
        "--split",
        choices=["source_test", "target_test"],
        default="source_test",
        help="Processed test split to evaluate.",
    )
    parser.add_argument("--reference-dir", default=None, help="Normal reference split for threshold fitting.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--output-dir", default="artifacts/evaluation", help="Root output directory.")
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Optional directory receiving plot copies, for example docs/assets.",
    )
    return parser.parse_args()


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    set_seed(int(config["seed"]))
    logger = get_logger("evaluate", log_dir=config["artifacts"]["logs_dir"])

    model_path = Path(args.model_path) if args.model_path else resolve_latest_model_path(config["artifacts"]["models_dir"])
    model = load_project_model(model_path)
    scorer = ReconstructionAnomalyScorer(model=model)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    input_shape = (int(config["spectrogram"]["n_mels"]), None)

    split_dir = Path(config["data"]["processed_dir"]) / args.split
    test_dataset = create_labeled_dataset(split_dir, batch_size=batch_size, input_shape=input_shape)
    scores, labels = scorer.score_dataset(test_dataset)
    if labels is None:
        raise ValueError(f"Evaluation split does not include labels: {split_dir}")

    reference_dir = Path(args.reference_dir or Path(config["data"]["processed_dir"]) / "train")
    reference_dataset = create_autoencoder_dataset(
        reference_dir,
        batch_size=batch_size,
        shuffle=False,
        input_shape=input_shape,
    )
    normal_reference_scores, _ = scorer.score_dataset(reference_dataset)
    threshold = threshold_from_config(normal_reference_scores, config)

    binary_predictions = classify_anomalies(scores, threshold.value)
    normalized_scores = score_to_probability(scores, threshold.value)
    severity_classifier = SeverityClassifier.from_config(config)
    severity = severity_classifier.predict(normalized_scores)
    requires_attention = severity != severity_classifier.labels[0]
    metrics = evaluate_all(labels, scores, threshold=threshold.value)

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.csv"
    metrics_path = output_dir / "metrics.json"
    score_plot_path = output_dir / "anomaly_score_distribution.png"
    roc_plot_path = output_dir / "roc_curve.png"

    pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "normalized_score": normalized_scores,
            "binary_prediction": binary_predictions,
            "severity": severity,
            "requires_attention": requires_attention,
        }
    ).to_csv(scores_path, index=False)

    fpr, tpr, _ = roc_curve(labels, scores)
    plot_anomaly_scores(scores, labels=labels, threshold=threshold.value, save_path=str(score_plot_path))
    plot_roc_curve(fpr, tpr, metrics["auc_roc"], save_path=str(roc_plot_path))

    if args.assets_dir:
        assets_dir = Path(args.assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        plot_anomaly_scores(
            scores,
            labels=labels,
            threshold=threshold.value,
            save_path=str(assets_dir / f"{args.split}_anomaly_scores.png"),
        )
        plot_roc_curve(
            fpr,
            tpr,
            metrics["auc_roc"],
            save_path=str(assets_dir / f"{args.split}_roc_curve.png"),
        )

    payload = {
        "split": args.split,
        "model_path": str(model_path),
        "reference_dir": str(reference_dir),
        "threshold": asdict(threshold),
        "metrics": metrics,
        "test_scores": asdict(summarize_scores(scores)),
        "normal_reference_scores": asdict(summarize_scores(normal_reference_scores)),
        "normal_test_scores": asdict(summarize_scores(scores[labels == 0])),
        "anomaly_test_scores": asdict(summarize_scores(scores[labels == 1])),
        "artifacts": {
            "scores_csv": str(scores_path),
            "score_plot": str(score_plot_path),
            "roc_plot": str(roc_plot_path),
        },
    }
    save_json(payload, metrics_path)

    run_id = "evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = MetadataTracker(config["artifacts"]["metadata_dir"]).save(
        run_id=run_id,
        params={"model_path": str(model_path), "split": args.split, "threshold_method": threshold.method},
        metrics=metrics,
        artifacts={"metrics_json": str(metrics_path), **payload["artifacts"]},
        experiment_name=config.get("mlflow", {}).get("experiment_name"),
    )

    logger.info("Evaluation complete for %s: AUC=%.4f, F1=%.4f", args.split, metrics["auc_roc"], metrics["f1"])
    logger.info("Threshold (%s): %.8f", threshold.method, threshold.value)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
