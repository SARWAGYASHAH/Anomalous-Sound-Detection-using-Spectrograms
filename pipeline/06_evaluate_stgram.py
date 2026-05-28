"""Evaluate STgram-MFN + per-section GMM anomaly scores.

Usage:
    python pipeline/06_evaluate_stgram.py --artifact-dir artifacts/models/stgram_mfn/run_...
    python pipeline/06_evaluate_stgram.py --split target_test
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.stgram_dataset import STgramWaveDataset, discover_wav_files  # noqa: E402
from src.models.stgram_mfn import build_stgram_mfn  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import evaluate_all  # noqa: E402
from src.utils.visualization import plot_anomaly_scores, plot_roc_curve  # noqa: E402


@dataclass(frozen=True)
class ThresholdResult:
    value: float
    method: str
    percentile: float | None = None
    mean: float | None = None
    std: float | None = None
    std_multiplier: float | None = None


@dataclass(frozen=True)
class ScoreSummary:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    percentile_95: float


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(base_path: str, override_path: str | None) -> dict[str, Any]:
    with open(base_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if override_path and Path(override_path).resolve() != Path(base_path).resolve():
        with open(override_path, "r", encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STgram-MFN anomaly detector")
    parser.add_argument("--config", default="config/stgram_mfn.yaml", help="STgram config YAML.")
    parser.add_argument("--base-config", default="config/default.yaml", help="Base config YAML.")
    parser.add_argument("--artifact-dir", default=None, help="Training run directory. Defaults to latest.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path. Defaults to best_model.pt.")
    parser.add_argument("--gmm-path", default=None, help="GMM joblib path. Defaults to run gmm_per_section.joblib.")
    parser.add_argument(
        "--split",
        choices=["source_test", "target_test"],
        default="source_test",
        help="Raw WAV test split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--output-dir", default=None, help="Root output directory.")
    parser.add_argument("--assets-dir", default=None, help="Optional directory for plot copies.")
    return parser.parse_args()


def resolve_artifact_dir(config: dict[str, Any], artifact_dir: str | None) -> Path:
    if artifact_dir:
        return Path(artifact_dir)
    output_root = Path(config.get("stgram", {}).get("output_dir", "artifacts/models/stgram_mfn"))
    latest_path = output_root / "latest.txt"
    if latest_path.exists():
        latest = latest_path.read_text(encoding="utf-8").strip()
        if latest:
            return Path(latest)
    runs = sorted(path for path in output_root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No STgram run found in {output_root}")
    return runs[-1]


def raw_split_dir(config: dict[str, Any], split: str) -> Path:
    stgram = config.get("stgram", {})
    explicit_key = f"raw_{split}_dir"
    if explicit_key in stgram:
        return Path(stgram[explicit_key])
    return Path("Data") / config["data"]["machine_type"] / split


def create_dataset(config: dict[str, Any], split: str, section_to_label: dict[str, int]) -> STgramWaveDataset:
    stgram_config = config["model"]["stgram_mfn"]
    files = discover_wav_files(raw_split_dir(config, split))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {raw_split_dir(config, split)}")
    return STgramWaveDataset(
        files,
        section_to_label=section_to_label,
        sample_rate=int(config["audio"]["sample_rate"]),
        duration=float(config["audio"]["duration"]),
        n_fft=int(config["spectrogram"]["n_fft"]),
        hop_length=int(config["spectrogram"]["hop_length"]),
        n_mels=int(stgram_config.get("n_mels", config["spectrogram"]["n_mels"])),
        power=float(config["spectrogram"]["power"]),
        normalize_mel=bool(config.get("stgram", {}).get("normalize_mel", True)),
        require_binary_label=split != "train",
    )


def create_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_torch_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def threshold_from_config(scores: np.ndarray, config: dict[str, Any]) -> ThresholdResult:
    inference_config = config.get("inference", {})
    method = str(inference_config.get("threshold_method", "percentile")).lower()
    if method == "percentile":
        percentile = float(inference_config.get("percentile", 95.0))
        return ThresholdResult(
            value=float(np.percentile(scores, percentile)),
            method="percentile",
            percentile=percentile,
        )
    if method == "mean_std":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        multiplier = float(inference_config.get("std_multiplier", 2.0))
        return ThresholdResult(
            value=mean + multiplier * std,
            method="mean_std",
            mean=mean,
            std=std,
            std_multiplier=multiplier,
        )
    if method == "fixed":
        fixed = inference_config.get("fixed_threshold")
        if fixed is None:
            raise ValueError("inference.fixed_threshold is required when threshold_method is fixed")
        return ThresholdResult(value=float(fixed), method="fixed")
    raise ValueError(f"Unsupported threshold method: {method}")


def summarize_scores(scores: np.ndarray) -> ScoreSummary:
    scores = np.asarray(scores, dtype=np.float64)
    return ScoreSummary(
        count=int(scores.size),
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        minimum=float(np.min(scores)),
        maximum=float(np.max(scores)),
        percentile_95=float(np.percentile(scores, 95.0)),
    )


def set_torch_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_loader(model, loader, gmms: dict[int, Any], device) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    sections: list[str] = []
    filepaths: list[str] = []
    section_labels: list[int] = []
    for batch in tqdm(loader, desc="score"):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        mels = batch["mel"].to(device, non_blocking=True)
        batch_section_labels = batch["section_label"].cpu().numpy()
        features = model.extract_features(waveforms, mels).cpu().numpy()
        for feature, section_label, binary_label, section, filepath in zip(
            features,
            batch_section_labels,
            batch["binary_label"].cpu().numpy(),
            batch["section"],
            batch["filepath"],
        ):
            section_label = int(section_label)
            if section_label not in gmms:
                raise KeyError(f"No GMM fitted for section label {section_label}")
            score = -float(gmms[section_label].score_samples(feature.reshape(1, -1))[0])
            scores.append(score)
            labels.append(int(binary_label))
            sections.append(section)
            filepaths.append(filepath)
            section_labels.append(section_label)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        sections,
        filepaths,
        np.asarray(section_labels, dtype=np.int64),
    )


def load_checkpoint_config(default_config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        return checkpoint_config
    return default_config


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    set_torch_seed(int(config["seed"]))
    logger = get_logger("evaluate_stgram", log_dir=config["artifacts"]["logs_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifact_dir = resolve_artifact_dir(config, args.artifact_dir)
    model_path = Path(args.model_path) if args.model_path else artifact_dir / "best_model.pt"
    gmm_path = Path(args.gmm_path) if args.gmm_path else artifact_dir / "gmm_per_section.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not gmm_path.exists():
        raise FileNotFoundError(f"GMM artifact not found: {gmm_path}")

    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    config = load_checkpoint_config(config, checkpoint)
    section_to_label = checkpoint["section_to_label"]
    model = build_stgram_mfn(config["model"]["stgram_mfn"], num_classes=len(section_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    gmms = joblib.load(gmm_path)

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(config.get("stgram", {}).get("num_workers", 2))
    test_dataset = create_dataset(config, args.split, section_to_label)
    train_dataset = create_dataset(config, "train", section_to_label)
    test_loader = create_loader(test_dataset, batch_size=batch_size, workers=workers)
    train_loader = create_loader(train_dataset, batch_size=batch_size, workers=workers)

    logger.info("=" * 60)
    logger.info("PIPELINE STAGE 6: Evaluation (STgram-MFN)")
    logger.info("=" * 60)
    logger.info("Device: %s", device)
    logger.info("Artifact dir: %s", artifact_dir)
    logger.info("Model: %s", model_path)
    logger.info("GMM: %s", gmm_path)
    logger.info("Split: %s", args.split)

    reference_scores, _, _, _, _ = score_loader(model, train_loader, gmms, device)
    threshold = threshold_from_config(reference_scores, config)
    scores, labels, sections, filepaths, section_labels = score_loader(model, test_loader, gmms, device)
    if np.any(labels < 0):
        raise ValueError(f"Evaluation split {args.split} did not provide binary normal/anomaly labels")

    metrics = evaluate_all(labels, scores, threshold=threshold.value)
    optimal_metrics = evaluate_all(labels, scores, threshold=None)
    fpr, tpr, _ = roc_curve(labels, scores)

    output_root = Path(args.output_dir or config.get("stgram", {}).get("evaluation_dir", "artifacts/evaluation_stgram"))
    output_dir = output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.csv"
    metrics_path = output_dir / "metrics.json"
    score_plot_path = output_dir / "anomaly_score_distribution.png"
    roc_plot_path = output_dir / "roc_curve.png"

    pd.DataFrame(
        {
            "filepath": filepaths,
            "section": sections,
            "section_label": section_labels,
            "label": labels,
            "score": scores,
            "binary_prediction": (scores >= threshold.value).astype(int),
        }
    ).to_csv(scores_path, index=False)
    plot_anomaly_scores(scores, labels=labels, threshold=threshold.value, save_path=str(score_plot_path))
    plot_roc_curve(fpr, tpr, metrics["auc_roc"], title=f"STgram-MFN ROC - {args.split}", save_path=str(roc_plot_path))

    if args.assets_dir:
        assets_dir = Path(args.assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        plot_anomaly_scores(
            scores,
            labels=labels,
            threshold=threshold.value,
            save_path=str(assets_dir / f"{args.split}_stgram_anomaly_scores.png"),
        )
        plot_roc_curve(
            fpr,
            tpr,
            metrics["auc_roc"],
            title=f"STgram-MFN ROC - {args.split}",
            save_path=str(assets_dir / f"{args.split}_stgram_roc_curve.png"),
        )

    payload = {
        "split": args.split,
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "gmm_path": str(gmm_path),
        "threshold": asdict(threshold),
        "metrics": metrics,
        "optimal_threshold_metrics": optimal_metrics,
        "test_scores": asdict(summarize_scores(scores)),
        "normal_reference_scores": asdict(summarize_scores(reference_scores)),
        "normal_test_scores": asdict(summarize_scores(scores[labels == 0])),
        "anomaly_test_scores": asdict(summarize_scores(scores[labels == 1])),
        "artifacts": {
            "scores_csv": str(scores_path),
            "score_plot": str(score_plot_path),
            "roc_plot": str(roc_plot_path),
        },
    }
    save_json(payload, metrics_path)
    logger.info("Evaluation complete for %s: AUC=%.4f F1=%.4f", args.split, metrics["auc_roc"], metrics["f1"])
    logger.info("Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
