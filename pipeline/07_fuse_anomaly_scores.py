"""Fuse v2 reconstruction and STgram-MFN anomaly scores.

Usage:
    python pipeline/07_fuse_anomaly_scores.py \
        --v2-scores artifacts/evaluation/source_test/scores.csv \
        --stgram-scores artifacts/evaluation_stgram/source_test/scores.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import evaluate_all  # noqa: E402
from src.utils.score_fusion import align_score_frames, normalize_scores, sweep_weighted_fusion  # noqa: E402
from src.utils.visualization import plot_anomaly_scores, plot_roc_curve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse anomaly score CSVs from v2 and STgram-MFN.")
    parser.add_argument("--v2-scores", default="artifacts/evaluation/source_test/scores.csv")
    parser.add_argument("--stgram-scores", default="artifacts/evaluation_stgram/source_test/scores.csv")
    parser.add_argument("--output-dir", default="artifacts/evaluation_fusion/source_test")
    parser.add_argument("--normalization", choices=["zscore", "minmax", "rank", "none"], default="rank")
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--v2-weight", type=float, default=None, help="Use a fixed v2 weight instead of sweeping.")
    parser.add_argument("--assets-dir", default=None, help="Optional directory for plot copies.")
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    v2_path = Path(args.v2_scores)
    stgram_path = Path(args.stgram_scores)
    if not v2_path.exists():
        raise FileNotFoundError(f"v2 score CSV not found: {v2_path}")
    if not stgram_path.exists():
        raise FileNotFoundError(f"STgram score CSV not found: {stgram_path}")

    aligned = align_score_frames(
        pd.read_csv(v2_path),
        pd.read_csv(stgram_path),
        left_name="v2",
        right_name="stgram",
    )
    labels = aligned["label"].to_numpy(dtype=np.int64)
    v2_scores = normalize_scores(aligned["v2_score"].to_numpy(dtype=np.float64), args.normalization)
    stgram_scores = normalize_scores(aligned["stgram_score"].to_numpy(dtype=np.float64), args.normalization)

    if args.v2_weight is None:
        best_weight, best_auc, fused_scores, sweep = sweep_weighted_fusion(
            labels,
            v2_scores,
            stgram_scores,
            step=float(args.weight_step),
        )
    else:
        best_weight = float(args.v2_weight)
        fused_scores = (best_weight * v2_scores) + ((1.0 - best_weight) * stgram_scores)
        best_auc = float(evaluate_all(labels, fused_scores, threshold=None)["auc_roc"])
        sweep = pd.DataFrame(
            [{"v2_weight": best_weight, "stgram_weight": 1.0 - best_weight, "auc_roc": best_auc}]
        )

    metrics = evaluate_all(labels, fused_scores, threshold=None)
    fpr, tpr, _ = roc_curve(labels, fused_scores)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.csv"
    metrics_path = output_dir / "metrics.json"
    sweep_path = output_dir / "weight_sweep.csv"
    score_plot_path = output_dir / "anomaly_score_distribution.png"
    roc_plot_path = output_dir / "roc_curve.png"

    aligned.assign(
        v2_score_normalized=v2_scores,
        stgram_score_normalized=stgram_scores,
        score=fused_scores,
        binary_prediction=(fused_scores >= metrics["threshold"]).astype(int),
    ).to_csv(scores_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    plot_anomaly_scores(fused_scores, labels=labels, threshold=metrics["threshold"], save_path=str(score_plot_path))
    plot_roc_curve(fpr, tpr, metrics["auc_roc"], title="Fused Anomaly Scores ROC", save_path=str(roc_plot_path))

    if args.assets_dir:
        assets_dir = Path(args.assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        plot_anomaly_scores(
            fused_scores,
            labels=labels,
            threshold=metrics["threshold"],
            save_path=str(assets_dir / "source_test_fused_anomaly_scores.png"),
        )
        plot_roc_curve(
            fpr,
            tpr,
            metrics["auc_roc"],
            title="Fused Anomaly Scores ROC",
            save_path=str(assets_dir / "source_test_fused_roc_curve.png"),
        )

    payload = {
        "v2_scores": str(v2_path),
        "stgram_scores": str(stgram_path),
        "normalization": args.normalization,
        "best_v2_weight": best_weight,
        "best_stgram_weight": 1.0 - best_weight,
        "best_swept_auc": best_auc,
        "metrics": metrics,
        "artifacts": {
            "scores_csv": str(scores_path),
            "weight_sweep_csv": str(sweep_path),
            "score_plot": str(score_plot_path),
            "roc_plot": str(roc_plot_path),
        },
    }
    save_json(payload, metrics_path)

    print(f"Best v2 weight: {best_weight:.2f}")
    print(f"Best STgram weight: {1.0 - best_weight:.2f}")
    print(f"Fused AUC: {metrics['auc_roc']:.4f}")
    print(f"Fused pAUC: {metrics['pauc']:.4f}")
    print(f"Fused F1: {metrics['f1']:.4f}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
