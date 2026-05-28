"""Utilities for combining anomaly score files from different models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def normalize_scores(scores: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a score vector while preserving score ordering."""
    scores = np.asarray(scores, dtype=np.float64)
    method = method.lower()
    if method == "none":
        return scores
    if method == "zscore":
        std = float(scores.std())
        if std <= 1e-12:
            return scores - float(scores.mean())
        return (scores - float(scores.mean())) / std
    if method == "minmax":
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum - minimum <= 1e-12:
            return scores - minimum
        return (scores - minimum) / (maximum - minimum)
    if method == "rank":
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
        if len(ranks) <= 1:
            return np.zeros_like(scores)
        return (ranks - 1.0) / (len(ranks) - 1.0)
    raise ValueError(f"Unknown score normalization method: {method}")


def align_score_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """Align two score frames by filepath when possible, otherwise by row order."""
    required = {"label", "score"}
    for name, frame in ((left_name, left), (right_name, right)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} scores are missing columns: {sorted(missing)}")

    if "filepath" in left.columns and "filepath" in right.columns:
        left_keyed = left.assign(_key=left["filepath"].astype(str).map(_basename))
        right_keyed = right.assign(_key=right["filepath"].astype(str).map(_basename))
        merged = left_keyed[["_key", "label", "score"]].merge(
            right_keyed[["_key", "label", "score"]],
            on="_key",
            suffixes=(f"_{left_name}", f"_{right_name}"),
        )
        if len(merged) == 0:
            raise ValueError("Could not align score files by filepath basename")
        labels_left = merged[f"label_{left_name}"].to_numpy()
        labels_right = merged[f"label_{right_name}"].to_numpy()
        if not np.array_equal(labels_left, labels_right):
            raise ValueError("Aligned score files have mismatched labels")
        return pd.DataFrame(
            {
                "label": labels_left.astype(np.int64),
                f"{left_name}_score": merged[f"score_{left_name}"].to_numpy(dtype=np.float64),
                f"{right_name}_score": merged[f"score_{right_name}"].to_numpy(dtype=np.float64),
                "key": merged["_key"],
            }
        )

    if len(left) != len(right):
        raise ValueError(f"Cannot align by row order: {left_name} has {len(left)} rows, {right_name} has {len(right)}")
    labels_left = left["label"].to_numpy(dtype=np.int64)
    labels_right = right["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels_left, labels_right):
        raise ValueError("Score files have mismatched row-order labels")
    return pd.DataFrame(
        {
            "label": labels_left,
            f"{left_name}_score": left["score"].to_numpy(dtype=np.float64),
            f"{right_name}_score": right["score"].to_numpy(dtype=np.float64),
            "key": np.arange(len(left)),
        }
    )


def sweep_weighted_fusion(
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    step: float = 0.01,
) -> tuple[float, float, np.ndarray, pd.DataFrame]:
    """Find the best left/right linear blend by AUC."""
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    rows = []
    best_auc = -np.inf
    best_weight = 0.0
    best_scores = right_scores
    for weight in weights:
        fused = (weight * left_scores) + ((1.0 - weight) * right_scores)
        auc = float(roc_auc_score(labels, fused))
        rows.append({"left_weight": float(weight), "right_weight": float(1.0 - weight), "auc_roc": auc})
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)
            best_scores = fused
    return best_weight, best_auc, best_scores, pd.DataFrame(rows)


def _basename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]
