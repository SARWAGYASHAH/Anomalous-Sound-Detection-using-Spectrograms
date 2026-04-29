"""
metrics.py - Anomaly detection evaluation metrics.

Provides AUC-ROC, partial AUC (pAUC), precision, recall, F1,
and a comprehensive evaluation summary.

Usage:
    from src.utils.metrics import compute_auc, compute_pauc, evaluate_all

    auc = compute_auc(y_true, anomaly_scores)
    pauc = compute_pauc(y_true, anomaly_scores, max_fpr=0.1)
    results = evaluate_all(y_true, anomaly_scores, threshold=0.5)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    Args:
        y_true: Ground truth labels (0 = normal, 1 = anomalous).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        AUC-ROC score (0.0 to 1.0).
    """
    try:
        auc = roc_auc_score(y_true, scores)
        logger.info(f"AUC-ROC: {auc:.4f}")
        return float(auc)
    except ValueError as e:
        logger.warning(f"Could not compute AUC-ROC: {e}")
        return 0.0


def compute_pauc(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.1,
) -> float:
    """
    Compute partial AUC (pAUC) up to a maximum false positive rate.

    pAUC is more relevant for anomaly detection because it focuses
    on the low-FPR region where the detector must perform well.

    Args:
        y_true: Ground truth labels (0 = normal, 1 = anomalous).
        scores: Anomaly scores (higher = more anomalous).
        max_fpr: Maximum false positive rate to consider (default: 0.1 = 10%).

    Returns:
        Partial AUC score (normalized to 0.0–1.0 range).
    """
    try:
        pauc = roc_auc_score(y_true, scores, max_fpr=max_fpr)
        logger.info(f"pAUC (max_fpr={max_fpr}): {pauc:.4f}")
        return float(pauc)
    except ValueError as e:
        logger.warning(f"Could not compute pAUC: {e}")
        return 0.0


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1-score for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).

    Returns:
        Dictionary with "precision", "recall", "f1" keys.
    """
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """
    Compute confusion matrix components.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with TP, TN, FP, FN counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    method: str = "f1",
) -> float:
    """
    Find the optimal anomaly threshold.

    Args:
        y_true: Ground truth labels.
        scores: Anomaly scores.
        method: Optimization criterion — "f1" (maximize F1) or "youden"
                (maximize Youden's J = sensitivity + specificity - 1).

    Returns:
        Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    if method == "youden":
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = float(thresholds[best_idx])
    elif method == "f1":
        # Compute F1 for each threshold
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(pr_thresholds[min(best_idx, len(pr_thresholds) - 1)])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'f1' or 'youden'.")

    logger.info(f"Optimal threshold ({method}): {optimal_threshold:.4f}")
    return optimal_threshold


def evaluate_all(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    max_fpr: float = 0.1,
) -> dict:
    """
    Run a comprehensive evaluation and return all metrics.

    If threshold is None, the optimal F1 threshold is computed automatically.

    Args:
        y_true: Ground truth labels (0 = normal, 1 = anomalous).
        scores: Anomaly scores (higher = more anomalous).
        threshold: Decision threshold. If None, auto-computed.
        max_fpr: Max FPR for pAUC calculation.

    Returns:
        Dictionary containing all metrics:
        {
            "auc_roc", "pauc", "threshold",
            "precision", "recall", "f1",
            "true_positives", "true_negatives",
            "false_positives", "false_negatives",
            "average_precision"
        }
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # Core metrics
    auc = compute_auc(y_true, scores)
    pauc = compute_pauc(y_true, scores, max_fpr=max_fpr)

    # Threshold
    if threshold is None:
        threshold = find_optimal_threshold(y_true, scores, method="f1")

    # Binary predictions
    y_pred = (scores >= threshold).astype(int)

    # Classification metrics
    prf = compute_precision_recall_f1(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    # Average precision (area under PR curve)
    try:
        ap = float(average_precision_score(y_true, scores))
    except ValueError:
        ap = 0.0

    results = {
        "auc_roc": auc,
        "pauc": pauc,
        "threshold": float(threshold),
        "average_precision": ap,
        **prf,
        **cm,
    }

    logger.info(f"Evaluation complete: AUC={auc:.4f}, F1={prf['f1']:.4f}, "
                f"Precision={prf['precision']:.4f}, Recall={prf['recall']:.4f}")

    return results
