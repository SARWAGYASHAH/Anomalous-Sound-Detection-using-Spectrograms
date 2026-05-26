"""Inference utilities for scoring, classification, and audio prediction."""

from src.inference.anomaly_scorer import (
    ReconstructionAnomalyScorer,
    ScoreSummary,
    ThresholdResult,
    classify_scores as classify_anomalies,
    compute_threshold,
    reconstruction_error,
    score_batch,
    score_dataset,
    score_to_probability,
    summarize_scores,
    threshold_from_config,
)
from src.inference.classifier import ClassificationResult, SeverityClassifier

__all__ = [
    "ClassificationResult",
    "ReconstructionAnomalyScorer",
    "ScoreSummary",
    "SeverityClassifier",
    "ThresholdResult",
    "classify_anomalies",
    "compute_threshold",
    "reconstruction_error",
    "score_batch",
    "score_dataset",
    "score_to_probability",
    "summarize_scores",
    "threshold_from_config",
]
