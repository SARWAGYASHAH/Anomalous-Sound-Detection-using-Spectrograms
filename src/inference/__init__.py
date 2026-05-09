"""
src.inference - Anomaly scoring, classification, and prediction.
"""

from src.inference.anomaly_scorer import (
    ReconstructionAnomalyScorer,
    ScoreSummary,
    compute_threshold,
    reconstruction_errors,
    summarize_scores,
    threshold_from_config,
)
from src.inference.classifier import SeverityClassifier
from src.inference.latent_scorer import (
    LatentMahalanobisScorer,
    LatentScorerMetadata,
    extract_latent_features,
    extract_latent_features_from_batch,
)

__all__ = [
    "LatentMahalanobisScorer",
    "LatentScorerMetadata",
    "ReconstructionAnomalyScorer",
    "ScoreSummary",
    "SeverityClassifier",
    "compute_threshold",
    "extract_latent_features",
    "extract_latent_features_from_batch",
    "reconstruction_errors",
    "summarize_scores",
    "threshold_from_config",
]
