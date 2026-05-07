"""
classifier.py - Convert anomaly scores into severity labels.
"""

from __future__ import annotations

import numpy as np


class SeverityClassifier:
    """
    Threshold-based anomaly severity classifier.

    Boundaries are interpreted as multipliers around the anomaly threshold:
        score < threshold * boundaries[0] -> labels[0]
        below threshold * boundaries[1]   -> labels[1]
        otherwise                         -> labels[2]

    With default boundaries [0.5, 0.8], this creates a conservative
    three-level severity output for reporting.
    """

    def __init__(
        self,
        threshold: float,
        labels: list[str] | tuple[str, str, str] = ("normal", "follow_up", "alert"),
        boundaries: list[float] | tuple[float, float] = (0.5, 0.8),
    ):
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if len(labels) != 3:
            raise ValueError("labels must contain exactly three labels.")
        if len(boundaries) != 2:
            raise ValueError("boundaries must contain exactly two values.")
        if boundaries[0] >= boundaries[1]:
            raise ValueError("boundaries must be increasing.")

        self.threshold = float(threshold)
        self.labels = tuple(labels)
        self.boundaries = tuple(float(value) for value in boundaries)

    @classmethod
    def from_config(cls, threshold: float, config: dict) -> "SeverityClassifier":
        """Create classifier from config['inference']['classification'] settings."""
        classification_config = config.get("inference", {}).get("classification", {})
        return cls(
            threshold=threshold,
            labels=classification_config.get("labels", ["normal", "follow_up", "alert"]),
            boundaries=classification_config.get("boundaries", [0.5, 0.8]),
        )

    def classify_score(self, score: float) -> str:
        """Classify one score."""
        if score < self.threshold * self.boundaries[0]:
            return self.labels[0]
        if score < self.threshold * self.boundaries[1]:
            return self.labels[1]
        return self.labels[2]

    def classify_scores(self, scores: np.ndarray) -> np.ndarray:
        """Classify a vector of scores."""
        return np.asarray([self.classify_score(float(score)) for score in scores])

    def to_binary_predictions(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores into binary anomaly predictions using the threshold."""
        return (np.asarray(scores) >= self.threshold).astype(np.int32)
