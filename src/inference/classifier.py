"""
classifier.py - Normal / follow-up / alert severity classification.

The anomaly scorer produces continuous scores where larger values are more
anomalous. This module converts those scores into operational labels using
ordered boundaries, for example:

    score < 0.5          -> normal
    0.5 <= score < 0.8   -> follow_up
    score >= 0.8         -> alert

The default labels and boundaries mirror ``config/default.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LABELS: tuple[str, str, str] = ("normal", "follow_up", "alert")
DEFAULT_BOUNDARIES: tuple[float, float] = (0.5, 0.8)


@dataclass(frozen=True)
class ClassificationResult:
    """Structured classification output for one anomaly score."""

    score: float
    label: str
    level: int

    @property
    def is_anomaly(self) -> bool:
        """Return True for every non-normal severity label."""
        return self.level > 0

    @property
    def requires_attention(self) -> bool:
        """Return True for follow-up and alert states."""
        return self.is_anomaly


def classify_score(
    score: float,
    boundaries: Sequence[float] = DEFAULT_BOUNDARIES,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> str:
    """
    Convert one anomaly score into a severity label.

    Args:
        score: Continuous anomaly score. Higher means more anomalous.
        boundaries: Ordered numeric cut points between labels.
        labels: Label names. Must contain exactly ``len(boundaries) + 1`` items.

    Returns:
        Severity label for the score.
    """
    classifier = SeverityClassifier(boundaries=boundaries, labels=labels)
    return classifier.predict_one(score)


def classify_scores(
    scores: np.ndarray | Iterable[float],
    boundaries: Sequence[float] = DEFAULT_BOUNDARIES,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> np.ndarray:
    """
    Convert anomaly scores into severity labels.

    Args:
        scores: One or more continuous anomaly scores.
        boundaries: Ordered numeric cut points between labels.
        labels: Label names. Must contain exactly ``len(boundaries) + 1`` items.

    Returns:
        1D array of string labels.
    """
    classifier = SeverityClassifier(boundaries=boundaries, labels=labels)
    return classifier.predict(scores)


def classify_scores_as_levels(
    scores: np.ndarray | Iterable[float],
    boundaries: Sequence[float] = DEFAULT_BOUNDARIES,
) -> np.ndarray:
    """
    Convert anomaly scores into integer severity levels.

    Returns:
        0 for the first label, 1 for the second label, and so on.
    """
    scores_arr = _validate_scores(scores)
    boundaries_tuple = _validate_boundaries(boundaries)
    return np.digitize(scores_arr, boundaries_tuple, right=False).astype(np.int32)


class SeverityClassifier:
    """Map continuous anomaly scores to ordered severity labels."""

    def __init__(
        self,
        boundaries: Sequence[float] = DEFAULT_BOUNDARIES,
        labels: Sequence[str] = DEFAULT_LABELS,
    ):
        self.boundaries = _validate_boundaries(boundaries)
        self.labels = _validate_labels(labels, expected_count=len(self.boundaries) + 1)
        logger.debug(
            "Initialized SeverityClassifier with labels=%s boundaries=%s",
            self.labels,
            self.boundaries,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SeverityClassifier":
        """
        Build a classifier from a full project config or classification section.

        Accepted shapes:

            {"inference": {"classification": {...}}}
            {"classification": {...}}
            {"labels": [...], "boundaries": [...]}
        """
        classification_config = config.get("inference", {}).get("classification")
        if classification_config is None:
            classification_config = config.get("classification", config)

        return cls(
            boundaries=classification_config.get("boundaries", DEFAULT_BOUNDARIES),
            labels=classification_config.get("labels", DEFAULT_LABELS),
        )

    def predict_one(self, score: float) -> str:
        """Return the severity label for one score."""
        level = self.predict_level(score)
        return self.labels[level]

    def predict_level(self, score: float) -> int:
        """Return the integer severity level for one score."""
        scores = _validate_scores([score])
        return int(np.digitize(scores, self.boundaries, right=False)[0])

    def predict(self, scores: np.ndarray | Iterable[float]) -> np.ndarray:
        """Return severity labels for a batch of scores."""
        levels = self.predict_levels(scores)
        labels_arr = np.asarray(self.labels, dtype=object)
        return labels_arr[levels].astype(str)

    def predict_levels(self, scores: np.ndarray | Iterable[float]) -> np.ndarray:
        """Return integer severity levels for a batch of scores."""
        scores_arr = _validate_scores(scores)
        return np.digitize(scores_arr, self.boundaries, right=False).astype(np.int32)

    def predict_results(self, scores: np.ndarray | Iterable[float]) -> list[ClassificationResult]:
        """Return structured results for a batch of scores."""
        scores_arr = _validate_scores(scores)
        levels = self.predict_levels(scores_arr)
        return [
            ClassificationResult(
                score=float(score),
                label=self.labels[int(level)],
                level=int(level),
            )
            for score, level in zip(scores_arr, levels)
        ]

    def to_dict(self) -> dict[str, list[float] | list[str]]:
        """Return a serializable representation of this classifier."""
        return {
            "labels": list(self.labels),
            "boundaries": [float(boundary) for boundary in self.boundaries],
        }


def _validate_scores(scores: np.ndarray | Iterable[float]) -> np.ndarray:
    try:
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        scores_arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)

    if scores_arr.size == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores must be finite numbers.")
    return scores_arr


def _validate_boundaries(boundaries: Sequence[float]) -> tuple[float, ...]:
    boundaries_arr = np.asarray(boundaries, dtype=np.float32).reshape(-1)
    if boundaries_arr.size == 0:
        raise ValueError("boundaries must contain at least one value.")
    if not np.all(np.isfinite(boundaries_arr)):
        raise ValueError("boundaries must be finite numbers.")
    if np.any(np.diff(boundaries_arr) <= 0):
        raise ValueError("boundaries must be strictly increasing.")
    return tuple(float(boundary) for boundary in boundaries_arr)


def _validate_labels(labels: Sequence[str], expected_count: int) -> tuple[str, ...]:
    labels_tuple = tuple(str(label) for label in labels)
    if len(labels_tuple) != expected_count:
        raise ValueError(
            "labels must contain exactly len(boundaries) + 1 values. "
            f"Expected {expected_count}, got {len(labels_tuple)}."
        )
    if any(label.strip() == "" for label in labels_tuple):
        raise ValueError("labels must not contain empty values.")
    return labels_tuple


__all__ = [
    "ClassificationResult",
    "DEFAULT_BOUNDARIES",
    "DEFAULT_LABELS",
    "SeverityClassifier",
    "classify_score",
    "classify_scores",
    "classify_scores_as_levels",
]
