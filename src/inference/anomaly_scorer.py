"""
anomaly_scorer.py - Reconstruction-error anomaly scoring.

The primary score is the per-sample mean squared reconstruction error:

    mean((original - reconstructed) ** 2)

Higher scores mean the autoencoder reconstructed the input poorly, which is
the signal used for anomaly detection. The module also includes lightweight
threshold helpers and optional latent-distance scoring utilities inspired by
PCA/Mahalanobis workflows.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from src.models import Conv2DAutoencoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

ThresholdMethod = Literal["percentile", "mean_std", "fixed"]


@dataclass(frozen=True)
class ThresholdResult:
    """Threshold value plus the method metadata used to compute it."""

    value: float
    method: ThresholdMethod
    percentile: float | None = None
    mean: float | None = None
    std: float | None = None
    std_multiplier: float | None = None


@dataclass(frozen=True)
class ScoreSummary:
    """Basic descriptive statistics for a set of anomaly scores."""

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    percentile_95: float


def reconstruction_error(
    original: np.ndarray | tf.Tensor,
    reconstructed: np.ndarray | tf.Tensor,
) -> np.ndarray:
    """
    Compute mean squared reconstruction error for each sample.

    Args:
        original: Original batch. Shape usually (batch, height, width, channels).
        reconstructed: Reconstructed batch with the same shape.

    Returns:
        1D numpy array of anomaly scores, one score per sample.
    """
    original_arr = np.asarray(original, dtype=np.float32)
    reconstructed_arr = np.asarray(reconstructed, dtype=np.float32)

    if original_arr.shape != reconstructed_arr.shape:
        raise ValueError(
            "original and reconstructed must have the same shape. "
            f"Got {original_arr.shape} and {reconstructed_arr.shape}."
        )

    if original_arr.ndim == 0:
        raise ValueError("Expected at least one sample dimension.")

    if original_arr.ndim == 1:
        original_arr = original_arr[np.newaxis, :]
        reconstructed_arr = reconstructed_arr[np.newaxis, :]

    axes = tuple(range(1, original_arr.ndim))
    return np.mean((original_arr - reconstructed_arr) ** 2, axis=axes).astype(np.float32)


def score_batch(
    model: tf.keras.Model,
    batch: np.ndarray | tf.Tensor,
    verbose: int = 0,
) -> np.ndarray:
    """
    Reconstruct a batch with a Keras model and return reconstruction scores.

    Args:
        model: Trained Keras autoencoder.
        batch: Input batch with channels-last shape.
        verbose: Retained for API compatibility; direct model invocation is quiet.

    Returns:
        1D numpy array of reconstruction-error scores.
    """
    batch_arr = np.asarray(batch, dtype=np.float32)
    reconstructed = model(tf.convert_to_tensor(batch_arr), training=False)
    return reconstruction_error(batch_arr, reconstructed)


def score_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Score every batch in a tf.data dataset.

    The dataset may yield either ``x`` or ``(x, label)`` batches. If labels are
    present, they are returned alongside the scores.
    """
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
            if len(batch) > 1:
                possible_labels = np.asarray(batch[1])
                if possible_labels.ndim <= 1:
                    labels.append(possible_labels)
        else:
            inputs = batch

        scores.append(score_batch(model, inputs, verbose=verbose))

    if not scores:
        raise ValueError("Cannot score an empty dataset.")

    all_scores = np.concatenate(scores).astype(np.float32)
    all_labels = np.concatenate(labels).astype(np.int32) if labels else None
    return all_scores, all_labels


def compute_threshold(
    scores: np.ndarray | list[float],
    method: ThresholdMethod = "percentile",
    percentile: float = 95.0,
    std_multiplier: float = 2.0,
    fixed_threshold: float | None = None,
) -> ThresholdResult:
    """
    Compute an anomaly threshold from normal/reference scores.

    Supported methods:
        percentile: ``np.percentile(scores, percentile)``
        mean_std: ``mean(scores) + std_multiplier * std(scores)``
        fixed: use ``fixed_threshold`` directly
    """
    scores_arr = _validate_scores(scores)
    method = method.lower()

    if method == "percentile":
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}.")
        value = float(np.percentile(scores_arr, percentile))
        return ThresholdResult(value=value, method="percentile", percentile=float(percentile))

    if method == "mean_std":
        mean = float(np.mean(scores_arr))
        std = float(np.std(scores_arr))
        value = mean + float(std_multiplier) * std
        return ThresholdResult(
            value=float(value),
            method="mean_std",
            mean=mean,
            std=std,
            std_multiplier=float(std_multiplier),
        )

    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold is required when method='fixed'.")
        return ThresholdResult(value=float(fixed_threshold), method="fixed")

    raise ValueError("Unknown threshold method. Use 'percentile', 'mean_std', or 'fixed'.")


def threshold_from_config(
    normal_scores: np.ndarray | list[float],
    config: dict,
) -> ThresholdResult:
    """Calculate the normal-data threshold specified by project configuration."""
    inference_config = config.get("inference", {})
    return compute_threshold(
        normal_scores,
        method=inference_config.get("threshold_method", "percentile"),
        percentile=float(inference_config.get("percentile", 95.0)),
        std_multiplier=float(inference_config.get("std_multiplier", 2.0)),
        fixed_threshold=inference_config.get("fixed_threshold"),
    )


def summarize_scores(scores: np.ndarray | list[float]) -> ScoreSummary:
    """Return descriptive statistics suitable for metrics JSON output."""
    scores_arr = _validate_scores(scores)
    return ScoreSummary(
        count=int(scores_arr.size),
        mean=float(np.mean(scores_arr)),
        std=float(np.std(scores_arr)),
        minimum=float(np.min(scores_arr)),
        maximum=float(np.max(scores_arr)),
        percentile_95=float(np.percentile(scores_arr, 95.0)),
    )


def classify_scores(
    scores: np.ndarray | list[float],
    threshold: float,
) -> np.ndarray:
    """
    Convert anomaly scores to binary labels.

    Returns:
        0 for normal, 1 for anomaly.
    """
    scores_arr = _validate_scores(scores)
    return (scores_arr >= float(threshold)).astype(np.int32)


def score_to_probability(
    scores: np.ndarray | list[float],
    threshold: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize scores relative to a threshold for downstream severity labels.

    A value below 1.0 is below threshold, and a value at/above 1.0 crosses the
    anomaly boundary. This is not a calibrated probability; it is a convenient
    normalized score for UI/prediction code.
    """
    scores_arr = _validate_scores(scores)
    return (scores_arr / (float(threshold) + eps)).astype(np.float32)


def latent_features(
    model: tf.keras.Model,
    batch: np.ndarray | tf.Tensor,
    verbose: int = 0,
) -> np.ndarray:
    """
    Extract flattened encoder features when the model exposes ``encode``.

    This supports the latent-distance style used in many autoencoder anomaly
    workflows. It is optional and not required for basic reconstruction scoring.
    """
    if not hasattr(model, "encode"):
        raise AttributeError("Model does not expose an encode() method for latent scoring.")

    batch_arr = np.asarray(batch, dtype=np.float32)
    encoded = model.encode(tf.convert_to_tensor(batch_arr), training=False)
    features = np.asarray(encoded)
    return features.reshape(features.shape[0], -1).astype(np.float32)


def mahalanobis_scores(
    features: np.ndarray,
    mean_vector: np.ndarray,
    inverse_covariance: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each feature vector.

    Args:
        features: Feature matrix with shape (n_samples, n_features).
        mean_vector: Reference normal-feature mean.
        inverse_covariance: Inverse covariance matrix for normal features.

    Returns:
        1D anomaly score array. Higher means farther from normal latent space.
    """
    features_arr = np.asarray(features, dtype=np.float64)
    mean_arr = np.asarray(mean_vector, dtype=np.float64).reshape(1, -1)
    inv_cov_arr = np.asarray(inverse_covariance, dtype=np.float64)

    if features_arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features_arr.shape}.")
    if mean_arr.shape[1] != features_arr.shape[1]:
        raise ValueError(
            "mean_vector length must match feature dimension. "
            f"Got {mean_arr.shape[1]} and {features_arr.shape[1]}."
        )
    if inv_cov_arr.shape != (features_arr.shape[1], features_arr.shape[1]):
        raise ValueError(
            "inverse_covariance must be square with feature dimension. "
            f"Got {inv_cov_arr.shape}, expected {(features_arr.shape[1], features_arr.shape[1])}."
        )

    diff = features_arr - mean_arr
    distances_sq = np.einsum("ij,jk,ik->i", diff, inv_cov_arr, diff)
    return np.sqrt(np.maximum(distances_sq, 0.0)).astype(np.float32)


def fit_mahalanobis_reference(features: np.ndarray, regularization: float = 1e-6) -> dict[str, np.ndarray]:
    """
    Fit mean and inverse covariance from normal/reference latent features.

    Returns a dict that can be saved with ``save_mahalanobis_reference``.
    """
    features_arr = np.asarray(features, dtype=np.float64)
    if features_arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features_arr.shape}.")
    if features_arr.shape[0] < 2:
        raise ValueError("At least two feature rows are required to fit covariance.")

    mean_vector = np.mean(features_arr, axis=0)
    covariance = np.cov(features_arr, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * float(regularization)
    inverse_covariance = np.linalg.pinv(covariance)

    return {
        "mean_vector": mean_vector.astype(np.float32),
        "inverse_covariance": inverse_covariance.astype(np.float32),
    }


def save_mahalanobis_reference(reference: dict[str, np.ndarray], filepath: str | Path) -> Path:
    """Save fitted Mahalanobis reference arrays to a compressed .npz file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)
    logger.info("Saved Mahalanobis reference to %s", path)
    return path


def load_mahalanobis_reference(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load Mahalanobis reference arrays saved by ``save_mahalanobis_reference``."""
    with np.load(filepath) as data:
        return {
            "mean_vector": data["mean_vector"],
            "inverse_covariance": data["inverse_covariance"],
        }


class ReconstructionAnomalyScorer:
    """Small stateful wrapper for scoring and thresholding reconstruction errors."""

    def __init__(
        self,
        threshold: float | None = None,
        threshold_method: ThresholdMethod = "percentile",
        percentile: float = 95.0,
        std_multiplier: float = 2.0,
        fixed_threshold: float | None = None,
        model: tf.keras.Model | None = None,
    ):
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.percentile = percentile
        self.std_multiplier = std_multiplier
        self.fixed_threshold = fixed_threshold
        self.threshold_result: ThresholdResult | None = None
        self.model = model

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        **kwargs,
    ) -> "ReconstructionAnomalyScorer":
        """Load a saved project autoencoder and create a reconstruction scorer."""
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"Conv2DAutoencoder": Conv2DAutoencoder},
            compile=False,
        )
        logger.info("Loaded Keras model for scoring: %s", model_path)
        return cls(model=model, **kwargs)

    def fit_threshold(self, normal_scores: np.ndarray | list[float]) -> ThresholdResult:
        """Fit and store a threshold from normal/reference reconstruction scores."""
        result = compute_threshold(
            normal_scores,
            method=self.threshold_method,
            percentile=self.percentile,
            std_multiplier=self.std_multiplier,
            fixed_threshold=self.fixed_threshold,
        )
        self.threshold = result.value
        self.threshold_result = result
        return result

    def score_batch(
        self,
        batch: np.ndarray | tf.Tensor,
        model: tf.keras.Model | None = None,
        verbose: int = 0,
    ) -> np.ndarray:
        """Return reconstruction scores for one batch."""
        model = model or self.model
        if model is None:
            raise ValueError("A Keras model is required for batch scoring.")
        return score_batch(model, batch, verbose=verbose)

    def score_dataset(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model | None = None,
        verbose: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return reconstruction scores, and labels when the dataset includes them."""
        model = model or self.model
        if model is None:
            raise ValueError("A Keras model is required for dataset scoring.")
        return score_dataset(model, dataset, verbose=verbose)

    def predict(self, scores: np.ndarray | list[float]) -> np.ndarray:
        """Convert scores to binary anomaly predictions using the stored threshold."""
        if self.threshold is None:
            raise ValueError("No threshold is set. Call fit_threshold() or pass threshold in the constructor.")
        return classify_scores(scores, self.threshold)


def _validate_scores(scores: np.ndarray | list[float]) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores_arr.size == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores must be finite numbers.")
    return scores_arr
