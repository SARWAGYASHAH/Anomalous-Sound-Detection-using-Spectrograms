"""
anomaly_scorer.py - Reconstruction-error anomaly scoring.

The Keras autoencoder reconstructs normal spectrograms. Higher reconstruction
error indicates a more anomalous sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from src.models import Conv2DAutoencoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoreSummary:
    """Summary statistics for anomaly scores."""

    mean: float
    std: float
    minimum: float
    maximum: float
    percentile_95: float


def reconstruction_errors(
    original: np.ndarray | tf.Tensor,
    reconstructed: np.ndarray | tf.Tensor,
    method: str = "mean",
    top_k_fraction: float = 0.05,
    frequency_band: tuple[float, float] | list[float] | None = None,
    frequency_bands: list[tuple[float, float]] | list[list[float]] | None = None,
    band_aggregation: str = "max",
) -> np.ndarray:
    """
    Compute per-sample reconstruction anomaly scores.

    Args:
        original: Batch of original spectrograms.
        reconstructed: Batch of reconstructed spectrograms.
        method: mean, top_k, frequency_band, frequency_top_k,
            frequency_bands, or frequency_band_top_k.
        top_k_fraction: Fraction of highest-error pixels to average for top-k methods.
        frequency_band: Optional fractional mel-bin range, such as (0.1, 0.6).
        frequency_bands: Optional list of fractional mel-bin ranges.
        band_aggregation: How to combine per-band scores into one sample score:
            max or mean.

    Returns:
        1D numpy array of anomaly scores.
    """
    original_np = np.asarray(original)
    reconstructed_np = np.asarray(reconstructed)

    if original_np.shape != reconstructed_np.shape:
        raise ValueError(
            f"Shape mismatch: original={original_np.shape}, reconstructed={reconstructed_np.shape}"
        )

    error = np.square(original_np - reconstructed_np)
    method = method.lower()

    if method in {"frequency_bands", "frequency_band_top_k"}:
        band_scores = frequency_band_errors(
            original_np,
            reconstructed_np,
            frequency_bands=frequency_bands,
            top_k_fraction=top_k_fraction,
            use_top_k=method == "frequency_band_top_k",
        )
        return _aggregate_band_scores(band_scores, band_aggregation)

    if method in {"frequency_band", "frequency_top_k"}:
        error = _select_frequency_band(error, frequency_band)

    if method in {"mean", "frequency_band"}:
        axes = tuple(range(1, error.ndim))
        return np.mean(error, axis=axes)

    if method in {"top_k", "frequency_top_k"}:
        return _top_k_error(error, top_k_fraction=top_k_fraction)

    raise ValueError(
        f"Unknown reconstruction score method '{method}'. "
        "Expected mean, top_k, frequency_band, frequency_top_k, "
        "frequency_bands, or frequency_band_top_k."
    )


def frequency_band_errors(
    original: np.ndarray | tf.Tensor,
    reconstructed: np.ndarray | tf.Tensor,
    frequency_bands: list[tuple[float, float]] | list[list[float]] | None = None,
    top_k_fraction: float = 0.05,
    use_top_k: bool = False,
) -> np.ndarray:
    """
    Compute reconstruction error separately for each frequency band.

    Returns an array with shape (num_samples, num_bands).
    """
    original_np = np.asarray(original)
    reconstructed_np = np.asarray(reconstructed)

    if original_np.shape != reconstructed_np.shape:
        raise ValueError(
            f"Shape mismatch: original={original_np.shape}, reconstructed={reconstructed_np.shape}"
        )

    if frequency_bands is None:
        frequency_bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    error = np.square(original_np - reconstructed_np)
    band_scores = []

    for band in frequency_bands:
        band_error = _select_frequency_band(error, band)
        if use_top_k:
            band_scores.append(_top_k_error(band_error, top_k_fraction=top_k_fraction))
        else:
            axes = tuple(range(1, band_error.ndim))
            band_scores.append(np.mean(band_error, axis=axes))

    return np.stack(band_scores, axis=1).astype(np.float32)


def _aggregate_band_scores(band_scores: np.ndarray, band_aggregation: str) -> np.ndarray:
    """Aggregate per-band scores into one score per sample for thresholding."""
    aggregation = band_aggregation.lower()
    if aggregation == "max":
        return np.max(band_scores, axis=1)
    if aggregation == "mean":
        return np.mean(band_scores, axis=1)
    raise ValueError(f"Unknown band_aggregation '{band_aggregation}'. Expected max or mean.")


def _select_frequency_band(
    error: np.ndarray,
    frequency_band: tuple[float, float] | list[float] | None,
) -> np.ndarray:
    """Select a fractional mel-bin range from an error map."""
    if frequency_band is None:
        frequency_band = (0.0, 1.0)
    if len(frequency_band) != 2:
        raise ValueError("frequency_band must contain exactly two values.")

    low, high = float(frequency_band[0]), float(frequency_band[1])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"frequency_band must be within [0, 1] and increasing, got {frequency_band}")

    num_bins = error.shape[1]
    low_idx = int(low * num_bins)
    high_idx = max(low_idx + 1, int(high * num_bins))
    return error[:, low_idx:high_idx, ...]


def _top_k_error(error: np.ndarray, top_k_fraction: float = 0.05) -> np.ndarray:
    """Average the highest-error pixels per sample."""
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")

    flat_error = error.reshape((error.shape[0], -1))
    k = max(1, int(flat_error.shape[1] * float(top_k_fraction)))
    top_values = np.partition(flat_error, kth=flat_error.shape[1] - k, axis=1)[:, -k:]
    return np.mean(top_values, axis=1)


def summarize_scores(scores: np.ndarray) -> ScoreSummary:
    """Return common summary statistics for anomaly scores."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        raise ValueError("Cannot summarize empty score array.")

    return ScoreSummary(
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        minimum=float(np.min(scores)),
        maximum=float(np.max(scores)),
        percentile_95=float(np.percentile(scores, 95)),
    )


def compute_threshold(
    normal_scores: np.ndarray,
    method: str = "mean_std",
    percentile: float = 95,
    std_multiplier: float = 2.5,
    fixed_threshold: float | None = None,
) -> float:
    """
    Compute an anomaly threshold from normal reconstruction scores.

    Supported methods:
        percentile: percentile of normal scores
        mean_std: mean + std_multiplier * std
        fixed: fixed_threshold
    """
    normal_scores = np.asarray(normal_scores, dtype=np.float32)
    if normal_scores.size == 0:
        raise ValueError("Cannot compute threshold from empty normal scores.")

    method = method.lower()
    if method == "percentile":
        return float(np.percentile(normal_scores, percentile))
    if method == "mean_std":
        return float(np.mean(normal_scores) + std_multiplier * np.std(normal_scores))
    if method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold is required when method='fixed'.")
        return float(fixed_threshold)

    raise ValueError(f"Unknown threshold method: {method}")


class ReconstructionAnomalyScorer:
    """Score tf.data batches using a trained Keras autoencoder."""

    def __init__(
        self,
        model: tf.keras.Model,
        score_config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.score_config = score_config or {}

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        score_config: dict[str, Any] | None = None,
    ) -> "ReconstructionAnomalyScorer":
        """Load a Keras model and create a scorer."""
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"Conv2DAutoencoder": Conv2DAutoencoder},
            compile=False,
        )
        logger.info("Loaded model for scoring: %s", model_path)
        return cls(model, score_config=score_config)

    def score_batch(self, batch: tf.Tensor | np.ndarray) -> np.ndarray:
        """Score one tensor/array batch."""
        reconstructed = self.model.predict(batch, verbose=0)
        return reconstruction_errors(
            batch,
            reconstructed,
            method=self.score_config.get("method", "mean"),
            top_k_fraction=float(self.score_config.get("top_k_fraction", 0.05)),
            frequency_band=self.score_config.get("frequency_band"),
            frequency_bands=self.score_config.get("frequency_bands"),
            band_aggregation=self.score_config.get("band_aggregation", "max"),
        )

    def score_dataset(
        self,
        dataset: tf.data.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Score a labeled dataset.

        Args:
            dataset: Batches of (spectrogram, label).

        Returns:
            Tuple of (scores, labels).
        """
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch, labels in dataset:
            all_scores.append(self.score_batch(batch))
            all_labels.append(np.asarray(labels))

        scores = np.concatenate(all_scores).astype(np.float32)
        labels = np.concatenate(all_labels).astype(np.int32)
        logger.info("Scored %s samples", len(scores))
        return scores, labels

    def score_unlabeled_dataset(self, dataset: tf.data.Dataset) -> np.ndarray:
        """Score a dataset that yields x or (x, x) batches."""
        all_scores: list[np.ndarray] = []

        for batch in dataset:
            if isinstance(batch, tuple):
                batch = batch[0]
            all_scores.append(self.score_batch(batch))

        scores = np.concatenate(all_scores).astype(np.float32)
        logger.info("Scored %s unlabeled samples", len(scores))
        return scores


def threshold_from_config(normal_scores: np.ndarray, config: dict[str, Any]) -> float:
    """Compute threshold using config['inference'] settings."""
    inference_config = config.get("inference", {})
    return compute_threshold(
        normal_scores=normal_scores,
        method=inference_config.get("threshold_method", "mean_std"),
        percentile=float(inference_config.get("percentile", 95)),
        std_multiplier=float(inference_config.get("std_multiplier", 2.5)),
        fixed_threshold=inference_config.get("fixed_threshold"),
    )
