"""
latent_scorer.py - Latent-space Mahalanobis anomaly scoring.

This scorer uses the trained autoencoder encoder as a feature extractor, fits
PCA on normal training features, then scores new samples by Mahalanobis
distance from the normal feature distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from src.models import Conv2DAutoencoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatentScorerMetadata:
    """Small metadata bundle saved beside scorer parameters."""

    feature_dim: int
    pca_components: int
    covariance_regularization: float
    num_fit_samples: int


class LatentMahalanobisScorer:
    """
    Score samples by distance from normal data in autoencoder latent space.

    The object stores all fitted preprocessing/statistical parameters as NumPy
    arrays so it can be saved and loaded without pickling sklearn objects.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        mean_vec: np.ndarray,
        inv_cov: np.ndarray,
        metadata: LatentScorerMetadata,
    ):
        self.model = model
        self.scaler_mean = scaler_mean.astype(np.float32)
        self.scaler_scale = scaler_scale.astype(np.float32)
        self.pca_mean = pca_mean.astype(np.float32)
        self.pca_components = pca_components.astype(np.float32)
        self.mean_vec = mean_vec.astype(np.float32)
        self.inv_cov = inv_cov.astype(np.float32)
        self.metadata = metadata

    @classmethod
    def fit(
        cls,
        model: tf.keras.Model,
        normal_dataset: tf.data.Dataset,
        pca_components: int = 32,
        covariance_regularization: float = 1e-6,
    ) -> "LatentMahalanobisScorer":
        """Fit PCA and Mahalanobis parameters from normal training samples."""
        features = extract_latent_features(model, normal_dataset)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(f"Expected at least 2 feature rows, got shape {features.shape}")

        max_components = min(features.shape[0] - 1, features.shape[1])
        n_components = min(int(pca_components), max_components)
        if n_components < 1:
            raise ValueError(f"Invalid PCA component count: {n_components}")

        scaler_mean = features.mean(axis=0)
        scaler_scale = features.std(axis=0)
        scaler_scale = np.where(scaler_scale < 1e-8, 1.0, scaler_scale)
        scaled_features = (features - scaler_mean) / scaler_scale

        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pca_features = pca.fit_transform(scaled_features)

        mean_vec = pca_features.mean(axis=0)
        covariance = np.cov(pca_features, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0], dtype=np.float32) * float(covariance_regularization)
        inv_cov = np.linalg.pinv(covariance)

        metadata = LatentScorerMetadata(
            feature_dim=int(features.shape[1]),
            pca_components=int(n_components),
            covariance_regularization=float(covariance_regularization),
            num_fit_samples=int(features.shape[0]),
        )
        logger.info(
            "Fitted latent Mahalanobis scorer: samples=%s, feature_dim=%s, pca_components=%s",
            metadata.num_fit_samples,
            metadata.feature_dim,
            metadata.pca_components,
        )

        return cls(
            model=model,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            mean_vec=mean_vec,
            inv_cov=inv_cov,
            metadata=metadata,
        )

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        params_path: str | Path,
    ) -> "LatentMahalanobisScorer":
        """Load a Keras model and fitted latent scorer parameters."""
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"Conv2DAutoencoder": Conv2DAutoencoder},
            compile=False,
        )
        return cls.load(params_path=params_path, model=model)

    @classmethod
    def load(
        cls,
        params_path: str | Path,
        model: tf.keras.Model,
    ) -> "LatentMahalanobisScorer":
        """Load fitted scorer parameters from a .npz file."""
        params_path = Path(params_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Latent scorer params not found: {params_path}")

        data = np.load(params_path, allow_pickle=False)
        metadata = LatentScorerMetadata(
            feature_dim=int(data["feature_dim"]),
            pca_components=int(data["pca_components"]),
            covariance_regularization=float(data["covariance_regularization"]),
            num_fit_samples=int(data["num_fit_samples"]),
        )
        logger.info("Loaded latent scorer params: %s", params_path)
        return cls(
            model=model,
            scaler_mean=data["scaler_mean"],
            scaler_scale=data["scaler_scale"],
            pca_mean=data["pca_mean"],
            pca_components=data["pca_components_array"],
            mean_vec=data["mean_vec"],
            inv_cov=data["inv_cov"],
            metadata=metadata,
        )

    def save(self, params_path: str | Path) -> Path:
        """Save fitted scorer parameters as a portable .npz file."""
        params_path = Path(params_path)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            params_path,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            pca_mean=self.pca_mean,
            pca_components_array=self.pca_components,
            mean_vec=self.mean_vec,
            inv_cov=self.inv_cov,
            feature_dim=np.asarray(self.metadata.feature_dim, dtype=np.int32),
            pca_components=np.asarray(self.metadata.pca_components, dtype=np.int32),
            covariance_regularization=np.asarray(
                self.metadata.covariance_regularization,
                dtype=np.float32,
            ),
            num_fit_samples=np.asarray(self.metadata.num_fit_samples, dtype=np.int32),
        )
        logger.info("Saved latent scorer params: %s", params_path)
        return params_path

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Apply saved standardization and PCA transform."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {features.shape}")
        if features.shape[1] != self.metadata.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.metadata.feature_dim}, "
                f"got {features.shape[1]}"
            )

        scaled = (features - self.scaler_mean) / self.scaler_scale
        return (scaled - self.pca_mean) @ self.pca_components.T

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Score extracted latent features."""
        pca_features = self.transform_features(features)
        diff = pca_features - self.mean_vec
        distances = np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff)
        return np.sqrt(np.maximum(distances, 0.0)).astype(np.float32)

    def score_batch(self, batch: tf.Tensor | np.ndarray) -> np.ndarray:
        """Score one batch of spectrograms."""
        features = extract_latent_features_from_batch(self.model, batch)
        return self.score_features(features)

    def score_dataset(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Score a labeled dataset that yields (spectrogram, label)."""
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch, labels in dataset:
            all_scores.append(self.score_batch(batch))
            all_labels.append(np.asarray(labels))

        scores = np.concatenate(all_scores).astype(np.float32)
        labels = np.concatenate(all_labels).astype(np.int32)
        logger.info("Latent scorer scored %s samples", len(scores))
        return scores, labels

    def score_unlabeled_dataset(self, dataset: tf.data.Dataset) -> np.ndarray:
        """Score an unlabeled dataset that yields x or (x, x)."""
        all_scores: list[np.ndarray] = []

        for batch in dataset:
            if isinstance(batch, tuple):
                batch = batch[0]
            all_scores.append(self.score_batch(batch))

        scores = np.concatenate(all_scores).astype(np.float32)
        logger.info("Latent scorer scored %s unlabeled samples", len(scores))
        return scores


def extract_latent_features(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Extract flattened encoder features from a dataset."""
    feature_batches: list[np.ndarray] = []

    for batch in dataset:
        if isinstance(batch, tuple):
            batch = batch[0]
        feature_batches.append(extract_latent_features_from_batch(model, batch))

    features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    logger.info("Extracted latent features with shape %s", features.shape)
    return features


def extract_latent_features_from_batch(
    model: tf.keras.Model,
    batch: tf.Tensor | np.ndarray,
) -> np.ndarray:
    """Extract flattened encoder features from one batch."""
    if hasattr(model, "encode"):
        encoded = model.encode(batch, training=False)
    elif hasattr(model, "encoder"):
        encoded = model.encoder(batch, training=False)
    else:
        raise AttributeError("Model must expose an encode() method or encoder attribute.")

    encoded_np = np.asarray(encoded)
    return encoded_np.reshape((encoded_np.shape[0], -1)).astype(np.float32)
