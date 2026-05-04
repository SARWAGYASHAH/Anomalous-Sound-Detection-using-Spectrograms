"""
losses.py - Keras loss helpers for spectrogram reconstruction.

Autoencoder training uses x as both input and target, so the default loss is
mean squared reconstruction error.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf


def reconstruction_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean squared reconstruction error over each sample."""
    error = tf.square(y_true - y_pred)
    axes = tf.range(1, tf.rank(error))
    return tf.reduce_mean(error, axis=axes)


def reconstruction_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean absolute reconstruction error over each sample."""
    error = tf.abs(y_true - y_pred)
    axes = tf.range(1, tf.rank(error))
    return tf.reduce_mean(error, axis=axes)


def ssim_reconstruction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    SSIM-inspired reconstruction loss.

    Inputs are expected to be normalized to [0, 1]. The loss is 1 - SSIM,
    averaged over the batch.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - ssim


def get_loss(name: str) -> str | Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Resolve a configured loss name to a Keras-compatible loss.

    Supported names:
        mse
        mae
        reconstruction_mse
        reconstruction_mae
        ssim
        custom
    """
    normalized = name.lower()

    if normalized == "mse":
        return "mse"
    if normalized == "mae":
        return "mae"
    if normalized in {"reconstruction_mse", "custom"}:
        return reconstruction_mse
    if normalized == "reconstruction_mae":
        return reconstruction_mae
    if normalized == "ssim":
        return ssim_reconstruction_loss

    raise ValueError(
        f"Unknown loss '{name}'. Expected one of: "
        "mse, mae, reconstruction_mse, reconstruction_mae, ssim, custom."
    )
