"""
dataset.py - TensorFlow data pipeline for processed spectrograms.

Wraps saved .npy spectrogram files into tf.data.Dataset objects for
autoencoder training and labeled anomaly evaluation.

Usage:
    from src.data.dataset import create_autoencoder_dataset, create_labeled_dataset

    train_ds = create_autoencoder_dataset("Data/processed/train", batch_size=32)
    test_ds = create_labeled_dataset("Data/processed/source_test", batch_size=32)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_MAP = {"normal": 0, "anomaly": 1}


def discover_spectrogram_files(
    data_dir: str | Path,
    labels: Iterable[str] | None = None,
) -> tuple[list[str], list[int]]:
    """
    Discover .npy spectrogram files organized by label directories.

    Expected layout:
        data_dir/
        |-- normal/
        |   |-- file_001.npy
        |-- anomaly/
            |-- file_002.npy

    Args:
        data_dir: Processed split directory, such as Data/processed/source_test.
        labels: Optional labels to include. Defaults to normal and anomaly.

    Returns:
        Tuple of file paths and integer labels.
    """
    data_path = Path(data_dir)
    selected_labels = list(labels) if labels is not None else list(LABEL_MAP)

    filepaths: list[str] = []
    targets: list[int] = []

    for label_name in selected_labels:
        if label_name not in LABEL_MAP:
            raise ValueError(f"Unknown label '{label_name}'. Expected one of {sorted(LABEL_MAP)}")

        label_dir = data_path / label_name
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        files = sorted(label_dir.glob("*.npy"))
        filepaths.extend(str(path) for path in files)
        targets.extend([LABEL_MAP[label_name]] * len(files))

    logger.info(
        f"Discovered {len(filepaths)} spectrograms in {data_path} "
        f"(normal={targets.count(0)}, anomaly={targets.count(1)})"
    )
    return filepaths, targets


def load_spectrogram_array(filepath: str | Path, add_channel: bool = True) -> np.ndarray:
    """
    Load a saved spectrogram as float32.

    Args:
        filepath: Path to a .npy spectrogram file.
        add_channel: When True, return channels-last shape (height, width, 1).

    Returns:
        Spectrogram array suitable for Keras models.
    """
    spectrogram = np.load(str(filepath)).astype(np.float32)

    if spectrogram.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, got shape {spectrogram.shape}: {filepath}")

    if add_channel:
        spectrogram = np.expand_dims(spectrogram, axis=-1)

    return spectrogram


def _load_spectrogram_tf(filepath: tf.Tensor, input_shape: tuple[int, int] | None) -> tf.Tensor:
    """TensorFlow wrapper around NumPy .npy loading."""

    def _load(path: bytes) -> np.ndarray:
        return load_spectrogram_array(path.decode("utf-8"), add_channel=True)

    spectrogram = tf.numpy_function(_load, [filepath], Tout=tf.float32)

    if input_shape is not None:
        spectrogram.set_shape((*input_shape, 1))
    else:
        spectrogram.set_shape((None, None, 1))

    return spectrogram


def create_autoencoder_dataset(
    data_dir: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    input_shape: tuple[int, int] | None = None,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Create a dataset for autoencoder training.

    Only normal samples are loaded. Each item is returned as (x, x), because
    the autoencoder learns to reconstruct its own input.
    """
    filepaths, _ = discover_spectrogram_files(data_dir, labels=["normal"])

    if not filepaths:
        raise FileNotFoundError(f"No normal .npy spectrograms found in {Path(data_dir) / 'normal'}")

    dataset = tf.data.Dataset.from_tensor_slices(filepaths)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path: _to_autoencoder_pair(path, input_shape),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        dataset = dataset.cache()

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_labeled_dataset(
    data_dir: str | Path,
    batch_size: int = 32,
    shuffle: bool = False,
    input_shape: tuple[int, int] | None = None,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Create a labeled dataset for evaluation.

    Returns batches of (spectrogram, label), where labels are:
        0 = normal
        1 = anomaly
    """
    filepaths, labels = discover_spectrogram_files(data_dir)

    if not filepaths:
        raise FileNotFoundError(f"No .npy spectrograms found in {data_dir}")

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, np.asarray(labels, dtype=np.int32)))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: (_load_spectrogram_tf(path, input_shape), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        dataset = dataset.cache()

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_labels(data_dir: str | Path) -> np.ndarray:
    """Return labels for all discovered spectrogram files in stable sorted order."""
    _, labels = discover_spectrogram_files(data_dir)
    return np.asarray(labels, dtype=np.int32)


def _to_autoencoder_pair(
    filepath: tf.Tensor,
    input_shape: tuple[int, int] | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    spectrogram = _load_spectrogram_tf(filepath, input_shape)
    return spectrogram, spectrogram
