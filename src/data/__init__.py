"""
src.data - Data loading, spectrogram extraction, and TensorFlow datasets.

Modules:
    audio_loader  - Load and resample .wav files
    spectrogram   - Extract mel/STFT/MFCC spectrograms
    dataset       - tf.data pipelines for processed spectrograms
"""

from src.data.audio_loader import AudioLoader
from src.data.spectrogram import SpectrogramExtractor
from src.data.dataset import (
    LABEL_MAP,
    create_autoencoder_dataset,
    create_labeled_dataset,
    discover_spectrogram_files,
    get_labels,
    load_spectrogram_array,
)

__all__ = [
    "AudioLoader",
    "SpectrogramExtractor",
    "LABEL_MAP",
    "create_autoencoder_dataset",
    "create_labeled_dataset",
    "discover_spectrogram_files",
    "get_labels",
    "load_spectrogram_array",
]
