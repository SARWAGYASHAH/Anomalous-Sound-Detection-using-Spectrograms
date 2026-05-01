"""
spectrogram.py - Convert audio waveforms to spectrograms and save as images/numpy.

Pipeline: Waveform → STFT → Mel filter bank → dB scale → (optional normalize) → save

Usage:
    from src.data.spectrogram import SpectrogramExtractor

    extractor = SpectrogramExtractor(n_fft=1024, hop_length=512, n_mels=128)
    mel_db = extractor.waveform_to_mel_spectrogram(waveform, sr=16000)
    extractor.save_as_image(mel_db, "output/spec.png")
    extractor.save_as_numpy(mel_db, "output/spec.npy")
"""

import os
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpectrogramExtractor:
    """
    Extract and save spectrograms from audio waveforms.

    Supports mel, STFT, and MFCC spectrogram types.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        power: float = 2.0,
        normalize: bool = True,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.normalize = normalize

    def waveform_to_mel_spectrogram(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Convert waveform to log-mel spectrogram in dB scale.

        Args:
            waveform: 1D numpy array of audio samples.
            sr: Sample rate.

        Returns:
            2D numpy array (n_mels × time_frames) in dB scale.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )

        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Optional normalization to [0, 1]
        if self.normalize:
            mel_db = self._normalize(mel_db)

        return mel_db

    def waveform_to_stft(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Compute STFT magnitude spectrogram in dB.

        Returns:
            2D numpy array (freq_bins × time_frames) in dB scale.
        """
        stft = np.abs(librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ))
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)

        if self.normalize:
            stft_db = self._normalize(stft_db)

        return stft_db

    def waveform_to_mfcc(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        n_mfcc: int = 40,
    ) -> np.ndarray:
        """
        Compute MFCCs from waveform.

        Returns:
            2D numpy array (n_mfcc × time_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if self.normalize:
            mfcc = self._normalize(mfcc)

        return mfcc

    def extract(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        spec_type: str = "mel",
    ) -> np.ndarray:
        """
        Extract spectrogram of the given type.

        Args:
            waveform: Audio waveform.
            sr: Sample rate.
            spec_type: "mel", "stft", or "mfcc".

        Returns:
            2D numpy spectrogram array.
        """
        if spec_type == "mel":
            return self.waveform_to_mel_spectrogram(waveform, sr)
        elif spec_type == "stft":
            return self.waveform_to_stft(waveform)
        elif spec_type == "mfcc":
            return self.waveform_to_mfcc(waveform, sr)
        else:
            raise ValueError(f"Unknown spectrogram type: {spec_type}")

    def save_as_image(
        self,
        spectrogram: np.ndarray,
        save_path: str | Path,
        sr: int = 16000,
        figsize: tuple = (10, 4),
    ) -> None:
        """
        Save spectrogram as a .png image.

        Args:
            spectrogram: 2D spectrogram array.
            save_path: Output file path.
            sr: Sample rate (for axis labels).
            figsize: Figure dimensions.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        img = librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        ax.set_title("")
        ax.axis("off")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout(pad=0)
        fig.savefig(
            str(save_path),
            bbox_inches="tight",
            pad_inches=0,
            dpi=100,
        )
        plt.close(fig)

    def save_as_numpy(
        self,
        spectrogram: np.ndarray,
        save_path: str | Path,
    ) -> None:
        """
        Save spectrogram as a .npy file for fast loading in training.

        Args:
            spectrogram: 2D spectrogram array.
            save_path: Output file path (should end with .npy).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), spectrogram)

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        d_min = data.min()
        d_max = data.max()
        if d_max - d_min == 0:
            return np.zeros_like(data)
        return (data - d_min) / (d_max - d_min)
