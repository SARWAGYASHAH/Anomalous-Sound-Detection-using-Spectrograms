"""
audio_loader.py - Load .wav files and resample to a target sample rate.

Handles single file loading, batch loading from directories,
and label extraction from DCASE-format filenames.

Usage:
    from src.data.audio_loader import AudioLoader

    loader = AudioLoader(target_sr=16000)
    waveform, sr = loader.load("path/to/file.wav")
    files = loader.discover_files("Data/gearbox/train")
"""

import os
from pathlib import Path

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioLoader:
    """
    Load and resample .wav audio files.

    Attributes:
        target_sr: Target sample rate for resampling.
        mono: Whether to convert to mono.
        duration: Expected clip duration in seconds (for validation).
    """

    def __init__(
        self,
        target_sr: int = 16000,
        mono: bool = True,
        duration: float | None = None,
    ):
        self.target_sr = target_sr
        self.mono = mono
        self.duration = duration

    def load(self, filepath: str | Path) -> tuple[np.ndarray, int]:
        """
        Load a single .wav file and resample to target_sr.

        Args:
            filepath: Path to the .wav file.

        Returns:
            Tuple of (waveform as np.ndarray, sample_rate as int).
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        waveform, sr = librosa.load(
            str(filepath),
            sr=self.target_sr,
            mono=self.mono,
        )

        # Validate duration if specified
        if self.duration is not None:
            expected_samples = int(self.target_sr * self.duration)
            if len(waveform) < expected_samples:
                # Pad short clips with zeros
                waveform = np.pad(
                    waveform,
                    (0, expected_samples - len(waveform)),
                    mode="constant",
                )
            elif len(waveform) > expected_samples:
                # Trim long clips
                waveform = waveform[:expected_samples]

        return waveform, self.target_sr

    def discover_files(self, directory: str | Path) -> list[dict]:
        """
        Discover all .wav files in a directory and extract metadata from filenames.

        DCASE filename format:
            section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav
            section_00_source_test_anomaly_0001.wav

        Returns:
            List of dicts with keys: "filepath", "filename", "label", "section".
        """
        directory = Path(directory)
        files = []

        for wav_file in sorted(directory.glob("*.wav")):
            label = self._extract_label(wav_file.name)
            section = self._extract_section(wav_file.name)

            files.append({
                "filepath": str(wav_file),
                "filename": wav_file.name,
                "label": label,
                "section": section,
            })

        logger.info(f"Discovered {len(files)} .wav files in {directory}")
        return files

    def load_batch(
        self,
        file_list: list[dict],
        max_files: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load a batch of audio files into a single numpy array.

        Args:
            file_list: List of file dicts from discover_files().
            max_files: Optional limit on number of files to load.

        Returns:
            Tuple of (waveforms array [N, samples], list of labels).
        """
        if max_files is not None:
            file_list = file_list[:max_files]

        waveforms = []
        labels = []

        for i, file_info in enumerate(file_list):
            waveform, _ = self.load(file_info["filepath"])
            waveforms.append(waveform)
            labels.append(file_info["label"])

            if (i + 1) % 500 == 0:
                logger.info(f"Loaded {i + 1}/{len(file_list)} files...")

        logger.info(f"Loaded {len(waveforms)} audio files")
        return np.array(waveforms), labels

    @staticmethod
    def _extract_label(filename: str) -> str:
        """Extract label (normal/anomaly) from DCASE filename."""
        if "anomaly" in filename:
            return "anomaly"
        elif "normal" in filename:
            return "normal"
        return "unknown"

    @staticmethod
    def _extract_section(filename: str) -> str:
        """Extract section ID (e.g., 'section_00') from DCASE filename."""
        parts = filename.split("_")
        if len(parts) >= 2 and parts[0] == "section":
            return f"section_{parts[1]}"
        return "unknown"
