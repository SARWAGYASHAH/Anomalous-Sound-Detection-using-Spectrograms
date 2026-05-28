"""Raw WAV dataset helpers for STgram-MFN training and evaluation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


SECTION_PATTERN = re.compile(r"section_\d+")


def discover_wav_files(directory: str | Path) -> list[Path]:
    """Return sorted WAV paths from a DCASE-style directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"WAV directory not found: {directory}")
    return sorted(directory.glob("*.wav"))


def extract_section(filepath: str | Path) -> str:
    """Extract section id such as section_00 from a filename."""
    match = SECTION_PATTERN.search(Path(filepath).name)
    if not match:
        raise ValueError(f"Could not extract section id from {filepath}")
    return match.group(0)


def extract_binary_label(filepath: str | Path) -> int:
    """Return 0 for normal and 1 for anomaly based on filename."""
    name = Path(filepath).name.lower()
    if "anomaly" in name:
        return 1
    if "normal" in name:
        return 0
    raise ValueError(f"Could not extract normal/anomaly label from {filepath}")


def build_section_label_map(filepaths: Iterable[str | Path]) -> dict[str, int]:
    """Build a deterministic section-to-class mapping from file paths."""
    sections = sorted({extract_section(path) for path in filepaths})
    return {section: index for index, section in enumerate(sections)}


class STgramWaveDataset(Dataset):
    """Load waveform and log-mel tensors for STgram-MFN."""

    def __init__(
        self,
        filepaths: Iterable[str | Path],
        section_to_label: dict[str, int],
        sample_rate: int = 16000,
        duration: float = 10.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        power: float = 2.0,
        normalize_mel: bool = True,
        require_binary_label: bool = False,
    ) -> None:
        self.filepaths = [Path(path) for path in filepaths]
        self.section_to_label = dict(section_to_label)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.normalize_mel = bool(normalize_mel)
        self.require_binary_label = bool(require_binary_label)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> dict:
        filepath = self.filepaths[index]
        section = extract_section(filepath)
        if section not in self.section_to_label:
            raise KeyError(f"Section {section} is missing from section_to_label")

        waveform = self._load_waveform(filepath)
        mel = self._waveform_to_log_mel(waveform)

        binary_label = extract_binary_label(filepath) if self.require_binary_label else -1
        return {
            "waveform": torch.from_numpy(waveform).float(),
            "mel": torch.from_numpy(mel).float(),
            "section_label": torch.tensor(self.section_to_label[section], dtype=torch.long),
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "section": section,
            "filepath": str(filepath),
        }

    def _load_waveform(self, filepath: Path) -> np.ndarray:
        waveform, _ = librosa.load(str(filepath), sr=self.sample_rate, mono=True)
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[0] < target_length:
            waveform = np.pad(waveform, (0, target_length - waveform.shape[0]))
        elif waveform.shape[0] > target_length:
            waveform = waveform[:target_length]
        return waveform.astype(np.float32, copy=False)

    def _waveform_to_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if self.normalize_mel:
            minimum = float(mel_db.min())
            maximum = float(mel_db.max())
            if maximum > minimum:
                mel_db = (mel_db - minimum) / (maximum - minimum)
            else:
                mel_db = np.zeros_like(mel_db)
        return mel_db.astype(np.float32, copy=False)
