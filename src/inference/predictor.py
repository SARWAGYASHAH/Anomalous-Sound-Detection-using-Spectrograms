"""End-to-end prediction for one audio file."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data import AudioLoader, SpectrogramExtractor, create_autoencoder_dataset
from src.inference.anomaly_scorer import (
    ReconstructionAnomalyScorer,
    ThresholdResult,
    score_to_probability,
    threshold_from_config,
)
from src.inference.classifier import SeverityClassifier
from src.models import Conv2DAutoencoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Serializable result of scoring one audio file."""

    audio_file: str
    model_path: str
    score: float
    normalized_score: float
    threshold: float
    threshold_method: str
    severity: str
    is_anomaly: bool
    requires_attention: bool
    crosses_threshold: bool
    spectrogram_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly prediction data."""
        return asdict(self)


def resolve_latest_model_path(models_dir: str | Path = "artifacts/models") -> Path:
    """Select the newest saved Keras model, preferring each run's best checkpoint."""
    base_dir = Path(models_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {base_dir}")

    def version_key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"v(\d+)", path.name)
        return (int(match.group(1)), path.name) if match else (-1, path.name)

    for version_dir in sorted((path for path in base_dir.iterdir() if path.is_dir()), key=version_key, reverse=True):
        for filename in ("model.keras", "best_model.keras", "final_model.keras"):
            candidate = version_dir / filename
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"No saved .keras model found below {base_dir}")


def load_project_model(model_path: str | Path) -> tf.keras.Model:
    """Load a serialized autoencoder, including project custom classes."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Conv2DAutoencoder": Conv2DAutoencoder},
        compile=False,
    )
    logger.info("Loaded model: %s", model_path)
    return model


def fit_reference_threshold(
    model: tf.keras.Model,
    config: dict[str, Any],
    reference_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> ThresholdResult:
    """Fit the configured anomaly threshold from normal training spectrograms."""
    reference_path = Path(reference_dir or Path(config["data"]["processed_dir"]) / "train")
    dataset = create_autoencoder_dataset(
        reference_path,
        batch_size=batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        input_shape=(int(config["spectrogram"]["n_mels"]), None),
    )
    scorer = ReconstructionAnomalyScorer(model=model)
    normal_scores, _ = scorer.score_dataset(dataset)
    if normal_scores.size == 0:
        raise ValueError(f"No normal reference samples found in {reference_path}")
    return threshold_from_config(normal_scores, config)


class AudioPredictor:
    """Load an audio file, compute reconstruction error, and assign severity."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: dict[str, Any],
        threshold: ThresholdResult,
        model_path: str | Path,
    ):
        self.model = model
        self.config = config
        self.threshold = threshold
        self.model_path = Path(model_path)
        self.scorer = ReconstructionAnomalyScorer(model=model, threshold=threshold.value)
        self.classifier = SeverityClassifier.from_config(config)

        audio_config = config["audio"]
        spectrogram_config = config["spectrogram"]
        self.loader = AudioLoader(
            target_sr=int(audio_config["sample_rate"]),
            mono=bool(audio_config["mono"]),
            duration=float(audio_config["duration"]),
        )
        self.extractor = SpectrogramExtractor(
            n_fft=int(spectrogram_config["n_fft"]),
            hop_length=int(spectrogram_config["hop_length"]),
            n_mels=int(spectrogram_config["n_mels"]),
            power=float(spectrogram_config["power"]),
            normalize=bool(spectrogram_config["normalize"]),
        )
        self.spec_type = spectrogram_config["type"]

    def predict_file(self, audio_file: str | Path) -> PredictionResult:
        """Predict anomaly severity for one WAV file."""
        audio_path = Path(audio_file)
        waveform, sample_rate = self.loader.load(audio_path)
        spectrogram = self.extractor.extract(waveform, sr=sample_rate, spec_type=self.spec_type).astype(np.float32)
        input_batch = spectrogram[np.newaxis, ..., np.newaxis]
        score = float(self.scorer.score_batch(input_batch)[0])
        normalized_score = float(score_to_probability([score], self.threshold.value)[0])
        classification = self.classifier.predict_results([normalized_score])[0]

        return PredictionResult(
            audio_file=str(audio_path),
            model_path=str(self.model_path),
            score=score,
            normalized_score=normalized_score,
            threshold=float(self.threshold.value),
            threshold_method=self.threshold.method,
            severity=classification.label,
            is_anomaly=classification.is_anomaly,
            requires_attention=classification.requires_attention,
            crosses_threshold=bool(score >= self.threshold.value),
            spectrogram_shape=tuple(int(value) for value in input_batch.shape[1:]),
        )
