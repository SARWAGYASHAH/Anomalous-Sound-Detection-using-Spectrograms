"""Backend services exposing model artifacts and WAV inference to the dashboard."""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
import yaml

from src.inference.anomaly_scorer import ThresholdResult
from src.inference.predictor import (
    AudioPredictor,
    fit_reference_threshold,
    load_project_model,
    resolve_latest_model_path,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardService:
    """Read project state and perform predictions for the web API."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        config_path: str | Path = "config/default.yaml",
    ):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.config_path = self.project_root / config_path
        with open(self.config_path, "r", encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle)

        self.artifacts_dir = self.project_root / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.evaluation_dir = self.artifacts_dir / "evaluation"
        self.predictions_dir = self.artifacts_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache: dict[str, tf.keras.Model] = {}
        self._threshold_cache: dict[str, ThresholdResult] = {}

    def health(self) -> dict[str, Any]:
        """Return application and model availability status."""
        try:
            model_path = self.resolve_model_path(None)
        except FileNotFoundError:
            model_path = None
        return {
            "status": "ready",
            "service": "SoundGuard AI",
            "framework": "TensorFlow / Keras",
            "model_available": model_path is not None,
            "active_model": str(model_path.relative_to(self.project_root)) if model_path else None,
            "evaluation_available": (self.evaluation_dir / "source_test" / "metrics.json").exists(),
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List versioned model artifacts available for analysis."""
        models: list[dict[str, Any]] = []
        if not self.models_dir.exists():
            return models

        def version_key(path: Path) -> tuple[int, str]:
            match = re.fullmatch(r"v(\d+)", path.name)
            return (int(match.group(1)), path.name) if match else (-1, path.name)

        for version_dir in sorted((path for path in self.models_dir.iterdir() if path.is_dir()), key=version_key, reverse=True):
            available_files = [
                name
                for name in ("model.keras", "best_model.keras", "final_model.keras", "config_snapshot.yaml", "training_log.csv")
                if (version_dir / name).exists()
            ]
            model_file = next((name for name in ("model.keras", "best_model.keras", "final_model.keras") if name in available_files), None)
            if not model_file:
                continue
            models.append(
                {
                    "version": version_dir.name,
                    "model_file": model_file,
                    "model_path": str((version_dir / model_file).relative_to(self.project_root)),
                    "files": available_files,
                    "modified_at": datetime.fromtimestamp((version_dir / model_file).stat().st_mtime).isoformat(),
                }
            )
        return models

    def dashboard(self) -> dict[str, Any]:
        """Compose the operational summary used by the main dashboard."""
        health = self.health()
        evaluation = self.evaluation("source_test", required=False)
        score_rows: list[dict[str, Any]] = []
        counts = {"normal": 0, "follow_up": 0, "alert": 0}
        if evaluation:
            score_path = self.project_root / evaluation["artifacts"]["scores_csv"]
            if score_path.exists():
                scores = pd.read_csv(score_path)
                for severity, count in scores["severity"].value_counts().items():
                    counts[str(severity)] = int(count)
                score_rows = [
                    {
                        "sample": int(index + 1),
                        "score": float(row.score),
                        "threshold": float(evaluation["threshold"]["value"]),
                        "severity": str(row.severity),
                    }
                    for index, row in scores.tail(32).reset_index(drop=True).iterrows()
                ]

        recent_predictions = []
        for path in sorted(self.predictions_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:6]:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload.setdefault(
                "created_at",
                datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            )
            recent_predictions.append(payload)

        processed_dir = self.project_root / self.config["data"]["processed_dir"]
        test_total = sum(len(list((processed_dir / "source_test" / label).glob("*.npy"))) for label in ("normal", "anomaly"))
        models = self.list_models()
        active_model = next((model for model in models if model["model_path"] == health["active_model"]), None)
        activity = [
            {
                "filename": prediction["audio_file"],
                "severity": prediction.get("severity", "normal"),
                "score": prediction.get("score"),
                "created_at": prediction["created_at"],
            }
            for prediction in recent_predictions[:4]
        ]
        return {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "health": health,
            "models": models,
            "active_model": active_model or (models[0] if models else None),
            "kpis": {
                "test_samples": test_total,
                "normal": counts["normal"],
                "follow_up": counts["follow_up"],
                "alert": counts["alert"],
                "auc": evaluation["metrics"]["auc_roc"] if evaluation else None,
            },
            "trend": score_rows,
            "recent_predictions": recent_predictions,
            "activity": activity,
            "evaluation": evaluation,
        }

    def evaluation(self, split: str, required: bool = True) -> dict[str, Any] | None:
        """Read saved evaluation metrics for a supported test split."""
        if split not in {"source_test", "target_test"}:
            raise ValueError("split must be source_test or target_test")
        metrics_path = self.evaluation_dir / split / "metrics.json"
        if not metrics_path.exists():
            if required:
                raise FileNotFoundError(f"Evaluation metrics not found for {split}")
            return None
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["plot_urls"] = {
            "scores": f"/artifacts/evaluation/{split}/anomaly_score_distribution.png",
            "roc": f"/artifacts/evaluation/{split}/roc_curve.png",
        }
        return payload

    def predict_audio(self, filename: str, content: bytes, model_version: str | None = None) -> dict[str, Any]:
        """Save an uploaded WAV temporarily and score it through the Keras pipeline."""
        if not content:
            raise ValueError("Uploaded audio file is empty.")
        safe_name = Path(filename).name
        if Path(safe_name).suffix.lower() != ".wav":
            raise ValueError("Only .wav audio files are supported.")

        model_path = self.resolve_model_path(model_version)
        model = self._load_model(model_path)
        threshold = self._resolve_threshold(model_path, model)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.predictions_dir) as handle:
            handle.write(content)
            temp_path = Path(handle.name)

        try:
            result = AudioPredictor(model, self.config, threshold, model_path).predict_file(temp_path)
            payload = result.to_dict()
            created_at = datetime.now().astimezone()
            payload["audio_file"] = safe_name
            payload["model_path"] = str(model_path.relative_to(self.project_root))
            payload["model_version"] = model_path.parent.name
            payload["created_at"] = created_at.isoformat(timespec="seconds")
            output_name = f"{Path(safe_name).stem}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}.json"
            output_path = self.predictions_dir / output_name
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            payload["result_url"] = f"/artifacts/predictions/{output_name}"
            return payload
        finally:
            temp_path.unlink(missing_ok=True)

    def resolve_model_path(self, version: str | None) -> Path:
        """Resolve an allowed model version or return the latest saved model."""
        if version is None or version == "latest":
            evaluated = self.evaluation("source_test", required=False)
            if evaluated:
                evaluated_path = self.project_root / evaluated["model_path"]
                if evaluated_path.exists():
                    return evaluated_path
            return resolve_latest_model_path(self.models_dir)
        if not re.fullmatch(r"v\d+", version):
            raise ValueError("model_version must look like v1, v2, or latest.")
        version_dir = self.models_dir / version
        for filename in ("model.keras", "best_model.keras", "final_model.keras"):
            candidate = version_dir / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No model artifact found for version {version}.")

    def _load_model(self, model_path: Path) -> tf.keras.Model:
        cache_key = str(model_path.resolve())
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = load_project_model(model_path)
        return self._model_cache[cache_key]

    def _resolve_threshold(self, model_path: Path, model: tf.keras.Model) -> ThresholdResult:
        cache_key = str(model_path.resolve())
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        source_metrics = self.evaluation("source_test", required=False)
        if source_metrics:
            evaluated_path = (self.project_root / source_metrics["model_path"]).resolve()
            if evaluated_path == model_path.resolve():
                threshold_data = source_metrics["threshold"]
                threshold = ThresholdResult(**threshold_data)
                self._threshold_cache[cache_key] = threshold
                return threshold

        threshold = fit_reference_threshold(model, self.config)
        self._threshold_cache[cache_key] = threshold
        return threshold
