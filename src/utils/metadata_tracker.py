"""
metadata_tracker.py - Save and load run metadata as JSON.

Each run produces a JSON file in artifacts/metadata/ containing
parameters, metrics, artifact paths, and MLflow run ID.

Usage:
    from src.utils.metadata_tracker import MetadataTracker

    tracker = MetadataTracker(metadata_dir="artifacts/metadata")
    tracker.save(run_id="20260429_210000", params={...}, metrics={...}, artifacts={...})
    data = tracker.load("20260429_210000")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataTracker:
    """Save and load per-run metadata as JSON files."""

    def __init__(self, metadata_dir: str = "artifacts/metadata"):
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        run_id: str,
        params: dict,
        metrics: dict | None = None,
        artifacts: dict | None = None,
        mlflow_run_id: str | None = None,
        experiment_name: str | None = None,
    ) -> Path:
        """
        Save run metadata to a JSON file.

        Args:
            run_id: Unique run identifier (e.g., timestamp string).
            params: Dictionary of run parameters (LR, epochs, seed, etc.).
            metrics: Dictionary of evaluation metrics (loss, AUC, etc.).
            artifacts: Dictionary of artifact paths (model, logs, etc.).
            mlflow_run_id: Optional MLflow run ID for cross-reference.
            experiment_name: Optional experiment name.

        Returns:
            Path to the saved JSON file.
        """
        metadata = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "mlflow_run_id": mlflow_run_id,
        }

        filepath = self.metadata_dir / f"run_{run_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Metadata saved: {filepath}")
        return filepath

    def load(self, run_id: str) -> dict:
        """
        Load metadata for a specific run.

        Args:
            run_id: Run identifier matching the filename.

        Returns:
            Metadata dictionary.
        """
        filepath = self.metadata_dir / f"run_{run_id}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Metadata loaded: {filepath}")
        return data

    def list_runs(self) -> list[str]:
        """List all run IDs that have saved metadata."""
        runs = []
        for f in sorted(self.metadata_dir.glob("run_*.json")):
            run_id = f.stem.replace("run_", "")
            runs.append(run_id)
        return runs
