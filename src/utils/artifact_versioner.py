"""
artifact_versioner.py - Auto-increment versioned artifact folders.

Manages versioned model storage under artifacts/models/:
    artifacts/models/v1/
    artifacts/models/v2/
    ...

Usage:
    from src.utils.artifact_versioner import ArtifactVersioner

    versioner = ArtifactVersioner(base_dir="artifacts/models")

    # Auto-increment mode:
    version_dir = versioner.get_next_version()   # → "artifacts/models/v3"

    # Manual mode:
    version_dir = versioner.get_version("v1_experiment_01")

    # Save config snapshot alongside model:
    versioner.save_config_snapshot(version_dir, config_dict)
"""

import os
import re
import yaml
import shutil
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactVersioner:
    """
    Manages versioned artifact directories for model saving.

    Supports two modes (controlled via config):
        - "auto": Scans existing v1/, v2/, ... and increments to the next version.
        - "manual": Uses a user-specified version string from config.
    """

    def __init__(self, base_dir: str = "artifacts/models"):
        """
        Args:
            base_dir: Root directory where versioned folders are created.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version(self) -> Path:
        """
        Auto-detect the next version number by scanning existing directories.

        Looks for folders matching the pattern v1, v2, v3, ... and returns
        the path for the next increment (e.g., v4).

        Returns:
            Path to the new version directory (created on disk).
        """
        existing_versions = []

        for item in self.base_dir.iterdir():
            if item.is_dir():
                match = re.match(r"^v(\d+)$", item.name)
                if match:
                    existing_versions.append(int(match.group(1)))

        next_version = max(existing_versions, default=0) + 1
        version_dir = self.base_dir / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created version directory: {version_dir}")
        return version_dir

    def get_version(self, version_name: str) -> Path:
        """
        Get or create a specific named version directory.

        Args:
            version_name: Version identifier (e.g., "v5", "v1_experiment_01").

        Returns:
            Path to the version directory (created on disk).
        """
        version_dir = self.base_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using version directory: {version_dir}")
        return version_dir

    def resolve_version(self, config: dict) -> Path:
        """
        Resolve the version directory based on config settings.

        Reads config["versioning"]["mode"]:
            - "auto" → calls get_next_version()
            - "manual" → calls get_version() with config["versioning"]["manual_version"]

        Args:
            config: Full config dictionary containing the "versioning" key.

        Returns:
            Path to the resolved version directory.
        """
        versioning_cfg = config.get("versioning", {})
        mode = versioning_cfg.get("mode", "auto")

        if mode == "manual":
            manual_version = versioning_cfg.get("manual_version")
            if not manual_version:
                logger.warning("Manual mode selected but no manual_version set. Falling back to auto.")
                return self.get_next_version()
            return self.get_version(manual_version)
        else:
            return self.get_next_version()

    def save_config_snapshot(self, version_dir: Path, config: dict) -> Path:
        """
        Save a YAML snapshot of the config used for this run.

        This ensures full reproducibility — the exact config that produced
        a model is stored alongside it.

        Args:
            version_dir: Path to the version directory (e.g., artifacts/models/v3).
            config: Full config dictionary to save.

        Returns:
            Path to the saved config snapshot file.
        """
        snapshot_path = Path(version_dir) / "config_snapshot.yaml"

        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config snapshot saved: {snapshot_path}")
        return snapshot_path

    def list_versions(self) -> list[str]:
        """
        List all existing version directories, sorted numerically.

        Returns:
            Sorted list of version directory names (e.g., ["v1", "v2", "v3"]).
        """
        versions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name != ".gitkeep":
                versions.append(item.name)

        # Sort: numeric versions first (v1, v2, ...), then alphabetical
        def sort_key(name):
            match = re.match(r"^v(\d+)$", name)
            return (0, int(match.group(1))) if match else (1, name)

        return sorted(versions, key=sort_key)

    def get_latest_version(self) -> Path | None:
        """
        Get the path to the latest (highest numbered) version directory.

        Returns:
            Path to the latest version, or None if no versions exist.
        """
        versions = self.list_versions()
        if not versions:
            logger.warning("No existing versions found.")
            return None

        latest = versions[-1]
        return self.base_dir / latest
