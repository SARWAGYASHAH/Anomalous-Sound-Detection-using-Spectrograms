"""
base_model.py - Shared Keras model interface for project models.

Concrete models should inherit from BaseModel and implement call().
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import tensorflow as tf


class BaseModel(tf.keras.Model, ABC):
    """
    Abstract parent class for trainable Keras models in the project.

    The base class keeps lightweight project metadata in one place while
    leaving architecture-specific computation to each concrete model.
    """

    def __init__(self, config: dict[str, Any] | None = None, name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self.project_config = config or {}

    @property
    def device(self) -> str:
        """
        Return the device TensorFlow is expected to use.

        TensorFlow places operations automatically, so this is informational.
        """
        gpus = tf.config.list_physical_devices("GPU")
        return "GPU" if gpus else "CPU"

    def get_project_config(self) -> dict[str, Any]:
        """Return a shallow copy of the project-specific model configuration."""
        return dict(self.project_config)

    def get_config(self) -> dict[str, Any]:
        """Return Keras serialization config with project metadata included."""
        return {
            **super().get_config(),
            "project_config": self.get_project_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseModel":
        """Recreate a model instance from Keras serialization config."""
        config = dict(config)
        project_config = config.pop("project_config", None)
        name = config.pop("name", None)
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(config=project_config, name=name)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: When True, count only trainable weights.

        Returns:
            Number of parameters.
        """
        weights = self.trainable_weights if trainable_only else self.weights
        return int(sum(tf.keras.backend.count_params(weight) for weight in weights))

    def save_checkpoint(
        self,
        filepath: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save the model and lightweight metadata.

        Args:
            filepath: Destination model path, usually ending in .keras.
            extra: Optional run metadata such as epoch, loss, or threshold.

        Returns:
            Path to the saved model.
        """
        checkpoint_path = Path(filepath)
        if checkpoint_path.suffix == "":
            checkpoint_path = checkpoint_path.with_suffix(".keras")

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.save(checkpoint_path)

        metadata = {
            "model_name": self.name,
            "config": self.get_project_config(),
            "extra": extra or {},
        }
        metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        return checkpoint_path

    def load_checkpoint(
        self,
        filepath: str | Path,
        custom_objects: dict[str, Any] | None = None,
    ) -> tf.keras.Model:
        """
        Load weights from a saved Keras model into this instance.

        Args:
            filepath: Source .keras model path.
            custom_objects: Optional Keras custom object mapping.

        Returns:
            Loaded Keras model. This instance also receives the loaded weights
            when its architecture is compatible and already built.
        """
        loaded_model = tf.keras.models.load_model(
            filepath,
            custom_objects=custom_objects,
            compile=False,
        )
        self.set_weights(loaded_model.get_weights())
        return loaded_model

    @abstractmethod
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Run a forward pass through the model.

        Args:
            inputs: Input tensor batch.
            training: Whether the model is running in training mode.

        Returns:
            Output tensor batch.
        """
        raise NotImplementedError
