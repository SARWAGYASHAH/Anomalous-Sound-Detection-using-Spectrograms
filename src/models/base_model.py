"""
base_model.py - Shared abstract interface for project models.

Concrete models should inherit from BaseModel and implement forward().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """
    Abstract parent class for all trainable models in the project.

    The base class keeps common metadata in one place while leaving the
    architecture-specific forward pass to each concrete model.
    """

    def __init__(self, config: dict[str, Any] | None = None, name: str | None = None):
        super().__init__()
        self.config = config or {}
        self.name = name or self.__class__.__name__

    @property
    def device(self) -> torch.device:
        """
        Return the device currently used by the model parameters.

        Models without parameters default to CPU.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_config(self) -> dict[str, Any]:
        """Return a shallow copy of the model configuration."""
        return dict(self.config)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: When True, count only parameters that require gradients.

        Returns:
            Number of parameters.
        """
        parameters = self.parameters()
        if trainable_only:
            parameters = (param for param in parameters if param.requires_grad)
        return sum(param.numel() for param in parameters)

    def save_checkpoint(
        self,
        filepath: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save model weights and lightweight metadata.

        Args:
            filepath: Destination checkpoint path.
            extra: Optional run metadata such as epoch, loss, or threshold.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_path = Path(filepath)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_name": self.name,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        filepath: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load model weights from a checkpoint produced by save_checkpoint().

        Args:
            filepath: Source checkpoint path.
            map_location: Device mapping used by torch.load.
            strict: Passed to load_state_dict.

        Returns:
            Full checkpoint dictionary for callers that need metadata.
        """
        checkpoint = torch.load(
            Path(filepath),
            map_location=map_location,
            weights_only=False,
        )
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        self.config = checkpoint.get("config", self.config)
        self.name = checkpoint.get("model_name", self.name)
        return checkpoint

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the model.

        Args:
            inputs: Input tensor batch.

        Returns:
            Output tensor batch.
        """
        raise NotImplementedError
