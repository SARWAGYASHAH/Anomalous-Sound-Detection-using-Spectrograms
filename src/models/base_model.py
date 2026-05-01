"""
base_model.py - Shared abstract interface for project models.

Concrete models should inherit from BaseModel and implement forward().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
