"""
seed.py - Central seed handler for reproducibility.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    logger.info(f"Random seed set to {seed} (random, numpy, tensorflow)")
