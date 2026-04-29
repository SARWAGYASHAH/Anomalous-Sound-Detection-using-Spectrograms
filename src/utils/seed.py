"""
seed.py - Central seed handler for reproducibility.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:
        logger.warning("PyTorch not installed. Skipping torch seed setup.")

    logger.info(f"Random seed set to {seed} (random, numpy, torch)")
