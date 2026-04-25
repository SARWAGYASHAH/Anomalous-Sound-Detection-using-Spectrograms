"""
logger.py - Structured logging with console and file output.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")

    # With file logging for a specific run:
    logger = get_logger(__name__, log_dir="artifacts/logs", run_id="20260425_221400")
"""

import logging
import os
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str | None = None,
    run_id: str | None = None,
) -> logging.Logger:
    """
    Create a structured logger with console output and optional file logging.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        log_dir: Directory for log files (e.g., "artifacts/logs").
                 If None, only console logging is enabled.
        run_id: Unique run identifier for the log filename.
                If None, a timestamp-based ID is generated.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ---------- Formatter ----------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- Console Handler ----------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---------- File Handler (optional) ----------
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_filename = f"run_{run_id}.log"
        log_path = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file: {log_path}")

    return logger
