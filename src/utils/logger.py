"""Logging utilities.

Created: 2026-05-31
Purpose: Configure structured logging for training, evaluation, and reporting workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create a console and optional file logger.

    Args:
        name: Logger name.
        log_file: Optional log-file path.
        level: Logging verbosity level.

    Returns:
        logging.Logger: Configured logger instance.

    Business Interpretation:
        This supports auditability by persisting model-development events and
        operational diagnostics.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
