"""Reproducibility helpers.

Created: 2026-05-31
Purpose: Enforce deterministic random-state initialization across supported libraries.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across supported libraries.

    Args:
        seed: Seed value to apply.

    Business Interpretation:
        This improves experiment reproducibility, which matters for model review,
        validation, and research traceability.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
