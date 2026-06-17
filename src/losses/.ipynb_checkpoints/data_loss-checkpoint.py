"""Supervised reserve-fitting loss.

Created: 2026-06-03
Purpose: Measure error between predicted reserves and classical actuarial reserves.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class DataLoss(BaseLoss):
    """Supervised fitting term using Huber loss.

    Why Huber instead of MSE:
        MSE squares large errors, so a handful of hard policies (long term,
        extreme age) dominate the gradient and pull the model away from the
        majority.  Huber is quadratic for small errors (like MSE) but linear
        for large ones, so outlier policies don't drown the signal from the
        typical policy.

        With standardised targets z ~ N(0,1), delta=1.0 means errors smaller
        than 1 standard deviation are treated quadratically and larger ones
        are clipped to linear — a natural threshold.
    """

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction)
        self.delta = delta

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del model, context
        targets = self.require_batch_tensor(batch, "target")
        return self.reduce(
            torch.nn.functional.huber_loss(
                predictions, targets,
                reduction="none",
                delta=self.delta,
            )
        )