"""Mortality monotonicity loss.

Created: 2026-06-03
Purpose: Encourage reserves to increase when mortality intensity increases.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class MortalityMonotonicityLoss(BaseLoss):
    """Penalize violations of ``dV/dμ >= 0``.

    Scientific Context:
        Higher mortality generally increases the expected present value of death
        benefits for term-life business, so reserves should not fall as
        mortality rises.

    Business Interpretation:
        This is a knowledge-informed control that discourages the model from
        learning reserve curves that contradict a basic actuarial intuition.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the mortality monotonicity penalty."""

        del model, context
        derivative = self.first_derivative(predictions=predictions, batch=batch, name="mortality")
        return self.reduce(torch.relu(-derivative))
