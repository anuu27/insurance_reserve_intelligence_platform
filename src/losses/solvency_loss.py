"""Solvency floor loss.

Created: 2026-06-03
Purpose: Penalize negative reserves that violate a non-negativity expectation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class SolvencyLoss(BaseLoss):
    """Penalize reserves below zero.

    Scientific Context:
        Negative reserves may occur in niche accounting settings, but many
        reserve-control experiments prefer a non-negative liability surface as a
        prudential modeling assumption.

    Business Interpretation:
        This loss is a simple solvency-style safety rail that blocks obviously
        problematic reserve outputs in governance dashboards.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the non-negativity penalty."""

        del model, batch, context
        return self.reduce(torch.relu(-predictions))
