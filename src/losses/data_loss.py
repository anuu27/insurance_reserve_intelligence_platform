"""Supervised reserve-fitting loss.

Created: 2026-06-03
Purpose: Measure squared error between predicted reserves and classical actuarial reserves.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class DataLoss(BaseLoss):
    """Classical supervised learning term for reserve prediction.

    Scientific Context:
        This loss anchors the neural network to benchmark reserves produced by
        the actuarial solver.

    Business Interpretation:
        It answers the most direct business question: how close is the model to
        the reserve amount the legacy valuation engine would report?
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute mean squared reserve error.

        Args:
            model: Reserve model. Unused here because the predictions are
                already supplied by the trainer.
            batch: Batch dictionary containing ``target``.
            predictions: Model reserve predictions.
            context: Shared execution context. Unused for this loss.

        Returns:
            torch.Tensor: Reduced ``L_data`` scalar.
        """

        del model, context
        targets = self.require_batch_tensor(batch, "target")
        return self.reduce((predictions - targets).pow(2))
