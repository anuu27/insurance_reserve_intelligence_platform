"""Interest-rate monotonicity loss.

Purpose: Enforce dV/dr <= 0 via finite-difference perturbation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX
from src.losses.base_loss import BaseLoss

_R_IDX = FEATURE_INDEX["interest_rate"]
_ABSOLUTE_RATE_DELTA = 0.005


class InterestRateMonotonicityLoss(BaseLoss):
    """Enforce dV/dr <= 0 using finite-difference output comparison."""

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del context

        features = batch["features"]

        if "interest_std" not in batch:
            raise KeyError(
                "InterestRateMonotonicityLoss requires batch['interest_std']. "
                "ReserveDataset.__getitem__ must return interest_std."
            )

        interest_std = batch["interest_std"].to(
            device=features.device,
            dtype=features.dtype,
        )

        delta_norm = _ABSOLUTE_RATE_DELTA / interest_std.clamp_min(1e-8)

        features_high = features.clone()
        features_high[:, _R_IDX] = features_high[:, _R_IDX] + delta_norm.squeeze(-1)

        pred_high = model(features_high)

        violation = pred_high - predictions

        return self.reduce(torch.relu(violation).pow(2))