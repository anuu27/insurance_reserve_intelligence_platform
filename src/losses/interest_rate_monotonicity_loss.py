"""Interest-rate monotonicity loss.

Created: 2026-06-03  Revised: 2026-06-11
Purpose: Enforce dV/dr <= 0 via finite-difference perturbation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

_R_IDX = FEATURE_INDEX["interest_rate"]
# Perturbation in normalised space: 0.5% absolute rate change / scale
_DELTA = 0.005 / FEATURE_SCALES["interest_rate"]


class InterestRateMonotonicityLoss(BaseLoss):
    """Enforce dV/dr <= 0 using finite-difference output comparison.

    Why finite difference instead of autograd:
        In normalised z-space the gradient dz/d(r_norm) is numerically tiny
        (~0.4) relative to dz/d(P_norm) (~4.0), so autograd-based penalties
        provide a near-zero gradient signal to the weight update — the model
        never learns to correct the sign.

        Finite difference compares model(r) vs model(r + Δr) directly.
        The penalty magnitude is proportional to how much the model violates
        the constraint, giving a strong, unambiguous training signal.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del context
        features = batch["features"]

        # Perturb interest rate upward by Δr
        features_high = features.clone()
        features_high[:, _R_IDX] = features_high[:, _R_IDX] + _DELTA

        pred_high = model(features_high)

        # dV/dr > 0 is a violation — penalise relu(V(r+Δ) - V(r))
        # Strong violations (large positive difference) get large penalties
        violation = pred_high - predictions
        return self.reduce(torch.relu(violation) * 10.0)  # scale up for clear signal