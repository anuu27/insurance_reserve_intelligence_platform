"""PDE residual loss in v=V/S z-space."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.losses.base_loss import BaseLoss

class PDEResidualLoss(BaseLoss):

    def forward(self, model, batch, predictions, context):

        del model

        dz_dt = self.first_derivative(
            predictions=predictions,
            batch=batch,
            name="time",
        )

        r = self.raw_feature(batch, "interest_rate")

        premium_ratio = self.raw_feature(
            batch,
            "premium",
        )
        S = self.raw_feature(batch, "sum_assured").clamp(min=1.0)

        mu = self.raw_feature(batch, "mortality")

        target_mean = (
            batch["target_mean"]
            .detach()
            .to(predictions.device)
        )

        target_std = (
            batch["target_std"]
            .detach()
            .to(predictions.device)
        )

        # Convert network output z back to v = V/S
        v = predictions * target_std + target_mean

        # Expected dv/dt from Thiele equation
        dv_dt = (
            r * v
            + premium_ratio
            - mu * (1.0 - v)
        )

        # Convert expected dv/dt to dz/dt
        dz_dt_expected = dv_dt / target_std

        residual = dz_dt - dz_dt_expected

        context["pde_residual"] = residual

        return self.reduce(residual.pow(2))