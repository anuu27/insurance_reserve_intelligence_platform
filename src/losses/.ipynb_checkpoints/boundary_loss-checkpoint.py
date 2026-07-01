"""Boundary condition loss — V(T) = 0 for term life."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss


class BoundaryLoss(BaseLoss):
    """Enforce V(T) = 0  →  z_T = (0 - μ_V) / σ_V = -μ_V/σ_V."""

    def forward(self, model, batch, predictions, context):
        features = self.require_batch_tensor(batch, "features")
        terms    = self.require_batch_tensor(batch, "term")
        t_mean   = batch["target_mean"].to(features.device)
        t_std    = batch["target_std"].to(features.device)

        boundary_features = features.clone()
        boundary_features[:, FEATURE_INDEX["time"]:FEATURE_INDEX["time"]+1] = (
            terms / FEATURE_SCALES["time"]
        )
        boundary_z = model(boundary_features)
        z_target   = -t_mean / t_std          # z such that V = 0
        context["boundary_predictions"] = boundary_z
        return self.reduce((boundary_z - z_target).pow(2))