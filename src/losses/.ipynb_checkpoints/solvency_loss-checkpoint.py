"""Solvency floor — V >= 0  →  z >= -μ_V/σ_V."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.losses.base_loss import BaseLoss


class SolvencyLoss(BaseLoss):
    def forward(self, model, batch, predictions, context):
        del model, context
        t_mean = batch["target_mean"].to(predictions.device)
        t_std  = batch["target_std"].to(predictions.device)
        z_floor = -t_mean / t_std
        return self.reduce(torch.relu(z_floor - predictions))