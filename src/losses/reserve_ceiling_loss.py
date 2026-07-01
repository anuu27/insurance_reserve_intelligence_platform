"""Ceiling loss — V≤S → v≤1 → z≤(1-μ)/σ."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.losses.base_loss import BaseLoss

class ReserveCeilingLoss(BaseLoss):
    def forward(self, model, batch, predictions, context):
        del model, context
        t_mean = batch["target_mean"].to(predictions.device)
        t_std  = batch["target_std"].to(predictions.device)
        z_ceil = (1.0 - t_mean) / t_std
        return self.reduce(torch.relu(predictions - z_ceil))