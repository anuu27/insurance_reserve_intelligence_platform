"""Supervised reserve fitting loss."""

from __future__ import annotations

import torch
from torch import nn


class DataLoss(nn.Module):
    """Mean squared error against classical reserve targets."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)
