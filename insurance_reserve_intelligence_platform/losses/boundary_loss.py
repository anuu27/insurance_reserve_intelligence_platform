"""Boundary-condition loss."""

from __future__ import annotations

import torch
from torch import nn


class BoundaryLoss(nn.Module):
    """Enforce V(T)=0 for term-life contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features: torch.Tensor, terms: torch.Tensor, model: nn.Module) -> torch.Tensor:
        boundary_features = features.clone()
        boundary_features[:, 0:1] = terms
        boundary_predictions = model(boundary_features)
        return self.criterion(boundary_predictions, torch.zeros_like(boundary_predictions))
