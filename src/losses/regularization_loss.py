"""Regularization loss."""

from __future__ import annotations

import torch
from torch import nn


class RegularizationLoss(nn.Module):
    """Compute L2 weight regularization."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model: nn.Module) -> torch.Tensor:
        parameters = list(model.parameters())
        penalties = [parameter.pow(2.0).sum() for parameter in parameters]
        if not penalties:
            return torch.tensor(0.0)
        return torch.stack(penalties).sum()
