"""Data loss using Huber."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.losses.base_loss import BaseLoss

class DataLoss(BaseLoss):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, model, batch, predictions, context):
        del model, context
        targets = self.require_batch_tensor(batch, "target")
        return self.reduce(torch.nn.functional.huber_loss(
            predictions, targets, reduction="none", delta=self.delta))