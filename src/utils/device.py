"""Device management for PyTorch execution."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(slots=True)
class DeviceContext:
    """Runtime device metadata."""

    device: torch.device
    mixed_precision: bool


class DeviceManager:
    """Resolve the best available execution device."""

    def __init__(self, prefer_mixed_precision: bool = True) -> None:
        self.context = DeviceContext(
            device=self._resolve_device(),
            mixed_precision=prefer_mixed_precision and torch.cuda.is_available(),
        )

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self.context.device

    @property
    def mixed_precision(self) -> bool:
        return self.context.mixed_precision

    def autocast_context(self) -> ContextManager[object]:
        """Return an autocast context when CUDA mixed precision is enabled."""

        if self.context.mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the configured device."""

        return tensor.to(self.context.device)
