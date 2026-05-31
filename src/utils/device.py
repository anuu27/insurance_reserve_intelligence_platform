"""Device management for PyTorch execution.

Created: 2026-05-31
Purpose: Resolve execution hardware and mixed-precision behavior for model workloads.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(slots=True)
class DeviceContext:
    """Runtime device metadata.

    Attributes:
        device: Selected execution device.
        mixed_precision: Whether mixed precision is enabled.
    """

    device: torch.device
    mixed_precision: bool


class DeviceManager:
    """Resolve the best available execution device.

    Business Interpretation:
        This keeps training and inference portable across developer laptops and
        accelerated hardware without changing business logic.
    """

    def __init__(self, prefer_mixed_precision: bool = True) -> None:
        """Initialize the device manager.

        Args:
            prefer_mixed_precision: Whether to enable mixed precision on CUDA.
        """
        self.context = DeviceContext(
            device=self._resolve_device(),
            mixed_precision=prefer_mixed_precision and torch.cuda.is_available(),
        )

    @staticmethod
    def _resolve_device() -> torch.device:
        """Resolve the preferred execution device.

        Returns:
            torch.device: CUDA, MPS, or CPU device in priority order.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Return the selected execution device.

        Returns:
            torch.device: Selected runtime device.
        """
        return self.context.device

    @property
    def mixed_precision(self) -> bool:
        """Return whether mixed precision is enabled.

        Returns:
            bool: Mixed-precision flag.
        """
        return self.context.mixed_precision

    def autocast_context(self) -> ContextManager[object]:
        """Return an autocast context when CUDA mixed precision is enabled."""

        if self.context.mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the configured device.

        Args:
            tensor: Tensor to move.

        Returns:
            torch.Tensor: Device-placed tensor.
        """

        return tensor.to(self.context.device)
