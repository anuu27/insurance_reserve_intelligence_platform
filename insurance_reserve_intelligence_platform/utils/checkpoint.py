"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Persist and restore training state."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, state: dict[str, Any]) -> Path:
        """Save a training checkpoint and return the resulting path."""

        path = self.directory / filename
        torch.save(state, path)
        return path

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
        """Load a previously saved checkpoint."""

        return torch.load(path, map_location=map_location, weights_only=False)
