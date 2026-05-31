"""Checkpoint persistence helpers.

Created: 2026-05-31
Purpose: Persist and restore model-training state for long-running reserve experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Persist and restore training state.

    Business Interpretation:
        This protects long-running actuarial ML experiments from interruption and
        enables reproducible model handoff.
    """

    def __init__(self, directory: str) -> None:
        """Initialize the checkpoint manager.

        Args:
            directory: Directory where checkpoints will be stored.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, state: dict[str, Any]) -> Path:
        """Save a training checkpoint and return the resulting path.

        Args:
            filename: Checkpoint filename.
            state: Serializable training state.

        Returns:
            Path: Full checkpoint path.
        """

        path = self.directory / filename
        torch.save(state, path)
        return path

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
        """Load a previously saved checkpoint.

        Args:
            path: Checkpoint path.
            map_location: Optional device remapping target.

        Returns:
            dict[str, Any]: Restored checkpoint state.
        """

        return torch.load(path, map_location=map_location, weights_only=False)
