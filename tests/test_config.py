"""Configuration tests.

Created: 2026-05-31
Purpose: Validate typed configuration loading behavior.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import ConfigLoader


def test_config_loader_reads_yaml() -> None:
    """Verify that the YAML configuration loads into the typed config object."""
    config = ConfigLoader.load(Path("configs/config.yaml"))
    assert config.project_name == "src"
    assert config.model.input_dim == 6
