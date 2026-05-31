"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

from insurance_reserve_intelligence_platform.utils.config import ConfigLoader


def test_config_loader_reads_yaml() -> None:
    config = ConfigLoader.load(Path("configs/config.yaml"))
    assert config.project_name == "insurance_reserve_intelligence_platform"
    assert config.model.input_dim == 6
