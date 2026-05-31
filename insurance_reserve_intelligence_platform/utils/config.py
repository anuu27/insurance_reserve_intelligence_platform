"""Configuration management utilities."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, cast, get_type_hints

import yaml


T = TypeVar("T")


def _coerce_dataclass(cls: Type[T], payload: Dict[str, Any]) -> T:
    """Recursively coerce a dictionary into a dataclass instance."""

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for item in fields(cls):
        if item.name in payload:
            value = payload[item.name]
        elif item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:
            value = item.default_factory()
        else:
            value = None

        hint = type_hints.get(item.name)
        if hint and is_dataclass(hint) and isinstance(value, dict):
            kwargs[item.name] = _coerce_dataclass(cast(Type[Any], hint), value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


@dataclass(slots=True)
class PathConfig:
    """File-system configuration."""

    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "artifacts/checkpoints"
    logs_dir: str = "artifacts/logs"
    reports_dir: str = "artifacts/reports"
    plots_dir: str = "artifacts/plots"
    tensorboard_dir: str = "artifacts/tensorboard"
    data_dir: str = "data"


@dataclass(slots=True)
class DataConfig:
    """Synthetic and tabular data generation configuration."""

    train_size: int = 512
    validation_size: int = 128
    test_size: int = 128
    time_steps: int = 40
    batch_size: int = 64
    num_workers: int = 0
    age_min: int = 25
    age_max: int = 70
    term_min: int = 5
    term_max: int = 30
    premium_min: float = 500.0
    premium_max: float = 5000.0
    interest_rate_min: float = 0.01
    interest_rate_max: float = 0.08
    sum_assured_min: float = 50_000.0
    sum_assured_max: float = 1_000_000.0
    mortality_scale: float = 0.0005
    mortality_shape: float = 1.08
    random_seed: int = 42


@dataclass(slots=True)
class SolverConfig:
    """Actuarial solver configuration."""

    method: str = "solve_ivp"
    integration_step: float = 0.25
    rtol: float = 1e-6
    atol: float = 1e-8


@dataclass(slots=True)
class ModelConfig:
    """PINN model configuration."""

    input_dim: int = 6
    hidden_dim: int = 128
    num_layers: int = 4
    activation: str = "tanh"
    dropout: float = 0.1


@dataclass(slots=True)
class LossConfig:
    """Loss term weights and regularization settings."""

    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_boundary: float = 1.0
    lambda_reg: float = 1e-5


@dataclass(slots=True)
class TrainerConfig:
    """Training loop configuration."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    checkpoint_every: int = 5
    resume_from: str | None = None
    mixed_precision: bool = True


@dataclass(slots=True)
class StressScenarioConfig:
    """Default shock amplitudes used by the stress tester."""

    mortality_shock: float = 0.15
    interest_rate_shock: float = -0.01
    inflation_shock: float = 0.05
    longevity_shock: float = -0.10
    lapse_shock: float = 0.10


@dataclass(slots=True)
class OptimizationConfig:
    """Optimization defaults."""

    learning_rate: float = 0.05
    steps: int = 150
    reserve_tolerance: float = 1e-4
    solvency_threshold: float = 0.0


@dataclass(slots=True)
class DigitalTwinConfig:
    """Digital twin simulation configuration."""

    forecast_horizon: int = 30
    scenario_steps: int = 12
    regime_names: list[str] = field(
        default_factory=lambda: ["base", "soft_recession", "inflationary", "mortality_crisis"]
    )


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level platform configuration."""

    project_name: str = "insurance_reserve_intelligence_platform"
    experiment_name: str = "actuary_twin_pinn"
    seed: int = 42
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    stress: StressScenarioConfig = field(default_factory=StressScenarioConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    digital_twin: DigitalTwinConfig = field(default_factory=DigitalTwinConfig)


class ConfigLoader:
    """Load YAML configuration into typed dataclasses."""

    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file."""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return _coerce_dataclass(ExperimentConfig, payload)


def ensure_directories(config: ExperimentConfig) -> None:
    """Create artifact directories required by the workflow."""

    for item in fields(config.paths):
        Path(getattr(config.paths, item.name)).mkdir(parents=True, exist_ok=True)
