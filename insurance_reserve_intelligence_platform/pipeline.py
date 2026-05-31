"""End-to-end assembly helpers for scripts and tests."""

from __future__ import annotations

from pathlib import Path

from insurance_reserve_intelligence_platform.actuarial.actuarial_solver import ThieleSolver
from insurance_reserve_intelligence_platform.data.dataloader import create_dataloader
from insurance_reserve_intelligence_platform.data.dataset import ReserveDataset
from insurance_reserve_intelligence_platform.data.mortality_loader import CSVMortalityLoader, MortalityDataSource
from insurance_reserve_intelligence_platform.data.simulator import PolicySimulator
from insurance_reserve_intelligence_platform.models.factory import ModelFactory
from insurance_reserve_intelligence_platform.utils.config import ExperimentConfig


def build_mortality_source(config: ExperimentConfig) -> MortalityDataSource | None:
    """Build an offline mortality source when a sample CSV is available."""

    candidate = Path(config.paths.data_dir) / "sample_mortality.csv"
    if candidate.exists():
        return CSVMortalityLoader(candidate)
    return None


def build_simulator(config: ExperimentConfig) -> PolicySimulator:
    """Construct a policy simulator from configuration."""

    return PolicySimulator(
        age_range=(config.data.age_min, config.data.age_max),
        term_range=(config.data.term_min, config.data.term_max),
        premium_range=(config.data.premium_min, config.data.premium_max),
        interest_rate_range=(config.data.interest_rate_min, config.data.interest_rate_max),
        sum_assured_range=(config.data.sum_assured_min, config.data.sum_assured_max),
        mortality_source=build_mortality_source(config),
        mortality_scale=config.data.mortality_scale,
        mortality_shape=config.data.mortality_shape,
        seed=config.data.random_seed,
    )


def build_solver(config: ExperimentConfig) -> ThieleSolver:
    """Construct the classical actuarial solver."""

    return ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )


def build_model(config: ExperimentConfig):
    """Construct the configured reserve model."""

    return ModelFactory.create_pinn(config.model)


def build_datasets(config: ExperimentConfig) -> tuple[ReserveDataset, ReserveDataset, ReserveDataset, list]:
    """Generate synthetic policies and datasets for training, validation, and testing."""

    simulator = build_simulator(config)
    solver = build_solver(config)
    train_policies = simulator.generate_random_policies(config.data.train_size)
    validation_policies = simulator.generate_random_policies(config.data.validation_size)
    test_policies = simulator.generate_random_policies(config.data.test_size)
    train_dataset = ReserveDataset(train_policies, solver, config.data.time_steps)
    validation_dataset = ReserveDataset(validation_policies, solver, config.data.time_steps)
    test_dataset = ReserveDataset(test_policies, solver, config.data.time_steps)
    return train_dataset, validation_dataset, test_dataset, test_policies


def build_dataloaders(config: ExperimentConfig):
    """Create train, validation, and test dataloaders."""

    train_dataset, validation_dataset, test_dataset, test_policies = build_datasets(config)
    train_loader = create_dataloader(train_dataset, config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)
    validation_loader = create_dataloader(validation_dataset, config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)
    test_loader = create_dataloader(test_dataset, config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)
    return train_loader, validation_loader, test_loader, test_dataset, test_policies
