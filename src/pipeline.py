"""End-to-end assembly helpers for scripts and tests.

Created: 2026-05-31
Purpose: Assemble the main research pipeline from config, simulation, solver, and model components.
"""

from __future__ import annotations

from pathlib import Path

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.dataloader import create_dataloader
from src.data.dataset import ReserveDataset
from src.data.mortality_loader import CSVMortalityLoader, MortalityDataSource
from src.data.simulator import PolicySimulator
from src.models.factory import ModelFactory
from src.utils.config import ExperimentConfig


def build_mortality_source(config: ExperimentConfig) -> MortalityDataSource | None:
    """Build an offline mortality source when a sample CSV is available.

    Args:
        config: Experiment configuration.

    Returns:
        MortalityDataSource | None: Offline mortality loader when data is present.

    Business Interpretation:
        This keeps the platform usable in offline or air-gapped actuarial research
        settings where external APIs are not available.
    """

    candidate = Path(config.paths.data_dir) / "sample_mortality.csv"
    if candidate.exists():
        return CSVMortalityLoader(candidate)
    return None


def build_simulator(config: ExperimentConfig) -> PolicySimulator:
    """Construct a policy simulator from configuration.

    Args:
        config: Experiment configuration.

    Returns:
        PolicySimulator: Configured policy simulator.

    Business Interpretation:
        This defines the synthetic portfolio generator used for model development
        and scenario experimentation.
    """

    return PolicySimulator(
        age_range=(config.data.age_min, config.data.age_max),
        term_range=(config.data.term_min, config.data.term_max),
        interest_rate_range=(config.data.interest_rate_min, config.data.interest_rate_max),
        sum_assured_range=(config.data.sum_assured_min, config.data.sum_assured_max),
        mortality_source=build_mortality_source(config),
        mortality_scale=config.data.mortality_scale,
        mortality_shape=config.data.mortality_shape,
        mortality_reference_age=config.data.mortality_reference_age,
        premium_loading=config.data.premium_loading,
        max_expiry_age=config.data.max_expiry_age,
        sum_assured_rounding=config.data.sum_assured_rounding,
        sum_assured_age_decay=config.data.sum_assured_age_decay,
        seed=config.data.random_seed,
    )


def build_solver(config: ExperimentConfig) -> ThieleSolver:
    """Construct the classical actuarial solver.

    Args:
        config: Experiment configuration.

    Returns:
        ThieleSolver: Configured actuarial solver.

    Business Interpretation:
        This wires the benchmark reserve engine used as the actuarial reference.
    """

    return ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )


def build_model(config: ExperimentConfig):
    """Construct the configured reserve model.

    Args:
        config: Experiment configuration.

    Returns:
        BaseReserveModel: Configured reserve model instance.

    Business Interpretation:
        This is the fast surrogate reserve engine used after training.
    """

    return ModelFactory.create_pinn(config.model)


def build_datasets(
    config: ExperimentConfig,
) -> tuple[ReserveDataset, ReserveDataset, ReserveDataset, list]:
    """Generate synthetic policies and datasets for training, validation, and testing.

    Args:
        config: Experiment configuration.

    Returns:
        tuple[ReserveDataset, ReserveDataset, ReserveDataset, list]: Train,
        validation, and test datasets plus held-out policies.

    Business Interpretation:
        This builds the controlled experimental population used to teach and
        validate the reserve surrogate.
    """

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
    """Create train, validation, and test dataloaders.

    Args:
        config: Experiment configuration.

    Returns:
        tuple: Train, validation, and test dataloaders plus test artifacts.

    Business Interpretation:
        This is the final assembly point before training or evaluation starts.
    """

    train_dataset, validation_dataset, test_dataset, test_policies = build_datasets(config)
    train_loader = create_dataloader(
        train_dataset,
        config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    test_loader = create_dataloader(
        test_dataset,
        config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, validation_loader, test_loader, test_dataset, test_policies
