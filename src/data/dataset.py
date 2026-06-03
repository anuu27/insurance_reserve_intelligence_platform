"""Datasets for reserve learning.

Created: 2026-05-31
Purpose: Build training records by pairing policies with classical reserve trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy


@dataclass(slots=True)
class ReserveRecord:
    """Single collocation record used for training or evaluation.

    Attributes:
        policy_id: Policy identifier associated with the record.
        features: Feature vector used by the PINN.
        reserve: Classical reserve target value.
        term: Policy maturity used for the boundary condition.
    """

    policy_id: str
    features: np.ndarray
    reserve: float
    term: float


class ReserveDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of reserve trajectories derived from classical actuarial solutions."""

    def __init__(self, policies: Iterable[Policy], solver: BaseActuarialSolver, time_steps: int) -> None:
        """Initialize the dataset.

        Args:
            policies: Policies to expand into reserve records.
            solver: Classical solver used to generate target trajectories.
            time_steps: Number of time steps per policy.
        """
        self.records = self._build_records(list(policies), solver=solver, time_steps=time_steps)

    @staticmethod
    def _build_records(policies: list[Policy], solver: BaseActuarialSolver, time_steps: int) -> list[ReserveRecord]:
        """Convert policies into reserve-learning records.

        Args:
            policies: Policies to transform.
            solver: Classical solver used to generate reserve trajectories.
            time_steps: Number of time steps per policy.

        Returns:
            list[ReserveRecord]: Flattened learning records.
        """
        records: list[ReserveRecord] = []
        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)
            for time_point, reserve in zip(trajectory.times, trajectory.reserves):
                mortality = policy.mortality_profile.intensity_at(time_point)
                features = np.asarray(
                    [time_point, float(policy.age), policy.interest_rate, policy.premium, policy.sum_assured, mortality],
                    dtype=np.float32,
                )
                records.append(
                    ReserveRecord(
                        policy_id=policy.policy_id,
                        features=features,
                        reserve=float(reserve),
                        term=float(policy.term),
                    )
                )
        return records

    def __len__(self) -> int:
        """Return the number of records in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retrieve one record as tensors.

        Args:
            index: Record index to fetch.

        Returns:
            dict[str, torch.Tensor]: Feature, target, and term tensors.
        """
        record = self.records[index]
        features = torch.tensor(record.features, dtype=torch.float32)

                # Feature normalization
        features[0] /= 30.0        # time
        features[1] /= 100.0       # age
        features[2] /= 0.1         # interest rate
        features[3] /= 10000.0     # premium
        features[4] /= 1000000.0   # sum assured
        features[5] /= 0.05        # mortality
        return {
            "features": features,
            "target": torch.tensor([record.reserve], dtype=torch.float32),
            "term": torch.tensor([record.term], dtype=torch.float32),
        }
