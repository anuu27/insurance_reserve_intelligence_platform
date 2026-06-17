"""Datasets for reserve learning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy

FEATURE_INDEX = {
    "time": 0,
    "age": 1,
    "interest_rate": 2,
    "premium": 3,
    "sum_assured": 4,
    "mortality": 5,
}

FEATURE_SCALES = {
    "time":          30.0,
    "age":           45.0,
    "interest_rate": 0.07,
    "premium":       2_000.0,
    "sum_assured":   950_000.0,
    "mortality":     0.05,
}


@dataclass(slots=True)
class ReserveRecord:
    policy_id: str
    features: np.ndarray
    reserve: float
    term: float


class ReserveDataset(Dataset):
    def __init__(self, policies, solver, time_steps,
                 target_mean=None, target_std=None):
        self.records = self._build_records(list(policies), solver=solver, time_steps=time_steps)

        v_values = np.array([
            r.reserve / max(r.features[FEATURE_INDEX["sum_assured"]], 1.0)
            for r in self.records
        ], dtype=np.float32)

        if target_mean is None:
            self.target_mean = float(v_values.mean())
        else:
            self.target_mean = float(target_mean)

        if target_std is None:
            std = float(v_values.std())
            self.target_std = std if std > 1e-8 else 1.0
        else:
            self.target_std = float(target_std)

    @staticmethod
    def _build_records(policies, solver, time_steps):
        records = []
        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)
            for time_point, reserve in zip(trajectory.times, trajectory.reserves):
                mortality = policy.mortality_profile.intensity_at(time_point)
                features = np.asarray([
                    time_point,
                    float(policy.age),
                    policy.interest_rate,
                    policy.premium,
                    policy.sum_assured,
                    mortality,
                ], dtype=np.float32)
                records.append(ReserveRecord(
                    policy_id=policy.policy_id,
                    features=features,
                    reserve=float(reserve),
                    term=float(policy.term),
                ))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw_features = torch.tensor(record.features, dtype=torch.float32)
        features = raw_features.clone()
        features[FEATURE_INDEX["time"]]          /= FEATURE_SCALES["time"]
        features[FEATURE_INDEX["age"]]           /= FEATURE_SCALES["age"]
        features[FEATURE_INDEX["interest_rate"]] /= FEATURE_SCALES["interest_rate"]
        features[FEATURE_INDEX["premium"]]       /= FEATURE_SCALES["premium"]
        features[FEATURE_INDEX["sum_assured"]]   /= FEATURE_SCALES["sum_assured"]
        features[FEATURE_INDEX["mortality"]]     /= FEATURE_SCALES["mortality"]

        sum_assured_val = float(raw_features[FEATURE_INDEX["sum_assured"]].item())
        scale = max(sum_assured_val, 1.0)
        v = record.reserve / scale
        z = (v - self.target_mean) / self.target_std

        return {
            "features":        features,
            "raw_features":    raw_features,
            "target":          torch.tensor([z], dtype=torch.float32),
            "sum_assured_scale": torch.tensor([scale], dtype=torch.float32),
            "target_mean":     torch.tensor([self.target_mean], dtype=torch.float32),
            "target_std":      torch.tensor([self.target_std], dtype=torch.float32),
            "term":            torch.tensor([record.term], dtype=torch.float32),
        }