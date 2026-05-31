"""Mortality data source abstractions and loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from insurance_reserve_intelligence_platform.actuarial.policy import MortalityProfile


class MortalityDataSource(ABC):
    """Abstract mortality data source."""

    @abstractmethod
    def load(self, age: int, term: int) -> MortalityProfile:
        """Load a mortality profile for the requested age and term."""


class _BaseFrameMortalityLoader(MortalityDataSource):
    """Base class for mortality tables represented as data frames."""

    def __init__(self, csv_path: str | Path | None = None, default_rate: float = 0.0005) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.default_rate = default_rate

    def _read_frame(self) -> pd.DataFrame:
        if self.csv_path is None:
            raise FileNotFoundError("No CSV path provided for offline mortality loading.")
        return pd.read_csv(self.csv_path)

    def _from_frame(self, frame: pd.DataFrame, age: int, term: int, source: str) -> MortalityProfile:
        age_column = "age" if "age" in frame.columns else frame.columns[0]
        rate_column = "mortality_rate" if "mortality_rate" in frame.columns else frame.columns[-1]
        filtered = frame[(frame[age_column] >= age) & (frame[age_column] <= age + term)].copy()
        if filtered.empty:
            ages = np.arange(age, age + term + 1, dtype=float)
            rates = self.default_rate * np.power(1.08, ages - age)
        else:
            ages = filtered[age_column].to_numpy(dtype=float)
            rates = filtered[rate_column].to_numpy(dtype=float)
        times = ages - float(age)
        return MortalityProfile(times=times, intensities=rates, source=source)


class HumanMortalityDatabaseLoader(_BaseFrameMortalityLoader):
    """Offline-compatible loader for Human Mortality Database extracts."""

    def load(self, age: int, term: int) -> MortalityProfile:
        frame = self._read_frame()
        return self._from_frame(frame, age=age, term=term, source="human_mortality_database")


class WHOMortalityLoader(_BaseFrameMortalityLoader):
    """Offline-compatible loader for WHO mortality extracts."""

    def load(self, age: int, term: int) -> MortalityProfile:
        frame = self._read_frame()
        return self._from_frame(frame, age=age, term=term, source="who")


class CSVMortalityLoader(_BaseFrameMortalityLoader):
    """Generic CSV mortality loader."""

    def load(self, age: int, term: int) -> MortalityProfile:
        frame = self._read_frame()
        return self._from_frame(frame, age=age, term=term, source="csv")
