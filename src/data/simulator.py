"""Policy simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.actuarial.policy import MortalityProfile, Policy
from src.data.mortality_loader import MortalityDataSource


@dataclass(slots=True)
class ScenarioDefinition:
    """Scenario overrides for synthetic policy generation."""

    interest_rate_shift: float = 0.0
    mortality_multiplier: float = 1.0
    premium_multiplier: float = 1.0
    sum_assured_multiplier: float = 1.0


class PolicySimulator:
    """Generate synthetic term-life policies."""

    def __init__(
        self,
        age_range: tuple[int, int],
        term_range: tuple[int, int],
        premium_range: tuple[float, float],
        interest_rate_range: tuple[float, float],
        sum_assured_range: tuple[float, float],
        mortality_source: MortalityDataSource | None = None,
        mortality_scale: float = 0.0005,
        mortality_shape: float = 1.08,
        seed: int = 42,
    ) -> None:
        self.age_range = age_range
        self.term_range = term_range
        self.premium_range = premium_range
        self.interest_rate_range = interest_rate_range
        self.sum_assured_range = sum_assured_range
        self.mortality_source = mortality_source
        self.mortality_scale = mortality_scale
        self.mortality_shape = mortality_shape
        self.rng = np.random.default_rng(seed)

    def _synthetic_mortality_profile(self, age: int, term: int, multiplier: float = 1.0) -> MortalityProfile:
        times = np.linspace(0.0, float(term), term + 1, dtype=float)
        ages = age + times
        rates = self.mortality_scale * np.power(self.mortality_shape, ages - age) * multiplier
        return MortalityProfile(times=times, intensities=rates, source="synthetic")

    def _mortality_profile(self, age: int, term: int, multiplier: float = 1.0) -> MortalityProfile:
        if self.mortality_source is not None:
            profile = self.mortality_source.load(age=age, term=term)
            return MortalityProfile(
                times=profile.times,
                intensities=profile.intensities * multiplier,
                source=profile.source,
            )
        return self._synthetic_mortality_profile(age=age, term=term, multiplier=multiplier)

    def _build_policy(
        self,
        policy_id: str,
        age: int,
        term: int,
        premium: float,
        interest_rate: float,
        sum_assured: float,
        mortality_multiplier: float = 1.0,
    ) -> Policy:
        return Policy(
            policy_id=policy_id,
            age=age,
            term=term,
            premium=premium,
            interest_rate=interest_rate,
            sum_assured=sum_assured,
            mortality_profile=self._mortality_profile(age=age, term=term, multiplier=mortality_multiplier),
        )

    def generate_random_policies(self, count: int) -> list[Policy]:
        """Generate randomly sampled policies."""

        policies: list[Policy] = []
        for index in range(count):
            age = int(self.rng.integers(self.age_range[0], self.age_range[1] + 1))
            term = int(self.rng.integers(self.term_range[0], self.term_range[1] + 1))
            premium = float(self.rng.uniform(*self.premium_range))
            interest_rate = float(self.rng.uniform(*self.interest_rate_range))
            sum_assured = float(self.rng.uniform(*self.sum_assured_range))
            policies.append(self._build_policy(f"policy_{index:05d}", age, term, premium, interest_rate, sum_assured))
        return policies

    def generate_stratified_policies(self, counts_by_age_band: dict[tuple[int, int], int]) -> list[Policy]:
        """Generate policies by age strata."""

        policies: list[Policy] = []
        start_index = 0
        for (age_min, age_max), count in counts_by_age_band.items():
            for offset in range(count):
                age = int(self.rng.integers(age_min, age_max + 1))
                term = int(self.rng.integers(self.term_range[0], self.term_range[1] + 1))
                premium = float(self.rng.uniform(*self.premium_range))
                interest_rate = float(self.rng.uniform(*self.interest_rate_range))
                sum_assured = float(self.rng.uniform(*self.sum_assured_range))
                policies.append(
                    self._build_policy(
                        f"stratified_{start_index + offset:05d}",
                        age,
                        term,
                        premium,
                        interest_rate,
                        sum_assured,
                    )
                )
            start_index += count
        return policies

    def generate_scenario_policies(self, base_policies: Iterable[Policy], scenario: ScenarioDefinition) -> list[Policy]:
        """Clone policies under a user-specified scenario."""

        stressed: list[Policy] = []
        for policy in base_policies:
            stressed.append(
                self._build_policy(
                    policy_id=f"{policy.policy_id}_scenario",
                    age=policy.age,
                    term=policy.term,
                    premium=policy.premium * scenario.premium_multiplier,
                    interest_rate=policy.interest_rate + scenario.interest_rate_shift,
                    sum_assured=policy.sum_assured * scenario.sum_assured_multiplier,
                    mortality_multiplier=scenario.mortality_multiplier,
                )
            )
        return stressed
