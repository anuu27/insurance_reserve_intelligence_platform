"""Closed-form helper expressions around the reserve PDE."""

from __future__ import annotations


def thiele_residual(dv_dt: float, reserve: float, interest_rate: float, premium: float, mortality: float, sum_assured: float) -> float:
    """Evaluate the scalar Thiele residual."""

    return dv_dt - interest_rate * reserve - premium + mortality * (sum_assured - reserve)
