"""PINN theory helpers."""

from __future__ import annotations


PINN_CONCEPTS: dict[str, str] = {
    "collocation_points": "Sampled points where the PDE residual is penalized.",
    "autodiff": "Automatic differentiation used to compute derivatives such as dV/dt.",
    "boundary_condition": "Constraint enforcing terminal reserve equal to zero for term life.",
}
