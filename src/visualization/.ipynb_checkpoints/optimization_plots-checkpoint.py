"""Optimization plotting utilities.

Created: 2026-07-02
Purpose: Generate business optimization figures for reports, papers, and decks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "premium": "Premium",
    "sum_assured": "Sum Assured",
    "interest_rate": "Interest Rate",
    "reserve": "Reserve",
    "model_reserve": "Raw Model Reserve",
    "profit": "Profit",
    "objective": "Objective",
}

def plot_metric_sweep(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot one optimization metric against one decision variable."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame[x_column], frame[y_column], color="#004c6d", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel(_label(x_column))
    axis.set_ylabel(_label(y_column))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_convergence(
    history: pd.DataFrame,
    y_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot an optimizer convergence trace."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if history.empty or y_column not in history:
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["iteration"], history[y_column], color="#7a3e00", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel(_label(y_column))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_standard_convergence_set(
    history: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    """Create objective, premium, reserve, and profit convergence charts."""

    output = Path(output_dir)
    created: list[Path] = []
    for column, title in (
        ("objective", "Optimization Convergence"),
        ("premium", "Premium Convergence"),
        ("reserve", "Reserve Convergence"),
        ("profit", "Profit Convergence"),
    ):
        path = output / f"{prefix}_{column}_convergence.png"
        plot_convergence(history, column, path, title)
        if path.exists():
            created.append(path)
    return created


def _label(column: str) -> str:
    """Return a human-readable axis label."""

    return column.replace("_", " ").title()
