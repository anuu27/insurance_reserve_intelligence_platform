"""Stress testing visualization utilities.

Created: 2026-05-31
Purpose: Visualize before-and-after reserve impacts from stress scenarios.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_stress_comparison(frame: pd.DataFrame, output_path: str) -> None:
    """Plot before/after reserve bars for a stress scenario.

    Args:
        frame: Data frame containing ``before_reserve`` and ``after_reserve`` columns.
        output_path: Destination path for the plot image.

    Business Interpretation:
        This chart provides a compact management view of how much a scenario
        changes reserve levels.
    """

    summary = frame[["before_reserve", "after_reserve"]].mean()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(summary.index, summary.values, color=["#2f4b7c", "#f95d6a"])
    axis.set_title("Stress Test Reserve Comparison")
    axis.set_ylabel("Average reserve")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
