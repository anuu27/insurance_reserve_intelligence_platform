"""Stress testing visualization utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_stress_comparison(frame: pd.DataFrame, output_path: str) -> None:
    """Plot before/after reserve bars for a stress scenario."""

    summary = frame[["before_reserve", "after_reserve"]].mean()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(summary.index, summary.values, color=["#2f4b7c", "#f95d6a"])
    axis.set_title("Stress Test Reserve Comparison")
    axis.set_ylabel("Average reserve")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
