"""Sensitivity visualization utilities.

Created: 2026-05-31
Purpose: Visualize reserve sensitivities for driver-attribution analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_sensitivities(frame: pd.DataFrame, output_path: str) -> None:
    """Plot mean sensitivities as a bar chart.

    Args:
        frame: Data frame containing sensitivity columns.
        output_path: Destination path for the plot image.

    Business Interpretation:
        This view helps non-technical stakeholders see which assumptions move
        reserve most strongly on average.
    """

    summary = frame.mean(numeric_only=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=axis, color=["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"])
    axis.set_title("Reserve Sensitivities")
    axis.set_ylabel("Average sensitivity")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
