"""Reserve plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_reserve_trajectory(frame: pd.DataFrame, output_path: str) -> None:
    """Plot reserve trajectory over time."""

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["time"], frame["reserve"], color="#004c6d", linewidth=2.0)
    axis.set_title("Reserve Trajectory")
    axis.set_xlabel("Time")
    axis.set_ylabel("Reserve")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
