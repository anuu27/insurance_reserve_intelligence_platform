"""Surface plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_path: str, title: str) -> None:
    """Plot a generic 3D surface."""

    fig = plt.figure(figsize=(9, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot_surface(x, y, z, cmap="viridis", edgecolor="none", alpha=0.9)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Reserve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
