import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.figure import Figure


def create_timestamped_dir(base_dir: str = ".", prefix: str = "run") -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_1d_plot(
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    hline: tuple[float, str] | None = None,
    grid: bool = False,
    dpi: int = 150,
) -> None:
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(x, y, color="blue", label="Current Estimate")

    if hline is not None:
        y_val, label = hline
        ax.axhline(y=y_val, color="red", linestyle="--", label=label)
        ax.legend()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(filepath, dpi=dpi)
