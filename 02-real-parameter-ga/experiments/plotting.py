"""Convergence plots: mean best-so-far value of every variant on a logarithmic axis."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from problems import Problem  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5

X_LABEL = "Počet ohodnocení účelové funkce"
Y_LABEL = "Nejlepší nalezená hodnota (log)"
FLOOR = 1e-300  # only guards the logarithmic axis against an exact zero

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

Curves = dict[str, np.ndarray]  # variant label -> (runs, budget) histories


def _draw(ax: plt.Axes, curves: Curves) -> None:
    for color, (label, histories) in zip(SERIES, curves.items()):
        mean = np.maximum(histories.mean(axis=0), FLOOR)
        ax.plot(np.arange(1, histories.shape[1] + 1), mean, color=color, linewidth=2, label=label)
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(1, max(len(h[0]) for h in curves.values()))


def save_convergence_plot(curves: Curves, problem: Problem, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    _draw(ax, curves)
    ax.set_title(f"{problem.title}, D = {problem.dim}", loc="left", pad=26)
    ax.text(
        0, 1.02, f"čára = průměr {len(next(iter(curves.values())))} běhů, minimalizace",
        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom",
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    # Outside the axes: with 80 decades on the Sphere panel there is no free corner.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save(fig, path)


def save_overview_plot(entries: Sequence[tuple[Problem, Curves]], path: Path) -> None:
    """All functions side by side, one panel each, shared legend."""
    fig, axes = plt.subplots(1, len(entries), figsize=(15, 4.8))
    for ax, (problem, curves) in zip(np.atleast_1d(axes), entries):
        _draw(ax, curves)
        ax.set_title(f"{problem.title}, D = {problem.dim}", loc="left")
        ax.set_xlabel(X_LABEL)
    np.atleast_1d(axes)[0].set_ylabel(Y_LABEL)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    runs = len(next(iter(entries[0][1].values())))
    fig.suptitle(f"Porovnání variant GA (čára = průměr {runs} běhů)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 1.0))
    _save(fig, path)


def save_parameter_plot(
    series: dict[str, tuple[np.ndarray, np.ndarray]], title: str, x_label: str, path: Path
) -> None:
    """Effect of one control parameter: x values vs. mean final value, one line per function."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for color, (label, (x, y)) in zip(SERIES, series.items()):
        ax.plot(x, np.maximum(y, FLOOR), color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Průměrná nejlepší hodnota (log)")
    ax.legend(loc="best")
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
