"""Plots: convergence of the DE variants, jDE parameter adaptation and the F x CR study."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from problems import Problem  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3

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


def _draw_convergence(ax: plt.Axes, curves: Curves) -> None:
    for color, (label, histories) in zip(SERIES, curves.items()):
        mean = np.maximum(histories.mean(axis=0), FLOOR)
        ax.plot(np.arange(1, histories.shape[1] + 1), mean, color=color, linewidth=2, label=label)
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(1, max(h.shape[1] for h in curves.values()))


def save_convergence_plot(curves: Curves, problem: Problem, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    _draw_convergence(ax, curves)
    ax.set_title(f"{problem.title}, D = {problem.dim}", loc="left", pad=26)
    ax.text(
        0, 1.02, f"čára = průměr {len(next(iter(curves.values())))} běhů, minimalizace",
        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom",
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save(fig, path)


def save_overview_plot(entries: Sequence[tuple[Problem, Curves]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(entries), figsize=(15, 4.8))
    axes = np.atleast_1d(axes)
    for ax, (problem, curves) in zip(axes, entries):
        _draw_convergence(ax, curves)
        ax.set_title(f"{problem.title}, D = {problem.dim}", loc="left")
        ax.set_xlabel(X_LABEL)
    axes[0].set_ylabel(Y_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    runs = len(next(iter(entries[0][1].values())))
    fig.suptitle(f"Porovnání verzí DE (čára = průměr {runs} běhů)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 1.0))
    _save(fig, path)


def save_adaptation_plot(
    entries: Sequence[tuple[Problem, np.ndarray, np.ndarray]], path: Path
) -> None:
    """How jDE moves the mean F and CR of the population during the run."""
    fig, axes = plt.subplots(1, len(entries), figsize=(15, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (problem, mean_f, mean_cr) in zip(axes, entries):
        generations = np.arange(1, mean_f.shape[1] + 1)
        for color, label, values in ((SERIES[0], "F", mean_f), (SERIES[1], "CR", mean_cr)):
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1)
            ax.fill_between(generations, mean - spread, mean + spread, color=color, alpha=0.18, linewidth=0)
            ax.plot(generations, mean, color=color, linewidth=2, label=label)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{problem.title}, D = {problem.dim}", loc="left")
        ax.set_xlabel("Generace")
    axes[0].set_ylabel("Průměrná hodnota v populaci")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Samoadaptace parametrů v jDE (čára = průměr běhů, pásmo = ±1 sm. odchylka)",
        x=0.01, ha="left", color=INK, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    _save(fig, path)


def save_parameter_study_plot(
    panels: Sequence[tuple[str, str, np.ndarray]],
    f_values: Sequence[float],
    cr_values: Sequence[float],
    n_rows: int,
    path: Path,
) -> None:
    """Heat maps of the mean final value over the F x CR grid (darker = worse)."""
    n_cols = len(panels) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows))
    for ax, (title, subtitle, matrix) in zip(np.ravel(axes), panels):
        norm = LogNorm(vmin=max(matrix.min(), 1e-12), vmax=matrix.max())
        ax.imshow(matrix, cmap="Blues", norm=norm, aspect="auto")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col, row, f"{value:.3g}", ha="center", va="center", fontsize=7.5,
                color="#ffffff" if norm(value) > 0.55 else INK,
            )
        ax.set_xticks(range(len(cr_values)), [f"{cr:g}" for cr in cr_values])
        ax.set_yticks(range(len(f_values)), [f"{f:g}" for f in f_values])
        ax.set_title(f"{title} – {subtitle}", loc="left", fontsize=10)
        ax.set_xlabel("CR")
        ax.set_ylabel("F")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Vliv F a CR: průměrná nejlepší hodnota (tmavší = horší)",
        x=0.01, ha="left", color=INK, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
