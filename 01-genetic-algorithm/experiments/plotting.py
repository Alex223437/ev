"""Convergence plots (mean best-so-far fitness over repeated runs)."""

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
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 (blue, orange)

X_LABEL = "Počet ohodnocení účelové funkce"
Y_LABEL = "Nejlepší nalezená fitness"
BAND_NOTE = "čára = průměr {runs} běhů, pásmo = ±1 směrodatná odchylka"

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


def _style_axes(ax: plt.Axes, problem: Problem, budget: int) -> None:
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(1, budget)
    ax.set_ylim(0, problem.optimum * 1.06)
    ax.axhline(problem.optimum, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        f"optimum = {problem.optimum}",
        xy=(1, problem.optimum),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )


def _draw_convergence(ax: plt.Axes, histories: np.ndarray, optimum: float, color: str, label: str | None = None) -> None:
    x = np.arange(1, histories.shape[1] + 1)
    mean = histories.mean(axis=0)
    std = histories.std(axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(mean)
    ax.fill_between(x, np.maximum(mean - std, 0), np.minimum(mean + std, optimum), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2, label=label)


def _title(problem: Problem) -> str:
    return f"{problem.title}, D = {problem.dim}"


def save_convergence_plot(histories: np.ndarray, problem: Problem, config_label: str, path: Path) -> None:
    """One instance: mean convergence curve with a ±std band."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _draw_convergence(ax, histories, problem.optimum, SERIES[0])
    _style_axes(ax, problem, histories.shape[1])
    ax.set_title(_title(problem), loc="left", pad=36)
    ax.text(
        0, 1.02, f"{BAND_NOTE.format(runs=len(histories))}\n{config_label}",
        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom",
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    _save(fig, path)


def save_overview_plot(entries: Sequence[tuple[Problem, np.ndarray]], path: Path) -> None:
    """All instances as small multiples: one row per problem, one column per dimension."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (problem, histories) in zip(axes.flat, entries):
        _draw_convergence(ax, histories, problem.optimum, SERIES[0])
        _style_axes(ax, problem, histories.shape[1])
        ax.set_title(_title(problem), loc="left")
    _label_grid(fig, axes)
    fig.suptitle(
        f"Průměrná konvergence GA ({BAND_NOTE.format(runs=len(entries[0][1]))})",
        x=0.01, ha="left", color=INK, fontsize=12,
    )
    fig.tight_layout()
    _save(fig, path)


def save_comparison_plot(
    entries: Sequence[tuple[Problem, dict[str, np.ndarray]]], title: str, path: Path
) -> None:
    """Small multiples comparing up to two variants (e.g. selection types) per instance."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (problem, variants) in zip(axes.flat, entries):
        for color, (label, histories) in zip(SERIES, variants.items()):
            _draw_convergence(ax, histories, problem.optimum, color, label)
        _style_axes(ax, problem, histories.shape[1])
        ax.set_title(_title(problem), loc="left")
    _label_grid(fig, axes)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.99, 0.995))
    _save(fig, path)


def _label_grid(fig: plt.Figure, axes: np.ndarray) -> None:
    for ax in axes[-1]:
        ax.set_xlabel(X_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
