"""Summary statistics of repeated runs (minimization)."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RunStatistics:
    """Statistics of the final best-so-far objective value over independent runs."""

    best: float
    worst: float
    mean: float
    median: float
    std: float  # sample standard deviation (ddof=1)
    runs: int

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(histories: np.ndarray) -> RunStatistics:
    """`histories` is a (runs, budget) matrix of best-so-far values."""
    final = histories[:, -1]
    return RunStatistics(
        best=float(final.min()),
        worst=float(final.max()),
        mean=float(final.mean()),
        median=float(np.median(final)),
        std=float(final.std(ddof=1)) if len(final) > 1 else 0.0,
        runs=len(final),
    )
