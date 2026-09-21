"""Summary statistics of repeated runs."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RunStatistics:
    """Statistics of the final best-so-far fitness over independent runs."""

    best: float
    worst: float
    mean: float
    median: float
    std: float  # sample standard deviation (ddof=1)
    runs: int
    successes: int  # runs that reached the optimum
    mean_evals_to_optimum: float | None  # averaged over successful runs only

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(histories: np.ndarray, optimum: float) -> RunStatistics:
    """`histories` is a (runs, budget) matrix of best-so-far values."""
    final = histories[:, -1]
    reached = histories >= optimum
    successful = reached[:, -1]
    # First evaluation (1-based) at which each successful run hit the optimum.
    evals_to_optimum = reached[successful].argmax(axis=1) + 1

    return RunStatistics(
        best=float(final.max()),
        worst=float(final.min()),
        mean=float(final.mean()),
        median=float(np.median(final)),
        std=float(final.std(ddof=1)) if len(final) > 1 else 0.0,
        runs=len(final),
        successes=int(successful.sum()),
        mean_evals_to_optimum=float(evals_to_optimum.mean()) if successful.any() else None,
    )


def normalized_auc(histories: np.ndarray, optimum: float) -> float:
    """Mean area under the convergence curves scaled to [0, 1]; higher means faster convergence."""
    return float(histories.mean() / optimum)
