"""Parent selection operators for minimization.

Every operator takes the objective values of the population (lower is better) and
returns indices of `n` selected individuals, drawn independently with replacement.
"""

from typing import Callable

import numpy as np

SelectionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def roulette_selection(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fitness-proportionate selection.

    Objective values are costs, so they are first turned into fitness by subtracting
    them from the worst value in the population (windowing). The worst individual then
    gets probability 0; if the whole population is equally good, selection is uniform.
    """
    values = np.asarray(values, dtype=np.float64)
    fitness = values.max() - values
    total = fitness.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.integers(0, len(values), size=n)
    return rng.choice(len(values), size=n, p=fitness / total)


def rank_selection(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Linear rank selection: the worst individual has rank 1, the best rank N, P(i) ~ rank.

    Only the ordering matters, so no transformation of the objective values is needed.
    Individuals with equal values share the average of their ranks.
    """
    _, inverse, counts = np.unique(-np.asarray(values, dtype=np.float64), return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]
    return rng.choice(len(values), size=n, p=ranks / ranks.sum())


SELECTIONS: dict[str, SelectionFn] = {
    "roulette": roulette_selection,
    "rank": rank_selection,
}
