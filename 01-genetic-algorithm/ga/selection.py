"""Parent selection operators.

Every operator takes the fitness vector of the population and returns indices of
`n` selected individuals (drawn independently, with replacement). Fitness is
assumed to be non-negative and maximized.
"""

from typing import Callable

import numpy as np

SelectionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def roulette_selection(fitness: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fitness-proportionate selection: P(i) = f_i / sum(f)."""
    fitness = np.asarray(fitness, dtype=np.float64)
    total = fitness.sum()
    if total <= 0:
        # All individuals have zero fitness (typical early on Leading ones): pick uniformly.
        return rng.integers(0, len(fitness), size=n)
    return rng.choice(len(fitness), size=n, p=fitness / total)


def rank_selection(fitness: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Linear rank selection: the worst individual has rank 1, the best rank N, P(i) ~ rank.

    Individuals with equal fitness share the average of their ranks, so they are
    equally likely to be selected.
    """
    _, inverse, counts = np.unique(fitness, return_inverse=True, return_counts=True)
    # A group of equal values occupies ranks (cum - count + 1) .. cum; take the midpoint.
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]
    return rng.choice(len(fitness), size=n, p=ranks / ranks.sum())


SELECTIONS: dict[str, SelectionFn] = {
    "roulette": roulette_selection,
    "rank": rank_selection,
}
