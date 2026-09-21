import numpy as np

from .base import Problem


class OneMax(Problem):
    """Number of ones in the string."""

    name = "onemax"
    title = "One-max"

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return population.sum(axis=1, dtype=np.int64)
