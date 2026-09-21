import numpy as np

from .base import Problem


class LeadingOnes(Problem):
    """Length of the uninterrupted block of ones at the left end of the string."""

    name = "leading_ones"
    title = "Leading ones"

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        # The cumulative product stays 1 until the first zero, so its sum is the prefix length.
        return np.cumprod(population, axis=1, dtype=np.int64).sum(axis=1)
