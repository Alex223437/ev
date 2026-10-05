import numpy as np

from .base import Problem


class Rosenbrock(Problem):
    """Non-separable valley: minimum 0 at (1, ..., 1), hard to follow the curved ridge."""

    name = "rosenbrock"
    title = "Rosenbrock"
    bounds = (-2.048, 2.048)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        head, tail = points[:, :-1], points[:, 1:]
        return np.sum(100.0 * (tail - head**2) ** 2 + (1.0 - head) ** 2, axis=1)
