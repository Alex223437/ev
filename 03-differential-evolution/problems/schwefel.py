import numpy as np

from .base import Problem


class Schwefel(Problem):
    """Multimodal and deceptive: the global minimum 0 at x_i = 420.9687 lies far from the
    best local minima, so a search that only refines locally gets stuck."""

    name = "schwefel"
    title = "Schwefel"
    bounds = (-500.0, 500.0)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return 418.9828872724339 * points.shape[1] - np.sum(
            points * np.sin(np.sqrt(np.abs(points))), axis=1
        )
