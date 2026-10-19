import numpy as np

from .base import Problem


class Sphere(Problem):
    """Separable, unimodal, smooth: f(x) = sum(x_i^2), minimum 0 at the origin."""

    name = "sphere"
    title = "Sphere"
    bounds = (-5.12, 5.12)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.sum(points**2, axis=1)
