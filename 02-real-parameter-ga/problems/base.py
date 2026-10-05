"""Real-parameter benchmark functions (minimization)."""

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """Continuous minimization problem on a box domain [lo, hi]^dim."""

    name: str = "problem"
    title: str = "Problem"
    bounds: tuple[float, float] = (-1.0, 1.0)
    optimum: float = 0.0

    def __init__(self, dim: int = 10) -> None:
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")
        self.dim = dim

    @property
    def lo(self) -> float:
        return self.bounds[0]

    @property
    def hi(self) -> float:
        return self.bounds[1]

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @abstractmethod
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Return the function value of every row of a (n, dim) matrix of real vectors."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"
