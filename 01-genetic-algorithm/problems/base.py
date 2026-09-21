"""Common interface of benchmark problems for the binary GA."""

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """Maximization problem over binary strings of a fixed length."""

    name: str = "problem"
    title: str = "Problem"

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")
        self.dim = dim

    @property
    def optimum(self) -> int:
        """Best achievable fitness (both benchmark problems reach `dim`)."""
        return self.dim

    @abstractmethod
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Return the fitness of every row of a (n, dim) binary matrix."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"
