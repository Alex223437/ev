"""A representation defines how individuals are stored and recombined.

The genetic algorithm itself only sees genotypes: it asks the representation for a
random population, for offspring of two parents, for mutated offspring and for the
real vectors to pass to the objective function. Swapping the representation is
therefore the only difference between the compared GA variants.
"""

from abc import ABC, abstractmethod

import numpy as np

from problems import Problem


class Representation(ABC):
    name: str = "representation"
    title: str = "Representation"

    def __init__(self, problem: Problem, mutation_rate: float) -> None:
        self.problem = problem
        self.mutation_rate = mutation_rate

    @abstractmethod
    def random_population(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Genotypes of `n` individuals drawn uniformly from the domain."""

    @abstractmethod
    def crossover(self, parents_a: np.ndarray, parents_b: np.ndarray, rng: np.random.Generator):
        """Single-point crossover of paired parents; returns two children per pair."""

    @abstractmethod
    def mutate(self, genotypes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutated copy of the given genotypes."""

    @abstractmethod
    def decode(self, genotypes: np.ndarray) -> np.ndarray:
        """Real vectors (n, dim) the objective function is evaluated on."""

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable description of the representation and its mutation."""


def single_point_mask(n_pairs: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the first part of each pair: a random cut point in 1 .. length-1."""
    points = rng.integers(1, length, size=n_pairs)
    return np.arange(length) < points[:, None]
