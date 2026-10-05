"""Real-valued representation: the chromosome is the vector of real parameters itself."""

import numpy as np

from problems import Problem

from .base import Representation, single_point_mask


class RealRepresentation(Representation):
    """Single-point crossover between dimensions (the cut never splits a number).

    Two mutation operators are available:
    - "gaussian": x + N(0, sigma^2) with sigma given as a fraction of the domain width,
      i.e. a local step whose size is a control parameter;
    - "uniform": the gene is replaced by a uniformly random value from the whole domain,
      i.e. a restart of that coordinate.
    Mutated values are clipped back into the domain.
    """

    def __init__(self, problem: Problem, mutation: str, mutation_rate: float, sigma_ratio: float | None = None) -> None:
        super().__init__(problem, mutation_rate)
        if mutation not in ("gaussian", "uniform"):
            raise ValueError(f"unknown mutation {mutation!r}, choose from ('gaussian', 'uniform')")
        if mutation == "gaussian" and sigma_ratio is None:
            raise ValueError("gaussian mutation needs sigma_ratio")
        self.mutation = mutation
        self.name = f"real_{mutation}"
        self.title = "Real + Gaussova mutace" if mutation == "gaussian" else "Real + rovnoměrná mutace"
        self.sigma_ratio = sigma_ratio
        self.sigma = None if sigma_ratio is None else sigma_ratio * problem.width

    def random_population(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.problem.lo, self.problem.hi, size=(n, self.problem.dim))

    def crossover(self, parents_a: np.ndarray, parents_b: np.ndarray, rng: np.random.Generator):
        mask = single_point_mask(len(parents_a), self.problem.dim, rng)
        return np.where(mask, parents_b, parents_a), np.where(mask, parents_a, parents_b)

    def mutate(self, genotypes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        hits = rng.random(genotypes.shape) < self.mutation_rate
        if self.mutation == "gaussian":
            mutated = genotypes + rng.normal(0.0, self.sigma, size=genotypes.shape)
        else:
            mutated = rng.uniform(self.problem.lo, self.problem.hi, size=genotypes.shape)
        return np.clip(np.where(hits, mutated, genotypes), self.problem.lo, self.problem.hi)

    def decode(self, genotypes: np.ndarray) -> np.ndarray:
        return genotypes

    def describe(self) -> str:
        if self.mutation == "gaussian":
            return f"reálné vektory, mutace N(0, sigma^2), sigma = {self.sigma_ratio:g} * rozsah = {self.sigma:.3g}"
        return "reálné vektory, mutace = nová náhodná hodnota z celého rozsahu"
