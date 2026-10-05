"""Binary representations: the chromosome is a bit string, 32 bits per dimension."""

import numpy as np

from problems import Problem

from .base import Representation, single_point_mask
from .codecs import CODECS, Codec


class BinaryRepresentation(Representation):
    """Bit string with bit-flip mutation and single-point crossover anywhere in the string.

    The cut point is not aligned to dimension boundaries, so a crossover can also split
    a single number in half - that is a property of the binary representation, not a bug.
    """

    def __init__(self, problem: Problem, codec: Codec, mutation_rate: float) -> None:
        super().__init__(problem, mutation_rate)
        self.codec = codec
        self.name = codec.name
        self.title = codec.title
        self.length = codec.length

    def random_population(self, n: int, rng: np.random.Generator) -> np.ndarray:
        points = rng.uniform(self.problem.lo, self.problem.hi, size=(n, self.problem.dim))
        return self.codec.encode(points)

    def crossover(self, parents_a: np.ndarray, parents_b: np.ndarray, rng: np.random.Generator):
        mask = single_point_mask(len(parents_a), self.length, rng)
        return np.where(mask, parents_b, parents_a), np.where(mask, parents_a, parents_b)

    def mutate(self, genotypes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        flips = rng.random(genotypes.shape) < self.mutation_rate
        return genotypes ^ flips.astype(genotypes.dtype)

    def decode(self, genotypes: np.ndarray) -> np.ndarray:
        return self.codec.decode(genotypes)

    def describe(self) -> str:
        return f"{self.codec.describe()}, bitová mutace"


def make_binary(name: str, problem: Problem, mutation_rate: float) -> BinaryRepresentation:
    return BinaryRepresentation(problem, CODECS[name](problem), mutation_rate)
