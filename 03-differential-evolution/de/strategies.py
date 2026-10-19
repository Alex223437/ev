"""Mutation strategies: how the mutant (noise) vector is built from the population."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

MutateFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def distinct_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """For every individual i return `count` distinct indices, all different from i.

    Drawing the smallest `count` of n random keys per row is a vectorized way of taking
    a uniformly random subset; the columns are then shuffled so that the roles
    (base vector, first and second difference vector) are assigned at random.
    """
    keys = rng.random((n, n))
    np.fill_diagonal(keys, np.inf)  # an individual is never its own donor
    chosen = np.argpartition(keys, count - 1, axis=1)[:, :count]
    order = np.argsort(rng.random(chosen.shape), axis=1)
    return np.take_along_axis(chosen, order, axis=1)


def rand1(population: np.ndarray, best: np.ndarray, donors: np.ndarray, f: np.ndarray) -> np.ndarray:
    """v_i = x_r1 + F * (x_r2 - x_r3)"""
    return population[donors[:, 0]] + f[:, None] * (population[donors[:, 1]] - population[donors[:, 2]])


def best1(population: np.ndarray, best: np.ndarray, donors: np.ndarray, f: np.ndarray) -> np.ndarray:
    """v_i = x_best + F * (x_r1 - x_r2)"""
    return best + f[:, None] * (population[donors[:, 0]] - population[donors[:, 1]])


@dataclass(frozen=True)
class Strategy:
    name: str
    title: str
    n_donors: int
    mutate: MutateFn


STRATEGIES: dict[str, Strategy] = {
    "rand1": Strategy("rand1", "DE/rand/1/bin", 3, rand1),
    "best1": Strategy("best1", "DE/best/1/bin", 2, best1),
}
