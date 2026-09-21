"""Crossover operators.

Each operator only decides *which genes* are exchanged between the two parents of
every pair: it returns a boolean mask of shape (n_pairs, dim). `recombine` then
performs the exchange, so all operators share the same swapping code.
"""

from typing import Callable

import numpy as np

CrossoverMaskFn = Callable[[int, int, np.random.Generator], np.ndarray]


def one_point(n_pairs: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Cut both parents at a random point in 1..dim-1 and swap the first parts."""
    points = rng.integers(1, dim, size=n_pairs)
    return np.arange(dim) < points[:, None]


def two_point(n_pairs: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Swap the segment between two distinct random cut points."""
    # Two distinct cut points per pair, drawn uniformly from 1..dim-1.
    cuts = np.sort(rng.random((n_pairs, dim - 1)).argsort(axis=1)[:, :2] + 1, axis=1)
    genes = np.arange(dim)
    return (genes >= cuts[:, :1]) & (genes < cuts[:, 1:])


def uniform(n_pairs: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Swap every gene independently with probability 0.5."""
    return rng.random((n_pairs, dim)) < 0.5


def recombine(
    parents_a: np.ndarray, parents_b: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create two children per pair by exchanging parent genes where `mask` is True."""
    children_a = np.where(mask, parents_b, parents_a)
    children_b = np.where(mask, parents_a, parents_b)
    return children_a, children_b


CROSSOVERS: dict[str, CrossoverMaskFn] = {
    "one_point": one_point,
    "two_point": two_point,
    "uniform": uniform,
}
