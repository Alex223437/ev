"""Mutation operator."""

import numpy as np


def bit_flip_mutation(population: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Walk through every bit and invert it with probability `rate`; returns a new array."""
    flips = rng.random(population.shape) < rate
    return population ^ flips.astype(population.dtype)
