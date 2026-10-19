"""Crossover (discrete recombination) between the parent and its mutant."""

import numpy as np


def binomial_crossover(
    parents: np.ndarray, mutants: np.ndarray, cr: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Take each parameter from the mutant with probability CR, otherwise from the parent.

    One randomly chosen parameter (j_rand) always comes from the mutant, so the trial
    vector is never an exact copy of the parent.
    """
    n, dim = parents.shape
    take_mutant = rng.random((n, dim)) < cr[:, None]
    take_mutant[np.arange(n), rng.integers(0, dim, size=n)] = True
    return np.where(take_mutant, mutants, parents)
