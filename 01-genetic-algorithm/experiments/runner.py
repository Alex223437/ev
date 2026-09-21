"""Repeated independent runs of the GA."""

import numpy as np

from ga import GAConfig, GeneticAlgorithm
from problems import Problem

BUDGET_PER_DIMENSION = 100

Instance = tuple[str, int]  # (problem name, dimension)


def instance_id(name: str, dim: int) -> str:
    return f"{name}_{dim}D"


def budget_for(dim: int) -> int:
    """Objective function evaluations allowed for one run: 100 * D."""
    return BUDGET_PER_DIMENSION * dim


def run_many(config: GAConfig, problem: Problem, runs: int, seed: int = 0) -> np.ndarray:
    """Run the GA `runs` times and return best-so-far histories as a (runs, budget) matrix.

    Run `i` uses seed `seed + i`, so different configurations are compared on the
    same random streams.
    """
    ga = GeneticAlgorithm(config)
    budget = budget_for(problem.dim)
    return np.stack(
        [ga.run(problem, budget, np.random.default_rng(seed + i)).history for i in range(runs)]
    )
