"""Repeated independent runs of one DE variant."""

import numpy as np

from de import DEConfig, DifferentialEvolution
from problems import Problem

BUDGET = 10_000  # objective function evaluations per run
DIMENSION = 10

Instance = tuple[str, str]  # (problem name, variant name)


def instance_id(problem_name: str, variant: str) -> str:
    return f"{problem_name}_{variant}"


def run_many(
    config: DEConfig, problem: Problem, runs: int, seed: int = 0, budget: int = BUDGET
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the variant `runs` times.

    Returns the best-so-far histories (runs, budget) and the per-generation mean F and
    CR of every run (runs, generations), which show how jDE adapts its parameters.
    """
    algorithm = DifferentialEvolution(config)
    results = [algorithm.run(problem, budget, np.random.default_rng(seed + i)) for i in range(runs)]
    return (
        np.stack([r.history for r in results]),
        np.stack([r.mean_f for r in results]),
        np.stack([r.mean_cr for r in results]),
    )
