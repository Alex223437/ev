"""Repeated independent runs of one GA variant."""

import numpy as np

from ga import GAConfig, GeneticAlgorithm
from problems import Problem
from representations import make_representation

BUDGET = 10_000  # objective function evaluations per run, as required by the assignment
DIMENSION = 10

Instance = tuple[str, str]  # (problem name, variant name)


def instance_id(problem_name: str, variant: str) -> str:
    return f"{problem_name}_{variant}"


def build(config: GAConfig, problem: Problem, variant: str) -> GeneticAlgorithm:
    representation = make_representation(variant, problem, config.mutation_rate, config.sigma_ratio)
    return GeneticAlgorithm(config, representation)


def run_many(
    config: GAConfig, problem: Problem, variant: str, runs: int, seed: int = 0, budget: int = BUDGET
) -> np.ndarray:
    """Run the variant `runs` times and return best-so-far histories as a (runs, budget) matrix.

    Run `i` uses seed `seed + i`, so different configurations are compared on the same
    random streams.
    """
    algorithm = build(config, problem, variant)
    return np.stack(
        [algorithm.run(problem, budget, np.random.default_rng(seed + i)).history for i in range(runs)]
    )
