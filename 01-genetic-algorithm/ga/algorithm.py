"""Generational genetic algorithm with elitism for binary strings."""

from dataclasses import dataclass

import numpy as np

from .config import GAConfig
from .crossover import CROSSOVERS, recombine
from .evaluator import BudgetedEvaluator, Objective
from .mutation import bit_flip_mutation
from .selection import SELECTIONS


@dataclass
class RunResult:
    history: np.ndarray  # best-so-far fitness after each evaluation, length == budget
    best_fitness: float
    best_solution: np.ndarray
    generations: int


class GeneticAlgorithm:
    def __init__(self, config: GAConfig) -> None:
        if config.selection not in SELECTIONS:
            raise ValueError(f"unknown selection {config.selection!r}, choose from {sorted(SELECTIONS)}")
        if config.crossover not in CROSSOVERS:
            raise ValueError(f"unknown crossover {config.crossover!r}, choose from {sorted(CROSSOVERS)}")
        self.config = config
        self._select = SELECTIONS[config.selection]
        self._crossover_mask = CROSSOVERS[config.crossover]

    def run(self, problem: Objective, budget: int, rng: np.random.Generator) -> RunResult:
        """Optimize `problem` until exactly `budget` objective function evaluations are spent."""
        evaluator = BudgetedEvaluator(problem, budget)
        mutation_rate = self.config.mutation_rate_for(problem.dim)

        population = rng.integers(0, 2, size=(min(self.config.pop_size, budget), problem.dim), dtype=np.uint8)
        fitness = evaluator(population)

        generations = 0
        while not evaluator.exhausted:
            population, fitness = self._next_generation(population, fitness, evaluator, mutation_rate, rng)
            generations += 1

        return RunResult(
            history=evaluator.history,
            best_fitness=evaluator.best_fitness,
            best_solution=evaluator.best_solution,
            generations=generations,
        )

    def _next_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        evaluator: BudgetedEvaluator,
        mutation_rate: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dim = population.shape[1]

        # 1. Elitism: the best individuals move over unchanged, their fitness is reused.
        elite = np.argsort(-fitness, kind="stable")[: cfg.n_elite]

        # 2. The rest of the new population is filled with offspring (the last
        #    generation is cut short when the budget runs out).
        n_children = min(cfg.pop_size - len(elite), evaluator.remaining)
        n_pairs = (n_children + 1) // 2

        # 3. Select two parents for every pair.
        parents = self._select(fitness, 2 * n_pairs, rng)
        parents_a, parents_b = population[parents[:n_pairs]], population[parents[n_pairs:]]

        # 4. Crossover with probability pc; pairs that do not cross yield copies of the parents.
        mask = self._crossover_mask(n_pairs, dim, rng)
        mask &= (rng.random(n_pairs) < cfg.crossover_rate)[:, None]
        children_a, children_b = recombine(parents_a, parents_b, mask)

        # 5. Mutate the offspring (dropping the surplus child of an odd count) and evaluate them.
        children = np.concatenate([children_a, children_b])[:n_children]
        children = bit_flip_mutation(children, mutation_rate, rng)
        children_fitness = evaluator(children)

        return (
            np.concatenate([population[elite], children]),
            np.concatenate([fitness[elite], children_fitness]),
        )
