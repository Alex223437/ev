"""Generational genetic algorithm with elitism, generic in the representation."""

from dataclasses import dataclass

import numpy as np

from representations import Representation

from .config import GAConfig
from .evaluator import BudgetedEvaluator, Objective
from .selection import SELECTIONS


@dataclass
class RunResult:
    history: np.ndarray  # best-so-far objective value after each evaluation, length == budget
    best_value: float
    best_point: np.ndarray
    generations: int


class GeneticAlgorithm:
    """The algorithm only talks to the representation, so all five variants share this code."""

    def __init__(self, config: GAConfig, representation: Representation) -> None:
        if config.selection not in SELECTIONS:
            raise ValueError(f"unknown selection {config.selection!r}, choose from {sorted(SELECTIONS)}")
        self.config = config
        self.representation = representation
        self._select = SELECTIONS[config.selection]

    def run(self, problem: Objective, budget: int, rng: np.random.Generator) -> RunResult:
        """Minimize `problem` until exactly `budget` objective function evaluations are spent."""
        evaluator = BudgetedEvaluator(problem, budget)
        population = self.representation.random_population(min(self.config.pop_size, budget), rng)
        values = evaluator(self.representation.decode(population))

        generations = 0
        while not evaluator.exhausted:
            population, values = self._next_generation(population, values, evaluator, rng)
            generations += 1

        return RunResult(
            history=evaluator.history,
            best_value=evaluator.best_value,
            best_point=evaluator.best_point,
            generations=generations,
        )

    def _next_generation(
        self,
        population: np.ndarray,
        values: np.ndarray,
        evaluator: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config

        # 1. Elitism: the best individuals move over unchanged, their values are reused.
        elite = np.argsort(values, kind="stable")[: cfg.n_elite]

        # 2. The rest of the new population is filled with offspring (the last
        #    generation is cut short when the budget runs out).
        n_children = min(cfg.pop_size - len(elite), evaluator.remaining)
        n_pairs = (n_children + 1) // 2

        # 3. Select two parents for every pair.
        parents = self._select(values, 2 * n_pairs, rng)
        parents_a, parents_b = population[parents[:n_pairs]], population[parents[n_pairs:]]

        # 4. Crossover with probability pc; pairs that do not cross yield copies of the parents.
        crossed_a, crossed_b = self.representation.crossover(parents_a, parents_b, rng)
        crosses = (rng.random(n_pairs) < cfg.crossover_rate)[:, None]
        children_a = np.where(crosses, crossed_a, parents_a)
        children_b = np.where(crosses, crossed_b, parents_b)

        # 5. Mutate the offspring (dropping the surplus child of an odd count) and evaluate them.
        children = np.concatenate([children_a, children_b])[:n_children]
        children = self.representation.mutate(children, rng)
        children_values = evaluator(self.representation.decode(children))

        return (
            np.concatenate([population[elite], children]),
            np.concatenate([values[elite], children_values]),
        )
