"""Objective function wrapper that enforces the evaluation budget."""

from typing import Protocol

import numpy as np


class Objective(Protocol):
    """Anything the GA can optimize: binary strings of length `dim`, maximized fitness."""

    dim: int

    def evaluate(self, population: np.ndarray) -> np.ndarray: ...


class BudgetedEvaluator:
    """Counts objective function evaluations and records the best-so-far fitness after each one.

    `history[i]` is the best fitness found within the first `i + 1` evaluations, so a
    finished run always yields an array of exactly `budget` values.
    """

    def __init__(self, problem: Objective, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"budget must be positive, got {budget}")
        self.problem = problem
        self.budget = budget
        self.evaluations = 0
        self.history = np.empty(budget, dtype=np.float64)
        self.best_fitness = -np.inf
        self.best_solution: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def __call__(self, population: np.ndarray) -> np.ndarray:
        n = len(population)
        if n > self.remaining:
            raise RuntimeError(f"requested {n} evaluations, only {self.remaining} left in the budget")

        fitness = self.problem.evaluate(population)
        running_best = np.maximum.accumulate(fitness.astype(np.float64))
        running_best = np.maximum(running_best, self.best_fitness)
        self.history[self.evaluations : self.evaluations + n] = running_best
        self.evaluations += n

        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = float(fitness[best_idx])
            self.best_solution = population[best_idx].copy()
        return fitness
