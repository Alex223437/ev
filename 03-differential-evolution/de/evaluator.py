"""Objective function wrapper that enforces the evaluation budget (minimization)."""

from typing import Protocol

import numpy as np


class Objective(Protocol):
    dim: int

    def evaluate(self, points: np.ndarray) -> np.ndarray: ...


class BudgetedEvaluator:
    """Counts objective function evaluations and records the best-so-far value after each one.

    `history[i]` is the lowest value found within the first `i + 1` evaluations, so a
    finished run always yields an array of exactly `budget` values.
    """

    def __init__(self, problem: Objective, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"budget must be positive, got {budget}")
        self.problem = problem
        self.budget = budget
        self.evaluations = 0
        self.history = np.empty(budget, dtype=np.float64)
        self.best_value = np.inf
        self.best_point: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        if n > self.remaining:
            raise RuntimeError(f"requested {n} evaluations, only {self.remaining} left in the budget")

        values = np.asarray(self.problem.evaluate(points), dtype=np.float64)
        running_best = np.minimum(np.minimum.accumulate(values), self.best_value)
        self.history[self.evaluations : self.evaluations + n] = running_best
        self.evaluations += n

        best_idx = int(np.argmin(values))
        if values[best_idx] < self.best_value:
            self.best_value = float(values[best_idx])
            self.best_point = points[best_idx].copy()
        return values
