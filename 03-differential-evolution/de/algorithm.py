"""Differential evolution: mutation -> crossover -> greedy selection."""

from dataclasses import dataclass

import numpy as np

from .config import DEConfig
from .crossover import binomial_crossover
from .evaluator import BudgetedEvaluator, Objective
from .parameters import FixedParameters, ParameterControl, SelfAdaptiveParameters
from .strategies import STRATEGIES, distinct_indices


@dataclass
class RunResult:
    history: np.ndarray  # best-so-far objective value after each evaluation, length == budget
    best_value: float
    best_point: np.ndarray
    generations: int
    mean_f: np.ndarray   # mean F in the population after each generation (jDE adaptation)
    mean_cr: np.ndarray  # mean CR in the population after each generation


def make_parameter_control(config: DEConfig, np_size: int) -> ParameterControl:
    if config.adaptive:
        return SelfAdaptiveParameters(
            np_size, config.f, config.cr, config.tau1, config.tau2, config.f_low, config.f_high
        )
    return FixedParameters(np_size, config.f, config.cr)


class DifferentialEvolution:
    def __init__(self, config: DEConfig) -> None:
        if config.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {config.strategy!r}, choose from {sorted(STRATEGIES)}")
        self.config = config
        self.strategy = STRATEGIES[config.strategy]

    def run(self, problem: Objective, budget: int, rng: np.random.Generator) -> RunResult:
        """Minimize `problem` until exactly `budget` objective function evaluations are spent."""
        cfg = self.config
        evaluator = BudgetedEvaluator(problem, budget)

        np_size = min(cfg.np_size, budget)
        population = rng.uniform(problem.lo, problem.hi, size=(np_size, problem.dim))
        values = evaluator(population)
        parameters = make_parameter_control(cfg, np_size)

        mean_f, mean_cr, generations = [], [], 0
        while not evaluator.exhausted:
            f, cr = parameters.propose(rng)

            # Mutation: the weighted difference vector shifts the base vector.
            donors = distinct_indices(np_size, self.strategy.n_donors, rng)
            mutants = self.strategy.mutate(population, population[np.argmin(values)], donors, f)

            # Crossover, then keep the trial vectors inside the domain.
            trials = np.clip(binomial_crossover(population, mutants, cr, rng), problem.lo, problem.hi)

            # The last generation is cut short when the budget runs out.
            evaluated = min(np_size, evaluator.remaining)
            trial_values = evaluator(trials[:evaluated])

            # Selection: the trial vector replaces its parent when it is not worse.
            success = np.zeros(np_size, dtype=bool)
            success[:evaluated] = trial_values <= values[:evaluated]
            won = success[:evaluated]
            population[:evaluated] = np.where(won[:, None], trials[:evaluated], population[:evaluated])
            values[:evaluated] = np.where(won, trial_values, values[:evaluated])

            parameters.accept(success)
            generation_f, generation_cr = parameters.means()
            mean_f.append(generation_f)
            mean_cr.append(generation_cr)
            generations += 1

        return RunResult(
            history=evaluator.history,
            best_value=evaluator.best_value,
            best_point=evaluator.best_point,
            generations=generations,
            mean_f=np.array(mean_f),
            mean_cr=np.array(mean_cr),
        )
