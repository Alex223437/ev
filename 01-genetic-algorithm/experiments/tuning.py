"""Grid search over the GA control parameters."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from ga import GAConfig
from problems import make_problem

from .runner import Instance, instance_id, run_many
from .statistics import normalized_auc, summarize

PARAM_GRID: dict[str, list] = {
    "pop_size": [5, 10, 20, 50, 100],
    "elite_ratio": [0.1, 0.2],
    "selection": ["roulette", "rank"],
    "crossover": ["one_point", "two_point", "uniform"],
    "crossover_rate": [0.8, 1.0],
    "mutation_rate": [0.005, 0.01, None],  # None = 1/D
}


@dataclass(frozen=True)
class TuningResult:
    problem: str
    dim: int
    config: GAConfig
    mean_final: float
    std_final: float
    successes: int
    runs: int
    auc: float

    def sort_key(self) -> tuple[float, float]:
        """Higher mean final fitness wins; ties are broken by faster convergence."""
        return (-self.mean_final, -self.auc)

    def to_row(self) -> dict:
        config = self.config.to_dict()
        if config["mutation_rate"] is None:
            config["mutation_rate"] = "1/D"
        return {
            **config,
            "mean_final": self.mean_final,
            "std_final": self.std_final,
            "success_rate": f"{self.successes}/{self.runs}",
            "auc": round(self.auc, 4),
        }


def grid_configs(grid: dict[str, list] = PARAM_GRID) -> list[GAConfig]:
    """The full (nominal) grid: every combination of parameter values."""
    return [GAConfig(**dict(zip(grid, values))) for values in product(*grid.values())]


def effective_key(config: GAConfig, dim: int) -> tuple:
    """Configurations with the same key behave identically on a D-dimensional problem,
    e.g. pm = 0.01 and pm = 1/D when D = 100, or two elite ratios giving the same elite count."""
    return (
        config.pop_size,
        config.n_elite,
        config.selection,
        config.crossover,
        config.crossover_rate,
        config.mutation_rate_for(dim),
    )


def unique_configs(dim: int, grid: dict[str, list] = PARAM_GRID) -> list[GAConfig]:
    """Grid configurations for dimension `dim`, keeping only one of each group of equivalent ones."""
    unique: dict[tuple, GAConfig] = {}
    for config in grid_configs(grid):
        unique.setdefault(effective_key(config, dim), config)
    return list(unique.values())


def evaluate_config(problem_name: str, dim: int, config: GAConfig, runs: int, seed: int) -> TuningResult:
    problem = make_problem(problem_name, dim)
    histories = run_many(config, problem, runs, seed)
    stats = summarize(histories, problem.optimum)
    return TuningResult(
        problem=problem_name,
        dim=dim,
        config=config,
        mean_final=stats.mean,
        std_final=stats.std,
        successes=stats.successes,
        runs=runs,
        auc=normalized_auc(histories, problem.optimum),
    )


def _evaluate_task(task: tuple) -> TuningResult:
    return evaluate_config(*task)


def tune(
    instances: Sequence[Instance],
    runs: int,
    seed: int,
    jobs: int | None = None,
    grid: dict[str, list] = PARAM_GRID,
) -> dict[Instance, list[TuningResult]]:
    """Evaluate every distinct grid configuration on every instance; results are sorted best first."""
    tasks = [(name, dim, config, runs, seed) for name, dim in instances for config in unique_configs(dim, grid)]
    if jobs == 1:
        results = [_evaluate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_evaluate_task, tasks, chunksize=16))

    ranking: dict[Instance, list[TuningResult]] = {instance: [] for instance in instances}
    for result in results:
        ranking[(result.problem, result.dim)].append(result)
    for results_of_instance in ranking.values():
        results_of_instance.sort(key=TuningResult.sort_key)
    return ranking


def parameter_effects(
    ranking: dict[Instance, list[TuningResult]], grid: dict[str, list] = PARAM_GRID
) -> list[dict]:
    """Marginal effect of every parameter value, separately for each instance: the AUC
    averaged over all configurations of the full grid that use this value.

    Equivalent configurations were evaluated only once, so each nominal configuration
    takes the result of its evaluated equivalent; this keeps the averages balanced.
    """
    configs = grid_configs(grid)
    auc_per_instance = {}
    for (name, dim), results in ranking.items():
        by_key = {effective_key(r.config, dim): r.auc for r in results}
        auc_per_instance[instance_id(name, dim)] = [by_key[effective_key(c, dim)] for c in configs]

    rows = []
    for param, values in grid.items():
        for value in values:
            uses_value = [getattr(config, param) == value for config in configs]
            row = {"parameter": param, "value": "1/D" if value is None else value}
            for iid, aucs in auc_per_instance.items():
                row[iid] = round(float(np.mean([auc for auc, used in zip(aucs, uses_value) if used])), 3)
            rows.append(row)
    return rows
