"""Grid search over the GA control parameters, separately for every (function, variant) pair."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from ga import GAConfig
from problems import make_problem
from representations import BITS_PER_DIM

from .runner import DIMENSION, Instance, run_many
from .statistics import summarize

Grid = dict[str, list]

# Parameters shared by all variants.
BASE_GRID: Grid = {
    "pop_size": [20, 50, 100],
    "elite_ratio": [0.1, 0.2],
    "selection": ["roulette", "rank"],
    "crossover_rate": [0.8, 1.0],
}

# Mutation means something different in each representation, so its grid differs too:
# per bit for the binary variants (1/320 = one flipped bit per chromosome on average),
# per gene for the real-valued ones.
ONE_BIT = 1.0 / (DIMENSION * BITS_PER_DIM)
MUTATION_GRIDS: dict[str, Grid] = {
    "binary": {"mutation_rate": [ONE_BIT, 0.01, 0.03]},
    "real_gaussian": {"mutation_rate": [0.1, 0.3, 1.0], "sigma_ratio": [0.01, 0.05, 0.1, 0.3]},
    "real_uniform": {"mutation_rate": [0.1, 0.3, 1.0]},
}


def grid_for(variant: str) -> Grid:
    key = variant if variant in MUTATION_GRIDS else "binary"
    return {**BASE_GRID, **MUTATION_GRIDS[key]}


def grid_configs(variant: str) -> list[GAConfig]:
    grid = grid_for(variant)
    return [GAConfig(**dict(zip(grid, values))) for values in product(*grid.values())]


@dataclass(frozen=True)
class TuningResult:
    problem: str
    variant: str
    config: GAConfig
    best_final: float
    mean_final: float
    median_final: float
    std_final: float
    runs: int

    def sort_key(self) -> tuple[float, float]:
        """Lower mean final value wins; ties are broken by the median."""
        return (self.mean_final, self.median_final)

    def to_row(self) -> dict:
        config = self.config.to_dict()
        return {
            **config,
            "mean_final": self.mean_final,
            "median_final": self.median_final,
            "best_final": self.best_final,
            "std_final": self.std_final,
        }


def evaluate_config(problem_name: str, variant: str, config: GAConfig, runs: int, seed: int) -> TuningResult:
    problem = make_problem(problem_name, DIMENSION)
    stats = summarize(run_many(config, problem, variant, runs, seed))
    return TuningResult(
        problem=problem_name,
        variant=variant,
        config=config,
        best_final=stats.best,
        mean_final=stats.mean,
        median_final=stats.median,
        std_final=stats.std,
        runs=runs,
    )


def _evaluate_task(task: tuple) -> TuningResult:
    return evaluate_config(*task)


def tune(
    instances: Sequence[Instance], runs: int, seed: int, jobs: int | None = None
) -> dict[Instance, list[TuningResult]]:
    """Evaluate every configuration of every instance; results are sorted best first."""
    tasks = [
        (problem_name, variant, config, runs, seed)
        for problem_name, variant in instances
        for config in grid_configs(variant)
    ]
    if jobs == 1:
        results = [_evaluate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_evaluate_task, tasks, chunksize=8))

    ranking: dict[Instance, list[TuningResult]] = {instance: [] for instance in instances}
    for result in results:
        ranking[(result.problem, result.variant)].append(result)
    for results_of_instance in ranking.values():
        results_of_instance.sort(key=TuningResult.sort_key)
    return ranking


def parameter_effects(
    ranking: dict[Instance, list[TuningResult]], variant: str, problem_names: Sequence[str]
) -> list[dict]:
    """For every parameter value: the median result over all configurations using it.

    The median is used instead of the mean because a single diverged configuration would
    otherwise dominate the column. Lower is better.
    """
    rows = []
    for param, values in grid_for(variant).items():
        for value in values:
            row = {"parameter": param, "value": value}
            for problem_name in problem_names:
                matching = [
                    r.mean_final
                    for r in ranking[(problem_name, variant)]
                    if getattr(r.config, param) == value
                ]
                row[problem_name] = float(np.median(matching)) if matching else None
            rows.append(row)
    return rows


def best_per_value(
    ranking: dict[Instance, list[TuningResult]], variant: str, param: str, problem_names: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Best achievable mean final value for each value of one parameter, per function."""
    values = np.array(grid_for(variant)[param], dtype=float)
    series = {}
    for problem_name in problem_names:
        results = ranking[(problem_name, variant)]
        series[problem_name] = (
            values,
            np.array([
                min(r.mean_final for r in results if getattr(r.config, param) == value)
                for value in grid_for(variant)[param]
            ]),
        )
    return series
