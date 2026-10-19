"""Sensitivity of the classic variants to F and CR.

The course materials recommend F = 0.5 and CR = 0.8, but also note that separable
functions prefer a much lower CR. The study runs the whole F x CR grid so that the
recommendation can be checked instead of taken on faith.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from de import variant_config
from problems import make_problem

from .runner import DIMENSION, run_many
from .statistics import summarize

F_VALUES = [0.3, 0.5, 0.7, 0.9]
CR_VALUES = [0.1, 0.2, 0.5, 0.8, 0.9]
STUDIED_VARIANTS = ("rand1", "best1")


@dataclass(frozen=True)
class StudyResult:
    problem: str
    variant: str
    f: float
    cr: float
    mean_final: float
    median_final: float
    best_final: float

    def to_row(self) -> dict:
        return {
            "problem": self.problem,
            "variant": self.variant,
            "f": self.f,
            "cr": self.cr,
            "mean_final": self.mean_final,
            "median_final": self.median_final,
            "best_final": self.best_final,
        }


def evaluate_point(problem_name: str, variant: str, f: float, cr: float, runs: int, seed: int) -> StudyResult:
    from dataclasses import replace

    problem = make_problem(problem_name, DIMENSION)
    config = replace(variant_config(variant), f=f, cr=cr)
    histories, _, _ = run_many(config, problem, runs, seed)
    stats = summarize(histories)
    return StudyResult(problem_name, variant, f, cr, stats.mean, stats.median, stats.best)


def _evaluate_task(task: tuple) -> StudyResult:
    return evaluate_point(*task)


def run_study(
    problem_names: Sequence[str], runs: int, seed: int, jobs: int | None = None
) -> list[StudyResult]:
    tasks = [
        (problem_name, variant, f, cr, runs, seed)
        for problem_name in problem_names
        for variant in STUDIED_VARIANTS
        for f, cr in product(F_VALUES, CR_VALUES)
    ]
    if jobs == 1:
        return [_evaluate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_evaluate_task, tasks, chunksize=4))


def grid_matrix(results: Sequence[StudyResult], problem_name: str, variant: str) -> np.ndarray:
    """Mean final values as a (len(F_VALUES), len(CR_VALUES)) matrix."""
    lookup = {(r.f, r.cr): r.mean_final for r in results if r.problem == problem_name and r.variant == variant}
    return np.array([[lookup[(f, cr)] for cr in CR_VALUES] for f in F_VALUES])
