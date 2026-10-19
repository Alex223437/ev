"""Differential evolution: DE/rand/1/bin, DE/best/1/bin and the self-adaptive jDE.

    python main.py all      # final experiments + F x CR parameter study
    python main.py run      # 10 runs per (function, variant) -> convergence plots and statistics
    python main.py study    # F x CR sensitivity of the two classic variants -> heat maps

Every function is solved in 10D with a budget of 10 000 objective function evaluations
and 10 independent runs. Control parameters follow the course materials: F = 0.5 and
CR = 0.8 for rand/1 and best/1; jDE starts from F = 0.5, CR = 0.9 with tau1 = tau2 = 0.1.
"""

import argparse
import os
import time
from pathlib import Path

from experiments.parameter_study import CR_VALUES, F_VALUES, STUDIED_VARIANTS, grid_matrix, run_study
from experiments.plotting import (
    save_adaptation_plot,
    save_convergence_plot,
    save_overview_plot,
    save_parameter_study_plot,
)
from experiments.runner import BUDGET, DIMENSION, instance_id, run_many
from experiments.statistics import summarize
from experiments.tables import markdown_table, write_csv
from de import VARIANT_TITLES, variant_config
from problems import make_problem

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

PROBLEM_NAMES = ("sphere", "rosenbrock", "schwefel")
VARIANT_NAMES = tuple(VARIANT_TITLES)

STATS_COLUMNS = {
    "problem": "Funkce",
    "variant": "Verze",
    "best": "Nejlepší",
    "worst": "Nejhorší",
    "mean": "Průměr",
    "median": "Medián",
    "std": "Sm. odchylka",
    "config": "Nastavení",
}
STUDY_COLUMNS = {
    "problem": "Funkce",
    "variant": "Verze",
    "f": "F",
    "cr": "CR",
    "mean_final": "Průměr",
    "median_final": "Medián",
    "best_final": "Nejlepší",
}


def cmd_run(args: argparse.Namespace) -> None:
    stats_rows, overview, adaptation = [], [], []

    for problem_name in PROBLEM_NAMES:
        problem = make_problem(problem_name, DIMENSION)
        curves = {}
        for variant in VARIANT_NAMES:
            config = variant_config(variant, args.np_size)
            histories, mean_f, mean_cr = run_many(config, problem, args.runs, args.seed)
            stats = summarize(histories)
            curves[VARIANT_TITLES[variant]] = histories
            stats_rows.append(
                {
                    "problem": problem.title,
                    "variant": VARIANT_TITLES[variant],
                    **stats.to_dict(),
                    "config": config.describe(),
                }
            )
            if config.adaptive:
                adaptation.append((problem, mean_f, mean_cr))
            print(
                f"{instance_id(problem_name, variant):<22} best={stats.best:<12.4g} mean={stats.mean:<12.4g} "
                f"median={stats.median:<12.4g} worst={stats.worst:<12.4g} std={stats.std:.4g}"
            )

        save_convergence_plot(curves, problem, PLOTS_DIR / f"{problem_name}.png")
        overview.append((problem, curves))

    write_csv(stats_rows, RESULTS_DIR / "stats.csv")
    (RESULTS_DIR / "stats.md").write_text(markdown_table(stats_rows, STATS_COLUMNS))
    save_overview_plot(overview, PLOTS_DIR / "comparison.png")
    save_adaptation_plot(adaptation, PLOTS_DIR / "jde_adaptation.png")
    print(f"\nresults written to {RESULTS_DIR}")


def cmd_study(args: argparse.Namespace) -> None:
    points = len(F_VALUES) * len(CR_VALUES) * len(STUDIED_VARIANTS) * len(PROBLEM_NAMES)
    print(f"F x CR study: {points} grid points x {args.runs} runs")
    start = time.perf_counter()
    results = run_study(PROBLEM_NAMES, args.runs, args.seed, args.jobs)
    print(f"finished in {time.perf_counter() - start:.0f} s")

    write_csv([r.to_row() for r in results], RESULTS_DIR / "parameter_study.csv")
    (RESULTS_DIR / "parameter_study.md").write_text(
        markdown_table([r.to_row() for r in results], STUDY_COLUMNS)
    )

    panels = [
        (VARIANT_TITLES[variant], make_problem(problem_name, DIMENSION).title,
         grid_matrix(results, problem_name, variant))
        for variant in STUDIED_VARIANTS
        for problem_name in PROBLEM_NAMES
    ]
    save_parameter_study_plot(panels, F_VALUES, CR_VALUES, len(STUDIED_VARIANTS), PLOTS_DIR / "parameter_study.png")

    for variant in STUDIED_VARIANTS:
        for problem_name in PROBLEM_NAMES:
            best = min(
                (r for r in results if r.variant == variant and r.problem == problem_name),
                key=lambda r: r.mean_final,
            )
            print(f"{instance_id(problem_name, variant):<22} best grid point: F={best.f:g}, CR={best.cr:g} -> {best.mean_final:.4g}")
    print(f"\nstudy written to {RESULTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "study", "all"])
    parser.add_argument("--runs", type=int, default=10, help="independent runs per setting (default: 10)")
    parser.add_argument("--np-size", type=int, default=None, help="population size NP (default: from the variant)")
    parser.add_argument("--seed", type=int, default=42, help="base random seed (default: 42)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes for the study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{len(PROBLEM_NAMES)} functions x {len(VARIANT_NAMES)} DE versions, D = {DIMENSION}, budget = {BUDGET} evaluations\n")
    if args.command in ("run", "all"):
        cmd_run(args)
    if args.command in ("study", "all"):
        cmd_study(args)


if __name__ == "__main__":
    main()
