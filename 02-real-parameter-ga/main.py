"""Genetic algorithm for real-parameter problems: five variants compared on three functions.

    python main.py all     # tuning + final experiments
    python main.py tune    # grid search of control parameters -> results/tuning/, results/best_configs.json
    python main.py run     # 10 runs per (function, variant) with the best configurations

Variants: IEEE 754, fixed-point and BCD binary encodings (32 bits per dimension) plus a
real-valued GA with gaussian and with uniform mutation. Every function is solved in 10D
with a budget of 10 000 objective function evaluations and 10 independent runs.
"""

import argparse
import json
import os
import time
from pathlib import Path

from experiments.plotting import save_convergence_plot, save_overview_plot, save_parameter_plot
from experiments.runner import BUDGET, DIMENSION, instance_id, run_many
from experiments.statistics import summarize
from experiments.tables import markdown_table, write_csv
from experiments.tuning import best_per_value, grid_configs, parameter_effects, tune
from ga import GAConfig
from problems import PROBLEMS, make_problem
from representations import VARIANTS, make_representation

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TUNING_DIR = RESULTS_DIR / "tuning"
BEST_CONFIGS_PATH = RESULTS_DIR / "best_configs.json"

PROBLEM_NAMES = ("sphere", "rosenbrock", "schwefel")
INSTANCES = [(problem_name, variant) for problem_name in PROBLEM_NAMES for variant in VARIANTS]

# Tuning uses different seeds than the final runs, so the reported statistics are
# not measured on the very runs the configuration was selected on.
TUNING_SEED_OFFSET = 1_000
TOP_CONFIGS_SHOWN = 5

PARAM_LABELS = {
    "pop_size": "N",
    "elite_ratio": "Elitismus",
    "selection": "Selekce",
    "crossover_rate": "pc",
    "mutation_rate": "pm",
    "sigma_ratio": "sigma / rozsah",
}
TUNING_COLUMNS = {
    **PARAM_LABELS,
    "mean_final": "Průměr",
    "median_final": "Medián",
    "best_final": "Nejlepší",
    "std_final": "Sm. odch.",
}
STATS_COLUMNS = {
    "problem": "Funkce",
    "variant": "Varianta",
    "best": "Nejlepší",
    "worst": "Nejhorší",
    "mean": "Průměr",
    "median": "Medián",
    "std": "Sm. odchylka",
    "config": "Nastavení",
}


def variant_title(variant: str, problem_name: str = "sphere") -> str:
    problem = make_problem(problem_name, DIMENSION)
    return make_representation(variant, problem, 0.01, 0.1).title


def cmd_tune(args: argparse.Namespace) -> None:
    n_configs = sum(len(grid_configs(variant)) for _, variant in INSTANCES)
    print(f"Grid search: {n_configs} (instance, configuration) pairs x {args.tuning_runs} runs")
    start = time.perf_counter()
    ranking = tune(INSTANCES, args.tuning_runs, args.seed + TUNING_SEED_OFFSET, args.jobs)
    print(f"finished in {time.perf_counter() - start:.0f} s\n")

    best_configs, top_tables = {}, []
    for (problem_name, variant), results in ranking.items():
        iid = instance_id(problem_name, variant)
        write_csv([result.to_row() for result in results], TUNING_DIR / f"{iid}.csv")

        winner = results[0]
        best_configs[iid] = {
            "problem": problem_name,
            "variant": variant,
            "config": winner.config.to_dict(),
            "mean_final": winner.mean_final,
        }
        title = f"{make_problem(problem_name, DIMENSION).title} – {variant_title(variant)}"
        top_tables.append(
            f"### {title}\n\n"
            + markdown_table([r.to_row() for r in results[:TOP_CONFIGS_SHOWN]], TUNING_COLUMNS)
        )
        print(f"{iid:<26} {winner.config.describe():<62} mean={winner.mean_final:.4g}")

    BEST_CONFIGS_PATH.write_text(json.dumps(best_configs, indent=2) + "\n")
    (TUNING_DIR / "top_configs.md").write_text("\n".join(top_tables))
    _write_parameter_effects(ranking)
    _write_parameter_plots(ranking)
    print()


def _write_parameter_effects(ranking) -> None:
    columns = {"parameter": "Parametr", "value": "Hodnota"}
    columns |= {name: make_problem(name, DIMENSION).title for name in PROBLEM_NAMES}
    for variant in VARIANTS:
        rows = parameter_effects(ranking, variant, PROBLEM_NAMES)
        write_csv(rows, TUNING_DIR / f"effects_{variant}.csv")
        readable = [{**row, "parameter": PARAM_LABELS[row["parameter"]]} for row in rows]
        (TUNING_DIR / f"effects_{variant}.md").write_text(markdown_table(readable, columns))


def _write_parameter_plots(ranking) -> None:
    """The assignment asks explicitly for experiments with sigma and the mutation rate."""
    titles = {name: make_problem(name, DIMENSION).title for name in PROBLEM_NAMES}
    for param, label, title in (
        ("sigma_ratio", "sigma / rozsah domény", "Vliv sigma (Gaussova mutace, nejlepší konfigurace)"),
        ("mutation_rate", "pravděpodobnost mutace pm", "Vliv pm (Gaussova mutace, nejlepší konfigurace)"),
    ):
        series = best_per_value(ranking, "real_gaussian", param, PROBLEM_NAMES)
        save_parameter_plot(
            {titles[name]: values for name, values in series.items()},
            title, label, PLOTS_DIR / f"real_gaussian_{param}.png",
        )


def load_best_configs() -> dict[str, GAConfig]:
    if not BEST_CONFIGS_PATH.exists():
        print(f"{BEST_CONFIGS_PATH.name} not found, using the default configuration (run `tune` first)")
        return {}
    data = json.loads(BEST_CONFIGS_PATH.read_text())
    return {iid: GAConfig.from_dict(entry["config"]) for iid, entry in data.items()}


def cmd_run(args: argparse.Namespace) -> None:
    configs = load_best_configs()
    stats_rows, overview = [], []

    for problem_name in PROBLEM_NAMES:
        problem = make_problem(problem_name, DIMENSION)
        curves = {}
        for variant in VARIANTS:
            iid = instance_id(problem_name, variant)
            config = configs.get(iid, GAConfig(sigma_ratio=0.05 if variant == "real_gaussian" else None))

            histories = run_many(config, problem, variant, args.runs, args.seed)
            stats = summarize(histories)
            curves[variant_title(variant)] = histories
            stats_rows.append(
                {
                    "problem": problem.title,
                    "variant": variant_title(variant),
                    **stats.to_dict(),
                    "config": config.describe(),
                }
            )
            print(
                f"{iid:<26} best={stats.best:<12.4g} mean={stats.mean:<12.4g} "
                f"median={stats.median:<12.4g} worst={stats.worst:<12.4g} std={stats.std:.4g}"
            )

        save_convergence_plot(curves, problem, PLOTS_DIR / f"{problem_name}.png")
        overview.append((problem, curves))

    write_csv(stats_rows, RESULTS_DIR / "stats.csv")
    (RESULTS_DIR / "stats.md").write_text(markdown_table(stats_rows, STATS_COLUMNS))
    save_overview_plot(overview, PLOTS_DIR / "comparison.png")
    print(f"\nresults written to {RESULTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tune", "run", "all"])
    parser.add_argument("--runs", type=int, default=10, help="independent runs per instance in `run` (default: 10)")
    parser.add_argument(
        "--tuning-runs", type=int, default=30,
        help="runs per configuration in `tune`; more runs make the choice less noisy (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=42, help="base random seed (default: 42)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes for tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{len(PROBLEMS)} functions x {len(VARIANTS)} variants, D = {DIMENSION}, budget = {BUDGET} evaluations\n")
    if args.command in ("tune", "all"):
        cmd_tune(args)
    if args.command in ("run", "all"):
        cmd_run(args)


if __name__ == "__main__":
    main()
