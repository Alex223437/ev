"""Binary genetic algorithm on One-max and Leading ones (10, 30 and 100 D).

    python main.py tune   # grid search of control parameters -> results/tuning/, results/best_configs.json
    python main.py run    # 10 runs per instance with the best configurations -> plots and statistics
    python main.py all    # tune, then run
"""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from experiments.plotting import save_comparison_plot, save_convergence_plot, save_overview_plot
from experiments.runner import instance_id, run_many
from experiments.statistics import summarize
from experiments.tables import markdown_table, write_csv
from experiments.tuning import PARAM_GRID, parameter_effects, tune, unique_configs
from ga import GAConfig
from problems import make_problem

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TUNING_DIR = RESULTS_DIR / "tuning"
BEST_CONFIGS_PATH = RESULTS_DIR / "best_configs.json"

PROBLEM_NAMES = ("onemax", "leading_ones")
DIMENSIONS = (10, 30, 100)
INSTANCES = [(name, dim) for name in PROBLEM_NAMES for dim in DIMENSIONS]

# Tuning uses different seeds than the final runs, so the reported statistics are
# not measured on the very runs the configuration was selected on.
TUNING_SEED_OFFSET = 1_000
TOP_CONFIGS_SHOWN = 5

TUNING_COLUMNS = {
    "pop_size": "N",
    "elite_ratio": "Elitismus",
    "selection": "Selekce",
    "crossover": "Křížení",
    "crossover_rate": "pc",
    "mutation_rate": "pm",
    "mean_final": "Průměr",
    "std_final": "Sm. odch.",
    "success_rate": "Optimum",
    "auc": "AUC",
}
STATS_COLUMNS = {
    "problem": "Úloha",
    "dim": "D",
    "budget": "Ohodnocení",
    "best": "Nejlepší",
    "worst": "Nejhorší",
    "mean": "Průměr",
    "median": "Medián",
    "std": "Sm. odchylka",
    "success_rate": "Optimum",
    "mean_evals_to_optimum": "Ohodnocení do optima (průměr)",
}
SELECTION_LABELS = {"roulette": "ruletová selekce", "rank": "pořadová selekce"}


def cmd_tune(args: argparse.Namespace) -> None:
    n_configs = sum(len(unique_configs(dim)) for _, dim in INSTANCES)
    print(f"Grid search: {n_configs} (instance, configuration) pairs x {args.tuning_runs} runs")
    start = time.perf_counter()
    ranking = tune(INSTANCES, args.tuning_runs, args.seed + TUNING_SEED_OFFSET, args.jobs)
    print(f"finished in {time.perf_counter() - start:.0f} s\n")

    best_configs, top_tables = {}, []
    for (name, dim), results in ranking.items():
        iid = instance_id(name, dim)
        rows = [result.to_row() for result in results]
        write_csv(rows, TUNING_DIR / f"{iid}.csv")

        winner = results[0]
        best_configs[iid] = {
            "problem": name,
            "dim": dim,
            "config": winner.config.to_dict(),
            "mean_final": winner.mean_final,
            "auc": winner.auc,
        }
        title = make_problem(name, dim).title
        top_tables.append(f"### {title}, D = {dim}\n\n" + markdown_table(rows[:TOP_CONFIGS_SHOWN], TUNING_COLUMNS))
        print(f"{iid:<17} {winner.config.describe():<60} mean={winner.mean_final:.2f}")

    BEST_CONFIGS_PATH.write_text(json.dumps(best_configs, indent=2) + "\n")
    (TUNING_DIR / "top_configs.md").write_text("\n".join(top_tables))

    effects = parameter_effects(ranking, PARAM_GRID)
    write_csv(effects, TUNING_DIR / "parameter_effects.csv")
    effect_columns = {"parameter": "Parametr", "value": "Hodnota"}
    effect_columns |= {
        instance_id(name, dim): f"{make_problem(name, dim).title} {dim}D" for name, dim in INSTANCES
    }
    readable = [{**row, "parameter": TUNING_COLUMNS[row["parameter"]]} for row in effects]
    (TUNING_DIR / "parameter_effects.md").write_text(markdown_table(readable, effect_columns))


def load_best_configs() -> dict[str, GAConfig]:
    if not BEST_CONFIGS_PATH.exists():
        print(f"{BEST_CONFIGS_PATH.name} not found, using the default configuration (run `tune` first)")
        return {}
    data = json.loads(BEST_CONFIGS_PATH.read_text())
    return {iid: GAConfig.from_dict(entry["config"]) for iid, entry in data.items()}


def cmd_run(args: argparse.Namespace) -> None:
    configs = load_best_configs()
    stats_rows, overview, selection_comparison = [], [], []

    for name, dim in INSTANCES:
        problem = make_problem(name, dim)
        iid = instance_id(name, dim)
        config = configs.get(iid, GAConfig())

        histories = run_many(config, problem, args.runs, args.seed)
        stats = summarize(histories, problem.optimum)
        stats_rows.append(
            {
                "problem": problem.title,
                "dim": dim,
                "budget": histories.shape[1],
                **stats.to_dict(),
                "success_rate": f"{stats.successes}/{stats.runs}",
                "config": config.describe(),
            }
        )
        save_convergence_plot(histories, problem, config.describe(), PLOTS_DIR / f"{iid}.png")
        overview.append((problem, histories))

        # Same configuration with the other selection type, on the same seeds.
        variants = {
            SELECTION_LABELS[selection]: (
                histories
                if selection == config.selection
                else run_many(replace(config, selection=selection), problem, args.runs, args.seed)
            )
            for selection in SELECTION_LABELS
        }
        selection_comparison.append((problem, variants))

        print(
            f"{iid:<17} best={stats.best:g} worst={stats.worst:g} mean={stats.mean:.2f} "
            f"median={stats.median:g} std={stats.std:.2f} optimum={stats.successes}/{stats.runs}"
        )

    write_csv(stats_rows, RESULTS_DIR / "stats.csv")
    (RESULTS_DIR / "stats.md").write_text(markdown_table(stats_rows, STATS_COLUMNS))
    save_overview_plot(overview, PLOTS_DIR / "overview.png")
    save_comparison_plot(
        selection_comparison,
        "Ruletová vs. pořadová selekce (ostatní parametry podle nejlepší konfigurace)",
        PLOTS_DIR / "selection_comparison.png",
    )
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
    if args.command in ("tune", "all"):
        cmd_tune(args)
    if args.command in ("run", "all"):
        cmd_run(args)


if __name__ == "__main__":
    main()
