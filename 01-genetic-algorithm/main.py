"""Binary genetic algorithm on One-max and Leading ones (10, 30 and 100 D).

    python main.py all                  # everything below for both grids, then compare
    python main.py tune --grid strict   # grid search -> results/<grid>/tuning/, results/<grid>/best_configs.json
    python main.py run --grid strict    # 10 runs per instance with the best configurations -> plots, statistics
    python main.py compare              # best 'strict' vs best 'extended' configurations -> results/comparison.*

Grids:
    strict    only what the assignment prescribes (one-point crossover, pm 0.5-1 %) - the main result
    extended  additional experiment that also tries other crossover types and pm = 1/D
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
from experiments.tuning import GRIDS, parameter_effects, tune, unique_configs
from ga import GAConfig
from problems import make_problem

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

PROBLEM_NAMES = ("onemax", "leading_ones")
DIMENSIONS = (10, 30, 100)
INSTANCES = [(name, dim) for name in PROBLEM_NAMES for dim in DIMENSIONS]

# Tuning uses different seeds than the final runs, so the reported statistics are
# not measured on the very runs the configuration was selected on.
TUNING_SEED_OFFSET = 1_000
TOP_CONFIGS_SHOWN = 5

GRID_LABELS = {"strict": "dle zadání", "extended": "rozšířené ladění"}
SELECTION_LABELS = {"roulette": "ruletová selekce", "rank": "pořadová selekce"}
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


def results_dir(grid_name: str) -> Path:
    return RESULTS_ROOT / grid_name


def instance_title(name: str, dim: int) -> str:
    return f"{make_problem(name, dim).title} {dim}D"


def cmd_tune(args: argparse.Namespace, grid_name: str) -> None:
    grid, out = GRIDS[grid_name], results_dir(grid_name)
    n_configs = sum(len(unique_configs(dim, grid)) for _, dim in INSTANCES)
    print(f"[{grid_name}] grid search: {n_configs} (instance, configuration) pairs x {args.tuning_runs} runs")
    start = time.perf_counter()
    ranking = tune(INSTANCES, grid, args.tuning_runs, args.seed + TUNING_SEED_OFFSET, args.jobs)
    print(f"finished in {time.perf_counter() - start:.0f} s\n")

    best_configs, top_tables = {}, []
    for (name, dim), results in ranking.items():
        iid = instance_id(name, dim)
        rows = [result.to_row() for result in results]
        write_csv(rows, out / "tuning" / f"{iid}.csv")

        winner = results[0]
        best_configs[iid] = {
            "problem": name,
            "dim": dim,
            "config": winner.config.to_dict(),
            "mean_final": winner.mean_final,
            "auc": winner.auc,
        }
        top_tables.append(f"### {instance_title(name, dim)}\n\n" + markdown_table(rows[:TOP_CONFIGS_SHOWN], TUNING_COLUMNS))
        print(f"{iid:<17} {winner.config.describe():<60} mean={winner.mean_final:.2f}")

    (out / "best_configs.json").write_text(json.dumps(best_configs, indent=2) + "\n")
    (out / "tuning" / "top_configs.md").write_text("\n".join(top_tables))

    effects = parameter_effects(ranking, grid)
    write_csv(effects, out / "tuning" / "parameter_effects.csv")
    effect_columns = {"parameter": "Parametr", "value": "Hodnota"}
    effect_columns |= {instance_id(name, dim): instance_title(name, dim) for name, dim in INSTANCES}
    readable = [{**row, "parameter": TUNING_COLUMNS[row["parameter"]]} for row in effects]
    (out / "tuning" / "parameter_effects.md").write_text(markdown_table(readable, effect_columns))
    print()


def load_best_configs(grid_name: str, required: bool = False) -> dict[str, GAConfig]:
    path = results_dir(grid_name) / "best_configs.json"
    if not path.exists():
        if required:
            raise SystemExit(f"{path} not found, run `tune --grid {grid_name}` first")
        print(f"{path} not found, using the default configuration (run `tune --grid {grid_name}` first)")
        return {}
    data = json.loads(path.read_text())
    return {iid: GAConfig.from_dict(entry["config"]) for iid, entry in data.items()}


def cmd_run(args: argparse.Namespace, grid_name: str) -> None:
    out = results_dir(grid_name)
    configs = load_best_configs(grid_name)
    stats_rows, overview, selection_comparison = [], [], []

    print(f"[{grid_name}] final runs")
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
        save_convergence_plot(histories, problem, config.describe(), out / "plots" / f"{iid}.png")
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

    write_csv(stats_rows, out / "stats.csv")
    (out / "stats.md").write_text(markdown_table(stats_rows, STATS_COLUMNS))
    save_overview_plot(overview, out / "plots" / "overview.png")
    save_comparison_plot(
        selection_comparison,
        "Ruletová vs. pořadová selekce (ostatní parametry podle nejlepší konfigurace)",
        out / "plots" / "selection_comparison.png",
    )
    print(f"results written to {out}\n")


def cmd_compare(args: argparse.Namespace) -> None:
    """Best configuration of each grid, run on the same seeds as the final experiments."""
    configs = {grid_name: load_best_configs(grid_name, required=True) for grid_name in GRID_LABELS}
    entries, rows = [], []

    for name, dim in INSTANCES:
        problem = make_problem(name, dim)
        iid = instance_id(name, dim)
        variants = {
            GRID_LABELS[grid_name]: run_many(configs[grid_name][iid], problem, args.runs, args.seed)
            for grid_name in GRID_LABELS
        }
        entries.append((problem, variants))

        row = {"problem": problem.title, "dim": dim}
        for grid_name, histories in zip(GRID_LABELS, variants.values()):
            stats = summarize(histories, problem.optimum)
            row[f"{grid_name}_mean"] = stats.mean
            row[f"{grid_name}_success"] = f"{stats.successes}/{stats.runs}"
            row[f"{grid_name}_evals"] = stats.mean_evals_to_optimum
        rows.append(row)

    columns = {"problem": "Úloha", "dim": "D"}
    for grid_name, label in GRID_LABELS.items():
        columns |= {
            f"{grid_name}_mean": f"Průměr ({label})",
            f"{grid_name}_success": f"Optimum ({label})",
            f"{grid_name}_evals": f"Ohodnocení do optima ({label})",
        }
    write_csv(rows, RESULTS_ROOT / "comparison.csv")
    (RESULTS_ROOT / "comparison.md").write_text(markdown_table(rows, columns))
    save_comparison_plot(
        entries,
        "Nejlepší nastavení dle zadání vs. z rozšířeného ladění",
        RESULTS_ROOT / "comparison.png",
    )
    print(f"comparison written to {RESULTS_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tune", "run", "compare", "all"])
    parser.add_argument(
        "--grid", choices=list(GRIDS), default="strict",
        help="parameter grid for `tune` and `run` (default: strict; `all` uses both)",
    )
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
    grid_names = list(GRIDS) if args.command == "all" else [args.grid]
    if args.command in ("tune", "all"):
        for grid_name in grid_names:
            cmd_tune(args, grid_name)
    if args.command in ("run", "all"):
        for grid_name in grid_names:
            cmd_run(args, grid_name)
    if args.command in ("compare", "all"):
        cmd_compare(args)


if __name__ == "__main__":
    main()
