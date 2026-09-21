from experiments.tuning import (
    GRIDS,
    TuningResult,
    effective_key,
    evaluate_config,
    grid_configs,
    parameter_effects,
    unique_configs,
)
from ga import GAConfig

SMALL_GRID = {
    "pop_size": [5, 10],
    "elite_ratio": [0.1, 0.2],
    "selection": ["rank"],
    "crossover": ["one_point"],
    "crossover_rate": [1.0],
    "mutation_rate": [0.01],
}


EXTENDED = GRIDS["extended"]


def test_strict_grid_stays_within_the_assignment():
    for config in grid_configs(GRIDS["strict"]):
        assert config.crossover == "one_point"
        assert 0.005 <= config.mutation_rate <= 0.01
        assert 0.1 <= config.elite_ratio <= 0.2


def test_unique_configs_have_no_equivalents():
    for grid in GRIDS.values():
        for dim in (10, 30, 100):
            keys = [effective_key(config, dim) for config in unique_configs(dim, grid)]
            assert len(keys) == len(set(keys))


def test_mutation_rate_equal_to_one_over_d_is_evaluated_once():
    # pm = 0.01 and pm = 1/D are the same thing for D = 100.
    assert len(unique_configs(100, EXTENDED)) < len(unique_configs(30, EXTENDED))
    assert {config.mutation_rate_for(100) for config in unique_configs(100, EXTENDED)} == {0.005, 0.01}


def test_unique_configs_cover_every_parameter_value():
    for grid in GRIDS.values():
        configs = unique_configs(30, grid)
        for name, values in grid.items():
            assert {getattr(config, name) for config in configs} == set(values)


def test_evaluate_config_summarizes_runs():
    result = evaluate_config("onemax", 10, GAConfig(pop_size=10), runs=3, seed=0)
    assert result.runs == 3
    assert result.mean_final == 10 and result.successes == 3
    assert 0 < result.auc <= 1


def test_parameter_effects_count_equivalent_configurations_for_every_value():
    # For N = 5 both elite ratios give one elite, so only one of them is evaluated;
    # its result must still count towards both 0.1 and 0.2.
    evaluated = unique_configs(10, SMALL_GRID)
    assert len(evaluated) == len(grid_configs(SMALL_GRID)) - 1
    auc = {5: 0.9, 10: 0.5}
    results = [
        TuningResult("onemax", 10, config, mean_final=10, std_final=0, successes=1, runs=1, auc=auc[config.pop_size])
        for config in evaluated
    ]
    rows = parameter_effects({("onemax", 10): results}, SMALL_GRID)
    elite_rows = {row["value"]: row["onemax_10D"] for row in rows if row["parameter"] == "elite_ratio"}
    assert elite_rows == {0.1: 0.7, 0.2: 0.7}
    # Parameters with a single value in the grid have no effect to show.
    assert {row["parameter"] for row in rows} == {"pop_size", "elite_ratio"}
