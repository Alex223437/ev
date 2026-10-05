import numpy as np
import pytest

from experiments.runner import build, run_many
from ga import BudgetedEvaluator, GAConfig, GeneticAlgorithm
from ga.selection import rank_selection, roulette_selection
from problems import make_problem
from representations import VARIANTS, make_representation


@pytest.mark.parametrize("variant", VARIANTS)
def test_run_spends_exactly_the_budget(variant, rng):
    problem = make_problem("sphere", 10)
    evaluations = []

    class CountingSphere(type(problem)):
        def evaluate(self, points):
            evaluations.append(len(points))
            return super().evaluate(points)

    config = GAConfig(pop_size=30, sigma_ratio=0.05)
    counting = CountingSphere(10)
    result = build(config, counting, variant).run(counting, 1000, rng)

    assert sum(evaluations) == 1000
    assert len(result.history) == 1000
    # Elites are never re-evaluated.
    assert max(evaluations[1:]) <= config.pop_size - config.n_elite


@pytest.mark.parametrize("variant", VARIANTS)
def test_history_is_non_increasing_and_matches_the_best_point(variant, rng):
    problem = make_problem("rosenbrock", 10)
    config = GAConfig(pop_size=20, sigma_ratio=0.05)
    result = build(config, problem, variant).run(problem, 2000, rng)
    assert np.all(np.diff(result.history) <= 0)
    assert result.history[-1] == result.best_value
    assert problem.evaluate(result.best_point[None, :])[0] == pytest.approx(result.best_value)
    assert np.all((result.best_point >= problem.lo) & (result.best_point <= problem.hi))


def test_same_seed_gives_same_run():
    problem = make_problem("schwefel", 10)
    config = GAConfig(pop_size=20, selection="roulette", mutation_rate=0.01)
    algorithm = build(config, problem, "bcd")
    first = algorithm.run(problem, 2000, np.random.default_rng(7))
    second = algorithm.run(problem, 2000, np.random.default_rng(7))
    assert np.array_equal(first.history, second.history)


def test_run_many_returns_one_history_per_run():
    problem = make_problem("sphere", 10)
    histories = run_many(GAConfig(pop_size=20), problem, "fixed_point", runs=3, seed=0, budget=500)
    assert histories.shape == (3, 500)
    # Different seeds give different runs.
    assert len({float(h[-1]) for h in histories}) > 1


def test_ga_improves_on_the_initial_population(rng):
    problem = make_problem("sphere", 10)
    config = GAConfig(pop_size=50, mutation_rate=0.01)
    result = build(config, problem, "fixed_point").run(problem, 5000, rng)
    assert result.best_value < result.history[49] / 100


def test_evaluator_refuses_to_exceed_budget(rng):
    problem = make_problem("sphere", 10)
    evaluator = BudgetedEvaluator(problem, budget=5)
    evaluator(rng.uniform(problem.lo, problem.hi, size=(5, 10)))
    assert evaluator.exhausted
    with pytest.raises(RuntimeError):
        evaluator(rng.uniform(problem.lo, problem.hi, size=(1, 10)))


def test_selection_prefers_lower_objective_values(rng):
    values = np.array([1.0, 5.0, 9.0])  # minimization: index 0 is the best
    for selection in (roulette_selection, rank_selection):
        shares = np.bincount(selection(values, 30_000, rng), minlength=3) / 30_000
        assert shares[0] > shares[1] > shares[2]


def test_rank_selection_is_proportional_to_rank(rng):
    picks = rank_selection(np.array([9.0, 1.0, 5.0]), 60_000, rng)
    shares = np.bincount(picks, minlength=3) / 60_000
    assert shares == pytest.approx([1 / 6, 3 / 6, 2 / 6], abs=0.01)


def test_roulette_falls_back_to_uniform_for_equal_values(rng):
    picks = roulette_selection(np.full(4, 7.0), 40_000, rng)
    assert np.bincount(picks, minlength=4) / 40_000 == pytest.approx([0.25] * 4, abs=0.01)


def test_elite_count():
    assert GAConfig(pop_size=20, elite_ratio=0.1).n_elite == 2
    assert GAConfig(pop_size=20, elite_ratio=0.0).n_elite == 0
    assert GAConfig(pop_size=5, elite_ratio=0.1).n_elite == 1
    assert GAConfig(pop_size=2, elite_ratio=0.9).n_elite == 1


def test_invalid_selection_name_is_rejected():
    problem = make_problem("sphere", 10)
    representation = make_representation("fixed_point", problem, 0.01)
    with pytest.raises(ValueError):
        GeneticAlgorithm(GAConfig(selection="tournament"), representation)
