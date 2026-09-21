import numpy as np
import pytest

from ga import BudgetedEvaluator, GAConfig, GeneticAlgorithm
from problems import LeadingOnes, OneMax


@pytest.mark.parametrize("pop_size", [7, 20, 50])
@pytest.mark.parametrize("elite_ratio", [0.0, 0.1, 0.2])
def test_run_spends_exactly_the_budget(pop_size, elite_ratio, rng):
    evaluations = []

    class CountingOneMax(OneMax):
        def evaluate(self, population):
            evaluations.append(len(population))
            return super().evaluate(population)

    config = GAConfig(pop_size=pop_size, elite_ratio=elite_ratio)
    result = GeneticAlgorithm(config).run(CountingOneMax(10), 1000, rng)

    assert sum(evaluations) == 1000
    assert len(result.history) == 1000
    # Elites are never re-evaluated, so each generation costs pop_size - n_elite evaluations at most.
    assert max(evaluations[1:]) <= pop_size - config.n_elite


@pytest.mark.parametrize("problem", [OneMax(30), LeadingOnes(30)])
def test_history_is_monotone_and_ends_with_best(problem, rng):
    result = GeneticAlgorithm(GAConfig()).run(problem, 3000, rng)
    assert np.all(np.diff(result.history) >= 0)
    assert result.history[-1] == result.best_fitness
    assert problem.evaluate(result.best_solution[None, :])[0] == result.best_fitness
    assert result.best_fitness <= problem.optimum


def test_same_seed_gives_same_run():
    ga = GeneticAlgorithm(GAConfig(selection="roulette", crossover="uniform"))
    first = ga.run(LeadingOnes(30), 3000, np.random.default_rng(7))
    second = ga.run(LeadingOnes(30), 3000, np.random.default_rng(7))
    assert np.array_equal(first.history, second.history)


def test_solves_small_onemax(rng):
    result = GeneticAlgorithm(GAConfig(pop_size=20)).run(OneMax(10), 1000, rng)
    assert result.best_fitness == 10


def test_evaluator_refuses_to_exceed_budget(rng):
    evaluator = BudgetedEvaluator(OneMax(10), budget=5)
    evaluator(rng.integers(0, 2, size=(5, 10), dtype=np.uint8))
    assert evaluator.exhausted
    with pytest.raises(RuntimeError):
        evaluator(rng.integers(0, 2, size=(1, 10), dtype=np.uint8))


def test_elite_count():
    assert GAConfig(pop_size=20, elite_ratio=0.1).n_elite == 2
    assert GAConfig(pop_size=20, elite_ratio=0.0).n_elite == 0
    # Enabled elitism always keeps the best individual...
    assert GAConfig(pop_size=5, elite_ratio=0.1).n_elite == 1
    # ...but never takes the whole population.
    assert GAConfig(pop_size=2, elite_ratio=0.9).n_elite == 1


def test_invalid_operator_name_is_rejected():
    with pytest.raises(ValueError):
        GeneticAlgorithm(GAConfig(selection="tournament"))
