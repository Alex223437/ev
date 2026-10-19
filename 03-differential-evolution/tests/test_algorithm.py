import numpy as np
import pytest

from de import DEConfig, DifferentialEvolution, variant_config
from de.evaluator import BudgetedEvaluator
from experiments.runner import run_many
from problems import make_problem

VARIANTS = ("rand1", "best1", "jde")


@pytest.mark.parametrize("variant", VARIANTS)
def test_run_spends_exactly_the_budget(variant, rng):
    problem = make_problem("sphere", 10)
    evaluations = []

    class CountingSphere(type(problem)):
        def evaluate(self, points):
            evaluations.append(len(points))
            return super().evaluate(points)

    config = variant_config(variant, np_size=30)
    counting = CountingSphere(10)
    result = DifferentialEvolution(config).run(counting, 1000, rng)

    assert sum(evaluations) == 1000
    assert len(result.history) == 1000
    # One trial vector per individual and generation, never more.
    assert max(evaluations) <= 30


@pytest.mark.parametrize("variant", VARIANTS)
def test_history_is_non_increasing_and_matches_the_best_point(variant, rng):
    problem = make_problem("rosenbrock", 10)
    result = DifferentialEvolution(variant_config(variant, np_size=20)).run(problem, 2000, rng)
    assert np.all(np.diff(result.history) <= 0)
    assert result.history[-1] == result.best_value
    assert problem.evaluate(result.best_point[None, :])[0] == pytest.approx(result.best_value)
    assert np.all((result.best_point >= problem.lo) & (result.best_point <= problem.hi))


@pytest.mark.parametrize("variant", VARIANTS)
def test_trial_vectors_stay_inside_the_domain(variant, rng):
    problem = make_problem("schwefel", 10)
    result = DifferentialEvolution(variant_config(variant, np_size=20)).run(problem, 2000, rng)
    assert np.all((result.best_point >= problem.lo) & (result.best_point <= problem.hi))


def test_same_seed_gives_same_run():
    problem = make_problem("schwefel", 10)
    algorithm = DifferentialEvolution(variant_config("jde", np_size=20))
    first = algorithm.run(problem, 2000, np.random.default_rng(7))
    second = algorithm.run(problem, 2000, np.random.default_rng(7))
    assert np.array_equal(first.history, second.history)
    assert np.array_equal(first.mean_f, second.mean_f)


@pytest.mark.parametrize("variant", VARIANTS)
def test_de_solves_the_sphere(variant, rng):
    problem = make_problem("sphere", 10)
    result = DifferentialEvolution(variant_config(variant)).run(problem, 10_000, rng)
    assert result.best_value < 1e-5


def test_classic_variants_keep_their_parameters_constant(rng):
    problem = make_problem("sphere", 10)
    result = DifferentialEvolution(variant_config("rand1", np_size=20)).run(problem, 2000, rng)
    assert np.all(result.mean_f == 0.5) and np.all(result.mean_cr == 0.8)


def test_jde_adapts_its_parameters(rng):
    problem = make_problem("schwefel", 10)
    result = DifferentialEvolution(variant_config("jde", np_size=20)).run(problem, 5000, rng)
    assert result.mean_f[0] != result.mean_f[-1] or result.mean_cr[0] != result.mean_cr[-1]
    assert np.all((result.mean_f >= 0.1) & (result.mean_f <= 0.9))
    assert np.all((result.mean_cr >= 0.0) & (result.mean_cr <= 1.0))


def test_run_many_returns_histories_and_parameter_traces():
    problem = make_problem("sphere", 10)
    histories, mean_f, mean_cr = run_many(variant_config("jde", np_size=20), problem, runs=3, seed=0, budget=1000)
    assert histories.shape == (3, 1000)
    # The initial population costs 20 evaluations, every generation another 20.
    assert mean_f.shape == mean_cr.shape == (3, (1000 - 20) // 20)
    assert len({float(h[-1]) for h in histories}) > 1  # different seeds, different runs


def test_evaluator_refuses_to_exceed_budget(rng):
    problem = make_problem("sphere", 10)
    evaluator = BudgetedEvaluator(problem, budget=5)
    evaluator(rng.uniform(problem.lo, problem.hi, size=(5, 10)))
    assert evaluator.exhausted
    with pytest.raises(RuntimeError):
        evaluator(rng.uniform(problem.lo, problem.hi, size=(1, 10)))


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        DEConfig(np_size=3)  # rand/1 needs the individual plus three donors
    with pytest.raises(ValueError):
        DEConfig(cr=1.5)
    with pytest.raises(ValueError):
        DEConfig(f=0.0)
    with pytest.raises(ValueError):
        variant_config("current-to-best")
    with pytest.raises(ValueError):
        DifferentialEvolution(DEConfig(strategy="rand2"))
