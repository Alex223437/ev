import numpy as np
import pytest

from problems import make_problem
from representations import VARIANTS, make_representation


def build(variant, problem_name="schwefel", mutation_rate=0.1):
    problem = make_problem(problem_name, 10)
    return problem, make_representation(variant, problem, mutation_rate, sigma_ratio=0.05)


@pytest.mark.parametrize("variant", VARIANTS)
def test_random_population_decodes_into_the_domain(variant, rng):
    problem, representation = build(variant)
    points = representation.decode(representation.random_population(64, rng))
    assert points.shape == (64, 10)
    assert np.all((points >= problem.lo) & (points <= problem.hi))


@pytest.mark.parametrize("variant", VARIANTS)
def test_crossover_only_exchanges_parent_genes(variant, rng):
    _, representation = build(variant)
    population = representation.random_population(20, rng)
    a, b = population[:10], population[10:]
    child_a, child_b = representation.crossover(a, b, rng)
    # Nothing is created or lost: the two children together hold the parents' genes.
    assert np.array_equal(child_a + child_b, a + b)
    assert not np.array_equal(child_a, a)


def test_real_crossover_cuts_between_dimensions(rng):
    _, representation = build("real_gaussian")
    a = np.zeros((50, 10))
    b = np.ones((50, 10))
    child_a, _ = representation.crossover(a, b, rng)
    # Every gene comes from one parent unchanged, so the child is a prefix of ones.
    assert set(np.unique(child_a)) <= {0.0, 1.0}
    assert np.all(np.diff(child_a, axis=1) <= 0)
    assert np.all(child_a[:, 0] == 1.0)


def test_binary_mutation_flips_the_expected_share_of_bits(rng):
    _, representation = build("fixed_point", mutation_rate=0.01)
    population = np.zeros((200, representation.length), dtype=np.uint8)
    mutated = representation.mutate(population, rng)
    assert mutated.dtype == np.uint8
    assert mutated.mean() == pytest.approx(0.01, rel=0.1)


def test_gaussian_mutation_makes_local_steps_inside_the_domain(rng):
    problem, representation = build("real_gaussian", mutation_rate=1.0)
    population = np.zeros((2000, 10))
    mutated = representation.mutate(population, rng)
    assert np.all((mutated >= problem.lo) & (mutated <= problem.hi))
    # sigma = 0.05 * 1000 = 50 for Schwefel.
    assert mutated.std() == pytest.approx(0.05 * problem.width, rel=0.1)


def test_uniform_mutation_replaces_genes_from_the_whole_range(rng):
    problem, representation = build("real_uniform", mutation_rate=1.0)
    mutated = representation.mutate(np.zeros((2000, 10)), rng)
    assert mutated.mean() == pytest.approx(0.0, abs=0.05 * problem.width)
    assert mutated.std() == pytest.approx(problem.width / np.sqrt(12), rel=0.1)


def test_gaussian_mutation_requires_sigma():
    problem = make_problem("sphere", 10)
    with pytest.raises(ValueError):
        make_representation("real_gaussian", problem, 0.1, sigma_ratio=None)


def test_unknown_variant_is_rejected():
    problem = make_problem("sphere", 10)
    with pytest.raises(ValueError):
        make_representation("gray_code", problem, 0.1)
