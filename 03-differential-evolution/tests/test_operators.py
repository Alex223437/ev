import numpy as np
import pytest

from de.crossover import binomial_crossover
from de.parameters import FixedParameters, SelfAdaptiveParameters
from de.strategies import STRATEGIES, best1, distinct_indices, rand1


# --- donor selection ---------------------------------------------------------

def test_donors_are_distinct_and_never_the_individual_itself(rng):
    for count in (2, 3):
        donors = distinct_indices(40, count, rng)
        assert donors.shape == (40, count)
        assert np.all(donors != np.arange(40)[:, None])
        for row in donors:
            assert len(set(row.tolist())) == count


def test_donors_cover_the_population_uniformly(rng):
    counts = np.bincount(distinct_indices(10, 3, rng)[:, 0], minlength=10)
    for _ in range(300):
        counts += np.bincount(distinct_indices(10, 3, rng)[:, 0], minlength=10)
    shares = counts / counts.sum()
    assert shares == pytest.approx(np.full(10, 0.1), abs=0.02)


# --- mutation ----------------------------------------------------------------

def test_rand1_matches_the_formula():
    population = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0], [10.0, 10.0]])
    donors = np.array([[1, 2, 3]])
    f = np.array([0.5])
    # x_r1 + F * (x_r2 - x_r3) = [1, 2] + 0.5 * ([3, 5] - [10, 10])
    assert rand1(population, population[0], donors, f)[0].tolist() == pytest.approx([-2.5, -0.5])


def test_best1_uses_the_best_individual():
    population = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]])
    best = np.array([9.0, 9.0])
    donors = np.array([[1, 2]])
    # x_best + F * (x_r1 - x_r2) = [9, 9] + 0.5 * ([1, 2] - [3, 5])
    assert best1(population, best, donors, np.array([0.5]))[0].tolist() == pytest.approx([8.0, 7.5])


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategies_produce_one_mutant_per_individual(name, rng):
    strategy = STRATEGIES[name]
    population = rng.normal(size=(20, 5))
    donors = distinct_indices(20, strategy.n_donors, rng)
    mutants = strategy.mutate(population, population[0], donors, np.full(20, 0.5))
    assert mutants.shape == population.shape


# --- crossover ---------------------------------------------------------------

def test_crossover_always_takes_at_least_one_gene_from_the_mutant(rng):
    parents = np.zeros((500, 8))
    mutants = np.ones((500, 8))
    trials = binomial_crossover(parents, mutants, np.zeros(500), rng)
    # CR = 0: exactly the guaranteed j_rand gene comes from the mutant.
    assert np.all(trials.sum(axis=1) == 1)


def test_crossover_with_cr_one_copies_the_mutant(rng):
    parents = np.zeros((50, 8))
    mutants = np.ones((50, 8))
    assert np.array_equal(binomial_crossover(parents, mutants, np.ones(50), rng), mutants)


def test_crossover_takes_about_cr_share_of_genes(rng):
    parents = np.zeros((2000, 20))
    mutants = np.ones((2000, 20))
    trials = binomial_crossover(parents, mutants, np.full(2000, 0.5), rng)
    assert trials.mean() == pytest.approx(0.5 + 0.5 / 20, rel=0.05)  # CR plus the guaranteed gene


def test_crossover_mixes_per_individual_cr(rng):
    parents = np.zeros((2, 100))
    mutants = np.ones((2, 100))
    trials = binomial_crossover(parents, mutants, np.array([0.0, 1.0]), rng)
    assert trials[0].sum() == 1 and trials[1].sum() == 100


# --- parameter control -------------------------------------------------------

def test_fixed_parameters_never_change(rng):
    parameters = FixedParameters(10, 0.5, 0.8)
    f, cr = parameters.propose(rng)
    parameters.accept(np.ones(10, dtype=bool))
    assert np.all(f == 0.5) and np.all(cr == 0.8)
    assert parameters.means() == (0.5, 0.8)


def test_jde_samples_f_inside_its_range(rng):
    parameters = SelfAdaptiveParameters(200, f=0.5, cr=0.9, tau1=1.0, tau2=1.0, f_low=0.1, f_high=0.9)
    f, cr = parameters.propose(rng)
    assert np.all((f >= 0.1) & (f <= 0.9))
    assert np.all((cr >= 0.0) & (cr <= 1.0))
    assert f.std() > 0.1 and cr.std() > 0.1  # tau = 1 re-samples everyone


def test_jde_keeps_parameters_of_successful_individuals_only(rng):
    parameters = SelfAdaptiveParameters(2, f=0.5, cr=0.9, tau1=1.0, tau2=1.0, f_low=0.1, f_high=0.9)
    f, cr = parameters.propose(rng)
    parameters.accept(np.array([True, False]))
    assert parameters._f[0] == f[0] and parameters._cr[0] == cr[0]  # winner propagates
    assert parameters._f[1] == 0.5 and parameters._cr[1] == 0.9     # loser keeps the old pair


def test_jde_does_not_change_parameters_without_proposal():
    parameters = SelfAdaptiveParameters(4, f=0.5, cr=0.9, tau1=0.1, tau2=0.1, f_low=0.1, f_high=0.9)
    with pytest.raises(RuntimeError):
        parameters.accept(np.ones(4, dtype=bool))
