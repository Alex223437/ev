import numpy as np
import pytest

from ga.crossover import CROSSOVERS, one_point, recombine, two_point
from ga.mutation import bit_flip_mutation
from ga.selection import rank_selection, roulette_selection


# --- crossover ---------------------------------------------------------------

def test_one_point_swaps_a_nonempty_proper_prefix(rng):
    mask = one_point(1000, 10, rng)
    prefix_lengths = mask.sum(axis=1)
    assert prefix_lengths.min() >= 1 and prefix_lengths.max() <= 9
    # Mask is a prefix: once False, it stays False.
    assert np.all(np.diff(mask.astype(int), axis=1) <= 0)


def test_two_point_swaps_one_inner_segment(rng):
    mask = two_point(1000, 10, rng)
    assert np.all(mask.sum(axis=1) >= 1)
    assert not mask[:, 0].any()
    # Exactly one contiguous block of True per row.
    edges = np.diff(np.pad(mask.astype(int), ((0, 0), (1, 1))), axis=1)
    assert np.all((edges == 1).sum(axis=1) == 1)


def test_one_point_example_from_assignment():
    a = np.array([[0, 0, 0, 0, 0]], dtype=np.uint8)
    b = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    mask = np.array([[True, True, False, False, False]])  # cut after the 2nd bit
    child_a, child_b = recombine(a, b, mask)
    assert child_a.tolist() == [[1, 1, 0, 0, 0]]
    assert child_b.tolist() == [[0, 0, 1, 1, 1]]


@pytest.mark.parametrize("name", sorted(CROSSOVERS))
def test_crossover_preserves_genes_per_position(name, rng):
    a = rng.integers(0, 2, size=(50, 20), dtype=np.uint8)
    b = rng.integers(0, 2, size=(50, 20), dtype=np.uint8)
    child_a, child_b = recombine(a, b, CROSSOVERS[name](50, 20, rng))
    # Genes are only exchanged between the two parents, never created or lost.
    assert np.array_equal(child_a + child_b, a + b)


# --- mutation ----------------------------------------------------------------

def test_mutation_with_zero_rate_is_identity(rng):
    population = rng.integers(0, 2, size=(20, 30), dtype=np.uint8)
    assert np.array_equal(bit_flip_mutation(population, 0.0, rng), population)


def test_mutation_with_full_rate_inverts_everything(rng):
    population = rng.integers(0, 2, size=(20, 30), dtype=np.uint8)
    assert np.array_equal(bit_flip_mutation(population, 1.0, rng), 1 - population)


def test_mutation_flips_expected_share_of_bits(rng):
    population = np.zeros((1000, 100), dtype=np.uint8)
    mutated = bit_flip_mutation(population, 0.01, rng)
    assert mutated.dtype == np.uint8
    assert mutated.mean() == pytest.approx(0.01, rel=0.1)


# --- selection ---------------------------------------------------------------

def test_roulette_is_proportional_to_fitness(rng):
    picks = roulette_selection(np.array([0, 1, 3]), 40_000, rng)
    shares = np.bincount(picks, minlength=3) / len(picks)
    assert shares == pytest.approx([0, 0.25, 0.75], abs=0.01)


def test_roulette_falls_back_to_uniform_for_zero_fitness(rng):
    picks = roulette_selection(np.zeros(4), 40_000, rng)
    assert np.bincount(picks, minlength=4) / len(picks) == pytest.approx([0.25] * 4, abs=0.01)


def test_rank_selection_is_proportional_to_rank(rng):
    # Ranks 1, 2, 3 regardless of the fitness scale.
    picks = rank_selection(np.array([5, 1000, 7]), 60_000, rng)
    shares = np.bincount(picks, minlength=3) / len(picks)
    assert shares == pytest.approx([1 / 6, 3 / 6, 2 / 6], abs=0.01)


def test_rank_selection_gives_ties_equal_chance(rng):
    # Two individuals tied for ranks 1-2 share rank 1.5 each; the best keeps rank 3.
    picks = rank_selection(np.array([2, 2, 9]), 60_000, rng)
    shares = np.bincount(picks, minlength=3) / len(picks)
    assert shares == pytest.approx([0.25, 0.25, 0.5], abs=0.01)
