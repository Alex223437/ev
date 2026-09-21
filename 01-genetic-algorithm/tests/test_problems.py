import numpy as np
import pytest

from problems import LeadingOnes, OneMax, make_problem


def bits(*rows: str) -> np.ndarray:
    return np.array([[int(c) for c in row] for row in rows], dtype=np.uint8)


def test_onemax_counts_ones():
    population = bits("0000", "1010", "1111", "0111")
    assert OneMax(4).evaluate(population).tolist() == [0, 2, 4, 3]


def test_leading_ones_counts_prefix():
    population = bits("0000", "1010", "1111", "0111", "1101")
    assert LeadingOnes(4).evaluate(population).tolist() == [0, 1, 4, 0, 2]


def test_optimum_is_dimension():
    assert OneMax(30).optimum == 30
    assert LeadingOnes(100).optimum == 100


def test_make_problem_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_problem("sphere", 10)
