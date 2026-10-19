import numpy as np
import pytest

from problems import PROBLEMS, Rosenbrock, Schwefel, Sphere, make_problem


def test_sphere_minimum_at_origin():
    assert Sphere(10).evaluate(np.zeros((1, 10)))[0] == 0.0


def test_rosenbrock_minimum_at_ones():
    assert Rosenbrock(10).evaluate(np.ones((1, 10)))[0] == pytest.approx(0.0)


def test_schwefel_minimum_at_known_point():
    optimum = np.full((1, 10), 420.9687)
    assert Schwefel(10).evaluate(optimum)[0] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_values_are_above_the_optimum(name, rng):
    problem = make_problem(name, 10)
    points = rng.uniform(problem.lo, problem.hi, size=(100, 10))
    values = problem.evaluate(points)
    assert values.shape == (100,)
    assert np.all(values >= problem.optimum - 1e-9)


def test_make_problem_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_problem("ackley", 10)
