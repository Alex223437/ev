from .base import Problem
from .rosenbrock import Rosenbrock
from .schwefel import Schwefel
from .sphere import Sphere

PROBLEMS: dict[str, type[Problem]] = {
    Sphere.name: Sphere,
    Rosenbrock.name: Rosenbrock,
    Schwefel.name: Schwefel,
}


def make_problem(name: str, dim: int = 10) -> Problem:
    try:
        return PROBLEMS[name](dim)
    except KeyError:
        raise ValueError(f"unknown problem {name!r}, choose from {sorted(PROBLEMS)}") from None


__all__ = ["PROBLEMS", "Problem", "Rosenbrock", "Schwefel", "Sphere", "make_problem"]
