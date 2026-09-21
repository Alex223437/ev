from .base import Problem
from .leading_ones import LeadingOnes
from .onemax import OneMax

PROBLEMS: dict[str, type[Problem]] = {
    OneMax.name: OneMax,
    LeadingOnes.name: LeadingOnes,
}


def make_problem(name: str, dim: int) -> Problem:
    try:
        return PROBLEMS[name](dim)
    except KeyError:
        raise ValueError(f"unknown problem {name!r}, choose from {sorted(PROBLEMS)}") from None


__all__ = ["PROBLEMS", "LeadingOnes", "OneMax", "Problem", "make_problem"]
