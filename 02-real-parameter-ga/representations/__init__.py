"""The five compared GA variants: three binary encodings and two real-valued ones."""

from problems import Problem

from .base import Representation
from .binary import BinaryRepresentation, make_binary
from .codecs import BITS_PER_DIM, CODECS
from .real import RealRepresentation

VARIANTS = ("ieee754", "fixed_point", "bcd", "real_gaussian", "real_uniform")
NEEDS_SIGMA = ("real_gaussian",)


def make_representation(variant: str, problem: Problem, mutation_rate: float, sigma_ratio: float | None = None) -> Representation:
    if variant in CODECS:
        return make_binary(variant, problem, mutation_rate)
    if variant == "real_gaussian":
        return RealRepresentation(problem, "gaussian", mutation_rate, sigma_ratio)
    if variant == "real_uniform":
        return RealRepresentation(problem, "uniform", mutation_rate)
    raise ValueError(f"unknown variant {variant!r}, choose from {list(VARIANTS)}")


def variant_title(variant: str, problem: Problem) -> str:
    return make_representation(variant, problem, 0.01, 0.1).title


__all__ = [
    "BITS_PER_DIM",
    "CODECS",
    "NEEDS_SIGMA",
    "VARIANTS",
    "BinaryRepresentation",
    "RealRepresentation",
    "Representation",
    "make_representation",
    "variant_title",
]
