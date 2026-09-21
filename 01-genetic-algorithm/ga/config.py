"""Control parameters of the genetic algorithm."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 50
    elite_ratio: float = 0.1
    selection: str = "rank"          # "roulette" | "rank"
    crossover: str = "one_point"     # "one_point" | "two_point" | "uniform"
    crossover_rate: float = 0.9
    mutation_rate: float | None = 0.01  # None means 1 / D

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be at least 2, got {self.pop_size}")
        if not 0.0 <= self.elite_ratio < 1.0:
            raise ValueError(f"elite_ratio must be in [0, 1), got {self.elite_ratio}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")
        if self.mutation_rate is not None and not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")

    @property
    def n_elite(self) -> int:
        """Number of individuals copied unchanged.

        With elitism enabled at least the best individual survives, and at least one
        slot is always left for offspring.
        """
        if self.elite_ratio == 0:
            return 0
        return min(max(1, round(self.elite_ratio * self.pop_size)), self.pop_size - 1)

    def mutation_rate_for(self, dim: int) -> float:
        return 1.0 / dim if self.mutation_rate is None else self.mutation_rate

    def describe(self) -> str:
        pm = "1/D" if self.mutation_rate is None else f"{self.mutation_rate:g}"
        return (
            f"N={self.pop_size}, elite={self.elite_ratio:.0%}, {self.selection}, "
            f"{self.crossover} (pc={self.crossover_rate:g}), pm={pm}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GAConfig":
        return cls(**data)
