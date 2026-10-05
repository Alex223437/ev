"""Control parameters of the genetic algorithm."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 50
    elite_ratio: float = 0.1
    selection: str = "rank"  # "roulette" | "rank"
    crossover_rate: float = 0.9
    mutation_rate: float = 0.01  # per bit (binary variants) or per gene (real variants)
    sigma_ratio: float | None = None  # gaussian mutation only, as a fraction of the domain width

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be at least 2, got {self.pop_size}")
        if not 0.0 <= self.elite_ratio < 1.0:
            raise ValueError(f"elite_ratio must be in [0, 1), got {self.elite_ratio}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if self.sigma_ratio is not None and self.sigma_ratio <= 0:
            raise ValueError(f"sigma_ratio must be positive, got {self.sigma_ratio}")

    @property
    def n_elite(self) -> int:
        """Number of individuals copied unchanged; at least one, and never the whole population."""
        if self.elite_ratio == 0:
            return 0
        return min(max(1, round(self.elite_ratio * self.pop_size)), self.pop_size - 1)

    def describe(self) -> str:
        sigma = "" if self.sigma_ratio is None else f", sigma={self.sigma_ratio:g}*rozsah"
        return (
            f"N={self.pop_size}, elite={self.elite_ratio:.0%}, {self.selection}, "
            f"pc={self.crossover_rate:g}, pm={self.mutation_rate:g}{sigma}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GAConfig":
        return cls(**data)
