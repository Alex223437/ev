"""Control parameters of differential evolution."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DEConfig:
    """Parameters of one DE variant.

    `f` and `cr` are the starting values; in the self-adaptive variant (jDE) every
    individual carries and evolves its own pair, so these are only the initial ones.
    """

    np_size: int = 50          # NP - population size
    f: float = 0.5             # F - mutation (scaling) constant
    cr: float = 0.8            # CR - crossover threshold
    strategy: str = "rand1"    # "rand1" | "best1"
    adaptive: bool = False     # jDE: self-adapting F and CR
    tau1: float = 0.1          # probability of re-sampling F (jDE)
    tau2: float = 0.1          # probability of re-sampling CR (jDE)
    f_low: float = 0.1         # lower bound of the sampled F (jDE)
    f_high: float = 0.9        # upper bound of the sampled F (jDE)

    def __post_init__(self) -> None:
        if self.np_size < 4:
            raise ValueError(f"np_size must be at least 4 (rand/1 needs 4 distinct individuals), got {self.np_size}")
        if self.f <= 0:
            raise ValueError(f"f must be positive, got {self.f}")
        if not 0.0 <= self.cr <= 1.0:
            raise ValueError(f"cr must be in [0, 1], got {self.cr}")
        if self.adaptive and not 0 < self.f_low <= self.f_high:
            raise ValueError(f"expected 0 < f_low <= f_high, got {self.f_low} and {self.f_high}")

    def describe(self) -> str:
        if self.adaptive:
            return (
                f"NP={self.np_size}, jDE (rand/1/bin), F0={self.f:g} z [{self.f_low:g}, {self.f_high:g}], "
                f"CR0={self.cr:g}, tau1=tau2={self.tau1:g}"
            )
        return f"NP={self.np_size}, {self.strategy}/bin, F={self.f:g}, CR={self.cr:g}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DEConfig":
        return cls(**data)
