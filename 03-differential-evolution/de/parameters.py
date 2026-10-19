"""Control of F and CR: fixed values, or the self-adaptation used by jDE."""

from abc import ABC, abstractmethod

import numpy as np


class ParameterControl(ABC):
    """Supplies the F and CR of every individual in one generation."""

    @abstractmethod
    def propose(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """F and CR to use for each individual's trial vector."""

    @abstractmethod
    def accept(self, success: np.ndarray) -> None:
        """Tell which trial vectors won; losers keep their previous parameters."""

    @abstractmethod
    def means(self) -> tuple[float, float]:
        """Mean F and CR in the population, for logging the adaptation."""


class FixedParameters(ParameterControl):
    """Classic DE: one F and one CR for the whole run."""

    def __init__(self, np_size: int, f: float, cr: float) -> None:
        self._f = np.full(np_size, float(f))
        self._cr = np.full(np_size, float(cr))

    def propose(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return self._f, self._cr

    def accept(self, success: np.ndarray) -> None:
        return None

    def means(self) -> tuple[float, float]:
        return float(self._f[0]), float(self._cr[0])


class SelfAdaptiveParameters(ParameterControl):
    """jDE (Brest, 2006): every individual carries its own F and CR.

    Before the trial vector is built, both values are re-sampled with probability
    tau1 resp. tau2. A successful trial vector passes its (possibly re-sampled)
    parameters on to the next generation, an unsuccessful one leaves the parent's
    original values in place - successful settings therefore spread through the
    population on their own.
    """

    def __init__(
        self, np_size: int, f: float, cr: float, tau1: float, tau2: float, f_low: float, f_high: float
    ) -> None:
        self._f = np.full(np_size, float(f))
        self._cr = np.full(np_size, float(cr))
        self.tau1, self.tau2 = tau1, tau2
        self.f_low, self.f_high = f_low, f_high
        self._proposed: tuple[np.ndarray, np.ndarray] | None = None

    def propose(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._f)
        sampled_f = self.f_low + rng.random(n) * (self.f_high - self.f_low)
        sampled_cr = rng.random(n)
        f = np.where(rng.random(n) < self.tau1, sampled_f, self._f)
        cr = np.where(rng.random(n) < self.tau2, sampled_cr, self._cr)
        self._proposed = (f, cr)
        return f, cr

    def accept(self, success: np.ndarray) -> None:
        if self._proposed is None:
            raise RuntimeError("accept() called before propose()")
        f, cr = self._proposed
        self._f = np.where(success, f, self._f)
        self._cr = np.where(success, cr, self._cr)
        self._proposed = None

    def means(self) -> tuple[float, float]:
        return float(self._f.mean()), float(self._cr.mean())
