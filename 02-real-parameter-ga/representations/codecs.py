"""Binary codecs: how a real number is stored in a fixed number of bits.

Every codec uses the same chromosome length - 32 bits per dimension - so the three
representations differ only in the meaning of those bits. Bits are stored as a uint8
matrix (n, dim * BITS_PER_DIM) with the most significant bit first.

Decoding always returns values inside the problem domain: patterns that fall outside
it (or are not a valid number at all) are clamped to the nearest bound.
"""

from abc import ABC, abstractmethod

import numpy as np

BITS_PER_DIM = 32


def bits_to_words(bits: np.ndarray, dim: int) -> np.ndarray:
    """(n, dim * 32) bits -> (n, dim) uint32 words, most significant bit first."""
    weights = (np.uint32(1) << np.arange(BITS_PER_DIM - 1, -1, -1, dtype=np.uint32))
    return (bits.reshape(len(bits), dim, BITS_PER_DIM).astype(np.uint32) * weights).sum(
        axis=2, dtype=np.uint32
    )


def words_to_bits(words: np.ndarray) -> np.ndarray:
    """(n, dim) uint32 words -> (n, dim * 32) bits."""
    shifts = np.arange(BITS_PER_DIM - 1, -1, -1, dtype=np.uint32)
    bits = (words[:, :, None] >> shifts) & np.uint32(1)
    return bits.reshape(len(words), -1).astype(np.uint8)


class Codec(ABC):
    """Translation between real vectors and their bit strings."""

    name: str = "codec"
    title: str = "Codec"

    def __init__(self, problem) -> None:
        self.problem = problem
        self.dim = problem.dim
        self.length = problem.dim * BITS_PER_DIM

    @abstractmethod
    def encode(self, points: np.ndarray) -> np.ndarray:
        """(n, dim) real values -> (n, length) bits."""

    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """(n, length) bits -> (n, dim) real values inside the domain."""

    def _clip(self, points: np.ndarray) -> np.ndarray:
        return np.clip(points, self.problem.lo, self.problem.hi)

    def describe(self) -> str:
        return self.title


class Ieee754Codec(Codec):
    """IEEE 754 binary32: 1 sign bit, 8 exponent bits, 23 mantissa bits.

    A flip in the exponent multiplies or divides the value by a power of two, so a
    single bit has a wildly different weight depending on where it sits. Patterns that
    decode to NaN or infinity are clamped to the domain bounds.
    """

    name = "ieee754"
    title = "IEEE 754"

    def encode(self, points: np.ndarray) -> np.ndarray:
        words = np.ascontiguousarray(points, dtype=np.float32).view(np.uint32)
        return words_to_bits(words)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        words = np.ascontiguousarray(bits_to_words(bits, self.dim))
        with np.errstate(invalid="ignore"):  # signaling NaN patterns raise on conversion
            values = words.view(np.float32).astype(np.float64)
        # NaN has no sensible position in the domain; infinities keep their sign.
        values = np.nan_to_num(values, nan=self.problem.hi, posinf=self.problem.hi, neginf=self.problem.lo)
        return self._clip(values)


class FixedPointCodec(Codec):
    """Fixed-point: 1 sign bit, a fixed number of integer bits and the rest for the fraction.

    The number of integer bits is the smallest that covers the domain, so the remaining
    bits give the finest resolution this domain allows.
    """

    name = "fixed_point"
    title = "Fixed-point"

    def __init__(self, problem) -> None:
        super().__init__(problem)
        limit = max(abs(problem.lo), abs(problem.hi))
        self.int_bits = max(1, int(np.ceil(np.log2(limit + 1.0))))
        self.frac_bits = BITS_PER_DIM - 1 - self.int_bits
        self.scale = float(2**self.frac_bits)
        self.max_magnitude = np.uint32(2**31 - 1)

    def encode(self, points: np.ndarray) -> np.ndarray:
        magnitude = np.minimum(np.round(np.abs(points) * self.scale), float(self.max_magnitude))
        words = magnitude.astype(np.uint32)
        words |= (points < 0).astype(np.uint32) << np.uint32(31)
        return words_to_bits(words)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        words = bits_to_words(bits, self.dim)
        magnitude = (words & self.max_magnitude).astype(np.float64) / self.scale
        values = np.where(words >> np.uint32(31) == 1, -magnitude, magnitude)
        return self._clip(values)

    def describe(self) -> str:
        return f"{self.title} (1 + {self.int_bits} + {self.frac_bits} bitů, krok {1 / self.scale:.2g})"


class BcdCodec(Codec):
    """Binary coded decimal: 4 bits per decimal digit, here 8 digits per dimension.

    The first nibble carries the sign (>= 8 means negative), the remaining 7 nibbles are
    decimal digits. Only 10 of the 16 nibble values are valid digits; the invalid ones
    are repaired modulo 10, which is why mutation in BCD is not symmetric.
    """

    name = "bcd"
    title = "BCD"
    nibble_weights = np.array([8, 4, 2, 1], dtype=np.uint32)

    def __init__(self, problem) -> None:
        super().__init__(problem)
        self.nibbles = BITS_PER_DIM // 4
        self.value_digits = self.nibbles - 1
        limit = max(abs(problem.lo), abs(problem.hi))
        self.int_digits = max(1, len(str(int(limit))))
        self.frac_digits = self.value_digits - self.int_digits
        self.scale = 10.0**self.frac_digits
        self.max_integer = 10**self.value_digits - 1

    def encode(self, points: np.ndarray) -> np.ndarray:
        integers = np.minimum(np.round(np.abs(points) * self.scale), float(self.max_integer)).astype(np.int64)
        powers = 10 ** np.arange(self.value_digits - 1, -1, -1, dtype=np.int64)
        digits = (integers[:, :, None] // powers) % 10
        sign = np.where(points < 0, 8, 0)[:, :, None]
        nibbles = np.concatenate([sign, digits], axis=2).astype(np.uint32)
        bits = (nibbles[:, :, :, None] >> np.arange(3, -1, -1, dtype=np.uint32)) & np.uint32(1)
        return bits.reshape(len(points), -1).astype(np.uint8)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        nibbles = (
            bits.reshape(len(bits), self.dim, self.nibbles, 4).astype(np.uint32) * self.nibble_weights
        ).sum(axis=3)
        digits = nibbles[:, :, 1:] % 10  # repair the six invalid nibble values
        powers = 10.0 ** np.arange(self.value_digits - 1, -1, -1)
        magnitude = (digits * powers).sum(axis=2) / self.scale
        values = np.where(nibbles[:, :, 0] >= 8, -magnitude, magnitude)
        return self._clip(values)

    def describe(self) -> str:
        return f"{self.title} ({self.int_digits} + {self.frac_digits} číslic, krok {1 / self.scale:.2g})"


CODECS: dict[str, type[Codec]] = {
    Ieee754Codec.name: Ieee754Codec,
    FixedPointCodec.name: FixedPointCodec,
    BcdCodec.name: BcdCodec,
}
