import numpy as np
import pytest

from problems import make_problem
from representations.codecs import BITS_PER_DIM, CODECS, BcdCodec, Ieee754Codec

PROBLEM_NAMES = ("sphere", "rosenbrock", "schwefel")
RESOLUTION = {"ieee754": 1e-4, "fixed_point": 1e-6, "bcd": 1e-3}


@pytest.mark.parametrize("problem_name", PROBLEM_NAMES)
@pytest.mark.parametrize("codec_name", sorted(CODECS))
def test_chromosome_is_32_bits_per_dimension(problem_name, codec_name, rng):
    problem = make_problem(problem_name, 10)
    codec = CODECS[codec_name](problem)
    bits = codec.encode(rng.uniform(problem.lo, problem.hi, size=(4, 10)))
    assert bits.shape == (4, 10 * BITS_PER_DIM)
    assert bits.dtype == np.uint8 and set(np.unique(bits)) <= {0, 1}


@pytest.mark.parametrize("problem_name", PROBLEM_NAMES)
@pytest.mark.parametrize("codec_name", sorted(CODECS))
def test_encode_decode_roundtrip(problem_name, codec_name, rng):
    problem = make_problem(problem_name, 10)
    codec = CODECS[codec_name](problem)
    points = rng.uniform(problem.lo, problem.hi, size=(50, 10))
    decoded = codec.decode(codec.encode(points))
    # Tolerance is relative to the domain width: each codec keeps its own resolution.
    assert np.allclose(decoded, points, atol=RESOLUTION[codec_name] * problem.width)


@pytest.mark.parametrize("problem_name", PROBLEM_NAMES)
@pytest.mark.parametrize("codec_name", sorted(CODECS))
def test_arbitrary_bits_decode_into_the_domain(problem_name, codec_name, rng):
    problem = make_problem(problem_name, 10)
    codec = CODECS[codec_name](problem)
    values = codec.decode(rng.integers(0, 2, size=(500, codec.length), dtype=np.uint8))
    assert values.shape == (500, 10)
    assert np.all(np.isfinite(values))
    assert np.all((values >= problem.lo) & (values <= problem.hi))


def test_ieee754_maps_nan_into_the_domain():
    problem = make_problem("sphere", 2)
    codec = Ieee754Codec(problem)
    # Exponent all ones with a non-zero mantissa is NaN.
    bits = np.zeros((1, codec.length), dtype=np.uint8)
    bits[0, 1:12] = 1
    bits[0, 20] = 1
    values = codec.decode(bits)
    assert np.all(np.isfinite(values))
    assert values[0, 0] == problem.hi


def test_bcd_repairs_invalid_nibbles():
    problem = make_problem("sphere", 2)
    codec = BcdCodec(problem)
    assert (codec.int_digits, codec.frac_digits) == (1, 6)
    bits = np.zeros((1, codec.length), dtype=np.uint8)
    bits[0, 4:8] = 1  # first value digit = 0b1111 = 15, an invalid BCD digit
    # 15 % 10 = 5 integer units, clipped to the domain bound 5.12.
    assert codec.decode(bits)[0, 0] == pytest.approx(5.0)


def test_bcd_sign_nibble():
    problem = make_problem("schwefel", 2)
    codec = BcdCodec(problem)
    points = np.array([[-123.4567, 0.0], [123.4567, 0.0]])
    bits = codec.encode(points)
    assert bits[0, 0] == 1 and bits[1, 0] == 0  # the sign nibble's high bit
    assert np.allclose(codec.decode(bits), points, atol=1e-4)
