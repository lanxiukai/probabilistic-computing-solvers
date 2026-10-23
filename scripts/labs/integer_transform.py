"""Exact NTT correlation certificate independent of solver FFT and bitset code."""

from functools import lru_cache
from math import isqrt

import numpy as np

MODULUS = 998244353
ROOT = 3


@lru_cache(maxsize=1)
def validate_modulus():
    # The factorization p-1 = 2**23 * 7 * 17 makes primitive-root checking complete.
    if any(MODULUS % divisor == 0 for divisor in range(2, isqrt(MODULUS) + 1)):
        raise AssertionError("NTT modulus is not prime")
    if MODULUS - 1 != 2**23 * 7 * 17:
        raise AssertionError("Incorrect group-order factorization")
    if any(pow(ROOT, (MODULUS - 1) // factor, MODULUS) == 1 for factor in (2, 7, 17)):
        raise AssertionError("NTT generator is not primitive")
    if (MODULUS - 1) ** 2 > np.iinfo(np.int64).max:
        raise AssertionError("Integer products could overflow")


@lru_cache(maxsize=16)
def permutation(length):
    bits = length.bit_length() - 1
    if 1 << bits != length or (MODULUS - 1) % length:
        raise ValueError("Unsupported transform length")
    source = np.arange(length, dtype=np.int64)
    target = np.zeros(length, dtype=np.int64)
    for _ in range(bits):
        target = (target << 1) | (source & 1)
        source >>= 1
    return target


@lru_cache(maxsize=48)
def twiddles(span, inverse):
    root = pow(ROOT, (MODULUS - 1) // span, MODULUS)
    if inverse:
        root = pow(root, MODULUS - 2, MODULUS)
    powers = np.empty(span // 2, dtype=np.int64)
    value = 1
    for i in range(len(powers)):
        powers[i] = value
        value = value * root % MODULUS
    return powers


def transform(values, inverse=False):
    validate_modulus()
    result = np.asarray(values, dtype=np.int64)[..., permutation(values.shape[-1])].copy()
    length = result.shape[-1]
    result %= MODULUS
    span = 2
    while span <= length:
        blocks = result.reshape(*result.shape[:-1], -1, span)
        a = blocks[..., : span // 2].copy()
        b = blocks[..., span // 2 :] * twiddles(span, inverse) % MODULUS
        blocks[..., : span // 2] = (a + b) % MODULUS
        blocks[..., span // 2 :] = (a - b) % MODULUS
        span *= 2
    if inverse:
        result = result * pow(length, MODULUS - 2, MODULUS) % MODULUS
    return result


def correlations(bits):
    bits = np.asarray(bits)
    if bits.ndim not in (1, 2) or not np.isin(bits, (0, 1)).all():
        raise AssertionError("Expected binary NTT input")
    size = bits.shape[-1]
    if not 2 <= size <= 32768 or 2 * size >= MODULUS:
        raise AssertionError("Correlation reconstruction bound fails")
    values = 2 * np.atleast_2d(bits).astype(np.int64) - 1
    length = 1 << (2 * size - 2).bit_length()
    forward = np.zeros((len(values), length), dtype=np.int64)
    reverse = forward.copy()
    forward[:, :size], reverse[:, :size] = values, values[:, ::-1]
    product = transform(forward) * transform(reverse) % MODULUS
    convolution = transform(product, inverse=True)
    if np.any(convolution[:, size - 1] != size):
        raise AssertionError("Zero-lag NTT correlation is wrong")
    result = convolution[:, size : 2 * size - 1]
    result = np.where(result > MODULUS // 2, result - MODULUS, result)
    bounds = np.arange(size - 1, 0, -1)
    if np.any(np.abs(result) > bounds) or np.any((result - bounds) % 2):
        raise AssertionError("Exact correlation magnitude or parity is impossible")
    return result[0] if bits.ndim == 1 else result


def one_flip_differences(bits, corr):
    """Exact CPU neighborhood certificate, without the solver FFT gain expression."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.isin(bits, (0, 1)).all():
        raise AssertionError("Expected one binary sequence")
    spins = 2 * bits.astype(np.int64) - 1
    if np.asarray(corr).shape != (len(spins) - 1,):
        raise AssertionError("Correlation length differs")
    result = np.empty(len(spins), dtype=np.int64)
    for site, spin in enumerate(spins):
        change = np.zeros(len(spins) - 1, dtype=np.int64)
        change[:site] -= 2 * spin * spins[:site][::-1]
        change[: len(spins) - site - 1] -= 2 * spin * spins[site + 1 :]
        result[site] = np.dot(change, 2 * corr + change)
    return result
