"""Deterministic prime and balanced-semiprime generation utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SemiprimeInstance:
    """A known-factor instance used to validate an inverse multiplier."""

    product: int
    factor_a: int
    factor_b: int
    product_bits: int
    factor_bits_a: int
    factor_bits_b: int


def is_prime(value: int) -> bool:
    """Deterministically test primality for unsigned 64-bit integers."""

    number = int(value)
    if number < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if number == prime:
            return True
        if number % prime == 0:
            return False

    exponent = number - 1
    shifts = 0
    while exponent % 2 == 0:
        shifts += 1
        exponent //= 2
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % number == 0:
            continue
        witness = pow(base, exponent, number)
        if witness in (1, number - 1):
            continue
        for _ in range(shifts - 1):
            witness = witness * witness % number
            if witness == number - 1:
                break
        else:
            return False
    return True


def _small_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for prime in range(2, math.isqrt(limit) + 1):
        if sieve[prime]:
            start = prime * prime
            sieve[start : limit + 1 : prime] = b"\x00" * ((limit - start) // prime + 1)
    return [index for index, flag in enumerate(sieve) if flag]


def iter_primes(
    start: int,
    stop: int,
    *,
    segment_size: int = 1 << 20,
) -> Iterator[int]:
    """Yield primes in the inclusive interval using bounded-size segments."""

    lower = max(2, int(start))
    upper = int(stop)
    if upper < lower:
        return
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    base_primes = _small_primes(math.isqrt(upper))
    for segment_start in range(lower, upper + 1, segment_size):
        segment_stop = min(segment_start + segment_size - 1, upper)
        flags = bytearray(b"\x01") * (segment_stop - segment_start + 1)
        for prime in base_primes:
            first = max(prime * prime, ((segment_start + prime - 1) // prime) * prime)
            if first > segment_stop:
                continue
            flags[first - segment_start : segment_stop - segment_start + 1 : prime] = b"\x00" * (
                (segment_stop - first) // prime + 1
            )
        for offset, flag in enumerate(flags):
            if flag:
                yield segment_start + offset


def primes_up_to(limit: int) -> tuple[int, ...]:
    """Materialize a prime table; callers choose limits that fit available memory."""

    return tuple(iter_primes(2, int(limit)))


def sample_semiprime_from_table(
    primes: Sequence[int],
    product_bits: int,
    *,
    rng: random.Random,
    require_distinct: bool = False,
    maximum_attempts: int = 100_000,
) -> SemiprimeInstance:
    """Sample two table entries whose product has the requested exact bit length."""

    if product_bits < 2:
        raise ValueError("product_bits must be at least two")
    candidates = [int(prime) for prime in primes if int(prime) >= 2]
    if not candidates:
        raise ValueError("Prime table is empty")
    composite = next((candidate for candidate in candidates if not is_prime(candidate)), None)
    if composite is not None:
        raise ValueError(f"Prime table contains a composite value: {composite}")
    for _ in range(maximum_attempts):
        left = rng.choice(candidates)
        right = rng.choice(candidates)
        if require_distinct and left == right:
            continue
        product = left * right
        if product.bit_length() == product_bits:
            return SemiprimeInstance(
                product=product,
                factor_a=left,
                factor_b=right,
                product_bits=product_bits,
                factor_bits_a=left.bit_length(),
                factor_bits_b=right.bit_length(),
            )
    raise RuntimeError("Could not sample a semiprime with the requested product width")


def generate_balanced_semiprime(
    product_bits: int,
    *,
    seed: int,
    require_distinct: bool = False,
    maximum_attempts: int = 100_000,
) -> SemiprimeInstance:
    """Generate a reproducible semiprime from equally wide prime factors."""

    if product_bits < 4 or product_bits % 2:
        raise ValueError("Balanced multiplier instances require an even product_bits >= 4")
    factor_bits = product_bits // 2
    lower = 1 << (factor_bits - 1)
    upper = (1 << factor_bits) - 1
    rng = random.Random(int(seed))
    for _ in range(maximum_attempts):
        left = rng.randrange(lower, upper + 1) | 1
        right = rng.randrange(lower, upper + 1) | 1
        if not is_prime(left) or not is_prime(right):
            continue
        if require_distinct and left == right:
            continue
        product = left * right
        if product.bit_length() == product_bits:
            return SemiprimeInstance(
                product=product,
                factor_a=left,
                factor_b=right,
                product_bits=product_bits,
                factor_bits_a=factor_bits,
                factor_bits_b=factor_bits,
            )
    raise RuntimeError("Could not generate a balanced semiprime within the attempt budget")
