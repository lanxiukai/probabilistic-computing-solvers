"""Native LABS energy, exact certificates, and constructive reference sequences."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import numpy as np


def validate_bits(bits, size=None):
    values = np.asarray(bits)
    if values.ndim not in (1, 2) or not np.isin(values, (0, 1)).all():
        raise ValueError("Expected one or more binary sequences")
    if size is not None and values.shape[-1] != size:
        raise ValueError("Sequence length differs from the model")
    return values.astype(np.uint8, copy=False)


def correlations(bits):
    values = validate_bits(bits)
    size = values.shape[-1]
    results = []
    for row in np.atleast_2d(values):
        number = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        mask = (1 << size) - 1
        result = []
        for lag in range(1, size):
            mask >>= 1
            different = ((number ^ (number >> lag)) & mask).bit_count()
            result.append(size - lag - 2 * different)
        results.append(result)
    array = np.asarray(results, dtype=np.int64)
    return array[0] if values.ndim == 1 else array


def energies(bits):
    corr = correlations(bits)
    return (corr * corr).sum(-1, dtype=np.int64)


def is_prime(value):
    return value >= 2 and all(value % d for d in range(2, isqrt(value) + 1))


def legendre(size, *, appended):
    """Construct the requested length, without using a stored optimized sequence."""
    if size < 8:
        raise ValueError("The constructive baseline is defined from length eight")
    # Middle real root of 4*T**3-30*T+27; theorem 2.1 of arXiv:1205.0626v2.
    ratio = 1.0
    if appended:
        ratio = 1.05
        for _ in range(8):
            ratio -= (4 * ratio**3 - 30 * ratio + 27) / (12 * ratio**2 - 30)
    center = max(3, round(size / ratio))
    prime = min(
        (p for p in range(max(3, center - 256), center + 257) if is_prime(p)),
        key=lambda p: (abs(p - size / ratio), p),
    )
    sequence = -np.ones(prime, dtype=np.int8)
    residues = (np.arange(1, prime, dtype=np.int64) ** 2) % prime
    sequence[residues] = 1
    sequence[0] = 1
    shift = round(prime * (0.75 - ratio / 2))
    selected = sequence[(np.arange(size) + shift) % prime]
    return ((selected + 1) // 2).astype(np.uint8), {
        "family": "appended-rotated-legendre" if appended else "quarter-rotated-legendre",
        "prime": prime,
        "length": size,
        "shift": shift,
        "asymptotic_ratio": ratio,
        "actual_length_ratio": size / prime,
        "proven_optimal_for_length": False,
    }


@dataclass(frozen=True)
class Instance:
    size: int

    def __post_init__(self):
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or not 2 <= self.size <= 32768
        ):
            raise ValueError("LABS sequence length must be between two and 32768")

    def energy(self, bits):
        return energies(validate_bits(bits, self.size))

    @property
    def active_variables(self):
        # For N >= 4, every variable occurs in a positive distance-two quadratic term.
        return {2: 0, 3: 2}.get(self.size, self.size)

    def exact_optimum(self):
        if self.size > 20:
            raise ValueError("Exhaustive reference is limited to twenty variables")
        best, witness = None, None
        for number in range(2 ** (self.size - 1)):
            bits = np.asarray([(number >> i) & 1 for i in range(self.size)], dtype=np.uint8)
            value = int(self.energy(bits))
            if best is None or value < best:
                best, witness = value, bits.tolist()
        return {"optimum_energy": best, "bits": witness, "method": "integer-bitset-enumeration"}
