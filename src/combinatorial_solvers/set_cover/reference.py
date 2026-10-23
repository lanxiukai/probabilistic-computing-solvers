"""Small CPU references for enumeration, greedy covers, and conditional kernels.

These are correctness tools, not a GPU solver or performance implementation.
"""

from __future__ import annotations

from itertools import product
from math import isfinite
from numbers import Integral

import numpy as np

from .model import Instance


def exact_optimum(instance: Instance):
    if instance.size > 20:
        raise ValueError("Enumeration is restricted to at most twenty columns")
    best, witness = sum(instance.costs) + 1, None
    for bits in product((0, 1), repeat=instance.size):
        certificate = instance.certificate(bits)
        if certificate["feasible"] and certificate["cost"] < best:
            best, witness = certificate["cost"], list(bits)
    return {"cost": best, "bits": witness, "method": "full-binary-enumeration"}


def greedy(instance: Instance):
    """Integer-ratio greedy selection followed by deterministic redundant deletion."""
    masks = [sum(1 << i for i in rows) for rows in instance.columns]
    remaining = (1 << len(instance.rows)) - 1
    selected = np.zeros(instance.size, dtype=np.int8)
    while remaining:
        best, best_gain = None, 0
        for j, mask in enumerate(masks):
            gain = (mask & remaining).bit_count()
            if gain and (
                best is None or instance.costs[j] * best_gain < instance.costs[best] * gain
            ):
                best, best_gain = j, gain
        if best is None:
            raise ValueError("The input has no feasible cover")
        selected[best] = 1
        remaining &= ~masks[best]
    for j in sorted(np.flatnonzero(selected), key=lambda j: (-instance.costs[j], int(j))):
        candidate = selected.copy()
        candidate[j] = 0
        if instance.certificate(candidate)["feasible"]:
            selected = candidate
    return selected


def conditional_distribution(instance, bits, block, beta, *, penalty=None):
    """Enumerate a block conditional; penalty=None means strict feasible support.

    A kernel schedule must choose blocks independently of the current state.
    Beta is in inverse original cost units, without implicit normalization.
    """
    values = instance.validate_bits(bits)
    block = tuple(block)
    if (
        not block
        or len(block) > 12
        or len(set(block)) != len(block)
        or any(
            isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < instance.size
            for j in block
        )
    ):
        raise ValueError("A block requires one to twelve distinct valid columns")
    if not isfinite(beta) or beta < 0:
        raise ValueError("Beta must be finite and nonnegative")
    if penalty is not None:
        penalty = instance.checked_penalty(penalty)
    states = np.asarray(list(product((0, 1), repeat=len(block))), dtype=np.int8)
    energies = np.full(len(states), np.inf)
    for i, local in enumerate(states):
        candidate = values.copy()
        candidate[list(block)] = local
        certificate = instance.certificate(candidate)
        if penalty is not None:
            energies[i] = certificate["cost"] + penalty * len(certificate["uncovered_rows"])
        elif certificate["feasible"]:
            energies[i] = certificate["cost"]
    valid = np.isfinite(energies)
    if not valid.any():
        raise ValueError("No feasible assignment exists within the selected block")
    probabilities = np.zeros(len(states))
    probabilities[valid] = np.exp(-beta * (energies[valid] - energies[valid].min()))
    probabilities /= probabilities.sum()
    return states, probabilities


def dual_bound(instance: Instance, numerators, denominator: int):
    """Check a rational feasible LP-dual witness using only integer arithmetic."""
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, Integral)
        or denominator <= 0
        or len(numerators) != len(instance.rows)
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in numerators)
    ):
        raise ValueError("Dual witness requires nonnegative integer row weights and a denominator")
    numerators = tuple(map(int, numerators))
    denominator = int(denominator)
    for j, rows in enumerate(instance.columns):
        if sum(numerators[i] for i in rows) > denominator * instance.costs[j]:
            raise ValueError("Dual witness violates a column inequality")
    numerator = sum(numerators)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "integer_cost_lower_bound": (numerator + denominator - 1) // denominator,
        "method": "rational-cover-dual-verification",
    }
