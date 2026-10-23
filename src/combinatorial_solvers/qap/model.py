"""Original integer QAP objective and independent permutation certificates."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    flow: np.ndarray
    distance: np.ndarray

    def __post_init__(self):
        for name in ("flow", "distance"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
                raise ValueError("QAP matrices must be square with order at least two")
            if value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError("Nonnegative integer matrices are required")
            if value.dtype.kind == "u" and int(value.max()) > (1 << 63) - 1:
                raise ValueError("Matrix entries must fit signed 64-bit storage")
            if int(value.max()) >= 1 << 62:
                raise ValueError("Matrix entries exceed the signed delta arithmetic envelope")
            value = value.astype(np.int64, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.flow.shape != self.distance.shape:
            raise ValueError("QAP matrix orders differ")
        if self.size**2 * int(self.flow.max()) * int(self.distance.max()) >= 2**52:
            raise ValueError("Objective exceeds the exact integer comparison envelope")

    @property
    def size(self):
        return len(self.flow)

    def certificate(self, permutation):
        p = np.asarray(permutation)
        feasible = (
            p.shape == (self.size,)
            and p.dtype.kind in "iu"
            and np.array_equal(np.sort(p), np.arange(self.size))
        )
        cost = None
        if feasible:
            # The constructor proves an objective bound below 2**52. Every
            # nonnegative product and partial sum therefore fits int64 exactly.
            # This evaluates the full original objective, not GPU deltas/caches.
            cost = int(np.sum(self.flow * self.distance[p[:, None], p[None, :]], dtype=np.int64))
        return {"feasible": bool(feasible), "cost": cost}

    def exhaustive_optimum(self):
        if self.size > 8:
            raise ValueError("Exhaustive reference is restricted to order eight")
        return min(self.certificate(p)["cost"] for p in itertools.permutations(range(self.size)))


def read_qaplib(path: Path):
    values = [int(token) for token in path.read_text().split()]
    n = values[0]
    if len(values) != 1 + 2 * n * n:
        raise ValueError("QAPLIB input must contain exactly two integer matrices")
    return Instance(
        np.array(values[1 : 1 + n * n]).reshape(n, n), np.array(values[1 + n * n :]).reshape(n, n)
    )


def read_solution(path: Path, instance: Instance, *, index_base=1, inverse=False):
    values = [int(token) for token in path.read_text().split()]
    if len(values) != instance.size + 2 or values[0] != instance.size:
        raise ValueError("Invalid QAPLIB solution header")
    if index_base not in (0, 1):
        raise ValueError("Reference index base must be zero or one")
    p = np.array(values[2:], dtype=np.int64) - index_base
    if inverse:
        if not np.array_equal(np.sort(p), np.arange(instance.size)):
            raise ValueError("Reference witness is not a permutation")
        p = np.argsort(p)
    certificate = instance.certificate(p)
    if not certificate["feasible"] or certificate["cost"] != values[1]:
        raise ValueError("Reference witness does not match the original integer objective")
    return p, certificate
