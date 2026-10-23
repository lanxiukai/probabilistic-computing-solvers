"""Strength-two covering arrays and complete original interaction certificates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    rows: int
    columns: int
    levels: int

    def __post_init__(self):
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (self.rows, self.columns, self.levels)
            )
            or self.rows < 1
            or self.columns < 2
            or self.levels not in (2, 3, 4)
        ):
            raise ValueError(
                "Positive rows, at least two columns and two to four levels are required"
            )

    @property
    def size(self):
        return self.rows * self.columns

    @property
    def pairs(self):
        return self.columns * (self.columns - 1) // 2

    def validate(self, array):
        a = np.asarray(array)
        if (
            a.shape != (self.rows, self.columns)
            or a.dtype.kind not in "iu"
            or not ((a >= 0) & (a < self.levels)).all()
        ):
            raise ValueError("Array entries are outside the declared dimensions or alphabet")
        return a

    def counts(self, array):
        a = self.validate(array)
        result = np.empty((self.pairs, self.levels**2), dtype=np.int32)
        offset = 0
        for i in range(self.columns - 1):
            width = self.columns - i - 1
            codes = a[:, i, None] * self.levels + a[:, i + 1 :]
            keys = codes + np.arange(width, dtype=np.int64) * self.levels**2
            result[offset : offset + width] = np.bincount(
                keys.ravel(), minlength=width * self.levels**2
            ).reshape(width, self.levels**2)
            offset += width
        return result

    def certificate(self, array):
        try:
            a = self.validate(array)
        except ValueError:
            return {"valid_assignment": False, "feasible": False, "missing_interactions": None}
        # Bit-set union is independent of both histogram counts and CUDA deltas.
        missing = 0
        for i in range(self.columns - 1):
            codes = (a[:, i, None] * self.levels + a[:, i + 1 :]).astype(np.uint32)
            present = np.bitwise_or.reduce(np.left_shift(np.uint32(1), codes), axis=0)
            missing += int((self.levels**2 - np.bitwise_count(present)).sum())
        return {
            "valid_assignment": True,
            "feasible": missing == 0,
            "missing_interactions": missing,
            "total_interactions": self.pairs * self.levels**2,
        }


def read_instance(path: Path):
    payload = json.loads(path.read_text())
    if payload.get("strength", 2) != 2:
        raise ValueError("Only strength-two arrays are implemented")
    return Instance(payload["rows"], payload["columns"], payload["levels"])
