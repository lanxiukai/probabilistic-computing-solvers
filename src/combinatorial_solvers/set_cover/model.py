"""Original covering constraints, OR-Library readers, and integer certificates."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    costs: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not self.costs or any(
            isinstance(c, bool) or not isinstance(c, Integral) or c <= 0 for c in self.costs
        ):
            raise ValueError("Costs must be positive integers")
        costs = tuple(map(int, self.costs))
        if not self.rows:
            raise ValueError("At least one required row is needed")
        rows = []
        for row in self.rows:
            if not row:
                raise ValueError("An empty required row makes the instance infeasible")
            if any(
                isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < len(costs)
                for j in row
            ):
                raise ValueError("Column indices must be integers within the instance")
            if len(set(row)) != len(row):
                raise ValueError("Repeated incidence in a required row")
            rows.append(tuple(sorted(map(int, row))))
        if sum(costs) + (max(costs) + 1) * len(rows) >= 2**52:
            raise ValueError("Default penalized energy must be exactly representable in FP64")
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "rows", tuple(rows))

    @property
    def size(self):
        return len(self.costs)

    @cached_property
    def columns(self):
        columns = [[] for _ in self.costs]
        for i, row in enumerate(self.rows):
            for j in row:
                columns[j].append(i)
        return tuple(tuple(column) for column in columns)

    def validate_bits(self, bits):
        values = np.asarray(bits)
        if (
            values.shape != (self.size,)
            or values.dtype.kind not in "biuf"
            or not np.isin(values, (0, 1)).all()
        ):
            raise ValueError("A certificate requires one binary value per column")
        return values.astype(np.int8, copy=True)

    def certificate(self, bits):
        """Recompute the original objective and every covering row, without caches."""
        values = self.validate_bits(bits)
        coverage = [sum(int(values[j]) for j in row) for row in self.rows]
        return {
            "cost": sum(c for c, x in zip(self.costs, values, strict=True) if x),
            "selected_columns": np.flatnonzero(values).tolist(),
            "uncovered_rows": [i for i, count in enumerate(coverage) if count == 0],
            "coverage": coverage,
            "feasible": all(coverage),
        }

    def checked_penalty(self, penalty):
        if (
            isinstance(penalty, bool)
            or not isinstance(penalty, Integral)
            or penalty <= 0
            or sum(self.costs) + int(penalty) * len(self.rows) >= 2**52
        ):
            raise ValueError("Penalty must be a positive integer within exact FP64 energy range")
        return int(penalty)

    def energy(self, bits, penalty):
        penalty = self.checked_penalty(penalty)
        certificate = self.certificate(bits)
        return certificate["cost"] + penalty * len(certificate["uncovered_rows"])

    def single_delta(self, bits, column, penalty):
        """Return E(x_j=1)-E(x_j=0) in original integer cost units."""
        values = self.validate_bits(bits)
        penalty = self.checked_penalty(penalty)
        if not isinstance(column, Integral) or not 0 <= column < self.size:
            raise ValueError("Column is outside the instance")
        critical = sum(
            not any(values[j] for j in self.rows[i] if j != column) for i in self.columns[column]
        )
        return self.costs[column] - penalty * critical

    def payload(self):
        return {"schema": "set-cover-instance-v1", "costs": self.costs, "rows": self.rows}

    def statistics(self):
        degrees = [len(row) for row in self.rows]
        return {
            "rows": len(self.rows),
            "columns": self.size,
            "incidences": sum(degrees),
            "row_degree_min": min(degrees),
            "row_degree_max": max(degrees),
            "empty_columns": sum(not column for column in self.columns),
            "cost_min": min(self.costs),
            "cost_max": max(self.costs),
            "native_auxiliary_bits": 0,
            "binary_slack_bits": sum((degree - 1).bit_length() for degree in degrees),
        }


def read_orlib(path: Path, format: str = "scp") -> Instance:
    """Read standard row-oriented SCP or column-oriented railway files explicitly."""
    if format not in {"scp", "rail"}:
        raise ValueError("Format must be scp or rail")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as stream:
        tokens = iter(stream.read().split())

    def take():
        try:
            return int(next(tokens))
        except StopIteration as error:
            raise ValueError("Truncated OR-Library instance") from error

    m, n = take(), take()
    if m <= 0 or n <= 0:
        raise ValueError("Row and column counts must be positive")
    rows = [[] for _ in range(m)]
    if format == "scp":
        costs = [take() for _ in range(n)]
        for i in range(m):
            degree = take()
            if not 0 <= degree <= n:
                raise ValueError("Invalid row degree")
            rows[i] = [take() - 1 for _ in range(degree)]
    else:
        costs = []
        for j in range(n):
            cost, degree = take(), take()
            costs.append(cost)
            if not 0 <= degree <= m:
                raise ValueError("Invalid column degree")
            for _ in range(degree):
                row = take() - 1
                if not 0 <= row < m:
                    raise ValueError("Row index is outside the instance")
                rows[row].append(j)
    if next(tokens, None) is not None:
        raise ValueError("Trailing tokens in OR-Library instance")
    return Instance(tuple(costs), tuple(tuple(row) for row in rows))


def read_json(path: Path) -> Instance:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "set-cover-instance-v1":
        raise ValueError("Unknown set-cover schema")
    return Instance(tuple(payload["costs"]), tuple(tuple(row) for row in payload["rows"]))
