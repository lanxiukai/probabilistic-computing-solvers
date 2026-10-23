"""Weighted soft clauses with integer objective certificates and WCNF input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from combinatorial_solvers.core.coloring import dsatur_coloring
from combinatorial_solvers.core.edge_model import EdgeIsingModel


@dataclass(frozen=True)
class Instance:
    variables: int
    literals: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        literals, weights = np.asarray(self.literals), np.asarray(self.weights)
        if not isinstance(self.variables, int) or self.variables < 1:
            raise ValueError("Variable count must be a positive integer")
        if literals.ndim != 2 or literals.dtype.kind not in "iu" or not len(literals):
            raise ValueError("Literals require a nonempty integer clause matrix")
        if weights.shape != (len(literals),) or weights.dtype.kind not in "iu":
            raise ValueError("One integer weight is required per clause")
        if np.any(weights <= 0) or sum(map(int, weights)) >= 2**52:
            raise ValueError("Positive weights with total below 2**52 are required")
        if np.any(literals > self.variables) or np.any(literals < -self.variables):
            raise ValueError("Literal exceeds the declared variable range")
        for row in literals:
            variables = np.abs(row[row != 0]).tolist()
            if not variables or len(set(variables)) != len(variables):
                raise ValueError("Clauses must be nonempty and contain distinct variables")
        object.__setattr__(self, "literals", literals.astype(np.int64, copy=True))
        object.__setattr__(self, "weights", weights.astype(np.int64, copy=True))

    @property
    def active_variables(self):
        return len(np.unique(np.abs(self.literals[self.literals != 0])))

    def cost(self, bits):
        bits = np.asarray(bits)
        if bits.shape[-1] != self.variables or not np.isin(bits, [0, 1]).all():
            raise ValueError("Expected binary assignments with the declared variable count")
        indices = np.maximum(np.abs(self.literals) - 1, 0)
        truth = np.where(self.literals > 0, bits[..., indices] == 1, bits[..., indices] == 0)
        satisfied = (truth & (self.literals != 0)).any(axis=-1)
        return ((~satisfied).astype(np.int64) * self.weights).sum(axis=-1, dtype=np.int64)

    def colors(self):
        edges = set()
        for row in self.literals:
            variables = sorted((np.abs(row[row != 0]) - 1).tolist())
            for i, left in enumerate(variables):
                for right in variables[i + 1 :]:
                    edges.add((left, right))
        edges = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
        primal = EdgeIsingModel(
            torch.tensor(edges[:, 0]),
            torch.tensor(edges[:, 1]),
            torch.ones(len(edges), dtype=torch.float64),
            torch.zeros(self.variables, dtype=torch.float64),
        )
        colors = dsatur_coloring(primal)
        assigned = np.empty(self.variables, dtype=np.int64)
        for color, nodes in enumerate(colors):
            assigned[list(nodes)] = color
        for row in self.literals:
            used = assigned[np.abs(row[row != 0]) - 1]
            if len(np.unique(used)) != len(used):
                raise AssertionError("A clause has simultaneous dependent updates")
        return colors

    def write_wcnf(self, path: Path):
        top = sum(map(int, self.weights)) + 1
        with path.open("x") as handle:
            handle.write(f"p wcnf {self.variables} {len(self.weights)} {top}\n")
            for weight, clause in zip(self.weights, self.literals, strict=True):
                row = [int(weight), *clause[clause != 0].tolist(), 0]
                handle.write(" ".join(map(str, row)) + "\n")

    def save(self, path: Path):
        np.savez_compressed(
            path, variables=np.int64(self.variables), literals=self.literals, weights=self.weights
        )


def from_wcnf(path: Path):
    variables = declared = top = None
    clauses, weights = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == "c":
            continue
        if fields[0] == "p":
            if fields[1] != "wcnf" or len(fields) not in {4, 5} or variables is not None:
                raise ValueError("Expected one old-format WCNF header")
            variables, declared = int(fields[2]), int(fields[3])
            top = int(fields[4]) if len(fields) == 5 else None
            continue
        if variables is None:
            raise ValueError("WCNF header must precede clauses")
        values = list(map(int, fields))
        if len(values) < 3 or values[-1] != 0 or 0 in values[1:-1]:
            raise ValueError("Each nonempty clause must occupy one zero-terminated line")
        if top is not None and values[0] >= top:
            raise ValueError("Hard clauses are outside the initial weighted soft-clause scope")
        weights.append(values[0])
        clauses.append(values[1:-1])
    if declared != len(clauses) or not clauses:
        raise ValueError("WCNF clause count mismatch")
    matrix = np.zeros((len(clauses), max(map(len, clauses))), dtype=np.int64)
    for i, clause in enumerate(clauses):
        matrix[i, : len(clause)] = clause
    return Instance(variables, matrix, np.asarray(weights, dtype=np.int64))


def instance_from_problem(problem):
    if "instance_file" in problem:
        if set(problem) != {"instance_file"}:
            raise ValueError("File and generated inputs cannot be mixed")
        return from_wcnf(Path(problem["instance_file"]))
    if set(problem) != {"variables", "clause_ratio", "arity", "weight_max", "instance_seed"}:
        raise ValueError("Unknown generated Max-SAT setting")
    n, ratio, arity, maximum, seed = (
        problem[k] for k in ("variables", "clause_ratio", "arity", "weight_max", "instance_seed")
    )
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, ratio, arity, maximum, seed)):
        raise ValueError("Generated settings must be integers")
    if n < 1 or ratio < 1 or not 1 <= arity <= min(n, 8) or maximum < 1 or seed < 0:
        raise ValueError("Invalid generated clause settings")
    rng = np.random.Generator(np.random.PCG64(seed))
    variables = rng.integers(1, n + 1, size=(n * ratio, arity))
    while True:
        ordered = np.sort(variables, axis=1)
        invalid = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
        if not invalid.any():
            break
        variables[invalid] = rng.integers(1, n + 1, size=(int(invalid.sum()), arity))
    signs = 2 * rng.integers(0, 2, size=variables.shape) - 1
    weights = rng.integers(1, maximum + 1, size=len(variables), dtype=np.int64)
    return Instance(n, variables * signs, weights)


def exact_optimum(instance: Instance):
    if instance.variables > 20:
        raise ValueError("Exact enumeration is limited to at most 20 variables")
    best_cost, best_bits = sum(map(int, instance.weights)) + 1, None
    for start in range(0, 1 << instance.variables, 512):
        values = np.arange(start, min(start + 512, 1 << instance.variables), dtype=np.uint64)
        bits = ((values[:, None] >> np.arange(instance.variables, dtype=np.uint64)) & 1).astype(
            np.uint8
        )
        costs = instance.cost(bits)
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best_cost, best_bits = int(costs[index]), bits[index].tolist()
    return {
        "method": "complete-integer-enumeration-v1",
        "optimum_cost": best_cost,
        "bits": best_bits,
    }
