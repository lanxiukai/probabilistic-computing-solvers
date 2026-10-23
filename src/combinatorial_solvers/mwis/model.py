"""Integer MWIS instances and an independently checkable pairwise encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from combinatorial_solvers.core.edge_model import EdgeIsingModel


@dataclass(frozen=True)
class Instance:
    weights: np.ndarray
    edges: np.ndarray

    def __post_init__(self):
        weights, edges = np.asarray(self.weights), np.asarray(self.edges)
        if weights.ndim != 1 or not len(weights) or weights.dtype.kind not in "iu":
            raise ValueError("Weights must be a nonempty integer vector")
        if np.any(weights <= 0) or sum(map(int, weights)) >= 2**52:
            raise ValueError("Weights must be positive with total below 2**52")
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
            raise ValueError("Edges must be an integer [m, 2] array")
        if np.any(edges < 0) or np.any(edges >= len(weights)):
            raise ValueError("Edge endpoint outside graph")
        if np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("Edges require left < right")
        keys = edges[:, 0].astype(np.int64) * len(weights) + edges[:, 1]
        if np.any(keys[1:] <= keys[:-1]):
            raise ValueError("Edges must be sorted and unique")
        object.__setattr__(self, "weights", weights.astype(np.int64, copy=True))
        object.__setattr__(self, "edges", edges.astype(np.int64, copy=True))

    @property
    def size(self):
        return len(self.weights)

    def adjacency(self):
        rows = [[] for _ in range(self.size)]
        for a, b in self.edges:
            rows[a].append(int(b))
            rows[b].append(int(a))
        return rows

    def certificate(self, bits):
        values = np.asarray(bits)
        if values.shape != (self.size,) or not np.all((values == 0) | (values == 1)):
            raise ValueError("Certificate requires one binary assignment")
        weight = sum(int(w) for w, chosen in zip(self.weights, values, strict=True) if chosen)
        conflicts = sum(bool(values[a] and values[b]) for a, b in self.edges)
        return {"weight": weight, "conflicts": conflicts, "feasible": conflicts == 0}

    def ising(self, penalty: int):
        if isinstance(penalty, bool) or not isinstance(penalty, int):
            raise ValueError("Penalty must be an integer")
        if penalty <= int(self.weights.max()):
            raise ValueError("Penalty must exceed every vertex weight")
        if penalty * len(self.edges) + sum(map(int, self.weights)) >= 2**52:
            raise ValueError("Encoding exceeds exact FP64 integer range")
        degree = np.bincount(self.edges.ravel(), minlength=self.size)
        return EdgeIsingModel(
            torch.from_numpy(self.edges[:, 0].copy()),
            torch.from_numpy(self.edges[:, 1].copy()),
            torch.full((len(self.edges),), -penalty / 4, dtype=torch.float64),
            torch.from_numpy(self.weights / 2 - penalty * degree / 4),
            constant=-float(self.weights.sum()) / 2 + penalty * len(self.edges) / 4,
        )

    def payload(self):
        return {
            "schema": "mwis-instance-v1",
            "weights": self.weights.tolist(),
            "edges": self.edges.tolist(),
        }

    def write_metis(self, path: Path):
        neighbors = self.adjacency()
        with path.open("x") as handle:
            handle.write(f"{self.size} {len(self.edges)} 10\n")
            for node, row in enumerate(neighbors):
                handle.write(
                    " ".join(map(str, [self.weights[node], *[x + 1 for x in sorted(row)]])) + "\n"
                )


def instance_from_problem(problem: dict) -> Instance:
    if "instance_file" in problem:
        if set(problem) != {"instance_file"}:
            raise ValueError("File input cannot be mixed with generated settings")
        payload = json.loads(Path(problem["instance_file"]).read_text())
        if payload.get("schema") != "mwis-instance-v1":
            raise ValueError("Unknown MWIS instance schema")
        return Instance(
            np.asarray(payload["weights"]),
            np.asarray(payload["edges"], dtype=np.int64).reshape(-1, 2),
        )
    if set(problem) != {"size", "mean_degree", "instance_seed", "weight_max"}:
        raise ValueError("Generated input needs size, mean_degree, instance_seed, weight_max")
    n, degree, seed, maximum = (
        problem[k] for k in ("size", "mean_degree", "instance_seed", "weight_max")
    )
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, degree, seed, maximum)):
        raise ValueError("Generated settings must be integers")
    if n < 1 or not 0 <= degree <= min(64, n - 1) or maximum < 1 or seed < 0:
        raise ValueError("Invalid sparse G(n,m) generation settings")
    rng = np.random.Generator(np.random.PCG64(seed))
    target = n * degree // 2
    keys: set[int] = set()
    # Rejection sampling produces uniform unordered edges without a dense matrix.
    while len(keys) < target:
        pairs = rng.integers(0, n, size=(2 * (target - len(keys)) + 16, 2))
        for a, b in pairs:
            if a != b:
                keys.add(int(min(a, b)) * n + int(max(a, b)))
                if len(keys) == target:
                    break
    ordered = np.asarray(sorted(keys), dtype=np.int64)
    edges = np.column_stack((ordered // n, ordered % n))
    weights = rng.integers(1, maximum + 1, size=n, dtype=np.int64)
    return Instance(weights, edges)


def exact_optimum(instance: Instance) -> dict:
    """Exhaustive integer branch recursion, restricted to correctness instances."""
    if instance.size > 24:
        raise ValueError("Exhaustive reference is limited to 24 vertices")
    adjacency = [0] * instance.size
    for a, b in instance.edges:
        adjacency[a] |= 1 << int(b)
        adjacency[b] |= 1 << int(a)
    best, mask = 0, 0

    def visit(remaining, selected, weight):
        nonlocal best, mask
        if weight > best:
            best, mask = weight, selected
        upper = weight + sum(
            int(instance.weights[i]) for i in range(instance.size) if remaining >> i & 1
        )
        if upper <= best or not remaining:
            return
        i = max(
            (i for i in range(instance.size) if remaining >> i & 1),
            key=lambda i: (adjacency[i] & remaining).bit_count(),
        )
        rest = remaining & ~(1 << i)
        visit(rest & ~adjacency[i], selected | (1 << i), weight + int(instance.weights[i]))
        visit(rest, selected, weight)

    visit((1 << instance.size) - 1, 0, 0)
    bits = [(mask >> i) & 1 for i in range(instance.size)]
    assert instance.certificate(bits) == {"weight": best, "conflicts": 0, "feasible": True}
    return {"method": "integer-exhaustive-branch-v1", "optimum": best, "bits": bits}
