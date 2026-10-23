"""Integer Max-Cut instances, certificates, and small exhaustive references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    size: int
    edges: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        edges, weights = np.asarray(self.edges), np.asarray(self.weights)
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 1:
            raise ValueError("Size must be a positive integer")
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
            raise ValueError("Edges must be an integer [m, 2] array")
        if weights.shape != (len(edges),) or weights.dtype.kind not in "iu" or not len(weights):
            raise ValueError("Expected one integer weight per edge in a nonempty graph")
        if np.any(weights == 0) or sum(abs(int(w)) for w in weights) >= 2**52:
            raise ValueError("Nonzero weights and an exactly representable objective are required")
        if np.any(edges < 0) or np.any(edges >= self.size) or np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("Edge endpoints require 0 <= left < right < size")
        keys = edges[:, 0].astype(np.int64) * self.size + edges[:, 1]
        if np.any(np.diff(keys) <= 0):
            raise ValueError("Edges must be sorted and unique")
        object.__setattr__(self, "edges", edges.astype(np.int64, copy=True))
        object.__setattr__(self, "weights", weights.astype(np.int64, copy=True))

    @property
    def active_variables(self):
        return len(np.unique(self.edges))

    def cut(self, bits):
        values = np.asarray(bits)
        if values.shape[-1] != self.size or not np.isin(values, (0, 1)).all():
            raise ValueError("A cut requires a binary assignment of the original vertices")
        result = np.zeros(values.shape[:-1], dtype=np.int64)
        for start in range(0, len(self.edges), 8192):
            edges, weights = self.edges[start : start + 8192], self.weights[start : start + 8192]
            result += ((values[..., edges[:, 0]] != values[..., edges[:, 1]]) * weights).sum(
                -1, dtype=np.int64
            )
        return result

    def adjacency(self):
        neighbors = [[] for _ in range(self.size)]
        for (a, b), w in zip(self.edges, self.weights, strict=True):
            neighbors[a].append((int(b), int(w)))
            neighbors[b].append((int(a), int(w)))
        return neighbors

    def colors(self):
        adjacency = self.adjacency()
        colors = np.full(self.size, -1, dtype=np.int64)
        for node in sorted(range(self.size), key=lambda i: (-len(adjacency[i]), i)):
            used = {colors[j] for j, _ in adjacency[node]}
            color = 0
            while color in used:
                color += 1
            colors[node] = color
        return [np.flatnonzero(colors == c) for c in range(int(colors.max()) + 1)]

    def save(self, filename):
        np.savez(filename, size=self.size, edges=self.edges, weights=self.weights)

    def write_mqlib(self, filename):
        with Path(filename).open("x") as output:
            output.write(f"{self.size} {len(self.edges)}\n")
            for (a, b), w in zip(self.edges, self.weights, strict=True):
                output.write(f"{a + 1} {b + 1} {w}\n")


def from_mqlib(filename):
    lines = Path(filename).read_text().splitlines()
    size, count = map(int, lines[0].split())
    rows = np.asarray(
        [list(map(int, line.split())) for line in lines[1:] if line.strip()], dtype=np.int64
    )
    if rows.shape != (count, 3):
        raise ValueError("Graph header and edge records disagree")
    edges = np.sort(rows[:, :2] - 1, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return Instance(size, edges[order], rows[order, 2])


def instance_from_problem(problem):
    if "instance_file" in problem:
        if set(problem) != {"instance_file"}:
            raise ValueError("File input and generator options cannot be combined")
        return from_mqlib(problem["instance_file"])
    size, degree, seed, maximum = (
        problem[k] for k in ("size", "mean_degree", "instance_seed", "weight_max")
    )
    signed = bool(problem.get("signed", False))
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (size, degree, seed, maximum)):
        raise ValueError("Generator options must be integers")
    if (
        not 2 <= size
        or not 1 <= degree <= min(128, size - 1)
        or seed < 0
        or not 1 <= maximum <= 1000
    ):
        raise ValueError("Invalid sparse G(n,m) settings")
    rng = np.random.Generator(np.random.PCG64(seed))
    target = size * degree // 2
    keys = set()
    while len(keys) < target:
        pairs = rng.integers(0, size, size=(2 * (target - len(keys)) + 16, 2))
        for a, b in pairs:
            if a != b:
                keys.add(int(min(a, b)) * size + int(max(a, b)))
                if len(keys) == target:
                    break
    keys = np.asarray(sorted(keys), dtype=np.int64)
    weights = rng.integers(1, maximum + 1, size=target, dtype=np.int64)
    if signed:
        weights *= 2 * rng.integers(0, 2, size=target, dtype=np.int64) - 1
    return Instance(size, np.column_stack((keys // size, keys % size)), weights)


def exact_optimum(instance):
    if instance.size > 20:
        raise ValueError("Exhaustive reference is limited to 20 vertices")
    # Global inversion preserves every edge cut; fix only the reference's last bit.
    best, witness = -1, None
    for start in range(0, 2 ** (instance.size - 1), 1024):
        numbers = np.arange(start, min(start + 1024, 2 ** (instance.size - 1)), dtype=np.uint64)
        bits = ((numbers[:, None] >> np.arange(instance.size, dtype=np.uint64)) & 1).astype(
            np.uint8
        )
        values = instance.cut(bits)
        index = int(values.argmax())
        if values[index] > best:
            best, witness = int(values[index]), bits[index].tolist()
    return {"optimum_cut": best, "bits": witness, "method": "exhaustive-global-inversion-v1"}
