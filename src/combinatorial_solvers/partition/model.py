"""Undirected positive integer graphs and exact cardinality certificates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass
class Graph:
    size: int
    edges: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.weights = np.asarray(self.weights, dtype=np.int64)
        if self.size < 4 or self.size % 2 or self.weights.shape != (len(self.edges),):
            raise ValueError("An even graph size and one weight per edge are required")
        if (
            np.any(self.edges < 0)
            or np.any(self.edges >= self.size)
            or np.any(self.edges[:, 0] >= self.edges[:, 1])
            or np.any(self.weights <= 0)
            or len(np.unique(self.edges, axis=0)) != len(self.edges)
            or int(self.weights.sum()) >= 2**30
        ):
            raise ValueError("Expected unique upper-triangle edges with bounded positive weights")

    def cut(self, bits):
        bits = np.asarray(bits)
        if bits.shape[-1] != self.size or np.any((bits != 0) & (bits != 1)):
            raise ValueError("Invalid partition labels")
        return np.sum(
            (bits[..., self.edges[:, 0]] != bits[..., self.edges[:, 1]]) * self.weights,
            axis=-1,
            dtype=np.int64,
        )

    def certify(self, bits):
        bits = np.asarray(bits)
        cut = self.cut(bits)
        if np.any(bits.sum(axis=-1) != self.size // 2):
            raise ValueError("Partition is not exactly balanced")
        return cut

    def adjacency(self):
        neighbors = [[] for _ in range(self.size)]
        for (i, j), w in zip(self.edges, self.weights, strict=True):
            neighbors[i].append((int(j), int(w)))
            neighbors[j].append((int(i), int(w)))
        return neighbors

    def write_metis(self, path):
        neighbors = self.adjacency()
        with Path(path).open("x") as handle:
            handle.write(f"{self.size} {len(self.edges)} 1\n")
            for row in neighbors:
                handle.write(" ".join(f"{j + 1} {w}" for j, w in sorted(row)) + "\n")

    def exact(self):
        if self.size > 22:
            raise ValueError("Enumeration is limited to 22 vertices")
        best, witness = int(self.weights.sum()) + 1, None
        for selected in combinations(range(1, self.size), self.size // 2 - 1):
            bits = np.zeros(self.size, dtype=np.int8)
            bits[[0, *selected]] = 1
            value = int(self.cut(bits))
            if value < best:
                best, witness = value, bits.copy()
        return {"cut": best, "bits": witness.tolist()}


def read_metis(path):
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if not line.lstrip().startswith("%")
    ]
    header = [int(x) for x in lines[0].split()]
    if len(header) not in (2, 3) or (len(header) == 3 and header[2] not in (0, 1)):
        raise ValueError("Only unit vertex weights and optional edge weights are supported")
    n, m = header[:2]
    weighted = len(header) == 3 and header[2] == 1
    if len(lines[1:]) != n:
        raise ValueError("METIS row count differs from the graph header")
    directed = {}
    for i, line in enumerate(lines[1:]):
        values = [int(x) for x in line.split()]
        if weighted and len(values) % 2:
            raise ValueError("Incomplete weighted adjacency row")
        row = zip(values[::2], values[1::2], strict=True) if weighted else ((j, 1) for j in values)
        for j, w in row:
            key = (i, j - 1)
            if key in directed:
                raise ValueError("Duplicate adjacency")
            directed[key] = w
    if len(directed) != 2 * m or any(directed.get((j, i)) != w for (i, j), w in directed.items()):
        raise ValueError("Graph is not symmetric or has the wrong edge count")
    upper = sorted((i, j, w) for (i, j), w in directed.items() if i < j)
    return Graph(n, [(i, j) for i, j, _ in upper], [w for _, _, w in upper])


def generate(config):
    n, seed = config["size"], config["seed"]
    rng = np.random.default_rng(seed)
    edges = {(i, i + 1) for i in range(n - 1)}
    edges.add((0, n - 1))
    family = config["family"]
    if family in ("random", "community"):
        target = n * config.get("degree", 12) // 2
        while len(edges) < target:
            i, j = (int(x) for x in rng.integers(n, size=2))
            if family == "community" and rng.random() < 0.8:
                j = (i // (n // 2)) * (n // 2) + int(rng.integers(n // 2))
            if i != j:
                edges.add((min(i, j), max(i, j)))
    elif family == "dense":
        for i in range(n - 1):
            edges.update((i, int(j)) for j in np.flatnonzero(rng.random(n - i - 1) < 0.5) + i + 1)
    else:
        raise ValueError("Unknown graph family")
    ordered = np.asarray(sorted(edges), dtype=np.int64)
    weights = rng.integers(1, config.get("maximum_weight", 1) + 1, len(ordered))
    return Graph(n, ordered, weights)
