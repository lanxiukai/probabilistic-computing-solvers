"""Simple-graph conflict objective and complete independent certificates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    size: int
    colors: int
    edges: tuple[tuple[int, int], ...]
    allowed: np.ndarray | None = None

    def __post_init__(self):
        if self.size < 2 or self.colors < 2:
            raise ValueError("At least two vertices and two colors are required")
        edges = tuple(sorted((min(u, v), max(u, v)) for u, v in self.edges))
        if len(set(edges)) != len(edges) or any(
            u < 0 or v >= self.size or u == v for u, v in edges
        ):
            raise ValueError("Edges must define a simple graph")
        allowed = (
            np.ones((self.size, self.colors), dtype=bool)
            if self.allowed is None
            else np.asarray(self.allowed, dtype=bool).copy()
        )
        if allowed.shape != (self.size, self.colors) or not allowed.any(axis=1).all():
            raise ValueError("Every vertex must have a nonempty allowed-color list")
        allowed.flags.writeable = False
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "allowed", allowed)

    def certificate(self, colors):
        c = np.asarray(colors)
        valid = (
            c.shape == (self.size,)
            and c.dtype.kind in "iu"
            and bool(((c >= 0) & (c < self.colors)).all())
        )
        lists = valid and bool(self.allowed[np.arange(self.size), c].all())
        conflicts = sum(int(c[u] == c[v]) for u, v in self.edges) if lists else None
        return {
            "valid_assignment": bool(lists),
            "conflicts": conflicts,
            "feasible": bool(lists and conflicts == 0),
        }

    def adjacency(self):
        rows = [[] for _ in range(self.size)]
        for u, v in self.edges:
            rows[u].append(v)
            rows[v].append(u)
        return rows

    def clique_certificate(self, vertices):
        values = tuple(vertices)
        valid_vertices = (
            len(values) > 0
            and all(isinstance(v, int) and not isinstance(v, bool) for v in values)
            and len(set(values)) == len(values)
            and all(0 <= v < self.size for v in values)
        )
        edge_set = set(self.edges)
        valid = valid_vertices and all(
            (min(u, v), max(u, v)) in edge_set
            for i, u in enumerate(values)
            for v in values[i + 1 :]
        )
        return {
            "vertices": list(values),
            "size": len(values),
            "valid_clique": bool(valid),
            "pair_checks": len(values) * (len(values) - 1) // 2,
        }

    def with_clique_symmetry(self, vertices):
        certificate = self.clique_certificate(vertices)
        if not certificate["valid_clique"] or len(vertices) > self.colors:
            raise ValueError("A valid clique fitting the palette is required")
        if not self.allowed.all():
            raise ValueError("Global color symmetry cannot be assumed for restricted lists")
        allowed = self.allowed.copy()
        for color, vertex in enumerate(vertices):
            allowed[vertex] = False
            allowed[vertex, color] = True
        return Instance(self.size, self.colors, self.edges, allowed)


def read_dimacs(path: Path, colors: int, allowed=None):
    size, count = None, None
    edges = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] in {"c", "%"}:
            continue
        if fields[0] == "p" and len(fields) == 4 and fields[1] in {"edge", "edges", "col"}:
            size, count = map(int, fields[2:])
        elif fields[0] == "e" and len(fields) == 3:
            edges.append((int(fields[1]) - 1, int(fields[2]) - 1))
        else:
            raise ValueError("Unsupported DIMACS record")
    if size is None or len(edges) != count:
        raise ValueError("DIMACS edge count differs from its header")
    return Instance(size, colors, tuple(edges), allowed)
