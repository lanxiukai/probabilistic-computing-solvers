"""Zero-field cubic Edwards-Anderson models with unique nearest-neighbor bonds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from cubic_spin_glass.core.edge_model import EdgeIsingModel


@dataclass(frozen=True)
class CubicInstance:
    length: int
    boundary: str
    disorder_seed: int | None
    model: EdgeIsingModel
    colors: tuple[torch.Tensor, torch.Tensor]
    coupling_id: str | None = None

    def energy_numpy(self, spins: np.ndarray) -> np.ndarray:
        """Independently certify integer energies from a unique-edge sum."""
        values = np.asarray(spins)
        if values.shape[-1] != self.model.size or not np.isin(values, [-1, 1]).all():
            raise ValueError("Certificates must contain bipolar spins of the correct size")
        left = self.model.edge_left.numpy()
        right = self.model.edge_right.numpy()
        weights = self.model.edge_couplings.numpy().astype(np.int64)
        return -(values[..., left] * values[..., right] * weights).sum(axis=-1)


def cubic_instance(length: int, boundary: str = "open", disorder_seed: int = 0) -> CubicInstance:
    """Generate iid +/-1 bonds; periodic checkerboards require even L >= 4.

    Coordinates use i = x + L*y + L*L*z. Periodic L=2 is deliberately rejected:
    a simple graph and a torus multigraph have different bond multiplicities.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer >= 2")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be open or periodic")
    if boundary == "periodic" and (length < 4 or length % 2):
        raise ValueError("Periodic checkerboard models require even length >= 4")
    if isinstance(disorder_seed, bool) or not isinstance(disorder_seed, int) or disorder_seed < 0:
        raise ValueError("disorder_seed must be a nonnegative integer")
    edges = []
    colors: list[list[int]] = [[], []]
    for z in range(length):
        for y in range(length):
            for x in range(length):
                coordinate = [x, y, z]
                left = x + length * y + length**2 * z
                colors[(x + y + z) % 2].append(left)
                for axis in range(3):
                    neighbor = coordinate.copy()
                    neighbor[axis] += 1
                    if neighbor[axis] == length:
                        if boundary == "open":
                            continue
                        neighbor[axis] = 0
                    right = neighbor[0] + length * neighbor[1] + length**2 * neighbor[2]
                    edges.append(tuple(sorted((left, right))))
    edges.sort()
    if len(set(edges)) != len(edges):
        raise AssertionError("Duplicate lattice bonds")
    endpoints = torch.tensor(edges, dtype=torch.long)
    rng = np.random.Generator(np.random.PCG64(disorder_seed))
    weights = 2 * rng.integers(0, 2, len(edges), dtype=np.int64) - 1
    model = EdgeIsingModel(
        endpoints[:, 0],
        endpoints[:, 1],
        torch.tensor(weights, dtype=torch.float64),
        torch.zeros(length**3, dtype=torch.float64),
        metadata={"length": length, "boundary": boundary, "disorder_seed": disorder_seed},
    )
    return CubicInstance(length, boundary, disorder_seed, model, tuple(map(torch.tensor, colors)))


def explicit_instance(length: int, boundary: str, bonds: list) -> CubicInstance:
    """Accept any +/-1 assignment on the complete declared cubic lattice."""
    template = cubic_instance(length, boundary)
    canonical = {}
    for bond in bonds:
        if not isinstance(bond, (list, tuple)) or len(bond) != 3:
            raise ValueError("Each bond must be [left, right, coupling]")
        a, b, weight = bond
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (a, b)):
            raise ValueError("Bond endpoints must be integers")
        if isinstance(weight, bool) or weight not in (-1, 1):
            raise ValueError("Explicit couplings must be +/-1")
        edge = tuple(sorted((a, b)))
        if edge in canonical:
            raise ValueError("Duplicate undirected bond")
        canonical[edge] = int(weight)
    edges = list(
        zip(template.model.edge_left.tolist(), template.model.edge_right.tolist(), strict=True)
    )
    if set(canonical) != set(edges):
        raise ValueError("Bonds must exactly cover the declared cubic nearest-neighbor graph")
    ordered = [[a, b, canonical[a, b]] for a, b in edges]
    identity = hashlib.sha256(json.dumps([length, boundary, ordered]).encode()).hexdigest()
    model = EdgeIsingModel(
        template.model.edge_left,
        template.model.edge_right,
        torch.tensor([row[2] for row in ordered], dtype=torch.float64),
        template.model.bias,
        metadata={"length": length, "boundary": boundary, "coupling_id": identity},
    )
    return replace(template, model=model, disorder_seed=None, coupling_id=identity)


def instance_from_problem(problem: dict) -> CubicInstance:
    """Load inline bonds or a self-contained JSON input; reject ambiguous sources."""
    if "instance_file" in problem:
        if "bonds" in problem or "disorder_seed" in problem:
            raise ValueError("instance_file cannot be combined with bonds or disorder_seed")
        payload = json.loads(Path(problem["instance_file"]).read_text())
        if set(payload) != {"length", "boundary", "bonds"}:
            raise ValueError("Instance JSON requires exactly length, boundary and bonds")
        for key in ("length", "boundary"):
            if key in problem and problem[key] != payload[key]:
                raise ValueError("Instance file and problem geometry disagree")
        return explicit_instance(**payload)
    if "bonds" in problem:
        if "disorder_seed" in problem:
            raise ValueError("bonds cannot be combined with disorder_seed")
        return explicit_instance(
            problem["length"], problem.get("boundary", "open"), problem["bonds"]
        )
    return cubic_instance(
        problem["length"], problem.get("boundary", "open"), problem.get("disorder_seed", 0)
    )


def instance_payload(instance: CubicInstance) -> dict:
    return {
        "length": instance.length,
        "boundary": instance.boundary,
        "bonds": [
            [a, b, int(j)]
            for a, b, j in zip(
                instance.model.edge_left.tolist(),
                instance.model.edge_right.tolist(),
                instance.model.edge_couplings.tolist(),
                strict=True,
            )
        ],
    }


def exact_ground_state(instance: CubicInstance) -> dict:
    """Exact min-sum layer transfer for open L <= 3, including degeneracy.

    All 2^(L^2) layer states participate. Costs and counts are integers; no
    heuristic reference or optimization-library tolerance is used.
    """
    length = instance.length
    if instance.boundary != "open" or length > 3:
        raise ValueError("Exact certification supports only open length 2 or 3")
    width = length**2
    states = 2 * ((np.arange(2**width)[:, None] >> np.arange(width)) & 1) - 1
    left = instance.model.edge_left.numpy()
    right = instance.model.edge_right.numpy()
    weights = instance.model.edge_couplings.numpy().astype(np.int64)
    intra = []
    for layer in range(length):
        mask = (left // width == layer) & (right // width == layer)
        intra.append(
            -(states[:, left[mask] % width] * states[:, right[mask] % width] * weights[mask]).sum(1)
        )
    costs = intra[0]
    counts = np.ones(len(states), dtype=np.int64)
    parents = []
    for layer in range(1, length):
        mask = (left // width == layer - 1) & (right // width == layer)
        interaction = (
            -(states[:, left[mask] % width] * weights[mask]) @ states[:, right[mask] % width].T
        )
        candidates = costs[:, None] + interaction
        minimum = candidates.min(axis=0)
        parents.append(candidates.argmin(axis=0))
        counts = ((candidates == minimum) * counts[:, None]).sum(axis=0)
        costs = minimum + intra[layer]
    energy = int(costs.min())
    degeneracy = int(counts[costs == energy].sum())
    index = int(costs.argmin())
    path = [states[index]]
    for parent in reversed(parents):
        index = int(parent[index])
        path.append(states[index])
    witness = np.concatenate(path[::-1]).astype(np.int8)
    if int(instance.energy_numpy(witness)) != energy:
        raise AssertionError("Exact ground-state witness failed independent certification")
    return {
        "method": "integer-min-sum-layer-transfer-v1",
        "energy": energy,
        "degeneracy": degeneracy,
        "states_per_layer": len(states),
        "spins": witness.tolist(),
    }
