"""Canonical pairwise Ising models stored as unique edges, without dense matrices."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from .model import ClampResult, IsingModel


@dataclass(frozen=True)
class EdgeIsingModel:
    """H=C-h*s-sum_edges J*s_i*s_j, with sorted unique i<j edges."""

    edge_left: torch.Tensor
    edge_right: torch.Tensor
    edge_couplings: torch.Tensor
    bias: torch.Tensor
    constant: float = 0.0
    names: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        left, right, weights = self.edge_left, self.edge_right, self.edge_couplings
        if left.ndim != 1 or right.shape != left.shape or weights.shape != left.shape:
            raise ValueError("Unique edge arrays must have the same one-dimensional shape")
        if left.dtype != torch.long or right.dtype != torch.long:
            raise TypeError("Edge endpoints must use torch.long")
        if (
            self.bias.ndim != 1
            or not self.bias.is_floating_point()
            or weights.dtype != self.bias.dtype
        ):
            raise TypeError("Edge weights and one-dimensional bias must share a floating dtype")
        if len({x.device for x in (left, right, weights, self.bias)}) != 1:
            raise ValueError("All edge-model arrays must share a device")
        if bool(torch.any((left < 0) | (left >= right) | (right >= self.size))):
            raise ValueError("Require in-range, upper-triangular edges")
        keys = left * self.size + right
        if len(keys) > 1 and bool(torch.any(keys[1:] <= keys[:-1])):
            raise ValueError("Edges must be lexicographically sorted and unique")
        if not math.isfinite(self.constant) or not bool(
            torch.isfinite(weights).all() and torch.isfinite(self.bias).all()
        ):
            raise ValueError("Ising coefficients must be finite")
        if bool((weights == 0).any()):
            raise ValueError("Zero edges must be removed")
        names = self.names or tuple(f"x_{i}" for i in range(self.size))
        if len(names) != self.size or len(set(names)) != self.size:
            raise ValueError("Names must be unique and match the model size")
        object.__setattr__(self, "names", tuple(names))

    @property
    def size(self):
        return self.bias.numel()

    @property
    def storage_bytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in (self.edge_left, self.edge_right, self.edge_couplings, self.bias)
        )

    def to(self, device=None, dtype=None):
        return EdgeIsingModel(
            self.edge_left.to(device=device),
            self.edge_right.to(device=device),
            self.edge_couplings.to(device=device, dtype=dtype),
            self.bias.to(device=device, dtype=dtype),
            self.constant,
            self.names,
            self.metadata,
        )

    def owned_cpu_fp64(self):
        return EdgeIsingModel(
            self.edge_left.detach().to(device="cpu", copy=True),
            self.edge_right.detach().to(device="cpu", copy=True),
            self.edge_couplings.detach().to(device="cpu", dtype=torch.float64, copy=True),
            self.bias.detach().to(device="cpu", dtype=torch.float64, copy=True),
            float(self.constant),
            tuple(self.names),
            dict(self.metadata),
        )

    def energy(self, spins):
        if spins.shape[-1] != self.size:
            raise ValueError("Spin shape does not match the edge model")
        values = spins.to(device=self.bias.device, dtype=self.bias.dtype)
        if self.size == 0:
            return torch.full(
                values.shape[:-1], self.constant, dtype=self.bias.dtype, device=self.bias.device
            )
        flat = values.reshape(-1, self.size)
        chunks = []
        for block in flat.split(128):
            pairs = (
                block[:, self.edge_left] * block[:, self.edge_right] * self.edge_couplings
            ).sum(1)
            chunks.append(self.constant - pairs - (block * self.bias).sum(1))
        return (
            torch.cat(chunks).reshape(values.shape[:-1])
            if chunks
            else torch.empty(values.shape[:-1], device=self.bias.device, dtype=self.bias.dtype)
        )

    def local_fields(self, spins, indices=None):
        from .padded_sparse import PaddedSparseIsing

        return PaddedSparseIsing.from_model(self).local_fields(spins, indices)

    def clamp(self, fixed):
        names = {name: i for i, name in enumerate(self.names)}
        resolved = {}
        for key, value in fixed.items():
            index = names[key] if isinstance(key, str) else int(key)
            if not 0 <= index < self.size:
                raise IndexError("Clamp index out of range")
            if value not in (-1, 1):
                raise ValueError("Fixed Ising values must be bipolar")
            if index in resolved and resolved[index] != int(value):
                raise ValueError("Conflicting clamps")
            resolved[index] = int(value)
        fixed_indices = tuple(sorted(resolved))
        free = tuple(i for i in range(self.size) if i not in resolved)
        mapping = {node: i for i, node in enumerate(free)}
        bias = self.bias.detach().cpu().tolist()
        constant = float(self.constant) - sum(bias[node] * spin for node, spin in resolved.items())
        edges = {}
        for a, b, w in zip(
            self.edge_left.cpu().tolist(),
            self.edge_right.cpu().tolist(),
            self.edge_couplings.cpu().tolist(),
            strict=True,
        ):
            if a in resolved and b in resolved:
                constant -= w * resolved[a] * resolved[b]
            elif a in resolved:
                bias[b] += w * resolved[a]
            elif b in resolved:
                bias[a] += w * resolved[b]
            else:
                edges[(mapping[a], mapping[b])] = w
        model = from_edges(
            edges,
            [bias[i] for i in free],
            constant,
            names=tuple(self.names[i] for i in free),
            dtype=self.bias.dtype,
            metadata={**self.metadata, "clamped_variables": len(resolved)},
        ).to(device=self.bias.device)
        return ClampResult(
            model, self.names, free, fixed_indices, tuple(resolved[i] for i in fixed_indices)
        )

    def to_dense(self):
        """Explicit conversion for small equivalence fixtures, never implicit."""
        coupling = torch.zeros(
            (self.size, self.size), dtype=self.bias.dtype, device=self.bias.device
        )
        coupling[self.edge_left, self.edge_right] = self.edge_couplings
        coupling[self.edge_right, self.edge_left] = self.edge_couplings
        return IsingModel(coupling, self.bias.clone(), self.constant, self.names, self.metadata)


def from_edges(edges, bias, constant, *, names, dtype, metadata):
    ordered = sorted((a, b, float(w)) for (a, b), w in edges.items() if w != 0)
    left = torch.tensor([a for a, _, _ in ordered], dtype=torch.long)
    right = torch.tensor([b for _, b, _ in ordered], dtype=torch.long)
    weights = torch.tensor([w for _, _, w in ordered], dtype=dtype)
    return EdgeIsingModel(
        left, right, weights, torch.tensor(bias, dtype=dtype), constant, names, metadata
    )


def edge_terms(model):
    """Expose the same upper-triangle ordering for dense and sparse models."""
    if isinstance(model, EdgeIsingModel):
        return model.edge_left, model.edge_right, model.edge_couplings
    left, right = torch.where(torch.triu(model.coupling != 0, diagonal=1))
    return left, right, model.coupling[left, right]
