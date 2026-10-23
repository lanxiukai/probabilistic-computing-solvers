"""Ground-state-preserving COPY-chain sparsification for pairwise Ising models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch

from .model import IsingModel


@dataclass(frozen=True)
class CopySparseModel:
    """A sparse model plus groups that decode duplicate variables."""

    model: IsingModel
    root_indices: tuple[int, ...]
    copy_groups: tuple[tuple[int, ...], ...]
    original_names: tuple[str, ...]
    copy_strengths: tuple[float, ...]

    def decode(self, sparse_spins: torch.Tensor) -> torch.Tensor:
        if sparse_spins.shape[-1] != self.model.size:
            raise ValueError("Sparse spin shape does not match the sparse model")
        roots = torch.tensor(self.root_indices, dtype=torch.long, device=sparse_spins.device)
        return sparse_spins.index_select(-1, roots)

    def copy_violations(self, sparse_spins: torch.Tensor) -> torch.Tensor:
        """Count duplicate spins that disagree with their logical root."""

        if sparse_spins.shape[-1] != self.model.size:
            raise ValueError("Sparse spin shape does not match the sparse model")
        result = torch.zeros(sparse_spins.shape[:-1], dtype=torch.int64, device=sparse_spins.device)
        for group in self.copy_groups:
            if len(group) < 2:
                continue
            index = torch.tensor(group, dtype=torch.long, device=sparse_spins.device)
            values = sparse_spins.index_select(-1, index)
            result += (values[..., 1:] != values[..., :1]).sum(dim=-1)
        return result


def copy_chain_sparsify(
    model: IsingModel,
    copy_strength: float | str = "safe",
    safety_factor: float = 1.05,
    margin: float = 1e-6,
) -> CopySparseModel:
    """Replace each logical edge endpoint by a COPY-linked port.

    The transformed graph has maximum degree at most three. With ``copy_strength="safe"``,
    each logical variable receives a conservative coupling larger than its total absolute
    external field. This preserves the original ground states after decoding, but the large
    dynamic range can slow annealing. A numeric strength is useful for empirical ablations.
    """

    if isinstance(copy_strength, str) and copy_strength != "safe":
        raise ValueError("copy_strength must be a positive number or 'safe'")
    if not isinstance(copy_strength, str) and (
        not math.isfinite(float(copy_strength)) or float(copy_strength) <= 0
    ):
        raise ValueError("copy_strength must be positive")
    if not math.isfinite(safety_factor) or safety_factor <= 1.0:
        raise ValueError("safety_factor must be greater than one")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")

    device = model.coupling.device
    dtype = model.coupling.dtype
    edges: list[tuple[int, int, float]] = []
    rows, columns = torch.where(torch.triu(model.coupling != 0, diagonal=1))
    for left, right in zip(rows.tolist(), columns.tolist(), strict=True):
        edges.append((left, right, float(model.coupling[left, right].item())))

    roots = list(range(model.size))
    next_index = model.size
    ports: list[dict[int, int]] = [dict() for _ in range(model.size)]
    names = list(model.names)
    for left, right, _ in edges:
        ports[left][right] = next_index
        names.append(f"{model.names[left]}__copy_for__{model.names[right]}")
        next_index += 1
        ports[right][left] = next_index
        names.append(f"{model.names[right]}__copy_for__{model.names[left]}")
        next_index += 1

    coupling = torch.zeros((next_index, next_index), dtype=dtype, device=device)
    bias = torch.zeros(next_index, dtype=dtype, device=device)
    bias[: model.size] = model.bias
    for left, right, weight in edges:
        left_port = ports[left][right]
        right_port = ports[right][left]
        coupling[left_port, right_port] += weight
        coupling[right_port, left_port] += weight

    strengths: list[float] = []
    groups: list[tuple[int, ...]] = []
    constant_compensation = 0.0
    for node in range(model.size):
        ordered_ports = [ports[node][neighbor] for neighbor in sorted(ports[node])]
        group = tuple([roots[node], *ordered_ports])
        groups.append(group)
        if isinstance(copy_strength, str):
            external_bound = float(model.bias[node].abs().item())
            external_bound += float(model.coupling[node].abs().sum().item())
            strength = safety_factor * external_bound + margin
            stored_strength = torch.tensor(strength, dtype=dtype, device=device)
            stored_bound = torch.tensor(external_bound, dtype=dtype, device=device)
            if not bool(torch.isfinite(stored_strength)) or not bool(
                stored_strength > stored_bound
            ):
                raise ValueError("Stored COPY strength does not exceed its safe bound")
            strength = float(stored_strength.item())
        else:
            strength = float(copy_strength)
        strengths.append(strength)
        for first, second in pairwise(group):
            coupling[first, second] += strength
            coupling[second, first] += strength
            constant_compensation += strength

    sparse = IsingModel(
        coupling=coupling,
        bias=bias,
        constant=model.constant + constant_compensation,
        names=tuple(names),
        metadata={
            **model.metadata,
            "representation": "copy_chain_sparse",
            "source_size": model.size,
            "source_edges": len(edges),
            "maximum_degree_bound": 3,
            "copy_strength_mode": copy_strength,
            "copy_constant_compensation": constant_compensation,
        },
    )
    return CopySparseModel(
        model=sparse,
        root_indices=tuple(roots),
        copy_groups=tuple(groups),
        original_names=model.names,
        copy_strengths=tuple(strengths),
    )
