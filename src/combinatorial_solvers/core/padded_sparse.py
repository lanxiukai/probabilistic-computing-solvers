"""Degree-padded sparse storage for pairwise Ising models.

The representation is intentionally simple: every vertex owns a fixed-width row of
neighbor indices and couplings, padded with zero-weight entries. It is a portable
PyTorch baseline for bounded-degree graphs and a stable interface for a future
Triton or custom CUDA kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .edge_model import edge_terms
from .model import IsingModel


@dataclass(frozen=True)
class PaddedSparseIsing:
    """An Ising model stored as degree-padded adjacency and unique edge arrays."""

    neighbors: torch.Tensor
    couplings: torch.Tensor
    bias: torch.Tensor
    edge_left: torch.Tensor
    edge_right: torch.Tensor
    edge_couplings: torch.Tensor
    constant: float

    def __post_init__(self) -> None:
        if self.neighbors.ndim != 2 or self.couplings.shape != self.neighbors.shape:
            raise ValueError("neighbors and couplings must have matching [variables, degree] shape")
        if self.neighbors.dtype != torch.long:
            raise TypeError("neighbors must use torch.long indices")
        if self.bias.ndim != 1 or self.bias.shape[0] != self.neighbors.shape[0]:
            raise ValueError("bias must match the padded adjacency variable dimension")
        if self.couplings.dtype != self.bias.dtype:
            raise TypeError("couplings and bias must share a floating-point dtype")
        if not self.couplings.is_floating_point():
            raise TypeError("sparse Ising coefficients must use a floating-point dtype")
        tensors = (
            self.neighbors,
            self.couplings,
            self.bias,
            self.edge_left,
            self.edge_right,
            self.edge_couplings,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all sparse Ising tensors must use the same device")
        if self.edge_left.dtype != torch.long or self.edge_right.dtype != torch.long:
            raise TypeError("edge endpoints must use torch.long indices")
        if self.edge_left.ndim != 1 or self.edge_right.shape != self.edge_left.shape:
            raise ValueError("edge endpoints must be matching one-dimensional tensors")
        if self.edge_couplings.shape != self.edge_left.shape:
            raise ValueError("edge couplings must match the unique edge arrays")
        if self.edge_couplings.dtype != self.bias.dtype:
            raise TypeError("edge couplings and bias must share dtype")
        if self.size and torch.any((self.neighbors < 0) | (self.neighbors >= self.size)):
            raise IndexError("neighbors contain an out-of-range variable index")
        if torch.any(~torch.isfinite(self.couplings)) or torch.any(~torch.isfinite(self.bias)):
            raise ValueError("sparse Ising coefficients must be finite")

    @classmethod
    def from_model(
        cls,
        model: IsingModel,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PaddedSparseIsing:
        """Build padded adjacency on CPU, then transfer only sparse storage."""

        selected_dtype = model.bias.dtype if dtype is None else dtype
        if selected_dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")
        host_bias = model.bias.detach().to(device="cpu", dtype=selected_dtype)
        source_left, source_right, source_weights = edge_terms(model)
        edge_left, edge_right = source_left.cpu(), source_right.cpu()
        edge_weights = source_weights.detach().to(device="cpu", dtype=selected_dtype)
        nonzero = edge_weights != 0
        edge_left, edge_right, edge_weights = (
            edge_left[nonzero],
            edge_right[nonzero],
            edge_weights[nonzero],
        )

        degrees = torch.zeros(model.size, dtype=torch.long)
        if edge_left.numel():
            degrees.scatter_add_(0, edge_left, torch.ones_like(edge_left))
            degrees.scatter_add_(0, edge_right, torch.ones_like(edge_right))
        maximum_degree = int(degrees.max().item()) if model.size else 0
        neighbors = torch.zeros((model.size, maximum_degree), dtype=torch.long)
        weights = torch.zeros((model.size, maximum_degree), dtype=selected_dtype)
        offsets = torch.zeros(model.size, dtype=torch.long)
        for left, right, weight in zip(
            edge_left.tolist(),
            edge_right.tolist(),
            edge_weights.tolist(),
            strict=True,
        ):
            left_offset = int(offsets[left].item())
            right_offset = int(offsets[right].item())
            neighbors[left, left_offset] = right
            weights[left, left_offset] = weight
            neighbors[right, right_offset] = left
            weights[right, right_offset] = weight
            offsets[left] += 1
            offsets[right] += 1

        selected_device = model.bias.device if device is None else torch.device(device)
        return cls(
            neighbors=neighbors.to(device=selected_device),
            couplings=weights.to(device=selected_device),
            bias=host_bias.to(device=selected_device),
            edge_left=edge_left.to(device=selected_device),
            edge_right=edge_right.to(device=selected_device),
            edge_couplings=edge_weights.to(device=selected_device),
            constant=float(model.constant),
        )

    @property
    def size(self) -> int:
        return int(self.bias.numel())

    @property
    def maximum_degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def edge_count(self) -> int:
        return int(self.edge_left.numel())

    @property
    def device(self) -> torch.device:
        return self.bias.device

    @property
    def dtype(self) -> torch.dtype:
        return self.bias.dtype

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.neighbors,
            self.couplings,
            self.bias,
            self.edge_left,
            self.edge_right,
            self.edge_couplings,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def local_fields(
        self,
        spins: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return local fields using only true or zero-padded neighbor entries."""

        if spins.shape[-1] != self.size:
            raise ValueError("Spin shape does not match the sparse model size")
        if spins.device != self.device:
            raise ValueError("spins and sparse model must use the same device")
        if indices is None:
            neighbors = self.neighbors
            couplings = self.couplings
            bias = self.bias
        else:
            selected = indices.to(device=self.device, dtype=torch.long)
            neighbors = self.neighbors.index_select(0, selected)
            couplings = self.couplings.index_select(0, selected)
            bias = self.bias.index_select(0, selected)
        values = spins.to(dtype=self.dtype)
        adjacent = values[..., neighbors]
        return (adjacent * couplings).sum(dim=-1) + bias

    def triton_local_fields(
        self,
        spins: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return local fields through the optional fused Triton CUDA kernel."""

        from .triton_sparse import triton_padded_local_fields

        return triton_padded_local_fields(
            spins,
            self.neighbors,
            self.couplings,
            self.bias,
            indices,
        )

    def triton_fused_counter_gibbs_update(
        self,
        spins: torch.Tensor,
        indices: torch.Tensor,
        *,
        beta: float,
        seed: int,
        counter_offset: int,
    ) -> torch.Tensor:
        """Fuse one color update under the explicit counter-v1 RNG protocol."""

        from .triton_sparse import triton_fused_counter_gibbs_update

        return triton_fused_counter_gibbs_update(
            spins,
            self.neighbors,
            self.couplings,
            self.bias,
            indices,
            beta=beta,
            seed=seed,
            counter_offset=counter_offset,
        )

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Evaluate the canonical Ising energy from unique undirected edges."""

        if spins.shape[-1] != self.size:
            raise ValueError("Spin shape does not match the sparse model size")
        if spins.device != self.device:
            raise ValueError("spins and sparse model must use the same device")
        values = spins.to(dtype=self.dtype)
        pair = -(
            values[..., self.edge_left] * values[..., self.edge_right] * self.edge_couplings
        ).sum(dim=-1)
        linear = -(values * self.bias).sum(dim=-1)
        return pair + linear + self.constant
