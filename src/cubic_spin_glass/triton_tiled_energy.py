"""Bounded, two-stage integer energy reduction for open EA lattices through L=32."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _edge_tiles(
    states,
    left,
    right,
    weights,
    partial,
    size: tl.constexpr,
    edges: tl.constexpr,
    tiles: tl.constexpr,
    block: tl.constexpr,
):
    replica, tile = tl.program_id(0), tl.program_id(1)
    edge = tile * block + tl.arange(0, block)
    mask = edge < edges
    first = tl.load(left + edge, mask, other=0)
    second = tl.load(right + edge, mask, other=0)
    a = tl.load(states + replica * size + first, mask, other=0).to(tl.int32)
    b = tl.load(states + replica * size + second, mask, other=0).to(tl.int32)
    weight = tl.load(weights + edge, mask, other=0).to(tl.int32)
    value = -tl.sum(a * b * weight, axis=0, dtype=tl.int32)
    tl.store(partial + replica * tiles + tile, value)


@triton.jit
def _sum_tiles(partial, result, tiles: tl.constexpr, block: tl.constexpr):
    replica = tl.program_id(0)
    tile = tl.arange(0, block)
    value = tl.load(partial + replica * tiles + tile, tile < tiles, other=0)
    energy = tl.sum(value, axis=0, dtype=tl.int32)
    tl.store(result + replica, energy.to(tl.float64))


class TiledCubicEnergy:
    """Keep each CTA bounded while summing at most 95,232 signed unit terms.

    Both reduction stages are exact int32 sums. Returned energies remain FP64;
    no probability, state, update ordering, or random-number interface changes.
    """

    def __init__(self, sparse, instance):
        if instance.boundary != "open" or not 2 <= instance.length <= 32:
            raise ValueError("lattice-v4 requires a complete open lattice with 2 <= L <= 32")
        if sparse.device.type != "cuda" or sparse.dtype != torch.float64:
            raise ValueError("lattice-v4 requires CUDA FP64 states")
        if (
            sparse.constant != 0
            or bool(sparse.bias.any())
            or not bool((sparse.edge_couplings.abs() == 1).all())
            or sparse.edge_count != 3 * instance.length**2 * (instance.length - 1)
            or sparse.edge_count > 95232
        ):
            raise ValueError("Tiled energy requires complete zero-field +/-1 cubic bonds")
        self.sparse = sparse
        self.block = 2048
        self.tiles = triton.cdiv(sparse.edge_count, self.block)

    def __call__(self, spins):
        sparse = self.sparse
        if (
            spins.device != sparse.device
            or spins.dtype != torch.float64
            or spins.shape[-1] != sparse.size
            or not spins.is_contiguous()
        ):
            raise ValueError("Tiled energy requires contiguous FP64 solver states")
        replicas = spins.numel() // sparse.size
        partial = torch.empty((replicas, self.tiles), device=spins.device, dtype=torch.int32)
        result = torch.empty(spins.shape[:-1], device=spins.device, dtype=torch.float64)
        _edge_tiles[(replicas, self.tiles)](
            spins,
            sparse.edge_left,
            sparse.edge_right,
            sparse.edge_couplings,
            partial,
            size=sparse.size,
            edges=sparse.edge_count,
            tiles=self.tiles,
            block=self.block,
            num_warps=4,
        )
        _sum_tiles[(replicas,)](
            partial,
            result,
            tiles=self.tiles,
            block=triton.next_power_of_2(self.tiles),
            num_warps=4,
        )
        return result
