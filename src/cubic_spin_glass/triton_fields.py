"""Exact integer local fields for bipolar, zero-field cubic EA states."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fields_kernel(
    states,
    neighbors,
    weights,
    result,
    size: tl.constexpr,
    sites: tl.constexpr,
    degree: tl.constexpr,
    block: tl.constexpr,
):
    replica = tl.program_id(1)
    site = tl.program_id(0) * block + tl.arange(0, block)
    mask = site < sites
    field = tl.full((block,), 0, tl.int32)
    for neighbor in tl.static_range(degree):
        offset = site * degree + neighbor
        index = tl.load(neighbors + offset, mask, other=0)
        weight = tl.load(weights + offset, mask, other=0)
        spin = tl.load(states + replica * size + index, mask, other=0).to(tl.int32)
        field += weight * spin
    tl.store(result + replica * sites + site, field.to(tl.float64), mask)


class CubicFields:
    """Fuse only the neighbor sum; FP64 probabilities and random draws stay in Torch.

    Every maintained spin is +/-1 and every nonzero bond is +/-1. Each field
    sums at most six unit terms, making int32 and FP64 sums exactly equal.
    """

    def __init__(self, sparse, colors):
        if sparse.device.type != "cuda" or sparse.dtype != torch.float64:
            raise ValueError("Fused cubic fields require CUDA FP64 states")
        if (
            sparse.maximum_degree > 6
            or bool(sparse.bias.any())
            or not bool((sparse.edge_couplings.abs() == 1).all())
        ):
            raise ValueError("Fused cubic fields require zero-field degree <= 6 +/-1 bonds")
        self.sparse = sparse
        self.cached = [
            (
                sparse.neighbors[color].to(torch.int32).contiguous(),
                sparse.couplings[color].to(torch.int32).contiguous(),
            )
            for color in colors
        ]

    def __call__(self, spins, color):
        sparse = self.sparse
        if (
            spins.device != sparse.device
            or spins.dtype != torch.float64
            or spins.shape[-1] != sparse.size
            or not spins.is_contiguous()
        ):
            raise ValueError("Fused cubic fields require contiguous FP64 solver states")
        neighbors, weights = self.cached[color]
        sites, degree = neighbors.shape
        result = torch.empty((*spins.shape[:-1], sites), device=spins.device, dtype=spins.dtype)
        _fields_kernel[(triton.cdiv(sites, 256), spins.numel() // sparse.size)](
            spins,
            neighbors,
            weights,
            result,
            size=sparse.size,
            sites=sites,
            degree=degree,
            block=256,
            num_warps=4,
        )
        return result
