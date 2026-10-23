"""Exact fused energy for bipolar, zero-field cubic EA states up to 16 cubed."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _energy_kernel(
    states,
    left,
    right,
    couplings,
    result,
    size: tl.constexpr,
    edges: tl.constexpr,
    block: tl.constexpr,
):
    replica = tl.program_id(0)
    edge = tl.arange(0, block)
    mask = edge < edges
    a = tl.load(left + edge, mask=mask, other=0)
    b = tl.load(right + edge, mask=mask, other=0)
    first = tl.load(states + replica * size + a, mask=mask, other=0).to(tl.int32)
    second = tl.load(states + replica * size + b, mask=mask, other=0).to(tl.int32)
    weight = tl.load(couplings + edge, mask=mask, other=0).to(tl.int32)
    energy = -tl.sum(first * second * weight, axis=0, dtype=tl.int32)
    tl.store(result + replica, energy.to(tl.float64))


class CubicEnergy:
    """Internal solver kernel: bipolar states are maintained by every update rule.

    At most 11,520 +/-1 edge terms are accumulated, so int32 is exact. Output
    remains FP64 and RNG generation, local fields and acceptance are untouched.
    The first invocation compiles in the ordinary preparation timing boundary.
    """

    def __init__(self, sparse, instance):
        if instance.boundary != "open" or not 2 <= instance.length <= 16:
            raise ValueError("lattice-v2 requires a complete open lattice with 2 <= L <= 16")
        if sparse.device.type != "cuda" or sparse.dtype != torch.float64:
            raise ValueError("lattice-v2 requires CUDA FP64 states")
        if (
            sparse.constant != 0
            or bool(sparse.bias.any())
            or not bool((sparse.edge_couplings.abs() == 1).all())
            or sparse.edge_count > 11520
        ):
            raise ValueError("Fused energy requires bounded zero-field +/-1 couplings")
        self.sparse = sparse
        self.block = triton.next_power_of_2(sparse.edge_count)

    def __call__(self, spins):
        sparse = self.sparse
        if (
            spins.device != sparse.device
            or spins.dtype != torch.float64
            or spins.shape[-1] != sparse.size
            or not spins.is_contiguous()
        ):
            raise ValueError("Fused energy requires contiguous FP64 solver states")
        result = torch.empty(spins.shape[:-1], device=spins.device, dtype=torch.float64)
        _energy_kernel[(spins.numel() // sparse.size,)](
            spins,
            sparse.edge_left,
            sparse.edge_right,
            sparse.edge_couplings,
            result,
            size=sparse.size,
            edges=sparse.edge_count,
            block=self.block,
            num_warps=8,
        )
        return result
