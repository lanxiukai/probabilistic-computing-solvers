"""Exact integer execution kernels for the existing wide Fermat proposal.

The restoring square root consumes two radicand bits per iteration. For a
radicand below 2**124 the final root fits 62 bits; the temporary remainder
fits unsigned 64 bits. No floating-point estimate or altered proposal mass
is introduced by these execution kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sqrt124(low, high, BLOCK: tl.constexpr):
    root = tl.full((BLOCK,), 0, tl.uint64)
    remainder = tl.full((BLOCK,), 0, tl.uint64)
    for iteration in range(62):
        shift = 122 - 2 * iteration
        pair = (
            tl.where(
                shift >= 64,
                high >> tl.maximum(shift - 64, 0),
                low >> tl.minimum(shift, 63),
            )
            & 3
        )
        remainder = (remainder << 2) | pair
        trial = (root << 2) | 1
        take = remainder >= trial
        remainder = tl.where(take, remainder - trial, remainder)
        root = (root << 1) | take.to(tl.uint64)
    return root


@triton.jit
def _sqrt_words(A0, A1, A2, A3, Out, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < SIZE
    w0 = tl.load(A0 + index, active, 0).to(tl.uint64)
    w1 = tl.load(A1 + index, active, 0).to(tl.uint64)
    w2 = tl.load(A2 + index, active, 0).to(tl.uint64)
    w3 = tl.load(A3 + index, active, 0).to(tl.uint64)
    root = _sqrt124(w0 | (w1 << 32), w2 | (w3 << 32), BLOCK)
    tl.store(Out + index, root.to(tl.int64), active)


@triton.jit
def _radius(
    X, Out, SIZE: tl.constexpr, NLOW: tl.constexpr, NHIGH: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < SIZE
    x = tl.load(X + index, active, 0).to(tl.uint64)
    low, high = x & 0xFFFFFFFF, x >> 32
    p00, p01, p11 = low * low, low * high, high * high
    middle = (p00 >> 32) + 2 * (p01 & 0xFFFFFFFF)
    upper = 2 * (p01 >> 32) + (p11 & 0xFFFFFFFF) + (middle >> 32)
    square_low = (p00 & 0xFFFFFFFF) | ((middle & 0xFFFFFFFF) << 32)
    square_high = (upper & 0xFFFFFFFF) | (((p11 >> 32) + (upper >> 32)) << 32)
    target_low = tl.full((BLOCK,), NLOW, tl.uint64)
    delta_low = square_low - target_low
    delta_high = (
        square_high - tl.full((BLOCK,), NHIGH, tl.uint64) - (square_low < target_low).to(tl.uint64)
    )
    y = _sqrt124(delta_low, delta_high, BLOCK)
    y = y - ((x + y + 1) & 1)
    tl.store(Out + index, y.to(tl.int64), active)


@triton.jit
def _small_qr(
    X,
    Table,
    Out,
    SIZE: tl.constexpr,
    MODULI: tl.constexpr,
    OFFSETS: tl.constexpr,
    NQR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < SIZE
    x = tl.load(X + index, active, 0).to(tl.uint64)
    low, high = (x & 0xFFFFFFFF).to(tl.uint32), (x >> 32).to(tl.uint32)
    matches = tl.full((BLOCK,), True, tl.int1)
    for i in tl.static_range(NQR):
        residue = ((low % MODULI[i]) + (high % MODULI[i]) * ((1 << 32) % MODULI[i])) % MODULI[i]
        matches &= tl.load(Table + OFFSETS[i] + residue, active, 0).to(tl.int1)
    tl.store(Out + index, matches, active)


def floor_sqrt_words(words):
    values = tuple(word.contiguous() for word in words)
    output = torch.empty(values[0].shape, device=values[0].device, dtype=torch.int64)
    if output.numel():
        _sqrt_words[(triton.cdiv(output.numel(), 128),)](
            *values, output, output.numel(), 128, num_warps=4
        )
    return output


def radius(x, target):
    value = x.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    if output.numel():
        _radius[(triton.cdiv(output.numel(), 128),)](
            value, output, output.numel(), target & ((1 << 64) - 1), target >> 64, 128, num_warps=4
        )
    return output


def pre_matches(sampler, x):
    value = x.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.bool)
    if output.numel():
        _small_qr[(triton.cdiv(output.numel(), 128),)](
            value,
            sampler.qr_flat,
            output,
            output.numel(),
            sampler.extra_moduli,
            sampler.qr_offsets,
            len(sampler.extra_moduli),
            128,
            num_warps=4,
        )
    return output
