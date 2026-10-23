"""Integer-only coordinate filter; never computes a factor or a square root."""

import torch
import triton
import triton.language as tl


@triton.jit
def _coordinate_filter(
    Words,
    CRT,
    QR,
    First,
    Counts,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    D_MASK: tl.constexpr,
    INDEX_COUNT: tl.constexpr,
    M: tl.constexpr,
    XMIN_MOD: tl.constexpr,
    XMIN: tl.constexpr,
    XMAX: tl.constexpr,
    RADICES: tl.constexpr,
    CRT_OFFSETS: tl.constexpr,
    NTABLES: tl.constexpr,
    MODULI: tl.constexpr,
    QR_OFFSETS: tl.constexpr,
    NQR: tl.constexpr,
    PRE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    replica, tile = tl.program_id(0), tl.program_id(1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    active = index < K
    base = Words + replica * K * 8 + index * 8
    w0 = tl.load(base, active, 0).to(tl.uint32)
    w1 = tl.load(base + 1, active, 0).to(tl.uint64)
    w2 = tl.load(base + 2, active, 0).to(tl.uint64)
    original_rank = ((w1 | ((w2 & 0x7FFFFFFF) << 32)) & D_MASK).to(tl.int64)
    rank = original_rank.to(tl.uint64)
    value = tl.full((BLOCK,), 0, tl.int64)
    for i in tl.static_range(NTABLES):
        next_rank = rank // RADICES[i]
        digit = (rank - next_rank * RADICES[i]).to(tl.int32)
        residue = tl.load(CRT + CRT_OFFSETS[i] + digit, active, 0)
        total = value + residue
        value = tl.where(total >= M, total - M, total)
        rank = next_rank
    # All division/remainder operands stay nonnegative, including translation.
    translated = value - XMIN_MOD
    residue_offset = tl.where(translated < 0, translated + M, translated)
    offset = residue_offset + (rank * M).to(tl.int64)
    x = (offset + XMIN).to(tl.int64)
    uniform = ((w0 >> 1) & 15) == 0
    biased = (~uniform) & (original_rank < INDEX_COUNT) & (x >= XMIN) & (x <= XMAX)
    w3 = tl.load(base + 3, active, 0).to(tl.uint64)
    w4 = tl.load(base + 4, active, 0).to(tl.uint64)
    w5 = tl.load(base + 5, active, 0).to(tl.uint64)
    w6 = tl.load(base + 6, active, 0).to(tl.uint64)
    a = w3 | (w4 << 32)
    b = w5 | (w6 << 32)
    if WIDTH < 64:
        a = a & ((1 << WIDTH) - 1)
        b = b & ((1 << WIDTH) - 1)
    ux = ((a >> 1) + (b >> 1) + (a & b & 1)).to(tl.int64)
    coordinate = tl.where(uniform, ux, x).to(tl.uint64)
    low = (coordinate & 0xFFFFFFFF).to(tl.uint32)
    high = (coordinate >> 32).to(tl.uint32)
    matches = tl.full((BLOCK,), True, tl.int1)
    for i in tl.static_range(NQR):
        remainder = ((low % MODULI[i]) + (high % MODULI[i]) * ((1 << 32) % MODULI[i])) % MODULI[i]
        allowed = tl.load(QR + QR_OFFSETS[i] + remainder, active, 0).to(tl.int1)
        matches = matches & allowed
    coin = ((w0 >> 5) & PRE_MASK) == 0
    eligible = active & (uniform | biased) & (matches | coin)
    first = tl.min(tl.where(eligible, index, K), 0)
    count = tl.sum(eligible.to(tl.int32), 0)
    tl.store(First + replica * BLOCKS + tile, first)
    tl.store(Counts + replica * BLOCKS + tile, count)


def preselect(sampler, words):
    block = 256
    replicas = len(words)
    blocks = triton.cdiv(sampler.retries, block)
    first = torch.empty((replicas, blocks), dtype=torch.long, device=words.device)
    counts = torch.empty((replicas, blocks), dtype=torch.int32, device=words.device)
    _coordinate_filter[(replicas, blocks)](
        words,
        sampler.crt_flat,
        sampler.qr_flat,
        first,
        counts,
        sampler.retries,
        blocks,
        sampler.width,
        sampler.envelope - 1,
        sampler.crt_count,
        sampler.crt_modulus,
        sampler.x_min % sampler.crt_modulus,
        sampler.signed_min,
        sampler.signed_max,
        sampler.crt_radices,
        sampler.crt_offsets,
        len(sampler.crt_radices),
        sampler.extra_moduli,
        sampler.qr_offsets,
        len(sampler.extra_moduli),
        sampler.pre_mask,
        block,
        num_warps=4,
    )
    selected = first.min(1).values
    valid = selected < sampler.retries
    return torch.where(valid, selected, 0), valid, counts.sum(1)
