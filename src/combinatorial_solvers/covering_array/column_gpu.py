"""Exact two-row same-column conditional moves for covering arrays."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def advance_column(
    X,
    Values,
    Best,
    Counts,
    Cost,
    BestCost,
    Visited,
    Changed,
    Noise,
    Parameters,
    Steps,
    Offset,
    Cycle,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    Pairs: tl.constexpr,
    KK: tl.constexpr,
    VV: tl.constexpr,
    PP: tl.constexpr,
    LL: tl.constexpr,
    Chunk: tl.constexpr,
    Method: tl.constexpr,
):
    replica = tl.program_id(0)
    k = tl.arange(0, KK)
    pattern = tl.arange(0, PP)
    levels = tl.arange(0, VV)
    cells = tl.arange(0, LL)
    valid = pattern < V * V
    new_a = pattern % V
    new_b = pattern // V
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    for step in range(Steps):
        urow = tl.load(Noise + replica * Chunk + step)
        ucolumn = tl.load(Noise + (R + replica) * Chunk + step)
        usecond = tl.load(Noise + (2 * R + replica) * Chunk + step)
        scheduled = Offset + step < N * K
        row_a = tl.where(
            scheduled, (Offset + step) // K, tl.minimum((urow * N).to(tl.int32), N - 1)
        )
        column = tl.where(
            scheduled, (Offset + step) % K, tl.minimum((ucolumn * K).to(tl.int32), K - 1)
        )
        row_b = tl.minimum((usecond * (N - 1)).to(tl.int32), N - 2)
        row_b += (row_b >= row_a).to(tl.int32)
        values_a = tl.load(Values + (replica * N + row_a) * K + k, k < K, 0)
        values_b = tl.load(Values + (replica * N + row_b) * K + k, k < K, 0)
        old_a = tl.sum(tl.where(k == column, values_a, 0), 0)
        old_b = tl.sum(tl.where(k == column, values_b, 0), 0)
        first = tl.minimum(column, k)
        second = tl.maximum(column, k)
        pair = first * (2 * K - first - 1) // 2 + second - first - 1
        mask = (k < K) & (k != column)
        old_code_a = tl.where(column < k, old_a * V + values_a, values_a * V + old_a)
        old_code_b = tl.where(column < k, old_b * V + values_b, values_b * V + old_b)
        new_code_a = tl.where(
            column < k,
            new_a[:, None] * V + values_a[None, :],
            values_a[None, :] * V + new_a[:, None],
        )
        new_code_b = tl.where(
            column < k,
            new_b[:, None] * V + values_b[None, :],
            values_b[None, :] * V + new_b[:, None],
        )
        delta = tl.full((PP,), 0, tl.int32)
        for slot in tl.static_range(4):
            if slot == 0:
                code = old_code_a[None, :] + tl.zeros((PP, 1), tl.int64)
                unique = tl.full((PP, KK), True, tl.int1)
            elif slot == 1:
                code = old_code_b[None, :] + tl.zeros((PP, 1), tl.int64)
                unique = code != old_code_a[None, :]
            elif slot == 2:
                code = new_code_a
                unique = (code != old_code_a[None, :]) & (code != old_code_b[None, :])
            else:
                code = new_code_b
                unique = (
                    (code != old_code_a[None, :])
                    & (code != old_code_b[None, :])
                    & (code != new_code_a)
                )
            count = tl.load(
                Counts + (replica * Pairs + pair[None, :]) * V * V + code,
                valid[:, None] & mask[None, :] & unique,
                0,
            )
            adjustment = (
                (code == new_code_a).to(tl.int32)
                + (code == new_code_b).to(tl.int32)
                - (code == old_code_a[None, :]).to(tl.int32)
                - (code == old_code_b[None, :]).to(tl.int32)
            )
            delta += tl.sum(
                tl.where(
                    valid[:, None] & mask[None, :] & unique,
                    ((count + adjustment) == 0).to(tl.int32) - (count == 0).to(tl.int32),
                    0,
                ),
                1,
            )
        beta = low * libdevice.exp(
            ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
        )
        u = tl.load(Noise + (3 * R + replica) * Chunk + step)
        old_pattern = old_a + V * old_b
        if Method == 1:
            proposed = tl.minimum((u * V * V).to(tl.int32), V * V - 1)
            proposed_delta = tl.sum(tl.where(pattern == proposed, delta, 0), 0)
            accept = tl.load(Noise + (4 * R + replica) * Chunk + step)
            selected = tl.where(
                accept < libdevice.exp(-beta * tl.maximum(proposed_delta, 0).to(tl.float64)),
                proposed,
                old_pattern,
            )
        else:
            minimum = tl.min(tl.where(valid, delta, 2147483647), 0)
            if Method == 2:
                weights = (valid & (delta == minimum)).to(tl.float64)
            else:
                weights = tl.where(
                    valid, libdevice.exp(-beta * (delta - minimum).to(tl.float64)), 0.0
                )
            selected = tl.min(
                tl.where(valid & (tl.cumsum(weights, 0) > u * tl.sum(weights, 0)), pattern, PP), 0
            )
        cost += tl.sum(tl.where(pattern == selected, delta, 0), 0)
        changed += selected != old_pattern
        picked_a = selected % V
        picked_b = selected // V
        picked_code_a = tl.where(column < k, picked_a * V + values_a, values_a * V + picked_a)
        picked_code_b = tl.where(column < k, picked_b * V + values_b, values_b * V + picked_b)
        tl.atomic_add(
            Counts + (replica * Pairs + pair) * V * V + old_code_a, -1, mask & (old_a != picked_a)
        )
        tl.atomic_add(
            Counts + (replica * Pairs + pair) * V * V + picked_code_a, 1, mask & (old_a != picked_a)
        )
        tl.atomic_add(
            Counts + (replica * Pairs + pair) * V * V + old_code_b, -1, mask & (old_b != picked_b)
        )
        tl.atomic_add(
            Counts + (replica * Pairs + pair) * V * V + picked_code_b, 1, mask & (old_b != picked_b)
        )
        tl.store(Values + (replica * N + row_a) * K + column, picked_a)
        tl.store(Values + (replica * N + row_b) * K + column, picked_b)
        tl.store(
            X + ((replica * N + row_a) * K + column) * V + levels,
            (levels == picked_a).to(tl.int8),
            levels < V,
        )
        tl.store(
            X + ((replica * N + row_b) * K + column) * V + levels,
            (levels == picked_b).to(tl.int8),
            levels < V,
        )
        tl.store(Visited + ((replica * N + row_a) * K + column) * V + levels, 1, levels < V)
        tl.store(Visited + ((replica * N + row_b) * K + column) * V + levels, 1, levels < V)
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            current = tl.load(Values + replica * N * K + cells, cells < N * K, 0)
            tl.store(Best + replica * N * K + cells, current, cells < N * K)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)
