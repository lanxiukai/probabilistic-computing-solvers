"""Persistent exact coverage counts and same-row joint categorical heat baths."""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pair(a, b, K: tl.constexpr):
    low = tl.minimum(a, b)
    high = tl.maximum(a, b)
    return low * (2 * K - low - 1) // 2 + high - low - 1


@triton.jit
def _advance(
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
    Width: tl.constexpr,
    Method: tl.constexpr,
):
    replica = tl.program_id(0)
    k = tl.arange(0, KK)
    p = tl.arange(0, PP)
    v = tl.arange(0, VV)
    cells = tl.arange(0, LL)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    for step in range(Steps):
        urow = tl.load(Noise + replica * Chunk + step)
        ui = tl.load(Noise + (R + replica) * Chunk + step)
        scheduled = Offset + step < N * K
        row = tl.where(scheduled, (Offset + step) // K, tl.minimum((urow * N).to(tl.int32), N - 1))
        i = tl.where(scheduled, (Offset + step) % K, tl.minimum((ui * K).to(tl.int32), K - 1))
        old_row = tl.load(Values + (replica * N + row) * K + k, k < K, 0)
        old_i = tl.sum(tl.where(k == i, old_row, 0), 0)
        new_i = p % V
        valid = p < (V if Width == 1 else V * V)
        if Width == 2:
            uj = tl.load(Noise + (2 * R + replica) * Chunk + step)
            j = tl.minimum((uj * (K - 1)).to(tl.int32), K - 2)
            j += (j >= i).to(tl.int32)
            old_j = tl.sum(tl.where(k == j, old_row, 0), 0)
            new_j = p // V
            old_pattern = old_i + V * old_j
        else:
            old_pattern = old_i
        pair_i = _pair(i, k, K)
        mask_i = (k < K) & (k != i)
        if Width == 2:
            mask_i &= k != j
        old_code_i = tl.where(i < k, old_i * V + old_row, old_row * V + old_i)
        new_code_i = tl.where(
            i < k, new_i[:, None] * V + old_row[None, :], old_row[None, :] * V + new_i[:, None]
        )
        old_count_i = tl.load(Counts + (replica * Pairs + pair_i) * V * V + old_code_i, mask_i, 0)
        new_count_i = tl.load(
            Counts + (replica * Pairs + pair_i[None, :]) * V * V + new_code_i,
            valid[:, None] & mask_i[None, :],
            0,
        )
        delta = tl.sum(
            tl.where(
                valid[:, None] & mask_i[None, :] & (new_code_i != old_code_i[None, :]),
                (old_count_i[None, :] == 1).to(tl.int32) - (new_count_i == 0).to(tl.int32),
                0,
            ),
            1,
        )
        if Width == 2:
            pair_j = _pair(j, k, K)
            mask_j = (k < K) & (k != i) & (k != j)
            old_code_j = tl.where(j < k, old_j * V + old_row, old_row * V + old_j)
            new_code_j = tl.where(
                j < k, new_j[:, None] * V + old_row[None, :], old_row[None, :] * V + new_j[:, None]
            )
            old_count_j = tl.load(
                Counts + (replica * Pairs + pair_j) * V * V + old_code_j, mask_j, 0
            )
            new_count_j = tl.load(
                Counts + (replica * Pairs + pair_j[None, :]) * V * V + new_code_j,
                valid[:, None] & mask_j[None, :],
                0,
            )
            delta += tl.sum(
                tl.where(
                    valid[:, None] & mask_j[None, :] & (new_code_j != old_code_j[None, :]),
                    (old_count_j[None, :] == 1).to(tl.int32) - (new_count_j == 0).to(tl.int32),
                    0,
                ),
                1,
            )
            pair_ij = _pair(i, j, K)
            old_code_ij = tl.where(i < j, old_i * V + old_j, old_j * V + old_i)
            new_code_ij = tl.where(i < j, new_i * V + new_j, new_j * V + new_i)
            old_count_ij = tl.load(Counts + (replica * Pairs + pair_ij) * V * V + old_code_ij)
            new_count_ij = tl.load(
                Counts + (replica * Pairs + pair_ij) * V * V + new_code_ij, valid, 0
            )
            delta += tl.where(
                valid & (new_code_ij != old_code_ij),
                (old_count_ij == 1).to(tl.int32) - (new_count_ij == 0).to(tl.int32),
                0,
            )
        beta = low * libdevice.exp(
            ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
        )
        u = tl.load(Noise + (3 * R + replica) * Chunk + step)
        if Method == 1:
            proposed = tl.minimum(
                (u * (V if Width == 1 else V * V)).to(tl.int32), (V if Width == 1 else V * V) - 1
            )
            proposed_delta = tl.sum(tl.where(p == proposed, delta, 0), 0)
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
                tl.where(valid & (tl.cumsum(weights, 0) > u * tl.sum(weights, 0)), p, PP), 0
            )
        cost += tl.sum(tl.where(p == selected, delta, 0), 0)
        changed += selected != old_pattern
        picked_i = selected % V
        picked_code_i = tl.where(i < k, picked_i * V + old_row, old_row * V + picked_i)
        picked_count_i = tl.load(
            Counts + (replica * Pairs + pair_i) * V * V + picked_code_i, mask_i, 0
        )
        tl.store(
            Counts + (replica * Pairs + pair_i) * V * V + old_code_i,
            old_count_i - 1,
            mask_i & (old_i != picked_i),
        )
        tl.store(
            Counts + (replica * Pairs + pair_i) * V * V + picked_code_i,
            picked_count_i + 1,
            mask_i & (old_i != picked_i),
        )
        tl.store(Values + (replica * N + row) * K + i, picked_i)
        tl.store(X + ((replica * N + row) * K + i) * V + v, (v == picked_i).to(tl.int8), v < V)
        tl.store(Visited + ((replica * N + row) * K + i) * V + v, 1, v < V)
        if Width == 2:
            picked_j = selected // V
            picked_code_j = tl.where(j < k, picked_j * V + old_row, old_row * V + picked_j)
            picked_count_j = tl.load(
                Counts + (replica * Pairs + pair_j) * V * V + picked_code_j, mask_j, 0
            )
            tl.store(
                Counts + (replica * Pairs + pair_j) * V * V + old_code_j,
                old_count_j - 1,
                mask_j & (old_j != picked_j),
            )
            tl.store(
                Counts + (replica * Pairs + pair_j) * V * V + picked_code_j,
                picked_count_j + 1,
                mask_j & (old_j != picked_j),
            )
            picked_code_ij = tl.where(i < j, picked_i * V + picked_j, picked_j * V + picked_i)
            picked_count_ij = tl.load(Counts + (replica * Pairs + pair_ij) * V * V + picked_code_ij)
            tl.store(
                Counts + (replica * Pairs + pair_ij) * V * V + old_code_ij,
                old_count_ij - 1,
                picked_code_ij != old_code_ij,
            )
            tl.store(
                Counts + (replica * Pairs + pair_ij) * V * V + picked_code_ij,
                picked_count_ij + 1,
                picked_code_ij != old_code_ij,
            )
            tl.store(Values + (replica * N + row) * K + j, picked_j)
            tl.store(X + ((replica * N + row) * K + j) * V + v, (v == picked_j).to(tl.int8), v < V)
            tl.store(Visited + ((replica * N + row) * K + j) * V + v, 1, v < V)
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            current = tl.load(Values + replica * N * K + cells, cells < N * K, 0)
            tl.store(Best + replica * N * K + cells, current, cells < N * K)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)


class CudaCoveringArray:
    def __init__(self, instance, initial, width=1, *, orientation="row", cache_parameters=True):
        if width not in (1, 2):
            raise ValueError("Covering-array block width must be one or two")
        if orientation not in {"row", "column"} or (
            orientation == "column" and (width != 2 or instance.rows < 2)
        ):
            raise ValueError("Column moves require two rows and block width two")
        initial = np.asarray(initial, dtype=np.int64)
        for a in initial:
            instance.validate(a)
        self.instance, self.r, self.width = instance, len(initial), width
        self.orientation = orientation
        self.cache_parameters = cache_parameters
        self.parameter_key = None
        self.parameters = None
        self.values = torch.tensor(initial, device="cuda")
        self.bits = torch.nn.functional.one_hot(self.values, instance.levels).to(torch.int8)
        self.best = self.values.clone()
        counts = np.array([instance.counts(a) for a in initial])
        self.counts = torch.tensor(counts, device="cuda")
        self.cost = torch.tensor((counts == 0).sum(axis=(1, 2)), dtype=torch.int64, device="cuda")
        self.best_cost = self.cost.clone()
        self.visited = torch.zeros_like(self.bits)
        self.changed = torch.zeros(self.r, dtype=torch.int64, device="cuda")

    def advance(
        self,
        noise,
        *,
        beta_start=0.1,
        beta_end=8.0,
        cycle_steps=4096,
        offset=0,
        method="heatbath",
        steps=None,
    ):
        if method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown transition method")
        instance = self.instance
        noise = torch.as_tensor(noise, dtype=torch.float64, device="cuda").contiguous()
        if noise.shape[:2] != (5, self.r):
            raise ValueError("Random tensor shape differs from the covering-array system")
        key = (beta_start, beta_end)
        if not self.cache_parameters or key != self.parameter_key:
            self.parameters = torch.tensor(
                [beta_start, math.log(beta_end / beta_start)], dtype=torch.float64, device="cuda"
            )
            self.parameter_key = key
        params = self.parameters
        chunk = noise.shape[-1]
        if self.orientation == "column":
            from .column_gpu import advance_column

            advance_column[(self.r,)](
                self.bits,
                self.values,
                self.best,
                self.counts,
                self.cost,
                self.best_cost,
                self.visited,
                self.changed,
                noise,
                params,
                steps or chunk,
                offset,
                cycle_steps,
                instance.rows,
                instance.columns,
                instance.levels,
                self.r,
                instance.pairs,
                triton.next_power_of_2(instance.columns),
                triton.next_power_of_2(instance.levels),
                triton.next_power_of_2(instance.levels**2),
                triton.next_power_of_2(instance.size),
                chunk,
                {"heatbath": 0, "metropolis": 1, "plateau": 2}[method],
                num_warps=8,
            )
            return
        _advance[(self.r,)](
            self.bits,
            self.values,
            self.best,
            self.counts,
            self.cost,
            self.best_cost,
            self.visited,
            self.changed,
            noise,
            params,
            steps or chunk,
            offset,
            cycle_steps,
            instance.rows,
            instance.columns,
            instance.levels,
            self.r,
            instance.pairs,
            triton.next_power_of_2(instance.columns),
            triton.next_power_of_2(instance.levels),
            triton.next_power_of_2(instance.levels**self.width),
            triton.next_power_of_2(instance.size),
            chunk,
            self.width,
            {"heatbath": 0, "metropolis": 1, "plateau": 2}[method],
            num_warps=4,
        )

    def state(self):
        return {
            key: getattr(self, value).cpu().numpy()
            for key, value in {
                "bits": "bits",
                "values": "values",
                "best_values": "best",
                "counts": "counts",
                "cost": "cost",
                "best_cost": "best_cost",
                "visited": "visited",
                "changed": "changed",
            }.items()
        }
