"""Exact local permutation energies and persistent one-hot block heat baths."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _advance(
    X,
    Perm,
    Best,
    Cost,
    BestCost,
    Visited,
    Changed,
    F,
    D,
    Patterns,
    Noise,
    Parameters,
    Steps,
    Offset,
    Cycle,
    N: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    NN: tl.constexpr,
    Chunk: tl.constexpr,
    Method: tl.constexpr,
    Factored: tl.constexpr,
):
    replica = tl.program_id(0)
    k = tl.arange(0, NN)
    candidate = tl.arange(0, PP)
    valid = candidate < P
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    for step in range(Steps):
        rows = tl.full((B,), 0, tl.int32)
        slots = tl.arange(0, B)
        for a in tl.static_range(B):
            u = tl.load(Noise + (a * R + replica) * Chunk + step)
            row = tl.minimum((u * N).to(tl.int32), N - 1)
            collision = tl.sum(((slots < a) & (rows == row)).to(tl.int32), 0) > 0
            while collision:
                row = (row + 1) % N
                collision = tl.sum(((slots < a) & (rows == row)).to(tl.int32), 0) > 0
            rows = tl.where(slots == a, row, rows)
        old_locations = tl.load(Perm + replica * N + rows)
        all_locations = tl.load(Perm + replica * N + k, k < N, 0)
        outside = k < N
        for a in tl.static_range(B):
            row = tl.sum(tl.where(slots == a, rows, 0), 0)
            outside &= k != row
        if Factored:
            # Every candidate reuses the same B locations. Load these outside
            # interactions once, rather than repeat them for all B! patterns.
            table_out = tl.load(
                D + old_locations[:, None] * N + all_locations[None, :],
                outside[None, :],
                0,
            )
            table_in = tl.load(
                D + all_locations[None, :] * N + old_locations[:, None],
                outside[None, :],
                0,
            )
        energy = tl.full((PP,), 0, tl.int64)
        for a in tl.static_range(B):
            row = tl.sum(tl.where(slots == a, rows, 0), 0)
            index = tl.load(Patterns + candidate * B + a, valid, 0)
            location = tl.gather(old_locations, index, 0)
            outgoing = tl.load(F + row * N + k, outside, 0)
            incoming = tl.load(F + k * N + row, outside, 0)
            if Factored:
                position_energy = tl.sum(
                    outgoing[None, :] * table_out + incoming[None, :] * table_in, 1
                )
                energy += tl.gather(position_energy, index, 0)
            else:
                d_out = tl.load(
                    D + location[:, None] * N + all_locations[None, :],
                    valid[:, None] & outside[None, :],
                    0,
                )
                d_in = tl.load(
                    D + all_locations[None, :] * N + location[:, None],
                    valid[:, None] & outside[None, :],
                    0,
                )
                energy += tl.sum(outgoing[None, :] * d_out + incoming[None, :] * d_in, 1)
            for b in tl.static_range(B):
                row_b = tl.sum(tl.where(slots == b, rows, 0), 0)
                index_b = tl.load(Patterns + candidate * B + b, valid, 0)
                location_b = tl.gather(old_locations, index_b, 0)
                energy += tl.load(F + row * N + row_b) * tl.load(D + location * N + location_b)
        old_energy = tl.sum(tl.where(candidate == 0, energy, 0), 0)
        beta = low * libdevice.exp(
            ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
        )
        u = tl.load(Noise + (B * R + replica) * Chunk + step)
        if Method == 1:
            proposed = tl.minimum((u * P).to(tl.int32), P - 1)
            delta = tl.sum(tl.where(candidate == proposed, energy, 0), 0) - old_energy
            accept_u = tl.load(Noise + ((B + 1) * R + replica) * Chunk + step)
            selected = tl.where(
                accept_u < libdevice.exp(-beta * tl.maximum(delta, 0).to(tl.float64)), proposed, 0
            )
        else:
            minimum = tl.min(tl.where(valid, energy, 0x7FFFFFFFFFFFFFFF), 0)
            if Method == 2:
                weights = (valid & (energy == minimum)).to(tl.float64)
            else:
                weights = tl.where(
                    valid, libdevice.exp(-beta * (energy - minimum).to(tl.float64)), 0.0
                )
            selected = tl.min(
                tl.where(valid & (tl.cumsum(weights, 0) > u * tl.sum(weights, 0)), candidate, PP), 0
            )
        selected_energy = tl.sum(tl.where(candidate == selected, energy, 0), 0)
        cost += selected_energy - old_energy
        changed += selected != 0
        for a in tl.static_range(B):
            row = tl.sum(tl.where(slots == a, rows, 0), 0)
            index = tl.load(Patterns + selected * B + a)
            location = tl.sum(tl.where(slots == index, old_locations, 0), 0)
            tl.store(
                X + replica * N * N + row * N + old_locations,
                (old_locations == location).to(tl.int8),
            )
            tl.store(Visited + replica * N * N + row * N + old_locations, 1)
            tl.store(Perm + replica * N + row, location)
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            p = tl.load(Perm + replica * N + k, k < N, 0)
            tl.store(Best + replica * N + k, p, k < N)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)


class CudaQAP:
    def __init__(self, instance, initial, width=2, num_warps=4, energy_layout="candidate"):
        if width not in (2, 4) or width > instance.size:
            raise ValueError("Block width must be two or four and no larger than the problem")
        if num_warps not in (4, 8, 16):
            raise ValueError("Supported CUDA warp counts are four, eight and sixteen")
        if energy_layout not in {"candidate", "factored"}:
            raise ValueError("Unknown local energy evaluation layout")
        initial = np.asarray(initial, dtype=np.int64)
        certificates = [instance.certificate(row) for row in initial]
        if not all(c["feasible"] for c in certificates):
            raise ValueError("Initial states must be permutations")
        self.n, self.r, self.width = instance.size, len(initial), width
        self.num_warps = num_warps
        self.energy_layout = energy_layout
        self.kernel_resources = {}
        self.perm = torch.tensor(initial, dtype=torch.int64, device="cuda")
        self.bits = torch.nn.functional.one_hot(self.perm, self.n).to(torch.int8)
        self.best = self.perm.clone()
        self.cost = torch.tensor(
            [c["cost"] for c in certificates], dtype=torch.int64, device="cuda"
        )
        self.best_cost = self.cost.clone()
        self.visited = torch.zeros_like(self.bits)
        self.changed = torch.zeros(self.r, dtype=torch.int64, device="cuda")
        self.flow = torch.tensor(instance.flow.copy(), device="cuda")
        self.distance = torch.tensor(instance.distance.copy(), device="cuda")
        self.patterns = torch.tensor(
            list(itertools.permutations(range(width))), dtype=torch.int64, device="cuda"
        )

    def advance(
        self,
        noise,
        *,
        beta_start=0.01,
        beta_end=1.0,
        cycle_steps=1000,
        offset=0,
        method="heatbath",
        steps=None,
    ):
        if method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown transition method")
        noise = torch.as_tensor(noise, dtype=torch.float64, device="cuda").contiguous()
        if noise.shape[:2] != (self.width + 2, self.r):
            raise ValueError("Random tensor shape differs from the declared block system")
        params = torch.tensor(
            [beta_start, math.log(beta_end / beta_start)], dtype=torch.float64, device="cuda"
        )
        chunk = noise.shape[-1]
        kernel = _advance[(self.r,)](
            self.bits,
            self.perm,
            self.best,
            self.cost,
            self.best_cost,
            self.visited,
            self.changed,
            self.flow,
            self.distance,
            self.patterns,
            noise,
            params,
            steps or chunk,
            offset,
            cycle_steps,
            self.n,
            self.r,
            self.width,
            len(self.patterns),
            triton.next_power_of_2(len(self.patterns)),
            triton.next_power_of_2(self.n),
            chunk,
            {"heatbath": 0, "metropolis": 1, "plateau": 2}[method],
            self.energy_layout == "factored",
            num_warps=self.num_warps,
        )
        if kernel.hash not in self.kernel_resources:
            self.kernel_resources[kernel.hash] = {
                "kernel_hash": kernel.hash,
                "registers_per_thread": int(kernel.n_regs),
                "local_bytes_per_thread": int(kernel.n_spills) * 4,
                "static_shared_bytes": int(kernel.metadata.shared),
                "num_warps": self.num_warps,
            }

    def state(self):
        return {
            key: getattr(self, value).cpu().numpy()
            for key, value in {
                "bits": "bits",
                "permutation": "perm",
                "best_permutation": "best",
                "cost": "cost",
                "best_cost": "best_cost",
                "visited": "visited",
                "changed": "changed",
            }.items()
        }
