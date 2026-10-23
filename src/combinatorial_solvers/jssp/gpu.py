"""Exact serial schedule decoding in permutation-block conditional updates."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .active_gpu import active_energy


@triton.jit(
    do_not_specialize=["Steps", "Offset", "Cycle", "Window"],
    do_not_specialize_on_alignment=["Steps", "Offset", "Cycle", "Window"],
)
def _advance(
    X,
    Sequence,
    Best,
    Cost,
    BestCost,
    Visited,
    Changed,
    Machines,
    Durations,
    Patterns,
    Noise,
    Parameters,
    Steps,
    Offset,
    Cycle,
    Window,
    J: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    JJ: tl.constexpr,
    MM: tl.constexpr,
    LL: tl.constexpr,
    Chunk: tl.constexpr,
    Method: tl.constexpr,
    Gather: tl.constexpr,
    Active: tl.constexpr,
):
    replica = tl.program_id(0)
    patterns = tl.arange(0, PP)
    slots = tl.arange(0, B)
    jobs = tl.arange(0, JJ)
    machines = tl.arange(0, MM)
    positions = tl.arange(0, LL)
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    for step in range(Steps):
        rows = tl.full((B,), 0, tl.int32)
        for a in tl.static_range(B):
            u = tl.load(Noise + (a * R + replica) * Chunk + step)
            if a == 0:
                row = tl.minimum((u * L).to(tl.int32), L - 1)
                anchor = row
            else:
                row = (anchor + 1 + tl.minimum((u * Window).to(tl.int32), Window - 1)) % L
            collision = tl.sum(((slots < a) & (rows == row)).to(tl.int32), 0) > 0
            while collision:
                row = (row + 1) % L
                collision = tl.sum(((slots < a) & (rows == row)).to(tl.int32), 0) > 0
            rows = tl.where(slots == a, row, rows)
        old_jobs = tl.load(Sequence + replica * L + rows)
        if Active:
            energy = active_energy(
                Sequence,
                Machines,
                Durations,
                Patterns,
                replica,
                rows,
                old_jobs,
                J,
                M,
                L,
                B,
                P,
                PP,
                JJ,
                MM,
            )
        else:
            operation = tl.full((PP, JJ), 0, tl.int32)
            job_ready = tl.full((PP, JJ), 0, tl.int32)
            machine_ready = tl.full((PP, MM), 0, tl.int32)
            for position in range(L):
                job = tl.full((PP,), 0, tl.int32) + tl.load(Sequence + replica * L + position)
                for a in tl.static_range(B):
                    row = tl.sum(tl.where(slots == a, rows, 0), 0)
                    index = tl.load(Patterns + patterns * B + a, patterns < P, 0)
                    proposed_job = tl.gather(old_jobs, index, 0)
                    job = tl.where(position == row, proposed_job, job)
                if Gather:
                    op = tl.gather(operation, job[:, None].to(tl.int32), 1).reshape(PP)
                else:
                    op = tl.sum(tl.where(jobs[None, :] == job[:, None], operation, 0), 1)
                machine = tl.load(Machines + job * M + op, patterns < P, 0)
                duration = tl.load(Durations + job * M + op, patterns < P, 0).to(tl.int32)
                if Gather:
                    ready_job = tl.gather(job_ready, job[:, None].to(tl.int32), 1).reshape(PP)
                    ready_machine = tl.gather(
                        machine_ready, machine[:, None].to(tl.int32), 1
                    ).reshape(PP)
                else:
                    ready_job = tl.sum(tl.where(jobs[None, :] == job[:, None], job_ready, 0), 1)
                    ready_machine = tl.sum(
                        tl.where(machines[None, :] == machine[:, None], machine_ready, 0), 1
                    )
                end = tl.maximum(ready_job, ready_machine) + duration
                operation += (jobs[None, :] == job[:, None]).to(tl.int32)
                job_ready = tl.where(jobs[None, :] == job[:, None], end[:, None], job_ready)
                machine_ready = tl.where(
                    machines[None, :] == machine[:, None], end[:, None], machine_ready
                )
            energy = tl.max(job_ready, 1)
        beta = low * libdevice.exp(
            ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
        )
        u = tl.load(Noise + (B * R + replica) * Chunk + step)
        if Method == 1:
            proposed = tl.minimum((u * P).to(tl.int32), P - 1)
            proposed_energy = tl.sum(tl.where(patterns == proposed, energy, 0), 0)
            accept_u = tl.load(Noise + ((B + 1) * R + replica) * Chunk + step)
            selected = tl.where(
                accept_u
                < libdevice.exp(-beta * tl.maximum(proposed_energy - cost, 0).to(tl.float64)),
                proposed,
                0,
            )
        else:
            minimum = tl.min(tl.where(patterns < P, energy, 2147483647), 0)
            if Method == 2:
                weights = ((patterns < P) & (energy == minimum)).to(tl.float64)
            else:
                weights = tl.where(
                    patterns < P, libdevice.exp(-beta * (energy - minimum).to(tl.float64)), 0.0
                )
            selected = tl.min(
                tl.where(
                    (patterns < P) & (tl.cumsum(weights, 0) > u * tl.sum(weights, 0)), patterns, PP
                ),
                0,
            )
        cost = tl.sum(tl.where(patterns == selected, energy, 0), 0).to(tl.int64)
        any_change = tl.full((), False, tl.int1)
        for a in tl.static_range(B):
            row = tl.sum(tl.where(slots == a, rows, 0), 0)
            index = tl.load(Patterns + selected * B + a)
            new_job = tl.sum(tl.where(slots == index, old_jobs, 0), 0)
            old_job = tl.sum(tl.where(slots == a, old_jobs, 0), 0)
            any_change |= old_job != new_job
            tl.store(Sequence + replica * L + row, new_job)
            tl.store(X + (replica * L + row) * J + jobs, (jobs == new_job).to(tl.int8), jobs < J)
            # Duplicate job labels do not create duplicate writes to the same address.
            available = tl.full((JJ,), False, tl.int1)
            for b in tl.static_range(B):
                label = tl.sum(tl.where(slots == b, old_jobs, 0), 0)
                available |= jobs == label
            tl.store(Visited + (replica * L + row) * J + jobs, 1, (jobs < J) & available)
        changed += any_change
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            current = tl.load(Sequence + replica * L + positions, positions < L, 0)
            tl.store(Best + replica * L + positions, current, positions < L)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)


class CudaJobShop:
    def __init__(self, instance, initial, width=2, schedule_decoder="append"):
        if width not in (2, 4) or width > instance.size:
            raise ValueError("Job-shop block width must be two or four")
        if schedule_decoder not in {"append", "active"}:
            raise ValueError("Unknown schedule decoder")
        if schedule_decoder == "active" and width != 2:
            raise ValueError("The active decoder currently supports two-position blocks only")
        initial = np.asarray(initial, dtype=np.int64)
        for row in initial:
            instance.validate_sequence(row)
        self.instance = instance
        self.schedule_decoder = schedule_decoder
        self.r, self.width = len(initial), width
        self.sequence = torch.tensor(initial, device="cuda")
        self.bits = torch.nn.functional.one_hot(self.sequence, instance.jobs).to(torch.int8)
        self.best = self.sequence.clone()
        self.cost = torch.tensor(
            [
                instance.certificate(instance.decode(row, schedule_decoder=schedule_decoder))[
                    "makespan"
                ]
                for row in initial
            ],
            dtype=torch.int64,
            device="cuda",
        )
        self.best_cost = self.cost.clone()
        self.visited = torch.zeros_like(self.bits)
        self.changed = torch.zeros(self.r, dtype=torch.int64, device="cuda")
        self.machines = torch.tensor(instance.machines.copy(), device="cuda")
        self.durations = torch.tensor(instance.durations.copy(), device="cuda")
        self.patterns = torch.tensor(
            list(itertools.permutations(range(width))), dtype=torch.int64, device="cuda"
        )

    def advance(
        self,
        noise,
        *,
        beta_start=0.01,
        beta_end=1.0,
        cycle_steps=1024,
        offset=0,
        window=None,
        method="heatbath",
        decoder="reduce",
        steps=None,
    ):
        if method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown update method")
        if decoder not in {"reduce", "gather"}:
            raise ValueError("Unknown exact decoder implementation")
        instance = self.instance
        noise = torch.as_tensor(noise, dtype=torch.float64, device="cuda").contiguous()
        if noise.shape[:2] != (self.width + 2, self.r):
            raise ValueError("Random tensor shape differs from the job-shop system")
        params = torch.tensor(
            [beta_start, math.log(beta_end / beta_start)], dtype=torch.float64, device="cuda"
        )
        chunk = noise.shape[-1]
        _advance[(self.r,)](
            self.bits,
            self.sequence,
            self.best,
            self.cost,
            self.best_cost,
            self.visited,
            self.changed,
            self.machines,
            self.durations,
            self.patterns,
            noise,
            params,
            steps or chunk,
            offset,
            cycle_steps,
            window or instance.size - 1,
            instance.jobs,
            instance.machine_count,
            instance.size,
            self.r,
            self.width,
            len(self.patterns),
            triton.next_power_of_2(len(self.patterns)),
            triton.next_power_of_2(instance.jobs),
            triton.next_power_of_2(instance.machine_count),
            triton.next_power_of_2(instance.size),
            chunk,
            {"heatbath": 0, "metropolis": 1, "plateau": 2}[method],
            decoder == "gather",
            self.schedule_decoder == "active",
            num_warps=4,
        )

    def state(self):
        return {
            key: getattr(self, value).cpu().numpy()
            for key, value in {
                "bits": "bits",
                "sequence": "sequence",
                "best_sequence": "best",
                "cost": "cost",
                "best_cost": "best_cost",
                "visited": "visited",
                "changed": "changed",
            }.items()
        }
