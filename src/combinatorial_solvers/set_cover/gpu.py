"""Persistent exact-integer covering updates with FP64 categorical decisions.

Each CUDA block owns one replica. A subset zeta transform evaluates all local
patterns from the histogram of uncovered row masks, without a pattern-by-row
matrix. Optional state-dependent block selection uses its conditional selection
mass in both Gibbs and Metropolis updates.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .blocks import Blocks


@triton.jit
def _advance(
    Bits,
    Coverage,
    Cost,
    Violations,
    Cardinality,
    ActiveList,
    Position,
    Visited,
    Changed,
    Flips,
    Nodes,
    Rows,
    Masks,
    Inverse,
    Weights,
    Noise,
    Steps,
    Offset,
    CycleSteps,
    Parameters,
    Penalty,
    N: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    U: tl.constexpr,
    NB: tl.constexpr,
    Banks: tl.constexpr,
    PerBank: tl.constexpr,
    Chunk: tl.constexpr,
    Hard: tl.constexpr,
    Method: tl.constexpr,
    Active: tl.constexpr,
    FirstSweep: tl.constexpr,
):
    replica = tl.program_id(0)
    beta_low = tl.load(Parameters)
    log_ratio = tl.load(Parameters + 1)
    scale = tl.load(Parameters + 2)
    epsilon = tl.load(Parameters + 3)
    penalty_integer = Penalty.to(tl.int64)
    slot = tl.arange(0, W)
    row_lane = tl.arange(0, U)
    patterns = tl.arange(0, 1 << W)
    cost = tl.load(Cost + replica)
    violations = tl.load(Violations + replica)
    cardinality = tl.load(Cardinality + replica)
    changed = tl.load(Changed + replica)
    flips = tl.load(Flips + replica)
    for iteration in range(Steps):
        step = Offset + iteration
        u_block = tl.load(Noise + replica * Chunk + iteration)
        u_pattern = tl.load(Noise + (R + replica) * Chunk + iteration)
        u_accept = tl.load(Noise + (2 * R + replica) * Chunk + iteration)
        scheduled = FirstSweep & (step < PerBank)
        use_active = Active & ~scheduled
        total_mass = epsilon * NB + Banks * cardinality
        block = tl.minimum((u_block * NB).to(tl.int32), NB - 1)
        if Active:
            if use_active:
                draw = u_block * total_mass
                if draw < epsilon * NB:
                    block = tl.minimum((draw / epsilon).to(tl.int32), NB - 1)
                else:
                    anchor_draw = tl.minimum(
                        (draw - epsilon * NB).to(tl.int32), Banks * cardinality - 1
                    )
                    anchor = tl.load(ActiveList + replica * N + anchor_draw // Banks)
                    block = tl.load(Inverse + (anchor_draw % Banks) * N + anchor)
        block = tl.where(scheduled, step, block)
        nodes = tl.load(Nodes + block * W + slot)
        real = nodes >= 0
        old_bits = tl.load(Bits + replica * N + nodes, real, 0).to(tl.int32)
        weights = tl.load(Weights + nodes, real, 0)
        old_pattern = tl.sum(old_bits * (1 << slot), 0)
        present = tl.sum(real.to(tl.int32) * (1 << slot), 0)
        old_local_count = tl.sum(old_bits, 0)
        old_local_cost = tl.sum(old_bits * weights, 0)
        row_ids = tl.load(Rows + block * U + row_lane)
        row_masks = tl.load(Masks + block * U + row_lane)
        row_valid = row_ids < M
        counts = tl.load(Coverage + replica * M + row_ids, row_valid, 0)
        old_contribution = tl.full((U,), 0, tl.int32)
        local_cost = tl.full((1 << W,), 0, tl.int64)
        local_count = tl.full((1 << W,), 0, tl.int32)
        for bit in tl.static_range(W):
            old_contribution += ((old_pattern >> bit) & 1) * ((row_masks >> bit) & 1)
            weight = tl.sum(tl.where(slot == bit, weights, 0), 0)
            local_cost += ((patterns >> bit) & 1) * weight
            local_count += (patterns >> bit) & 1
        outside = counts - old_contribution
        phase = (step % CycleSteps).to(tl.float64) / tl.maximum(CycleSteps - 1, 1)
        gamma = beta_low * libdevice.exp(log_ratio * phase) / scale
        if Method == 1:
            # A uniform block proposal needs one energy evaluation, not enumeration.
            proposal = tl.minimum((u_pattern * (1 << W)).to(tl.int32), (1 << W) - 1)
            proposal_bits = (proposal >> slot) & 1
            proposal_count = tl.sum(tl.where(real, proposal_bits, 0), 0)
            proposal_cost = tl.sum(proposal_bits * weights, 0)
            proposal_contribution = tl.full((U,), 0, tl.int32)
            for bit in tl.static_range(W):
                proposal_contribution += ((proposal >> bit) & 1) * ((row_masks >> bit) & 1)
            proposal_uncovered = tl.sum(
                (row_valid & (outside + proposal_contribution == 0)).to(tl.int32), 0
            )
            old_uncovered = tl.sum((row_valid & (counts == 0)).to(tl.int32), 0)
            possible = (proposal & ~present) == 0
            old_energy = old_local_cost.to(tl.float64)
            selected_energy = proposal_cost.to(tl.float64)
            if Hard:
                possible &= proposal_uncovered == 0
            else:
                old_energy += penalty_integer * old_uncovered
                selected_energy += penalty_integer * proposal_uncovered
            selected_mass = tl.full((), 1, tl.float64)
            old_mass = tl.full((), 1, tl.float64)
            if Active:
                if use_active:
                    selected_mass = (epsilon + proposal_count) / (
                        epsilon * NB + Banks * (cardinality - old_local_count + proposal_count)
                    )
                    old_mass = (epsilon + old_local_count) / total_mass
            accept = possible & (
                u_accept
                < tl.minimum(
                    1.0,
                    libdevice.exp(-gamma * (selected_energy - old_energy))
                    * selected_mass
                    / old_mass,
                )
            )
            new_pattern = tl.where(accept, proposal, old_pattern)
        else:
            histogram = tl.histogram(row_masks, 1 << W, mask=row_valid & (outside == 0))
            zeta = tl.where(patterns == 0, 0, histogram)
            for bit in tl.static_range(W):
                partner = tl.gather(zeta, patterns ^ (1 << bit), 0)
                zeta += tl.where((patterns & (1 << bit)) != 0, partner, 0)
            uncovered = tl.gather(zeta, ((1 << W) - 1) ^ patterns, 0)
            valid = (patterns & ~present) == 0
            if Hard:
                valid &= uncovered == 0
                energy = local_cost.to(tl.float64)
            else:
                energy = (local_cost + penalty_integer * uncovered).to(tl.float64)
            if Method == 0:
                mass = tl.full((1 << W,), 1, tl.float64)
                if Active:
                    if use_active:
                        mass = (epsilon + local_count) / (
                            epsilon * NB + Banks * (cardinality - old_local_count + local_count)
                        )
                minimum = tl.min(tl.where(valid, energy, float("inf")), 0)
                probability = tl.where(valid, libdevice.exp(-gamma * (energy - minimum)) * mass, 0)
                threshold = u_pattern * tl.sum(probability, 0)
                cumulative = tl.cumsum(probability, 0)
                new_pattern = tl.min(
                    tl.where(valid & (cumulative > threshold), patterns, 1 << W), 0
                )
                last = tl.max(tl.where(valid, patterns, -1), 0)
                new_pattern = tl.where(new_pattern == (1 << W), last, new_pattern)
            elif Method == 2:
                minimum = tl.min(tl.where(valid, energy, float("inf")), 0)
                proposal = tl.min(tl.where(valid & (energy == minimum), patterns, 1 << W), 0)
                old_energy = tl.sum(tl.where(patterns == old_pattern, energy, 0), 0)
                new_pattern = tl.where(minimum < old_energy, proposal, old_pattern)
            else:
                minimum = tl.min(tl.where(valid, energy, float("inf")), 0)
                tied = valid & (energy == minimum)
                rank = (u_pattern * tl.sum(tied.to(tl.int32), 0)).to(tl.int32)
                cumulative = tl.cumsum(tied.to(tl.int32), 0)
                new_pattern = tl.min(tl.where(tied & (cumulative > rank), patterns, 1 << W), 0)
        new_bits = (new_pattern >> slot) & 1
        new_local_count = tl.sum(tl.where(real, new_bits, 0), 0)
        new_local_cost = tl.sum(new_bits * weights, 0)
        new_contribution = tl.full((U,), 0, tl.int32)
        for bit in tl.static_range(W):
            new_contribution += ((new_pattern >> bit) & 1) * ((row_masks >> bit) & 1)
        new_counts = outside + new_contribution
        violations += tl.sum(((new_counts == 0) & row_valid).to(tl.int32), 0)
        violations -= tl.sum(((counts == 0) & row_valid).to(tl.int32), 0)
        tl.store(Coverage + replica * M + row_ids, new_counts, row_valid)
        tl.store(Bits + replica * N + nodes, new_bits, real)
        tl.store(Visited + replica * N + nodes, 1, real)
        flips += tl.sum((real & (old_bits != new_bits)).to(tl.int32), 0)
        changed += new_pattern != old_pattern
        cost += new_local_cost - old_local_cost
        if Active:
            for bit in tl.static_range(W):
                column = tl.sum(tl.where(slot == bit, nodes, 0), 0)
                old = (old_pattern >> bit) & 1
                new = (new_pattern >> bit) & 1
                if (column >= 0) & (old != new):
                    if new == 0:
                        position = tl.load(Position + replica * N + column)
                        last_column = tl.load(ActiveList + replica * N + cardinality - 1)
                        tl.store(ActiveList + replica * N + position, last_column)
                        tl.store(Position + replica * N + last_column, position)
                        tl.store(ActiveList + replica * N + cardinality - 1, -1)
                        tl.store(Position + replica * N + column, -1)
                        cardinality -= 1
                    else:
                        tl.store(ActiveList + replica * N + cardinality, column)
                        tl.store(Position + replica * N + column, cardinality)
                        cardinality += 1
                    tl.debug_barrier()
        else:
            cardinality += new_local_count - old_local_count
        tl.debug_barrier()
    tl.store(Cost + replica, cost)
    tl.store(Violations + replica, violations)
    tl.store(Cardinality + replica, cardinality)
    tl.store(Changed + replica, changed)
    tl.store(Flips + replica, flips)


class CudaCover:
    def __init__(
        self, instance, blocks: Blocks, states, *, active=False, epsilon=0.0625, cache_initial=True
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for native covering updates")
        self.instance, self.blocks = instance, blocks
        states = np.asarray([instance.validate_bits(s) for s in np.atleast_2d(states)])
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Block-selection epsilon must be positive")
        self.active, self.epsilon = active, float(epsilon)
        self.replicas = len(states)

        def device(a):
            return torch.as_tensor(np.asarray(a).copy(), device="cuda")

        self.bits = device(states.astype(np.int8))
        certificates, cache = [], {}
        for state in states:
            key = state.tobytes()
            if not cache_initial or key not in cache:
                cache[key] = instance.certificate(state)
            certificates.append(cache[key])
        self.coverage = device(np.asarray([c["coverage"] for c in certificates], dtype=np.int32))
        self.cost = device(np.asarray([c["cost"] for c in certificates], dtype=np.int64))
        self.violations = device(
            np.asarray([len(c["uncovered_rows"]) for c in certificates], dtype=np.int32)
        )
        self.cardinality = device(states.sum(-1, dtype=np.int32))
        active_list = np.full(states.shape, -1, dtype=np.int32)
        position = active_list.copy()
        for r, state in enumerate(states):
            selected = np.flatnonzero(state)
            active_list[r, : len(selected)] = selected
            position[r, selected] = np.arange(len(selected))
        self.active_list, self.position = device(active_list), device(position)
        self.visited = torch.zeros_like(self.bits)
        self.changed = torch.zeros(self.replicas, dtype=torch.int64, device="cuda")
        self.flips = self.changed.clone()
        self.nodes, self.rows, self.masks = map(device, (blocks.nodes, blocks.rows, blocks.masks))
        self.inverse = device(blocks.inverse)
        self.weights = device(np.asarray(instance.costs, dtype=np.int64))

    def advance(
        self,
        noise,
        *,
        offset=0,
        steps=None,
        cycle_steps=1000,
        beta_low=0.1,
        beta_high=16.0,
        penalty=None,
        method="heatbath",
        hard=False,
        first_sweep=True,
    ):
        if method not in {"heatbath", "metropolis", "greedy", "plateau"}:
            raise ValueError("Unknown covering update method")
        noise = torch.as_tensor(noise, dtype=torch.float64, device="cuda").contiguous()
        if noise.ndim != 3 or noise.shape[:2] != (3, self.replicas):
            raise ValueError("Noise requires three streams per replica and chunk")
        if cycle_steps < 1 or beta_low <= 0 or beta_high < beta_low:
            raise ValueError("Invalid temperature schedule")
        steps = noise.shape[2] if steps is None else steps
        if not 0 < steps <= noise.shape[2] or offset < 0:
            raise ValueError("Invalid step range")
        penalty = self.instance.checked_penalty(
            max(self.instance.costs) + 1 if penalty is None else penalty
        )
        parameter_key = (
            float(beta_low),
            float(np.log(beta_high / beta_low)),
            float(max(self.instance.costs)),
            self.epsilon,
        )
        if getattr(self, "_parameter_key", None) != parameter_key:
            self.parameters = torch.tensor(parameter_key, dtype=torch.float64, device="cuda")
            self._parameter_key = parameter_key
        if hard and bool(self.violations.any()):
            raise ValueError("Feasible updates require a feasible initial state")
        _advance[(self.replicas,)](
            self.bits,
            self.coverage,
            self.cost,
            self.violations,
            self.cardinality,
            self.active_list,
            self.position,
            self.visited,
            self.changed,
            self.flips,
            self.nodes,
            self.rows,
            self.masks,
            self.inverse,
            self.weights,
            noise,
            steps,
            offset,
            cycle_steps,
            self.parameters,
            penalty,
            N=self.instance.size,
            M=len(self.instance.rows),
            R=self.replicas,
            W=self.blocks.width,
            U=self.blocks.rows.shape[1],
            NB=len(self.blocks.nodes),
            Banks=self.blocks.banks,
            PerBank=self.blocks.per_bank,
            Chunk=noise.shape[2],
            Hard=hard,
            Method={"heatbath": 0, "metropolis": 1, "greedy": 2, "plateau": 3}[method],
            Active=self.active,
            FirstSweep=first_sweep,
            num_warps=4,
            enable_fp_fusion=False,
        )

    def state(self):
        return {
            name: getattr(self, name).cpu().numpy()
            for name in (
                "bits",
                "coverage",
                "cost",
                "violations",
                "cardinality",
                "active_list",
                "position",
                "visited",
                "changed",
                "flips",
            )
        }
