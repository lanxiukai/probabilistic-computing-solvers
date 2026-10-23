"""Persistent raw colored Gibbs for the fixed P7 sparse multiplier.

One GPU block owns one independent replica. Block barriers are therefore enough
between colors; no neighboring physical spins update together. This fuses time
steps, never logical variables. The legacy numerical mode preserves counter-v1
trajectories, while uint32_tail avoids cancellation in rare flip probabilities.
"""

# pyright: reportInvalidTypeForm=false

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import triton
import triton.language as tl

from sparse_factorization.core.coloring import dsatur_coloring, validate_coloring
from sparse_factorization.core.rng import derive_stream_seed

from .compiler import ConditionedFactorizer

ProbabilityMode = Literal["legacy_fp32", "uint32_tail"]


def tail_thresholds(fields: torch.Tensor, beta: float) -> torch.Tensor:
    """Return 32-bit integer tail thresholds without subtracting from one."""

    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    probabilities = torch.sigmoid(-2.0 * float(beta) * fields.float().abs())
    return (probabilities * (1 << 32)).to(torch.int64)


@triton.jit(do_not_specialize=["beta", "seed", "counter_offset", "global_sweep", "sweeps"])
def _persistent_sparse_kernel(
    states_pointer,
    best_pointer,
    hit_pointer,
    energy_pointer,
    best_energy_pointer,
    first_pointer,
    neighbors_pointer,
    weights_pointer,
    bias_pointer,
    color_indices_pointer,
    color_counts_pointer,
    color_prefixes_pointer,
    factor_a_weights_pointer,
    factor_b_weights_pointer,
    beta_pointer,
    beta,
    seed,
    counter_offset,
    global_sweep,
    sweeps,
    target: tl.constexpr,
    variables: tl.constexpr,
    replicas: tl.constexpr,
    maximum_degree: tl.constexpr,
    color_count: tl.constexpr,
    color_block: tl.constexpr,
    state_block: tl.constexpr,
    uint32_tail: tl.constexpr,
    per_replica_beta: tl.constexpr,
):
    replica = tl.program_id(0)
    inverse_temperature = tl.load(beta_pointer + replica) if per_replica_beta else beta
    lane = tl.arange(0, color_block)
    whole = tl.arange(0, state_block)
    energy = tl.load(energy_pointer + replica)
    best_energy = tl.load(best_energy_pointer + replica)
    first = tl.load(first_pointer + replica)
    for sweep in tl.range(0, sweeps, disable_licm=True):
        for color in tl.range(0, color_count, loop_unroll_factor=1):
            count = tl.load(color_counts_pointer + color)
            prefix = tl.load(color_prefixes_pointer + color)
            mask = lane < count
            variable = tl.load(color_indices_pointer + color * color_block + lane, mask, other=0)
            previous = tl.load(
                states_pointer + replica * variables + variable, mask, other=0, volatile=True
            ).to(tl.float32)
            field = tl.load(bias_pointer + variable, mask, other=0.0)
            for degree in tl.static_range(maximum_degree):
                neighbor = tl.load(neighbors_pointer + degree * variables + variable, mask, other=0)
                weight = tl.load(weights_pointer + degree * variables + variable, mask, other=0.0)
                other = tl.load(
                    states_pointer + replica * variables + neighbor, mask, other=0, volatile=True
                ).to(tl.float32)
                field += weight * other
            offset = (
                counter_offset
                + sweep.to(tl.int64) * (replicas * variables)
                + prefix.to(tl.int64) * replicas
                + replica * count.to(tl.int64)
                + lane
            )
            if uint32_tail:
                word = tl.randint(seed, offset)
                threshold = (
                    tl.sigmoid(-2.0 * inverse_temperature * tl.abs(field)) * 4294967296.0
                ).to(tl.uint64)
                preferred = tl.where(field >= 0.0, 1.0, -1.0)
                updated = tl.where(word.to(tl.uint64) < threshold, -preferred, preferred)
            else:
                updated = tl.where(
                    tl.rand(seed, offset) < tl.sigmoid(2.0 * inverse_temperature * field), 1.0, -1.0
                )
            energy -= tl.sum(tl.where(mask, (updated - previous) * field, 0.0), 0)
            tl.store(states_pointer + replica * variables + variable, updated.to(tl.int8), mask)
            # Every node of this replica belongs to this block, including all colors.
            tl.debug_barrier()
        if energy < best_energy:
            current = tl.load(
                states_pointer + replica * variables + whole,
                whole < variables,
                other=0,
                volatile=True,
            )
            tl.store(best_pointer + replica * variables + whole, current, whole < variables)
            best_energy = energy
        if (energy == 0.0) & (first < 0):
            current = tl.load(
                states_pointer + replica * variables + whole,
                whole < variables,
                other=0,
                volatile=True,
            )
            a = tl.sum(
                tl.where(
                    current > 0,
                    tl.load(factor_a_weights_pointer + whole, whole < variables, other=0),
                    0,
                ),
                0,
            ).to(tl.uint64)
            b = tl.sum(
                tl.where(
                    current > 0,
                    tl.load(factor_b_weights_pointer + whole, whole < variables, other=0),
                    0,
                ),
                0,
            ).to(tl.uint64)
            if (a > 1) & (b > 1) & (a * b == tl.full((), target, tl.uint64)):
                tl.store(hit_pointer + replica * variables + whole, current, whole < variables)
                first = global_sweep + sweep.to(tl.int64) + 1
    tl.store(energy_pointer + replica, energy)
    tl.store(best_energy_pointer + replica, best_energy)
    tl.store(first_pointer + replica, first)


class PersistentSparseGibbs:
    """GPU-resident independent chains with exact logical counter accounting."""

    array_names = (
        "states",
        "best_states",
        "first_hit_states",
        "energies",
        "best_energies",
        "first_hit_sweeps",
    )

    def __init__(
        self,
        problem: ConditionedFactorizer,
        initial_states: torch.Tensor,
        *,
        seed: int,
        probability_mode: ProbabilityMode = "uint32_tail",
        allow_research_widths: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Persistent sparse Gibbs requires CUDA")
        compiled = problem.compiled
        if (
            compiled.representation != "compact_copy_tree"
            or compiled.netlist.profile != "aadit"
            or compiled.model.metadata["copy_weight_multiplier"] != 1.0
            or any(name.startswith("factor_") for name in problem.fixed_logical_bits)
        ):
            raise ValueError("Persistent Gibbs requires product-only unit-COPY Aadit sparse models")
        historical_p7 = (
            compiled.netlist.product_bits == 32
            and problem.target == 4_277_546_633
            and compiled.model.metadata["factor_copy_topology"] == "balanced_tree"
        )
        research_width = allow_research_widths and compiled.netlist.product_bits in {
            24,
            32,
            40,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            104,
            112,
            120,
            128,
        }
        if compiled.netlist.product_bits > 16 and not (historical_p7 or research_width):
            raise ValueError(
                "Wider execution requires the historical P7 graph or explicit sparse-study scope"
            )
        if probability_mode not in {"legacy_fp32", "uint32_tail"}:
            raise ValueError("Unknown Gibbs probability mode")
        if initial_states.ndim != 2 or initial_states.shape[1] != problem.model.size:
            raise ValueError("Initial states have the wrong physical shape")
        if not bool(torch.all((initial_states == -1) | (initial_states == 1))):
            raise ValueError("Initial states must contain bipolar physical spins")
        self.problem = problem
        self.replicas, self.variables = initial_states.shape
        self.probability_mode = probability_mode
        self.seed = int(seed)
        self.stream_name = (
            "triton_counter_gibbs_v1"
            if probability_mode == "legacy_fp32"
            else "triton_philox_uint32_tail_v1"
        )
        self.counter_seed = derive_stream_seed(seed, self.stream_name)
        self.counter_offset = 0
        self.global_sweep = 0
        self.colors = dsatur_coloring(problem.model)
        validate_coloring(problem.model, self.colors)
        self.states = initial_states.to(device="cuda", dtype=torch.int8).contiguous()
        self.best_states = self.states.clone()
        self.first_hit_states = torch.zeros_like(self.states)
        self.energies = problem.model.energy(initial_states.cpu().float()).float().cuda()
        self.best_energies = self.energies.clone()
        self.first_hit_sweeps = torch.full((self.replicas,), -1, dtype=torch.int64, device="cuda")
        initial_hits = (self.energies == 0) & problem.certificate_mask(initial_states.cpu()).cuda()
        self.first_hit_sweeps[initial_hits] = 0
        self.first_hit_states[initial_hits] = self.states[initial_hits]

        from sparse_factorization.core.edge_model import EdgeIsingModel
        from sparse_factorization.core.padded_sparse import PaddedSparseIsing

        if isinstance(problem.model, EdgeIsingModel):
            sparse_model = PaddedSparseIsing.from_model(
                problem.model, device="cpu", dtype=torch.float32
            )
            if not torch.equal(
                sparse_model.edge_couplings, sparse_model.edge_couplings.round()
            ) or not torch.equal(sparse_model.bias, sparse_model.bias.round()):
                raise ValueError("Incremental tracking requires integer Aadit coefficients")
            self.maximum_degree = sparse_model.maximum_degree
            neighbors, weights = sparse_model.neighbors.int(), sparse_model.couplings
        else:
            coupling = problem.model.coupling.cpu()
            if not torch.equal(coupling, coupling.round()) or not torch.equal(
                problem.model.bias, problem.model.bias.round()
            ):
                raise ValueError(
                    "Incremental energy tracking requires the integer Aadit coefficients"
                )
            self.maximum_degree = int(torch.count_nonzero(coupling, dim=1).max())
            neighbors = torch.zeros((self.variables, self.maximum_degree), dtype=torch.int32)
            weights = torch.zeros((self.variables, self.maximum_degree), dtype=torch.float32)
            for index in range(self.variables):
                adjacent = torch.where(coupling[index] != 0)[0]
                neighbors[index, : len(adjacent)] = adjacent.int()
                weights[index, : len(adjacent)] = coupling[index, adjacent]
        self.color_block = triton.next_power_of_2(max(map(len, self.colors)))
        indices = torch.zeros((len(self.colors), self.color_block), dtype=torch.int32)
        counts, prefixes, offset = [], [], 0
        for color, nodes in enumerate(self.colors):
            indices[color, : len(nodes)] = torch.tensor(nodes, dtype=torch.int32)
            counts.append(len(nodes))
            prefixes.append(offset)
            offset += len(nodes)
        to_free = {node: i for i, node in enumerate(problem.clamp.free_indices)}
        factor_weights = []
        for roots in (compiled.factor_a_indices, compiled.factor_b_indices):
            dtype = torch.int64 if len(roots) > 31 else torch.int32
            vector = torch.zeros(self.variables, dtype=dtype)
            for bit, node in enumerate(roots):
                vector[to_free[node]] = (1 << bit) if bit < 63 else -(1 << 63)
            factor_weights.append(vector.cuda())
        self._arguments = (
            neighbors.T.contiguous().cuda(),
            weights.T.contiguous().cuda(),
            problem.model.bias.float().cuda(),
            indices.cuda(),
            torch.tensor(counts, dtype=torch.int32, device="cuda"),
            torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
            *factor_weights,
        )

    def run_chunk(self, beta: float | torch.Tensor, sweeps: int) -> None:
        """Advance a short GPU launch, checking every native sweep for first passage."""

        if self.problem.compiled.netlist.product_bits > 64:
            raise ValueError("Above 64 bits this engine is storage for circuit-only studies")
        vector_beta = isinstance(beta, torch.Tensor)
        if vector_beta:
            if beta.shape != (self.replicas,) or not bool(
                torch.all(torch.isfinite(beta) & (beta >= 0))
            ):
                raise ValueError("Replica temperatures must be finite, nonnegative and slot-sized")
            beta_pointer = beta.to(device="cuda", dtype=torch.float32).contiguous()
            scalar_beta = 0.0
        else:
            scalar_beta = float(beta)
            beta_pointer = self.energies
        if not math.isfinite(scalar_beta) or scalar_beta < 0 or not 0 < sweeps <= 16384:
            raise ValueError("Use finite nonnegative beta and 1..16384 sweeps per GPU launch")
        end_offset = self.counter_offset + self.replicas * self.variables * sweeps
        if not 0 <= self.counter_offset < end_offset < (1 << 63):
            raise ValueError("The logical counter range must fit signed 64-bit integers")
        _persistent_sparse_kernel[(self.replicas,)](
            self.states,
            self.best_states,
            self.first_hit_states,
            self.energies,
            self.best_energies,
            self.first_hit_sweeps,
            *self._arguments,
            beta_pointer,
            scalar_beta,
            self.counter_seed,
            self.counter_offset,
            self.global_sweep,
            int(sweeps),
            self.problem.target,
            variables=self.variables,
            replicas=self.replicas,
            maximum_degree=self.maximum_degree,
            color_count=len(self.colors),
            color_block=self.color_block,
            state_block=triton.next_power_of_2(self.variables),
            uint32_tail=self.probability_mode == "uint32_tail",
            per_replica_beta=vector_beta,
            num_warps=8,
            enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        self.global_sweep += int(sweeps)
        self.counter_offset = end_offset

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy a completed launch boundary for atomic checkpointing."""

        return {
            **{
                name: getattr(self, name).detach().cpu().numpy().copy() for name in self.array_names
            },
            "global_sweep": np.asarray(self.global_sweep, dtype=np.int64),
            "counter_offset": np.asarray(self.counter_offset, dtype=np.int64),
        }

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        """Restore a completed native trajectory boundary, including first-hit history."""

        for name in self.array_names:
            current = getattr(self, name)
            value = torch.as_tensor(snapshot[name], device="cuda", dtype=current.dtype)
            if value.shape != current.shape:
                raise ValueError(f"Checkpoint shape differs for {name}")
            current.copy_(value)
        self.global_sweep = int(snapshot["global_sweep"])
        self.counter_offset = int(snapshot["counter_offset"])
