"""Exact input-validated reuse of an accepted state's energy and body log density."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .circuit_triton import _energy_and_density


@triton.jit
def _evaluate_old(
    States,
    Full,
    Beta,
    CachedStates,
    CachedBeta,
    CachedEnergy,
    CachedDensity,
    Valid,
    LastHit,
    Context,
    ContextWeight,
    Nodes,
    PortWeight,
    LogQ,
    GateEnergy,
    Edges,
    Energy,
    Density,
    Variables: tl.constexpr,
    StateBlock: tl.constexpr,
    GroupsTotal: tl.constexpr,
    FullSize: tl.constexpr,
    EdgesTotal: tl.constexpr,
    GroupBlock: tl.constexpr,
    EdgeBlock: tl.constexpr,
):
    replica = tl.program_id(0)
    hit = tl.full((), False, tl.int1)
    if tl.load(Valid + replica):
        columns = tl.arange(0, StateBlock)
        mask = columns < Variables
        current = tl.load(States + replica * Variables + columns, mask, other=0)
        previous = tl.load(CachedStates + replica * Variables + columns, mask, other=0)
        same_state = tl.sum(((current != previous) & mask).to(tl.int32), axis=0) == 0
        hit = same_state & (tl.load(Beta + replica) == tl.load(CachedBeta + replica))
    if hit:
        tl.store(Energy + replica, tl.load(CachedEnergy + replica))
        tl.store(Density + replica, tl.load(CachedDensity + replica))
    else:
        # Keep the original FP64 reduction and its arithmetic order on a miss.
        _energy_and_density(
            Full,
            Beta,
            Context,
            ContextWeight,
            Nodes,
            PortWeight,
            LogQ,
            None,
            GateEnergy,
            Edges,
            Energy,
            Density,
            GroupsTotal,
            FullSize,
            EdgesTotal,
            GroupBlock,
            EdgeBlock,
            False,
        )
    tl.store(LastHit + replica, hit)


@triton.jit
def _commit_metadata(
    Energy,
    Forward,
    Reverse,
    Accepted,
    Beta,
    CachedEnergy,
    CachedDensity,
    CachedBeta,
    Valid,
    LastHit,
    Hits,
    Replicas: tl.constexpr,
    Block: tl.constexpr,
):
    replica = tl.program_id(0) * Block + tl.arange(0, Block)
    mask = replica < Replicas
    accepted = tl.load(Accepted + replica, mask, other=False)
    forward = tl.load(Forward + replica, mask, other=0)
    reverse = tl.load(Reverse + replica, mask, other=0)
    tl.store(CachedEnergy + replica, tl.load(Energy + replica, mask, other=0), mask)
    tl.store(CachedDensity + replica, tl.where(accepted, forward, reverse), mask)
    tl.store(CachedBeta + replica, tl.load(Beta + replica, mask, other=0), mask)
    tl.store(Valid + replica, True, mask)
    hit = tl.load(LastHit + replica, mask, other=False).to(tl.int64)
    tl.store(Hits + replica, tl.load(Hits + replica, mask, other=0) + hit, mask)


class ValidatedStateCache:
    """Own the graph's actual-state output; verify all spins and beta before reuse.

    Lookup tables belong to a fixed proposal and must not be mutated in place.
    The public proposal result remains an independent clone of graph outputs.
    """

    def __init__(self, states, beta_indices):
        self.states = torch.empty_like(states, memory_format=torch.contiguous_format)
        self.beta = torch.empty_like(beta_indices, memory_format=torch.contiguous_format)
        self.energy = torch.empty(len(states), dtype=torch.float64, device=states.device)
        self.density = torch.empty_like(self.energy)
        self.valid = torch.zeros(len(states), dtype=torch.bool, device=states.device)
        self.last_hit = torch.empty_like(self.valid)
        self.hits = torch.zeros(len(states), dtype=torch.int64, device=states.device)
        self.calls = 0

    @torch.inference_mode()
    def reset(self):
        self.valid.zero_()
        self.hits.zero_()
        self.calls = 0

    def evaluate(self, proposal, states, full, beta_indices):
        energy, density = torch.empty_like(self.energy), torch.empty_like(self.density)
        _evaluate_old[(len(states),)](
            states,
            full,
            beta_indices,
            self.states,
            self.beta,
            self.energy,
            self.density,
            self.valid,
            self.last_hit,
            proposal.context_nodes,
            proposal.context_weights,
            proposal.group_nodes,
            proposal.port_weights,
            proposal.log_probabilities,
            proposal.gate_energies,
            proposal.copy_edges,
            energy,
            density,
            Variables=states.shape[1],
            StateBlock=triton.next_power_of_2(states.shape[1]),
            GroupsTotal=proposal.group_count,
            FullSize=full.shape[1],
            EdgesTotal=len(proposal.copy_edges),
            GroupBlock=triton.next_power_of_2(proposal.group_count),
            EdgeBlock=triton.next_power_of_2(max(1, len(proposal.copy_edges))),
            num_warps=8,
        )
        return energy, density

    def commit(self, result, beta_indices):
        # The preceding torch.where writes self.states as the graph's actual output.
        # Same-stream kernel ordering completes that write before marking it valid.
        _commit_metadata[(triton.cdiv(len(self.states), 256),)](
            result.energies,
            result.log_q_forward_body,
            result.log_q_reverse_body,
            result.accepted,
            beta_indices,
            self.energy,
            self.density,
            self.beta,
            self.valid,
            self.last_hit,
            self.hits,
            Replicas=len(self.states),
            Block=256,
            num_warps=4,
        )

    def diagnostics(self):
        hits = int(self.hits.sum())
        checks = self.calls * len(self.states)
        return {
            "calls": self.calls,
            "replica_checks": checks,
            "replica_hits": hits,
            "hit_fraction": hits / checks if checks else None,
            "state_output_bytes": self.states.numel() * self.states.element_size(),
            "metadata_bytes": sum(
                t.numel() * t.element_size()
                for t in (
                    self.beta,
                    self.energy,
                    self.density,
                    self.valid,
                    self.last_hit,
                    self.hits,
                )
            ),
            "validation": "Exact equality of every free physical spin and the beta index",
        }
