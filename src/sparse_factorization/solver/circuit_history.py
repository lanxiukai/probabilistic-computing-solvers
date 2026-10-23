"""Per-event physical history updates without dynamic-index host synchronization."""

from __future__ import annotations

import triton
import triton.language as tl

HISTORY_FIELDS = 7


@triton.jit
def _commit_history(
    States,
    Energies,
    BestStates,
    BestEnergies,
    FirstStates,
    FirstSweeps,
    ActualStates,
    ActualEnergy,
    ProposalEnergy,
    Accepted,
    RootChanged,
    AcceptedCounts,
    RootCounts,
    Payload,
    RawSweep,
    Variables: tl.constexpr,
    Block: tl.constexpr,
):
    replica = tl.program_id(0)
    columns = tl.arange(0, Block)
    valid = columns < Variables
    offsets = replica * Variables + columns
    old = tl.load(States + offsets, valid, other=0)
    state = tl.load(ActualStates + offsets, valid, other=0)
    old_energy = tl.load(Energies + replica).to(tl.float64)
    exact_energy = tl.load(ActualEnergy + replica)
    energy = exact_energy.to(tl.float32)
    best = tl.load(BestEnergies + replica)
    first = tl.load(FirstSweeps + replica)
    improved = energy < best
    fresh = (energy == 0) & (first < 0)
    first = tl.where(fresh, RawSweep, first)
    accepted = tl.load(Accepted + replica).to(tl.int64)
    roots = (accepted != 0) & tl.load(RootChanged + replica)
    zero_proposal = tl.load(ProposalEnergy + replica) == 0

    # Scalar metadata loads are replicated across warps. All warps must read the
    # old best/first markers before a leader stores the new markers, otherwise
    # late readers can suppress part of a vector history write.
    tl.debug_barrier()

    tl.store(States + offsets, state, valid)
    tl.store(Energies + replica, energy)
    tl.store(BestStates + offsets, state, valid & improved)
    tl.store(BestEnergies + replica, energy, improved)
    tl.store(FirstStates + offsets, state, valid & fresh)
    tl.store(FirstSweeps + replica, first)
    tl.store(AcceptedCounts + replica, tl.load(AcceptedCounts + replica) + accepted)
    tl.store(RootCounts + replica, tl.load(RootCounts + replica) + roots.to(tl.int64))

    base = Payload + replica * 7
    tl.store(base, first)
    tl.store(base + 1, tl.sum(((old != state) & valid).to(tl.int32), axis=0))
    # Preserve the FP64 delta bits; first-sweep ordinals remain genuine int64.
    tl.store(base + 2, (exact_energy - old_energy).to(tl.int64, bitcast=True))
    tl.store(base + 3, accepted)
    tl.store(base + 4, roots.to(tl.int64))
    tl.store(base + 5, fresh.to(tl.int64))
    tl.store(base + 6, zero_proposal.to(tl.int64))


def commit_history(engine, result, accepted_counts, root_counts, payload):
    """Commit every replica and transfer one completed event payload to the host."""
    _commit_history[(engine.replicas,)](
        engine.states,
        engine.energies,
        engine.best_states,
        engine.best_energies,
        engine.first_hit_states,
        engine.first_hit_sweeps,
        result.states,
        result.energies,
        result.proposed_energies,
        result.accepted,
        result.root_changed,
        accepted_counts,
        root_counts,
        payload,
        engine.global_sweep,
        Variables=engine.variables,
        Block=triton.next_power_of_2(engine.variables),
        num_warps=4,
    )
    # Blocking host copy completes this event before native observation or PT.
    return payload.cpu().numpy()
