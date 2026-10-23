"""Hybrid tour search with CPU AB-cycle selection and CPU/CUDA local descent.

The inner energy is two-factor length / distance_scale plus a penalty per extra
subtour. Its connectivity term is nonquadratic. Binary heat-bath updates sample
this fixed auxiliary model; changing parents, annealing, explicit repair, and
polishing make the complete solver an optimization heuristic, not a Boltzmann
sampler for the original TSP. CUDA accelerates local descent only; this is not
native probabilistic-device execution.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from random import Random
from time import perf_counter

import numpy as np

from .hybrid_backend import LocalSearchBackend
from .hybrid_local import first_descent
from .hybrid_moves import (
    ABCycle,
    Tour,
    alternating_cycles,
    canonical_tour,
    edge_frequencies,
    factor_components,
    nearest_candidate_edges,
    repair_subtours,
    select_uncertain_cycles,
    selected_edges,
    tour_edges,
)
from .instance import TSPInstance, tour_length

EventSink = Callable[[dict], None]


class BudgetExpired(RuntimeError):
    """A cooperative wall-clock budget expired between bounded operations."""


@dataclass(frozen=True)
class Deadline:
    stop: float

    def check(self) -> None:
        if perf_counter() >= self.stop:
            raise BudgetExpired("The hybrid wall-clock budget has expired")

    def remaining(self) -> float:
        return max(0.0, self.stop - perf_counter())


@dataclass(frozen=True)
class HybridConfig:
    max_seconds: float
    kind: str = "hybrid"
    device: str = "cpu"
    seed: int = 0
    rounds: int = 20
    elite_size: int = 8
    candidate_neighbors: int = 12
    max_cycles: int = 12
    selector: str = "pbit"
    replicas: int = 8
    sweeps: int = 32
    max_evaluations: int = 4096
    proposals_per_round: int = 8
    beta_start: float = 0.1
    beta_stop: float = 8.0
    subtour_penalty: float = 2.0
    local_search_passes: int = 4
    local_search: str = "two_opt"
    stop_when_covered: bool = True
    rng_checkpoint_interval: int = 1
    local_search_batch_size: int = 1
    cuda_tile: int = 256

    def __post_init__(self) -> None:
        if self.kind != "hybrid" or self.device not in {"cpu", "cuda"}:
            raise ValueError("The hybrid requires kind=hybrid and device=cpu or cuda")
        if self.selector not in {"pbit", "metropolis", "greedy", "enumerate", "classical"}:
            raise ValueError("Unknown hybrid selector")
        if self.local_search not in {"two_opt", "two_opt_or_opt"}:
            raise ValueError("solver.local_search must be two_opt or two_opt_or_opt")
        for name in (
            "rounds",
            "elite_size",
            "candidate_neighbors",
            "max_cycles",
            "replicas",
            "sweeps",
            "max_evaluations",
            "proposals_per_round",
            "local_search_passes",
            "rng_checkpoint_interval",
            "local_search_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"solver.{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("solver.seed must be a nonnegative integer")
        if self.max_cycles > 20:
            raise ValueError("The CPU reference selector supports at most 20 free cycle bits")
        for name in ("max_seconds", "beta_start", "beta_stop", "subtour_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"solver.{name} must be finite and positive")
        if self.selector == "enumerate" and self.max_evaluations < 1 << self.max_cycles:
            raise ValueError("Enumeration requires max_evaluations >= 2**max_cycles")
        if not isinstance(self.stop_when_covered, bool):
            raise ValueError("solver.stop_when_covered must be a boolean")
        if self.cuda_tile not in {128, 256, 512, 1024}:
            raise ValueError("solver.cuda_tile must be 128, 256, 512 or 1024")


class ElitePool:
    """Keep certified, orientation-deduplicated elites with exact integer costs."""

    def __init__(self, instance: TSPInstance, capacity: int):
        self.instance, self.capacity = instance, capacity
        self.lengths: dict[Tour, int] = {}

    def add(self, tour: Sequence[int]) -> tuple[Tour, int]:
        canonical = canonical_tour(self.instance, tour)
        length = tour_length(self.instance, canonical, validate=False)
        self.lengths[canonical] = length
        ordered = sorted(self.lengths, key=lambda item: (self.lengths[item], item))
        self.lengths = {item: self.lengths[item] for item in ordered[: self.capacity]}
        return canonical, length

    @property
    def tours(self) -> tuple[Tour, ...]:
        return tuple(self.lengths)

    @property
    def best(self) -> Tour:
        if not self.lengths:
            raise ValueError("An elite pool cannot be queried before initialization")
        return next(iter(self.lengths))

    def payload(self) -> list[dict]:
        return [{"tour": tour, "length": length} for tour, length in self.lengths.items()]


def perturb_tour(tour: Tour, rng: Random) -> Tour:
    """Make a genuine four-cut double bridge, including a nonzero first segment."""
    if len(tour) >= 8:
        a, b, c, d = sorted(rng.sample(range(1, len(tour)), 4))
        return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
    a, b = sorted(rng.sample(range(1, len(tour)), 2))
    values = list(tour)
    values[a], values[b] = values[b], values[a]
    return tuple(values)


def polish_tour(
    instance: TSPInstance,
    tour: Tour,
    passes: int,
    deadline: Deadline,
    method: str = "two_opt",
    backend: LocalSearchBackend | None = None,
) -> Tour:
    """Bound local descent by passes; check the cooperative deadline between passes."""
    if backend is not None:
        return backend.solve((tour,), deadline.remaining)[0]
    return first_descent(
        instance, tour, passes, deadline.remaining, or_opt=method == "two_opt_or_opt"
    )


@dataclass(frozen=True)
class FactorState:
    mask: int
    length: int
    components: int

    def score(self, scale: float, penalty: float) -> float:
        return self.length / scale + penalty * (self.components - 1)


@dataclass(frozen=True)
class SelectionResult:
    states: tuple[FactorState, ...]
    evaluations: int
    proposals: int
    accepted: int
    stop_reason: str
    enumeration_complete: bool


def select_cycle_bits(
    instance: TSPInstance,
    parent: Tour,
    cycles: tuple[ABCycle, ...],
    config: HybridConfig,
    scale: float,
    rng: Random,
    deadline: Deadline,
    emit: EventSink,
    *,
    record_rng_state: bool = True,
) -> SelectionResult:
    """Select coupled cycle bits and retain only actually visited bit states.

    An energy evaluation counts a cache miss. Both proposal attempts and unique
    evaluations are reported. Enumeration is exhaustive only for the auxiliary
    fixed-parent model, not for the complete TSP or its repaired-tour objective.
    """
    base = tour_edges(parent)
    base_length = sum(int(instance.distance[a, b]) for a, b in base)
    cycle_deltas = tuple(
        sum(int(instance.distance[a, b]) for a, b in cycle.added)
        - sum(int(instance.distance[a, b]) for a, b in cycle.removed)
        for cycle in cycles
    )
    cache: dict[int, FactorState] = {}
    visited: set[int] = set()
    proposals = accepted = 0

    def evaluate(mask: int) -> FactorState | None:
        deadline.check()
        if mask in cache:
            return cache[mask]
        if len(cache) >= config.max_evaluations:
            return None
        bonds = selected_edges(base, cycles, mask)
        components = factor_components(instance.dimension, bonds)
        length = base_length + sum(
            delta for index, delta in enumerate(cycle_deltas) if mask & (1 << index)
        )
        state = FactorState(mask, length, len(components))
        cache[mask] = state
        return state

    reason = "sweep_limit"
    complete = False
    states: list[FactorState] = []
    try:
        if config.selector == "enumerate":
            for mask in range(1 << len(cycles)):
                state = evaluate(mask)
                if state is None:
                    reason = "evaluation_limit"
                    break
                visited.add(mask)
            else:
                reason, complete = "enumerated_auxiliary_model", True
        else:
            masks = [0]
            if config.replicas > 1:
                masks.append((1 << len(cycles)) - 1)
            masks.extend(rng.getrandbits(len(cycles)) for _ in range(config.replicas - len(masks)))
            for mask in masks:
                state = evaluate(mask)
                if state is None:
                    reason = "evaluation_limit"
                    break
                states.append(state)
                visited.add(mask)
            emit({"stage": "selector_initial", "states": [asdict(s) for s in states]})
            for sweep in range(config.sweeps):
                if reason == "evaluation_limit":
                    break
                # Once all masks have been visited, later sweeps cannot add a proposal.
                # This is an optimization stopping rule, not an equilibrium claim.
                if config.stop_when_covered and len(visited) == 1 << len(cycles):
                    reason = "state_space_covered"
                    break
                fraction = sweep / max(1, config.sweeps - 1)
                beta = config.beta_start + fraction * (config.beta_stop - config.beta_start)
                for replica, current in enumerate(states):
                    order = list(range(len(cycles)))
                    rng.shuffle(order)
                    for bit in order:
                        proposal = evaluate(current.mask ^ (1 << bit))
                        if proposal is None:
                            reason = "evaluation_limit"
                            break
                        proposals += 1
                        # Subtract exact integer lengths before converting to floating point.
                        delta = (
                            proposal.length - current.length
                        ) / scale + config.subtour_penalty * (
                            proposal.components - current.components
                        )
                        scaled = beta * delta
                        if config.selector == "pbit":
                            tail = math.exp(-abs(scaled))
                            probability = tail / (1.0 + tail) if scaled >= 0 else 1.0 / (1.0 + tail)
                        elif config.selector == "metropolis":
                            probability = math.exp(-max(0.0, scaled))
                        else:
                            probability = float(delta <= 0)
                        if rng.random() < probability:
                            current = proposal
                            accepted += 1
                            visited.add(current.mask)
                        states[replica] = current
                    if reason == "evaluation_limit":
                        break
                emit(
                    {
                        "stage": "selector_sweep",
                        "sweep": sweep + 1,
                        "beta": beta,
                        "states": [asdict(s) for s in states],
                        "evaluations": len(cache),
                        "proposals": proposals,
                        "accepted": accepted,
                    }
                )
    except BudgetExpired:
        reason = "wall_clock_limit"
    finally:
        emit(
            {
                "stage": "selector_final",
                "states": [asdict(s) for s in states],
                "visited_states": [asdict(cache[mask]) for mask in sorted(visited)],
                "evaluations": len(cache),
                "proposals": proposals,
                "accepted": accepted,
                "rng_state": rng.getstate() if record_rng_state else None,
            }
        )
    ranked = sorted(
        (cache[mask] for mask in visited),
        key=lambda state: (state.score(scale, config.subtour_penalty), state.length, state.mask),
    )
    result = SelectionResult(
        tuple(ranked[: config.proposals_per_round]),
        len(cache),
        proposals,
        accepted,
        reason,
        complete,
    )
    emit({"stage": "selection_complete", **asdict(result)})
    return result


def search(
    instance: TSPInstance,
    pool: ElitePool,
    config: HybridConfig,
    deadline: Deadline,
    emit: EventSink,
    backend: LocalSearchBackend | None = None,
) -> dict:
    """Continue classical search and optionally inject certified hybrid offspring."""
    if config.device == "cuda" and backend is None:
        raise ValueError("CUDA search requires an initialized local-search backend")
    classical_rng = Random(config.seed + 1)
    cycle_rng = Random(config.seed + 2)
    bit_rng = Random(config.seed + 3)
    upper = instance.distance[np.triu_indices(instance.dimension, k=1)]
    positive = upper[upper > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    rounds = energy_evaluations = 0
    stopped = "round_limit"
    phase_seconds = {
        "candidates": 0.0,
        "classical": 0.0,
        "selection": 0.0,
        "repair": 0.0,
        "polishing": 0.0,
    }

    def observe(stage: str, tour: Tour, **details) -> None:
        certified, length = pool.add(tour)
        emit(
            {
                "stage": stage,
                "tour": certified,
                "length": length,
                "elite_pool": pool.payload(),
                **details,
            }
        )

    try:
        phase = perf_counter()
        nearest = (
            frozenset()
            if config.selector == "classical"
            else nearest_candidate_edges(instance, config.candidate_neighbors, deadline.check)
        )
        phase_seconds["candidates"] += perf_counter() - phase
        for round_index in range(config.rounds):
            deadline.check()
            emit(
                {
                    "stage": "round_start",
                    "round": round_index,
                    "rng_states": {
                        "classical": classical_rng.getstate(),
                        "cycles": cycle_rng.getstate(),
                        "bits": bit_rng.getstate(),
                    }
                    if round_index % config.rng_checkpoint_interval == 0
                    else None,
                }
            )
            phase = perf_counter()
            parent = pool.tours[round_index % len(pool.tours)]
            kicked = perturb_tour(parent, classical_rng)
            classical = polish_tour(
                instance, kicked, config.local_search_passes, deadline, config.local_search, backend
            )
            observe("classical_candidate", classical, round=round_index)
            phase_seconds["classical"] += perf_counter() - phase
            if config.selector == "classical":
                rounds += 1
                continue
            deadline.check()

            alternatives = [tour for tour in pool.tours if tour != parent]
            if not alternatives:
                alternatives = [canonical_tour(instance, kicked)]
            parent_edges = tour_edges(parent)
            donor = max(
                alternatives,
                key=lambda tour: (len(tour_edges(tour) - parent_edges), tour),
            )
            phase = perf_counter()
            consensus_tours = tuple(dict.fromkeys((*pool.tours, parent, donor)))
            frequencies = edge_frequencies(consensus_tours)
            candidates = nearest | frozenset(frequencies) | tour_edges(donor)
            all_cycles = alternating_cycles(parent, donor, cycle_rng)
            cycles = select_uncertain_cycles(
                instance, all_cycles, frequencies, candidates, config.max_cycles, cycle_rng
            )
            emit(
                {
                    "stage": "cycle_model",
                    "round": round_index,
                    "parent": parent,
                    "donor": donor,
                    "edge_support_population": consensus_tours,
                    "cycles": [asdict(c) for c in cycles],
                    "available_cycles": len(all_cycles),
                    "distance_scale": scale,
                    "subtour_penalty": config.subtour_penalty,
                }
            )
            phase_seconds["candidates"] += perf_counter() - phase
            if not cycles:
                emit({"stage": "no_alternating_cycles", "round": round_index})
                rounds += 1
                continue
            phase = perf_counter()
            result = select_cycle_bits(
                instance,
                parent,
                cycles,
                config,
                scale,
                bit_rng,
                deadline,
                lambda event, index=round_index: emit({"round": index, **event}),
                record_rng_state=round_index % config.rng_checkpoint_interval == 0,
            )
            energy_evaluations += result.evaluations
            phase_seconds["selection"] += perf_counter() - phase
            if result.stop_reason == "wall_clock_limit":
                stopped = "wall_clock_limit"
                break
            for offset in range(0, len(result.states), config.local_search_batch_size):
                states = result.states[offset : offset + config.local_search_batch_size]
                repaired_tours = []
                for state in states:
                    deadline.check()
                    phase = perf_counter()
                    bonds = selected_edges(parent_edges, cycles, state.mask)
                    repaired, repairs = repair_subtours(instance, bonds, candidates, deadline.check)
                    observe(
                        "repaired_candidate",
                        repaired,
                        round=round_index,
                        selected_state=asdict(state),
                        repairs=repairs,
                    )
                    phase_seconds["repair"] += perf_counter() - phase
                    repaired_tours.append(repaired)
                deadline.check()
                phase = perf_counter()
                polished_tours = (
                    backend.solve(repaired_tours, deadline.remaining)
                    if backend is not None
                    else tuple(
                        polish_tour(
                            instance,
                            tour,
                            config.local_search_passes,
                            deadline,
                            config.local_search,
                        )
                        for tour in repaired_tours
                    )
                )
                for state, polished in zip(states, polished_tours, strict=True):
                    observe("polished_candidate", polished, round=round_index, mask=state.mask)
                phase_seconds["polishing"] += perf_counter() - phase
            rounds += 1
    except BudgetExpired:
        stopped = "wall_clock_limit"
    return {
        "stop_reason": stopped,
        "completed_rounds": rounds,
        "completed_selection_energy_evaluations": energy_evaluations,
        "completed_phase_seconds": phase_seconds,
        "elite_pool": pool.payload(),
        "rng_states": {
            "classical": classical_rng.getstate(),
            "cycles": cycle_rng.getstate(),
            "bits": bit_rng.getstate(),
        },
        "timing_note": "Phase sums omit interrupted operations; enclosing wall time includes them.",
        "sampling_claim": "annealed auxiliary two-factor model with explicit repair; heuristic TSP",
        "replica_role": "coupled proposal generation, not independent TSP trials",
    }
