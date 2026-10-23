"""Target-correct sequential auxiliary proposals for discrete Ising blocks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .junction_tree import JunctionTreePlan, sample_junction_tree_conditional
from .model import IsingModel
from .rng import RngAuditRecorder


@dataclass(frozen=True)
class SequentialHastingsResult:
    """One vectorized block proposal and its accepted target-energy change."""

    states: torch.Tensor
    accepted_energy_delta: torch.Tensor
    accepted: torch.Tensor
    proposed_change: torch.Tensor
    forward_log_probability: torch.Tensor
    reverse_log_probability: torch.Tensor
    log_acceptance_ratio: torch.Tensor


@dataclass(frozen=True)
class StateDependentMixtureHastingsResult:
    """One exact-block mixture proposal corrected for state-dependent selection."""

    states: torch.Tensor
    proposed_states: torch.Tensor
    accepted_energy_delta: torch.Tensor
    selected_plan_indices: torch.Tensor
    accepted: torch.Tensor
    proposed_change: torch.Tensor
    forward_selector_probability: torch.Tensor
    reverse_selector_probability: torch.Tensor
    log_acceptance_ratio: torch.Tensor
    zero_reverse_probability: torch.Tensor
    plan_update_counts: tuple[int, ...]
    conditional_random_draws: int
    random_draws: int


SelectorProbabilityFunction = Callable[[torch.Tensor], torch.Tensor]


def _validate_inputs(
    model: IsingModel,
    states: torch.Tensor,
    nodes: torch.Tensor,
) -> torch.Tensor:
    if states.ndim != 2 or states.shape[1] != model.size or states.dtype != torch.int8:
        raise ValueError("states must be int8 bipolar values with shape [replicas, model.size]")
    if states.device != model.coupling.device:
        raise ValueError("states and model must share one device")
    if torch.any((states != -1) & (states != 1)):
        raise ValueError("states must contain only -1 and +1")
    selected = nodes.to(device=states.device, dtype=torch.long).reshape(-1)
    if selected.numel() == 0 or selected.unique().numel() != selected.numel():
        raise ValueError("nodes must be a non-empty set")
    if int(selected.min()) < 0 or int(selected.max()) >= model.size:
        raise IndexError("nodes contain an out-of-range index")
    return selected


def _beta_column(
    beta: float | torch.Tensor,
    *,
    replicas: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(beta, dtype=dtype, device=device)
    if values.ndim == 0:
        values = values.expand(replicas)
    if values.shape != (replicas,) or not bool(torch.all(torch.isfinite(values))):
        raise ValueError("beta must be finite and scalar or have shape [replicas]")
    if bool(torch.any(values <= 0)):
        raise ValueError("beta must be positive")
    return values


def _clipped_logits(
    fields: torch.Tensor,
    beta: torch.Tensor,
    proposal_logit_clip: float,
) -> torch.Tensor:
    if not math.isfinite(proposal_logit_clip) or proposal_logit_clip <= 0:
        raise ValueError("proposal_logit_clip must be finite and positive")
    return torch.clamp(
        2.0 * beta * fields,
        min=-float(proposal_logit_clip),
        max=float(proposal_logit_clip),
    )


def sequential_proposal_log_probability(
    proposal_model: IsingModel,
    start_states: torch.Tensor,
    end_states: torch.Tensor,
    nodes: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    proposal_logit_clip: float = 12.0,
) -> torch.Tensor:
    """Return log q(end block | start) for a deterministic node scan.

    Spins outside ``nodes`` must be identical. Inside the block, each requested end
    spin is forced in scan order while subsequent local fields are updated exactly.
    The finite logit clip is part of the proposal definition and guarantees numerical
    full support; the Hastings ratio corrects it back to the original target.
    """

    selected = _validate_inputs(proposal_model, start_states, nodes)
    if end_states.shape != start_states.shape or end_states.dtype != torch.int8:
        raise ValueError("end_states must match start_states")
    if end_states.device != start_states.device or torch.any(
        (end_states != -1) & (end_states != 1)
    ):
        raise ValueError("end_states must be bipolar values on the same device")
    outside = torch.ones(proposal_model.size, dtype=torch.bool, device=start_states.device)
    outside[selected] = False
    if bool(torch.any(start_states[:, outside] != end_states[:, outside])):
        raise ValueError("A block proposal cannot change spins outside nodes")

    dtype = proposal_model.coupling.dtype
    beta_values = _beta_column(
        beta,
        replicas=start_states.shape[0],
        dtype=dtype,
        device=start_states.device,
    )
    current = start_states.index_select(1, selected).to(dtype)
    forced = end_states.index_select(1, selected).to(dtype)
    fields = proposal_model.local_fields(start_states, selected)
    internal = proposal_model.coupling.index_select(0, selected).index_select(1, selected)
    log_probability = torch.zeros(start_states.shape[0], dtype=dtype, device=start_states.device)
    for position in range(selected.numel()):
        logits = _clipped_logits(fields[:, position], beta_values, proposal_logit_clip)
        chosen = forced[:, position]
        log_probability.add_(functional.logsigmoid(chosen * logits))
        delta = chosen - current[:, position]
        if position + 1 < selected.numel():
            fields[:, position + 1 :].add_(
                delta[:, None] * internal[position, position + 1 :][None, :]
            )
        current[:, position] = chosen
    return log_probability


def _target_block_energy_delta(
    target_model: IsingModel,
    start_states: torch.Tensor,
    proposed_states: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    dtype = target_model.coupling.dtype
    old = start_states.index_select(1, selected).to(dtype)
    new = proposed_states.index_select(1, selected).to(dtype)
    internal = target_model.coupling.index_select(0, selected).index_select(1, selected)
    full_fields = target_model.local_fields(start_states, selected)
    boundary = full_fields - old @ internal.mT
    old_energy = -torch.sum(old * boundary, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r", old, internal, old
    )
    new_energy = -torch.sum(new * boundary, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r", new, internal, new
    )
    return new_energy - old_energy


def sequential_auxiliary_hastings_step(
    target_model: IsingModel,
    proposal_model: IsingModel,
    states: torch.Tensor,
    nodes: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator,
    proposal_logit_clip: float = 12.0,
    rng_recorder: RngAuditRecorder | None = None,
) -> SequentialHastingsResult:
    """Apply one full-support auxiliary Gibbs-scan proposal with MH correction."""

    selected = _validate_inputs(target_model, states, nodes)
    if (
        proposal_model.names != target_model.names
        or proposal_model.coupling.dtype != target_model.coupling.dtype
        or proposal_model.coupling.device != target_model.coupling.device
    ):
        raise ValueError("Target and proposal models must share variables, dtype, and device")
    dtype = target_model.coupling.dtype
    beta_values = _beta_column(
        beta,
        replicas=states.shape[0],
        dtype=dtype,
        device=states.device,
    )
    proposed = states.clone()
    current = states.index_select(1, selected).to(dtype)
    fields = proposal_model.local_fields(states, selected)
    internal = proposal_model.coupling.index_select(0, selected).index_select(1, selected)
    forward = torch.zeros(states.shape[0], dtype=dtype, device=states.device)
    for position in range(selected.numel()):
        logits = _clipped_logits(fields[:, position], beta_values, proposal_logit_clip)
        uniforms = torch.rand(
            states.shape[0], dtype=dtype, device=states.device, generator=generator
        )
        if rng_recorder is not None:
            rng_recorder.record("hastings_proposal_uniform", states.shape[0])
        chosen = (uniforms < torch.sigmoid(logits)).mul(2).sub(1).to(dtype)
        forward.add_(functional.logsigmoid(chosen * logits))
        delta = chosen - current[:, position]
        if position + 1 < selected.numel():
            fields[:, position + 1 :].add_(
                delta[:, None] * internal[position, position + 1 :][None, :]
            )
        current[:, position] = chosen
        proposed[:, selected[position]] = chosen.to(torch.int8)

    reverse = sequential_proposal_log_probability(
        proposal_model,
        proposed,
        states,
        selected,
        beta_values,
        proposal_logit_clip=proposal_logit_clip,
    )
    target_delta = _target_block_energy_delta(target_model, states, proposed, selected)
    log_acceptance = -beta_values * target_delta + reverse - forward
    acceptance_uniform = torch.rand(
        states.shape[0], dtype=dtype, device=states.device, generator=generator
    )
    if rng_recorder is not None:
        rng_recorder.record("hastings_acceptance_uniform", states.shape[0])
    accepted = torch.log(acceptance_uniform) < torch.minimum(
        torch.zeros_like(log_acceptance), log_acceptance
    )
    proposed_change = torch.any(proposed[:, selected] != states[:, selected], dim=1)
    updated = states.clone()
    updated[:, selected] = torch.where(
        accepted[:, None], proposed[:, selected], states[:, selected]
    )
    return SequentialHastingsResult(
        states=updated,
        accepted_energy_delta=torch.where(accepted, target_delta, 0.0),
        accepted=accepted,
        proposed_change=proposed_change,
        forward_log_probability=forward,
        reverse_log_probability=reverse,
        log_acceptance_ratio=log_acceptance,
    )


def _selector_probability_matrix(
    selector_probabilities: SelectorProbabilityFunction,
    states: torch.Tensor,
    *,
    plan_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    probabilities = torch.as_tensor(
        selector_probabilities(states),
        device=states.device,
        dtype=dtype,
    )
    if probabilities.shape != (states.shape[0], plan_count):
        raise ValueError("selector probabilities must have shape [replicas, plans]")
    if bool(torch.any(~torch.isfinite(probabilities))) or bool(torch.any(probabilities < 0)):
        raise ValueError("selector probabilities must be finite and nonnegative")
    sums = torch.sum(probabilities, dim=1)
    tolerance = max(1e-12, 32.0 * torch.finfo(dtype).eps)
    if not bool(torch.all(torch.abs(sums - 1.0) <= tolerance)):
        raise ValueError("selector probabilities must sum to one")
    return probabilities


def state_dependent_junction_tree_mixture_hastings_step(
    target_model: IsingModel,
    states: torch.Tensor,
    plans: Sequence[JunctionTreePlan],
    beta: float | torch.Tensor,
    *,
    selector_probabilities: SelectorProbabilityFunction,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
) -> StateDependentMixtureHastingsResult:
    """Correct a state-dependent mixture of fixed-support exact Gibbs kernels.

    A plan label ``k`` is selected with probability ``rho(k | x)`` and its exact
    target conditional proposes ``y``.  Because every fixed-plan Gibbs kernel is
    reversible for the target, the Metropolis-Hastings ratio reduces to
    ``rho(k | y) / rho(k | x)``.  Zero reverse selector probability produces a
    deterministic rejection and therefore preserves, but may not explore, the target.
    """

    if states.ndim != 2 or states.shape[1] != target_model.size or states.dtype != torch.int8:
        raise ValueError("states must be int8 bipolar values with shape [replicas, model.size]")
    if states.device != target_model.coupling.device:
        raise ValueError("states and model must share one device")
    if states.shape[0] == 0 or bool(torch.any((states != -1) & (states != 1))):
        raise ValueError("states must contain at least one bipolar replica")
    prepared = tuple(plans)
    if not prepared:
        raise ValueError("plans must be non-empty")
    expected_nodes = set(range(target_model.size))
    for plan in prepared:
        if not plan.nodes or set(plan.nodes) & set(plan.boundary_nodes):
            raise ValueError("each plan must have a non-empty support disjoint from its boundary")
        if set(plan.nodes) | set(plan.boundary_nodes) != expected_nodes:
            raise ValueError("each plan must partition the model into support and boundary nodes")

    dtype = target_model.coupling.dtype
    beta_values = _beta_column(
        beta,
        replicas=states.shape[0],
        dtype=dtype,
        device=states.device,
    )
    forward_matrix = _selector_probability_matrix(
        selector_probabilities,
        states,
        plan_count=len(prepared),
        dtype=dtype,
    )
    selector_uniforms = torch.rand(
        states.shape[0],
        dtype=dtype,
        device=states.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("state_dependent_selector_uniform", states.shape[0])
    cumulative = torch.cumsum(forward_matrix, dim=1)
    cumulative[:, -1] = 1.0
    selected = torch.sum(selector_uniforms[:, None] >= cumulative, dim=1).to(torch.long)
    rows = torch.arange(states.shape[0], dtype=torch.long, device=states.device)
    forward = forward_matrix[rows, selected]
    if bool(torch.any(forward <= 0)):
        raise RuntimeError("the categorical selector chose a zero-probability plan")

    proposed = states.clone()
    conditional_random_draws = 0
    update_counts: list[int] = []
    for plan_index, plan in enumerate(prepared):
        replica_indices = torch.nonzero(selected == plan_index, as_tuple=False).flatten()
        update_counts.append(int(replica_indices.numel()))
        if replica_indices.numel() == 0:
            continue
        sampled = sample_junction_tree_conditional(
            target_model,
            plan,
            states.index_select(0, replica_indices),
            beta_values.index_select(0, replica_indices),
            generator=generator,
            rng_recorder=rng_recorder,
        )
        proposed.index_copy_(0, replica_indices, sampled.states)
        conditional_random_draws += sampled.random_draws

    reverse_matrix = _selector_probability_matrix(
        selector_probabilities,
        proposed,
        plan_count=len(prepared),
        dtype=dtype,
    )
    reverse = reverse_matrix[rows, selected]
    negative_infinity = torch.full_like(reverse, -torch.inf)
    reverse_log_probability = torch.where(reverse > 0, torch.log(reverse), negative_infinity)
    log_acceptance = reverse_log_probability - torch.log(forward)
    acceptance_uniforms = torch.rand(
        states.shape[0],
        dtype=dtype,
        device=states.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("state_dependent_acceptance_uniform", states.shape[0])
    accepted = torch.log(acceptance_uniforms) < torch.minimum(
        torch.zeros_like(log_acceptance), log_acceptance
    )
    proposed_change = torch.any(proposed != states, dim=1)
    proposed_energy_delta = target_model.energy(proposed) - target_model.energy(states)
    updated = torch.where(accepted[:, None], proposed, states)
    return StateDependentMixtureHastingsResult(
        states=updated,
        proposed_states=proposed,
        accepted_energy_delta=torch.where(accepted, proposed_energy_delta, 0.0),
        selected_plan_indices=selected,
        accepted=accepted,
        proposed_change=proposed_change,
        forward_selector_probability=forward,
        reverse_selector_probability=reverse,
        log_acceptance_ratio=log_acceptance,
        zero_reverse_probability=reverse <= 0,
        plan_update_counts=tuple(update_counts),
        conditional_random_draws=conditional_random_draws,
        random_draws=2 * states.shape[0] + conditional_random_draws,
    )
