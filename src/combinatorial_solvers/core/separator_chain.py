"""Exact heat-bath updates for group chains with separator-compressed messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .rng import RngAuditRecorder


@dataclass(frozen=True)
class SeparatorGatePlan:
    """Enumerated states and internal scores for one disjoint chain group."""

    positions: torch.Tensor
    candidates: torch.Tensor
    internal_scores: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.positions.numel())

    @property
    def state_count(self) -> int:
        return int(self.candidates.shape[0])


@dataclass(frozen=True)
class SeparatorLinkPlan:
    """Compressed state maps and interaction scores across one chain separator."""

    left_state_codes: torch.Tensor
    right_state_codes: torch.Tensor
    separator_scores: torch.Tensor
    left_separator_size: int
    right_separator_size: int
    left_group_order: torch.Tensor | None = None

    @property
    def left_state_count(self) -> int:
        return int(self.separator_scores.shape[0])

    @property
    def right_state_count(self) -> int:
        return int(self.separator_scores.shape[1])


@dataclass(frozen=True)
class SeparatorChainPlan:
    """Prepared exact conditional whose cost is exponential in group/separator width."""

    nodes: torch.Tensor
    internal_coupling: torch.Tensor
    gates: tuple[SeparatorGatePlan, ...]
    links: tuple[SeparatorLinkPlan, ...]
    treewidth_upper_bound: int

    @property
    def node_count(self) -> int:
        return int(self.nodes.numel())

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def candidate_evaluations_per_replica(self) -> int:
        unary = sum(gate.state_count for gate in self.gates)
        separators = sum(int(link.separator_scores.numel()) for link in self.links)
        return unary + separators


def _normalize_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(int(node) for node in group) for group in groups if group)
    if not normalized:
        raise ValueError("A separator chain requires at least one non-empty group")
    if any(len(group) != len(set(group)) for group in normalized):
        raise ValueError("A separator-chain group cannot repeat a node")
    flattened = [node for group in normalized for node in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Separator-chain groups must be disjoint")
    return normalized


def _candidate_states(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    codes = torch.arange(1 << size, dtype=torch.int64, device=device)
    return ((codes[:, None] >> torch.arange(size, device=device)) & 1).mul(2).sub(1).to(dtype)


def _state_codes(candidates: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
    bits = candidates.index_select(1, positions).gt(0).to(torch.long)
    weights = 1 << torch.arange(positions.numel(), device=candidates.device)
    return torch.sum(bits * weights[None, :], dim=1)


def separator_chain_treewidth_upper_bound(
    coupling: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> int:
    """Validate nearest-group coupling and return the explicit message-width bound."""

    normalized = _normalize_groups(groups)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be square")
    flattened = [node for group in normalized for node in group]
    if any(node < 0 or node >= coupling.shape[0] for node in flattened):
        raise IndexError("A separator-chain group contains an out-of-range node")
    model_device = coupling.device
    group_nodes = [
        torch.tensor(group, dtype=torch.long, device=model_device) for group in normalized
    ]
    width = max(len(group) - 1 for group in normalized)
    for left_index, left in enumerate(group_nodes):
        for right_index, right in enumerate(group_nodes[left_index + 1 :], left_index + 1):
            cross = coupling.index_select(0, left).index_select(1, right)
            if not bool(torch.any(cross != 0)):
                continue
            if right_index != left_index + 1:
                raise ValueError("Separator-chain groups have a non-nearest cross coupling")
            left_size = int(torch.count_nonzero(torch.any(cross != 0, dim=1)).item())
            right_size = int(torch.count_nonzero(torch.any(cross != 0, dim=0)).item())
            width = max(width, left_size + right_size - 1)
    return width


def prepare_separator_chain(
    coupling: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    maximum_treewidth: int,
    message_reduction: str = "masked_v1",
) -> SeparatorChainPlan:
    """Prepare a chain conditional with arbitrary nearest-group separator edges."""

    if maximum_treewidth < 0:
        raise ValueError("maximum_treewidth must be nonnegative")
    if message_reduction not in {"masked_v1", "grouped_v1"}:
        raise ValueError("Unknown separator message reduction")
    normalized = _normalize_groups(groups)
    width = separator_chain_treewidth_upper_bound(coupling, normalized)
    if width > maximum_treewidth:
        raise ValueError(
            f"Separator-chain treewidth {width} exceeds maximum_treewidth={maximum_treewidth}"
        )
    flattened = [node for group in normalized for node in group]

    model_device = coupling.device
    nodes_model = torch.tensor(flattened, dtype=torch.long, device=model_device)
    internal = coupling.index_select(0, nodes_model).index_select(1, nodes_model)
    internal = internal.to(device=device, dtype=dtype)
    nodes = nodes_model.to(device=device)

    gates: list[SeparatorGatePlan] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    for group in normalized:
        stop = start + len(group)
        positions = torch.arange(start, stop, dtype=torch.long, device=device)
        candidates = _candidate_states(len(group), device=device, dtype=dtype)
        local = internal[start:stop, start:stop]
        scores = 0.5 * torch.einsum("qi,ij,qj->q", candidates, local, candidates)
        gates.append(
            SeparatorGatePlan(
                positions=positions,
                candidates=candidates,
                internal_scores=scores,
            )
        )
        offsets.append((start, stop))
        start = stop

    links: list[SeparatorLinkPlan] = []
    for index in range(len(gates) - 1):
        left_start, left_stop = offsets[index]
        right_start, right_stop = offsets[index + 1]
        cross = internal[left_start:left_stop, right_start:right_stop]
        left_positions = torch.where(torch.any(cross != 0, dim=1))[0]
        right_positions = torch.where(torch.any(cross != 0, dim=0))[0]
        left_separator = _candidate_states(int(left_positions.numel()), device=device, dtype=dtype)
        right_separator = _candidate_states(
            int(right_positions.numel()), device=device, dtype=dtype
        )
        selected_cross = cross.index_select(0, left_positions).index_select(1, right_positions)
        separator_scores = left_separator @ selected_cross @ right_separator.T
        left_codes = _state_codes(gates[index].candidates, left_positions)
        links.append(
            SeparatorLinkPlan(
                left_state_codes=left_codes,
                right_state_codes=_state_codes(gates[index + 1].candidates, right_positions),
                separator_scores=separator_scores,
                left_separator_size=int(left_positions.numel()),
                right_separator_size=int(right_positions.numel()),
                left_group_order=(
                    torch.argsort(left_codes, stable=True)
                    if message_reduction == "grouped_v1"
                    else None
                ),
            )
        )
    return SeparatorChainPlan(
        nodes=nodes,
        internal_coupling=internal,
        gates=tuple(gates),
        links=tuple(links),
        treewidth_upper_bound=width,
    )


def _beta_vector(
    beta: float | torch.Tensor,
    *,
    replicas: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(beta, torch.Tensor):
        values = beta.to(device=device, dtype=dtype)
        if values.shape != (replicas,):
            raise ValueError("Per-replica beta must have shape [replicas]")
        return values
    return torch.full((replicas,), float(beta), dtype=dtype, device=device)


def _draw_logits(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    uniforms = torch.rand(
        (logits.shape[0], 1),
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("separator_corridor_uniform", logits.shape[0])
    return torch.sum(uniforms > torch.cumsum(probabilities, dim=-1), dim=-1).clamp_max(
        logits.shape[-1] - 1
    )


def _compress_logits(
    logits: torch.Tensor,
    state_codes: torch.Tensor,
    state_count: int,
) -> torch.Tensor:
    return torch.stack(
        [torch.logsumexp(logits[:, state_codes == state], dim=1) for state in range(state_count)],
        dim=1,
    )


def _compress_logits_grouped(
    logits: torch.Tensor,
    group_order: torch.Tensor,
    state_count: int,
) -> torch.Tensor:
    """Reduce full bag enumeration in stable separator-code groups.

    Every d-spin code occurs 2**(k-d) times among all k-spin states. The
    permutation is prepared from graph structure, never from sampled logits.
    """

    grouped = logits.index_select(1, group_order).reshape(logits.shape[0], state_count, -1)
    return torch.logsumexp(grouped, dim=2)


def sample_separator_chain(
    plan: SeparatorChainPlan,
    selected: torch.Tensor,
    fields: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw an exact conditional chain state using separator-state messages."""

    if selected.ndim != 2 or selected.shape != fields.shape:
        raise ValueError("selected and fields must share shape [replicas, chain_nodes]")
    if selected.shape[1] != plan.node_count:
        raise ValueError("selected chain width does not match the separator plan")
    if torch.any((selected != -1) & (selected != 1)):
        raise ValueError("selected must contain only -1 and +1")
    replicas = selected.shape[0]
    dtype = plan.internal_coupling.dtype
    values = selected.to(dtype=dtype)
    boundary_fields = fields.to(dtype=dtype) - values @ plan.internal_coupling
    beta_values = _beta_vector(
        beta,
        replicas=replicas,
        device=values.device,
        dtype=dtype,
    )

    gate_logits: list[torch.Tensor] = []
    for gate in plan.gates:
        boundary = boundary_fields.index_select(1, gate.positions)
        scores = boundary @ gate.candidates.T + gate.internal_scores[None, :]
        gate_logits.append(beta_values[:, None] * scores)

    forward = [gate_logits[0]]
    for index, logits in enumerate(gate_logits[1:]):
        link = plan.links[index]
        compressed = (
            _compress_logits(forward[-1], link.left_state_codes, link.left_state_count)
            if link.left_group_order is None
            else _compress_logits_grouped(forward[-1], link.left_group_order, link.left_state_count)
        )
        transition = beta_values[:, None, None] * link.separator_scores[None, :, :]
        right_message = torch.logsumexp(compressed[:, :, None] + transition, dim=1)
        forward.append(logits + right_message[:, link.right_state_codes])

    choices: list[torch.Tensor] = [torch.empty(0, device=values.device, dtype=torch.long)] * len(
        plan.gates
    )
    choices[-1] = _draw_logits(
        forward[-1],
        generator=generator,
        rng_recorder=rng_recorder,
    )
    for index in range(len(plan.gates) - 2, -1, -1):
        link = plan.links[index]
        right_codes = link.right_state_codes[choices[index + 1]]
        separator_scores = link.separator_scores[:, right_codes].T
        cross_by_state = separator_scores[:, link.left_state_codes]
        logits = forward[index] + beta_values[:, None] * cross_by_state
        choices[index] = _draw_logits(
            logits,
            generator=generator,
            rng_recorder=rng_recorder,
        )

    updated = torch.empty_like(values)
    for gate, choice in zip(plan.gates, choices, strict=True):
        updated[:, gate.positions] = gate.candidates[choice]
    old_energy = -torch.sum(values * boundary_fields, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r", values, plan.internal_coupling, values
    )
    new_energy = -torch.sum(updated * boundary_fields, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r", updated, plan.internal_coupling, updated
    )
    return updated.to(torch.int8), new_energy - old_energy
