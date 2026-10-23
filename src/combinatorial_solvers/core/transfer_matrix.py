"""Exact batched heat-bath updates for low-treewidth chains of spin groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .rng import RngAuditRecorder


@dataclass(frozen=True)
class TransferMatrixGatePlan:
    """Enumerated states and internal scores for one bounded gate bag."""

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
class TransferMatrixChainPlan:
    """Prepared exact conditional for a chain of bounded gate bags."""

    nodes: torch.Tensor
    internal_coupling: torch.Tensor
    gates: tuple[TransferMatrixGatePlan, ...]
    cross_scores: tuple[torch.Tensor, ...]
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
        transitions = sum(int(scores.numel()) for scores in self.cross_scores)
        return unary + transitions


def _normalized_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(int(node) for node in group) for group in groups if group)
    if not normalized:
        raise ValueError("A transfer-matrix chain requires at least one non-empty group")
    if any(len(group) != len(set(group)) for group in normalized):
        raise ValueError("A transfer-matrix group cannot repeat a node")
    flattened = [node for group in normalized for node in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Transfer-matrix groups must be disjoint")
    return normalized


def transfer_matrix_treewidth_upper_bound(
    coupling: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> int:
    """Validate a single-edge-linked bag chain and return its treewidth bound."""

    normalized = _normalized_groups(groups)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be square")
    flattened = [node for group in normalized for node in group]
    if any(node < 0 or node >= coupling.shape[0] for node in flattened):
        raise IndexError("A transfer-matrix group contains an out-of-range node")

    model_device = coupling.device
    tensors = [torch.tensor(group, dtype=torch.long, device=model_device) for group in normalized]
    has_cross_edge = False
    for left_index, left in enumerate(tensors):
        for right_index, right in enumerate(tensors[left_index + 1 :], left_index + 1):
            cross = coupling.index_select(0, left).index_select(1, right)
            nonzero = int(torch.count_nonzero(cross).item())
            if not nonzero:
                continue
            has_cross_edge = True
            if right_index != left_index + 1:
                raise ValueError("Transfer-matrix groups have a non-nearest cross coupling")
            if nonzero > 1:
                raise ValueError("Adjacent transfer-matrix groups must share at most one edge")

    bag_width = max(len(group) - 1 for group in normalized)
    return max(bag_width, 1 if has_cross_edge else 0)


def prepare_transfer_matrix_chain(
    coupling: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    maximum_treewidth: int,
) -> TransferMatrixChainPlan:
    """Prepare exact transfer factors for one conditioned group chain."""

    if maximum_treewidth < 0:
        raise ValueError("maximum_treewidth must be nonnegative")
    normalized = _normalized_groups(groups)
    width = transfer_matrix_treewidth_upper_bound(coupling, normalized)
    if width > maximum_treewidth:
        raise ValueError(
            f"Transfer-matrix treewidth {width} exceeds maximum_treewidth={maximum_treewidth}"
        )

    model_device = coupling.device
    flat_nodes = tuple(node for group in normalized for node in group)
    nodes_model = torch.tensor(flat_nodes, dtype=torch.long, device=model_device)
    internal = coupling.index_select(0, nodes_model).index_select(1, nodes_model)
    internal = internal.to(device=device, dtype=dtype)
    nodes = nodes_model.to(device=device)

    gates: list[TransferMatrixGatePlan] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    for group in normalized:
        stop = start + len(group)
        positions = torch.arange(start, stop, dtype=torch.long, device=device)
        codes = torch.arange(1 << len(group), dtype=torch.int64, device=device)
        candidates = (
            ((codes[:, None] >> torch.arange(len(group), device=device)) & 1)
            .mul(2)
            .sub(1)
            .to(dtype)
        )
        local = internal[start:stop, start:stop]
        internal_scores = 0.5 * torch.einsum(
            "qi,ij,qj->q",
            candidates,
            local,
            candidates,
        )
        gates.append(
            TransferMatrixGatePlan(
                positions=positions,
                candidates=candidates,
                internal_scores=internal_scores,
            )
        )
        offsets.append((start, stop))
        start = stop

    cross_scores: list[torch.Tensor] = []
    for gate_index in range(len(gates) - 1):
        left_start, left_stop = offsets[gate_index]
        right_start, right_stop = offsets[gate_index + 1]
        cross = internal[left_start:left_stop, right_start:right_stop]
        cross_scores.append(
            gates[gate_index].candidates @ cross @ gates[gate_index + 1].candidates.T
        )

    return TransferMatrixChainPlan(
        nodes=nodes,
        internal_coupling=internal,
        gates=tuple(gates),
        cross_scores=tuple(cross_scores),
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
    random_values = torch.rand(
        (logits.shape[0], 1),
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("treewidth_corridor_uniform", logits.shape[0])
    return torch.sum(random_values > torch.cumsum(probabilities, dim=-1), dim=-1).clamp_max(
        logits.shape[-1] - 1
    )


def sample_transfer_matrix_chain(
    plan: TransferMatrixChainPlan,
    selected: torch.Tensor,
    fields: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one exact conditional chain state and return its target-energy delta."""

    if selected.ndim != 2 or selected.shape != fields.shape:
        raise ValueError("selected and fields must share shape [replicas, chain_nodes]")
    if selected.shape[1] != plan.node_count:
        raise ValueError("selected chain width does not match the transfer plan")
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
        score = boundary @ gate.candidates.T + gate.internal_scores[None, :]
        gate_logits.append(beta_values[:, None] * score)

    forward = [gate_logits[0]]
    for gate_index, logits in enumerate(gate_logits[1:]):
        transition = beta_values[:, None, None] * plan.cross_scores[gate_index][None, :, :]
        forward.append(logits + torch.logsumexp(forward[-1][:, :, None] + transition, dim=1))

    choices: list[torch.Tensor] = [torch.empty(0, device=values.device, dtype=torch.long)] * len(
        plan.gates
    )
    choices[-1] = _draw_logits(
        forward[-1],
        generator=generator,
        rng_recorder=rng_recorder,
    )
    for gate_index in range(len(plan.gates) - 2, -1, -1):
        transition = plan.cross_scores[gate_index][:, choices[gate_index + 1]].T
        logits = forward[gate_index] + beta_values[:, None] * transition
        choices[gate_index] = _draw_logits(
            logits,
            generator=generator,
            rng_recorder=rng_recorder,
        )

    updated = torch.empty_like(values)
    for gate, choice in zip(plan.gates, choices, strict=True):
        updated[:, gate.positions] = gate.candidates[choice]

    old_energy = -torch.sum(values * boundary_fields, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r",
        values,
        plan.internal_coupling,
        values,
    )
    new_energy = -torch.sum(updated * boundary_fields, dim=1) - 0.5 * torch.einsum(
        "ri,ij,rj->r",
        updated,
        plan.internal_coupling,
        updated,
    )
    return updated.to(torch.int8), new_energy - old_energy
