"""Batched colored-Gibbs annealing with auditable update semantics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .coloring import validate_coloring
from .hastings import sequential_auxiliary_hastings_step
from .junction_tree import JunctionTreePlan, sample_junction_tree_conditional
from .macro import (
    InvolutiveMovePlan,
    involutive_codeword_metropolis_step,
    locally_balanced_involutive_metropolis_step,
)
from .model import IsingModel
from .padded_sparse import PaddedSparseIsing
from .rng import CounterRngAuditRecorder, RngAuditRecorder, RngStreamAudit, derive_stream_seed
from .separator_chain import (
    SeparatorChainPlan,
    prepare_separator_chain,
    sample_separator_chain,
)
from .transfer_matrix import (
    TransferMatrixChainPlan,
    prepare_transfer_matrix_chain,
    sample_transfer_matrix_chain,
)

SuccessFunction = Callable[[torch.Tensor], torch.Tensor]
FirstHitCallback = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
FirstHitEventCallback = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None]
PredicateEvaluationCallback = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int],
    None,
]
JunctionTreeEventCallback = Callable[[int, int, torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class AnnealSnapshot:
    """States and additive-Hamiltonian diagnostics at one recorded sweep."""

    sweep: int
    beta: float
    states: torch.Tensor
    target_energies: torch.Tensor
    base_energies: torch.Tensor | None
    component_energies: torch.Tensor | None
    component_weights: torch.Tensor | None
    target_endpoint_mask: torch.Tensor | None


TraceCallback = Callable[[AnnealSnapshot], None]


@dataclass(frozen=True)
class AnnealTracePoint:
    sweep: int
    beta: float
    energy_mean: float
    energy_std: float
    energy_q05: float
    energy_q25: float
    energy_median: float
    energy_q75: float
    energy_q95: float
    best_energy_mean: float
    best_energy_std: float
    success_rate: float | None


@dataclass(frozen=True)
class AnnealResult:
    final_states: torch.Tensor
    best_states: torch.Tensor
    best_energies: torch.Tensor
    first_hit_sweeps: torch.Tensor
    trace: tuple[AnnealTracePoint, ...]
    kernel_seconds: float
    total_sweeps: int
    seed: int
    device: str
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    final_energies: torch.Tensor | None = None
    final_success_mask: torch.Tensor | None = None
    local_gibbs_node_updates: int = 0
    beta_stage_sweeps: tuple[int, ...] = ()
    field_backend: str = "dense"
    gibbs_kernel: str = "eager_torch"
    rng_protocol: str = "pytorch_generator"
    fused_gibbs_calls: int = 0
    energy_tracking: str = "recompute"
    backend_storage_bytes: int | None = None
    backend_prepare_seconds: float = 0.0
    metropolis_block_count: int = 0
    metropolis_block_color_count: int = 0
    metropolis_block_every: int = 0
    metropolis_block_rule: str = "metropolis"
    exact_block_count: int = 0
    exact_block_color_count: int = 0
    exact_block_max_size: int = 0
    exact_block_every: int = 0
    exact_block_color_mode: str = "cycle"
    exact_block_update_events: int = 0
    exact_block_candidate_evaluations: int = 0
    exact_block_plan_group_count: int = 0
    exact_block_field_evaluation_calls: int = 0
    exact_block_field_node_evaluations: int = 0
    exact_block_workspace_bytes: int = 0
    exact_block_rng_stream: str = "sampler"
    junction_tree_node_count: int = 0
    junction_tree_boundary_count: int = 0
    junction_tree_treewidth_upper_bound: int = 0
    junction_tree_conditional_table_states_per_replica: int = 0
    junction_tree_every: int = 0
    junction_tree_start_sweep: int = 1
    junction_tree_stop_sweep: int | None = None
    junction_tree_update_events: int = 0
    junction_tree_random_draws: int = 0
    junction_tree_maximum_forward_table_states_per_replica: int = 0
    junction_tree_replica_chunk_size: int | None = None
    junction_tree_chunk_count_per_event: int = 0
    junction_tree_rng_stream: str | None = None
    junction_tree_plan_count: int = 0
    junction_tree_plan_node_counts: tuple[int, ...] = ()
    junction_tree_plan_boundary_counts: tuple[int, ...] = ()
    junction_tree_plan_treewidth_upper_bounds: tuple[int, ...] = ()
    junction_tree_plan_conditional_table_states_per_replica: tuple[int, ...] = ()
    junction_tree_plan_update_events: tuple[int, ...] = ()
    junction_tree_explicit_event_sweeps: tuple[int, ...] = ()
    junction_tree_explicit_event_plan_indices: tuple[int, ...] = ()
    junction_tree_explicit_event_replica_ranges: tuple[tuple[int, int], ...] = ()
    junction_tree_beta_override: float | None = None
    junction_tree_explicit_event_beta_overrides: tuple[float, ...] = ()
    treewidth_corridor_count: int = 0
    treewidth_corridor_color_count: int = 0
    treewidth_corridor_max_nodes: int = 0
    treewidth_corridor_max_treewidth: int = 0
    treewidth_corridor_every: int = 0
    treewidth_corridor_color_mode: str = "cycle"
    treewidth_corridor_message_mode: str = "dense_transition"
    treewidth_corridor_update_events: int = 0
    treewidth_corridor_gate_update_events: int = 0
    treewidth_corridor_candidate_evaluations: int = 0
    treewidth_corridor_field_evaluation_calls: int = 0
    treewidth_corridor_field_node_evaluations: int = 0
    hastings_block_count: int = 0
    hastings_every: int = 0
    hastings_block_mode: str = "cycle"
    hastings_proposal_logit_clip: float | None = None
    hastings_update_events: int = 0
    hastings_proposals: int = 0
    hastings_accepts: int = 0
    hastings_proposed_changes: int = 0
    hastings_accepted_changes: int = 0
    codeword_macro_move_count: int = 0
    codeword_macro_move_names: tuple[str, ...] = ()
    codeword_macro_plan_kind: str | None = None
    codeword_macro_support: str | None = None
    codeword_macro_rule: str = "uniform"
    codeword_macro_uniform_mixture: float = 0.0
    codeword_macro_every: int = 0
    codeword_macro_start_sweep: int = 1
    codeword_macro_stop_sweep: int | None = None
    codeword_macro_update_events: int = 0
    codeword_macro_proposals: int = 0
    codeword_macro_accepts: int = 0
    codeword_macro_candidate_evaluations: int = 0
    codeword_macro_downhill_proposals: int = 0
    codeword_macro_downhill_accepts: int = 0
    codeword_macro_selected_move_counts: tuple[int, ...] = ()
    codeword_macro_accepted_move_counts: tuple[int, ...] = ()
    codeword_macro_storage_bytes: int = 0
    local_field_proposal_clip: float | None = None
    gibbs_uniform_dtype: str = "coefficient"
    local_field_clip_proposals: int = 0
    local_field_clip_accepts: int = 0
    local_field_clip_proposed_flips: int = 0
    local_field_clip_accepted_flips: int = 0
    gibbs_sweep_every: int = 1
    restart_every_sweeps: int = 0
    restart_count: int = 0
    parallel_tempering_beta_ladder: tuple[float, ...] = ()
    parallel_tempering_exchange_every: int = 0
    parallel_tempering_exchange_attempts: tuple[int, ...] = ()
    parallel_tempering_exchange_accepts: tuple[int, ...] = ()
    parallel_tempering_walkers_visited_both_endpoints: int = 0
    hamiltonian_component_name: str | None = None
    hamiltonian_target_weight: float | None = None
    hamiltonian_weight_schedule: tuple[float, ...] = ()
    hamiltonian_tempering_weights: tuple[float, ...] = ()
    hamiltonian_tempering_target_slot: int | None = None
    hamiltonian_success_policy: str = "target_weight_only"
    hamiltonian_tempering_exchange_every: int = 0
    hamiltonian_tempering_exchange_attempts: tuple[int, ...] = ()
    hamiltonian_tempering_exchange_accepts: tuple[int, ...] = ()
    hamiltonian_tempering_walkers_visited_both_endpoints: int = 0
    first_base_zero_sweeps: torch.Tensor | None = None
    first_component_zero_sweeps: torch.Tensor | None = None
    first_component_nonzero_sweeps: torch.Tensor | None = None
    first_component_rezero_sweeps: torch.Tensor | None = None
    first_joint_zero_sweeps: torch.Tensor | None = None
    independent_units: int | None = None
    dependency_unit: str = "independent_chain"
    rng_audits: tuple[RngStreamAudit, ...] = ()


@dataclass(frozen=True)
class _MetropolisBlockColor:
    nodes: torch.Tensor
    node_to_block: torch.Tensor
    internal_coupling: torch.Tensor
    block_count: int


@dataclass(frozen=True)
class _ExactBlockGroup:
    field_positions: torch.Tensor
    internal_coupling: torch.Tensor
    candidates: torch.Tensor
    candidate_internal_energies: torch.Tensor
    block_count: int
    block_size: int


@dataclass(frozen=True)
class _ExactBlockColor:
    nodes: torch.Tensor
    groups: tuple[_ExactBlockGroup, ...]
    block_count: int


@dataclass(frozen=True)
class _TreewidthCorridor:
    field_positions: torch.Tensor
    plan: TransferMatrixChainPlan | SeparatorChainPlan


@dataclass(frozen=True)
class _TreewidthCorridorColor:
    nodes: torch.Tensor
    corridors: tuple[_TreewidthCorridor, ...]


def linear_beta_schedule(start: float, stop: float, steps: int) -> tuple[float, ...]:
    if start <= 0 or stop <= 0 or steps <= 0:
        raise ValueError("Beta endpoints and step count must be positive")
    if steps == 1:
        return (float(stop),)
    return tuple(float(value) for value in torch.linspace(start, stop, steps).tolist())


def _random_spins(
    replicas: int,
    size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    bits = torch.randint(0, 2, (replicas, size), device=device, generator=generator)
    return bits.mul(2).sub(1).to(torch.int8)


def _walker_order(
    states: torch.Tensor,
    energies: torch.Tensor,
    walker_at_slot: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state and energy tensors indexed by persistent walker identity."""

    if walker_at_slot is None:
        return states, energies
    ordered_states = torch.empty_like(states)
    ordered_energies = torch.empty_like(energies)
    ordered_states[walker_at_slot] = states
    ordered_energies[walker_at_slot] = energies
    return ordered_states, ordered_energies


def _hamiltonian_exchange_log_acceptance(
    beta: float,
    left_target_energy: torch.Tensor,
    right_target_energy: torch.Tensor,
    left_component_energy: torch.Tensor,
    right_component_energy: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    target_weight: float,
) -> torch.Tensor:
    """Return exact adjacent-swap log acceptance for an additive Hamiltonian family."""

    current = (
        left_target_energy
        + (left_weight - target_weight) * left_component_energy
        + right_target_energy
        + (right_weight - target_weight) * right_component_energy
    )
    crossed = (
        right_target_energy
        + (left_weight - target_weight) * right_component_energy
        + left_target_energy
        + (right_weight - target_weight) * left_component_energy
    )
    return -float(beta) * (crossed - current)


def _exact_block_candidate_energies(
    selected: torch.Tensor,
    fields: torch.Tensor,
    internal_coupling: torch.Tensor,
    candidates: torch.Tensor,
    candidate_internal_energies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return boundary fields and conditional energies for enumerated block states."""

    internal_fields = torch.einsum("rbk,bkl->rbl", selected, internal_coupling)
    boundary_fields = fields - internal_fields
    energies = candidate_internal_energies[None, :, :] - torch.einsum(
        "qk,rbk->rbq",
        candidates,
        boundary_fields,
    )
    return boundary_fields, energies


def _trace_point(
    sweep: int,
    beta: float,
    energies: torch.Tensor,
    best_energies: torch.Tensor,
    success: torch.Tensor | None,
) -> AnnealTracePoint:
    current = energies.float()
    best = best_energies.float()
    quantiles = torch.quantile(
        current,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=current.device),
    )
    return AnnealTracePoint(
        sweep=sweep,
        beta=float(beta),
        energy_mean=float(current.mean().item()),
        energy_std=float(current.std(unbiased=False).item()),
        energy_q05=float(quantiles[0].item()),
        energy_q25=float(quantiles[1].item()),
        energy_median=float(quantiles[2].item()),
        energy_q75=float(quantiles[3].item()),
        energy_q95=float(quantiles[4].item()),
        best_energy_mean=float(best.mean().item()),
        best_energy_std=float(best.std(unbiased=False).item()),
        success_rate=None if success is None else float(success.float().mean().item()),
    )


def _prepare_metropolis_blocks(
    model: IsingModel,
    blocks: Sequence[Sequence[int]],
    block_colors: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_MetropolisBlockColor, ...]:
    normalized = tuple(tuple(int(index) for index in block) for block in blocks)
    if not normalized or any(not block for block in normalized):
        raise ValueError("metropolis_blocks must contain non-empty blocks")
    flattened = [index for block in normalized for index in block]
    if any(len(block) != len(set(block)) for block in normalized):
        raise ValueError("A Metropolis block cannot repeat a node")
    if any(index < 0 or index >= model.size for index in flattened):
        raise IndexError("metropolis_blocks contain an out-of-range node")
    normalized_colors = tuple(tuple(int(index) for index in color) for color in block_colors)
    color_entries = [index for color in normalized_colors for index in color]
    if sorted(color_entries) != list(range(len(normalized))):
        raise ValueError("metropolis_block_colors must partition every block exactly once")

    plans: list[_MetropolisBlockColor] = []
    model_device = model.coupling.device
    for color in normalized_colors:
        if not color:
            raise ValueError("metropolis_block_colors cannot contain empty colors")
        for left_position, left_block in enumerate(color):
            left_nodes = torch.tensor(normalized[left_block], dtype=torch.long, device=model_device)
            for right_block in color[left_position + 1 :]:
                right_nodes = torch.tensor(
                    normalized[right_block], dtype=torch.long, device=model_device
                )
                if set(normalized[left_block]).intersection(normalized[right_block]):
                    raise ValueError("Metropolis blocks in one color must be disjoint")
                cross = model.coupling.index_select(0, left_nodes).index_select(1, right_nodes)
                if bool(torch.any(cross != 0)):
                    raise ValueError("Blocks in one Metropolis color must have no cross coupling")
        nodes: list[int] = []
        node_to_block: list[int] = []
        for local_block, global_block in enumerate(color):
            nodes.extend(normalized[global_block])
            node_to_block.extend([local_block] * len(normalized[global_block]))
        node_tensor_model = torch.tensor(nodes, dtype=torch.long, device=model_device)
        internal = model.coupling.index_select(0, node_tensor_model).index_select(
            1, node_tensor_model
        )
        plans.append(
            _MetropolisBlockColor(
                nodes=node_tensor_model.to(device=device),
                node_to_block=torch.tensor(
                    node_to_block,
                    dtype=torch.long,
                    device=device,
                ),
                internal_coupling=internal.to(device=device, dtype=dtype),
                block_count=len(color),
            )
        )
    return tuple(plans)


def _prepare_exact_blocks(
    model: IsingModel,
    blocks: Sequence[Sequence[int]],
    block_colors: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
    *,
    replicas: int,
    maximum_block_size: int,
    workspace_bytes: int,
    plan_block_limit: int = 32,
) -> tuple[_ExactBlockColor, ...]:
    """Prepare exact conditional block-Gibbs plans grouped by color and size."""

    normalized = tuple(tuple(int(index) for index in block) for block in blocks)
    if not normalized or any(not block for block in normalized):
        raise ValueError("exact_blocks must contain non-empty blocks")
    if replicas <= 0 or maximum_block_size <= 0 or plan_block_limit <= 0:
        raise ValueError("Replica count, exact block size, and plan limit must be positive")
    if workspace_bytes <= 0:
        raise ValueError("Exact block workspace must be positive")
    if any(len(block) != len(set(block)) for block in normalized):
        raise ValueError("An exact block cannot repeat a node")
    if any(len(block) > maximum_block_size for block in normalized):
        raise ValueError("An exact block exceeds exact_block_max_size")
    flattened = [index for block in normalized for index in block]
    if any(index < 0 or index >= model.size for index in flattened):
        raise IndexError("exact_blocks contain an out-of-range node")
    normalized_colors = tuple(tuple(int(index) for index in color) for color in block_colors)
    color_entries = [index for color in normalized_colors for index in color]
    if sorted(color_entries) != list(range(len(normalized))):
        raise ValueError("exact_block_colors must partition every block exactly once")

    result: list[_ExactBlockColor] = []
    model_device = model.coupling.device
    element_size = torch.empty((), dtype=dtype).element_size()
    for color in normalized_colors:
        if not color:
            raise ValueError("exact_block_colors cannot contain empty colors")
        for left_position, left_block in enumerate(color):
            left_values = normalized[left_block]
            left_nodes = torch.tensor(left_values, dtype=torch.long, device=model_device)
            for right_block in color[left_position + 1 :]:
                right_values = normalized[right_block]
                if set(left_values).intersection(right_values):
                    raise ValueError("Exact blocks in one color must be disjoint")
                right_nodes = torch.tensor(
                    right_values,
                    dtype=torch.long,
                    device=model_device,
                )
                cross = model.coupling.index_select(0, left_nodes).index_select(1, right_nodes)
                if bool(torch.any(cross != 0)):
                    raise ValueError("Exact blocks in one color must have no cross coupling")

        color_nodes_values = tuple(node for block in color for node in normalized[block])
        color_nodes = torch.tensor(color_nodes_values, dtype=torch.long, device=device)
        field_position = {node: position for position, node in enumerate(color_nodes_values)}
        groups: list[_ExactBlockGroup] = []
        for size in sorted({len(normalized[index]) for index in color}):
            sized = [index for index in color if len(normalized[index]) == size]
            codes = torch.arange(1 << size, dtype=torch.int64, device=device)
            candidates = (
                ((codes[:, None] >> torch.arange(size, dtype=torch.int64, device=device)) & 1)
                .mul(2)
                .sub(1)
                .to(dtype)
            )
            candidate_count = 1 << size
            estimated_bytes_per_block = 4 * replicas * candidate_count * element_size
            workspace_block_limit = max(1, workspace_bytes // estimated_bytes_per_block)
            block_limit = min(plan_block_limit, workspace_block_limit)
            for start in range(0, len(sized), block_limit):
                chunk = sized[start : start + block_limit]
                field_positions = torch.tensor(
                    [[field_position[node] for node in normalized[index]] for index in chunk],
                    dtype=torch.long,
                    device=device,
                )
                node_matrix_model = torch.tensor(
                    [normalized[index] for index in chunk],
                    dtype=torch.long,
                    device=model_device,
                )
                flat_model = node_matrix_model.reshape(-1)
                internal = (
                    model.coupling.index_select(0, flat_model)
                    .index_select(1, flat_model)
                    .reshape(len(chunk), size, len(chunk), size)
                )
                block_index = torch.arange(len(chunk), device=model_device)
                internal = internal[block_index, :, block_index, :]
                internal = internal.to(device=device, dtype=dtype)
                candidate_internal = -0.5 * torch.einsum(
                    "qk,bkl,ql->bq",
                    candidates,
                    internal,
                    candidates,
                )
                groups.append(
                    _ExactBlockGroup(
                        field_positions=field_positions,
                        internal_coupling=internal,
                        candidates=candidates,
                        candidate_internal_energies=candidate_internal,
                        block_count=len(chunk),
                        block_size=size,
                    )
                )
        result.append(
            _ExactBlockColor(
                nodes=color_nodes,
                groups=tuple(groups),
                block_count=len(color),
            )
        )
    return tuple(result)


def _prepare_treewidth_corridors(
    model: IsingModel,
    corridors: Sequence[Sequence[Sequence[int]]],
    corridor_colors: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
    *,
    maximum_treewidth: int,
    message_mode: str,
) -> tuple[_TreewidthCorridorColor, ...]:
    """Prepare conflict-free transfer-matrix corridors grouped by color."""

    normalized = tuple(
        tuple(tuple(int(node) for node in gate) for gate in corridor if gate)
        for corridor in corridors
    )
    if not normalized or any(not corridor for corridor in normalized):
        raise ValueError("treewidth_corridors must contain non-empty gate chains")
    normalized_colors = tuple(tuple(int(index) for index in color) for color in corridor_colors)
    color_entries = [index for color in normalized_colors for index in color]
    if sorted(color_entries) != list(range(len(normalized))):
        raise ValueError("treewidth_corridor_colors must partition every corridor exactly once")

    model_device = model.coupling.device
    if message_mode == "dense_transition":
        prepared: tuple[TransferMatrixChainPlan | SeparatorChainPlan, ...] = tuple(
            prepare_transfer_matrix_chain(
                model.coupling,
                corridor,
                device=device,
                dtype=dtype,
                maximum_treewidth=maximum_treewidth,
            )
            for corridor in normalized
        )
    elif message_mode == "separator_compressed":
        prepared = tuple(
            prepare_separator_chain(
                model.coupling,
                corridor,
                device=device,
                dtype=dtype,
                maximum_treewidth=maximum_treewidth,
            )
            for corridor in normalized
        )
    else:
        raise ValueError(
            "treewidth_corridor_message_mode must be 'dense_transition' or 'separator_compressed'"
        )
    result: list[_TreewidthCorridorColor] = []
    for color in normalized_colors:
        if not color:
            raise ValueError("treewidth_corridor_colors cannot contain empty colors")
        for left_position, left_index in enumerate(color):
            left = prepared[left_index].nodes.to(model_device)
            for right_index in color[left_position + 1 :]:
                right = prepared[right_index].nodes.to(model_device)
                if set(left.tolist()).intersection(right.tolist()):
                    raise ValueError("Treewidth corridors in one color must be disjoint")
                cross = model.coupling.index_select(0, left).index_select(1, right)
                if bool(torch.any(cross != 0)):
                    raise ValueError("Treewidth corridors in one color must have no cross coupling")

        node_values = tuple(
            int(node)
            for corridor_index in color
            for node in prepared[corridor_index].nodes.tolist()
        )
        nodes = torch.tensor(node_values, dtype=torch.long, device=device)
        position = {node: index for index, node in enumerate(node_values)}
        color_corridors = tuple(
            _TreewidthCorridor(
                field_positions=torch.tensor(
                    [position[int(node)] for node in prepared[corridor_index].nodes.tolist()],
                    dtype=torch.long,
                    device=device,
                ),
                plan=prepared[corridor_index],
            )
            for corridor_index in color
        )
        result.append(_TreewidthCorridorColor(nodes=nodes, corridors=color_corridors))
    return tuple(result)


@torch.inference_mode()
def run_colored_gibbs(
    model: IsingModel,
    colors: Sequence[Sequence[int]],
    betas: Sequence[float],
    sweeps_per_beta: int,
    replicas: int,
    seed: int,
    *,
    beta_stage_sweeps: Sequence[int] | None = None,
    initial_states: torch.Tensor | None = None,
    success: SuccessFunction | None = None,
    record_every: int = 1,
    record_sweeps: Sequence[int] | None = None,
    randomize_color_order: bool = False,
    stop_when_all_succeeded: bool = False,
    success_energy_upper_bound: float | None = None,
    on_first_hit: FirstHitCallback | None = None,
    on_first_hit_event: FirstHitEventCallback | None = None,
    on_predicate_evaluation: PredicateEvaluationCallback | None = None,
    on_junction_tree_event: JunctionTreeEventCallback | None = None,
    field_backend: str = "dense",
    gibbs_kernel: str = "eager_torch",
    energy_tracking: str = "recompute",
    padded_model: PaddedSparseIsing | None = None,
    metropolis_blocks: Sequence[Sequence[int]] | None = None,
    metropolis_block_colors: Sequence[Sequence[int]] | None = None,
    metropolis_block_every: int = 0,
    metropolis_block_rule: str = "metropolis",
    exact_blocks: Sequence[Sequence[int]] | None = None,
    exact_block_colors: Sequence[Sequence[int]] | None = None,
    exact_block_every: int = 0,
    exact_block_max_size: int = 8,
    exact_block_color_mode: str = "cycle",
    exact_block_workspace_mib: float = 512.0,
    exact_block_rng_seed: int | None = None,
    junction_tree_plan: JunctionTreePlan | None = None,
    junction_tree_plans: Sequence[JunctionTreePlan] | None = None,
    junction_tree_every: int = 0,
    junction_tree_start_sweep: int = 1,
    junction_tree_stop_sweep: int | None = None,
    junction_tree_rng_seed: int | None = None,
    junction_tree_replica_chunk_size: int | None = None,
    junction_tree_event_sweeps: Sequence[int] | None = None,
    junction_tree_event_plan_indices: Sequence[int] | None = None,
    junction_tree_event_replica_ranges: Sequence[tuple[int, int]] | None = None,
    junction_tree_beta_override: float | None = None,
    junction_tree_event_beta_overrides: Sequence[float] | None = None,
    treewidth_corridors: Sequence[Sequence[Sequence[int]]] | None = None,
    treewidth_corridor_colors: Sequence[Sequence[int]] | None = None,
    treewidth_corridor_every: int = 0,
    treewidth_corridor_max_treewidth: int = 4,
    treewidth_corridor_color_mode: str = "cycle",
    treewidth_corridor_message_mode: str = "dense_transition",
    hastings_blocks: Sequence[Sequence[int]] | None = None,
    hastings_proposal_model: IsingModel | None = None,
    hastings_every: int = 0,
    hastings_block_mode: str = "cycle",
    hastings_proposal_logit_clip: float = 12.0,
    codeword_macro_plan: InvolutiveMovePlan | None = None,
    codeword_macro_every: int = 0,
    codeword_macro_rule: str = "uniform",
    codeword_macro_uniform_mixture: float = 0.0,
    codeword_macro_start_sweep: int = 1,
    codeword_macro_stop_sweep: int | None = None,
    local_field_proposal_clip: float | None = None,
    gibbs_uniform_dtype: torch.dtype | None = None,
    gibbs_sweep_every: int = 1,
    restart_every_sweeps: int = 0,
    restart_states: torch.Tensor | None = None,
    parallel_tempering_betas: Sequence[float] | None = None,
    parallel_tempering_exchange_every: int = 0,
    hamiltonian_component: IsingModel | None = None,
    hamiltonian_component_padded: PaddedSparseIsing | None = None,
    hamiltonian_component_name: str = "component",
    hamiltonian_target_weight: float = 1.0,
    hamiltonian_weight_schedule: Sequence[float] | None = None,
    hamiltonian_tempering_weights: Sequence[float] | None = None,
    hamiltonian_tempering_exchange_every: int = 0,
    hamiltonian_success_policy: str = "target_weight_only",
    component_zero_tolerance: float = 1e-5,
    on_trace: TraceCallback | None = None,
    external_rng_audits: Sequence[RngStreamAudit] = (),
    counter_rng_start_offset: int = 0,
) -> AnnealResult:
    """Run color-block Gibbs and optional block-Metropolis replica chains.

    ``success`` is evaluated at initialization and after every sweep, so first-hit sweep
    zero is retained. When ``success_energy_upper_bound`` is set, both the certificate
    predicate and the energy threshold must pass. ``on_predicate_evaluation`` observes
    the raw predicate and target energy before that threshold is applied; callbacks must
    treat their tensor arguments as read-only and must not draw from sampler RNG streams.
    Trace statistics can be stored more sparsely with ``record_every`` or at explicit
    ``record_sweeps`` checkpoints.
    ``field_backend="padded_sparse"`` avoids dense matrix
    products for bounded-degree graphs. ``field_backend="triton_sparse"`` uses the same
    padded storage with a fused CUDA local-field kernel.
    ``energy_tracking="incremental"`` applies the exact color-block energy delta; this
    relies on the validated absence of edges within every color block.

    ``gibbs_sweep_every=0`` disables site-wise Gibbs updates. This is useful for
    auditable non-local optimization kernels when the Metropolis blocks partition
    the state, but it no longer models local p-bit dynamics. ``exact_blocks`` enumerate
    every state of each bounded block and draw from its exact conditional distribution.
    Conflict-free blocks in one color update simultaneously; overlapping colors are
    scanned deterministically or all at once. ``treewidth_corridors`` instead updates
    arbitrarily long chains of bounded gate bags by exact transfer-matrix forward-filtering
    and backward-sampling. Its exponential cost depends on the audited treewidth rather
    than the total corridor node count. ``hastings_blocks`` use a sequential auxiliary
    Gibbs scan with explicit forward/reverse proposal probabilities and a Metropolis-
    Hastings correction to preserve the original fixed-beta target. A
    ``codeword_macro_plan`` selects a self-inverse transport. ``uniform`` applies the
    ordinary exact Metropolis correction; the locally balanced rules enumerate the
    complete labelled neighborhood and apply a full forward/reverse Hastings correction.

    ``restart_every_sweeps`` redraws independent states at fixed sweep boundaries
    while retaining first-hit and best-state history. ``restart_states`` may supply an
    auditable external initialization at every boundary; otherwise the sampler stream
    draws independent spins. ``parallel_tempering_betas``
    partitions replicas into independent ladders, tracks persistent walker identity
    across adjacent exchanges, and reports the ladder rather than each slot as the
    independent unit. Restarts and replica exchange are deliberately mutually
    exclusive in this implementation.

    An optional additive component defines
    ``H(w) = H(target) + (w - target_weight) * H_component``. A common weight
    schedule gives non-homogeneous optimization. A Hamiltonian-tempering ladder
    instead runs physical updates at distinct weights and performs exact adjacent
    replica exchange. The default success policy admits only the target-weight slot.
    Optimization problems whose positive-weight Hamiltonians have a proven shared ground
    set may explicitly use ``shared_ground_state`` to accept a certified solution at any
    weight while still evaluating its energy under the target Hamiltonian.

    ``counter_rng_start_offset`` is an exact logical-spin offset for the versioned
    counter-based Triton kernels. Supplying the final offset and final states from a
    previous invocation resumes the same trajectory without reusing random values.
    """

    if replicas <= 0 or sweeps_per_beta <= 0 or record_every <= 0:
        raise ValueError("replicas, sweeps_per_beta, and record_every must be positive")
    if not betas or any(not math.isfinite(float(beta)) or float(beta) <= 0 for beta in betas):
        raise ValueError("betas must be a non-empty sequence of finite positive values")
    if success_energy_upper_bound is not None and not math.isfinite(
        float(success_energy_upper_bound)
    ):
        raise ValueError("success_energy_upper_bound must be finite when provided")
    if success_energy_upper_bound is not None and success is None:
        raise ValueError("success_energy_upper_bound requires a success function")
    if on_first_hit is not None and success is None:
        raise ValueError("on_first_hit requires a success function")
    if on_first_hit_event is not None and success is None:
        raise ValueError("on_first_hit_event requires a success function")
    if on_predicate_evaluation is not None and success is None:
        raise ValueError("on_predicate_evaluation requires a success function")
    if hamiltonian_success_policy not in {"target_weight_only", "shared_ground_state"}:
        raise ValueError(
            "hamiltonian_success_policy must be 'target_weight_only' or 'shared_ground_state'"
        )
    if field_backend not in {"dense", "padded_sparse", "triton_sparse"}:
        raise ValueError("field_backend must be 'dense', 'padded_sparse', or 'triton_sparse'")
    if gibbs_kernel not in {
        "eager_torch",
        "triton_external_uniform_bridge",
        "triton_counter_reference",
        "triton_counter_fused",
    }:
        raise ValueError("Unsupported gibbs_kernel")
    if energy_tracking not in {"recompute", "incremental"}:
        raise ValueError("energy_tracking must be 'recompute' or 'incremental'")
    if field_backend == "dense" and padded_model is not None:
        raise ValueError("padded_model is only valid with a sparse field backend")
    if metropolis_block_every < 0:
        raise ValueError("metropolis_block_every must be nonnegative")
    if metropolis_block_rule not in {"metropolis", "heat_bath"}:
        raise ValueError("metropolis_block_rule must be 'metropolis' or 'heat_bath'")
    if exact_block_every < 0:
        raise ValueError("exact_block_every must be nonnegative")
    if exact_block_max_size <= 0:
        raise ValueError("exact_block_max_size must be positive")
    if exact_block_color_mode not in {"cycle", "all"}:
        raise ValueError("exact_block_color_mode must be 'cycle' or 'all'")
    if not math.isfinite(exact_block_workspace_mib) or exact_block_workspace_mib <= 0:
        raise ValueError("exact_block_workspace_mib must be finite and positive")
    if treewidth_corridor_every < 0:
        raise ValueError("treewidth_corridor_every must be nonnegative")
    if treewidth_corridor_max_treewidth < 0:
        raise ValueError("treewidth_corridor_max_treewidth must be nonnegative")
    if treewidth_corridor_color_mode not in {"cycle", "all"}:
        raise ValueError("treewidth_corridor_color_mode must be 'cycle' or 'all'")
    if treewidth_corridor_message_mode not in {"dense_transition", "separator_compressed"}:
        raise ValueError(
            "treewidth_corridor_message_mode must be 'dense_transition' or 'separator_compressed'"
        )
    if hastings_every < 0:
        raise ValueError("hastings_every must be nonnegative")
    if hastings_block_mode not in {"cycle", "all"}:
        raise ValueError("hastings_block_mode must be 'cycle' or 'all'")
    if not math.isfinite(hastings_proposal_logit_clip) or hastings_proposal_logit_clip <= 0:
        raise ValueError("hastings_proposal_logit_clip must be finite and positive")
    if codeword_macro_every < 0:
        raise ValueError("codeword_macro_every must be nonnegative")
    if codeword_macro_rule not in {"uniform", "locally_balanced_barker", "locally_balanced_sqrt"}:
        raise ValueError(
            "codeword_macro_rule must be 'uniform', 'locally_balanced_barker', or "
            "'locally_balanced_sqrt'"
        )
    if not math.isfinite(codeword_macro_uniform_mixture) or not (
        0.0 <= codeword_macro_uniform_mixture < 1.0
    ):
        raise ValueError("codeword_macro_uniform_mixture must be finite and in [0, 1)")
    if codeword_macro_rule == "uniform" and codeword_macro_uniform_mixture != 0.0:
        raise ValueError("Uniform macro rules cannot add a uniform proposal mixture")
    if codeword_macro_start_sweep < 1:
        raise ValueError("codeword_macro_start_sweep must be positive")
    if codeword_macro_stop_sweep is not None and (
        codeword_macro_stop_sweep < codeword_macro_start_sweep
    ):
        raise ValueError("codeword_macro_stop_sweep must not precede the start sweep")
    if local_field_proposal_clip is not None and (
        not math.isfinite(float(local_field_proposal_clip)) or float(local_field_proposal_clip) <= 0
    ):
        raise ValueError("local_field_proposal_clip must be finite and positive")
    if gibbs_uniform_dtype not in {None, torch.float32, torch.float64}:
        raise ValueError("gibbs_uniform_dtype must be torch.float32, torch.float64, or None")
    if counter_rng_start_offset < 0 or counter_rng_start_offset >= 1 << 63:
        raise ValueError("counter_rng_start_offset must fit a nonnegative signed 64-bit integer")
    if counter_rng_start_offset and gibbs_kernel not in {
        "triton_counter_reference",
        "triton_counter_fused",
    }:
        raise ValueError("counter_rng_start_offset requires a counter-based Triton kernel")
    if gibbs_sweep_every < 0:
        raise ValueError("gibbs_sweep_every must be nonnegative")
    if local_field_proposal_clip is not None and gibbs_sweep_every == 0:
        raise ValueError("A clipped local-field proposal requires local Gibbs updates")
    if restart_every_sweeps < 0:
        raise ValueError("restart_every_sweeps must be nonnegative")
    if gibbs_kernel != "eager_torch":
        if field_backend != "triton_sparse":
            raise ValueError(
                "Experimental Triton Gibbs kernels require field_backend='triton_sparse'"
            )
        if energy_tracking != "incremental":
            raise ValueError(
                "Experimental Triton Gibbs kernels require incremental energy tracking"
            )
        if initial_states is None:
            raise ValueError("Experimental Triton Gibbs kernels require explicit initial_states")
        if randomize_color_order or local_field_proposal_clip is not None:
            raise ValueError(
                "Experimental Triton Gibbs kernels require unclipped deterministic colors"
            )
        if gibbs_uniform_dtype is not None:
            raise ValueError("Experimental Triton Gibbs kernels fix the uniform dtype to float32")
    if metropolis_block_every > 0 and (
        metropolis_blocks is None or metropolis_block_colors is None
    ):
        raise ValueError("Metropolis block updates require blocks and block colors")
    if metropolis_block_every == 0 and (
        metropolis_blocks is not None or metropolis_block_colors is not None
    ):
        raise ValueError("Metropolis blocks require a positive metropolis_block_every")
    if exact_block_every > 0 and (exact_blocks is None or exact_block_colors is None):
        raise ValueError("Exact block updates require blocks and block colors")
    if exact_block_every == 0 and (exact_blocks is not None or exact_block_colors is not None):
        raise ValueError("Exact blocks require a positive exact_block_every")
    if exact_block_rng_seed is not None and exact_blocks is None:
        raise ValueError("A separate exact-block RNG stream requires exact blocks")
    if exact_block_rng_seed is not None and not 0 <= exact_block_rng_seed < (1 << 63):
        raise ValueError("exact_block_rng_seed must fit a nonnegative signed 64-bit integer")
    if junction_tree_plan is not None and junction_tree_plans is not None:
        raise ValueError("Specify either junction_tree_plan or junction_tree_plans, not both")
    if junction_tree_plans is not None and not junction_tree_plans:
        raise ValueError("junction_tree_plans must be non-empty when provided")
    junction_tree_plan_cycle = (
        (junction_tree_plan,)
        if junction_tree_plan is not None
        else tuple(junction_tree_plans or ())
    )
    explicit_junction_tree_sweeps = tuple(
        int(value) for value in (junction_tree_event_sweeps or ())
    )
    explicit_junction_tree_plan_indices = tuple(
        int(value) for value in (junction_tree_event_plan_indices or ())
    )
    explicit_junction_tree_replica_ranges = tuple(
        (int(start), int(stop)) for start, stop in (junction_tree_event_replica_ranges or ())
    )
    explicit_junction_tree_beta_overrides = tuple(
        float(value) for value in (junction_tree_event_beta_overrides or ())
    )
    requested_total_sweeps = (
        len(betas) * sweeps_per_beta
        if beta_stage_sweeps is None
        else sum(int(value) for value in beta_stage_sweeps)
    )
    if junction_tree_event_sweeps is not None and not explicit_junction_tree_sweeps:
        raise ValueError("junction_tree_event_sweeps must be non-empty when provided")
    if explicit_junction_tree_sweeps and (
        tuple(sorted(set(explicit_junction_tree_sweeps))) != explicit_junction_tree_sweeps
        or any(
            sweep < 1 or sweep > requested_total_sweeps for sweep in explicit_junction_tree_sweeps
        )
    ):
        raise ValueError("Explicit junction-tree sweeps must be unique, sorted, and in range")
    if junction_tree_event_plan_indices is not None and not explicit_junction_tree_sweeps:
        raise ValueError("Explicit junction-tree plan indices require explicit event sweeps")
    if junction_tree_event_plan_indices is not None and (
        len(explicit_junction_tree_plan_indices) != len(explicit_junction_tree_sweeps)
        or any(
            index < 0 or index >= len(junction_tree_plan_cycle)
            for index in explicit_junction_tree_plan_indices
        )
    ):
        raise ValueError("Explicit junction-tree plan indices must match valid event sweeps")
    if junction_tree_event_replica_ranges is not None and not explicit_junction_tree_sweeps:
        raise ValueError("Explicit junction-tree replica ranges require explicit event sweeps")
    if junction_tree_event_replica_ranges is not None and (
        len(explicit_junction_tree_replica_ranges) != len(explicit_junction_tree_sweeps)
        or any(
            start < 0 or stop <= start or stop > replicas
            for start, stop in explicit_junction_tree_replica_ranges
        )
    ):
        raise ValueError("Explicit junction-tree replica ranges must match events and replicas")
    if explicit_junction_tree_sweeps and not junction_tree_plan_cycle:
        raise ValueError("Explicit junction-tree events require a prepared plan cycle")
    if junction_tree_beta_override is not None and (
        not math.isfinite(float(junction_tree_beta_override))
        or float(junction_tree_beta_override) <= 0
    ):
        raise ValueError("junction_tree_beta_override must be finite and positive")
    if junction_tree_beta_override is not None and not junction_tree_plan_cycle:
        raise ValueError("junction_tree_beta_override requires a prepared plan cycle")
    if junction_tree_event_beta_overrides is not None and not explicit_junction_tree_sweeps:
        raise ValueError("Explicit junction-tree beta overrides require explicit event sweeps")
    if junction_tree_event_beta_overrides is not None and (
        len(explicit_junction_tree_beta_overrides) != len(explicit_junction_tree_sweeps)
        or any(
            not math.isfinite(value) or value <= 0
            for value in explicit_junction_tree_beta_overrides
        )
    ):
        raise ValueError("Explicit junction-tree beta overrides must match events and be positive")
    if junction_tree_beta_override is not None and junction_tree_event_beta_overrides is not None:
        raise ValueError("Specify scalar or per-event junction-tree beta overrides, not both")
    if junction_tree_every < 0:
        raise ValueError("junction_tree_every must be nonnegative")
    if junction_tree_start_sweep < 1:
        raise ValueError("junction_tree_start_sweep must be positive")
    if junction_tree_stop_sweep is not None and (
        junction_tree_stop_sweep < junction_tree_start_sweep
    ):
        raise ValueError("junction_tree_stop_sweep must not precede its start sweep")
    if junction_tree_every > 0 and not junction_tree_plan_cycle:
        raise ValueError("Junction-tree updates require at least one prepared plan")
    if junction_tree_every == 0 and junction_tree_plan_cycle:
        raise ValueError("A junction-tree plan cycle requires a positive update interval")
    if junction_tree_plan_cycle and junction_tree_rng_seed is None:
        raise ValueError("Junction-tree updates require a separate RNG seed")
    if junction_tree_rng_seed is not None and not junction_tree_plan_cycle:
        raise ValueError("A junction-tree RNG seed requires at least one prepared plan")
    if junction_tree_rng_seed is not None and not 0 <= junction_tree_rng_seed < (1 << 63):
        raise ValueError("junction_tree_rng_seed must fit a nonnegative signed 64-bit integer")
    if on_junction_tree_event is not None and not junction_tree_plan_cycle:
        raise ValueError("on_junction_tree_event requires a junction-tree plan cycle")
    if junction_tree_replica_chunk_size is not None and junction_tree_replica_chunk_size <= 0:
        raise ValueError("junction_tree_replica_chunk_size must be positive")
    if junction_tree_replica_chunk_size is not None and not junction_tree_plan_cycle:
        raise ValueError("A junction-tree replica chunk size requires a prepared plan cycle")
    for prepared_junction_tree_plan in junction_tree_plan_cycle:
        partition = (
            *prepared_junction_tree_plan.nodes,
            *prepared_junction_tree_plan.boundary_nodes,
        )
        if len(partition) != model.size or sorted(partition) != list(range(model.size)):
            raise ValueError("Every junction-tree plan must partition the Ising model")
    if treewidth_corridor_every > 0 and (
        treewidth_corridors is None or treewidth_corridor_colors is None
    ):
        raise ValueError("Treewidth corridor updates require corridors and colors")
    if treewidth_corridor_every == 0 and (
        treewidth_corridors is not None or treewidth_corridor_colors is not None
    ):
        raise ValueError("Treewidth corridors require a positive update interval")
    if hastings_every > 0 and (hastings_blocks is None or hastings_proposal_model is None):
        raise ValueError("Hastings repair requires blocks and an auxiliary proposal model")
    if hastings_every == 0 and (hastings_blocks is not None or hastings_proposal_model is not None):
        raise ValueError("Hastings blocks require a positive hastings_every")
    if codeword_macro_every > 0 and codeword_macro_plan is None:
        raise ValueError("Codeword macro updates require a codeword_macro_plan")
    if codeword_macro_every == 0 and codeword_macro_plan is not None:
        raise ValueError("A codeword_macro_plan requires a positive update interval")
    if codeword_macro_plan is not None and codeword_macro_plan.state_size != model.size:
        raise ValueError("The codeword macro plan does not match the Ising model size")
    if gibbs_kernel != "eager_torch" and (
        metropolis_block_every
        or exact_block_every
        or junction_tree_every
        or treewidth_corridor_every
        or hastings_every
        or codeword_macro_every
        or restart_every_sweeps
        or parallel_tempering_betas
        or hamiltonian_component is not None
        or hamiltonian_weight_schedule
        or hamiltonian_tempering_weights
    ):
        raise ValueError("Experimental Triton Gibbs kernels currently support local Gibbs only")
    if (
        gibbs_sweep_every == 0
        and metropolis_block_every == 0
        and exact_block_every == 0
        and junction_tree_every == 0
        and treewidth_corridor_every == 0
        and hastings_every == 0
        and codeword_macro_every == 0
    ):
        raise ValueError(
            "At least one Gibbs, Metropolis, exact block, corridor, Hastings, or macro kernel "
            "is required"
        )
    tempering_ladder = tuple(float(value) for value in (parallel_tempering_betas or ()))
    weight_schedule = tuple(float(value) for value in (hamiltonian_weight_schedule or ()))
    hamiltonian_ladder = tuple(float(value) for value in (hamiltonian_tempering_weights or ()))
    if not math.isfinite(hamiltonian_target_weight) or hamiltonian_target_weight <= 0:
        raise ValueError("hamiltonian_target_weight must be finite and positive")
    if not math.isfinite(component_zero_tolerance) or component_zero_tolerance < 0:
        raise ValueError("component_zero_tolerance must be finite and nonnegative")
    if weight_schedule or hamiltonian_ladder:
        if hamiltonian_component is None:
            raise ValueError("Hamiltonian weight controls require an additive component")
    if hamiltonian_success_policy == "shared_ground_state" and not (
        weight_schedule or hamiltonian_ladder
    ):
        raise ValueError("shared_ground_state success requires Hamiltonian weight controls")
    if hamiltonian_component is not None:
        if hamiltonian_component.names != model.names:
            raise ValueError("Hamiltonian component must share the target model variable order")
        if (weight_schedule or hamiltonian_ladder) and (
            metropolis_block_every
            or exact_block_every
            or junction_tree_every
            or treewidth_corridor_every
            or hastings_every
            or codeword_macro_every
            or restart_every_sweeps
        ):
            raise ValueError(
                "Dynamic Hamiltonian controls do not support blocks, corridors, macros, or restarts"
            )
        if field_backend == "dense" and hamiltonian_component_padded is not None:
            raise ValueError("A padded Hamiltonian component requires a sparse field backend")
    else:
        if hamiltonian_component_padded is not None:
            raise ValueError("A padded component requires hamiltonian_component")
    resolved_beta_stage_sweeps = (
        (sweeps_per_beta,) * len(betas)
        if beta_stage_sweeps is None
        else tuple(int(value) for value in beta_stage_sweeps)
    )
    if len(resolved_beta_stage_sweeps) != len(betas) or any(
        value <= 0 for value in resolved_beta_stage_sweeps
    ):
        raise ValueError("beta_stage_sweeps must contain one positive dwell per beta")
    total_requested_sweeps = sum(resolved_beta_stage_sweeps)
    expected_restart_states = (
        0 if restart_every_sweeps == 0 else (total_requested_sweeps - 1) // restart_every_sweeps
    )
    explicit_record_sweeps = frozenset(int(value) for value in (record_sweeps or ()))
    if any(value <= 0 or value > total_requested_sweeps for value in explicit_record_sweeps):
        raise ValueError("record_sweeps must lie inside the positive requested sweep range")
    if weight_schedule:
        if len(weight_schedule) != total_requested_sweeps:
            raise ValueError("Hamiltonian weight schedule must contain one value per sweep")
        if any(not math.isfinite(value) or value <= 0 for value in weight_schedule):
            raise ValueError("Hamiltonian weight schedule values must be finite and positive")
        if not math.isclose(weight_schedule[-1], hamiltonian_target_weight, abs_tol=1e-12):
            raise ValueError("Hamiltonian weight schedule must end at the target weight")
    if hamiltonian_ladder:
        if len(hamiltonian_ladder) < 2:
            raise ValueError("Hamiltonian tempering requires at least two weights")
        if any(not math.isfinite(value) or value <= 0 for value in hamiltonian_ladder):
            raise ValueError("Hamiltonian-tempering weights must be finite and positive")
        if any(
            left >= right
            for left, right in zip(hamiltonian_ladder, hamiltonian_ladder[1:], strict=False)
        ):
            raise ValueError("Hamiltonian-tempering weights must be strictly increasing")
        target_slots = [
            index
            for index, weight in enumerate(hamiltonian_ladder)
            if math.isclose(weight, hamiltonian_target_weight, abs_tol=1e-12, rel_tol=0.0)
        ]
        if len(target_slots) != 1:
            raise ValueError(
                "Hamiltonian-tempering ladder must contain the target weight exactly once"
            )
        if hamiltonian_tempering_exchange_every <= 0:
            raise ValueError("Hamiltonian tempering requires a positive exchange interval")
        if replicas % len(hamiltonian_ladder):
            raise ValueError("replicas must be divisible by the Hamiltonian ladder size")
    elif hamiltonian_tempering_exchange_every:
        raise ValueError("A Hamiltonian exchange interval requires a weight ladder")
    if weight_schedule and hamiltonian_ladder:
        raise ValueError("Use either a common weight schedule or Hamiltonian tempering")
    if tempering_ladder and hamiltonian_ladder:
        raise ValueError("Temperature and Hamiltonian tempering cannot be combined")
    if tempering_ladder:
        if len(tempering_ladder) < 2:
            raise ValueError("Parallel tempering requires at least two beta values")
        if any(not math.isfinite(value) or value <= 0 for value in tempering_ladder):
            raise ValueError("Parallel-tempering betas must be finite and positive")
        if any(
            left >= right
            for left, right in zip(tempering_ladder, tempering_ladder[1:], strict=False)
        ):
            raise ValueError("Parallel-tempering betas must be strictly increasing")
        if replicas % len(tempering_ladder):
            raise ValueError("replicas must be divisible by the parallel-tempering ladder size")
        if parallel_tempering_exchange_every <= 0:
            raise ValueError("Parallel tempering requires a positive exchange interval")
        if restart_every_sweeps:
            raise ValueError("Restarts and parallel tempering cannot be combined")
    elif parallel_tempering_exchange_every:
        raise ValueError("A PT exchange interval requires parallel_tempering_betas")
    if restart_states is not None and not restart_every_sweeps:
        raise ValueError("restart_states require a positive restart_every_sweeps")
    if restart_states is not None and initial_states is None:
        raise ValueError("External restart states require explicit initial_states")
    if restart_every_sweeps and initial_states is not None and restart_states is None:
        raise ValueError("Explicit initial_states require explicit restart_states when restarting")
    validate_coloring(model, colors)

    hamiltonian_target_slot = (
        next(
            index
            for index, weight in enumerate(hamiltonian_ladder)
            if math.isclose(weight, hamiltonian_target_weight, abs_tol=1e-12, rel_tol=0.0)
        )
        if hamiltonian_ladder
        else None
    )

    sparse: PaddedSparseIsing | None = None
    if field_backend == "dense":
        device = model.coupling.device
        coefficient_dtype = model.coupling.dtype
        local_fields = model.local_fields
        energy = model.energy
        backend_storage_bytes = (
            model.coupling.numel() * model.coupling.element_size()
            + model.bias.numel() * model.bias.element_size()
        )
    else:
        sparse = padded_model or PaddedSparseIsing.from_model(model)
        if sparse.size != model.size:
            raise ValueError("padded_model size does not match the Ising model")
        device = sparse.device
        coefficient_dtype = sparse.dtype
        local_fields = (
            sparse.triton_local_fields if field_backend == "triton_sparse" else sparse.local_fields
        )
        energy = sparse.energy
        backend_storage_bytes = sparse.storage_bytes
    if gibbs_kernel != "eager_torch" and (
        device.type != "cuda" or coefficient_dtype != torch.float32
    ):
        raise ValueError("Experimental Triton Gibbs kernels require CUDA FP32 tensors")

    prepared_restart_states = None
    if restart_states is not None:
        if restart_states.shape != (expected_restart_states, replicas, model.size):
            raise ValueError(
                "restart_states must have shape [restart_boundaries, replicas, model.size]"
            )
        source_restart_states = restart_states.to(device=device)
        if torch.any((source_restart_states != -1) & (source_restart_states != 1)):
            raise ValueError("restart_states must contain only -1 and +1")
        prepared_restart_states = source_restart_states.to(dtype=torch.int8).clone()

    component_local_fields = None
    component_energy = None
    component_sparse = None
    if hamiltonian_component is not None:
        if field_backend == "dense":
            component_model = hamiltonian_component.to(
                device=device,
                dtype=coefficient_dtype,
            )
            component_local_fields = component_model.local_fields
            component_energy = component_model.energy
            backend_storage_bytes += (
                component_model.coupling.numel() * component_model.coupling.element_size()
                + component_model.bias.numel() * component_model.bias.element_size()
            )
        else:
            component_sparse = hamiltonian_component_padded or PaddedSparseIsing.from_model(
                hamiltonian_component,
                device=device,
                dtype=coefficient_dtype,
            )
            if component_sparse.size != model.size:
                raise ValueError("Padded Hamiltonian component size mismatch")
            component_local_fields = (
                component_sparse.triton_local_fields
                if field_backend == "triton_sparse"
                else component_sparse.local_fields
            )
            component_energy = component_sparse.energy
            backend_storage_bytes += component_sparse.storage_bytes

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    rng_recorder = RngAuditRecorder(generator, seed=seed, stream_name="sampler")
    exact_block_generator = generator
    exact_block_rng_recorder = rng_recorder
    separate_exact_block_rng_recorder = None
    if exact_block_rng_seed is not None:
        exact_block_generator = torch.Generator(device=device)
        exact_block_generator.manual_seed(exact_block_rng_seed)
        separate_exact_block_rng_recorder = RngAuditRecorder(
            exact_block_generator,
            seed=exact_block_rng_seed,
            stream_name="exact_blocks",
        )
        exact_block_rng_recorder = separate_exact_block_rng_recorder
    junction_tree_generator = None
    junction_tree_rng_recorder = None
    junction_tree_device_model = None
    if junction_tree_plan_cycle:
        if junction_tree_rng_seed is None:
            raise RuntimeError("Junction-tree RNG seed disappeared after validation")
        junction_tree_generator = torch.Generator(device=device)
        junction_tree_generator.manual_seed(junction_tree_rng_seed)
        junction_tree_rng_recorder = RngAuditRecorder(
            junction_tree_generator,
            seed=junction_tree_rng_seed,
            stream_name="junction_tree",
        )
        junction_tree_device_model = model.to(device=device, dtype=coefficient_dtype)
        backend_storage_bytes += (
            junction_tree_device_model.coupling.numel()
            * junction_tree_device_model.coupling.element_size()
            + junction_tree_device_model.bias.numel()
            * junction_tree_device_model.bias.element_size()
        )
    counter_rng_recorder = None
    counter_seed = None
    if gibbs_kernel in {"triton_counter_reference", "triton_counter_fused"}:
        counter_seed = derive_stream_seed(seed, "triton_counter_gibbs_v1")
        counter_rng_recorder = CounterRngAuditRecorder(
            seed=counter_seed,
            device=device,
            stream_name="triton_counter_gibbs_v1",
            start_offset=counter_rng_start_offset,
        )
    hastings_generator = None
    hastings_rng_recorder = None
    if hastings_blocks is not None:
        hastings_seed = derive_stream_seed(seed, "hastings_repair")
        hastings_generator = torch.Generator(device=device)
        hastings_generator.manual_seed(hastings_seed)
        hastings_rng_recorder = RngAuditRecorder(
            hastings_generator,
            seed=hastings_seed,
            stream_name="hastings_repair",
        )
    codeword_macro_generator = None
    codeword_macro_rng_recorder = None
    prepared_codeword_macro = None
    if codeword_macro_plan is not None:
        codeword_macro_seed = derive_stream_seed(seed, "codeword_transport")
        codeword_macro_generator = torch.Generator(device=device)
        codeword_macro_generator.manual_seed(codeword_macro_seed)
        codeword_macro_rng_recorder = RngAuditRecorder(
            codeword_macro_generator,
            seed=codeword_macro_seed,
            stream_name="codeword_transport",
        )
        prepared_codeword_macro = codeword_macro_plan.to(device)
        backend_storage_bytes += prepared_codeword_macro.storage_bytes
    if initial_states is None:
        rng_recorder.record("independent_initialization", replicas * model.size)
        states = _random_spins(replicas, model.size, device, generator)
    else:
        if initial_states.shape != (replicas, model.size):
            raise ValueError("initial_states must have shape [replicas, model.size]")
        source_states = initial_states.to(device=device)
        if torch.any((source_states != -1) & (source_states != 1)):
            raise ValueError("initial_states must contain only -1 and +1")
        states = source_states.to(dtype=torch.int8).clone()

    color_tensors = [torch.tensor(block, dtype=torch.long, device=device) for block in colors]
    block_plans = (
        _prepare_metropolis_blocks(
            model,
            metropolis_blocks,
            metropolis_block_colors,
            device,
            coefficient_dtype,
        )
        if metropolis_blocks is not None and metropolis_block_colors is not None
        else ()
    )
    exact_block_plans = (
        _prepare_exact_blocks(
            model,
            exact_blocks,
            exact_block_colors,
            device,
            coefficient_dtype,
            replicas=replicas,
            maximum_block_size=exact_block_max_size,
            workspace_bytes=int(exact_block_workspace_mib * 1024 * 1024),
        )
        if exact_blocks is not None and exact_block_colors is not None
        else ()
    )
    treewidth_corridor_plans = (
        _prepare_treewidth_corridors(
            model,
            treewidth_corridors,
            treewidth_corridor_colors,
            device,
            coefficient_dtype,
            maximum_treewidth=treewidth_corridor_max_treewidth,
            message_mode=treewidth_corridor_message_mode,
        )
        if treewidth_corridors is not None and treewidth_corridor_colors is not None
        else ()
    )
    hastings_node_tensors: tuple[torch.Tensor, ...] = ()
    hastings_target_device_model = None
    hastings_proposal_device_model = None
    if hastings_blocks is not None:
        normalized_hastings = tuple(
            tuple(int(index) for index in block) for block in hastings_blocks
        )
        if not normalized_hastings or any(
            not block or len(block) != len(set(block)) for block in normalized_hastings
        ):
            raise ValueError("hastings_blocks must contain non-empty node sets")
        if any(
            index < 0 or index >= model.size for block in normalized_hastings for index in block
        ):
            raise IndexError("hastings_blocks contain an out-of-range node")
        if hastings_proposal_model is None or hastings_proposal_model.names != model.names:
            raise ValueError("The Hastings proposal model must share the target variable order")
        hastings_node_tensors = tuple(
            torch.tensor(block, dtype=torch.long, device=device) for block in normalized_hastings
        )
        hastings_target_device_model = model.to(device=device, dtype=coefficient_dtype)
        hastings_proposal_device_model = hastings_proposal_model.to(
            device=device, dtype=coefficient_dtype
        )
        backend_storage_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                hastings_target_device_model.coupling,
                hastings_target_device_model.bias,
                hastings_proposal_device_model.coupling,
                hastings_proposal_device_model.bias,
            )
        )
    target_energies = energy(states)
    component_energies = component_energy(states) if component_energy is not None else None
    ladder_size = len(tempering_ladder) or len(hamiltonian_ladder)
    walker_count = replicas // ladder_size if ladder_size else replicas
    walker_at_slot = (
        torch.arange(replicas, dtype=torch.long, device=device) if ladder_size else None
    )
    base_tempering_betas = (
        torch.tensor(tempering_ladder, dtype=coefficient_dtype, device=device).repeat(walker_count)
        if tempering_ladder
        else None
    )
    hamiltonian_slot_weights = (
        torch.tensor(hamiltonian_ladder, dtype=coefficient_dtype, device=device).repeat(
            walker_count
        )
        if hamiltonian_ladder
        else None
    )
    exchange_attempts = torch.zeros(
        max(0, len(tempering_ladder) - 1), dtype=torch.int64, device=device
    )
    exchange_accepts = torch.zeros_like(exchange_attempts)
    hamiltonian_exchange_attempts = torch.zeros(
        max(0, len(hamiltonian_ladder) - 1), dtype=torch.int64, device=device
    )
    hamiltonian_exchange_accepts = torch.zeros_like(hamiltonian_exchange_attempts)
    visited_hot = torch.zeros(replicas, dtype=torch.bool, device=device)
    visited_cold = torch.zeros(replicas, dtype=torch.bool, device=device)
    if walker_at_slot is not None:
        slot_grid = walker_at_slot.view(walker_count, ladder_size)
        visited_hot[slot_grid[:, 0]] = True
        visited_cold[slot_grid[:, -1]] = True

    def ordered_values(values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None or walker_at_slot is None:
            return values
        ordered = torch.empty_like(values)
        ordered[walker_at_slot] = values
        return ordered

    def ordered_weights(common_weight: float) -> torch.Tensor | None:
        if hamiltonian_slot_weights is not None:
            return ordered_values(hamiltonian_slot_weights)
        if hamiltonian_component is not None:
            return torch.full(
                (replicas,),
                common_weight,
                dtype=coefficient_dtype,
                device=device,
            )
        return None

    def target_endpoint_mask(common_weight: float) -> torch.Tensor | None:
        if hamiltonian_slot_weights is not None:
            slot_mask = torch.zeros(replicas, dtype=torch.bool, device=device)
            if hamiltonian_target_slot is None:
                raise RuntimeError("Hamiltonian target slot was not resolved")
            slot_mask.view(walker_count, ladder_size)[:, hamiltonian_target_slot] = True
            if walker_at_slot is None:
                return slot_mask
            ordered = torch.zeros_like(slot_mask)
            ordered[walker_at_slot] = slot_mask
            return ordered
        if weight_schedule:
            enabled = math.isclose(common_weight, hamiltonian_target_weight, abs_tol=1e-12)
            return torch.full((replicas,), enabled, dtype=torch.bool, device=device)
        return None

    def success_weight_mask(common_weight: float) -> torch.Tensor | None:
        if hamiltonian_success_policy == "shared_ground_state":
            return None
        return target_endpoint_mask(common_weight)

    def update_first_passage(
        sweep_value: int,
        ordered_target: torch.Tensor,
        ordered_component: torch.Tensor | None,
    ) -> None:
        if ordered_component is None:
            return
        ordered_base = ordered_target - hamiltonian_target_weight * ordered_component
        base_zero = torch.abs(ordered_base) <= component_zero_tolerance
        component_zero = torch.abs(ordered_component) <= component_zero_tolerance
        component_nonzero = ~component_zero
        new_base = base_zero & (first_base_zero < 0)
        new_component = component_zero & (first_component_zero < 0)
        new_component_nonzero = component_nonzero & (first_component_nonzero < 0)
        component_was_nonzero.logical_or_(component_nonzero)
        new_component_rezero = component_zero & component_was_nonzero & (first_component_rezero < 0)
        new_joint = base_zero & component_zero & (first_joint_zero < 0)
        first_base_zero[new_base] = sweep_value
        first_component_zero[new_component] = sweep_value
        first_component_nonzero[new_component_nonzero] = sweep_value
        first_component_rezero[new_component_rezero] = sweep_value
        first_joint_zero[new_joint] = sweep_value

    def emit_trace(
        sweep_value: int,
        beta_value: float,
        common_weight: float,
        succeeded: torch.Tensor | None,
    ) -> None:
        trace_beta = beta_value * (tempering_ladder[-1] if tempering_ladder else 1.0)
        traces.append(
            _trace_point(
                sweep_value,
                trace_beta,
                ordered_target_energies,
                best_energies,
                first_hit >= 0 if success else succeeded,
            )
        )
        if on_trace is not None:
            ordered_component = ordered_values(component_energies)
            on_trace(
                AnnealSnapshot(
                    sweep=sweep_value,
                    beta=trace_beta,
                    states=ordered_states,
                    target_energies=ordered_target_energies,
                    base_energies=(
                        None
                        if ordered_component is None
                        else ordered_target_energies - hamiltonian_target_weight * ordered_component
                    ),
                    component_energies=ordered_component,
                    component_weights=ordered_weights(common_weight),
                    target_endpoint_mask=target_endpoint_mask(common_weight),
                )
            )

    common_weight = weight_schedule[0] if weight_schedule else hamiltonian_target_weight
    ordered_states, ordered_target_energies = _walker_order(states, target_energies, walker_at_slot)
    best_energies = ordered_target_energies.clone()
    best_states = ordered_states.clone()
    first_hit = torch.full((replicas,), -1, dtype=torch.int64, device=device)
    first_base_zero = torch.full_like(first_hit, -1)
    first_component_zero = torch.full_like(first_hit, -1)
    first_component_nonzero = torch.full_like(first_hit, -1)
    first_component_rezero = torch.full_like(first_hit, -1)
    component_was_nonzero = torch.zeros_like(first_hit, dtype=torch.bool)
    first_joint_zero = torch.full_like(first_hit, -1)
    traces: list[AnnealTracePoint] = []
    update_first_passage(0, ordered_target_energies, ordered_values(component_energies))
    initial_succeeded = None
    if success is not None:
        initial_succeeded = success(ordered_states)
        if initial_succeeded.shape != (replicas,) or initial_succeeded.dtype != torch.bool:
            raise ValueError("success must return a boolean tensor with shape [replicas]")
        if on_predicate_evaluation is not None:
            on_predicate_evaluation(
                ordered_states,
                ordered_target_energies,
                initial_succeeded,
                0,
            )
        if success_energy_upper_bound is not None:
            initial_succeeded = initial_succeeded & (
                ordered_target_energies <= float(success_energy_upper_bound)
            )
        endpoint = success_weight_mask(common_weight)
        if endpoint is not None:
            initial_succeeded = initial_succeeded & endpoint
        first_hit[initial_succeeded] = 0
        if on_first_hit is not None:
            on_first_hit(ordered_states, ordered_target_energies, initial_succeeded)
        if on_first_hit_event is not None:
            on_first_hit_event(
                ordered_states,
                ordered_target_energies,
                initial_succeeded,
                0,
            )
    final_succeeded = initial_succeeded
    emit_trace(0, float(betas[0]), common_weight, initial_succeeded)

    backend_prepare_seconds = 0.0
    if field_backend == "triton_sparse":
        prepare_start = time.perf_counter()
        warm_states = states.clone()
        warm_block = color_tensors[0]
        warm_fields = local_fields(warm_states, warm_block)
        if gibbs_kernel == "triton_external_uniform_bridge":
            from .triton_sparse import triton_external_uniform_gibbs_write

            warm_probabilities = torch.sigmoid(2.0 * float(betas[0]) * warm_fields)
            triton_external_uniform_gibbs_write(
                warm_states,
                warm_probabilities,
                torch.zeros_like(warm_probabilities),
                warm_block,
            )
        elif gibbs_kernel == "triton_counter_reference":
            from .triton_sparse import triton_counter_gibbs_update_from_fields

            if counter_seed is None:
                raise RuntimeError("Counter RNG seed was not prepared")
            triton_counter_gibbs_update_from_fields(
                warm_states,
                warm_fields,
                warm_block,
                beta=float(betas[0]),
                seed=counter_seed,
                counter_offset=0,
            )
        elif gibbs_kernel == "triton_counter_fused":
            if counter_seed is None:
                raise RuntimeError("Counter RNG seed was not prepared")
            if sparse is None:
                raise RuntimeError("Fused counter Gibbs requires a padded sparse model")
            sparse.triton_fused_counter_gibbs_update(
                warm_states,
                warm_block,
                beta=float(betas[0]),
                seed=counter_seed,
                counter_offset=0,
            )
        if component_local_fields is not None:
            component_local_fields(states, color_tensors[0])
        torch.cuda.synchronize(device)
        backend_prepare_seconds = time.perf_counter() - prepare_start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    sweep = 0
    restart_count = 0
    exact_block_trigger_count = 0
    exact_block_update_events = 0
    exact_block_candidate_evaluations = 0
    exact_block_field_evaluation_calls = 0
    exact_block_field_node_evaluations = 0
    junction_tree_update_events = 0
    junction_tree_random_draws = 0
    junction_tree_maximum_forward_states = 0
    junction_tree_chunk_count_per_event = 0
    junction_tree_plan_update_events = [0] * len(junction_tree_plan_cycle)
    explicit_junction_tree_event_ordinals = {
        event_sweep: event_index
        for event_index, event_sweep in enumerate(explicit_junction_tree_sweeps)
    }
    treewidth_corridor_trigger_count = 0
    treewidth_corridor_update_events = 0
    treewidth_corridor_gate_update_events = 0
    treewidth_corridor_candidate_evaluations = 0
    treewidth_corridor_field_evaluation_calls = 0
    treewidth_corridor_field_node_evaluations = 0
    hastings_trigger_count = 0
    hastings_update_events = 0
    hastings_proposals = 0
    hastings_accepts = 0
    hastings_proposed_changes = 0
    hastings_accepted_changes = 0
    codeword_macro_update_events = 0
    codeword_macro_proposals = 0
    codeword_macro_accepts = 0
    codeword_macro_candidate_evaluations = 0
    codeword_macro_downhill_proposals = 0
    codeword_macro_downhill_accepts = 0
    codeword_macro_selected_move_counts = (
        torch.zeros(prepared_codeword_macro.move_count, dtype=torch.int64, device=device)
        if prepared_codeword_macro is not None
        else None
    )
    codeword_macro_accepted_move_counts = (
        torch.zeros(prepared_codeword_macro.move_count, dtype=torch.int64, device=device)
        if prepared_codeword_macro is not None
        else None
    )
    local_field_clip_proposals = 0
    local_field_clip_accepts = 0
    local_field_clip_proposed_flips = 0
    local_field_clip_accepted_flips = 0
    fused_gibbs_calls = 0
    all_succeeded_initially = (
        stop_when_all_succeeded
        and initial_succeeded is not None
        and bool(torch.all(initial_succeeded))
    )
    beta_stages = zip(betas, resolved_beta_stage_sweeps, strict=True)
    for beta_value, stage_sweeps in () if all_succeeded_initially else beta_stages:
        beta = float(beta_value)
        for _ in range(stage_sweeps):
            next_sweep = sweep + 1
            common_weight = (
                weight_schedule[next_sweep - 1] if weight_schedule else hamiltonian_target_weight
            )
            if restart_every_sweeps and sweep > 0 and sweep % restart_every_sweeps == 0:
                if prepared_restart_states is None:
                    rng_recorder.record("restart_initialization", replicas * model.size)
                    states = _random_spins(replicas, model.size, device, generator)
                else:
                    states = prepared_restart_states[restart_count].clone()
                target_energies = energy(states)
                restart_count += 1
                improved = target_energies < best_energies
                best_energies = torch.where(improved, target_energies, best_energies)
                best_states = torch.where(improved[:, None], states, best_states)
                if success is not None:
                    restarted_success = success(states)
                    if (
                        restarted_success.shape != (replicas,)
                        or restarted_success.dtype != torch.bool
                    ):
                        raise ValueError(
                            "success must return a boolean tensor with shape [replicas]"
                        )
                    if on_predicate_evaluation is not None:
                        on_predicate_evaluation(
                            states,
                            target_energies,
                            restarted_success,
                            sweep,
                        )
                    if success_energy_upper_bound is not None:
                        restarted_success = restarted_success & (
                            target_energies <= float(success_energy_upper_bound)
                        )
                    restarted_hits = restarted_success & (first_hit < 0)
                    first_hit[restarted_hits] = sweep
                    if on_first_hit is not None:
                        on_first_hit(states, target_energies, restarted_hits)
                    if on_first_hit_event is not None:
                        on_first_hit_event(
                            states,
                            target_energies,
                            restarted_hits,
                            sweep,
                        )
            effective_betas = (
                base_tempering_betas * beta if base_tempering_betas is not None else None
            )
            if gibbs_sweep_every > 0 and next_sweep % gibbs_sweep_every == 0:
                if randomize_color_order:
                    rng_recorder.record("color_order", len(color_tensors))
                    order = torch.randperm(
                        len(color_tensors), generator=generator, device=device
                    ).tolist()
                else:
                    order = range(len(color_tensors))
                for color_index in order:
                    block = color_tensors[color_index]
                    previous = states.index_select(1, block)
                    component_fields = None
                    if gibbs_kernel == "triton_counter_fused":
                        if counter_seed is None or counter_rng_recorder is None:
                            raise RuntimeError("Counter RNG state was not prepared")
                        if sparse is None:
                            raise RuntimeError("Fused counter Gibbs requires a padded sparse model")
                        counter_offset = counter_rng_recorder.reserve(
                            "gibbs_uniform",
                            replicas * block.numel(),
                        )
                        target_fields = sparse.triton_fused_counter_gibbs_update(
                            states,
                            block,
                            beta=beta,
                            seed=counter_seed,
                            counter_offset=counter_offset,
                        )
                        updated = states.index_select(1, block)
                        fused_gibbs_calls += 1
                    else:
                        target_fields = local_fields(states, block)
                        component_fields = (
                            component_local_fields(states, block)
                            if component_local_fields is not None
                            else None
                        )
                        fields = target_fields
                        if component_fields is not None:
                            if hamiltonian_slot_weights is not None:
                                weight_delta: float | torch.Tensor = (
                                    hamiltonian_slot_weights - hamiltonian_target_weight
                                )[:, None]
                            else:
                                weight_delta = common_weight - hamiltonian_target_weight
                            fields = target_fields + weight_delta * component_fields
                        selected_beta: float | torch.Tensor = (
                            beta if effective_betas is None else effective_betas[:, None]
                        )
                        dimensionless_fields = fields * selected_beta
                        proposal_fields = (
                            dimensionless_fields
                            if local_field_proposal_clip is None
                            else torch.clamp(
                                dimensionless_fields,
                                min=-float(local_field_proposal_clip),
                                max=float(local_field_proposal_clip),
                            )
                        )
                        if gibbs_kernel == "triton_counter_reference":
                            from .triton_sparse import triton_counter_gibbs_update_from_fields

                            if counter_seed is None or counter_rng_recorder is None:
                                raise RuntimeError("Counter RNG state was not prepared")
                            counter_offset = counter_rng_recorder.reserve(
                                "gibbs_uniform",
                                replicas * block.numel(),
                            )
                            triton_counter_gibbs_update_from_fields(
                                states,
                                target_fields,
                                block,
                                beta=beta,
                                seed=counter_seed,
                                counter_offset=counter_offset,
                            )
                            updated = states.index_select(1, block)
                        else:
                            probabilities = torch.sigmoid(2.0 * proposal_fields)
                            random_values = torch.rand(
                                probabilities.shape,
                                dtype=(
                                    probabilities.dtype
                                    if gibbs_uniform_dtype is None
                                    else gibbs_uniform_dtype
                                ),
                                device=device,
                                generator=generator,
                            )
                            rng_recorder.record("gibbs_uniform", probabilities.numel())
                            if gibbs_kernel == "triton_external_uniform_bridge":
                                from .triton_sparse import triton_external_uniform_gibbs_write

                                triton_external_uniform_gibbs_write(
                                    states,
                                    probabilities,
                                    random_values,
                                    block,
                                )
                                updated = states.index_select(1, block)
                                fused_gibbs_calls += 1
                            else:
                                updated = (
                                    (random_values < probabilities).mul(2).sub(1).to(torch.int8)
                                )
                            if local_field_proposal_clip is not None:
                                proposed_flip = updated != previous
                                acceptance_log_ratio = (updated - previous).to(
                                    coefficient_dtype
                                ) * (dimensionless_fields - proposal_fields)
                                acceptance_uniform = torch.rand(
                                    acceptance_log_ratio.shape,
                                    dtype=coefficient_dtype,
                                    device=device,
                                    generator=generator,
                                )
                                rng_recorder.record(
                                    "local_field_clip_acceptance_uniform",
                                    acceptance_log_ratio.numel(),
                                )
                                accepted = torch.log(acceptance_uniform) < torch.minimum(
                                    torch.zeros_like(acceptance_log_ratio),
                                    acceptance_log_ratio,
                                )
                                updated = torch.where(accepted, updated, previous)
                                local_field_clip_proposals += accepted.numel()
                                local_field_clip_accepts += int(
                                    torch.count_nonzero(accepted).item()
                                )
                                local_field_clip_proposed_flips += int(
                                    torch.count_nonzero(proposed_flip).item()
                                )
                                local_field_clip_accepted_flips += int(
                                    torch.count_nonzero(accepted & proposed_flip).item()
                                )
                    if energy_tracking == "incremental":
                        spin_delta = (previous - updated).to(dtype=coefficient_dtype)
                        target_delta = (spin_delta * target_fields).sum(dim=1)
                        target_energies.add_(target_delta)
                        if component_energies is not None and component_fields is not None:
                            component_energies.add_((spin_delta * component_fields).sum(dim=1))
                    if gibbs_kernel == "eager_torch":
                        states[:, block] = updated

            sweep = next_sweep
            if block_plans and sweep % metropolis_block_every == 0:
                for plan in block_plans:
                    selected = states.index_select(1, plan.nodes)
                    fields = local_fields(states, plan.nodes)
                    internal_fields = selected.to(coefficient_dtype) @ plan.internal_coupling.mT
                    node_deltas = 2.0 * selected.to(coefficient_dtype) * (fields - internal_fields)
                    deltas = torch.zeros(
                        (replicas, plan.block_count),
                        dtype=coefficient_dtype,
                        device=device,
                    )
                    deltas.scatter_add_(
                        1,
                        plan.node_to_block.expand(replicas, -1),
                        node_deltas,
                    )
                    random_values = torch.rand(
                        deltas.shape,
                        dtype=coefficient_dtype,
                        device=device,
                        generator=generator,
                    )
                    rng_recorder.record("metropolis_block_uniform", deltas.numel())
                    if metropolis_block_rule == "metropolis":
                        accepted_blocks = torch.log(random_values) < torch.minimum(
                            torch.zeros_like(deltas),
                            -(beta if effective_betas is None else effective_betas[:, None])
                            * deltas,
                        )
                    else:
                        accepted_blocks = random_values < torch.sigmoid(
                            -(beta if effective_betas is None else effective_betas[:, None])
                            * deltas
                        )
                    accepted_nodes = accepted_blocks.index_select(1, plan.node_to_block)
                    states[:, plan.nodes] = torch.where(
                        accepted_nodes,
                        -selected,
                        selected,
                    )
                    if energy_tracking == "incremental":
                        target_energies.add_(torch.where(accepted_blocks, deltas, 0.0).sum(dim=1))

            if exact_block_plans and sweep % exact_block_every == 0:
                exact_block_trigger_count += 1
                if exact_block_color_mode == "all":
                    selected_colors = range(len(exact_block_plans))
                else:
                    selected_colors = ((exact_block_trigger_count - 1) % len(exact_block_plans),)
                for exact_color in selected_colors:
                    exact_block_update_events += 1
                    color_plan = exact_block_plans[exact_color]
                    color_fields = local_fields(states, color_plan.nodes)
                    color_states = states.index_select(1, color_plan.nodes).to(coefficient_dtype)
                    exact_block_field_evaluation_calls += 1
                    exact_block_field_node_evaluations += replicas * color_plan.nodes.numel()
                    for plan in color_plan.groups:
                        flat_positions = plan.field_positions.reshape(-1)
                        selected = color_states.index_select(1, flat_positions).reshape(
                            replicas, plan.block_count, plan.block_size
                        )
                        fields = color_fields.index_select(1, flat_positions).reshape(
                            replicas,
                            plan.block_count,
                            plan.block_size,
                        )
                        boundary_fields, candidate_energies = _exact_block_candidate_energies(
                            selected,
                            fields,
                            plan.internal_coupling,
                            plan.candidates,
                            plan.candidate_internal_energies,
                        )
                        selected_beta: float | torch.Tensor = beta
                        if effective_betas is not None:
                            selected_beta = effective_betas[:, None, None]
                        probabilities = torch.softmax(
                            -selected_beta * candidate_energies,
                            dim=-1,
                        )
                        random_values = torch.rand(
                            (replicas, plan.block_count, 1),
                            dtype=coefficient_dtype,
                            device=device,
                            generator=exact_block_generator,
                        )
                        exact_block_rng_recorder.record(
                            "exact_block_heat_bath_uniform",
                            replicas * plan.block_count,
                        )
                        candidate_index = torch.sum(
                            random_values > torch.cumsum(probabilities, dim=-1),
                            dim=-1,
                        ).clamp_max(plan.candidates.shape[0] - 1)
                        updated = plan.candidates[candidate_index]
                        color_states[:, flat_positions] = updated.reshape(
                            replicas,
                            plan.block_count * plan.block_size,
                        )
                        if energy_tracking == "incremental":
                            old_energies = -torch.sum(selected * boundary_fields, dim=-1) - (
                                0.5
                                * torch.einsum(
                                    "rbk,bkl,rbl->rb",
                                    selected,
                                    plan.internal_coupling,
                                    selected,
                                )
                            )
                            new_energies = candidate_energies.gather(
                                2,
                                candidate_index[:, :, None],
                            ).squeeze(2)
                            target_energies.add_((new_energies - old_energies).sum(dim=1))
                        exact_block_candidate_evaluations += (
                            replicas * plan.block_count * plan.candidates.shape[0]
                        )
                    states[:, color_plan.nodes] = color_states.to(torch.int8)

            if hastings_node_tensors and sweep % hastings_every == 0:
                if hastings_target_device_model is None or hastings_proposal_device_model is None:
                    raise RuntimeError("Hastings device models were not prepared")
                if hastings_generator is None or hastings_rng_recorder is None:
                    raise RuntimeError("Hastings RNG stream was not prepared")
                hastings_trigger_count += 1
                if hastings_block_mode == "all":
                    selected_hastings_blocks = range(len(hastings_node_tensors))
                else:
                    selected_hastings_blocks = (
                        (hastings_trigger_count - 1) % len(hastings_node_tensors),
                    )
                selected_beta: float | torch.Tensor = (
                    beta if effective_betas is None else effective_betas
                )
                for block_index in selected_hastings_blocks:
                    update = sequential_auxiliary_hastings_step(
                        hastings_target_device_model,
                        hastings_proposal_device_model,
                        states,
                        hastings_node_tensors[block_index],
                        selected_beta,
                        generator=hastings_generator,
                        proposal_logit_clip=hastings_proposal_logit_clip,
                        rng_recorder=hastings_rng_recorder,
                    )
                    states = update.states
                    if energy_tracking == "incremental":
                        target_energies.add_(update.accepted_energy_delta)
                    hastings_update_events += 1
                    hastings_proposals += replicas
                    hastings_accepts += int(torch.count_nonzero(update.accepted).item())
                    hastings_proposed_changes += int(
                        torch.count_nonzero(update.proposed_change).item()
                    )
                    hastings_accepted_changes += int(
                        torch.count_nonzero(update.accepted & update.proposed_change).item()
                    )

            if treewidth_corridor_plans and sweep % treewidth_corridor_every == 0:
                treewidth_corridor_trigger_count += 1
                if treewidth_corridor_color_mode == "all":
                    selected_corridor_colors = range(len(treewidth_corridor_plans))
                else:
                    selected_corridor_colors = (
                        (treewidth_corridor_trigger_count - 1) % len(treewidth_corridor_plans),
                    )
                for corridor_color in selected_corridor_colors:
                    color_plan = treewidth_corridor_plans[corridor_color]
                    color_fields = local_fields(states, color_plan.nodes)
                    color_states = states.index_select(1, color_plan.nodes)
                    treewidth_corridor_field_evaluation_calls += 1
                    treewidth_corridor_field_node_evaluations += replicas * color_plan.nodes.numel()
                    selected_beta: float | torch.Tensor = beta
                    if effective_betas is not None:
                        selected_beta = effective_betas
                    for corridor in color_plan.corridors:
                        selected = color_states.index_select(1, corridor.field_positions)
                        fields = color_fields.index_select(1, corridor.field_positions)
                        if isinstance(corridor.plan, SeparatorChainPlan):
                            updated, target_delta = sample_separator_chain(
                                corridor.plan,
                                selected,
                                fields,
                                selected_beta,
                                generator=generator,
                                rng_recorder=rng_recorder,
                            )
                        else:
                            updated, target_delta = sample_transfer_matrix_chain(
                                corridor.plan,
                                selected,
                                fields,
                                selected_beta,
                                generator=generator,
                                rng_recorder=rng_recorder,
                            )
                        color_states[:, corridor.field_positions] = updated
                        if energy_tracking == "incremental":
                            target_energies.add_(target_delta)
                        treewidth_corridor_update_events += 1
                        treewidth_corridor_gate_update_events += corridor.plan.gate_count
                        treewidth_corridor_candidate_evaluations += (
                            replicas * corridor.plan.candidate_evaluations_per_replica
                        )
                    states[:, color_plan.nodes] = color_states

            macro_due = (
                prepared_codeword_macro is not None
                and codeword_macro_start_sweep <= sweep
                and (codeword_macro_stop_sweep is None or sweep <= codeword_macro_stop_sweep)
                and sweep % codeword_macro_every == 0
            )
            if macro_due:
                if codeword_macro_generator is None or codeword_macro_rng_recorder is None:
                    raise RuntimeError("Codeword macro RNG stream was not prepared")
                selected_beta = beta if effective_betas is None else effective_betas
                macro_current_energies = (
                    target_energies if energy_tracking == "incremental" else energy(states)
                )
                if codeword_macro_rule == "uniform":
                    update = involutive_codeword_metropolis_step(
                        states,
                        macro_current_energies,
                        selected_beta,
                        prepared_codeword_macro,
                        energy,
                        generator=codeword_macro_generator,
                        rng_recorder=codeword_macro_rng_recorder,
                    )
                    codeword_macro_candidate_evaluations += replicas
                else:
                    update = locally_balanced_involutive_metropolis_step(
                        states,
                        macro_current_energies,
                        selected_beta,
                        prepared_codeword_macro,
                        energy,
                        rule=codeword_macro_rule.removeprefix("locally_balanced_"),
                        uniform_mixture=codeword_macro_uniform_mixture,
                        generator=codeword_macro_generator,
                        rng_recorder=codeword_macro_rng_recorder,
                    )
                    codeword_macro_candidate_evaluations += update.candidate_evaluations
                states = update.states
                if energy_tracking == "incremental":
                    target_energies.add_(update.accepted_energy_delta)
                codeword_macro_update_events += 1
                codeword_macro_proposals += replicas
                codeword_macro_accepts += int(torch.count_nonzero(update.accepted).item())
                downhill = update.proposed_energy_delta < 0
                codeword_macro_downhill_proposals += int(torch.count_nonzero(downhill).item())
                codeword_macro_downhill_accepts += int(
                    torch.count_nonzero(downhill & update.accepted).item()
                )
                if (
                    codeword_macro_selected_move_counts is None
                    or codeword_macro_accepted_move_counts is None
                ):
                    raise RuntimeError("Codeword macro move counters were not prepared")
                codeword_macro_selected_move_counts.add_(
                    torch.bincount(update.selectors, minlength=prepared_codeword_macro.move_count)
                )
                codeword_macro_accepted_move_counts.add_(
                    torch.bincount(
                        update.selectors[update.accepted],
                        minlength=prepared_codeword_macro.move_count,
                    )
                )

            if explicit_junction_tree_sweeps:
                junction_tree_due = sweep in explicit_junction_tree_event_ordinals
            else:
                junction_tree_due = (
                    bool(junction_tree_plan_cycle)
                    and junction_tree_start_sweep <= sweep
                    and (junction_tree_stop_sweep is None or sweep <= junction_tree_stop_sweep)
                    and sweep % junction_tree_every == 0
                )
            if junction_tree_due:
                if (
                    junction_tree_device_model is None
                    or junction_tree_generator is None
                    or junction_tree_rng_recorder is None
                    or not junction_tree_plan_cycle
                ):
                    raise RuntimeError("Junction-tree sampler was not prepared")
                if junction_tree_event_plan_indices is not None:
                    junction_tree_plan_index = explicit_junction_tree_plan_indices[
                        explicit_junction_tree_event_ordinals[sweep]
                    ]
                else:
                    junction_tree_plan_index = junction_tree_update_events % len(
                        junction_tree_plan_cycle
                    )
                selected_junction_tree_plan = junction_tree_plan_cycle[junction_tree_plan_index]
                if junction_tree_event_replica_ranges is None:
                    event_replica_start, event_replica_stop = 0, replicas
                else:
                    event_replica_start, event_replica_stop = explicit_junction_tree_replica_ranges[
                        explicit_junction_tree_event_ordinals[sweep]
                    ]
                event_ordinal = explicit_junction_tree_event_ordinals.get(sweep)
                selected_beta = (
                    explicit_junction_tree_beta_overrides[event_ordinal]
                    if junction_tree_event_beta_overrides is not None and event_ordinal is not None
                    else float(junction_tree_beta_override)
                    if junction_tree_beta_override is not None
                    else beta
                    if effective_betas is None
                    else effective_betas[event_replica_start:event_replica_stop]
                )
                junction_update = sample_junction_tree_conditional(
                    junction_tree_device_model,
                    selected_junction_tree_plan,
                    states[event_replica_start:event_replica_stop],
                    selected_beta,
                    generator=junction_tree_generator,
                    rng_recorder=junction_tree_rng_recorder,
                    replica_chunk_size=junction_tree_replica_chunk_size,
                )
                updated_states = states.clone()
                updated_states[event_replica_start:event_replica_stop] = junction_update.states
                if on_junction_tree_event is not None:
                    on_junction_tree_event(
                        sweep,
                        junction_tree_plan_index,
                        states,
                        updated_states,
                    )
                states = updated_states
                if energy_tracking == "incremental":
                    target_energies[event_replica_start:event_replica_stop].add_(
                        junction_update.energy_delta
                    )
                junction_tree_update_events += 1
                junction_tree_plan_update_events[junction_tree_plan_index] += 1
                junction_tree_random_draws += junction_update.random_draws
                junction_tree_maximum_forward_states = max(
                    junction_tree_maximum_forward_states,
                    junction_update.maximum_forward_table_states_per_replica,
                )
                junction_tree_chunk_count_per_event = max(
                    junction_tree_chunk_count_per_event,
                    junction_update.chunk_count,
                )

            if energy_tracking == "recompute":
                target_energies = energy(states)
                if component_energy is not None:
                    component_energies = component_energy(states)
            elif component_energy is not None and (
                (block_plans and sweep % metropolis_block_every == 0)
                or (exact_block_plans and sweep % exact_block_every == 0)
                or (treewidth_corridor_plans and sweep % treewidth_corridor_every == 0)
                or (hastings_node_tensors and sweep % hastings_every == 0)
                or macro_due
                or junction_tree_due
            ):
                component_energies = component_energy(states)
            if tempering_ladder and sweep % parallel_tempering_exchange_every == 0:
                parity = ((sweep // parallel_tempering_exchange_every) - 1) % 2
                slot_grid = torch.arange(replicas, device=device).view(walker_count, ladder_size)
                left = slot_grid[:, parity : ladder_size - 1 : 2].reshape(-1)
                right = left + 1
                if left.numel():
                    left_beta = effective_betas.index_select(0, left)
                    right_beta = effective_betas.index_select(0, right)
                    log_acceptance = (left_beta - right_beta) * (
                        target_energies.index_select(0, left)
                        - target_energies.index_select(0, right)
                    )
                    random_values = torch.rand(
                        log_acceptance.shape,
                        dtype=coefficient_dtype,
                        device=device,
                        generator=generator,
                    )
                    rng_recorder.record("parallel_tempering_exchange_uniform", left.numel())
                    accepted = torch.log(random_values) < torch.minimum(
                        torch.zeros_like(log_acceptance), log_acceptance
                    )
                    accepted_left = left[accepted]
                    accepted_right = right[accepted]
                    if accepted_left.numel():
                        left_states = states.index_select(0, accepted_left).clone()
                        states[accepted_left] = states.index_select(0, accepted_right)
                        states[accepted_right] = left_states
                        left_energies = target_energies.index_select(0, accepted_left).clone()
                        target_energies[accepted_left] = target_energies.index_select(
                            0, accepted_right
                        )
                        target_energies[accepted_right] = left_energies
                        if component_energies is not None:
                            left_component = component_energies.index_select(
                                0, accepted_left
                            ).clone()
                            component_energies[accepted_left] = component_energies.index_select(
                                0, accepted_right
                            )
                            component_energies[accepted_right] = left_component
                        left_walkers = walker_at_slot.index_select(0, accepted_left).clone()
                        walker_at_slot[accepted_left] = walker_at_slot.index_select(
                            0, accepted_right
                        )
                        walker_at_slot[accepted_right] = left_walkers
                    edge_indices = torch.arange(
                        parity,
                        ladder_size - 1,
                        2,
                        dtype=torch.long,
                        device=device,
                    )
                    exchange_attempts[edge_indices] += walker_count
                    accepted_by_edge = accepted.view(walker_count, -1).sum(dim=0)
                    exchange_accepts[edge_indices] += accepted_by_edge
                slot_walkers = walker_at_slot.view(walker_count, ladder_size)
                visited_hot[slot_walkers[:, 0]] = True
                visited_cold[slot_walkers[:, -1]] = True

            if hamiltonian_ladder and sweep % hamiltonian_tempering_exchange_every == 0:
                if component_energies is None or hamiltonian_slot_weights is None:
                    raise RuntimeError("Hamiltonian exchange requires component energies")
                parity = ((sweep // hamiltonian_tempering_exchange_every) - 1) % 2
                slot_grid = torch.arange(replicas, device=device).view(walker_count, ladder_size)
                left = slot_grid[:, parity : ladder_size - 1 : 2].reshape(-1)
                right = left + 1
                if left.numel():
                    left_weight = hamiltonian_slot_weights.index_select(0, left)
                    right_weight = hamiltonian_slot_weights.index_select(0, right)
                    left_target = target_energies.index_select(0, left)
                    right_target = target_energies.index_select(0, right)
                    left_component = component_energies.index_select(0, left)
                    right_component = component_energies.index_select(0, right)
                    log_acceptance = _hamiltonian_exchange_log_acceptance(
                        beta,
                        left_target,
                        right_target,
                        left_component,
                        right_component,
                        left_weight,
                        right_weight,
                        hamiltonian_target_weight,
                    )
                    random_values = torch.rand(
                        log_acceptance.shape,
                        dtype=coefficient_dtype,
                        device=device,
                        generator=generator,
                    )
                    rng_recorder.record("hamiltonian_tempering_exchange_uniform", left.numel())
                    accepted = torch.log(random_values) < torch.minimum(
                        torch.zeros_like(log_acceptance), log_acceptance
                    )
                    accepted_left = left[accepted]
                    accepted_right = right[accepted]
                    if accepted_left.numel():
                        left_states = states.index_select(0, accepted_left).clone()
                        states[accepted_left] = states.index_select(0, accepted_right)
                        states[accepted_right] = left_states
                        left_target_values = target_energies.index_select(0, accepted_left).clone()
                        target_energies[accepted_left] = target_energies.index_select(
                            0, accepted_right
                        )
                        target_energies[accepted_right] = left_target_values
                        left_component_values = component_energies.index_select(
                            0, accepted_left
                        ).clone()
                        component_energies[accepted_left] = component_energies.index_select(
                            0, accepted_right
                        )
                        component_energies[accepted_right] = left_component_values
                        left_walkers = walker_at_slot.index_select(0, accepted_left).clone()
                        walker_at_slot[accepted_left] = walker_at_slot.index_select(
                            0, accepted_right
                        )
                        walker_at_slot[accepted_right] = left_walkers
                    edge_indices = torch.arange(
                        parity,
                        ladder_size - 1,
                        2,
                        dtype=torch.long,
                        device=device,
                    )
                    hamiltonian_exchange_attempts[edge_indices] += walker_count
                    hamiltonian_exchange_accepts[edge_indices] += accepted.view(
                        walker_count, -1
                    ).sum(dim=0)
                slot_walkers = walker_at_slot.view(walker_count, ladder_size)
                visited_hot[slot_walkers[:, 0]] = True
                visited_cold[slot_walkers[:, -1]] = True

            ordered_states, ordered_target_energies = _walker_order(
                states, target_energies, walker_at_slot
            )
            ordered_component_energies = ordered_values(component_energies)
            update_first_passage(sweep, ordered_target_energies, ordered_component_energies)
            improved = ordered_target_energies < best_energies
            best_energies = torch.where(improved, ordered_target_energies, best_energies)
            best_states = torch.where(improved[:, None], ordered_states, best_states)

            succeeded = None
            if success is not None:
                succeeded = success(ordered_states)
                if succeeded.shape != (replicas,) or succeeded.dtype != torch.bool:
                    raise ValueError("success must return a boolean tensor with shape [replicas]")
                if on_predicate_evaluation is not None:
                    on_predicate_evaluation(
                        ordered_states,
                        ordered_target_energies,
                        succeeded,
                        sweep,
                    )
                if success_energy_upper_bound is not None:
                    succeeded = succeeded & (
                        ordered_target_energies <= float(success_energy_upper_bound)
                    )
                endpoint = success_weight_mask(common_weight)
                if endpoint is not None:
                    succeeded = succeeded & endpoint
                new_hits = succeeded & (first_hit < 0)
                first_hit[new_hits] = sweep
                if on_first_hit is not None:
                    on_first_hit(ordered_states, ordered_target_energies, new_hits)
                if on_first_hit_event is not None:
                    on_first_hit_event(
                        ordered_states,
                        ordered_target_energies,
                        new_hits,
                        sweep,
                    )
                final_succeeded = succeeded

            if sweep == 1 or sweep % record_every == 0 or sweep in explicit_record_sweeps:
                emit_trace(sweep, beta, common_weight, succeeded)
            if stop_when_all_succeeded and success is not None and bool(torch.all(first_hit >= 0)):
                break
        if stop_when_all_succeeded and success is not None and bool(torch.all(first_hit >= 0)):
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not traces or traces[-1].sweep != sweep:
        emit_trace(sweep, float(betas[-1]), common_weight, None)

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    rng_audits = (
        *tuple(external_rng_audits),
        rng_recorder.finish(),
        *(
            (separate_exact_block_rng_recorder.finish(),)
            if separate_exact_block_rng_recorder is not None
            else ()
        ),
        *((junction_tree_rng_recorder.finish(),) if junction_tree_rng_recorder is not None else ()),
        *((hastings_rng_recorder.finish(),) if hastings_rng_recorder is not None else ()),
        *(
            (codeword_macro_rng_recorder.finish(),)
            if codeword_macro_rng_recorder is not None
            else ()
        ),
        *((counter_rng_recorder.finish(),) if counter_rng_recorder is not None else ()),
    )
    return AnnealResult(
        final_states=ordered_states.cpu(),
        best_states=best_states.cpu(),
        best_energies=best_energies.cpu(),
        first_hit_sweeps=first_hit.cpu(),
        trace=tuple(traces),
        kernel_seconds=elapsed,
        total_sweeps=sweep,
        seed=int(seed),
        device=str(device),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        final_energies=ordered_target_energies.cpu(),
        final_success_mask=(None if final_succeeded is None else final_succeeded.cpu()),
        local_gibbs_node_updates=(
            replicas * model.size * (sweep // gibbs_sweep_every) if gibbs_sweep_every > 0 else 0
        ),
        beta_stage_sweeps=resolved_beta_stage_sweeps,
        field_backend=field_backend,
        gibbs_kernel=gibbs_kernel,
        rng_protocol=(
            "triton_philox_counter_v1" if counter_rng_recorder is not None else "pytorch_generator"
        ),
        fused_gibbs_calls=fused_gibbs_calls,
        energy_tracking=energy_tracking,
        backend_storage_bytes=backend_storage_bytes,
        backend_prepare_seconds=backend_prepare_seconds,
        metropolis_block_count=(0 if metropolis_blocks is None else len(metropolis_blocks)),
        metropolis_block_color_count=len(block_plans),
        metropolis_block_every=metropolis_block_every,
        metropolis_block_rule=metropolis_block_rule,
        exact_block_count=(0 if exact_blocks is None else len(exact_blocks)),
        exact_block_color_count=len(exact_block_plans),
        exact_block_max_size=(
            0 if exact_blocks is None else max(len(block) for block in exact_blocks)
        ),
        exact_block_every=exact_block_every,
        exact_block_color_mode=exact_block_color_mode,
        exact_block_update_events=exact_block_update_events,
        exact_block_candidate_evaluations=exact_block_candidate_evaluations,
        exact_block_plan_group_count=sum(
            len(color_plan.groups) for color_plan in exact_block_plans
        ),
        exact_block_field_evaluation_calls=exact_block_field_evaluation_calls,
        exact_block_field_node_evaluations=exact_block_field_node_evaluations,
        exact_block_workspace_bytes=(
            0 if not exact_block_plans else int(exact_block_workspace_mib * 1024 * 1024)
        ),
        exact_block_rng_stream=(
            "exact_blocks" if separate_exact_block_rng_recorder is not None else "sampler"
        ),
        junction_tree_node_count=(
            max((len(plan.nodes) for plan in junction_tree_plan_cycle), default=0)
        ),
        junction_tree_boundary_count=(
            max((len(plan.boundary_nodes) for plan in junction_tree_plan_cycle), default=0)
        ),
        junction_tree_treewidth_upper_bound=(
            max(
                (plan.treewidth_upper_bound for plan in junction_tree_plan_cycle),
                default=0,
            )
        ),
        junction_tree_conditional_table_states_per_replica=(
            max(
                (plan.conditional_table_states_per_replica for plan in junction_tree_plan_cycle),
                default=0,
            )
        ),
        junction_tree_every=junction_tree_every,
        junction_tree_start_sweep=junction_tree_start_sweep,
        junction_tree_stop_sweep=junction_tree_stop_sweep,
        junction_tree_update_events=junction_tree_update_events,
        junction_tree_random_draws=junction_tree_random_draws,
        junction_tree_maximum_forward_table_states_per_replica=(
            junction_tree_maximum_forward_states
        ),
        junction_tree_replica_chunk_size=junction_tree_replica_chunk_size,
        junction_tree_chunk_count_per_event=junction_tree_chunk_count_per_event,
        junction_tree_rng_stream=(None if junction_tree_rng_recorder is None else "junction_tree"),
        junction_tree_plan_count=len(junction_tree_plan_cycle),
        junction_tree_plan_node_counts=tuple(len(plan.nodes) for plan in junction_tree_plan_cycle),
        junction_tree_plan_boundary_counts=tuple(
            len(plan.boundary_nodes) for plan in junction_tree_plan_cycle
        ),
        junction_tree_plan_treewidth_upper_bounds=tuple(
            plan.treewidth_upper_bound for plan in junction_tree_plan_cycle
        ),
        junction_tree_plan_conditional_table_states_per_replica=tuple(
            plan.conditional_table_states_per_replica for plan in junction_tree_plan_cycle
        ),
        junction_tree_plan_update_events=tuple(junction_tree_plan_update_events),
        junction_tree_explicit_event_sweeps=explicit_junction_tree_sweeps,
        junction_tree_explicit_event_plan_indices=explicit_junction_tree_plan_indices,
        junction_tree_explicit_event_replica_ranges=explicit_junction_tree_replica_ranges,
        junction_tree_beta_override=(
            None if junction_tree_beta_override is None else float(junction_tree_beta_override)
        ),
        junction_tree_explicit_event_beta_overrides=(explicit_junction_tree_beta_overrides),
        treewidth_corridor_count=(0 if treewidth_corridors is None else len(treewidth_corridors)),
        treewidth_corridor_color_count=len(treewidth_corridor_plans),
        treewidth_corridor_max_nodes=max(
            (
                corridor.plan.node_count
                for color_plan in treewidth_corridor_plans
                for corridor in color_plan.corridors
            ),
            default=0,
        ),
        treewidth_corridor_max_treewidth=max(
            (
                corridor.plan.treewidth_upper_bound
                for color_plan in treewidth_corridor_plans
                for corridor in color_plan.corridors
            ),
            default=0,
        ),
        treewidth_corridor_every=treewidth_corridor_every,
        treewidth_corridor_color_mode=treewidth_corridor_color_mode,
        treewidth_corridor_message_mode=treewidth_corridor_message_mode,
        treewidth_corridor_update_events=treewidth_corridor_update_events,
        treewidth_corridor_gate_update_events=treewidth_corridor_gate_update_events,
        treewidth_corridor_candidate_evaluations=(treewidth_corridor_candidate_evaluations),
        treewidth_corridor_field_evaluation_calls=(treewidth_corridor_field_evaluation_calls),
        treewidth_corridor_field_node_evaluations=(treewidth_corridor_field_node_evaluations),
        hastings_block_count=len(hastings_node_tensors),
        hastings_every=hastings_every,
        hastings_block_mode=hastings_block_mode,
        hastings_proposal_logit_clip=(
            float(hastings_proposal_logit_clip) if hastings_node_tensors else None
        ),
        hastings_update_events=hastings_update_events,
        hastings_proposals=hastings_proposals,
        hastings_accepts=hastings_accepts,
        hastings_proposed_changes=hastings_proposed_changes,
        hastings_accepted_changes=hastings_accepted_changes,
        codeword_macro_move_count=(
            0 if prepared_codeword_macro is None else prepared_codeword_macro.move_count
        ),
        codeword_macro_move_names=(
            () if prepared_codeword_macro is None else prepared_codeword_macro.move_names
        ),
        codeword_macro_plan_kind=(
            None if prepared_codeword_macro is None else prepared_codeword_macro.plan_kind
        ),
        codeword_macro_support=(
            None if prepared_codeword_macro is None else prepared_codeword_macro.support_name
        ),
        codeword_macro_rule=codeword_macro_rule,
        codeword_macro_uniform_mixture=float(codeword_macro_uniform_mixture),
        codeword_macro_every=codeword_macro_every,
        codeword_macro_start_sweep=codeword_macro_start_sweep,
        codeword_macro_stop_sweep=codeword_macro_stop_sweep,
        codeword_macro_update_events=codeword_macro_update_events,
        codeword_macro_proposals=codeword_macro_proposals,
        codeword_macro_accepts=codeword_macro_accepts,
        codeword_macro_candidate_evaluations=codeword_macro_candidate_evaluations,
        codeword_macro_downhill_proposals=codeword_macro_downhill_proposals,
        codeword_macro_downhill_accepts=codeword_macro_downhill_accepts,
        codeword_macro_selected_move_counts=(
            ()
            if codeword_macro_selected_move_counts is None
            else tuple(int(value) for value in codeword_macro_selected_move_counts.cpu().tolist())
        ),
        codeword_macro_accepted_move_counts=(
            ()
            if codeword_macro_accepted_move_counts is None
            else tuple(int(value) for value in codeword_macro_accepted_move_counts.cpu().tolist())
        ),
        codeword_macro_storage_bytes=(
            0 if prepared_codeword_macro is None else prepared_codeword_macro.storage_bytes
        ),
        local_field_proposal_clip=(
            None if local_field_proposal_clip is None else float(local_field_proposal_clip)
        ),
        gibbs_uniform_dtype=(
            "coefficient" if gibbs_uniform_dtype is None else str(gibbs_uniform_dtype)
        ),
        local_field_clip_proposals=local_field_clip_proposals,
        local_field_clip_accepts=local_field_clip_accepts,
        local_field_clip_proposed_flips=local_field_clip_proposed_flips,
        local_field_clip_accepted_flips=local_field_clip_accepted_flips,
        gibbs_sweep_every=gibbs_sweep_every,
        restart_every_sweeps=restart_every_sweeps,
        restart_count=restart_count,
        parallel_tempering_beta_ladder=tempering_ladder,
        parallel_tempering_exchange_every=parallel_tempering_exchange_every,
        parallel_tempering_exchange_attempts=tuple(
            int(value) for value in exchange_attempts.cpu().tolist()
        ),
        parallel_tempering_exchange_accepts=tuple(
            int(value) for value in exchange_accepts.cpu().tolist()
        ),
        parallel_tempering_walkers_visited_both_endpoints=int(
            torch.count_nonzero(visited_hot & visited_cold).item() if tempering_ladder else 0
        ),
        hamiltonian_component_name=(
            hamiltonian_component_name if hamiltonian_component is not None else None
        ),
        hamiltonian_target_weight=(
            float(hamiltonian_target_weight) if hamiltonian_component is not None else None
        ),
        hamiltonian_weight_schedule=weight_schedule,
        hamiltonian_tempering_weights=hamiltonian_ladder,
        hamiltonian_tempering_target_slot=hamiltonian_target_slot,
        hamiltonian_success_policy=hamiltonian_success_policy,
        hamiltonian_tempering_exchange_every=hamiltonian_tempering_exchange_every,
        hamiltonian_tempering_exchange_attempts=tuple(
            int(value) for value in hamiltonian_exchange_attempts.cpu().tolist()
        ),
        hamiltonian_tempering_exchange_accepts=tuple(
            int(value) for value in hamiltonian_exchange_accepts.cpu().tolist()
        ),
        hamiltonian_tempering_walkers_visited_both_endpoints=int(
            torch.count_nonzero(visited_hot & visited_cold).item() if hamiltonian_ladder else 0
        ),
        first_base_zero_sweeps=(
            first_base_zero.cpu() if hamiltonian_component is not None else None
        ),
        first_component_zero_sweeps=(
            first_component_zero.cpu() if hamiltonian_component is not None else None
        ),
        first_component_nonzero_sweeps=(
            first_component_nonzero.cpu() if hamiltonian_component is not None else None
        ),
        first_component_rezero_sweeps=(
            first_component_rezero.cpu() if hamiltonian_component is not None else None
        ),
        first_joint_zero_sweeps=(
            first_joint_zero.cpu() if hamiltonian_component is not None else None
        ),
        independent_units=walker_count,
        dependency_unit=(
            "parallel_tempering_ladder"
            if tempering_ladder
            else ("hamiltonian_tempering_ladder" if hamiltonian_ladder else "independent_chain")
        ),
        rng_audits=rng_audits,
    )
