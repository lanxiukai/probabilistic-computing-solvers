"""Exact binary Ising conditionals by bucket elimination and backward sampling."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .model import IsingModel
from .rng import RngAuditRecorder


@dataclass(frozen=True)
class JunctionTreeElimination:
    """One eliminated variable and its filled active-neighbor separator."""

    variable: int
    separator: tuple[int, ...]


@dataclass(frozen=True)
class JunctionTreePlan:
    """Prepared elimination geometry for one exact conditional block."""

    nodes: tuple[int, ...]
    boundary_nodes: tuple[int, ...]
    eliminations: tuple[JunctionTreeElimination, ...]
    treewidth_upper_bound: int
    conditional_table_states_per_replica: int


@dataclass(frozen=True)
class JunctionTreeSample:
    """Updated states and exact target-energy deltas."""

    states: torch.Tensor
    energy_delta: torch.Tensor
    conditional_log_partition: torch.Tensor
    random_draws: int
    maximum_forward_table_states_per_replica: int
    replica_chunk_size: int
    chunk_count: int


@dataclass(frozen=True)
class JunctionTreeConditionalEvaluation:
    """Exact conditional log partitions computed without sampling or state updates."""

    conditional_log_partition: torch.Tensor
    maximum_forward_table_states_per_replica: int
    replica_chunk_size: int
    chunk_count: int


@dataclass(frozen=True)
class JunctionTreeMinimum:
    """Constructed conditional minimizers; these are not sampled states or native hits."""

    minimizing_states: torch.Tensor
    minimum_energy_delta: torch.Tensor
    maximum_forward_table_states_per_replica: int
    replica_chunk_size: int
    chunk_count: int


@dataclass(frozen=True)
class JunctionTreePayloadMobility:
    """Exact probability that at least one fixed payload spin changes."""

    full_conditional_log_partition: torch.Tensor
    free_conditional_log_partition: torch.Tensor
    full_log_stay_probability: torch.Tensor
    full_stay_probability: torch.Tensor
    full_escape_probability: torch.Tensor
    log_payload_stay_probability: torch.Tensor
    payload_stay_probability: torch.Tensor
    payload_escape_probability: torch.Tensor
    full_maximum_forward_table_states_per_replica: int
    free_maximum_forward_table_states_per_replica: int
    replica_chunk_size: int
    chunk_count: int


@dataclass
class _LogFactor:
    scope: tuple[int, ...]
    values: torch.Tensor


@dataclass
class _BackwardTable:
    variable: int
    separator: tuple[int, ...]
    log_odds: torch.Tensor


def minimum_degree_elimination_order(
    coupling: torch.Tensor,
    nodes: Sequence[int],
) -> tuple[tuple[int, ...], int]:
    """Return a deterministic min-degree-with-fill order and its width upper bound."""

    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be square")
    selected = tuple(int(node) for node in nodes)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("nodes must be a non-empty set of unique indices")
    if any(node < 0 or node >= coupling.shape[0] for node in selected):
        raise IndexError("junction-tree node is out of range")
    selected_set = set(selected)
    graph = {node: set() for node in selected}
    rows, columns = torch.where(torch.triu(coupling != 0, diagonal=1))
    for left, right in zip(rows.tolist(), columns.tolist(), strict=True):
        if left in selected_set and right in selected_set:
            graph[left].add(right)
            graph[right].add(left)
    heap = [(len(neighbors), node) for node, neighbors in graph.items()]
    heapq.heapify(heap)
    order: list[int] = []
    width = 0
    while graph:
        while True:
            degree, node = heapq.heappop(heap)
            if node in graph and degree == len(graph[node]):
                break
        neighbors = sorted(graph[node])
        width = max(width, degree)
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
        for neighbor in neighbors:
            graph[neighbor].discard(node)
        del graph[node]
        affected = set(neighbors)
        for neighbor in neighbors:
            affected.update(graph[neighbor])
        for affected_node in affected:
            if affected_node in graph:
                heapq.heappush(heap, (len(graph[affected_node]), affected_node))
        order.append(node)
    return tuple(order), width


def prepare_junction_tree(
    coupling: torch.Tensor,
    nodes: Sequence[int],
    elimination_order: Sequence[int],
    *,
    maximum_treewidth: int,
) -> JunctionTreePlan:
    """Validate an elimination order and prepare exact conditional separators."""

    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be square")
    selected = tuple(int(node) for node in nodes)
    order = tuple(int(node) for node in elimination_order)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("nodes must be a non-empty set of unique indices")
    if set(order) != set(selected) or len(order) != len(selected):
        raise ValueError("elimination_order must contain every selected node exactly once")
    if maximum_treewidth < 0:
        raise ValueError("maximum_treewidth must be nonnegative")
    if any(node < 0 or node >= coupling.shape[0] for node in selected):
        raise IndexError("junction-tree node is out of range")

    selected_set = set(selected)
    boundary = tuple(index for index in range(coupling.shape[0]) if index not in selected_set)
    graph = {node: set() for node in selected}
    rows, columns = torch.where(torch.triu(coupling != 0, diagonal=1))
    for left, right in zip(rows.tolist(), columns.tolist(), strict=True):
        if left in selected_set and right in selected_set:
            graph[left].add(right)
            graph[right].add(left)
    position = {node: index for index, node in enumerate(order)}
    eliminations: list[JunctionTreeElimination] = []
    width = 0
    conditional_states = 0
    for variable in order:
        separator = tuple(sorted(graph[variable], key=position.__getitem__))
        if any(position[node] <= position[variable] for node in separator):
            raise RuntimeError("elimination graph retained an already eliminated node")
        width = max(width, len(separator))
        conditional_states += 1 << len(separator)
        neighbors = list(separator)
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
        for neighbor in neighbors:
            graph[neighbor].discard(variable)
        del graph[variable]
        eliminations.append(JunctionTreeElimination(variable=variable, separator=separator))
    if width > maximum_treewidth:
        raise ValueError(
            f"Junction-tree width {width} exceeds maximum_treewidth={maximum_treewidth}"
        )
    return JunctionTreePlan(
        nodes=selected,
        boundary_nodes=boundary,
        eliminations=tuple(eliminations),
        treewidth_upper_bound=width,
        conditional_table_states_per_replica=conditional_states,
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


def _expanded_values(
    factor: _LogFactor,
    union_scope: tuple[int, ...],
) -> torch.Tensor:
    scope_positions = [union_scope.index(variable) for variable in factor.scope]
    if scope_positions != sorted(scope_positions):
        raise RuntimeError("Factor scope order is incompatible with the elimination scope")
    shape = [factor.values.shape[0]] + [
        2 if variable in factor.scope else 1 for variable in union_scope
    ]
    return factor.values.reshape(shape)


def _junction_tree_forward_elimination(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    beta_values: torch.Tensor,
    *,
    reduction: str = "logsumexp",
) -> tuple[list[_BackwardTable], torch.Tensor, int]:
    """Build branch differences and sum-product or max-sum aggregate scores.

    With max reduction, log_odds stores the difference of optimal branch scores
    instead of marginal log odds. Sampling callers retain the logsumexp default.
    """

    if reduction not in {"logsumexp", "max"}:
        raise ValueError("Unknown junction-tree reduction")
    replicas = states.shape[0]
    dtype = model.coupling.dtype
    device = states.device
    order = tuple(step.variable for step in plan.eliminations)
    position = {node: index for index, node in enumerate(order)}
    node_tensor = torch.tensor(plan.nodes, dtype=torch.long, device=device)
    boundary_tensor = torch.tensor(plan.boundary_nodes, dtype=torch.long, device=device)
    coupling = model.coupling
    effective_bias = model.bias.index_select(0, node_tensor).expand(replicas, -1).clone()
    if plan.boundary_nodes:
        boundary_coupling = coupling.index_select(0, boundary_tensor).index_select(1, node_tensor)
        effective_bias.add_(
            states.index_select(1, boundary_tensor).to(dtype=dtype) @ boundary_coupling
        )
    node_position = {node: index for index, node in enumerate(plan.nodes)}
    spin_values = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
    buckets: list[list[_LogFactor]] = [[] for _ in order]
    for node in plan.nodes:
        local_index = node_position[node]
        values = beta_values[:, None] * effective_bias[:, local_index, None] * spin_values[None, :]
        buckets[position[node]].append(_LogFactor(scope=(node,), values=values))
    auxiliary_coupling = coupling.index_select(0, node_tensor).index_select(1, node_tensor)
    rows, columns = torch.where(torch.triu(auxiliary_coupling != 0, diagonal=1))
    pair_spins = spin_values[:, None] * spin_values[None, :]
    for local_left, local_right in zip(rows.tolist(), columns.tolist(), strict=True):
        left = plan.nodes[local_left]
        right = plan.nodes[local_right]
        scope = tuple(sorted((left, right), key=position.__getitem__))
        coefficient = auxiliary_coupling[local_left, local_right]
        values = beta_values[:, None, None] * coefficient * pair_spins[None, :, :]
        buckets[min(position[left], position[right])].append(_LogFactor(scope=scope, values=values))

    backward: list[_BackwardTable] = []
    maximum_forward_states = 0
    conditional_log_partition = torch.zeros(replicas, dtype=dtype, device=device)
    for step_index, elimination in enumerate(plan.eliminations):
        factors = buckets[step_index]
        if not factors:
            raise RuntimeError("An elimination bucket unexpectedly contains no factors")
        union_scope = (elimination.variable, *elimination.separator)
        if any(not set(factor.scope).issubset(union_scope) for factor in factors):
            raise RuntimeError("An elimination bucket contains an out-of-scope factor")
        combined = (
            _expanded_values(factors[0], union_scope)
            .expand(replicas, *(2 for _ in union_scope))
            .clone()
        )
        for factor in factors[1:]:
            combined.add_(_expanded_values(factor, union_scope))
        maximum_forward_states = max(
            maximum_forward_states,
            1 << len(union_scope),
        )
        log_odds = (combined[:, 1] - combined[:, 0]).contiguous()
        backward.append(
            _BackwardTable(
                variable=elimination.variable,
                separator=elimination.separator,
                log_odds=log_odds,
            )
        )
        message = (
            torch.logsumexp(combined, dim=1) if reduction == "logsumexp" else combined.amax(dim=1)
        )
        buckets[step_index] = []
        if elimination.separator:
            next_index = min(position[node] for node in elimination.separator)
            buckets[next_index].append(_LogFactor(scope=elimination.separator, values=message))
        else:
            conditional_log_partition.add_(message)
    return backward, conditional_log_partition, maximum_forward_states


def _selected_branch_difference(table: _BackwardTable, states: torch.Tensor) -> torch.Tensor:
    """Look up a backward branch difference using the already assigned separator."""

    flattened = table.log_odds.reshape(states.shape[0], -1)
    if not table.separator:
        return flattened[:, 0]
    separator = torch.tensor(table.separator, dtype=torch.long, device=states.device)
    bits = states.index_select(1, separator).gt(0).to(torch.long)
    weights = 1 << torch.arange(
        len(table.separator) - 1, -1, -1, dtype=torch.long, device=states.device
    )
    codes = torch.sum(bits * weights[None, :], dim=1)
    return flattened.gather(1, codes[:, None]).squeeze(1)


def _sample_junction_tree_batch(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    beta_values: torch.Tensor,
    *,
    generator: torch.Generator | None,
    uniforms_by_variable: dict[int, torch.Tensor] | None,
    rng_recorder: RngAuditRecorder | None = None,
) -> JunctionTreeSample:
    """Draw one replica batch, optionally consuming caller-provided uniforms."""

    replicas = states.shape[0]
    dtype = model.coupling.dtype
    device = states.device
    backward, conditional_log_partition, maximum_forward_states = (
        _junction_tree_forward_elimination(model, plan, states, beta_values)
    )

    old_energy = model.energy(states)
    updated = states.clone()
    random_draws = 0
    for table in reversed(backward):
        selected_log_odds = _selected_branch_difference(table, updated)
        if uniforms_by_variable is None:
            if generator is None:
                raise RuntimeError("Junction-tree sampling requires uniforms or a generator")
            uniforms = torch.rand(
                replicas,
                dtype=dtype,
                device=device,
                generator=generator,
            )
        else:
            uniforms = uniforms_by_variable[table.variable]
            if uniforms.shape != (replicas,):
                raise ValueError("Pre-generated junction-tree uniforms have the wrong shape")
        random_draws += replicas
        if rng_recorder is not None:
            rng_recorder.record("junction_tree_backward_uniform", replicas)
        plus = uniforms < torch.sigmoid(selected_log_odds)
        updated[:, table.variable] = plus.to(torch.int8).mul(2).sub(1)
    new_energy = model.energy(updated)
    return JunctionTreeSample(
        states=updated,
        energy_delta=new_energy - old_energy,
        conditional_log_partition=conditional_log_partition,
        random_draws=random_draws,
        maximum_forward_table_states_per_replica=maximum_forward_states,
        replica_chunk_size=replicas,
        chunk_count=1,
    )


def evaluate_junction_tree_conditional_log_partition(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    replica_chunk_size: int | None = None,
) -> JunctionTreeConditionalEvaluation:
    """Evaluate exact block-conditional log partitions without drawing samples."""

    if states.ndim != 2 or states.shape[1] != model.size:
        raise ValueError("states must have shape [replicas, model.size]")
    if torch.any((states != -1) & (states != 1)):
        raise ValueError("states must contain only -1 and +1")
    if model.coupling.device != states.device:
        raise ValueError("model and states must share a device")
    replicas = states.shape[0]
    if replicas <= 0:
        raise ValueError("states must contain at least one replica")
    beta_values = _beta_vector(
        beta,
        replicas=replicas,
        device=states.device,
        dtype=model.coupling.dtype,
    )
    if torch.any(~torch.isfinite(beta_values)) or torch.any(beta_values < 0):
        raise ValueError("beta must be finite and nonnegative")
    chunk_size = replicas if replica_chunk_size is None else int(replica_chunk_size)
    if chunk_size <= 0:
        raise ValueError("replica_chunk_size must be positive")
    chunk_size = min(chunk_size, replicas)

    partitions: list[torch.Tensor] = []
    maximum_forward_states = 0
    for start in range(0, replicas, chunk_size):
        stop = min(start + chunk_size, replicas)
        _, partition, forward_states = _junction_tree_forward_elimination(
            model,
            plan,
            states[start:stop],
            beta_values[start:stop],
        )
        partitions.append(partition)
        maximum_forward_states = max(maximum_forward_states, forward_states)
    return JunctionTreeConditionalEvaluation(
        conditional_log_partition=torch.cat(partitions, dim=0),
        maximum_forward_table_states_per_replica=maximum_forward_states,
        replica_chunk_size=chunk_size,
        chunk_count=len(partitions),
    )


def _conditional_log_weight(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    beta: float | torch.Tensor,
) -> torch.Tensor:
    """Return the selected-spin log weight with boundary-only terms removed."""

    replicas = states.shape[0]
    dtype = model.coupling.dtype
    device = states.device
    beta_values = _beta_vector(
        beta,
        replicas=replicas,
        device=device,
        dtype=dtype,
    )
    nodes = torch.tensor(plan.nodes, dtype=torch.long, device=device)
    boundary = torch.tensor(plan.boundary_nodes, dtype=torch.long, device=device)
    selected = states.index_select(1, nodes).to(dtype=dtype)
    effective_bias = model.bias.index_select(0, nodes).expand(replicas, -1).clone()
    if plan.boundary_nodes:
        effective_bias.add_(
            states.index_select(1, boundary).to(dtype=dtype)
            @ model.coupling.index_select(0, boundary).index_select(1, nodes)
        )
    internal = model.coupling.index_select(0, nodes).index_select(1, nodes)
    potential = torch.sum(selected * effective_bias, dim=1)
    potential.add_(0.5 * torch.einsum("bi,ij,bj->b", selected, internal, selected))
    return beta_values * potential


@torch.inference_mode()
def minimize_junction_tree_conditional(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    *,
    replica_chunk_size: int | None = None,
) -> JunctionTreeMinimum:
    """Minimize the original Hamiltonian with every exterior spin held fixed.

    Reuses the prepared bucket graph, unary/boundary factors and pair accounting.
    Max-sum scores use beta=1; the minimum is independent of a sampling temperature.
    Backtracking chooses -1 on exact ties and never draws random numbers.
    """

    if states.ndim != 2 or states.shape[1] != model.size or states.shape[0] == 0:
        raise ValueError("states must have nonempty shape [replicas, model.size]")
    if not bool(torch.all((states == -1) | (states == 1))):
        raise ValueError("states must be bipolar")
    if model.coupling.device != states.device:
        raise ValueError("model and states must share a device")
    replicas = states.shape[0]
    chunk_size = replicas if replica_chunk_size is None else int(replica_chunk_size)
    if chunk_size <= 0:
        raise ValueError("replica_chunk_size must be positive")
    chunk_size = min(chunk_size, replicas)
    minima, deltas = [], []
    maximum_forward_states = 0
    for start in range(0, replicas, chunk_size):
        selected = states[start : start + chunk_size]
        beta = torch.ones(len(selected), dtype=model.coupling.dtype, device=states.device)
        backward, score, forward_states = _junction_tree_forward_elimination(
            model, plan, selected, beta, reduction="max"
        )
        updated = selected.clone()
        for table in reversed(backward):
            plus = _selected_branch_difference(table, updated) > 0
            updated[:, table.variable] = plus.to(updated.dtype).mul(2).sub(1)
        minima.append(updated)
        deltas.append(_conditional_log_weight(model, plan, selected, 1.0) - score)
        maximum_forward_states = max(maximum_forward_states, forward_states)
    return JunctionTreeMinimum(
        minimizing_states=torch.cat(minima),
        minimum_energy_delta=torch.cat(deltas),
        maximum_forward_table_states_per_replica=maximum_forward_states,
        replica_chunk_size=chunk_size,
        chunk_count=len(minima),
    )


def evaluate_junction_tree_payload_mobility(
    model: IsingModel,
    full_plan: JunctionTreePlan,
    free_plan: JunctionTreePlan,
    states: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    replica_chunk_size: int | None = None,
) -> JunctionTreePayloadMobility:
    """Evaluate the exact probability that a payload changes under a full-block draw.

    ``free_plan.nodes`` are allowed to vary while every node in
    ``full_plan.nodes - free_plan.nodes`` is fixed to its current value. The restricted
    event mass follows

    ``log Z_free + log w_full(x) - log w_free(x) - log Z_full``.

    This identity avoids enumerating the payload and distinguishes useful payload
    movement from changes confined to a mobile seed sub-block.
    """

    full_nodes = set(full_plan.nodes)
    free_nodes = set(free_plan.nodes)
    if not free_nodes < full_nodes:
        raise ValueError("free_plan.nodes must be a proper non-empty subset of full_plan.nodes")

    full = evaluate_junction_tree_conditional_log_partition(
        model,
        full_plan,
        states,
        beta,
        replica_chunk_size=replica_chunk_size,
    )
    free = evaluate_junction_tree_conditional_log_partition(
        model,
        free_plan,
        states,
        beta,
        replica_chunk_size=replica_chunk_size,
    )
    full_log_stay = (
        _conditional_log_weight(model, full_plan, states, beta) - full.conditional_log_partition
    )
    clamped_full_log_stay = torch.minimum(
        full_log_stay,
        torch.zeros_like(full_log_stay),
    )
    full_stay = torch.exp(clamped_full_log_stay)
    full_escape = -torch.expm1(clamped_full_log_stay)
    log_stay = (
        free.conditional_log_partition
        + _conditional_log_weight(model, full_plan, states, beta)
        - _conditional_log_weight(model, free_plan, states, beta)
        - full.conditional_log_partition
    )
    clamped_log_stay = torch.minimum(log_stay, torch.zeros_like(log_stay))
    stay = torch.exp(clamped_log_stay)
    escape = -torch.expm1(clamped_log_stay)
    return JunctionTreePayloadMobility(
        full_conditional_log_partition=full.conditional_log_partition,
        free_conditional_log_partition=free.conditional_log_partition,
        full_log_stay_probability=full_log_stay,
        full_stay_probability=full_stay,
        full_escape_probability=full_escape,
        log_payload_stay_probability=log_stay,
        payload_stay_probability=stay,
        payload_escape_probability=escape,
        full_maximum_forward_table_states_per_replica=(
            full.maximum_forward_table_states_per_replica
        ),
        free_maximum_forward_table_states_per_replica=(
            free.maximum_forward_table_states_per_replica
        ),
        replica_chunk_size=full.replica_chunk_size,
        chunk_count=full.chunk_count,
    )


def sample_junction_tree_conditional(
    model: IsingModel,
    plan: JunctionTreePlan,
    states: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
    replica_chunk_size: int | None = None,
) -> JunctionTreeSample:
    """Draw the exact block conditional while preserving every boundary spin.

    A bounded ``replica_chunk_size`` pre-generates the same variable-major uniform
    stream as the unchunked sampler and then runs independent replica slices through
    variable elimination. Chunking changes table memory, not the conditional law,
    logical draw count, generator offset, or replica order.
    """

    if states.ndim != 2 or states.shape[1] != model.size:
        raise ValueError("states must have shape [replicas, model.size]")
    if torch.any((states != -1) & (states != 1)):
        raise ValueError("states must contain only -1 and +1")
    if model.coupling.device != states.device:
        raise ValueError("model and states must share a device")
    replicas = states.shape[0]
    if replicas <= 0:
        raise ValueError("states must contain at least one replica")
    dtype = model.coupling.dtype
    device = states.device
    beta_values = _beta_vector(
        beta,
        replicas=replicas,
        device=device,
        dtype=dtype,
    )
    if torch.any(~torch.isfinite(beta_values)) or torch.any(beta_values < 0):
        raise ValueError("beta must be finite and nonnegative")
    chunk_size = replicas if replica_chunk_size is None else int(replica_chunk_size)
    if chunk_size <= 0:
        raise ValueError("replica_chunk_size must be positive")
    chunk_size = min(chunk_size, replicas)
    if chunk_size == replicas:
        return _sample_junction_tree_batch(
            model,
            plan,
            states,
            beta_values,
            generator=generator,
            uniforms_by_variable=None,
            rng_recorder=rng_recorder,
        )

    uniforms_by_variable: dict[int, torch.Tensor] = {}
    for elimination in reversed(plan.eliminations):
        uniforms_by_variable[elimination.variable] = torch.rand(
            replicas,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        if rng_recorder is not None:
            rng_recorder.record("junction_tree_backward_uniform", replicas)

    samples: list[JunctionTreeSample] = []
    for start in range(0, replicas, chunk_size):
        stop = min(start + chunk_size, replicas)
        samples.append(
            _sample_junction_tree_batch(
                model,
                plan,
                states[start:stop],
                beta_values[start:stop],
                generator=None,
                uniforms_by_variable={
                    variable: uniforms[start:stop]
                    for variable, uniforms in uniforms_by_variable.items()
                },
                rng_recorder=None,
            )
        )
    return JunctionTreeSample(
        states=torch.cat([sample.states for sample in samples], dim=0),
        energy_delta=torch.cat([sample.energy_delta for sample in samples], dim=0),
        conditional_log_partition=torch.cat(
            [sample.conditional_log_partition for sample in samples], dim=0
        ),
        random_draws=sum(sample.random_draws for sample in samples),
        maximum_forward_table_states_per_replica=max(
            sample.maximum_forward_table_states_per_replica for sample in samples
        ),
        replica_chunk_size=chunk_size,
        chunk_count=len(samples),
    )
