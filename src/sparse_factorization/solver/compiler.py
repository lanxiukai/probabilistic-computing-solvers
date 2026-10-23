"""Fused and compact COPY-tree compilers for factorization netlists."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from sparse_factorization.core.edge_model import EdgeIsingModel, edge_terms
from sparse_factorization.core.model import ClampResult, IsingModel, QUBOModel, bits_to_spins

from .gates import COPY, get_gate
from .netlist import MultiplierNetlist


def _edge_count(model: IsingModel) -> int:
    if isinstance(model, EdgeIsingModel):
        return model.edge_left.numel()
    return int(torch.count_nonzero(torch.triu(model.coupling, diagonal=1)).item())


def _maximum_degree(model: IsingModel) -> int:
    if model.size == 0:
        return 0
    if isinstance(model, EdgeIsingModel):
        left, right, _ = edge_terms(model)
        return int(torch.bincount(torch.cat((left, right)), minlength=model.size).max())
    return int(torch.count_nonzero(model.coupling, dim=1).max().item())


def _decode_unsigned(spins: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    if len(indices) > 64:
        raise ValueError("Tensor integer decoding supports at most 64 bits")
    if len(indices) == 64:
        # Avoid signed sums and unsigned arithmetic kernels: decode on the host.
        selected = spins.detach().cpu()[..., list(indices)].reshape(-1, 64).tolist()
        integers = [sum(((int(s) + 1) // 2) << i for i, s in enumerate(row)) for row in selected]
        return torch.tensor(integers, dtype=torch.uint64, device=spins.device).reshape(
            spins.shape[:-1]
        )
    selected = torch.tensor(indices, dtype=torch.long, device=spins.device)
    bits = spins.index_select(-1, selected).add(1).div(2).to(torch.int64)
    weights = torch.tensor(
        [1 << index for index in range(len(indices))],
        dtype=torch.int64,
        device=spins.device,
    )
    return (bits * weights).sum(dim=-1)


def factor_pair_mask(factors: torch.Tensor, target: int) -> torch.Tensor:
    """Certify nontrivial integer products without signed overflow above 63 bits."""
    left, right = factors[..., 0], factors[..., 1]
    if (
        target < (1 << 63)
        and factors.dtype != torch.uint64
        and bool(((factors >= 0) & (factors < (1 << 32))).all())
    ):
        return (left > 1) & (right > 1) & (left * right == target)
    pairs = factors.detach().cpu().reshape(-1, 2).tolist()
    matches = [a > 1 and b > 1 and a * b == target for a, b in pairs]
    return torch.tensor(matches, dtype=torch.bool, device=factors.device).reshape(left.shape)


@dataclass(frozen=True)
class PhysicalGate:
    """One gate penalty placed on physical multiplier nodes."""

    name: str
    gate: str
    nodes: tuple[int, ...]
    column: int


@dataclass(frozen=True)
class FactorHamiltonianComponents:
    """Additive gate and unit-weight COPY Hamiltonian components."""

    gate_model: IsingModel
    copy_model: IsingModel
    gates: tuple[PhysicalGate, ...]

    def __post_init__(self) -> None:
        if self.gate_model.names != self.copy_model.names:
            raise ValueError("Factor Hamiltonian components must share one variable order")


@dataclass(frozen=True)
class CompiledMultiplier:
    """A physical Ising graph plus its logical multiplier interpretation."""

    model: IsingModel
    netlist: MultiplierNetlist
    representation: str
    logical_to_physical: Mapping[str, tuple[int, ...]]
    logical_roots: Mapping[str, int]
    copy_edges: tuple[tuple[int, int], ...] = ()
    components: FactorHamiltonianComponents | None = None

    def __post_init__(self) -> None:
        if set(self.logical_to_physical) != set(self.netlist.signal_names):
            raise ValueError("Physical mapping does not cover every logical signal")
        if set(self.logical_roots) != set(self.netlist.signal_names):
            raise ValueError("Logical roots do not cover every logical signal")

    @property
    def factor_a_indices(self) -> tuple[int, ...]:
        return tuple(self.logical_roots[signal] for signal in self.netlist.factor_a)

    @property
    def factor_b_indices(self) -> tuple[int, ...]:
        return tuple(self.logical_roots[signal] for signal in self.netlist.factor_b)

    @property
    def product_indices(self) -> tuple[int, ...]:
        return tuple(self.logical_roots[signal] for signal in self.netlist.product)

    def encode_logical_bits(self, assignment: Mapping[str, int]) -> torch.Tensor:
        """Expand a complete logical assignment to all physical COPY nodes."""

        missing = set(self.netlist.signal_names) - set(assignment)
        if missing:
            raise ValueError(f"Logical assignment is missing signals: {sorted(missing)}")
        bits = torch.empty(self.model.size, dtype=torch.int8)
        assigned = torch.zeros(self.model.size, dtype=torch.bool)
        for signal, indices in self.logical_to_physical.items():
            value = int(assignment[signal])
            if value not in (0, 1):
                raise ValueError("Logical assignments must contain only zero or one")
            index = torch.tensor(indices, dtype=torch.long)
            bits[index] = value
            assigned[index] = True
        if not bool(torch.all(assigned)):
            raise RuntimeError("Some physical multiplier nodes lack a logical signal")
        return bits

    def forward_spins(self, factor_a: int, factor_b: int) -> torch.Tensor:
        assignment = self.netlist.forward_assignment(factor_a, factor_b)
        return bits_to_spins(self.encode_logical_bits(assignment))

    def decode_factors(self, full_spins: torch.Tensor) -> torch.Tensor:
        if full_spins.shape[-1] != self.model.size:
            raise ValueError("Spin shape does not match the compiled multiplier")
        left = _decode_unsigned(full_spins, self.factor_a_indices)
        right = _decode_unsigned(full_spins, self.factor_b_indices)
        return torch.stack((left, right), dim=-1)

    def decode_product(self, full_spins: torch.Tensor) -> torch.Tensor:
        if full_spins.shape[-1] != self.model.size:
            raise ValueError("Spin shape does not match the compiled multiplier")
        return _decode_unsigned(full_spins, self.product_indices)

    def copy_violations(self, full_spins: torch.Tensor) -> torch.Tensor:
        if full_spins.shape[-1] != self.model.size:
            raise ValueError("Spin shape does not match the compiled multiplier")
        result = torch.zeros(full_spins.shape[:-1], dtype=torch.int64, device=full_spins.device)
        for left, right in self.copy_edges:
            result += (full_spins[..., left] != full_spins[..., right]).to(torch.int64)
        return result

    def clamp_logical(self, assignments: Mapping[str, int]) -> ClampResult:
        fixed: dict[int, int] = {}
        for signal, bit_value in assignments.items():
            if signal not in self.logical_to_physical:
                raise ValueError(f"Unknown multiplier signal: {signal}")
            bit = int(bit_value)
            if bit not in (0, 1):
                raise ValueError("Logical clamps must contain only zero or one")
            spin = 2 * bit - 1
            for index in self.logical_to_physical[signal]:
                fixed[index] = spin
        return self.model.clamp(fixed)


@dataclass(frozen=True)
class ConditionedFactorizer:
    """A multiplier conditioned on zero pins, product bits, and declared priors."""

    compiled: CompiledMultiplier
    clamp: ClampResult
    target: int
    fixed_logical_bits: Mapping[str, int]
    components: FactorHamiltonianComponents | None = None

    @property
    def model(self) -> IsingModel:
        return self.clamp.model

    def reconstruct(self, free_spins: torch.Tensor) -> torch.Tensor:
        return self.clamp.reconstruct(free_spins)

    def decode_factors(self, free_spins: torch.Tensor) -> torch.Tensor:
        return self.compiled.decode_factors(self.reconstruct(free_spins))

    def certificate_mask(self, free_spins: torch.Tensor) -> torch.Tensor:
        return factor_pair_mask(self.decode_factors(free_spins), self.target)


def _assemble_qubo(
    names: list[str],
    gate_nodes: list[tuple[int, ...]],
    netlist: MultiplierNetlist,
    *,
    dtype: torch.dtype,
    copy_edges: tuple[tuple[int, int], ...] = (),
    copy_weight: float = 1.0,
    metadata: Mapping[str, object],
) -> tuple[QUBOModel, QUBOModel, QUBOModel]:
    size = len(names)
    quadratic = torch.zeros((size, size), dtype=dtype)
    linear = torch.zeros(size, dtype=dtype)
    constant = 0.0
    gate_quadratic = torch.zeros_like(quadratic)
    gate_linear = torch.zeros_like(linear)
    gate_constant = 0.0
    copy_quadratic = torch.zeros_like(quadratic)
    copy_linear = torch.zeros_like(linear)
    copy_constant = 0.0

    for instance, physical in zip(netlist.gates, gate_nodes, strict=True):
        local = get_gate(instance.gate).qubo(netlist.profile, dtype=dtype)
        for local_index, global_index in enumerate(physical):
            linear[global_index] += local.linear[local_index]
            gate_linear[global_index] += local.linear[local_index]
        for left in range(local.size):
            for right in range(left + 1, local.size):
                value = local.quadratic[left, right]
                if bool(value != 0):
                    global_left = physical[left]
                    global_right = physical[right]
                    quadratic[global_left, global_right] += value
                    quadratic[global_right, global_left] += value
                    gate_quadratic[global_left, global_right] += value
                    gate_quadratic[global_right, global_left] += value
        constant += float(local.constant)
        gate_constant += float(local.constant)

    if copy_edges:
        copy = COPY.qubo(netlist.profile, dtype=dtype)
        for left, right in copy_edges:
            linear[left] += copy_weight * copy.linear[0]
            linear[right] += copy_weight * copy.linear[1]
            copy_linear[left] += copy.linear[0]
            copy_linear[right] += copy.linear[1]
            coefficient = copy_weight * copy.quadratic[0, 1]
            quadratic[left, right] += coefficient
            quadratic[right, left] += coefficient
            copy_quadratic[left, right] += copy.quadratic[0, 1]
            copy_quadratic[right, left] += copy.quadratic[0, 1]
            constant += copy_weight * float(copy.constant)
            copy_constant += float(copy.constant)

    total = QUBOModel(
        quadratic=quadratic,
        linear=linear,
        constant=constant,
        names=tuple(names),
        metadata=dict(metadata),
    )
    gate_component = QUBOModel(
        quadratic=gate_quadratic,
        linear=gate_linear,
        constant=gate_constant,
        names=tuple(names),
        metadata={**metadata, "hamiltonian_component": "gate"},
    )
    copy_component = QUBOModel(
        quadratic=copy_quadratic,
        linear=copy_linear,
        constant=copy_constant,
        names=tuple(names),
        metadata={
            **metadata,
            "hamiltonian_component": "copy",
            "copy_weight_multiplier": 1.0,
        },
    )
    return total, gate_component, copy_component


def compile_fused_multiplier(
    netlist: MultiplierNetlist,
    *,
    dtype: torch.dtype = torch.float32,
) -> CompiledMultiplier:
    """Fuse every occurrence of a logical signal into one Ising node."""

    names = list(netlist.signal_names)
    lookup = {name: index for index, name in enumerate(names)}
    gate_nodes = [tuple(lookup[signal] for signal in gate.signals) for gate in netlist.gates]
    qubo, _, _ = _assemble_qubo(
        names,
        gate_nodes,
        netlist,
        dtype=dtype,
        metadata={
            "problem": "factorization",
            "representation": "fused",
            "factor_bits": netlist.factor_bits,
            "product_bits": netlist.product_bits,
            "penalty_profile": netlist.profile,
            "nonnegative_term_compilation": True,
            "requires_zero_clamp": True,
        },
    )
    model = qubo.to_ising()
    expected_nodes = 3 * netlist.factor_bits**2 + netlist.factor_bits
    if model.size != expected_nodes:
        raise RuntimeError(f"Fused multiplier size mismatch: {model.size} != {expected_nodes}")
    metadata = {
        **model.metadata,
        "edge_count": _edge_count(model),
        "maximum_degree": _maximum_degree(model),
    }
    model = IsingModel(
        model.coupling,
        model.bias,
        model.constant,
        model.names,
        metadata,
    )
    groups = {name: (index,) for index, name in enumerate(names)}
    roots = {name: index for index, name in enumerate(names)}
    return CompiledMultiplier(
        model=model,
        netlist=netlist,
        representation="fused",
        logical_to_physical=groups,
        logical_roots=roots,
    )


def _copy_tree_extra_nodes(leaves: int, maximum_degree: int) -> int:
    count = 0
    current = leaves
    while current > 1:
        current = math.ceil(current / (maximum_degree - 1))
        count += current
    return count


def _inplace_balanced_copy_tree(
    leaves: list[int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Build an ordered balanced binary tree using only existing gate ports."""

    if not leaves:
        raise ValueError("An in-place COPY tree requires at least one leaf")
    edges: list[tuple[int, int]] = []

    def recurse(start: int, stop: int) -> int:
        midpoint = (start + stop - 1) // 2
        root = leaves[midpoint]
        if start < midpoint:
            edges.append((root, recurse(start, midpoint)))
        if midpoint + 1 < stop:
            edges.append((root, recurse(midpoint + 1, stop)))
        return root

    root = recurse(0, len(leaves))
    return root, tuple(edges)


def compile_compact_sparse_multiplier(
    netlist: MultiplierNetlist,
    *,
    maximum_degree: int = 5,
    copy_weight: float = 1.0,
    factor_copy_topology: str = "balanced_tree",
    dtype: torch.dtype = torch.float32,
    model_storage: str = "dense_v1",
) -> CompiledMultiplier:
    """Compile an Aadit-style compact gate-port graph with COPY trees.

    Every gate port remains a separate physical node. Point-to-point logical nets
    receive one COPY edge, while each high-fanout factor input uses either the
    count-matched ``maximum_degree - 1``-ary tree, a carry-ordered leaf chain,
    or an ordered balanced tree whose internal vertices are existing gate ports.
    Because the graph is assembled from unchanged non-negative gate penalties and
    non-negative equality penalties, any satisfiable multiplier assignment has zero
    energy and every zero-energy state satisfies all gates and COPY relations.

    The construction matches the published compact node/edge formulas and degree
    bound, but its metadata deliberately calls it ``aadit_style`` rather than a
    cycle-accurate reproduction of the unpublished implementation.
    """

    if model_storage not in {"dense_v1", "edge_v1"}:
        raise ValueError("Unknown model storage")
    if model_storage == "edge_v1" and (
        netlist.profile != "aadit" or copy_weight != 1 or factor_copy_topology != "carry_chain"
    ):
        raise ValueError("Edge compilation currently requires unit-COPY Aadit carry_chain")
    if maximum_degree < 5:
        raise ValueError("The five-node full-adder topology requires maximum_degree >= 5")
    if not math.isfinite(copy_weight) or copy_weight <= 0:
        raise ValueError("copy_weight must be finite and positive")
    if factor_copy_topology not in {
        "balanced_tree",
        "carry_chain",
        "inplace_balanced_tree",
    }:
        raise ValueError(
            "factor_copy_topology must be 'balanced_tree', 'carry_chain', or "
            "'inplace_balanced_tree'"
        )

    names: list[str] = []
    gate_nodes: list[tuple[int, ...]] = []
    occurrences: dict[str, list[int]] = {signal: [] for signal in netlist.signal_names}
    for instance in netlist.gates:
        physical: list[int] = []
        template = get_gate(instance.gate)
        for port, signal in zip(template.ports, instance.signals, strict=True):
            index = len(names)
            names.append(f"{instance.name}.{port}")
            physical.append(index)
            occurrences[signal].append(index)
        gate_nodes.append(tuple(physical))

    factor_signals = set(netlist.factor_a) | set(netlist.factor_b)
    copy_edges: list[tuple[int, int]] = []
    groups: dict[str, tuple[int, ...]] = {}
    roots: dict[str, int] = {}
    for signal in netlist.signal_names:
        leaves = list(occurrences[signal])
        if not leaves:
            raise RuntimeError(f"Logical signal has no physical occurrence: {signal}")
        added: list[int] = []
        if signal in factor_signals and factor_copy_topology == "balanced_tree":
            current = leaves
            level = 0
            while len(current) > 1:
                parents: list[int] = []
                for group_index, start in enumerate(range(0, len(current), maximum_degree - 1)):
                    children = current[start : start + maximum_degree - 1]
                    parent = len(names)
                    names.append(f"{signal}__copy_tree[{level},{group_index}]")
                    added.append(parent)
                    parents.append(parent)
                    copy_edges.extend((parent, child) for child in children)
                current = parents
                level += 1
            root = current[0]
        elif signal in factor_signals and factor_copy_topology == "carry_chain":
            copy_edges.extend(zip(leaves, leaves[1:], strict=False))
            root = leaves[(len(leaves) - 1) // 2]
        elif signal in factor_signals:
            root, inplace_edges = _inplace_balanced_copy_tree(leaves)
            copy_edges.extend(inplace_edges)
        else:
            if len(leaves) > 2:
                raise RuntimeError(
                    f"Non-factor signal unexpectedly has fanout greater than two: {signal}"
                )
            if len(leaves) == 2:
                copy_edges.append((leaves[0], leaves[1]))
            root = leaves[0]
        groups[signal] = tuple([*leaves, *added])
        roots[signal] = root

    metadata = {
        "problem": "factorization",
        "representation": "compact_copy_tree",
        "sparse_topology": "aadit_style_count_matched",
        "factor_copy_topology": factor_copy_topology,
        "factor_copy_leaf_order": "array_multiplier_gate_order_equals_arithmetic_column_order",
        "logical_root_policy": (
            "tree_root"
            if factor_copy_topology == "balanced_tree"
            else (
                "lower_median_leaf"
                if factor_copy_topology == "carry_chain"
                else "ordered_recursive_lower_median_leaf"
            )
        ),
        "exact_hardware_reproduction": False,
        "ground_state_guarantee": "termwise_nonnegative_penalties",
        "factor_bits": netlist.factor_bits,
        "product_bits": netlist.product_bits,
        "penalty_profile": netlist.profile,
        "maximum_degree_requested": maximum_degree,
        "copy_weight_multiplier": float(copy_weight),
        "nonnegative_term_compilation": True,
        "requires_zero_clamp": True,
    }
    if model_storage == "edge_v1":
        from .edge_compiler import assemble_edge_models

        model, gate_model, copy_model = assemble_edge_models(
            names, gate_nodes, netlist, tuple(copy_edges), metadata, dtype
        )
    else:
        qubo, gate_qubo, copy_qubo = _assemble_qubo(
            names,
            gate_nodes,
            netlist,
            dtype=dtype,
            copy_edges=tuple(copy_edges),
            copy_weight=float(copy_weight),
            metadata=metadata,
        )
        model, gate_model, copy_model = qubo.to_ising(), gate_qubo.to_ising(), copy_qubo.to_ising()
        reconstructed_coupling = gate_model.coupling + float(copy_weight) * copy_model.coupling
        reconstructed_bias = gate_model.bias + float(copy_weight) * copy_model.bias
        reconstructed_constant = gate_model.constant + float(copy_weight) * copy_model.constant
        if not torch.allclose(model.coupling, reconstructed_coupling, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Gate and COPY couplings do not reconstruct the sparse model")
        if not torch.allclose(model.bias, reconstructed_bias, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Gate and COPY biases do not reconstruct the sparse model")
        if not math.isclose(model.constant, reconstructed_constant, abs_tol=1e-5, rel_tol=1e-6):
            raise RuntimeError("Gate and COPY constants do not reconstruct the sparse model")
    width = netlist.factor_bits
    extra_per_factor = (
        _copy_tree_extra_nodes(width, maximum_degree)
        if factor_copy_topology == "balanced_tree"
        else 0
    )
    expected_nodes = 8 * width**2 - 5 * width + 2 * width * extra_per_factor
    if model.size != expected_nodes:
        raise RuntimeError(f"Sparse multiplier size mismatch: {model.size} != {expected_nodes}")
    actual_maximum_degree = _maximum_degree(model)
    if actual_maximum_degree > maximum_degree:
        raise RuntimeError(
            f"Sparse compiler exceeded degree bound: {actual_maximum_degree} > {maximum_degree}"
        )
    metadata = {
        **model.metadata,
        "edge_count": _edge_count(model),
        "maximum_degree": actual_maximum_degree,
        "copy_edges": len(copy_edges),
    }
    model = replace(model, metadata=metadata)
    return CompiledMultiplier(
        model=model,
        netlist=netlist,
        representation="compact_copy_tree",
        logical_to_physical=groups,
        logical_roots=roots,
        copy_edges=tuple(copy_edges),
        components=FactorHamiltonianComponents(
            gate_model=gate_model,
            copy_model=copy_model,
            gates=tuple(
                PhysicalGate(
                    name=instance.name,
                    gate=instance.gate,
                    nodes=physical,
                    column=sum(
                        int(value)
                        for value in instance.name.partition("[")[2].rstrip("]").split(",")
                    ),
                )
                for instance, physical in zip(netlist.gates, gate_nodes, strict=True)
            ),
        ),
    )


def condition_product(
    compiled: CompiledMultiplier,
    target: int,
    *,
    fix_odd_factors: bool = False,
    fix_factor_msbs: bool = False,
) -> ConditionedFactorizer:
    """Exactly clamp multiplier constants and little-endian product bits."""

    product_limit = 1 << compiled.netlist.product_bits
    if not 0 <= int(target) < product_limit:
        raise ValueError("target does not fit the multiplier product width")
    target = int(target)
    fixed: dict[str, int] = {signal: 0 for signal in compiled.netlist.zero_signals}
    for index, signal in enumerate(compiled.netlist.product):
        fixed[signal] = (target >> index) & 1

    if fix_odd_factors:
        if target % 2 == 0:
            raise ValueError("Odd-factor preprocessing is invalid for an even target")
        fixed[compiled.netlist.factor_a[0]] = 1
        fixed[compiled.netlist.factor_b[0]] = 1
    if fix_factor_msbs:
        fixed[compiled.netlist.factor_a[-1]] = 1
        fixed[compiled.netlist.factor_b[-1]] = 1

    clamp = compiled.clamp_logical(fixed)
    components = None
    if compiled.components is not None:
        fixed_physical = dict(zip(clamp.fixed_indices, clamp.fixed_spins, strict=True))
        gate_clamp = compiled.components.gate_model.clamp(fixed_physical)
        copy_clamp = compiled.components.copy_model.clamp(fixed_physical)
        if (
            gate_clamp.free_indices != clamp.free_indices
            or copy_clamp.free_indices != clamp.free_indices
        ):
            raise RuntimeError("Conditioned Hamiltonian components use a different free order")
        components = FactorHamiltonianComponents(
            gate_model=gate_clamp.model,
            copy_model=copy_clamp.model,
            gates=compiled.components.gates,
        )
    return ConditionedFactorizer(
        compiled=compiled,
        clamp=clamp,
        target=target,
        fixed_logical_bits=fixed,
        components=components,
    )


def verify_factorization(factor_a: int, factor_b: int, target: int) -> bool:
    """Return the exact non-trivial integer factor certificate."""

    left = int(factor_a)
    right = int(factor_b)
    product = int(target)
    return 1 < left < product and 1 < right < product and left * right == product
