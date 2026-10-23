"""Full-support circuit-directed proposals with an explicit Metropolis-Hastings ratio."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .circuit_deferred_radius import DeferredRadiusProposal
from .circuit_fermat import FilteredFermatRootProposal
from .circuit_fermat_crt import FilteredCRTFermatRootProposal
from .circuit_fermat_crt_wide import FilteredWideCRTFermatRootProposal
from .circuit_modular import UNIT_GROUP_MODES
from .circuit_quotient import QUOTIENT_MODES, QuotientRootProposal
from .circuit_range import (
    PUBLIC_RANGE_MODES,
    WIDE_PUBLIC_RANGE_MODES,
    PublicRangeRootProposal,
    WidePublicRangeRootProposal,
)
from .circuit_root_filter import FERMAT_MODES, FILTER_MODES, FilteredQuotientRootProposal
from .compiler import ConditionedFactorizer
from .gates import get_gate

UINT32_MASS = 1 << 32
CDF_LAYOUTS = ("beta_group_context", "group_beta_context", "group_context_beta")


@dataclass(frozen=True)
class CircuitNoise:
    """Optional externally supplied noise for deterministic implementation checks."""

    root_choices: torch.Tensor
    body_uniforms: torch.Tensor
    acceptance_uniforms: torch.Tensor
    root_masks: torch.Tensor | None = None
    root_uniforms: torch.Tensor | None = None
    modular_uniforms: torch.Tensor | None = None


@dataclass(frozen=True)
class CircuitProposalResult:
    """The actual accepted physical state and auditable proposal statistics."""

    states: torch.Tensor
    energies: torch.Tensor
    proposed_energies: torch.Tensor
    accepted: torch.Tensor
    root_changed: torch.Tensor
    log_q_forward_body: torch.Tensor
    log_q_reverse_body: torch.Tensor
    log_acceptance_ratio: torch.Tensor


@dataclass(frozen=True)
class CircuitArithmeticResult(CircuitProposalResult):
    """P7.29-only read-only arithmetic counters before physical-state acceptance."""

    arithmetic_metrics: torch.Tensor


class CircuitDirectedProposal:
    """Propose roots, then regenerate the full body or its selected descendant groups.

    The physical carry-chain root remains the compiler's median leaf. COPY paths
    are oriented away from that leaf, rather than changing its interpretation.
    Each gate group assigns its non-root free ports jointly; incoming COPY edges
    connect it only to already assigned parents. Target product pins remain fixed.

    Per-gate probabilities use clipped relative logits, followed by a uint32 CDF.
    The cached log density is computed from the actual integer bin masses, not
    from the pre-quantized softmax. Symmetric root labels cancel from the MH ratio;
    state-dependent root selection requires its explicit reverse correction.
    This is a stochastic full-support proposal, never a codeword
    projection or an exact draw from the target conditional.
    """

    def __init__(
        self,
        problem: ConditionedFactorizer,
        betas: Sequence[float],
        *,
        device: torch.device | str = "cpu",
        logit_clip: float = 12.0,
        proposal_beta_scale: float = 1.0,
        root_policy: str = "lazy_single_flip_v1",
        backend: str = "torch",
        cdf_layout: str = "beta_group_context",
        old_state_cache: str = "disabled",
        cdf_search: str = "scan32_v1",
        lookahead_strength: float = 0.0,
        regrowth_mode: str = "full_v1",
        root_guidance: str = "disabled",
        modular_root_mode: str = "disabled",
        guidance_execution: str = "legacy_v1",
    ) -> None:
        self.problem = problem
        if root_guidance not in {"disabled", "uniform_v1", "partition_v1"}:
            raise ValueError("Unknown root guidance mode")
        if root_guidance != "disabled" and (
            root_policy != "uniform_0_1_2_4_v2" or regrowth_mode != "full_v1"
        ):
            raise ValueError("Root guidance requires the declared multi-root/full-body protocol")
        if (
            modular_root_mode
            not in {"disabled", "control_v1", "involution_mixture_v1"}
            | UNIT_GROUP_MODES
            | PUBLIC_RANGE_MODES
            | QUOTIENT_MODES
            | FILTER_MODES
        ):
            raise ValueError("Unknown modular root mode")
        if modular_root_mode != "disabled" and root_guidance != "partition_v1":
            raise ValueError("Modular root modes require partition guidance and full body")
        self.modular_root_mode = modular_root_mode
        self.root_guidance = root_guidance
        if guidance_execution not in {"legacy_v1", "skip_unused_v1"}:
            raise ValueError("Unknown root guidance execution mode")
        self.guidance_execution = guidance_execution
        self.skip_unused_guidance = guidance_execution == "skip_unused_v1" and (
            modular_root_mode == "quotient_only_v1" or modular_root_mode in FILTER_MODES
        )
        if modular_root_mode in FERMAT_MODES and not self.skip_unused_guidance:
            raise ValueError("Fermat proposals require explicit skip_unused_v1 execution")
        if regrowth_mode not in {"full_v1", "root_descendants_v1"}:
            raise ValueError("Unknown circuit regrowth mode")
        if regrowth_mode != "full_v1" and old_state_cache != "disabled":
            raise ValueError("Partial regrowth requires the full-body density cache disabled")
        self.regrowth_mode = regrowth_mode
        if not np.isfinite(lookahead_strength) or not 0 <= lookahead_strength <= 1:
            raise ValueError("Use a finite backward-message strength between zero and one")
        self.lookahead_strength = float(lookahead_strength)
        if root_policy not in {"lazy_single_flip_v1", "lazy_single_flip_v2", "uniform_0_1_2_4_v2"}:
            raise ValueError("Unknown symmetric root proposal")
        self.root_policy = root_policy
        self.device = torch.empty(0, device=device).device
        if backend not in {"torch", "triton_layers_v1", "triton_graph_v1"}:
            raise ValueError("Unknown circuit proposal backend")
        if backend in {"triton_layers_v1", "triton_graph_v1"} and self.device.type != "cuda":
            raise ValueError("The Triton layer backend requires CUDA")
        if cdf_layout not in CDF_LAYOUTS:
            raise ValueError("Unknown circuit sampling CDF layout")
        if cdf_layout != CDF_LAYOUTS[0] and backend == "torch":
            raise ValueError("Noncanonical sampling CDF layouts require a Triton backend")
        if old_state_cache not in {"disabled", "validated_v1"}:
            raise ValueError("Unknown old-state cache mode")
        if old_state_cache != "disabled" and backend != "triton_graph_v1":
            raise ValueError("Validated old-state caching requires the Graph backend")
        if cdf_search not in {"scan32_v1", "binary5_v1"}:
            raise ValueError("Unknown integer CDF search method")
        if cdf_search != "scan32_v1" and backend == "torch":
            raise ValueError("Binary CDF search requires a Triton backend")
        self.cdf_search = cdf_search
        self.old_state_cache = old_state_cache
        self.cdf_layout = cdf_layout
        self.backend = backend
        self._graph_runtime = None
        self.reuse_graph_noise = False
        self.beta_values = np.asarray(betas, dtype=np.float64)
        if (
            self.beta_values.ndim != 1
            or not len(self.beta_values)
            or not np.all(np.isfinite(self.beta_values) & (self.beta_values > 0))
        ):
            raise ValueError("Use a nonempty finite positive beta table")
        if (
            not 0 < logit_clip <= 16
            or not np.isfinite(proposal_beta_scale)
            or proposal_beta_scale <= 0
        ):
            raise ValueError("Use bounded finite proposal logits and a positive beta scale")
        compiled = problem.compiled
        if (
            compiled.representation != "compact_copy_tree"
            or compiled.model.metadata.get("factor_copy_topology") != "carry_chain"
        ):
            raise ValueError("Circuit proposals require the declared compact carry-chain model")
        if (
            compiled.components is None
            or compiled.model.metadata["copy_weight_multiplier"] != 1
            or compiled.netlist.profile != "aadit"
        ):
            raise ValueError("Circuit proposals require physical gate provenance and unit COPY")
        self.logit_clip = float(logit_clip)
        self.proposal_beta_scale = float(proposal_beta_scale)
        self.gates = compiled.components.gates
        self.group_count = len(self.gates)
        roots = (*compiled.factor_a_indices, *compiled.factor_b_indices)
        self.root_count = len(roots)
        fixed = dict(zip(problem.clamp.fixed_indices, problem.clamp.fixed_spins, strict=True))
        if set(roots).intersection(fixed):
            raise ValueError("Do not clamp factor roots in the circuit proposal")
        owner = {node: i for i, gate in enumerate(self.gates) for node in gate.nodes}
        if len(owner) != sum(len(gate.nodes) for gate in self.gates):
            raise ValueError("Physical gate groups must be disjoint")
        if set(problem.clamp.free_indices) - set(owner):
            raise ValueError("Free internal COPY nodes need a separately defined proposal group")
        roots_set = set(roots)
        source = {
            signal: compiled.logical_roots[signal]
            for signal in (*compiled.netlist.factor_a, *compiled.netlist.factor_b)
        }
        for logical, physical in zip(compiled.netlist.gates, self.gates, strict=True):
            start = 2 if physical.gate == "and" else 3
            for signal, node in zip(logical.signals[start:], physical.nodes[start:], strict=True):
                source[signal] = node
        edges = {node: [] for node in range(compiled.model.size)}
        for left, right in compiled.copy_edges:
            edges[left].append(right)
            edges[right].append(left)
        incoming: list[list[tuple[int, int]]] = [[] for _ in self.gates]
        directed_edges = 0
        for signal, occurrences in compiled.logical_to_physical.items():
            if all(node in fixed for node in occurrences):
                if len({fixed[node] for node in occurrences}) != 1:
                    raise ValueError("A fully fixed COPY component is inconsistent")
                continue
            if any(node in fixed for node in occurrences):
                raise ValueError("Partial physical COPY conditioning is not supported")
            root = source[signal]
            visited = {root}
            pending = deque([root])
            while pending:
                parent = pending.popleft()
                for child in edges[parent]:
                    if child in visited:
                        continue
                    visited.add(child)
                    pending.append(child)
                    gate_index = owner[child]
                    incoming[gate_index].append((parent, self.gates[gate_index].nodes.index(child)))
                    directed_edges += 1
            if visited != set(occurrences):
                raise ValueError("COPY orientation did not cover one logical component")
        expected_edges = sum(not (a in fixed and b in fixed) for a, b in compiled.copy_edges)
        if directed_edges != expected_edges:
            raise ValueError("Every nonconstant COPY penalty must appear exactly once")

        contexts, root_ports, parents = [], [], []
        for i, gate in enumerate(self.gates):
            local_roots = [
                (node, port) for port, node in enumerate(gate.nodes) if node in roots_set
            ]
            context_nodes = [node for node, _ in local_roots] + [node for node, _ in incoming[i]]
            if len(context_nodes) > 3:
                raise ValueError("Unexpected circuit context width")
            contexts.append(context_nodes)
            root_ports.append(local_roots)
            parents.append({owner[node] for node, _ in incoming[i] if node not in roots_set})
        levels: list[list[int]] = []
        assigned: set[int] = set()
        while len(assigned) < self.group_count:
            ready = [
                i for i in range(self.group_count) if i not in assigned and parents[i] <= assigned
            ]
            if not ready:
                raise ValueError("The gate/COPY proposal dependency graph contains a cycle")
            levels.append(ready)
            assigned.update(ready)

        codes = np.arange(32, dtype=np.int64)
        candidate_bits = (codes[:, None] >> np.arange(5)) & 1
        candidate_spins = 2 * candidate_bits - 1
        gate_energies = np.zeros((self.group_count, 32), dtype=np.float64)
        group_nodes = np.zeros((self.group_count, 5), dtype=np.int64)
        port_weights = np.zeros_like(group_nodes)
        context_nodes = np.zeros((self.group_count, 3), dtype=np.int64)
        context_weights = np.zeros_like(context_nodes)
        potential = np.zeros((self.group_count, 8, 32), dtype=np.float64)
        allowed = np.zeros((self.group_count, 8, 32), dtype=bool)
        for i, gate in enumerate(self.gates):
            size = len(gate.nodes)
            bits = torch.from_numpy(candidate_bits[: 1 << size, :size]).double()
            energy = (
                get_gate(gate.gate)
                .qubo(compiled.netlist.profile, dtype=torch.float64)
                .energy(bits)
                .numpy()
            )
            gate_energies[i, : len(energy)] = energy
            group_nodes[i, :size] = gate.nodes
            port_weights[i, :size] = 1 << np.arange(size)
            context_nodes[i, : len(contexts[i])] = contexts[i]
            context_weights[i, : len(contexts[i])] = 1 << np.arange(len(contexts[i]))
            for context in range(8):
                valid = codes < 1 << size
                for port, node in enumerate(gate.nodes):
                    if node in fixed:
                        valid &= candidate_spins[:, port] == fixed[node]
                for bit, (_, port) in enumerate(root_ports[i]):
                    valid &= candidate_bits[:, port] == ((context >> bit) & 1)
                score = gate_energies[i].copy()
                for bit, (_, port) in enumerate(incoming[i], len(root_ports[i])):
                    score += 2 * (candidate_bits[:, port] != ((context >> bit) & 1))
                if not valid.any():
                    raise ValueError("A gate context has no physically allowed assignment")
                allowed[i, context] = valid
                potential[i, context] = score
        logits = -self.beta_values[:, None, None, None] * self.proposal_beta_scale * potential[None]
        logits = np.where(allowed[None], logits, -np.inf)
        lookahead_span = 0.0
        if self.lookahead_strength:
            from .circuit_lookahead import backward_unary_bias

            lookahead = backward_unary_bias(
                logits,
                [gate.nodes for gate in self.gates],
                contexts,
                levels,
                roots_set,
                strength=self.lookahead_strength,
            )
            lookahead_span = float(np.ptp(lookahead, axis=-1).max())
            logits = logits + lookahead[:, :, None, :]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        weights = np.where(allowed[None], np.exp(np.maximum(shifted, -self.logit_clip)), 0.0)
        probabilities = weights / weights.sum(axis=-1, keepdims=True)
        cumulative = np.floor(np.cumsum(probabilities, axis=-1) * UINT32_MASS).astype(np.int64)
        last = np.max(np.where(allowed, codes, -1), axis=-1)
        cumulative = np.where(
            codes[None, None, None, :] >= last[None, :, :, None], UINT32_MASS, cumulative
        )
        counts = np.diff(
            cumulative, axis=-1, prepend=np.zeros((*cumulative.shape[:-1], 1), dtype=np.int64)
        )
        if np.any(counts[:, allowed] <= 0) or np.any(counts[:, ~allowed] != 0):
            raise ValueError("Quantized proposal must preserve exactly its allowed full support")
        log_probability = np.full(counts.shape, -np.inf, dtype=np.float64)
        np.log(counts, out=log_probability, where=counts > 0)
        log_probability -= np.log(float(UINT32_MASS))

        def tensor(value, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        self.betas = tensor(self.beta_values)
        self.root_nodes = tensor(roots, torch.long)
        self.group_root_words = self.root_bit_words = None
        if self.regrowth_mode != "full_v1":
            from .circuit_regrowth import root_ancestor_words

            words = root_ancestor_words(
                [gate.nodes for gate in self.gates], contexts, levels, roots, set(fixed)
            )
            self.group_root_words = tensor(words)
            self.root_bit_words = tensor(
                np.asarray([1 << i for i in range(self.root_count)], dtype=np.uint64).view(np.int64)
            )
        self.free_nodes = tensor(problem.clamp.free_indices, torch.long)
        fixed_template = np.ones(compiled.model.size, dtype=np.int8)
        for node, value in fixed.items():
            fixed_template[node] = value
        self.fixed_template = tensor(fixed_template)
        self.group_nodes = tensor(group_nodes)
        self.port_weights = tensor(port_weights)
        mutable = np.zeros(compiled.model.size, dtype=bool)
        mutable[list(problem.clamp.free_indices)] = True
        mutable[list(roots)] = False
        self.mutable_ports = tensor(mutable[group_nodes] & (port_weights != 0))
        self.context_nodes = tensor(context_nodes)
        self.context_weights = tensor(context_weights)
        self.group_ids = tensor(np.arange(self.group_count))
        self.cumulative = tensor(cumulative)
        order = {
            "beta_group_context": (0, 1, 2, 3),
            "group_beta_context": (1, 0, 2, 3),
            "group_context_beta": (1, 2, 0, 3),
        }[cdf_layout]
        self.sampling_cumulative = self.cumulative.permute(order).contiguous()
        self.log_probabilities = tensor(log_probability)
        self.gate_energies = tensor(gate_energies)
        self.copy_edges = tensor(compiled.copy_edges, torch.long).reshape(-1, 2)
        self.port_shifts = tensor(np.arange(5))
        self.stages = []
        for level in levels:
            ids = tensor(level, torch.long)
            real = np.flatnonzero(port_weights[level].ravel())
            nodes = group_nodes[level].ravel()[real]
            self.stages.append((ids, tensor(nodes), tensor(real)))
        self.root_sampler = None
        if self.root_guidance == "partition_v1" and self.modular_root_mode not in FERMAT_MODES:
            from .circuit_roots import ContextRootProposal

            self.root_sampler = ContextRootProposal(
                contexts, roots, context_nodes, context_weights, logits, self.device
            )
        self.modular_sampler = None
        if self.modular_root_mode != "disabled":
            from .circuit_modular import ModularRootProposal, UnitGroupRootProposal

            if len(compiled.factor_a_indices) != len(compiled.factor_b_indices):
                raise ValueError("Modular roots require equal input widths")
            if self.modular_root_mode in FERMAT_MODES:
                sampler_class = (
                    DeferredRadiusProposal
                    if self.modular_root_mode == "fermat_deferred262144_v1"
                    else FilteredWideCRTFermatRootProposal
                    if self.modular_root_mode == "fermat_crt128_filter256_v1"
                    else FilteredCRTFermatRootProposal
                    if self.modular_root_mode == "fermat_crt_filter256_v1"
                    else FilteredFermatRootProposal
                )
                self.modular_sampler = sampler_class(
                    len(compiled.factor_a_indices),
                    problem.target,
                    self.device,
                    mode=self.modular_root_mode,
                )
            elif self.modular_root_mode in FILTER_MODES:
                self.modular_sampler = FilteredQuotientRootProposal(
                    len(compiled.factor_a_indices),
                    problem.target,
                    self.device,
                    mode=self.modular_root_mode,
                )
            elif self.modular_root_mode in QUOTIENT_MODES:
                self.modular_sampler = QuotientRootProposal(
                    len(compiled.factor_a_indices),
                    problem.target,
                    self.device,
                    pure=self.modular_root_mode == "quotient_only_v1",
                )
            elif self.modular_root_mode in PUBLIC_RANGE_MODES:
                range_class = (
                    WidePublicRangeRootProposal
                    if self.modular_root_mode in WIDE_PUBLIC_RANGE_MODES
                    else PublicRangeRootProposal
                )
                self.modular_sampler = range_class(
                    len(compiled.factor_a_indices),
                    problem.target,
                    self.device,
                    bounded=self.modular_root_mode
                    in {"public_range_first8_v1", "public_range24_first8_v1"},
                )
            else:
                sampler_class = (
                    UnitGroupRootProposal
                    if self.modular_root_mode in UNIT_GROUP_MODES
                    else ModularRootProposal
                )
                self.modular_sampler = sampler_class(len(compiled.factor_a_indices), self.device)
        self.geometry = {
            "modular_root_mode": self.modular_root_mode,
            "modular_root_details": None
            if self.modular_sampler is None
            else self.modular_sampler.metadata,
            "root_guidance": self.root_guidance,
            "guidance_execution": self.guidance_execution,
            "skips_unused_guidance": self.skip_unused_guidance,
            "root_guidance_details": None
            if self.root_sampler is None
            else self.root_sampler.metadata,
            "backend": self.backend,
            "cdf_layout": self.cdf_layout,
            "cdf_search": self.cdf_search,
            "old_state_cache": self.old_state_cache,
            "regrowth_mode": self.regrowth_mode,
            "group_root_words": None
            if self.group_root_words is None
            else self.group_root_words.cpu().tolist(),
            "zero_root_mask_regrowth": "full_body",
            "groups": self.group_count,
            "stages": [list(level) for level in levels],
            "root_nodes": list(roots),
            "root_policy": "compiler_lower_median_leaf",
            "root_move_policy": self.root_policy,
            "free_spins": problem.model.size,
            "maximum_context_bits": max(map(len, contexts)),
            "nonconstant_copy_terms": directed_edges,
            "gate_terms": self.group_count,
            "lookup_table_bytes": int(cumulative.nbytes + log_probability.nbytes),
            "categorical_mass": UINT32_MASS,
            "logit_clip": self.logit_clip,
            "proposal_beta_scale": self.proposal_beta_scale,
            "lookahead_strength": self.lookahead_strength,
            "lookahead": "backward_unary_v1" if self.lookahead_strength else "disabled",
            "maximum_lookahead_logit_span": lookahead_span,
            "known_factor_information_used": False,
        }

    def expand(self, states: torch.Tensor) -> torch.Tensor:
        if (
            states.ndim != 2
            or states.shape[1] != self.problem.model.size
            or states.device != self.device
        ):
            raise ValueError("States must match the conditioned model and proposal device")
        full = self.fixed_template.expand(len(states), -1).clone()
        full[:, self.free_nodes] = states
        return full

    def _contexts(self, full: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        bits = full[:, self.context_nodes[ids]].long().add(1).div(2, rounding_mode="floor")
        return (bits * self.context_weights[ids]).sum(dim=-1)

    def _codes(self, full: torch.Tensor) -> torch.Tensor:
        bits = full[:, self.group_nodes].long().add(1).div(2, rounding_mode="floor")
        return (bits * self.port_weights).sum(dim=-1)

    def regrowth_groups(self, noise: CircuitNoise) -> torch.Tensor | None:
        """Select a fixed descendant closure; a zero root mask refreshes the full body."""
        if self.regrowth_mode == "full_v1":
            return None
        if noise.root_masks is None:
            selected = torch.where(
                noise.root_choices < self.root_count,
                self.root_bit_words[noise.root_choices.clamp_max(self.root_count - 1)],
                0,
            )
        else:
            selected = (noise.root_masks * self.root_bit_words).sum(1)
        return (selected[:, None] == 0) | (
            (selected[:, None] & self.group_root_words[None, :]) != 0
        )

    def body_log_probability(
        self,
        full: torch.Tensor,
        beta_indices: torch.Tensor,
        active_groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self._contexts(full, self.group_ids)
        code = self._codes(full)
        terms = self.log_probabilities[
            beta_indices[:, None], self.group_ids[None, :], context, code
        ]
        if active_groups is not None:
            terms = torch.where(active_groups, terms, 0.0)
        return terms.sum(dim=1)

    def energy(self, full: torch.Tensor) -> torch.Tensor:
        gates = self.gate_energies[self.group_ids[None, :], self._codes(full)].sum(dim=1)
        copies = (full[:, self.copy_edges[:, 0]] != full[:, self.copy_edges[:, 1]]).sum(dim=1)
        return gates + 2 * copies

    def draw_noise(
        self, replicas: int, generator: torch.Generator, *, modular_out=None
    ) -> CircuitNoise:
        """Draw the original fixed noise layout, including unused body words."""
        masks = None
        if self.root_policy == "lazy_single_flip_v1":
            root_choices = torch.randint(
                2 * self.root_count, (replicas,), device=self.device, generator=generator
            )
        else:
            root_choices = torch.randint(
                UINT32_MASS, (replicas,), device=self.device, generator=generator
            )
            ranking = torch.rand(
                (replicas, self.root_count),
                dtype=torch.float64,
                device=self.device,
                generator=generator,
            ).argsort(dim=1)
            if self.root_policy == "lazy_single_flip_v2":
                size = root_choices & 1
            else:
                sizes = torch.tensor([0, 1, 2, 4], device=self.device)
                size = sizes[root_choices & 3]
            masks = torch.zeros((replicas, self.root_count), dtype=torch.bool, device=self.device)
            masks.scatter_(
                1,
                ranking,
                torch.arange(self.root_count, device=self.device)[None, :] < size[:, None],
            )
        root_uniforms = (
            None
            if self.root_guidance == "disabled"
            else torch.randint(UINT32_MASS, (replicas, 4), device=self.device, generator=generator)
        )
        modular_uniforms = (
            None
            if self.modular_root_mode == "disabled"
            else torch.randint(
                UINT32_MASS,
                (replicas, self.modular_sampler.noise_words),
                device=self.device,
                generator=generator,
                **({"out": modular_out} if modular_out is not None else {}),
            )
        )
        return CircuitNoise(
            root_choices,
            torch.randint(
                UINT32_MASS,
                (replicas, self.group_count),
                device=self.device,
                generator=generator,
            ),
            torch.rand(replicas, dtype=torch.float64, device=self.device, generator=generator),
            masks,
            root_uniforms,
            modular_uniforms,
        )

    @torch.inference_mode()
    def step(
        self,
        states: torch.Tensor,
        beta_indices: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: CircuitNoise | None = None,
    ) -> CircuitProposalResult:
        """Return one declared root/body MH update without modifying the input."""

        old = self.expand(states)
        replicas = len(states)
        if (
            beta_indices.shape != (replicas,)
            or beta_indices.dtype != torch.long
            or beta_indices.device != self.device
        ):
            raise ValueError("beta_indices must be a per-replica integer vector on the device")
        if noise is None:
            if generator is None:
                raise ValueError("A generator or explicit noise is required")
            modular_out = None
            if self.reuse_graph_noise and self._graph_runtime is not None:
                existing = self._graph_runtime.noise.modular_uniforms
                if existing is not None and existing.shape == (
                    replicas,
                    self.modular_sampler.noise_words,
                ):
                    modular_out = existing
            noise = self.draw_noise(replicas, generator, modular_out=modular_out)
        if self.root_guidance != "disabled" and (
            noise.root_uniforms is None
            or noise.root_uniforms.shape != (replicas, 4)
            or noise.root_uniforms.dtype != torch.int64
            or noise.root_uniforms.device != self.device
        ):
            raise ValueError("The v3 noise layout requires four uint32 root words per replica")
        if self.modular_root_mode != "disabled" and (
            noise.modular_uniforms is None
            or noise.modular_uniforms.shape != (replicas, self.modular_sampler.noise_words)
            or noise.modular_uniforms.dtype != torch.int64
            or noise.modular_uniforms.device != self.device
        ):
            version = (
                "v7"
                if self.modular_root_mode in FERMAT_MODES
                else "v6"
                if self.modular_root_mode in FILTER_MODES
                else "v5"
                if self.modular_root_mode in PUBLIC_RANGE_MODES | QUOTIENT_MODES
                else "v4"
            )
            raise ValueError(
                f"The {version} noise layout requires "
                f"{self.modular_sampler.noise_words} modular words per replica"
            )
        if self.backend == "triton_graph_v1":
            from .circuit_graph import CircuitGraph

            if self._graph_runtime is None or not self._graph_runtime.matches(states, noise):
                self._graph_runtime = CircuitGraph(self, states, old, beta_indices, noise)
            return self._graph_runtime.run(states, old, beta_indices, noise)
        return self._step_from_noise(states, old, beta_indices, noise)

    def _step_from_noise(
        self,
        states: torch.Tensor,
        old: torch.Tensor,
        beta_indices: torch.Tensor,
        noise: CircuitNoise,
        *,
        old_cache=None,
    ) -> CircuitProposalResult:
        """Deterministic calculation shared by eager launches and graph replay."""
        replicas = len(states)
        proposed = old.clone()
        root_forward = None
        if self.skip_unused_guidance:
            # The pure arithmetic branch below supplies every root and change flag.
            # Keep drawing all original guidance words outside this deterministic path.
            pass
        elif self.root_sampler is not None:
            sizes = self.root_sampler.cardinalities[noise.root_choices & 3]
            masks, root_order, root_forward = self.root_sampler.sample(
                old, beta_indices, noise.root_uniforms, sizes
            )
            change = masks.any(1)
            proposed[:, self.root_nodes] *= torch.where(masks, -1, 1).to(torch.int8)
        elif noise.root_masks is None:
            if self.root_policy != "lazy_single_flip_v1":
                raise ValueError("The v2 noise layout includes an explicit root mask")
            change = noise.root_choices < self.root_count
            selected_root = self.root_nodes[noise.root_choices.clamp_max(self.root_count - 1)]
            replica_ids = torch.arange(replicas, device=self.device)
            proposed[replica_ids, selected_root] *= torch.where(change, -1, 1).to(torch.int8)
        else:
            if (
                noise.root_masks.shape != (replicas, self.root_count)
                or noise.root_masks.dtype != torch.bool
            ):
                raise ValueError("Root masks must be a per-replica Boolean array")
            change = noise.root_masks.any(dim=1)
            proposed[:, self.root_nodes] *= torch.where(noise.root_masks, -1, 1).to(torch.int8)
        modular_branch = None
        range_details = None
        quotient_details = None
        filter_details = None
        if (
            self.modular_root_mode
            in {"involution_mixture_v1", "unit_group_mixture_v1"}
            | PUBLIC_RANGE_MODES
            | QUOTIENT_MODES
            | FILTER_MODES
        ):
            modular_branch = (noise.modular_uniforms[:, 0] & 1) == 0
            if (
                self.modular_root_mode == "quotient_only_v1"
                or self.modular_root_mode in FILTER_MODES
            ):
                modular_branch = torch.ones_like(modular_branch)
            roots = old[:, self.root_nodes]
            if self.modular_root_mode in FILTER_MODES:
                modular_roots, filter_details = self.modular_sampler.propose_filtered(
                    roots, noise.modular_uniforms
                )
            elif self.modular_root_mode in QUOTIENT_MODES:
                modular_roots, quotient_details = self.modular_sampler.propose_with_details(
                    roots, noise.modular_uniforms
                )
            elif self.modular_root_mode in PUBLIC_RANGE_MODES:
                modular_roots, range_details = self.modular_sampler.propose_with_details(
                    roots, noise.modular_uniforms
                )
            else:
                modular_roots = self.modular_sampler.propose(roots, noise.modular_uniforms)
            proposed[:, self.root_nodes] = torch.where(
                modular_branch[:, None], modular_roots, proposed[:, self.root_nodes]
            )
            change = (proposed[:, self.root_nodes] != roots).any(1)
        active_groups = self.regrowth_groups(noise)
        if self.backend == "torch":
            for ids, nodes, real in self.stages:
                context = self._contexts(proposed, ids)
                cdf = self.cumulative[beta_indices[:, None], ids[None, :], context]
                codes = (noise.body_uniforms[:, ids, None] >= cdf).sum(dim=-1)
                ports = ((codes[:, :, None] >> self.port_shifts) & 1).mul(2).sub(1).to(torch.int8)
                values = ports.reshape(replicas, -1)[:, real]
                if active_groups is not None:
                    selected = active_groups[:, ids, None].expand(-1, -1, 5)
                    values = torch.where(
                        selected.reshape(replicas, -1)[:, real], values, proposed[:, nodes]
                    )
                proposed[:, nodes] = values
            forward = self.body_log_probability(proposed, beta_indices, active_groups)
            reverse = self.body_log_probability(old, beta_indices, active_groups)
            old_energy, proposed_energy = self.energy(old), self.energy(proposed)
        else:
            from .circuit_triton import energy_and_density, sample_body

            sample_body(self, proposed, beta_indices, noise.body_uniforms, active_groups)
            proposed_energy, forward = energy_and_density(
                self, proposed, beta_indices, active_groups
            )
            old_energy, reverse = (
                energy_and_density(self, old, beta_indices, active_groups)
                if old_cache is None
                else old_cache.evaluate(self, states, old, beta_indices)
            )
        log_acceptance = (
            -self.betas[beta_indices] * (proposed_energy - old_energy) + reverse - forward
        )
        if root_forward is not None:
            root_reverse = self.root_sampler.log_probability(
                proposed, beta_indices, root_order, sizes
            )
            root_correction = root_reverse - root_forward
            if modular_branch is not None:
                root_correction = torch.where(
                    modular_branch, self._arithmetic_root_correction(old, proposed), root_correction
                )
            log_acceptance += root_correction
        elif self.skip_unused_guidance:
            # Skipping the unused partition calculation must not remove the actual
            # quotient/filter mass ratio. Preserve its original arithmetic order.
            log_acceptance += self._arithmetic_root_correction(old, proposed)
        accepted = torch.log(noise.acceptance_uniforms) < log_acceptance.clamp_max(0)
        if filter_details is not None:
            # Failed root rejection banks contribute only diagonal proposal mass.
            accepted &= filter_details["valid"]
        actual = torch.where(
            accepted[:, None],
            proposed[:, self.free_nodes],
            states,
            **({"out": old_cache.states} if old_cache is not None else {}),
        )
        observation = {}
        result_class = CircuitProposalResult
        if self.modular_root_mode in FILTER_MODES:
            result_class = CircuitArithmeticResult
            observation["arithmetic_metrics"] = self.modular_sampler.observe_filtered(
                old[:, self.root_nodes],
                proposed[:, self.root_nodes],
                proposed_energy,
                accepted,
                filter_details,
            )
        elif self.modular_root_mode in QUOTIENT_MODES:
            result_class = CircuitArithmeticResult
            observation["arithmetic_metrics"] = self.modular_sampler.observe_quotient(
                old[:, self.root_nodes],
                proposed[:, self.root_nodes],
                modular_branch,
                proposed_energy,
                accepted,
                quotient_details,
            )
        elif self.modular_root_mode in UNIT_GROUP_MODES | PUBLIC_RANGE_MODES:
            result_class = CircuitArithmeticResult
            branch = torch.zeros_like(accepted) if modular_branch is None else modular_branch
            observer = (
                self.modular_sampler.observe
                if range_details is None
                else self.modular_sampler.observe_range
            )
            observation["arithmetic_metrics"] = observer(
                old[:, self.root_nodes],
                proposed[:, self.root_nodes],
                branch,
                self.problem.target,
                proposed_energy,
                accepted,
                **({} if range_details is None else {"details": range_details}),
            )
        result = result_class(
            states=actual,
            energies=torch.where(accepted, proposed_energy, old_energy),
            proposed_energies=proposed_energy,
            accepted=accepted,
            root_changed=change,
            log_q_forward_body=forward,
            log_q_reverse_body=reverse,
            log_acceptance_ratio=log_acceptance,
            **observation,
        )
        if old_cache is not None:
            old_cache.commit(result, beta_indices)
        return result

    def _arithmetic_root_correction(self, old: torch.Tensor, proposed: torch.Tensor):
        if self.modular_root_mode in FILTER_MODES:
            return self.modular_sampler.log_weighted_mass(
                old[:, self.root_nodes]
            ) - self.modular_sampler.log_weighted_mass(proposed[:, self.root_nodes])
        if self.modular_root_mode in QUOTIENT_MODES:
            return self.modular_sampler.log_probability(
                old[:, self.root_nodes]
            ) - self.modular_sampler.log_probability(proposed[:, self.root_nodes])
        return 0.0
