"""Release-facing model, initialization and independent certificate helpers.

The established Rademacher, sparse-model and witness-replay semantics are retained
from the audited research drivers without importing their historical campaigns.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sparse_factorization.core.coloring import dsatur_coloring, validate_coloring
from sparse_factorization.core.model import ClampResult, IsingModel
from sparse_factorization.core.padded_sparse import PaddedSparseIsing
from sparse_factorization.core.rng import derive_stream_seed
from sparse_factorization.solver.compiler import (
    ConditionedFactorizer,
    compile_compact_sparse_multiplier,
    condition_product,
)
from sparse_factorization.solver.netlist import build_unsigned_array_multiplier
from sparse_factorization.solver.persistent_circuit import PersistentCircuit
from sparse_factorization.solver.persistent_gibbs import PersistentSparseGibbs
from sparse_factorization.solver.publication_events import (
    SELECTED_CONTEXT_LAYOUT,
    selected_context,
)


@dataclass(frozen=True)
class SparseProblem:
    compiled: Any
    conditioned: ConditionedFactorizer
    colors: tuple[tuple[int, ...], ...]


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        host = value.detach().cpu().contiguous().numpy()
    else:
        host = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def build_sparse_problem(specification: Mapping[str, Any]) -> SparseProblem:
    product_bits = int(specification["product_bits"])
    if product_bits < 4 or product_bits % 2:
        raise ValueError("product_bits must be an even integer of at least four")
    target = int(specification["target"])
    netlist = build_unsigned_array_multiplier(
        product_bits // 2,
        profile=str(specification.get("penalty_profile", "aadit")),
    )
    compiled = compile_compact_sparse_multiplier(
        netlist,
        maximum_degree=int(specification.get("maximum_degree", 5)),
        copy_weight=float(specification.get("copy_weight", 1.0)),
        factor_copy_topology=str(specification.get("factor_copy_topology", "balanced_tree")),
        dtype=torch.float32,
        model_storage=str(specification.get("model_storage", "dense_v1")),
    )
    conditioned = condition_product(
        compiled,
        target,
        fix_odd_factors=bool(specification.get("fix_odd_factors", False)),
        fix_factor_msbs=bool(specification.get("fix_factor_msbs", False)),
    )
    colors = dsatur_coloring(conditioned.model)
    validate_coloring(conditioned.model, colors)
    return SparseProblem(compiled=compiled, conditioned=conditioned, colors=colors)


def independent_initial_states(
    problem: ConditionedFactorizer,
    replicas: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    stream_seed = derive_stream_seed(seed, "p7_independent_physical_initialization_v1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stream_seed)
    bits = torch.randint(
        0,
        2,
        (replicas, problem.model.size),
        generator=generator,
        dtype=torch.int8,
    )
    states = bits.mul(2).sub(1)
    return states, {
        "algorithm": "independent_rademacher_physical_spins_v1",
        "base_seed": int(seed),
        "stream_seed": int(stream_seed),
        "replicas": int(replicas),
        "variables": int(problem.model.size),
        "state_sha256": _tensor_sha256(states),
        "outcome_information_used": False,
    }


@dataclass(frozen=True)
class IndependentCertificateContext:
    """An owned CPU/FP64 snapshot, independent of subsequent sampler mutations.

    Construct from the independently rebuilt verifier problem, once per replay.
    Tensors are private implementation state and must not be mutated by callers.
    """

    _clamp: ClampResult
    _full_model: IsingModel
    _factor_indices: tuple[tuple[int, ...], tuple[int, ...]]

    @classmethod
    def from_problem(cls, problem: ConditionedFactorizer) -> IndependentCertificateContext:
        def snapshot(model: IsingModel) -> IsingModel:
            from sparse_factorization.core.edge_model import EdgeIsingModel

            if isinstance(model, EdgeIsingModel):
                return model.owned_cpu_fp64()
            # .to() alone can alias an already-CPU/FP64 source. Always own storage.
            return IsingModel(
                coupling=model.coupling.detach().to(device="cpu", dtype=torch.float64, copy=True),
                bias=model.bias.detach().to(device="cpu", dtype=torch.float64, copy=True),
                constant=float(model.constant),
                names=tuple(model.names),
            )

        clamp = problem.clamp
        return cls(
            _clamp=ClampResult(
                model=snapshot(problem.model),
                original_names=tuple(clamp.original_names),
                free_indices=tuple(clamp.free_indices),
                fixed_indices=tuple(clamp.fixed_indices),
                fixed_spins=tuple(clamp.fixed_spins),
            ),
            _full_model=snapshot(problem.compiled.model),
            _factor_indices=(
                tuple(problem.compiled.factor_a_indices),
                tuple(problem.compiled.factor_b_indices),
            ),
        )

    def certify(self, free_state: torch.Tensor, *, target: int, tolerance: float) -> dict[str, Any]:
        if free_state.ndim != 1 or free_state.shape[0] != self._clamp.model.size:
            raise ValueError("free_state must contain exactly one conditioned model state")
        # Check before narrowing: fractional/wrapped values must not become valid spins.
        if not bool(torch.all((free_state == -1) | (free_state == 1))):
            raise ValueError("free_state must contain only bipolar -1 or +1 values")
        host = free_state.detach().to(device="cpu", dtype=torch.int8)
        reconstructed = self._clamp.reconstruct(host)
        factor_a, factor_b = (
            sum(((int(reconstructed[node]) + 1) // 2) << bit for bit, node in enumerate(indices))
            for indices in self._factor_indices
        )
        conditioned_energy = float(self._clamp.model.energy(host.to(torch.float64)).item())
        full_energy = float(self._full_model.energy(reconstructed.to(torch.float64)).item())
        integer_valid = factor_a > 1 and factor_b > 1 and factor_a * factor_b == int(target)
        energy_valid = conditioned_energy <= float(tolerance) and full_energy <= float(tolerance)
        return {
            "factor_a": factor_a,
            "factor_b": factor_b,
            "target": int(target),
            "product": factor_a * factor_b,
            "integer_certificate": integer_valid,
            "conditioned_energy_fp64": conditioned_energy,
            "full_energy_fp64": full_energy,
            "energy_tolerance": float(tolerance),
            "energy_certificate": energy_valid,
            "valid_bipolar_state": True,
            "certified": bool(integer_valid and energy_valid),
            "state_sha256": _tensor_sha256(host),
        }


def _problem(config: dict):
    return build_sparse_problem(
        {
            "product_bits": config["product_bits"],
            "target": config["target"],
            "factor_copy_topology": config["topology"],
            "fix_odd_factors": False,
            "fix_factor_msbs": False,
            "model_storage": config.get("model_storage", "dense_v1"),
        }
    )


def temperature_values(config: dict) -> np.ndarray:
    return np.geomspace(*config["beta"], config["temperature_slots"]).astype(np.float32)


def make_engine(config: dict, *, initial_states: torch.Tensor | None = None):
    sparse = _problem(config)
    if initial_states is None:
        initial_states, initialization = independent_initial_states(
            sparse.conditioned, config["replicas"], config["initialization_seed"]
        )
    else:
        initialization = None
    engine = PersistentSparseGibbs(
        sparse.conditioned, initial_states, seed=config["sampler_seed"], allow_research_widths=True
    )
    betas = torch.from_numpy(
        np.tile(
            temperature_values(config),
            config["replicas"] // config["temperature_slots"],
        )
    ).cuda()
    circuit = (
        None
        if config["algorithm"] == "fixed_beta_mixture"
        else PersistentCircuit(
            engine,
            config["circuit_seed"],
            betas,
            logit_clip=config["logit_clip"],
            proposal_beta_scale=config["proposal_beta_scale"],
            root_policy=config["root_policy"],
            exchange_every=config.get("exchange_every", 0),
            exchange_seed=config.get("exchange_seed"),
            backend=config.get("circuit_backend", "torch"),
            history_backend=config.get("history_backend", "torch_host_v1"),
            cdf_layout=config.get("cdf_layout", "beta_group_context"),
            old_state_cache=config.get("old_state_cache", "disabled"),
            cdf_search=config.get("cdf_search", "scan32_v1"),
            lookahead_strength=config.get("lookahead_strength", 0.0),
            regrowth_mode=config.get("regrowth_mode", "full_v1"),
            root_guidance=config.get("root_guidance", "disabled"),
            modular_root_mode=config.get("modular_root_mode", "disabled"),
            guidance_execution=config.get("guidance_execution", "legacy_v1"),
        )
    )
    if circuit is not None and circuit.proposal.modular_sampler is not None:
        from .execution_profiles import configure

        circuit.proposal.reuse_graph_noise = configure(
            circuit.proposal.modular_sampler, config.get("execution_profile", "reference_v1")
        )
    return sparse, engine, betas, circuit, initialization


def replay_packet(config: dict, packet: dict[str, np.ndarray]) -> list[dict]:
    """Replay each observed boundary once, then certify all representative group hits."""
    first_context = int(packet["witness_context"][0])
    compact = str(packet.get("context_layout", "full_population_v1")) == SELECTED_CONTEXT_LAYOUT
    sparse, engine, betas, circuit, _ = make_engine(
        config,
        initial_states=None
        if compact
        else torch.from_numpy(packet["context_states"][first_context]),
    )
    # Other slots do not influence this MH/Gibbs boundary, but the full original
    # population is retained to preserve Philox shapes and draw addresses.
    # Its unused slots keep the original Rademacher template, never factor data.
    template = engine.states.clone() if compact else None
    padded = PaddedSparseIsing.from_model(
        sparse.conditioned.model, device="cuda", dtype=torch.float32
    )
    verifier = IndependentCertificateContext.from_problem(sparse.conditioned)
    reports = {}
    for context in np.unique(packet["witness_context"]):
        indices = np.flatnonzero(packet["witness_context"] == context)
        kinds = {str(packet["witness_kind"][i]) for i in indices}
        if len(kinds) != 1:
            raise ValueError("One boundary context cannot describe different transition kinds")
        kind = kinds.pop()
        if compact:
            slots, selected = selected_context(packet, int(context), config["replicas"])
            states = template.clone()
            states[slots.tolist()] = torch.from_numpy(selected).cuda()
        else:
            states = torch.from_numpy(packet["context_states"][context]).cuda()
        engine.states.copy_(states)
        engine.energies.copy_(padded.energy(states))
        engine.best_states.copy_(states)
        engine.best_energies.copy_(engine.energies)
        engine.global_sweep = int(packet["context_raw_sweep"][context])
        engine.counter_offset = int(packet["context_gibbs_offset"][context])
        engine.first_hit_sweeps.fill_(-1)
        engine.first_hit_states.zero_()
        zero = engine.energies == 0
        engine.first_hit_sweeps[zero] = engine.global_sweep
        engine.first_hit_states[zero] = states[zero]
        if circuit:
            circuit.events = int(packet["context_auxiliary_events"][context])
            circuit.generator.set_state(
                torch.from_numpy(packet["context_auxiliary_rng_state"][context])
            )
        if kind == "gibbs":
            stop = max(int(packet["witness_raw_sweep"][i]) for i in indices)
            engine.run_chunk(betas, stop - engine.global_sweep)
        elif kind == "circuit":
            if circuit is None:
                raise ValueError("A circuit witness requires its declared proposal kernel")
            circuit.step(betas)
        elif kind != "initial":
            raise ValueError("Unknown native event kind")
        for index in indices:
            slot = int(packet["witness_slot"][index])
            actual = engine.first_hit_states[slot].cpu()
            certificate = verifier.certify(actual, target=config["target"], tolerance=0)
            if not (
                certificate["certified"]
                and np.array_equal(actual.numpy(), packet["witness_state"][index])
                and int(engine.first_hit_sweeps[slot]) == int(packet["witness_raw_sweep"][index])
            ):
                raise ValueError("A native circuit-study transition failed replay")
            if kind == "circuit" and circuit.events != int(
                packet["witness_auxiliary_event"][index]
            ):
                raise ValueError("Circuit event ordinal differs during replay")
            reports[int(index)] = {
                "group": int(packet["witness_group"][index]),
                "slot": slot,
                "kind": kind,
                "raw_sweep": int(packet["witness_raw_sweep"][index]),
                "circuit_event": int(packet["witness_auxiliary_event"][index]),
                "transition": int(packet["witness_transition"][index]),
                "wall_seconds_lower": float(packet["witness_wall_lower"][index]),
                "wall_seconds_upper": float(packet["witness_wall_upper"][index]),
                "transition_replay_passed": True,
                **certificate,
            }
    return [reports[i] for i in range(len(reports))]
