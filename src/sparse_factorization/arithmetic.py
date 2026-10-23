"""Same-noise selected-root integer control, with no physical body or MH update."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .core.rng import derive_stream_seed
from .runner import validate_config
from .solver.circuit_deferred_radius import DeferredRadiusProposal
from .solver.circuit_fermat import FilteredFermatRootProposal
from .solver.circuit_fermat_crt import FilteredCRTFermatRootProposal
from .solver.circuit_fermat_crt_wide import FilteredWideCRTFermatRootProposal
from .solver.circuit_proposal import CircuitDirectedProposal
from .solver.circuit_root_filter import FilteredQuotientRootProposal
from .solver.integer_words import exact_product_match
from .solver.publication_io import atomic_json, canonical_digest
from .solver.publication_runtime import _problem, independent_initial_states
from .stream import ChunkWriter

SAMPLERS = {
    "quotient_filter256_v1": (FilteredQuotientRootProposal, "p736_filtered_circuit_cuda_philox_v6"),
    "fermat_filter256_v1": (FilteredFermatRootProposal, "p745_fermat_circuit_cuda_philox_v7"),
    "fermat_crt_filter256_v1": (
        FilteredCRTFermatRootProposal,
        "p747_crt_fermat_circuit_cuda_philox_v8",
    ),
    "fermat_crt128_filter256_v1": (
        FilteredWideCRTFermatRootProposal,
        "p748_crt128_circuit_cuda_philox_v9",
    ),
    "fermat_deferred262144_v1": (
        DeferredRadiusProposal,
        "p749_deferred_radius_circuit_cuda_philox_v10",
    ),
}


def setup(config: dict):
    problem = _problem(config)
    initial, initialization = independent_initial_states(
        problem.conditioned, config["replicas"], config["initialization_seed"]
    )
    free = list(problem.conditioned.clamp.free_indices)
    root_indices = [
        free.index(i)
        for i in (*problem.compiled.factor_a_indices, *problem.compiled.factor_b_indices)
    ]
    roots = initial[:, root_indices].cuda()
    cls, namespace = SAMPLERS[config["modular_root_mode"]]
    sampler = cls(
        config["product_bits"] // 2, config["target"], "cuda", mode=config["modular_root_mode"]
    )
    proxy = SimpleNamespace(
        device=torch.device("cuda"),
        root_count=config["product_bits"],
        root_policy=config["root_policy"],
        root_guidance=config["root_guidance"],
        modular_root_mode=config["modular_root_mode"],
        modular_sampler=sampler,
        group_count=len(problem.compiled.components.gates),
    )
    from .solver.execution_profiles import configure

    proxy.reuse_graph_noise = configure(sampler, config.get("execution_profile", "reference_v1"))
    generator = torch.Generator(device="cuda").manual_seed(
        derive_stream_seed(config["circuit_seed"], namespace)
    )
    return roots, sampler, proxy, generator, initialization


class SelectedGraph:
    def __init__(self, sampler, roots, words):
        self.words = words.clone()

        def calculate():
            proposed, details = sampler.propose_filtered(roots, self.words)
            pairs = sampler.decode(proposed)
            matches = exact_product_match(pairs, sampler.target) & details["valid"]
            counts = torch.stack((details["valid"], details["eligible_count"], matches), 1).long()
            return pairs, counts

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                calculate()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = calculate()
        torch.cuda.current_stream().wait_stream(stream)

    def run(self, words):
        self.words.copy_(words)
        self.graph.replay()
        return self.output


@torch.inference_mode()
def run_arithmetic(config: dict, output: Path, source_sha256: str) -> dict:
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)
    started = time.perf_counter()
    roots, sampler, proxy, generator, initialization = setup(config)
    rng = generator.get_state()
    noise = CircuitDirectedProposal.draw_noise(proxy, len(roots), generator)
    graph = SelectedGraph(sampler, roots, noise.modular_uniforms)
    generator.set_state(rng)
    counts = torch.zeros((len(roots), 3), dtype=torch.int64, device="cuda")
    events = 0
    host_roots = roots.cpu().numpy()

    def snapshot():
        return {
            "states": host_roots.copy(),
            "best_states": host_roots.copy(),
            "circuit_rng_state": generator.get_state().numpy().copy(),
            "circuit_events": np.asarray(events, dtype=np.int64),
            "arithmetic_counts": counts.cpu().numpy().copy(),
        }

    writer = ChunkWriter(
        output / "stream",
        {
            "run_id": config["run_id"],
            "arm": "selected_root_integer",
            "metric_columns": 3,
            "state_kind": "initial_factor_roots_only_not_physical_states",
            "config": config,
            "config_sha256": canonical_digest(config),
            "source_sha256": source_sha256,
            "maximum_events": config["maximum_events"],
            "replicas": len(roots),
            "free_spins": config["product_bits"],
        },
        snapshot(),
    )
    capacity = min(config["chunk_events"], config["maximum_events"])
    buffer = torch.empty((capacity, 32, 3), dtype=torch.int64, device="cuda")
    filled, times, certificates = 0, [], []
    setup_seconds = time.perf_counter() - started
    begin = time.perf_counter()
    status, first_time, first_event = "event_limit", None, None

    def seal():
        nonlocal filled, times
        if filled:
            writer.submit(
                events,
                {
                    "checkpoint.npz": snapshot(),
                    "metrics.npz": {
                        "cumulative_by_position": buffer[:filled].cpu().numpy().copy(),
                        "event_wall": np.asarray(times),
                    },
                    "first_hits.npz": {"selected_integer_hits": counts[:, 2].cpu().numpy().copy()},
                },
            )
            filled, times = 0, []

    try:
        for event in range(1, config["maximum_events"] + 1):
            if time.perf_counter() - begin >= config["maximum_sampling_seconds"]:
                status = "sampling_deadline"
                break
            if torch.cuda.memory_reserved() > config["maximum_reserved_vram_gib"] * 1024**3:
                raise RuntimeError("Frozen GPU memory limit exceeded")
            lower = time.perf_counter() - begin
            noise = CircuitDirectedProposal.draw_noise(
                proxy,
                len(roots),
                generator,
                modular_out=graph.words if proxy.reuse_graph_noise else None,
            )
            pairs, selected = graph.run(noise.modular_uniforms)
            counts += selected
            buffer[filled].copy_(counts.reshape(-1, 32, 3).sum(0))
            locations = torch.nonzero(selected[:, 2]).flatten().cpu().tolist()
            upper = time.perf_counter() - begin
            events, filled = event, filled + 1
            times.append((lower, upper, upper))
            for slot in locations:
                a, b = (int(v) & ((1 << 64) - 1) for v in pairs[slot].cpu().tolist())
                if a <= 1 or b <= 1 or a * b != config["target"]:
                    raise ValueError("Integer control certificate failed")
                certificates.append({"event": event, "slot": slot, "factor_a": a, "factor_b": b})
            if locations:
                first_event, first_time = event, upper
                status = (
                    "integer_hit" if upper <= config["maximum_sampling_seconds"] else "late_hit"
                )
            if filled == capacity or locations:
                seal()
            if locations:
                break
    except BaseException as error:
        status = "failed"
        atomic_json(output / "failure.json", {"error": repr(error)})
        seal()
        writer.close(status)
        raise
    finally:
        seal()
        writer.drain()
    summary = {
        "run_id": config["run_id"],
        "arm": "selected_root_integer",
        "target": config["target"],
        "product_bits": config["product_bits"],
        "events": events,
        "status": status,
        "success": status == "integer_hit",
        "first_event": first_event,
        "first_hit_seconds": first_time,
        "sampling_seconds": time.perf_counter() - begin,
        "setup_seconds": setup_seconds,
        "cell_wall_seconds": time.perf_counter() - started,
        "certificates": certificates,
        "initialization": initialization,
        "source_sha256": source_sha256,
        "independent_trials": 0,
        "limitations": "Same full noise layout; selected candidates only; integer-only success",
    }
    atomic_json(output / "summary.json", summary)
    writer.close(status, summary)
    return summary
