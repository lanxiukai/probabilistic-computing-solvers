"""Bounded production-kernel runs with immutable streaming checkpoints."""

from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
import torch

from .solver.publication_events import NativeEventRecorder
from .solver.publication_io import atomic_json, canonical_digest
from .solver.publication_runtime import IndependentCertificateContext, _problem, make_engine
from .stream import ChunkWriter

MODES = {
    "quotient_filter256_v1",
    "fermat_filter256_v1",
    "fermat_crt_filter256_v1",
    "fermat_crt128_filter256_v1",
    "fermat_deferred262144_v1",
}


def validate_config(config: dict) -> None:
    width = config["product_bits"]
    if not isinstance(width, int) or width < 4 or width > 128 or width % 2:
        raise ValueError("Require even product widths in [4, 128]")
    if config["target"].bit_length() != width or config["target"] % 2 == 0:
        raise ValueError("Require an odd target with the declared product bit length")
    if config["topology"] != "carry_chain" or config["model_storage"] != "edge_v1":
        raise ValueError("Release runs use compact sparse carry_chain and edge storage")
    if config["algorithm"] != "circuit_only" or config["exchange_every"] != 0:
        raise ValueError("Release protocol uses circuit MH without PT")
    if config["modular_root_mode"] not in MODES:
        raise ValueError("Unsupported release proposal")
    if config["replicas"] <= 0 or config["replicas"] % 32 or config["temperature_slots"] != 32:
        raise ValueError("Require positive populations divisible by 32")
    if config["beta"] != [12.0, 12.0]:
        raise ValueError("Release protocol freezes beta=12")
    for key in ("maximum_events", "chunk_events"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError("Require positive integer event limits")
    for key in ("maximum_sampling_seconds", "maximum_reserved_vram_gib"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Require finite positive resource limits")
    for key in ("initialization_seed", "sampler_seed", "circuit_seed", "exchange_seed"):
        if not isinstance(config[key], int) or not 0 <= config[key] < 2**63:
            raise ValueError("Invalid role seed")


def checkpoint(engine, circuit) -> dict:
    return {
        **engine.snapshot(),
        "circuit_rng_state": circuit.generator.get_state().numpy().copy(),
        "circuit_events": np.asarray(circuit.events, dtype=np.int64),
        "circuit_uniform_draws": np.asarray(circuit.uniform_draws, dtype=np.int64),
        "arithmetic_counts": circuit.arithmetic_counts.cpu().numpy().copy(),
        "accepted_by_replica": circuit.accepted_by_replica.cpu().numpy().copy(),
        "roots_by_replica": circuit.roots_by_replica.cpu().numpy().copy(),
    }


def restore_checkpoint(engine, circuit, saved: dict) -> None:
    """Restore a completed no-PT boundary, invalidating only the derived cache."""
    engine.restore(saved)
    circuit.reset_statistics()
    circuit.generator.set_state(torch.from_numpy(saved["circuit_rng_state"]))
    circuit.events = int(saved["circuit_events"])
    circuit.uniform_draws = int(saved["circuit_uniform_draws"])
    for name in ("arithmetic_counts", "accepted_by_replica", "roots_by_replica"):
        getattr(circuit, name).copy_(torch.as_tensor(saved[name], device="cuda"))
    circuit.first_hit_sweeps_host = saved["first_hit_sweeps"].copy()


def first_hits(recorder) -> dict:
    return {
        "first_hit_sweeps": recorder.engine.first_hit_sweeps.cpu().numpy().copy(),
        "first_hit_kind": recorder.kinds.copy(),
        "first_hit_auxiliary_event": recorder.events.copy(),
        "first_hit_transition": recorder.transitions.copy(),
        "first_hit_wall_lower": recorder.lower.copy(),
        "first_hit_wall_upper": recorder.upper.copy(),
    }


def certificate_verifier(config: dict):
    """Build the independent CPU verifier lazily, only when a witness arrives."""
    context = None

    def certify(packet: dict) -> list[dict]:
        nonlocal context
        if context is None:
            context = IndependentCertificateContext.from_problem(_problem(config).conditioned)
        result = []
        for index, state in enumerate(packet["witness_state"]):
            checked = context.certify(torch.from_numpy(state), target=config["target"], tolerance=0)
            if (
                not checked["certified"]
                or checked["full_energy_fp64"] != 0
                or checked["conditioned_energy_fp64"] != 0
            ):
                raise ValueError("Strict witness certificate failed")
            result.append(
                {
                    "slot": int(packet["witness_slot"][index]),
                    "event": int(packet["witness_auxiliary_event"][index]),
                    **checked,
                }
            )
        return result

    return certify


@torch.inference_mode()
def run_physical(config: dict, output: Path, source_sha256: str) -> dict:
    validate_config(config)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)
    sparse, engine, betas, circuit, initialization = make_engine(config)
    initial = engine.snapshot()
    rng = circuit.generator.get_state()
    for _ in range(circuit.warmup_steps):
        circuit.step(betas)
        circuit.after_observation()
    circuit.generator.set_state(rng)
    circuit.reset_statistics()
    engine.restore(initial)
    torch.cuda.synchronize()
    recorder = NativeEventRecorder(engine, betas, circuit, 32, context_layout="selected_slots_v1")
    recorder.observe(recorder.context(), "initial", 0.0, 0.0, 0)
    identity = {
        "run_id": config["run_id"],
        "config_sha256": canonical_digest(config),
        "source_sha256": source_sha256,
        "config": config,
        "maximum_events": config["maximum_events"],
        "replicas": engine.replicas,
        "free_spins": engine.variables,
        "full_spins": sparse.compiled.model.size,
        "geometry": circuit.proposal.geometry,
        "circuit_rng": {"namespace": circuit.stream_name, "effective_seed": circuit.rng_seed},
        "timing": "sampling wall includes capture and compression backpressure; setup separate",
    }
    initial_files = {"first_hits.npz": first_hits(recorder)}
    if recorder.witnesses:
        initial_files["witnesses.npz"] = recorder.packet()
    writer = ChunkWriter(output / "stream", identity, checkpoint(engine, circuit), initial_files)
    capacity = min(config["chunk_events"], config["maximum_events"])
    counts = torch.empty((capacity, 32, 19), dtype=torch.int64, device="cuda")
    times, filled = [], 0
    capture_seconds = 0.0
    setup_seconds = time.perf_counter() - started
    begin = time.perf_counter()
    status = "event_limit"
    interrupted = False

    def stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGTERM, stop)

    def seal():
        nonlocal filled, times, capture_seconds
        if filled == 0:
            return
        before = time.perf_counter()
        payload = {
            "checkpoint.npz": checkpoint(engine, circuit),
            "metrics.npz": {
                "cumulative_by_position": counts[:filled].cpu().numpy().copy(),
                "event_wall": np.asarray(times, dtype=np.float64),
            },
            "first_hits.npz": first_hits(recorder),
        }
        if recorder.witnesses:
            payload["witnesses.npz"] = recorder.packet()
        writer.submit(circuit.events, payload)
        filled, times = 0, []
        capture_seconds += time.perf_counter() - before

    error = None
    torch.cuda.reset_peak_memory_stats()
    try:
        if recorder.group_seen.any():
            status = "initial_hit"
        else:
            for event in range(1, config["maximum_events"] + 1):
                if interrupted:
                    status = "interrupted"
                    break
                if time.perf_counter() - begin >= config["maximum_sampling_seconds"]:
                    status = "sampling_deadline"
                    break
                if torch.cuda.memory_reserved() > config["maximum_reserved_vram_gib"] * 1024**3:
                    raise RuntimeError("Frozen GPU memory limit exceeded")
                before = recorder.context()
                lower = time.perf_counter() - begin
                circuit.step(betas)
                upper = time.perf_counter() - begin
                recorder.observe(before, "circuit", lower, upper, event)
                counts[filled].copy_(circuit.arithmetic_counts.reshape(-1, 32, 19).sum(0))
                hit = bool(recorder.group_seen.any())
                stopping_hit = hit and config.get("stop_on_first_native_hit", True)
                if not stopping_hit:
                    circuit.after_observation()
                filled += 1
                times.append((lower, upper, time.perf_counter() - begin))
                if stopping_hit:
                    status = (
                        "strict_hit" if upper <= config["maximum_sampling_seconds"] else "late_hit"
                    )
                if filled == capacity or stopping_hit:
                    seal()
                if stopping_hit:
                    break
    except BaseException as caught:
        status = "interrupted" if isinstance(caught, KeyboardInterrupt) else "failed"
        error = caught
        atomic_json(output / "failure.json", {"phase": "sampling", "error": repr(caught)})
    finally:
        signal.signal(signal.SIGTERM, previous)
        # Completed events are preserved even when a later update fails.
        if filled and circuit.events != writer.last_event + filled:
            from .solver.publication_io import atomic_npz

            atomic_npz(
                output / "interrupted_metrics.npz",
                cumulative_by_position=counts[:filled].cpu().numpy(),
                event_wall=np.asarray(times),
            )
            atomic_npz(output / "interrupted_state.npz", **checkpoint(engine, circuit))
            filled = 0
        if filled and len(times) == filled:
            seal()
        torch.cuda.synchronize()
        sampling_seconds = time.perf_counter() - begin
        writer.drain()
    certificates = []
    certification_started = time.perf_counter()
    try:
        if recorder.witnesses and error is None:
            from .solver.publication_runtime import replay_packet

            certificates = replay_packet(config, recorder.packet())
            if not all(
                c["certified"]
                and c["transition_replay_passed"]
                and c["full_energy_fp64"] == c["conditioned_energy_fp64"] == 0
                for c in certificates
            ):
                raise ValueError("Independent transition certificate failed")
        peak = torch.cuda.max_memory_reserved()
        if peak > config["maximum_reserved_vram_gib"] * 1024**3:
            raise RuntimeError("Total reserved-memory cap exceeded during certification")
    except BaseException:
        writer.close("certification_failed")
        raise
    summary = {
        "run_id": config["run_id"],
        "arm": "physical",
        "target": config["target"],
        "product_bits": config["product_bits"],
        "status": status,
        "events": circuit.events,
        "success": status in {"strict_hit", "initial_hit"} and bool(certificates),
        "first_event": int(recorder.events[recorder.seen].min()) if recorder.seen.any() else None,
        "first_hit_seconds": float(recorder.upper[recorder.seen].min())
        if recorder.seen.any()
        else None,
        "sampling_seconds": sampling_seconds,
        "setup_seconds": setup_seconds,
        "capture_seconds": capture_seconds,
        "certification_seconds": time.perf_counter() - certification_started,
        "cell_wall_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": peak,
        "witnesses": certificates,
        "initialization": initialization,
        "source_sha256": source_sha256,
        "independent_trials": int(
            not config.get("replay", False) and not config.get("cost_only", False)
        ),
        "transition_replay": "passed" if certificates else "no_witness",
        "full_origin_replay": False,
    }
    atomic_json(output / "summary.json", summary)
    writer.close(status, summary)
    if error is not None:
        raise error
    return summary
