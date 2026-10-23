"""Stream-following origin replay; restored checkpoints are a separate audit mode."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .runner import checkpoint
from .solver.publication_io import atomic_json, read_json
from .solver.publication_runtime import make_engine
from .stream import arrays, validate_segment


def compare_arrays(actual: dict, expected: dict) -> int:
    if set(actual) != set(expected):
        raise ValueError("Checkpoint field set differs")
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"Replay differs at {key}")
    return len(actual)


@torch.inference_mode()
def replay_stream(stream: Path, output: Path, max_seconds: float, *, watch=False) -> dict:
    from .solver.publication_io import digest

    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Replay needs its own positive finite wall budget")
    output.mkdir(parents=True, exist_ok=False)
    identity = read_json(stream / "identity.json")
    from .repro import source_identity

    if source_identity(Path(__file__).resolve().parents[2])["sha256"] != identity["source_sha256"]:
        raise ValueError("Origin replay must use the exact archived execution source")
    if identity.get("arm"):
        raise ValueError("Origin replay here applies to physical trajectories")
    config = identity["config"]
    _, engine, betas, circuit, _ = make_engine(config)
    initial = engine.snapshot()
    rng = circuit.generator.get_state()
    for _ in range(circuit.warmup_steps):
        circuit.step(betas)
        circuit.after_observation()
    circuit.generator.set_state(rng)
    circuit.reset_statistics()
    engine.restore(initial)
    validate_segment(stream / "initial", identity, None, 0)
    checked = compare_arrays(checkpoint(engine, circuit), arrays(stream / "initial/checkpoint.npz"))
    previous = digest(stream / "initial/manifest.json")
    index, start = 1, 0
    begun = time.perf_counter()
    status = "partial"
    try:
        while True:
            if time.perf_counter() - begun >= max_seconds:
                status = "replay_deadline"
                break
            path = stream / "chunks" / f"chunk-{index:06d}"
            if not path.exists():
                if (stream / "terminal.json").exists():
                    final = read_json(stream / "terminal.json")
                    if final["events"] != start or final["chunks"] != index - 1:
                        raise ValueError("Replay ended before complete event coverage")
                    status = "complete"
                    break
                if not watch:
                    break
                time.sleep(1)
                continue
            manifest = validate_segment(path, identity, previous, start)
            expected = arrays(path / "metrics.npz")["cumulative_by_position"]
            for row in expected:
                if time.perf_counter() - begun >= max_seconds:
                    raise TimeoutError("Replay budget exhausted inside a segment")
                circuit.step(betas)
                actual = circuit.arithmetic_counts.reshape(-1, 32, 19).sum(0).cpu().numpy()
                if not np.array_equal(actual, row):
                    raise ValueError("Per-event arithmetic counters differ")
                if not np.any(circuit.first_hit_sweeps_host >= 0) or not config.get(
                    "stop_on_first_native_hit", True
                ):
                    circuit.after_observation()
            checked += compare_arrays(checkpoint(engine, circuit), arrays(path / "checkpoint.npz"))
            previous = digest(path / "manifest.json")
            start = manifest["end_event_inclusive"]
            atomic_json(
                output / f"chunk-{index:06d}.json",
                {"events": start, "manifest_sha256": previous, "non_timing_arrays_equal": True},
            )
            index += 1
    except BaseException as error:
        status = "replay_deadline" if isinstance(error, TimeoutError) else "failed"
        atomic_json(output / "failure.json", {"error": repr(error), "completed_events": start})
        raise
    finally:
        result = {
            "status": status,
            "completed_events": start,
            "chunks": index - 1,
            "array_comparisons": checked,
            "additional_independent_trials": 0,
            "elapsed_seconds": time.perf_counter() - begun,
            "full_origin_replay": status == "complete",
        }
        atomic_json(output / "summary.json", result)
    return result
