"""Bounded CUDA covering-array search with cold states, state histories and work replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gpu import CudaCoveringArray


@dataclass(frozen=True)
class Settings:
    replicas: int = 64
    seed: int = 2026102301
    width: int = 1
    orientation: str = "row"
    cache_parameters: bool = True
    prewarm_full_chunks: bool = True
    method: str = "heatbath"
    beta_start: float = 0.1
    beta_end: float = 8.0
    cycle_steps: int = 4096
    chunk_steps: int = 256
    max_steps: int = 100000000
    max_seconds: float = 10
    snapshot_every_chunks: int = 32
    quality_max_step: int | None = None

    def validate(self):
        for name in (
            "replicas",
            "seed",
            "width",
            "cycle_steps",
            "chunk_steps",
            "max_steps",
            "snapshot_every_chunks",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError("Invalid integer setting: " + name)
        if self.replicas > 1024 or self.width not in (1, 2):
            raise ValueError("Unsupported replica count or block width")
        if self.method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown update method")
        if self.orientation not in {"row", "column"} or (
            self.orientation == "column" and self.width != 2
        ):
            raise ValueError("Invalid covering-array block orientation")
        if not isinstance(self.cache_parameters, bool) or not isinstance(
            self.prewarm_full_chunks, bool
        ):
            raise ValueError("Implementation switches must be boolean")
        if not 0 < self.beta_start <= self.beta_end < float(
            "inf"
        ) or not 0 < self.max_seconds < float("inf"):
            raise ValueError("Invalid temperature endpoints or time limit")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid replay quality boundary")


def solve(instance, settings: Settings, *, evidence_directory: Path | None = None):
    settings.validate()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    prepared = perf_counter()
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    initial = rng.integers(
        0,
        instance.levels,
        size=(settings.replicas, instance.rows, instance.columns),
        dtype=np.int64,
    )
    if evidence_directory is not None:
        np.savez_compressed(evidence_directory / "initial-state.npz", values=initial)
    engine = CudaCoveringArray(
        instance,
        initial,
        settings.width,
        orientation=settings.orientation,
        cache_parameters=settings.cache_parameters,
    )
    initial_cost = engine.cost.cpu().numpy()
    arguments = dict(
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        cycle_steps=settings.cycle_steps,
        method=settings.method,
    )
    generator = torch.Generator(device="cuda").manual_seed(settings.seed ^ 0x24680)
    initial_rng = generator.get_state().cpu().numpy()
    best = initial.copy()
    best_cost = initial_cost.copy()
    quality_snapshot = (0, best, best_cost)
    committed_boundary = (0, initial_rng.copy())
    transition_pending = False
    pending_noise = None
    snapshots, best_snapshots, snapshot_steps = [initial.copy()], [initial.copy()], [0]
    curve_steps, curve_seconds, curve_cost, curve_best, eligible = [], [], [], [], []
    steps, chunks, quality_step = 0, 0, 0

    def checkpoint(name):
        if evidence_directory is None:
            return
        committed_step, committed_rng = committed_boundary
        saved_quality_step, saved_best, saved_cost = quality_snapshot
        # Save the last atomic CPU boundary even if device-state capture fails.
        np.savez_compressed(
            evidence_directory / ("boundary-" + name),
            committed_step=np.array(committed_step),
            committed_rng=committed_rng,
            quality_step=np.array(saved_quality_step),
            quality_best_values=saved_best,
            quality_best_cost=saved_cost,
            transition_pending=np.array(transition_pending),
        )
        pending = {}
        if transition_pending and pending_noise is not None:
            pending["pending_noise"] = pending_noise.cpu().numpy()
        np.savez_compressed(
            evidence_directory / name,
            values=engine.values.cpu().numpy(),
            bits=engine.bits.cpu().numpy(),
            best_values=engine.best.cpu().numpy(),
            cost=engine.cost.cpu().numpy(),
            best_cost=engine.best_cost.cpu().numpy(),
            visited=engine.visited.cpu().numpy(),
            changed=engine.changed.cpu().numpy(),
            quality_best_values=saved_best,
            quality_best_cost=saved_cost,
            step=np.array(committed_step),
            quality_step=np.array(saved_quality_step),
            rng=generator.get_state().cpu().numpy(),
            committed_rng=committed_rng,
            state_boundary_known=np.array(not transition_pending),
            **pending,
        )

    try:
        checkpoint("checkpoint-initial.npz")
        warm = CudaCoveringArray(
            instance,
            initial,
            settings.width,
            orientation=settings.orientation,
            cache_parameters=settings.cache_parameters,
        )
        noise_shape = (5, settings.replicas, settings.chunk_steps)
        warm_noise = torch.full(noise_shape, 0.5, dtype=torch.float64, device="cuda")
        if settings.prewarm_full_chunks:
            first = min(settings.chunk_steps, settings.max_steps)
            for offset in (0, first, 2 * first, 16 * first):
                warm.advance(warm_noise, steps=first, offset=offset, **arguments)
            tail = settings.max_steps % settings.chunk_steps
            if tail:
                warm.advance(warm_noise, steps=tail, offset=settings.max_steps - tail, **arguments)
        else:
            warm.advance(warm_noise, steps=1, **arguments)
        torch.cuda.synchronize()
        del warm
        preparation_seconds = perf_counter() - prepared
        started = perf_counter()
        while steps < settings.max_steps and perf_counter() - started < settings.max_seconds:
            count = min(settings.chunk_steps, settings.max_steps - steps)
            transition_pending = True
            noise = torch.rand(noise_shape, dtype=torch.float64, device="cuda", generator=generator)
            pending_noise = noise
            engine.advance(noise, offset=steps, steps=count, **arguments)
            torch.cuda.synchronize()
            committed_boundary = (steps + count, generator.get_state().cpu().numpy())
            steps = committed_boundary[0]
            transition_pending = False
            pending_noise = None
            chunks += 1
            cost = engine.cost.cpu().numpy()
            prefix_best = engine.best_cost.cpu().numpy()
            elapsed = perf_counter() - started
            accepted = elapsed <= settings.max_seconds and (
                settings.quality_max_step is None or steps <= settings.quality_max_step
            )
            if accepted:
                quality_snapshot = (steps, engine.best.cpu().numpy(), prefix_best.copy())
                quality_step, best, best_cost = quality_snapshot
            curve_steps.append(steps)
            curve_seconds.append(elapsed)
            curve_cost.append(cost)
            curve_best.append(prefix_best)
            eligible.append(accepted)
            if evidence_directory is not None:
                with (evidence_directory / "observations.jsonl").open("a") as trace:
                    trace.write(
                        json.dumps(
                            {
                                "step": steps,
                                "seconds": elapsed,
                                "eligible": accepted,
                                "cost": cost.tolist(),
                                "best_cost": prefix_best.tolist(),
                            }
                        )
                        + "\n"
                    )
            if chunks % settings.snapshot_every_chunks == 0:
                snapshots.append(engine.values.cpu().numpy())
                best_snapshots.append(engine.best.cpu().numpy())
                snapshot_steps.append(steps)
                checkpoint(f"checkpoint-{steps:012d}.npz")
    except BaseException as error:
        if evidence_directory is not None:
            (evidence_directory / "sampling-failure.json").write_text(
                json.dumps(
                    {
                        "error": repr(error),
                        "step": committed_boundary[0],
                        "quality_step": quality_snapshot[0],
                        "state_boundary_known": not transition_pending,
                        "scope": (
                            "Pending device state is diagnostic; recover from the last committed "
                            "boundary by full-origin fixed-work replay, not by direct resume"
                        ),
                    }
                )
                + "\n"
            )
            try:
                checkpoint("checkpoint-interrupted.npz")
            except BaseException as capture_error:
                (evidence_directory / "capture-failure.json").write_text(
                    json.dumps({"error": repr(capture_error)}) + "\n"
                )
        raise
    search_seconds = perf_counter() - started
    checkpoint("checkpoint-final.npz")
    state = engine.state()
    arrays = {
        "initial_values": initial,
        "initial_cost": initial_cost,
        "quality_best_values": best,
        "quality_best_cost": best_cost,
        "snapshot_steps": np.array(snapshot_steps),
        "snapshots": np.array(snapshots),
        "best_snapshots": np.array(best_snapshots),
        "curve_steps": np.array(curve_steps),
        "curve_seconds": np.array(curve_seconds),
        "curve_cost": np.array(curve_cost),
        "curve_best": np.array(curve_best),
        "curve_eligible": np.array(eligible),
        "initial_rng": initial_rng,
        "final_rng": generator.get_state().cpu().numpy(),
        **state,
    }
    return arrays, {
        "method": settings.method,
        "settings": asdict(settings),
        "pbit_heatbath_participation": settings.method == "heatbath",
        "physical_update_pbits_per_model": instance.size * instance.levels
        if settings.method == "heatbath"
        else 0,
        "model_binary_variables": instance.size * instance.levels,
        "parallel_replicas": settings.replicas,
        "initialization": "independent_uniform_array_entries",
        "initial_best_cost": int(initial_cost.min()),
        "best_feasible_cost": int(best_cost.min()),
        "returned_best_cost": int(state["best_cost"].min()),
        "replicas_improved": int((best_cost < initial_cost).sum()),
        "completed_steps": steps,
        "quality_max_step": quality_step,
        "preparation_seconds": preparation_seconds,
        "search_seconds": search_seconds,
        "deadline_overshoot_seconds": max(0, search_seconds - settings.max_seconds),
        "visited_assignment_bits_min": int(state["visited"].sum(axis=(1, 2, 3)).min()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "observation": (
            "per-chunk best of every accepted transition; deadline crossing chunk excluded"
        ),
    }
