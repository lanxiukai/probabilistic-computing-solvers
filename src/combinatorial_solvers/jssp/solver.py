"""Bounded CUDA job-shop search with cold states, state histories and work replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gpu import CudaJobShop


@dataclass(frozen=True)
class Settings:
    replicas: int = 64
    seed: int = 2026102301
    width: int = 2
    method: str = "heatbath"
    decoder: str = "reduce"
    schedule_decoder: str = "append"
    beta_start: float = 0.001
    beta_end: float = 0.1
    cycle_steps: int = 1024
    chunk_steps: int = 8
    window: int | None = None
    max_steps: int = 100000000
    max_seconds: float = 10
    snapshot_every_chunks: int = 32
    checkpoint_interval_seconds: float | None = None
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
        if self.replicas > 1024 or self.width not in (2, 4):
            raise ValueError("Unsupported replica count or block width")
        if self.method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown update method")
        if self.decoder not in {"reduce", "gather"}:
            raise ValueError("Unknown exact decoder implementation")
        if self.schedule_decoder not in {"append", "active"}:
            raise ValueError("Unknown schedule decoder")
        if self.schedule_decoder == "active" and self.width != 2:
            raise ValueError("The active decoder currently supports two-position blocks only")
        if not 0 < self.beta_start <= self.beta_end < float(
            "inf"
        ) or not 0 < self.max_seconds < float("inf"):
            raise ValueError("Invalid temperature endpoints or time limit")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid replay quality boundary")
        if self.checkpoint_interval_seconds is not None and (
            isinstance(self.checkpoint_interval_seconds, bool)
            or not isinstance(self.checkpoint_interval_seconds, (int, float))
            or not 0 < self.checkpoint_interval_seconds < float("inf")
        ):
            raise ValueError("The checkpoint interval must be positive and finite")
        if self.window is not None and (
            isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1
        ):
            raise ValueError("The position window must be a positive integer")


def solve(instance, settings: Settings, *, evidence_directory=None):
    settings.validate()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    prepared = perf_counter()
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    initial = np.array(
        [
            rng.permutation(np.repeat(np.arange(instance.jobs), instance.machine_count))
            for _ in range(settings.replicas)
        ]
    )
    evidence = Path(evidence_directory) if evidence_directory is not None else None
    if evidence is not None:
        evidence.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(evidence / "initial-state.npz", initial_sequence=initial)
        (evidence / "initialization.json").write_text(
            json.dumps(
                {
                    "settings": asdict(settings),
                    "numpy_rng_after_initialization": rng.bit_generator.state,
                },
                indent=2,
            )
            + "\n"
        )
    engine = CudaJobShop(instance, initial, settings.width, settings.schedule_decoder)
    initial_cost = engine.cost.cpu().numpy()
    arguments = dict(
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        cycle_steps=settings.cycle_steps,
        window=settings.window,
        method=settings.method,
        decoder=settings.decoder,
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

    def checkpoint(label):
        if evidence is None:
            return
        destination = evidence / ("checkpoint-" + label + ".npz")
        committed_step, committed_rng = committed_boundary
        saved_quality_step, saved_best, saved_cost = quality_snapshot
        # The committed CPU-side boundary survives a failed device-state transfer.
        np.savez_compressed(
            evidence / ("boundary-" + label + ".npz"),
            committed_steps=np.array(committed_step),
            committed_rng=committed_rng,
            quality_max_step=np.array(saved_quality_step),
            quality_best_sequence=saved_best,
            quality_best_cost=saved_cost,
            transition_pending=np.array(transition_pending),
        )
        pending = {}
        if transition_pending and pending_noise is not None:
            pending["pending_noise"] = pending_noise.cpu().numpy()
        np.savez_compressed(
            destination,
            **engine.state(),
            quality_best_sequence=saved_best,
            quality_best_cost=saved_cost,
            completed_steps=np.array(committed_step),
            quality_max_step=np.array(saved_quality_step),
            torch_rng=generator.get_state().cpu().numpy(),
            committed_rng=committed_rng,
            state_boundary_known=np.array(not transition_pending),
            **pending,
        )
        (evidence / "recovery.json").write_text(
            json.dumps(
                {
                    "checkpoint": destination.name,
                    "steps": committed_step,
                    "quality_max_step": saved_quality_step,
                    "seed": settings.seed,
                    "state_boundary_known": not transition_pending,
                    "scope": (
                        "A pending transition may have partially changed captured device state; "
                        "use the last committed boundary and preserved noise for recovery auditing"
                        if transition_pending
                        else "Complete state and RNG boundary"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    curve_file = (evidence / "curve.jsonl").open("a") if evidence is not None else None
    try:
        checkpoint("initial")
        warm = CudaJobShop(instance, initial, settings.width, settings.schedule_decoder)
        noise_shape = (settings.width + 2, settings.replicas, settings.chunk_steps)
        warm.advance(
            torch.full(noise_shape, 0.5, dtype=torch.float64, device="cuda"),
            steps=settings.chunk_steps,
            **arguments,
        )
        torch.cuda.synchronize()
        del warm
        preparation_seconds = perf_counter() - prepared
        started = perf_counter()
        last_checkpoint_finished = started
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
            if curve_file is not None:
                curve_file.write(
                    json.dumps(
                        {
                            "steps": steps,
                            "seconds": elapsed,
                            "cost": cost.tolist(),
                            "best_cost": prefix_best.tolist(),
                            "eligible": accepted,
                        }
                    )
                    + "\n"
                )
                curve_file.flush()
            coarse_snapshot = chunks % settings.snapshot_every_chunks == 0
            if coarse_snapshot:
                snapshots.append(engine.sequence.cpu().numpy())
                best_snapshots.append(engine.best.cpu().numpy())
                snapshot_steps.append(steps)
            timed_checkpoint = (
                evidence is not None
                and settings.checkpoint_interval_seconds is not None
                and perf_counter() - last_checkpoint_finished
                >= settings.checkpoint_interval_seconds
            )
            if coarse_snapshot or timed_checkpoint:
                checkpoint(str(steps))
                last_checkpoint_finished = perf_counter()
        search_seconds = perf_counter() - started
        checkpoint("final")
    except BaseException as error:
        if evidence is not None:
            try:
                checkpoint("partial")
            except BaseException as recovery_error:
                (evidence / "recovery-failure.json").write_text(
                    json.dumps(
                        {"original_error": repr(error), "recovery_error": repr(recovery_error)}
                    )
                    + "\n"
                )
        raise
    finally:
        if curve_file is not None:
            curve_file.close()
    state = engine.state()
    arrays = {
        "initial_sequence": initial,
        "initial_cost": initial_cost,
        "quality_best_sequence": best,
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
        "physical_update_pbits_per_model": instance.size * instance.jobs
        if settings.method == "heatbath"
        else 0,
        "model_binary_variables": instance.size * instance.jobs,
        "parallel_replicas": settings.replicas,
        "initialization": "independent_uniform_job_multiset_sequences",
        "classical_support": (
            "serial_append_decoder"
            if settings.schedule_decoder == "append"
            else "earliest_gap_insertion_decoder"
        ),
        "initial_best_cost": int(initial_cost.min()),
        "best_feasible_cost": int(best_cost.min()),
        "returned_best_cost": int(state["best_cost"].min()),
        "replicas_improved": int((best_cost < initial_cost).sum()),
        "pbit_incremental_best_reduction": int(initial_cost.min() - best_cost.min())
        if settings.method == "heatbath"
        else None,
        "completed_steps": steps,
        "quality_max_step": quality_step,
        "preparation_seconds": preparation_seconds,
        "search_seconds": search_seconds,
        "deadline_overshoot_seconds": max(0, search_seconds - settings.max_seconds),
        "visited_assignment_bits_min": int(state["visited"].sum(axis=(1, 2)).min()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "observation": (
            "per-chunk best of every accepted transition; deadline crossing chunk excluded"
        ),
        "checkpoint_observation": (
            "Optional wall-clock checkpoints occur at the next completed chunk boundary; "
            "their interval starts at search start or the previous checkpoint completion. "
            "They preserve full state and RNG without adding coarse trajectory snapshots."
        ),
    }
