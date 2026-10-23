"""Bounded CUDA QAP search with cold states, state histories and work replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gpu import CudaQAP


@dataclass(frozen=True)
class Settings:
    replicas: int = 64
    seed: int = 2026102301
    width: int = 2
    num_warps: int = 4
    energy_layout: str = "candidate"
    method: str = "heatbath"
    beta_start: float = 0.001
    beta_end: float = 0.1
    cycle_steps: int = 4096
    chunk_steps: int = 256
    max_steps: int = 100000000
    max_seconds: float = 10
    snapshot_every_chunks: int = 32
    checkpoint_seconds: float = 2.0
    quality_max_step: int | None = None

    def validate(self):
        for name in (
            "replicas",
            "seed",
            "width",
            "num_warps",
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
        if self.replicas > 1024 or self.width not in (2, 4) or self.num_warps not in (4, 8, 16):
            raise ValueError("Unsupported replica count or block width")
        if self.method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown update method")
        if self.energy_layout not in {"candidate", "factored"}:
            raise ValueError("Unknown local energy evaluation layout")
        if not 0 < self.beta_start <= self.beta_end < float(
            "inf"
        ) or not 0 < self.max_seconds < float("inf"):
            raise ValueError("Invalid temperature endpoints or time limit")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid replay quality boundary")
        if not 0 < self.checkpoint_seconds < float("inf"):
            raise ValueError("Checkpoint interval must be positive and finite")


def solve(instance, settings: Settings, *, evidence_directory=None):
    settings.validate()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    prepared = perf_counter()
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    initial = np.array([rng.permutation(instance.size) for _ in range(settings.replicas)])
    evidence = Path(evidence_directory) if evidence_directory is not None else None
    if evidence is not None:
        evidence.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            evidence / "initial-state.npz",
            initial_permutation=initial,
            seed=np.array(settings.seed, dtype=np.int64),
            numpy_rng=np.frombuffer(json.dumps(rng.bit_generator.state).encode(), dtype=np.uint8),
        )
    engine = CudaQAP(instance, initial, settings.width, settings.num_warps, settings.energy_layout)
    initial_cost = engine.cost.cpu().numpy()
    arguments = dict(
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        cycle_steps=settings.cycle_steps,
        method=settings.method,
    )
    warm = CudaQAP(instance, initial, settings.width, settings.num_warps, settings.energy_layout)
    noise_shape = (settings.width + 2, settings.replicas, settings.chunk_steps)
    warm_noise = torch.full(noise_shape, 0.5, dtype=torch.float64, device="cuda")
    # Triton can specialize integer zero/one and alignment. Prepare the actual
    # full and final chunk counts at both zero and nonzero search offsets.
    warm_counts = {min(settings.chunk_steps, settings.max_steps)}
    remainder = settings.max_steps % settings.chunk_steps
    if remainder:
        warm_counts.add(remainder)
    warm_offsets = (0, settings.chunk_steps, 2 * settings.chunk_steps, 16 * settings.chunk_steps)
    for count in sorted(warm_counts):
        for offset in warm_offsets:
            warm.advance(warm_noise, steps=count, offset=offset, **arguments)
    torch.cuda.synchronize()
    warm_resources = warm.kernel_resources.copy()
    del warm, warm_noise
    generator = torch.Generator(device="cuda").manual_seed(settings.seed ^ 0x24680)
    initial_rng = generator.get_state().cpu().numpy()
    committed = (0, initial_rng.copy())
    quality_state = (0, initial.copy(), initial_cost.copy())
    pending = None
    snapshots, best_snapshots, snapshot_steps = [initial.copy()], [initial.copy()], [0]
    curve_steps, curve_seconds, curve_cost, curve_best, eligible = [], [], [], [], []
    curve_file = (evidence / "curve.jsonl").open("x") if evidence is not None else None
    steps, chunks = 0, 0

    def checkpoint(name, status):
        if evidence is None:
            return
        path = evidence / ("checkpoint-" + name + ".npz")
        boundary_path = evidence / ("boundary-" + name + ".npz")
        completed_steps, committed_rng = committed
        quality_step, quality_best, quality_cost = quality_state
        # Preserve the last known boundary without touching CUDA first. A
        # poisoned device must not prevent recovery of its committed RNG prefix.
        np.savez_compressed(
            boundary_path,
            completed_steps=np.array(completed_steps),
            committed_generator_state=committed_rng,
            quality_max_step=np.array(quality_step),
            quality_best_permutation=quality_best,
            quality_best_cost=quality_cost,
            initial_rng=initial_rng,
        )
        recovery_arrays = {}
        if pending is not None and pending["noise"] is not None:
            recovery_arrays["pending_noise"] = pending["noise"].cpu().numpy()
        np.savez_compressed(
            path,
            **engine.state(),
            quality_best_permutation=quality_best,
            quality_best_cost=quality_cost,
            completed_steps=np.array(completed_steps),
            quality_max_step=np.array(quality_step),
            generator_state=generator.get_state().cpu().numpy(),
            committed_generator_state=committed_rng,
            initial_rng=initial_rng,
            **recovery_arrays,
        )
        (evidence / "recovery.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "checkpoint": path.name,
                    "boundary_checkpoint": boundary_path.name,
                    "completed_steps": completed_steps,
                    "quality_max_step": quality_step,
                    "pending_chunk": (
                        {key: value for key, value in pending.items() if key != "noise"}
                        if pending is not None
                        else None
                    ),
                    "state_boundary": (
                        "may_include_uncommitted_chunk" if pending is not None else "committed"
                    ),
                    "settings": asdict(settings),
                    "replay": (
                        "Restore the archived source and replay the initial seed "
                        "to completed_steps; use committed_generator_state for that "
                        "boundary. A partial physical state may include pending updates "
                        "and must not be labeled as the committed state. No automatic "
                        "resume or wall-clock equivalence is claimed."
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    checkpoint("0", "initialized")
    preparation_seconds = perf_counter() - prepared
    started = perf_counter()
    last_checkpoint = started
    try:
        while steps < settings.max_steps and perf_counter() - started < settings.max_seconds:
            count = min(settings.chunk_steps, settings.max_steps - steps)
            pending = {"offset": steps, "steps": count, "phase": "drawing", "noise": None}
            noise = torch.rand(noise_shape, dtype=torch.float64, device="cuda", generator=generator)
            pending = {"offset": steps, "steps": count, "phase": "updating", "noise": noise}
            engine.advance(noise, offset=steps, steps=count, **arguments)
            torch.cuda.synchronize()
            committed = (steps + count, generator.get_state().cpu().numpy())
            steps = committed[0]
            pending = None
            chunks += 1
            cost = engine.cost.cpu().numpy()
            prefix_best = engine.best_cost.cpu().numpy()
            elapsed = perf_counter() - started
            accepted = elapsed <= settings.max_seconds and (
                settings.quality_max_step is None or steps <= settings.quality_max_step
            )
            if accepted:
                quality_state = (steps, engine.best.cpu().numpy(), prefix_best.copy())
            curve_steps.append(steps)
            curve_seconds.append(elapsed)
            curve_cost.append(cost)
            curve_best.append(prefix_best)
            eligible.append(accepted)
            if curve_file is not None:
                curve_file.write(
                    json.dumps(
                        {
                            "step": steps,
                            "seconds": elapsed,
                            "cost": cost.tolist(),
                            "best_cost": prefix_best.tolist(),
                            "eligible": accepted,
                        }
                    )
                    + "\n"
                )
                curve_file.flush()
            if chunks % settings.snapshot_every_chunks == 0:
                snapshots.append(engine.perm.cpu().numpy())
                best_snapshots.append(engine.best.cpu().numpy())
                snapshot_steps.append(steps)
            if perf_counter() - last_checkpoint >= settings.checkpoint_seconds:
                checkpoint(str(steps), "running")
                last_checkpoint = perf_counter()
    except BaseException:
        if evidence is not None:
            try:
                checkpoint("partial", "interrupted")
            except BaseException as recovery_error:
                (evidence / "recovery-error.json").write_text(
                    json.dumps(
                        {
                            "error": repr(recovery_error),
                            "completed_steps": committed[0],
                            "boundary_checkpoint": "boundary-partial.npz",
                        }
                    )
                    + "\n"
                )
        raise
    finally:
        if curve_file is not None:
            curve_file.close()
    search_seconds = perf_counter() - started
    checkpoint("final", "completed")
    state = engine.state()
    quality_step, best, best_cost = quality_state
    participated = settings.method == "heatbath" and steps > 0
    arrays = {
        "initial_permutation": initial,
        "initial_cost": initial_cost,
        "quality_best_permutation": best,
        "quality_best_cost": best_cost,
        "snapshot_steps": np.array(snapshot_steps),
        "snapshots": np.array(snapshots),
        "best_snapshots": np.array(best_snapshots),
        "curve_steps": np.array(curve_steps),
        "curve_seconds": np.array(curve_seconds),
        "curve_cost": np.asarray(curve_cost, dtype=np.int64).reshape(-1, settings.replicas),
        "curve_best": np.asarray(curve_best, dtype=np.int64).reshape(-1, settings.replicas),
        "curve_eligible": np.asarray(eligible, dtype=bool),
        "initial_rng": initial_rng,
        "final_rng": generator.get_state().cpu().numpy(),
        **state,
    }
    return arrays, {
        "method": settings.method,
        "settings": asdict(settings),
        "pbit_heatbath_participation": participated,
        "physical_update_pbits_per_model": instance.size**2 if participated else 0,
        "model_binary_variables": instance.size**2,
        "binary_state_representation": (
            "Persistent constrained one-hot assignments with a checked permutation cache"
        ),
        "parallel_replicas": settings.replicas,
        "initialization": "independent_uniform_permutations",
        "independent_trial_unit": "one complete seeded replica ensemble",
        "optimized_classical_prefix_or_polishing": False,
        "initial_best_cost": int(initial_cost.min()),
        "best_feasible_cost": int(best_cost.min()),
        "returned_best_cost": int(state["best_cost"].min()),
        "replicas_improved": int((best_cost < initial_cost).sum()),
        "best_cost_reduction_from_initial_ensemble": int(initial_cost.min() - best_cost.min()),
        "replica_cost_reduction_min": int((initial_cost - best_cost).min()),
        "replica_cost_reduction_median": float(np.median(initial_cost - best_cost)),
        "completed_steps": steps,
        "quality_max_step": quality_step,
        "preparation_seconds": preparation_seconds,
        "warmup": (
            "Full and possible final chunk counts at zero and nonzero offsets; "
            "warm states and draws are separate from the cold search ensemble."
        ),
        "warmup_kernel_variants": list(warm_resources.values()),
        "search_kernel_variants": list(engine.kernel_resources.values()),
        "unprepared_kernel_hashes": sorted(set(engine.kernel_resources) - set(warm_resources)),
        "kernel_resource_scope": (
            "CUDA function registers, static shared memory, and local memory bytes per thread; "
            "local bytes include compiler local storage and are not a count of spill instructions."
        ),
        "search_seconds": search_seconds,
        "deadline_overshoot_seconds": max(0, search_seconds - settings.max_seconds),
        "visited_assignment_bits_min": int(state["visited"].sum(axis=(1, 2)).min()),
        "visited_assignment_semantics": (
            "Ports considered in candidate blocks, including unselected patterns; "
            "not a count of accepted binary flips."
        ),
        "returned_work_scope": (
            "Visitation and block-change counts cover all completed updates, including a "
            "retained deadline-crossing chunk; quality uses its separate eligible boundary."
        ),
        "nonidentity_block_transitions_total": int(state["changed"].sum()),
        "nonidentity_block_transitions_min": int(state["changed"].min()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "observation": (
            "per-chunk best of every accepted transition; deadline crossing chunk excluded"
        ),
        "checkpoint_cost": (
            "Periodic checkpoint I/O is included in search time; initialization and final archive "
            "I/O are outside search time and included in complete runner time."
        ),
    }
