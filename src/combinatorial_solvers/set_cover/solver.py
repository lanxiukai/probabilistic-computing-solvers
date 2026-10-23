"""Bounded GPU covering search with original-state certificates and work replays."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .baseline import fast_greedy
from .blocks import build_blocks
from .gpu import CudaCover


@dataclass(frozen=True)
class Settings:
    method: str = "heatbath"
    representation: str = "penalty"
    width: int = 1
    banks: int = 1
    block_seed: int = 2026102200
    block_policy: str = "minhash"
    cache_initial: bool = True
    selection: str = "uniform"
    epsilon: float = 0.0625
    initialization: str = "random"
    replicas: int = 16
    seed: int = 2026102201
    beta_start: float = 0.05
    beta_end: float = 8.0
    cycle_sweeps: int = 4
    chunk_steps: int = 256
    max_steps: int = 2_000_000
    max_seconds: float = 10.0
    snapshot_every_chunks: int = 32
    quality_max_step: int | None = None

    def validate(self):
        if self.method not in {"heatbath", "metropolis", "greedy", "plateau"}:
            raise ValueError("Unknown native method")
        if self.representation not in {"penalty", "feasible"}:
            raise ValueError("Unknown representation")
        if self.selection not in {"uniform", "active"}:
            raise ValueError("Unknown block selection")
        if self.block_policy not in {"minhash", "cost-minhash", "greedy-pivot"}:
            raise ValueError("Unknown block construction policy")
        if self.initialization not in {"random", "greedy", "all"}:
            raise ValueError("Unknown initialization")
        if self.representation == "feasible" and self.initialization == "random":
            raise ValueError("Feasible kernels need a declared feasible start")
        for name in (
            "replicas",
            "cycle_sweeps",
            "chunk_steps",
            "max_steps",
            "snapshot_every_chunks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.seed < 0 or self.block_seed < 0 or not 1 <= self.replicas <= 128:
            raise ValueError("Invalid seeds or replica count")
        if not all(
            isfinite(x) and x > 0
            for x in (self.beta_start, self.beta_end, self.max_seconds, self.epsilon)
        ):
            raise ValueError("Positive finite temperature, duration and selection mass required")
        if self.beta_end < self.beta_start:
            raise ValueError("Annealing endpoints are reversed")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Quality work boundary must be within the replay")


def solve(instance, settings: Settings, *, evidence_directory: Path | None = None):
    settings.validate()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the performance solver")
    prepared = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    init_start = perf_counter()
    if settings.initialization == "random":
        rng = np.random.Generator(np.random.PCG64(settings.seed ^ 0x13579))
        initial = rng.integers(0, 2, size=(settings.replicas, instance.size), dtype=np.int8)
    else:
        single = (
            fast_greedy(instance)
            if settings.initialization == "greedy"
            else np.ones(instance.size, dtype=np.int8)
        )
        initial = np.repeat(single[None, :], settings.replicas, axis=0)
    initialization_seconds = perf_counter() - init_start
    if evidence_directory is not None:
        np.savez_compressed(evidence_directory / "native-initial.npz", initial_bits=initial)
    block_start = perf_counter()
    blocks = build_blocks(
        instance, settings.width, settings.banks, settings.block_seed, policy=settings.block_policy
    )
    block_seconds = perf_counter() - block_start
    engine = CudaCover(
        instance,
        blocks,
        initial,
        active=settings.selection == "active",
        epsilon=settings.epsilon,
        cache_initial=settings.cache_initial,
    )
    penalty = max(instance.costs) + 1
    scale = max(instance.costs)
    arguments = {
        "cycle_steps": blocks.per_bank * settings.cycle_sweeps,
        "beta_low": settings.beta_start * scale,
        "beta_high": settings.beta_end * scale,
        "penalty": penalty,
        "hard": settings.representation == "feasible",
        "method": settings.method,
    }
    warm_start = perf_counter()
    warm = CudaCover(
        instance,
        blocks,
        initial,
        active=settings.selection == "active",
        epsilon=settings.epsilon,
        cache_initial=settings.cache_initial,
    )
    warm.advance(
        torch.full(
            (3, settings.replicas, settings.chunk_steps), 0.5, dtype=torch.float64, device="cuda"
        ),
        steps=1,
        **arguments,
    )
    torch.cuda.synchronize()
    del warm
    warmup_seconds = perf_counter() - warm_start
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.seed ^ 0x24680)
    initial_rng = generator.get_state().cpu().numpy()
    initial_cost = engine.cost.cpu().numpy()
    initial_violations = engine.violations.cpu().numpy()
    best_cost = np.where(initial_violations == 0, initial_cost, -1)
    best_bits = engine.bits.clone()
    best_energy = initial_cost + penalty * initial_violations
    best_energy_bits = engine.bits.clone()
    snapshot_steps, snapshots = [0], [initial.copy()]
    best_snapshots = [initial.copy()]
    steps_log, seconds_log, costs_log, violations_log, best_log, eligibility = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    torch.cuda.synchronize()
    preparation_seconds = perf_counter() - prepared
    started = perf_counter()
    steps, chunks, quality_last_step = 0, 0, 0
    failure = None
    try:
        while steps < settings.max_steps and perf_counter() - started < settings.max_seconds:
            count = min(settings.chunk_steps, settings.max_steps - steps)
            noise = torch.rand(
                (3, settings.replicas, settings.chunk_steps),
                generator=generator,
                dtype=torch.float64,
                device="cuda",
            )
            engine.advance(noise, offset=steps, steps=count, **arguments)
            current_cost = engine.cost.cpu().numpy()
            current_violations = engine.violations.cpu().numpy()
            elapsed = perf_counter() - started
            steps += count
            chunks += 1
            eligible = (
                elapsed <= settings.max_seconds
                if settings.quality_max_step is None
                else steps <= settings.quality_max_step
            )
            if eligible:
                quality_last_step = steps
                improved = (current_violations == 0) & (
                    (best_cost < 0) | (current_cost < best_cost)
                )
                if improved.any():
                    best_bits[torch.as_tensor(improved, device="cuda")] = engine.bits[
                        torch.as_tensor(improved, device="cuda")
                    ]
                    best_cost = np.where(improved, current_cost, best_cost)
                energy = current_cost + penalty * current_violations
                lower = energy < best_energy
                if lower.any():
                    best_energy_bits[torch.as_tensor(lower, device="cuda")] = engine.bits[
                        torch.as_tensor(lower, device="cuda")
                    ]
                    best_energy = np.minimum(best_energy, energy)
            steps_log.append(steps)
            seconds_log.append(elapsed)
            costs_log.append(current_cost.copy())
            violations_log.append(current_violations.copy())
            best_log.append(best_cost.copy())
            eligibility.append(eligible)
            if chunks % settings.snapshot_every_chunks == 0:
                snapshot_steps.append(steps)
                snapshots.append(engine.bits.cpu().numpy())
                best_snapshots.append(best_bits.cpu().numpy())
        torch.cuda.synchronize()
    except BaseException as error:
        failure = repr(error)
        if evidence_directory is not None:
            np.savez_compressed(
                evidence_directory / "partial-observations.npz",
                snapshot_steps=np.asarray(snapshot_steps),
                snapshots=np.stack(snapshots),
                best_snapshots=np.stack(best_snapshots),
                observation_steps=np.asarray(steps_log),
                observation_seconds=np.asarray(seconds_log),
                observation_cost=np.asarray(costs_log),
                observation_violations=np.asarray(violations_log),
                observation_best_cost=np.asarray(best_log),
                observation_eligible=np.asarray(eligibility),
            )
    sampling_seconds = perf_counter() - started
    export_start = perf_counter()
    final = engine.state()
    if snapshot_steps[-1] != steps:
        snapshot_steps.append(steps)
        snapshots.append(final["bits"].copy())
        best_snapshots.append(best_bits.cpu().numpy())
    arrays = {
        "initial_bits": initial,
        "initial_rng": initial_rng,
        "final_rng": generator.get_state().cpu().numpy(),
        "best_bits": best_bits.cpu().numpy(),
        "best_cost": best_cost,
        "best_energy_bits": best_energy_bits.cpu().numpy(),
        "best_energy": best_energy,
        "final_bits": final["bits"],
        "final_coverage": final["coverage"],
        "final_cost": final["cost"],
        "final_violations": final["violations"],
        "final_cardinality": final["cardinality"],
        "visited": final["visited"],
        "changed_blocks": final["changed"],
        "flipped_bits": final["flips"],
        "snapshot_steps": np.asarray(snapshot_steps),
        "snapshots": np.stack(snapshots),
        "best_snapshots": np.stack(best_snapshots),
        "observation_steps": np.asarray(steps_log),
        "observation_seconds": np.asarray(seconds_log),
        "observation_cost": np.asarray(costs_log),
        "observation_violations": np.asarray(violations_log),
        "observation_best_cost": np.asarray(best_log),
        "observation_eligible": np.asarray(eligibility),
    }
    if settings.selection == "active":
        arrays["active_list"], arrays["position"] = final["active_list"], final["position"]
    state_transfer_seconds = perf_counter() - export_start
    valid = best_cost >= 0
    summary = {
        "method": settings.method,
        "representation": settings.representation,
        "initialization": settings.initialization,
        "selection": settings.selection,
        "block_width": settings.width,
        "block_banks": settings.banks,
        "block_policy": settings.block_policy,
        "cache_initial": settings.cache_initial,
        "block_count": len(blocks.nodes),
        "block_row_stride": blocks.rows.shape[1],
        "replicas": settings.replicas,
        "seed": settings.seed,
        "penalty": penalty,
        "inverse_temperature_units": "inverse-original-cost",
        "initialization_seconds": initialization_seconds,
        "block_preparation_seconds": block_seconds,
        "warmup_seconds": warmup_seconds,
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "state_transfer_seconds": state_transfer_seconds,
        "completed_steps": steps,
        "quality_max_step": quality_last_step,
        "completed_chunks": chunks,
        "cycle_steps": arguments["cycle_steps"],
        "failure": failure,
        "initial_best_cost": int(initial_cost[initial_violations == 0].min())
        if (initial_violations == 0).any()
        else None,
        "best_feasible_cost": int(best_cost[valid].min()) if valid.any() else None,
        "feasible_replicas": int(valid.sum()),
        "minimum_visited_variables_per_replica": int(final["visited"].sum(-1).min()),
        "model_binary_variables": instance.size,
        "auxiliary_pbits": 0,
        "physical_update_pbits_per_model": instance.size
        if settings.method in {"heatbath", "metropolis"}
        else 0,
        "pbit_heatbath_participation": settings.method == "heatbath",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "capture": (
            "initial/final, periodic current/best states, every chunk objective and feasibility; "
            "no every-step state export"
        ),
        "stop_reason": "failure"
        if failure
        else "step_limit"
        if steps == settings.max_steps
        else "time_limit",
    }
    return arrays, summary
