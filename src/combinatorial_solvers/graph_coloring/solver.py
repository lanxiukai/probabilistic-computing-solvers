"""Bounded CUDA coloring search with cold states, state histories and work replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gpu import CudaColoring
from .tempering import exchange_ranks


@dataclass(frozen=True)
class Settings:
    replicas: int = 64
    seed: int = 2026102301
    width: int = 1
    method: str = "heatbath"
    beta_start: float = 0.1
    beta_end: float = 8.0
    cycle_steps: int = 4096
    chunk_steps: int = 256
    max_steps: int = 100000000
    max_seconds: float = 10
    snapshot_every_chunks: int | None = 32
    quality_max_step: int | None = None
    checkpoint_seconds: float = 2.0
    tempering_ladder_size: int = 1
    exchange_every_chunks: int = 8
    stop_when_all_groups_solved: bool = False
    cache_boltzmann_weights: bool = True
    warmup_steps: int | None = None
    reuse_preparation: bool = True

    def validate(self):
        for name in (
            "replicas",
            "seed",
            "width",
            "cycle_steps",
            "chunk_steps",
            "max_steps",
            "tempering_ladder_size",
            "exchange_every_chunks",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError("Invalid integer setting: " + name)
        if self.replicas > 1024 or self.width not in (1, 2, 3, 4):
            raise ValueError("Unsupported replica count or block width")
        if self.method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown update method")
        if not 0 < self.beta_start <= self.beta_end < float(
            "inf"
        ) or not 0 < self.max_seconds < float("inf"):
            raise ValueError("Invalid temperature endpoints or time limit")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid replay quality boundary")
        if not 0 < self.checkpoint_seconds < float("inf"):
            raise ValueError("Invalid checkpoint interval")
        if self.snapshot_every_chunks is not None and (
            isinstance(self.snapshot_every_chunks, bool)
            or not isinstance(self.snapshot_every_chunks, int)
            or self.snapshot_every_chunks < 1
        ):
            raise ValueError("Snapshot interval must be a positive integer or None")
        if self.tempering_ladder_size != 1 and (
            self.tempering_ladder_size % 2 or self.replicas % self.tempering_ladder_size
        ):
            raise ValueError("An even tempering ladder must divide the replica count")
        if self.tempering_ladder_size > 1 and self.method == "plateau":
            raise ValueError("Temperature exchange requires finite-temperature transitions")
        if not isinstance(self.stop_when_all_groups_solved, bool):
            raise ValueError("Early-stop selection must be a boolean")
        if not isinstance(self.cache_boltzmann_weights, bool):
            raise ValueError("Weight-cache selection must be a boolean")
        if not isinstance(self.reuse_preparation, bool):
            raise ValueError("Preparation reuse must be a boolean")
        if self.warmup_steps is not None and (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or not 1 <= self.warmup_steps <= self.chunk_steps
        ):
            raise ValueError("Warmup steps must fit the declared chunk")


def solve(instance, settings: Settings, *, evidence_directory: Path | None = None):
    settings.validate()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    prepared = perf_counter()
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    choices = (
        [np.flatnonzero(row) for row in instance.allowed] if settings.reuse_preparation else None
    )
    initial = np.array(
        [
            [
                rng.choice(choices[vertex] if choices is not None else np.flatnonzero(row))
                for vertex, row in enumerate(instance.allowed)
            ]
            for _ in range(settings.replicas)
        ]
    )
    evidence = Path(evidence_directory) if evidence_directory is not None else None
    if evidence is not None:
        evidence.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            evidence / "initial-state.npz",
            colors=initial,
            initialization_rng=np.frombuffer(
                json.dumps(rng.bit_generator.state).encode(), dtype=np.uint8
            ),
        )
    engine = CudaColoring(instance, initial, settings.width)
    initial_cost = engine.cost.cpu().numpy()
    tempering = settings.tempering_ladder_size > 1
    beta_grid = np.geomspace(settings.beta_start, settings.beta_end, settings.tempering_ladder_size)
    ranks = np.tile(
        np.arange(settings.tempering_ladder_size, dtype=np.int64),
        settings.replicas // settings.tempering_ladder_size,
    )
    exchange_rng = np.random.Generator(np.random.PCG64(settings.seed ^ 0x73129))
    initial_exchange_rng = np.frombuffer(
        json.dumps(exchange_rng.bit_generator.state).encode(), dtype=np.uint8
    )
    arguments = dict(
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        cycle_steps=settings.cycle_steps,
        method=settings.method,
        cache_boltzmann_weights=settings.cache_boltzmann_weights,
        temperature_epoch=0,
    )
    if tempering:
        arguments["beta_by_replica"] = torch.tensor(
            beta_grid[ranks], dtype=torch.float64, device="cuda"
        )
    warm = (
        engine.clone()
        if settings.reuse_preparation
        else CudaColoring(instance, initial, settings.width)
    )
    noise_shape = (max(4, settings.width + 2), settings.replicas, settings.chunk_steps)
    warm.advance(
        torch.full(noise_shape, 0.5, dtype=torch.float64, device="cuda"),
        steps=settings.warmup_steps or settings.chunk_steps,
        **arguments,
    )
    torch.cuda.synchronize()
    del warm
    generator = torch.Generator(device="cuda").manual_seed(settings.seed ^ 0x24680)
    initial_rng = generator.get_state().cpu().numpy()
    best = initial.copy()
    best_cost = initial_cost.copy()
    snapshots, best_snapshots, snapshot_steps = [initial.copy()], [initial.copy()], [0]
    curve_steps, curve_seconds, curve_cost, curve_best, eligible = [], [], [], [], []
    curve_ranks = []
    (
        exchange_steps,
        exchange_cost,
        exchange_uniform,
        exchange_labels,
        exchange_accept,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    steps, chunks, quality_step = 0, 0, 0
    latest_checkpoint = None
    phase = "idle"
    proposed_chunk = 0
    pending_noise = None
    pending_exchange_uniform = None
    pending_offset, pending_steps = 0, 0
    committed = (
        steps,
        chunks,
        best,
        best_cost,
        quality_step,
        initial_rng.copy(),
        ranks.copy(),
        initial_exchange_rng.copy(),
    )

    def checkpoint(name, status):
        nonlocal latest_checkpoint
        if evidence is None:
            return
        target = evidence / (name + ".npz")
        temporary = evidence / (name + ".pending.npz")
        confirmed_boundary = proposed_chunk == committed[1]
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **engine.state(),
                quality_best_colors=committed[2],
                quality_best_cost=committed[3],
                completed_steps=np.array(committed[0]),
                quality_max_step=np.array(committed[4]),
                chunks=np.array(committed[1]),
                rng=committed[5],
                current_rng=generator.get_state().cpu().numpy(),
                committed_beta_rank=committed[6],
                committed_exchange_rng=committed[7],
                confirmed_chunk_boundary=np.array(confirmed_boundary),
                phase=np.array(phase),
                pending_noise=np.empty(0) if pending_noise is None else pending_noise.cpu().numpy(),
                pending_noise_offset=np.array(pending_offset),
                pending_noise_steps=np.array(pending_steps),
                pending_exchange_uniform=np.empty(0)
                if pending_exchange_uniform is None
                else pending_exchange_uniform,
                seed=np.array(settings.seed),
                beta_rank=ranks,
                exchange_events=np.array(len(exchange_steps)),
                exchange_rng=np.frombuffer(
                    json.dumps(exchange_rng.bit_generator.state).encode(),
                    dtype=np.uint8,
                ),
            )
        temporary.replace(target)
        if confirmed_boundary:
            latest_checkpoint = target.name
        (evidence / "recovery.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "latest_complete_checkpoint": latest_checkpoint,
                    "latest_state_file": target.name,
                    "confirmed_chunk_boundary": confirmed_boundary,
                    "phase": phase,
                    "completed_steps": committed[0],
                    "quality_max_step": committed[4],
                    "settings": asdict(settings),
                    "scope": (
                        "Recovery state and fixed-work replay evidence; "
                        "no automatic resume is claimed."
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    checkpoint("checkpoint-0", "running")
    preparation_seconds = perf_counter() - prepared
    started = perf_counter()
    last_checkpoint = started
    curve_file = (evidence / "curve.jsonl").open("a") if evidence is not None else None
    exchange_file = (evidence / "exchange.jsonl").open("a") if evidence is not None else None
    try:
        while steps < settings.max_steps and perf_counter() - started < settings.max_seconds:
            count = min(settings.chunk_steps, settings.max_steps - steps)
            proposed_chunk = chunks + 1
            pending_offset, pending_steps = steps, count
            phase = "randomness"
            noise = torch.rand(noise_shape, dtype=torch.float64, device="cuda", generator=generator)
            pending_noise = noise
            phase = "transition"
            engine.advance(noise, offset=steps, steps=count, **arguments)
            torch.cuda.synchronize()
            phase = "observation"
            steps += count
            chunks += 1
            cost = engine.cost.cpu().numpy()
            prefix_best = engine.best_cost.cpu().numpy()
            elapsed = perf_counter() - started
            accepted = elapsed <= settings.max_seconds and (
                settings.quality_max_step is None or steps <= settings.quality_max_step
            )
            if accepted:
                best = engine.best.cpu().numpy()
                best_cost = prefix_best.copy()
                quality_step = steps
            curve_steps.append(steps)
            curve_seconds.append(elapsed)
            curve_cost.append(cost)
            curve_best.append(prefix_best)
            eligible.append(accepted)
            curve_ranks.append(ranks.copy())
            if curve_file is not None:
                curve_file.write(
                    json.dumps(
                        {
                            "step": steps,
                            "seconds": elapsed,
                            "eligible": accepted,
                            "cost": cost.tolist(),
                            "best_cost": prefix_best.tolist(),
                            "beta_rank": ranks.tolist(),
                        }
                    )
                    + "\n"
                )
                curve_file.flush()
            if (
                settings.snapshot_every_chunks is not None
                and chunks % settings.snapshot_every_chunks == 0
            ):
                snapshots.append(engine.color.cpu().numpy())
                best_snapshots.append(engine.best.cpu().numpy())
                snapshot_steps.append(steps)
            if tempering and chunks % settings.exchange_every_chunks == 0:
                phase = "temperature_exchange"
                uniforms = exchange_rng.random(
                    (
                        settings.replicas // settings.tempering_ladder_size,
                        settings.tempering_ladder_size // 2,
                    )
                )
                uniforms = np.maximum(uniforms, np.nextafter(0.0, 1.0))
                pending_exchange_uniform = uniforms
                ranks, exchanged = exchange_ranks(
                    cost,
                    ranks,
                    beta_grid,
                    settings.tempering_ladder_size,
                    len(exchange_steps) % 2,
                    uniforms,
                )
                exchange_steps.append(steps)
                exchange_cost.append(cost.copy())
                exchange_uniform.append(uniforms)
                exchange_labels.append(ranks.copy())
                exchange_accept.append(exchanged)
                if exchange_file is not None:
                    exchange_file.write(
                        json.dumps(
                            {
                                "step": steps,
                                "cost": cost.tolist(),
                                "uniform": uniforms.tolist(),
                                "beta_rank": ranks.tolist(),
                                "accepted": exchanged.tolist(),
                            }
                        )
                        + "\n"
                    )
                    exchange_file.flush()
                arguments["beta_by_replica"].copy_(
                    torch.as_tensor(beta_grid[ranks], dtype=torch.float64, device="cuda")
                )
                arguments["temperature_epoch"] = len(exchange_steps)
            committed = (
                steps,
                chunks,
                best,
                best_cost,
                quality_step,
                generator.get_state().cpu().numpy().copy(),
                ranks.copy(),
                np.frombuffer(
                    json.dumps(exchange_rng.bit_generator.state).encode(),
                    dtype=np.uint8,
                ),
            )
            phase = "idle"
            pending_noise = None
            pending_exchange_uniform = None
            pending_offset, pending_steps = steps, 0
            if perf_counter() - last_checkpoint >= settings.checkpoint_seconds:
                checkpoint("checkpoint-" + str(steps), "running")
                last_checkpoint = perf_counter()
            if (
                settings.stop_when_all_groups_solved
                and (best_cost.reshape(-1, settings.tempering_ladder_size).min(axis=1) == 0).all()
            ):
                break
    except BaseException:
        try:
            checkpoint("checkpoint-partial", "interrupted")
        except BaseException as recovery_error:
            if evidence is not None:
                (evidence / "recovery-error.json").write_text(
                    json.dumps(
                        {
                            "error": repr(recovery_error),
                            "latest_complete_checkpoint": latest_checkpoint,
                            "phase": phase,
                            "committed_steps": committed[0],
                            "committed_rng": committed[5].tolist(),
                            "committed_beta_rank": committed[6].tolist(),
                            "committed_exchange_rng": committed[7].tobytes().decode(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
        raise
    finally:
        if curve_file is not None:
            curve_file.close()
        if exchange_file is not None:
            exchange_file.close()
    search_seconds = perf_counter() - started
    checkpoint("checkpoint-final", "completed")
    state = engine.state()
    arrays = {
        "initial_colors": initial,
        "initial_cost": initial_cost,
        "quality_best_colors": best,
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
        "beta_grid": beta_grid,
        "final_beta_rank": ranks,
        "curve_beta_rank": np.asarray(curve_ranks),
        "exchange_steps": np.asarray(exchange_steps, dtype=np.int64),
        "exchange_cost": np.asarray(exchange_cost, dtype=np.int64),
        "exchange_uniform": np.asarray(exchange_uniform, dtype=np.float64),
        "exchange_beta_rank": np.asarray(exchange_labels, dtype=np.int64),
        "exchange_accepted": np.asarray(exchange_accept, dtype=bool),
        "initial_exchange_rng": initial_exchange_rng,
        "final_exchange_rng": np.frombuffer(
            json.dumps(exchange_rng.bit_generator.state).encode(), dtype=np.uint8
        ),
        **state,
    }
    return arrays, {
        "method": settings.method,
        "settings": asdict(settings),
        "pbit_heatbath_participation": bool(
            settings.method == "heatbath" and (instance.allowed.sum(axis=1) > 1).any()
        ),
        "physical_update_pbits_per_model": int(
            instance.allowed[instance.allowed.sum(axis=1) > 1].sum()
        )
        if settings.method == "heatbath"
        else 0,
        "model_binary_variables": instance.size * instance.colors,
        "parallel_replicas": settings.replicas,
        "independent_replica_groups": settings.replicas // settings.tempering_ladder_size,
        "tempering_ladder_size": settings.tempering_ladder_size,
        "zero_conflict_groups": int(
            (best_cost.reshape(-1, settings.tempering_ladder_size).min(axis=1) == 0).sum()
        ),
        "exchange_events": len(exchange_steps),
        "exchange_accepted_pairs": int(np.asarray(exchange_accept).sum()),
        "temperature_scope": (
            "Constant temperature labels with adjacent Metropolis exchanges within "
            "independent ladders; labels move, persistent binary states stay in their chains."
        )
        if tempering
        else "Independent chains with a common cyclic inverse-temperature schedule.",
        "initialization": "independent_uniform_allowed_colors",
        "initial_best_cost": int(initial_cost.min()),
        "initial_zero_conflict_hits": int((initial_cost == 0).sum()),
        "pbit_new_zero_conflict_hits": int(((initial_cost > 0) & (best_cost == 0)).sum())
        if settings.method == "heatbath"
        else 0,
        "best_feasible_cost": int(best_cost.min()),
        "returned_best_cost": int(state["best_conflicts"].min()),
        "replicas_improved": int((best_cost < initial_cost).sum()),
        "completed_steps": steps,
        "quality_max_step": quality_step,
        "preparation_seconds": preparation_seconds,
        "search_seconds": search_seconds,
        "deadline_overshoot_seconds": max(0, search_seconds - settings.max_seconds),
        "visited_assignment_bits_min": int(state["visited"].sum(axis=(1, 2)).min()),
        "visited_unclamped_assignment_bits_min": int(
            state["visited"][:, instance.allowed.sum(axis=1) > 1].sum(axis=(1, 2)).min()
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "observation": (
            "per-chunk best of every accepted transition; deadline crossing chunk excluded"
        ),
        "checkpoint_timing": (
            "Periodic checkpoint and curve costs are inside the search wall time; "
            "initial/final checkpoints are outside it."
        ),
        "warmup_steps": settings.warmup_steps or settings.chunk_steps,
        "preparation_reuse": settings.reuse_preparation,
        "boltzmann_weight_cache_active": bool(
            settings.cache_boltzmann_weights
            and settings.method != "plateau"
            and (tempering or settings.beta_start == settings.beta_end)
        ),
    }
