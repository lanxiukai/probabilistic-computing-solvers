"""Budgeted likelihood search with explicit feasible-state observations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .kernels import Factors
from .model import SCALE, quadratic_initial


@dataclass(frozen=True)
class Settings:
    method: str = "gibbs"
    representation: str = "native"
    replicas: int = 4
    device: str = "cuda"
    seed: int = 1
    penalty: int = 8
    penalty_start: float | None = None
    auxiliary_refinement: str = "none"
    beta_start: float = 0.05
    beta_end: float = 1.5
    cycle_sweeps: int = 256
    memory_strength: float = 0.0
    gain_only: bool = False
    hold_probability: float = 0.9
    max_seconds: float = 5.0
    max_steps: int | None = None
    quality_max_step: int | None = None
    diagnostic_sweeps: int = 32


def solve(instance, cfg: Settings, initial=None):
    if (
        cfg.method not in ("gibbs", "parallel")
        or cfg.representation not in ("native", "quadratic")
        or (cfg.method == "parallel" and cfg.representation != "native")
        or cfg.replicas < 1
        or cfg.penalty < 1
        or not isinstance(cfg.penalty, int)
        or (cfg.penalty_start is not None and not 0 <= cfg.penalty_start <= cfg.penalty)
        or cfg.auxiliary_refinement not in ("none", "single", "pair")
        or (
            cfg.auxiliary_refinement != "none"
            and (cfg.representation != "quadratic" or cfg.method != "gibbs")
        )
        or cfg.beta_start <= 0
        or cfg.beta_end < cfg.beta_start
        or cfg.cycle_sweeps < 2
        or not 0 <= cfg.hold_probability < 1
        or not 0 <= cfg.memory_strength <= 1
        or cfg.max_seconds <= 0
        or cfg.diagnostic_sweeps < 1
    ):
        raise ValueError("Invalid LDPC stochastic settings")
    started = perf_counter()
    device = torch.device(cfg.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    factors = Factors(instance, cfg.representation, device)
    if initial is None:
        initial = np.zeros((instance.frames, instance.code.size), dtype=np.int8)
    initial = np.asarray(initial, dtype=np.int8)
    if initial.shape != (instance.frames, instance.code.size):
        raise ValueError("Invalid warm-start dimensions")
    instance.code.syndrome(initial)
    labels = np.repeat(initial[:, None, :], cfg.replicas, axis=1)
    if cfg.representation == "quadratic":
        labels = quadratic_initial(instance, labels)
    bits = torch.as_tensor(labels, dtype=torch.int8, device=device).clone()
    cache = factors.cache(bits)
    cost = factors.cost(bits)
    violations = factors.violations(cache)
    initial_cost, initial_violations = cost.cpu().numpy().copy(), violations.cpu().numpy().copy()
    best_bits, best_cost, best_violations = bits.clone(), cost.clone(), violations.clone()
    best_step = torch.zeros_like(cost)
    best_seconds = torch.zeros_like(cost, dtype=torch.float64)
    visited = torch.zeros_like(bits, dtype=torch.bool)
    changed = torch.zeros_like(bits, dtype=torch.bool)
    memory = torch.zeros_like(bits, dtype=torch.float64) if cfg.method == "parallel" else None
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    initial_rng = generator.get_state().cpu().numpy().copy()
    multiplier = int(instance.costs.sum(axis=-1).max()) + 1
    frame_index = torch.arange(instance.frames, device=device)
    events = {"frame": [], "step": [], "seconds": [], "cost": [], "bits": []}
    last_feasible_cost = np.full(instance.frames, np.iinfo(np.int64).max, dtype=np.int64)
    first_feasible_seconds = np.full(instance.frames, np.nan)
    first_feasible_step = np.full(instance.frames, -1, dtype=np.int64)

    def winner():
        return (best_violations * multiplier + best_cost).argmin(dim=1)

    def feasible_events(step, elapsed):
        selected = winner()
        v = best_violations[frame_index, selected].cpu().numpy()
        c = best_cost[frame_index, selected].cpu().numpy()
        for frame in np.flatnonzero((v == 0) & (c < last_feasible_cost)):
            events["frame"].append(int(frame))
            events["step"].append(step)
            events["seconds"].append(elapsed)
            events["cost"].append(int(c[frame]))
            events["bits"].append(best_bits[frame, selected[frame]].cpu().numpy().copy())
            last_feasible_cost[frame] = c[frame]
            if first_feasible_step[frame] < 0:
                first_feasible_step[frame] = step
                first_feasible_seconds[frame] = elapsed

    feasible_events(0, 0.0)
    diag_step, diag_seconds, diag_v, diag_cost = [], [], [], []

    def diagnostic(step, elapsed):
        selected = winner()
        diag_step.append(step)
        diag_seconds.append(elapsed)
        diag_v.append(best_violations[frame_index, selected].cpu().numpy().copy())
        diag_cost.append(best_cost[frame_index, selected].cpu().numpy().copy())

    diagnostic(0, 0.0)
    sync()
    setup_seconds = perf_counter() - started
    search = perf_counter()
    step, quality_step, elapsed = 0, 0, 0.0
    extra = {"none": 0, "single": 2, "pair": 1}[cfg.auxiliary_refinement]
    group_count = len(factors.groups) + extra if cfg.method == "gibbs" else 1
    while cfg.max_steps is None or step < cfg.max_steps:
        if cfg.max_steps is None and perf_counter() - search >= cfg.max_seconds:
            break
        sweep = step // group_count
        phase = (sweep % cfg.cycle_sweeps) / (cfg.cycle_sweeps - 1)
        beta = cfg.beta_start * (cfg.beta_end / cfg.beta_start) ** phase
        penalty = (
            cfg.penalty * SCALE
            if cfg.penalty_start is None
            else round(SCALE * (cfg.penalty_start + (cfg.penalty - cfg.penalty_start) * phase))
        )
        if cfg.method == "gibbs":
            position = step % group_count
            if cfg.auxiliary_refinement == "pair" and position == len(factors.groups):
                uniform = torch.rand(
                    (*bits.shape[:-1], instance.code.checks),
                    dtype=torch.float64,
                    device=device,
                    generator=generator,
                )
                flips = factors.auxiliary_pair_update(bits, cache, beta, penalty, uniform)
                visited[..., instance.code.size :] = True
                changed[..., instance.code.size :] |= flips
            else:
                group = (
                    factors.groups[position]
                    if position < len(factors.groups)
                    else factors.auxiliary_groups[position - len(factors.groups)]
                )
                nodes = group[0]
                uniform = torch.rand(
                    (*bits.shape[:-1], len(nodes)),
                    dtype=torch.float64,
                    device=device,
                    generator=generator,
                )
                flips = factors.colored_update(bits, cache, cost, group, beta, penalty, uniform)
                visited[..., nodes] = True
                changed[..., nodes] |= flips
        else:
            uniform = torch.rand(
                bits.shape, dtype=torch.float64, device=device, generator=generator
            )
            active = (
                torch.rand(bits.shape, dtype=torch.float64, device=device, generator=generator)
                >= cfg.hold_probability
            )
            flips = factors.parallel_update(
                bits,
                cache,
                cost,
                memory,
                beta,
                penalty,
                cfg.memory_strength,
                cfg.gain_only,
                uniform,
                active,
            )
            visited |= active
            changed |= flips
        step += 1
        violations = factors.violations(cache)
        improved = (violations < best_violations) | (
            (violations == best_violations) & (cost < best_cost)
        )
        sync()
        elapsed = perf_counter() - search
        eligible = (
            step <= cfg.quality_max_step
            if cfg.quality_max_step is not None
            else elapsed <= cfg.max_seconds
        )
        if eligible:
            quality_step = step
            if bool(improved.any()):
                best_bits[improved] = bits[improved]
                best_cost[improved] = cost[improved]
                best_violations[improved] = violations[improved]
                best_step[improved] = step
                best_seconds[improved] = elapsed
                feasible_events(step, elapsed)
        if step % (cfg.diagnostic_sweeps * group_count) == 0:
            diagnostic(step, elapsed)
        if not eligible and cfg.max_steps is None:
            break
    sync()
    search_seconds = perf_counter() - search
    if not torch.equal(cache, factors.cache(bits)) or not torch.equal(cost, factors.cost(bits)):
        raise AssertionError("Incremental LDPC factor cache or cost drifted")
    if diag_step[-1] != step:
        diagnostic(step, elapsed)
    selected = winner()
    decoded_physical = best_bits[frame_index, selected].cpu().numpy()
    selected_cost = best_cost[frame_index, selected].cpu().numpy()
    selected_v = best_violations[frame_index, selected].cpu().numpy()
    event_bits = np.asarray(events["bits"], dtype=np.int8).reshape(-1, factors.size)
    arrays = {
        "initial_bits": labels,
        "initial_cost": initial_cost,
        "initial_violations": initial_violations,
        "best_bits": best_bits.cpu().numpy(),
        "best_cost": best_cost.cpu().numpy(),
        "best_violations": best_violations.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "best_seconds": best_seconds.cpu().numpy(),
        "final_bits": bits.cpu().numpy(),
        "final_cost": cost.cpu().numpy(),
        "final_violations": violations.cpu().numpy(),
        "final_factor_cache": cache[..., :-1].cpu().numpy(),
        "final_memory": memory.cpu().numpy() if memory is not None else np.empty(0),
        "decoded_error": decoded_physical[:, : instance.code.size],
        "decoded_physical": decoded_physical,
        "winner_replica": selected.cpu().numpy(),
        "visited": visited.cpu().numpy(),
        "changed": changed.cpu().numpy(),
        "colors": factors.colors,
        "initial_rng": initial_rng,
        "final_rng": generator.get_state().cpu().numpy(),
        "event_frame": np.asarray(events["frame"], dtype=np.int64),
        "event_step": np.asarray(events["step"], dtype=np.int64),
        "event_seconds": np.asarray(events["seconds"]),
        "event_cost": np.asarray(events["cost"], dtype=np.int64),
        "event_bits": event_bits,
        "first_feasible_step": first_feasible_step,
        "first_feasible_seconds": first_feasible_seconds,
        "diagnostic_step": np.asarray(diag_step),
        "diagnostic_seconds": np.asarray(diag_seconds),
        "diagnostic_best_violations": np.asarray(diag_v),
        "diagnostic_best_cost": np.asarray(diag_cost),
    }
    initial_valid = np.any(initial_violations == 0, axis=1)
    return arrays, {
        "method": cfg.method,
        "representation": cfg.representation,
        "frames": instance.frames,
        "replicas": cfg.replicas,
        "valid_frames": int(np.count_nonzero(selected_v == 0)),
        "initial_valid_frames": int(initial_valid.sum()),
        "new_valid_frames": int(np.count_nonzero((selected_v == 0) & ~initial_valid)),
        "feasible_cost_sum": int(selected_cost[selected_v == 0].sum()),
        "physical_update_pbits_per_model": factors.size,
        "auxiliary_pbits": factors.size - instance.code.size,
        "response_states_per_model": factors.size if memory is not None else 0,
        "visited_variables_min": int(visited.sum(-1).min()),
        "completed_steps": step,
        "quality_max_step": quality_step,
        "colors": len(factors.groups),
        "extra_auxiliary_steps_per_sweep": extra,
        "auxiliary_refinement": cfg.auxiliary_refinement,
        "penalty_start": cfg.penalty_start,
        "completed_sweeps": step / group_count,
        "setup_seconds": setup_seconds,
        "search_seconds": search_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
