"""Budgeted, exactly balanced conditional heat-bath and matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .blocks import make_banks, prepare_groups, update, validate_banks


@dataclass(frozen=True)
class Settings:
    method: str = "heatbath"
    width: int = 2
    banks: int = 8
    replicas: int = 8
    seed: int = 1
    bank_seed: int = 173
    device: str = "cuda"
    beta_start: float = 0.1
    beta_end: float = 8.0
    cycle_sweeps: int = 64
    max_seconds: float = 15.0
    max_sweeps: int | None = None
    quality_sweeps: int | None = None


def solve(graph, settings: Settings, initial=None):
    cfg = settings
    if (
        cfg.replicas < 1
        or cfg.max_seconds <= 0
        or cfg.cycle_sweeps < 2
        or cfg.beta_start <= 0
        or cfg.beta_end < cfg.beta_start
        or cfg.method not in ("heatbath", "metropolis", "greedy")
        or (cfg.method == "metropolis" and cfg.width != 2)
    ):
        raise ValueError("Invalid bisection settings")
    device = torch.device(cfg.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    started = perf_counter()
    banks = make_banks(graph, cfg.width, cfg.banks, cfg.bank_seed)
    validate_banks(graph, banks)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    groups = prepare_groups(graph, banks, device)
    rng = np.random.default_rng(cfg.seed)
    if initial is None:
        labels = np.stack(
            [
                rng.permutation(np.repeat(np.arange(2, dtype=np.int8), graph.size // 2))
                for _ in range(cfg.replicas)
            ]
        )
    else:
        graph.certify(initial)
        labels = np.tile(initial, (cfg.replicas, 1)).astype(np.int8)
    initial_cut = graph.certify(labels)
    spins = torch.as_tensor(2 * labels - 1, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    initial_rng = generator.get_state().cpu().numpy().copy()
    patterns = torch.tensor(
        [[2 * ((value >> i) & 1) - 1 for i in range(cfg.width)] for value in range(2**cfg.width)],
        dtype=torch.float64,
        device=device,
    )
    edges = torch.as_tensor(graph.edges, device=device)
    weights = torch.as_tensor(graph.weights, device=device)

    def cut():
        return ((spins[:, edges[:, 0]] != spins[:, edges[:, 1]]) * weights).sum(-1)

    best = spins.clone()
    best_cut = cut()
    trace_bits, trace_cut = [labels.copy()], [initial_cut.copy()]
    trace_sweep, trace_seconds, trace_eligible = [0], [0.0], [True]
    visited = torch.zeros(graph.size, dtype=torch.bool, device=device)
    changed = torch.zeros((cfg.replicas, graph.size), dtype=torch.bool, device=device)
    sync()
    setup_seconds = perf_counter() - started
    search = perf_counter()
    completed, quality_sweeps, quiet_sweeps = 0, 0, 0
    last_elapsed = 0.0
    while cfg.max_sweeps is None or completed < cfg.max_sweeps:
        if cfg.max_sweeps is None and perf_counter() - search >= cfg.max_seconds:
            break
        phase = (completed % cfg.cycle_sweeps) / (cfg.cycle_sweeps - 1)
        beta = cfg.beta_start * (cfg.beta_end / cfg.beta_start) ** phase
        before = spins.clone()
        for group in groups[completed % cfg.banks]:
            uniform = torch.rand(
                (cfg.replicas, len(group[0])),
                dtype=torch.float64,
                device=device,
                generator=generator,
            )
            update(spins, group, patterns, beta, cfg.method, uniform)
            visited[group[0].reshape(-1)] = True
        changed |= before != spins
        completed += 1
        value = cut()
        sync()
        elapsed = perf_counter() - search
        last_elapsed = elapsed
        eligible = (
            completed <= cfg.quality_sweeps
            if cfg.quality_sweeps is not None
            else elapsed <= cfg.max_seconds
        )
        improved = value < best_cut
        if eligible:
            best[improved] = spins[improved]
            best_cut = torch.minimum(best_cut, value)
            quality_sweeps = completed
        # Preserve every new per-chain incumbent and the last observation.
        if bool(improved.any()) or not eligible:
            trace_bits.append(((spins + 1) / 2).to(torch.int8).cpu().numpy().copy())
            trace_cut.append(value.cpu().numpy().copy())
            trace_sweep.append(completed)
            trace_seconds.append(elapsed)
            trace_eligible.append(eligible)
        if not eligible and cfg.max_sweeps is None:
            break
        if cfg.method == "greedy":
            quiet_sweeps = quiet_sweeps + 1 if torch.equal(before, spins) else 0
            if quiet_sweeps == cfg.banks:
                break
    sync()
    if trace_sweep[-1] != completed:
        trace_bits.append(((spins + 1) / 2).to(torch.int8).cpu().numpy().copy())
        trace_cut.append(cut().cpu().numpy().copy())
        trace_sweep.append(completed)
        trace_seconds.append(last_elapsed)
        trace_eligible.append(completed <= quality_sweeps)
    arrays = {
        "initial_bits": labels,
        "initial_cut": initial_cut,
        "best_bits": ((best + 1) / 2).to(torch.int8).cpu().numpy(),
        "best_cut": best_cut.cpu().numpy(),
        "final_bits": ((spins + 1) / 2).to(torch.int8).cpu().numpy(),
        "final_cut": cut().cpu().numpy(),
        "trace_bits": np.asarray(trace_bits, dtype=np.int8),
        "trace_cut": np.asarray(trace_cut),
        "trace_sweep": np.asarray(trace_sweep),
        "trace_seconds": np.asarray(trace_seconds),
        "trace_eligible": np.asarray(trace_eligible),
        "visited": visited.cpu().numpy(),
        "changed": changed.cpu().numpy(),
        "initial_rng": initial_rng,
        "final_rng": generator.get_state().cpu().numpy(),
        "bank_nodes": np.stack([b for b, _ in banks]),
        "bank_colors": np.stack([c for _, c in banks]),
    }
    summary = {
        "method": cfg.method,
        "width": cfg.width,
        "best_cut": int(best_cut.min()),
        "replicas": cfg.replicas,
        "physical_update_pbits_per_model": graph.size if cfg.method != "greedy" else 0,
        "auxiliary_pbits": 0,
        "visited_variables": int(visited.sum()),
        "changed_variables_per_replica": changed.sum(-1).cpu().tolist(),
        "completed_sweeps": completed,
        "quality_sweeps": quality_sweeps,
        "setup_seconds": setup_seconds,
        "search_seconds": perf_counter() - search,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "maximum_bank_colors": max(int(c.max()) + 1 for _, c in banks),
        "greedy_bank_stationary": cfg.method == "greedy" and quiet_sweeps == cfg.banks,
    }
    return arrays, summary
