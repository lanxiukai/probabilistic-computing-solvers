"""Colored MWIS heat-bath and Metropolis baselines with raw-state evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from combinatorial_solvers.core.coloring import dsatur_coloring

from .model import Instance
from .pair_updates import prepare as prepare_pairs
from .pair_updates import update as update_pairs


@dataclass(frozen=True)
class Settings:
    method: str = "pbit"
    device: str = "cpu"
    seed: int = 0
    chains: int = 16
    max_sweeps: int = 4096
    max_seconds: float = 5.0
    cycle_sweeps: int = 256
    observe_every: int = 16
    beta_min: float = 0.1
    beta_max: float = 16.0
    penalty_multiplier: int = 2
    dtype: str = "float32"
    quality_max_sweep: int | None = None

    def __post_init__(self):
        if self.method not in {
            "pbit",
            "metropolis",
            "pair-gibbs",
            "pair-metropolis",
        } or self.device not in {"cpu", "cuda"}:
            raise ValueError("Unknown method or device")
        positive = (
            self.chains,
            self.max_sweeps,
            self.cycle_sweeps,
            self.observe_every,
            self.penalty_multiplier,
        )
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in positive):
            raise ValueError("Counts and penalty multiplier must be positive integers")
        if self.cycle_sweeps < 2 or self.seed < 0 or self.dtype not in {"float32", "float64"}:
            raise ValueError("Invalid cycle, seed, or precision")
        if self.quality_max_sweep is not None and not (
            isinstance(self.quality_max_sweep, int)
            and 0 <= self.quality_max_sweep <= self.max_sweeps
        ):
            raise ValueError("Quality cutoff must be within the fixed horizon")
        if (
            not 0 < self.beta_min <= self.beta_max
            or not np.isfinite([self.beta_min, self.beta_max, self.max_seconds]).all()
            or self.max_seconds <= 0
        ):
            raise ValueError("Finite positive temperature and time settings required")


def update_color(bits, nodes, neighbors, mask, weights, penalty, beta, scale, uniforms, method):
    counts = (bits[:, neighbors] * mask).sum(dim=-1)
    cost_one = -weights + penalty * counts
    current = bits[:, nodes]
    if method == "pbit":
        updated = (uniforms < torch.sigmoid(-beta * cost_one / scale)).to(bits.dtype)
    elif method == "metropolis":
        delta = (1 - 2 * current) * cost_one
        accept = uniforms < torch.exp(torch.clamp(-beta * delta / scale, max=0))
        updated = torch.where(accept, 1 - current, current)
    else:
        raise ValueError("Unknown update method")
    bits[:, nodes] = updated


def solve(instance: Instance, settings: Settings):
    prepared = perf_counter()
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    dtype = getattr(torch, settings.dtype)
    maximum_weight = int(instance.weights.max())
    penalty = settings.penalty_multiplier * maximum_weight + 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = instance.ising(penalty)
    pair_arrays = {}
    if settings.method.startswith("pair-"):
        blocks, pairs, assigned, block_colors = prepare_pairs(instance, device, dtype)
        pair_arrays = {
            "pair_vertices": pairs,
            "pair_assignment": assigned,
            "pair_colors": block_colors,
        }
    else:
        colors = dsatur_coloring(model)
        adjacency = instance.adjacency()
        blocks = []
        for color in colors:
            width = max((len(adjacency[i]) for i in color), default=0)
            neighbors = np.zeros((len(color), width), dtype=np.int64)
            mask = np.zeros((len(color), width), dtype=np.float64)
            for row, node in enumerate(color):
                neighbors[row, : len(adjacency[node])] = adjacency[node]
                mask[row, : len(adjacency[node])] = 1
            blocks.append(
                (
                    torch.tensor(color, device=device),
                    torch.tensor(neighbors, device=device),
                    torch.tensor(mask, dtype=dtype, device=device),
                    torch.tensor(instance.weights[list(color)], dtype=dtype, device=device),
                )
            )
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    rng_initial = generator.get_state().cpu().numpy().copy()
    bits = torch.randint(
        0, 2, (settings.chains, instance.size), generator=generator, device=device
    ).to(dtype)
    initial = bits.to(torch.uint8).cpu().numpy().copy()
    rng_after_initial = generator.get_state().cpu().numpy().copy()
    weights = torch.tensor(instance.weights, dtype=torch.int64, device=device)
    left = torch.tensor(instance.edges[:, 0], device=device)
    right = torch.tensor(instance.edges[:, 1], device=device)
    best_bits = torch.zeros_like(bits)
    best_weight = torch.full((settings.chains,), -1, dtype=torch.int64, device=device)
    best_sweep = torch.full_like(best_weight, -1)
    trace = []
    interrupted = None

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def observe(sweep, elapsed, in_budget):
        in_budget = in_budget and (
            settings.quality_max_sweep is None or sweep <= settings.quality_max_sweep
        )
        current_weight = (bits.to(torch.int64) * weights).sum(dim=1)
        conflicts = (bits[:, left] * bits[:, right]).to(torch.int64).sum(dim=1)
        improved = (conflicts == 0) & (current_weight > best_weight) & in_budget
        best_weight.copy_(torch.where(improved, current_weight, best_weight))
        best_sweep.copy_(torch.where(improved, sweep, best_sweep))
        best_bits.copy_(torch.where(improved[:, None], bits, best_bits))
        trace.append(
            (
                sweep,
                elapsed,
                in_budget,
                current_weight.cpu().numpy().copy(),
                conflicts.cpu().numpy().copy(),
                best_weight.cpu().numpy().copy(),
            )
        )

    observe(0, 0.0, True)
    synchronize()
    preparation_seconds = perf_counter() - prepared
    started = perf_counter()
    completed = 0
    try:
        with torch.no_grad():
            for sweep in range(1, settings.max_sweeps + 1):
                fraction = ((sweep - 1) % settings.cycle_sweeps) / (settings.cycle_sweeps - 1)
                beta = settings.beta_min * (settings.beta_max / settings.beta_min) ** fraction
                if pair_arrays:
                    for block in blocks:
                        uniforms = torch.rand(
                            (settings.chains, len(block["a"]), 3),
                            generator=generator,
                            device=device,
                            dtype=dtype,
                        )
                        update_pairs(
                            bits, block, penalty, beta / maximum_weight, uniforms, settings.method
                        )
                else:
                    for nodes, neighbors, mask, local_weights in blocks:
                        uniforms = torch.rand(
                            (settings.chains, len(nodes)),
                            generator=generator,
                            device=device,
                            dtype=dtype,
                        )
                        update_color(
                            bits,
                            nodes,
                            neighbors,
                            mask,
                            local_weights,
                            penalty,
                            beta,
                            maximum_weight,
                            uniforms,
                            settings.method,
                        )
                completed = sweep
                synchronize()
                elapsed = perf_counter() - started
                last = elapsed >= settings.max_seconds or sweep == settings.max_sweeps
                if sweep % settings.observe_every == 0 or last:
                    observe(sweep, elapsed, elapsed <= settings.max_seconds)
                if last:
                    break
    except BaseException as error:
        interrupted = {"type": type(error).__name__, "message": str(error)}
    synchronize()
    sampling_seconds = perf_counter() - started
    arrays = {
        **pair_arrays,
        "initial_bits": initial,
        "final_bits": bits.to(torch.uint8).cpu().numpy(),
        "best_bits": best_bits.to(torch.uint8).cpu().numpy(),
        "best_weight": best_weight.cpu().numpy(),
        "best_sweep": best_sweep.cpu().numpy(),
        "rng_initial": rng_initial,
        "rng_after_initial": rng_after_initial,
        "rng_final": generator.get_state().cpu().numpy(),
        "trace_sweep": np.asarray([x[0] for x in trace], dtype=np.int64),
        "trace_seconds": np.asarray([x[1] for x in trace]),
        "trace_in_budget": np.asarray([x[2] for x in trace]),
        "trace_current_weight": np.stack([x[3] for x in trace]),
        "trace_conflicts": np.stack([x[4] for x in trace]),
        "trace_best_weight": np.stack([x[5] for x in trace]),
    }
    observed = arrays["best_weight"] >= 0
    summary = {
        "settings": asdict(settings),
        "variables": instance.size,
        "physical_update_pbits_per_model": instance.size,
        "auxiliary_pbits": 0,
        "edges": len(instance.edges),
        "colors": len(blocks),
        "random_draws_per_block": 3 if pair_arrays else 1,
        "kernel_scope": "two-variable joint conditional kernel"
        if pair_arrays
        else "single-site kernel",
        "penalty": penalty,
        "completed_sweeps": completed,
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "sampling_budget_overrun_seconds": max(0.0, sampling_seconds - settings.max_seconds),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "feasible_chains": int(observed.sum()),
        "best_feasible_weight": int(arrays["best_weight"].max()) if observed.any() else None,
        "failure": interrupted,
        "observation_scope": (
            "Initial state and completed observation sweeps; only observations within "
            "sampling budget improve native best. No repair or greedy augmentation."
        ),
    }
    return arrays, summary
