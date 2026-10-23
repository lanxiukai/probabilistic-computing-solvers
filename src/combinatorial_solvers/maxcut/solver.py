"""Properly colored heat-bath/Metropolis annealing and replica exchange."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class Settings:
    method: str = "pbit"
    schedule: str = "anneal"
    device: str = "cpu"
    seed: int = 0
    chains: int = 16
    max_sweeps: int = 1000000
    max_seconds: float = 5.0
    cycle_sweeps: int = 512
    observe_every: int = 4
    swap_every: int = 4
    beta_min: float = 0.1
    beta_max: float = 16.0
    quality_max_sweep: int | None = None

    def __post_init__(self):
        if self.method not in {"pbit", "metropolis"} or self.schedule not in {"anneal", "pt"}:
            raise ValueError("Unsupported update or schedule")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Unsupported device")
        for value in (
            self.chains,
            self.max_sweeps,
            self.cycle_sweeps,
            self.observe_every,
            self.swap_every,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Population and loop counts must be positive integers")
        if self.cycle_sweeps < 2 or self.schedule == "pt" and self.chains < 2:
            raise ValueError("Insufficient schedule or population size")
        if (
            not np.isfinite([self.max_seconds, self.beta_min, self.beta_max]).all()
            or not self.max_seconds > 0
            or not 0 < self.beta_min <= self.beta_max
        ):
            raise ValueError("Invalid time or inverse temperatures")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Invalid seed")
        if (
            self.quality_max_sweep is not None
            and not 0 <= self.quality_max_sweep <= self.max_sweeps
        ):
            raise ValueError("Quality cutoff outside the horizon")


def prepare_blocks(instance, device):
    adjacency = instance.adjacency()
    if max(sum(abs(w) for _, w in row) for row in adjacency) >= 2**24:
        raise ValueError("Conditional integer field exceeds exact FP32 range")
    blocks = []
    for nodes in instance.colors():
        width = max(len(adjacency[i]) for i in nodes)
        neighbors = np.zeros((len(nodes), width), dtype=np.int64)
        weights = np.zeros((len(nodes), width), dtype=np.float32)
        for i, node in enumerate(nodes):
            for j, (other, w) in enumerate(adjacency[node]):
                neighbors[i, j], weights[i, j] = other, w
        blocks.append(
            {
                "nodes": torch.tensor(nodes, device=device),
                "neighbors": torch.tensor(neighbors, device=device),
                "weights": torch.tensor(weights, device=device),
            }
        )
    return blocks


def update(spins, block, beta, uniforms, method):
    field = (spins[:, block["neighbors"]] * block["weights"]).sum(-1)
    current = spins[:, block["nodes"]]
    if method == "pbit":
        updated = 2 * (uniforms < torch.sigmoid(-2 * beta * field)).to(spins.dtype) - 1
    elif method == "metropolis":
        probability = torch.exp(torch.clamp(2 * beta * current * field, max=0))
        updated = torch.where(uniforms < probability, -current, current)
    else:
        raise ValueError("Unknown update")
    spins[:, block["nodes"]] = updated


def exchange(spins, energy, betas, left, uniforms):
    right = left + 1
    log_ratio = (betas[left] - betas[right]) * (energy[left] - energy[right])
    accepted = uniforms < torch.exp(torch.clamp(log_ratio, max=0))
    a, b = spins[left].clone(), spins[right].clone()
    spins[left] = torch.where(accepted[:, None], b, a)
    spins[right] = torch.where(accepted[:, None], a, b)
    return accepted


def solve(instance, settings):
    prepared = perf_counter()
    device = torch.device(settings.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    blocks = prepare_blocks(instance, device)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    rng_initial = generator.get_state().cpu().numpy().copy()
    spins = (
        2
        * torch.randint(0, 2, (settings.chains, instance.size), device=device, generator=generator)
        - 1
    ).to(torch.float32)
    initial = ((spins + 1) / 2).to(torch.uint8).cpu().numpy().copy()
    rng_after_initial = generator.get_state().cpu().numpy().copy()
    a, b = (torch.tensor(instance.edges[:, i], device=device) for i in (0, 1))
    weights = torch.tensor(instance.weights, device=device)
    scale = max(abs(int(w)) for w in instance.weights)
    betas = (
        torch.logspace(
            np.log10(settings.beta_min), np.log10(settings.beta_max), settings.chains, device=device
        )
        / scale
    )
    best_cut = torch.full(
        (settings.chains,),
        -sum(abs(int(w)) for w in instance.weights) - 1,
        device=device,
        dtype=torch.int64,
    )
    best_bits = torch.zeros_like(spins)
    best_sweep = torch.zeros(settings.chains, device=device, dtype=torch.int64)
    swap_proposals = torch.zeros(settings.chains - 1, device=device, dtype=torch.int64)
    swap_accepted = torch.zeros_like(swap_proposals)
    traces = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def cut_values():
        values = torch.zeros(settings.chains, device=device, dtype=torch.int64)
        for start in range(0, len(a), 8192):
            values += (
                (spins[:, a[start : start + 8192]] != spins[:, b[start : start + 8192]]).to(
                    torch.int64
                )
                * weights[start : start + 8192]
            ).sum(-1)
        return values

    def observe(sweep, elapsed):
        within = elapsed <= settings.max_seconds and (
            settings.quality_max_sweep is None or sweep <= settings.quality_max_sweep
        )
        cuts = cut_values()
        improved = (cuts > best_cut) & within
        best_bits.copy_(torch.where(improved[:, None], (spins + 1) / 2, best_bits))
        best_cut.copy_(torch.where(improved, cuts, best_cut))
        best_sweep.copy_(torch.where(improved, sweep, best_sweep))
        traces.append(
            (sweep, elapsed, within, cuts.cpu().numpy().copy(), best_cut.cpu().numpy().copy())
        )

    observe(0, 0.0)
    synchronize()
    preparation_seconds = perf_counter() - prepared
    started, completed, failure = perf_counter(), 0, None
    try:
        with torch.no_grad():
            for sweep in range(1, settings.max_sweeps + 1):
                if settings.schedule == "anneal":
                    fraction = ((sweep - 1) % settings.cycle_sweeps) / (settings.cycle_sweeps - 1)
                    beta = (
                        settings.beta_min
                        * (settings.beta_max / settings.beta_min) ** fraction
                        / scale
                    )
                else:
                    beta = betas[:, None]
                for block in blocks:
                    uniforms = torch.rand(
                        (settings.chains, len(block["nodes"])), device=device, generator=generator
                    )
                    update(spins, block, beta, uniforms, settings.method)
                if settings.schedule == "pt" and sweep % settings.swap_every == 0:
                    energy = int(instance.weights.sum()) - 2 * cut_values()
                    parity = (sweep // settings.swap_every - 1) % 2
                    left = torch.arange(parity, settings.chains - 1, 2, device=device)
                    uniforms = torch.rand(len(left), device=device, generator=generator)
                    accepted = exchange(spins, energy, betas, left, uniforms)
                    swap_proposals[left] += 1
                    swap_accepted[left] += accepted.to(torch.int64)
                completed = sweep
                synchronize()
                elapsed = perf_counter() - started
                last = elapsed >= settings.max_seconds or sweep == settings.max_sweeps
                if sweep % settings.observe_every == 0 or last:
                    observe(sweep, elapsed)
                if last:
                    break
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    synchronize()
    sampling_seconds = perf_counter() - started
    before = perf_counter()
    arrays = {
        "initial_bits": initial,
        "final_bits": ((spins + 1) / 2).to(torch.uint8).cpu().numpy(),
        "best_bits": best_bits.to(torch.uint8).cpu().numpy(),
        "best_cut": best_cut.cpu().numpy(),
        "best_sweep": best_sweep.cpu().numpy(),
        "rng_initial": rng_initial,
        "rng_after_initial": rng_after_initial,
        "rng_final": generator.get_state().cpu().numpy(),
        "swap_proposals": swap_proposals.cpu().numpy(),
        "swap_accepted": swap_accepted.cpu().numpy(),
        "trace_sweep": np.asarray([r[0] for r in traces], dtype=np.int64),
        "trace_seconds": np.asarray([r[1] for r in traces]),
        "trace_in_budget": np.asarray([r[2] for r in traces]),
        "trace_cut": np.stack([r[3] for r in traces]),
        "trace_best_cut": np.stack([r[4] for r in traces]),
    }
    return arrays, {
        "settings": asdict(settings),
        "best_cut": int(arrays["best_cut"].max()),
        "physical_update_pbits_per_model": instance.size,
        "objective_active_variables": instance.active_variables,
        "auxiliary_pbits": 0,
        "colors": len(blocks),
        "edges": len(instance.weights),
        "completed_sweeps": completed,
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "state_transfer_seconds": perf_counter() - before,
        "failure": failure,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
