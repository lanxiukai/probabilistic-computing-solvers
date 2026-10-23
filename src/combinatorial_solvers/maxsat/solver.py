"""Native higher-order single-site updates for weighted soft clauses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from .model import Instance


@dataclass(frozen=True)
class Settings:
    method: str = "pbit"
    device: str = "cpu"
    seed: int = 0
    chains: int = 16
    max_sweeps: int = 1000000
    max_seconds: float = 5.0
    cycle_sweeps: int = 512
    observe_every: int = 4
    beta_min: float = 0.1
    beta_max: float = 16.0
    dtype: str = "float32"
    quality_max_sweep: int | None = None
    stop_at_zero: bool = False
    target_cost: int | None = None

    def __post_init__(self):
        if self.method not in {"pbit", "metropolis"} or self.device not in {"cpu", "cuda"}:
            raise ValueError("Unknown Max-SAT method or device")
        for value in (self.chains, self.max_sweeps, self.cycle_sweeps, self.observe_every):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Loop and population counts must be positive integers")
        if self.cycle_sweeps < 2 or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Invalid cycle length or seed")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("Unsupported probability precision")
        if not np.isfinite([self.max_seconds, self.beta_min, self.beta_max]).all():
            raise ValueError("Time and inverse temperatures must be finite")
        if not self.max_seconds > 0 or not 0 < self.beta_min <= self.beta_max:
            raise ValueError("Invalid time or inverse-temperature limits")
        if (
            self.quality_max_sweep is not None
            and not 0 <= self.quality_max_sweep <= self.max_sweeps
        ):
            raise ValueError("Quality cutoff outside the horizon")
        if self.target_cost is not None and (
            isinstance(self.target_cost, bool)
            or not isinstance(self.target_cost, int)
            or self.target_cost < 0
        ):
            raise ValueError("Target must be a nonnegative integer")


def prepare_blocks(instance, device, dtype):
    occurrences = [[] for _ in range(instance.variables)]
    for clause, row in enumerate(instance.literals):
        for position, literal in enumerate(row):
            if literal:
                occurrences[abs(int(literal)) - 1].append((clause, position))
    blocks = []
    other_width = instance.literals.shape[1] - 1
    for nodes in instance.colors():
        width = max(map(lambda node: len(occurrences[node]), nodes), default=0)
        others = np.zeros((len(nodes), width, other_width), dtype=np.int64)
        positive = np.zeros_like(others, dtype=bool)
        valid = np.zeros_like(others, dtype=bool)
        coefficients = np.zeros((len(nodes), width), dtype=np.float64)
        for row, node in enumerate(nodes):
            for slot, (clause, position) in enumerate(occurrences[node]):
                literal = instance.literals[clause, position]
                rest = [
                    int(x) for j, x in enumerate(instance.literals[clause]) if j != position and x
                ]
                others[row, slot, : len(rest)] = np.abs(rest) - 1
                positive[row, slot, : len(rest)] = np.asarray(rest) > 0
                valid[row, slot, : len(rest)] = True
                coefficients[row, slot] = -int(instance.weights[clause]) * (
                    1 if literal > 0 else -1
                )
        blocks.append(
            {
                "nodes": torch.tensor(nodes, device=device),
                "others": torch.tensor(others, device=device),
                "positive": torch.tensor(positive, device=device),
                "valid": torch.tensor(valid, device=device),
                "coefficients": torch.tensor(coefficients, device=device, dtype=dtype),
            }
        )
    return blocks


def cost_one_minus_zero(bits, block):
    values = bits[:, block["others"]]
    false_literals = torch.where(block["positive"], 1 - values, values)
    other_false = torch.where(block["valid"], false_literals, 1).prod(dim=-1)
    return (other_false * block["coefficients"]).sum(dim=-1)


def update(bits, block, beta_over_scale, uniforms, method):
    delta = cost_one_minus_zero(bits, block)
    nodes = block["nodes"]
    if method == "pbit":
        bits[:, nodes] = (uniforms < torch.sigmoid(-beta_over_scale * delta)).to(bits.dtype)
    elif method == "metropolis":
        current = bits[:, nodes]
        probability = torch.exp(torch.clamp(-beta_over_scale * (1 - 2 * current) * delta, max=0))
        bits[:, nodes] = torch.where(uniforms < probability, 1 - current, current)
    else:
        raise ValueError("Unknown update rule")


def solve(instance: Instance, settings: Settings, *, sampling_deadline=None):
    prepared = perf_counter()
    device, dtype = torch.device(settings.device), getattr(torch, settings.dtype)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    blocks = prepare_blocks(instance, device, dtype)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    rng_initial = generator.get_state().cpu().numpy().copy()
    bits = torch.randint(
        0, 2, (settings.chains, instance.variables), generator=generator, device=device
    ).to(dtype)
    initial = bits.to(torch.uint8).cpu().numpy().copy()
    rng_after_initial = generator.get_state().cpu().numpy().copy()
    indices = torch.tensor(np.maximum(np.abs(instance.literals) - 1, 0), device=device)
    positive = torch.tensor(instance.literals > 0, device=device)
    valid = torch.tensor(instance.literals != 0, device=device)
    weights = torch.tensor(instance.weights, device=device, dtype=torch.int64)
    best_cost = torch.full(
        (settings.chains,), sum(map(int, instance.weights)) + 1, device=device, dtype=torch.int64
    )
    best_bits = torch.zeros_like(bits)
    best_sweep = torch.zeros_like(best_cost)
    traces = []
    search_budget = settings.max_seconds

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def observe(sweep, elapsed):
        within = elapsed <= search_budget and (
            settings.quality_max_sweep is None or sweep <= settings.quality_max_sweep
        )
        values = bits[:, indices]
        truth = torch.where(positive, values > 0, values == 0) & valid
        cost = ((~truth.any(-1)).to(torch.int64) * weights).sum(-1)
        improved = (cost < best_cost) & within
        best_bits.copy_(torch.where(improved[:, None], bits, best_bits))
        best_cost.copy_(torch.where(improved, cost, best_cost))
        best_sweep.copy_(torch.where(improved, sweep, best_sweep))
        traces.append(
            (sweep, elapsed, within, cost.cpu().numpy().copy(), best_cost.cpu().numpy().copy())
        )

    observe(0, 0.0)
    synchronize()
    preparation_seconds = perf_counter() - prepared
    started, completed, failure = perf_counter(), 0, None
    if sampling_deadline is not None:
        search_budget = min(search_budget, sampling_deadline - started)
    scale = int(instance.weights.max())
    target = (
        settings.target_cost
        if settings.target_cost is not None
        else (0 if settings.stop_at_zero else None)
    )
    horizon = 0 if target is not None and traces[-1][4].min() <= target else settings.max_sweeps
    if search_budget <= 0:
        horizon = 0
        failure = {"type": "PreparationBudgetExceeded", "message": "No sampling budget remains"}
    try:
        with torch.no_grad():
            for sweep in range(1, horizon + 1):
                fraction = ((sweep - 1) % settings.cycle_sweeps) / (settings.cycle_sweeps - 1)
                beta = settings.beta_min * (settings.beta_max / settings.beta_min) ** fraction
                for block in blocks:
                    uniforms = torch.rand(
                        (settings.chains, len(block["nodes"])),
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                    update(bits, block, beta / scale, uniforms, settings.method)
                completed = sweep
                synchronize()
                elapsed = perf_counter() - started
                last = elapsed >= search_budget or sweep == settings.max_sweeps
                if sweep % settings.observe_every == 0 or last:
                    observe(sweep, elapsed)
                if last or (target is not None and traces[-1][4].min() <= target):
                    break
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    synchronize()
    sampling_seconds = perf_counter() - started
    transfer_started = perf_counter()
    arrays = {
        "initial_bits": initial,
        "final_bits": bits.to(torch.uint8).cpu().numpy(),
        "best_bits": best_bits.to(torch.uint8).cpu().numpy(),
        "best_cost": best_cost.cpu().numpy(),
        "best_sweep": best_sweep.cpu().numpy(),
        "rng_initial": rng_initial,
        "rng_after_initial": rng_after_initial,
        "rng_final": generator.get_state().cpu().numpy(),
        "trace_sweep": np.asarray([r[0] for r in traces], dtype=np.int64),
        "trace_seconds": np.asarray([r[1] for r in traces]),
        "trace_in_budget": np.asarray([r[2] for r in traces]),
        "trace_cost": np.stack([r[3] for r in traces]),
        "trace_best_cost": np.stack([r[4] for r in traces]),
    }
    summary = {
        "settings": asdict(settings),
        "physical_update_pbits_per_model": instance.variables,
        "objective_active_variables": instance.active_variables,
        "auxiliary_pbits": 0,
        "representation": "native higher-order clause conditionals; not pairwise quadratization",
        "clauses": len(instance.weights),
        "colors": len(blocks),
        "completed_sweeps": completed,
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "effective_search_seconds": search_budget,
        "state_transfer_seconds": perf_counter() - transfer_started,
        "best_cost": int(arrays["best_cost"].min()),
        "failure": failure,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
    return arrays, summary
