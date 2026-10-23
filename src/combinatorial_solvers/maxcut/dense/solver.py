"""Dense block heat bath, stalled synchronous p-bits, and a dSB control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class Settings:
    method: str = "block-gibbs"
    device: str = "cpu"
    seed: int = 0
    chains: int = 16
    block_size: int = 8
    max_steps: int = 1000000
    max_seconds: float = 5.0
    cycle_steps: int = 1024
    observe_every: int = 8
    beta_min: float = 0.1
    beta_max: float = 8.0
    update_rate: float = 0.1
    step_size: float = 1.0
    sca_pin: float = 8.0
    sca_temperature_start: float = 12.0
    sca_temperature_end: float = 5.0
    onsager_gain: float = 1.05
    onsager_end_ramp: bool = True
    field_mode: str = "full"
    sparse_field_fraction: float = 0.125
    quality_max_step: int | None = None

    def __post_init__(self):
        if self.method not in {
            "block-gibbs",
            "stalled-pbit",
            "dsb",
            "sca",
            "onsager",
        } or self.device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("Unknown dense method or device")
        for value in (
            self.chains,
            self.max_steps,
            self.cycle_steps,
            self.observe_every,
            self.block_size,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Loop and population counts must be positive integers")
        if (
            self.cycle_steps < 2
            or self.block_size > 10
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("Invalid schedule, block size, or seed")
        if (
            not np.isfinite(
                [self.max_seconds, self.beta_min, self.beta_max, self.update_rate, self.step_size]
            ).all()
            or not self.max_seconds > 0
            or not 0 < self.beta_min <= self.beta_max
            or not 0 < self.update_rate <= 1
            or not 0 < self.step_size <= 1.25
        ):
            raise ValueError("Invalid time, temperature, activity, or integration step")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid quality cutoff")
        if self.field_mode not in {"full", "flips"} or not 0 < self.sparse_field_fraction <= 0.25:
            raise ValueError("Invalid field update policy")
        if (
            not np.isfinite(
                [
                    self.sca_pin,
                    self.sca_temperature_start,
                    self.sca_temperature_end,
                    self.onsager_gain,
                ]
            ).all()
            or self.sca_pin < 0
            or not 0 < self.sca_temperature_end <= self.sca_temperature_start
            or self.onsager_gain < 0
        ):
            raise ValueError("Invalid SCA temperature or correction")


def sign(values):
    return torch.where(values >= 0, 1.0, -1.0)


def patterns(width, device, dtype):
    number = torch.arange(2**width, device=device)
    return (2 * ((number[:, None] >> torch.arange(width, device=device)) & 1) - 1).to(dtype)


def block_probabilities(current, field, internal, candidates, beta):
    outside = field - current @ internal
    energy = outside @ candidates.T + 0.5 * (candidates * (candidates @ internal)).sum(-1)
    return torch.softmax(-beta * energy, dim=-1)


def block_update(spins, field, matrix, nodes, candidates, beta, uniforms):
    current = spins[:, nodes]
    rows = matrix[nodes]
    internal = rows[:, nodes]
    probability = block_probabilities(current, field[:, nodes], internal, candidates, beta)
    index = (uniforms[:, None] >= probability.cumsum(-1)).sum(-1).clamp_max(len(candidates) - 1)
    proposed = candidates[index]
    field.add_((proposed - current) @ rows)
    spins[:, nodes] = proposed


def stalled_update(spins, field, beta, uniforms, activity, rate):
    proposed = 2 * (uniforms < torch.sigmoid(-2 * beta * field)).to(spins.dtype) - 1
    selected = activity < rate
    spins.copy_(torch.where(selected, proposed, spins))
    return selected


def dsb_update(position, momentum, field, ramp, coupling, step_size):
    # J_paper = -W_MaxCut, a0 = 1; symplectic Euler updates momentum first.
    momentum.add_(step_size * (-(1 - ramp) * position - coupling * field))
    position.add_(step_size * momentum)
    outside = position.abs() > 1
    position.copy_(torch.where(outside, sign(position), position))
    momentum.masked_fill_(outside, 0)


def sca_update(spins, field, previous, echo, temperature, pinning, gain, uniforms):
    """Paper Algorithm 1, with J=-W and FP64 decision coefficients."""
    z = -spins.to(torch.float64) * field - gain * echo[:, None] * spins * previous + pinning
    keep = torch.clamp(z / (4 * temperature) + 0.5, 0, 1)
    flipped = uniforms > keep
    coefficient = (z.abs() < 2 * temperature).sum(-1).to(torch.float64) / (2 * temperature)
    old = spins.clone()
    spins.copy_(torch.where(flipped, -spins, spins))
    previous.copy_(old)
    echo.copy_(coefficient)
    return flipped


def onsager_ramp(step, steps, enabled):
    if not enabled or step < 0.7 * steps:
        return 1.0
    return (steps - 1 - step) / max(1.0, steps - 1 - 0.7 * steps)


def advance_field(old, new, field, matrix, mode, fraction):
    if mode == "full":
        return new @ matrix, "full", 0
    active = torch.nonzero((new != old).any(0), as_tuple=False).flatten()
    if not len(active):
        return field, "unchanged", 0
    if len(active) <= fraction * matrix.shape[0]:
        field.add_((new[:, active] - old[:, active]) @ matrix[active])
        return field, "incremental", len(active)
    return new @ matrix, "full", 0


def solve(instance, settings):
    if settings.method in {"sca", "onsager"} and instance.family != "sk":
        raise ValueError(
            "The counted Onsager coefficient is restricted to complete unit signed graphs"
        )
    prepared = perf_counter()
    device = torch.device(settings.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    matrix = torch.from_numpy(instance.matrix()).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    rng_initial = generator.get_state().cpu().numpy().copy()
    spins = (
        2
        * torch.randint(0, 2, (settings.chains, instance.size), generator=generator, device=device)
        - 1
    ).to(torch.float32)
    initial = ((spins + 1) / 2).to(torch.uint8).cpu().numpy().copy()
    rng_after_initial = generator.get_state().cpu().numpy().copy()
    visited = torch.zeros(instance.size, device=device, dtype=torch.bool)
    position, momentum = torch.empty(0, device=device), torch.empty(0, device=device)
    restarts = 0
    previous = (
        spins.clone() if settings.method in {"sca", "onsager"} else torch.empty(0, device=device)
    )
    echo = torch.zeros(settings.chains, device=device, dtype=torch.float64)
    last_flipped = torch.zeros_like(spins, dtype=torch.bool)
    flip_counts = torch.zeros(settings.chains, device=device, dtype=torch.int64)
    flip_back_counts = torch.zeros_like(flip_counts)
    if settings.method in {"sca", "onsager"}:
        restarts = 1

    def initialize_continuous(restart):
        nonlocal position, momentum, spins, restarts
        if restart:
            spins = (
                2 * torch.randint(0, 2, spins.shape, generator=generator, device=device) - 1
            ).to(torch.float32)
        magnitude = (
            torch.rand(spins.shape, generator=generator, device=device).clamp_min(
                torch.finfo(torch.float32).eps
            )
            * 0.01
        )
        position = spins * magnitude
        momentum = (2 * torch.rand(spins.shape, generator=generator, device=device) - 1) * 0.01
        restarts += 1

    if settings.method == "dsb":
        initialize_continuous(False)
    initial_position = position.cpu().numpy().copy()
    initial_momentum = momentum.cpu().numpy().copy()
    blocks = (
        [
            torch.arange(i, min(i + settings.block_size, instance.size), device=device)
            for i in range(0, instance.size, settings.block_size)
        ]
        if settings.method == "block-gibbs"
        else []
    )
    candidates = {
        width: patterns(width, device, torch.float32) for width in {len(b) for b in blocks}
    }
    permutation = torch.empty(0, device=device, dtype=torch.int64)
    best_cut = torch.full(
        (settings.chains,), -instance.edge_count - 1, device=device, dtype=torch.int64
    )
    best_bits = torch.zeros_like(spins)
    best_step = torch.zeros(settings.chains, device=device, dtype=torch.int64)
    field = spins @ matrix
    field_operations = {"full": 1, "incremental": 0, "unchanged": 0}
    max_selected_rows = 0
    traces = []
    # Unit signed or 0/1 couplings: RMS magnitude, including absent off-diagonal edges.
    rms = np.sqrt(2 * instance.edge_count / (instance.size * (instance.size - 1)))
    scale = rms * np.sqrt(instance.size)
    if scale == 0:
        raise ValueError("The generated dense graph has no active edge")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def observe(step, started=None):
        current_field = field
        integer_field = current_field.to(torch.int64)
        if not torch.equal(current_field, integer_field.to(torch.float32)):
            raise AssertionError("Dense local fields lost integer exactness")
        twice_energy = (spins.to(torch.int64) * integer_field).sum(-1)
        numerator = 2 * instance.total_weight - twice_energy
        if torch.any(numerator % 4):
            raise AssertionError("Dense cut parity identity failed")
        cuts = numerator // 4
        synchronize()
        elapsed = 0.0 if started is None else perf_counter() - started
        within = elapsed <= settings.max_seconds and (
            settings.quality_max_step is None or step <= settings.quality_max_step
        )
        improved = (cuts > best_cut) & within
        best_cut.copy_(torch.where(improved, cuts, best_cut))
        best_bits.copy_(torch.where(improved[:, None], (spins + 1) / 2, best_bits))
        best_step.copy_(torch.where(improved, step, best_step))
        traces.append(
            (step, elapsed, within, cuts.cpu().numpy().copy(), best_cut.cpu().numpy().copy())
        )

    observe(0)
    synchronize()
    preparation_seconds = perf_counter() - prepared
    started, completed, failure = perf_counter(), 0, None
    try:
        with torch.no_grad():
            for step in range(1, settings.max_steps + 1):
                phase = (step - 1) % settings.cycle_steps
                ramp = phase / (settings.cycle_steps - 1)
                beta = settings.beta_min * (settings.beta_max / settings.beta_min) ** ramp / scale
                if settings.method == "block-gibbs":
                    cursor = (step - 1) % len(blocks)
                    if cursor == 0:
                        permutation = torch.randperm(
                            len(blocks), generator=generator, device=device
                        )
                        order = permutation.cpu().tolist()
                    nodes = blocks[order[cursor]]
                    uniforms = torch.rand(settings.chains, generator=generator, device=device)
                    block_update(
                        spins, field, matrix, nodes, candidates[len(nodes)], beta, uniforms
                    )
                    visited[nodes] = True
                elif settings.method == "stalled-pbit":
                    old_spins = spins.clone()
                    uniforms = torch.rand(spins.shape, generator=generator, device=device)
                    activity = torch.rand(spins.shape, generator=generator, device=device)
                    selected = stalled_update(
                        spins, field, beta, uniforms, activity, settings.update_rate
                    )
                    visited |= selected.any(0)
                    field, kind, rows = advance_field(
                        old_spins,
                        spins,
                        field,
                        matrix,
                        settings.field_mode,
                        settings.sparse_field_fraction,
                    )
                    field_operations[kind] += 1
                    max_selected_rows = max(max_selected_rows, rows)
                elif settings.method in {"sca", "onsager"}:
                    if phase == 0 and step > 1:
                        spins = (
                            2 * torch.randint(0, 2, spins.shape, generator=generator, device=device)
                            - 1
                        ).to(torch.float32)
                        previous.copy_(spins)
                        echo.zero_()
                        last_flipped.zero_()
                        restarts += 1
                        field = spins @ matrix
                        field_operations["full"] += 1
                    temperature = (
                        settings.sca_temperature_start
                        * (settings.sca_temperature_end / settings.sca_temperature_start) ** ramp
                        * scale
                        / np.sqrt(2000.0)
                    )
                    pinning = settings.sca_pin * scale / np.sqrt(2000.0)
                    gain = settings.onsager_gain if settings.method == "onsager" else 0.0
                    gain *= onsager_ramp(phase, settings.cycle_steps, settings.onsager_end_ramp)
                    old_spins = spins.clone()
                    uniforms = torch.rand(spins.shape, generator=generator, device=device)
                    flipped = sca_update(
                        spins, field, previous, echo, temperature, pinning, gain, uniforms
                    )
                    flip_counts += flipped.sum(-1)
                    flip_back_counts += (flipped & last_flipped).sum(-1)
                    last_flipped.copy_(flipped)
                    visited.fill_(True)
                    field, kind, rows = advance_field(
                        old_spins,
                        spins,
                        field,
                        matrix,
                        settings.field_mode,
                        settings.sparse_field_fraction,
                    )
                    field_operations[kind] += 1
                    max_selected_rows = max(max_selected_rows, rows)
                else:
                    if phase == 0 and step > 1:
                        initialize_continuous(True)
                        field = spins @ matrix
                        field_operations["full"] += 1
                    old_spins = spins
                    dsb_update(position, momentum, field, ramp, 0.5 / scale, settings.step_size)
                    if not torch.isfinite(position).all() or not torch.isfinite(momentum).all():
                        raise FloatingPointError("Nonfinite dSB state")
                    spins = sign(position)
                    visited.fill_(True)
                    field, kind, rows = advance_field(
                        old_spins,
                        spins,
                        field,
                        matrix,
                        settings.field_mode,
                        settings.sparse_field_fraction,
                    )
                    field_operations[kind] += 1
                    max_selected_rows = max(max_selected_rows, rows)
                completed = step
                synchronize()
                last = (
                    perf_counter() - started >= settings.max_seconds or step == settings.max_steps
                )
                if step % settings.observe_every == 0 or last:
                    observe(step, started)
                if last:
                    break
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    synchronize()
    sampling_seconds = perf_counter() - started
    field_error = float((field - spins @ matrix).abs().max().item())
    if field_error not in (None, 0.0):
        failure = {"type": "FieldDrift", "message": str(field_error)}
    before = perf_counter()
    arrays = {
        "initial_bits": initial,
        "final_bits": ((spins + 1) / 2).to(torch.uint8).cpu().numpy(),
        "best_bits": best_bits.to(torch.uint8).cpu().numpy(),
        "best_cut": best_cut.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "visited_variables": visited.cpu().numpy(),
        "initial_position": initial_position,
        "initial_momentum": initial_momentum,
        "final_position": position.cpu().numpy(),
        "final_momentum": momentum.cpu().numpy(),
        "last_block_permutation": permutation.cpu().numpy(),
        "previous_spins": previous.cpu().numpy(),
        "echo_coefficient": echo.cpu().numpy(),
        "last_flip_mask": last_flipped.cpu().numpy(),
        "flip_counts": flip_counts.cpu().numpy(),
        "flip_back_counts": flip_back_counts.cpu().numpy(),
        "rng_initial": rng_initial,
        "rng_after_initial": rng_after_initial,
        "rng_final": generator.get_state().cpu().numpy(),
        "trace_step": np.asarray([r[0] for r in traces], dtype=np.int64),
        "trace_seconds": np.asarray([r[1] for r in traces]),
        "trace_in_budget": np.asarray([r[2] for r in traces]),
        "trace_cut": np.stack([r[3] for r in traces]),
        "trace_best_cut": np.stack([r[4] for r in traces]),
    }
    return arrays, {
        "settings": asdict(settings),
        "best_cut": int(arrays["best_cut"].max()),
        "completed_steps": completed,
        "model_binary_variables": instance.size,
        "physical_update_pbits_per_model": instance.size if settings.method != "dsb" else 0,
        "continuous_states_per_replica": 2 * instance.size if settings.method == "dsb" else 0,
        "objective_active_variables": instance.active_variables,
        "auxiliary_pbits": 0,
        "visited_variables": int(arrays["visited_variables"].sum()),
        "matrix_bytes": matrix.numel() * matrix.element_size(),
        "coupling_rms": float(rms),
        "field_scale": float(scale),
        "field_error": field_error,
        "field_operations": field_operations,
        "maximum_selected_field_rows": max_selected_rows,
        "continuous_initializations": restarts if settings.method == "dsb" else 0,
        "anneals_started": max(1, restarts),
        "history_binary_bits_per_replica": instance.size
        if settings.method in {"sca", "onsager"}
        else 0,
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "state_transfer_seconds": perf_counter() - before,
        "failure": failure,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
