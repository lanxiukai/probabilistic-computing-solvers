"""Colored Ising annealing, tempering, and population/restart controls."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import label

from cubic_spin_glass.core.padded_sparse import PaddedSparseIsing

from .model import CubicInstance

ALGORITHMS = (
    "metropolis-sa",
    "pbit-sa",
    "pbit-pt",
    "pbit-pt-icm",
    "metropolis-rsa",
    "metropolis-pa",
)


def _phase(name):
    """Domain-local profiling hook; the ordinary path performs no synchronization."""
    return nullcontext()


@dataclass(frozen=True)
class SolverConfig:
    algorithm: str = "pbit-pt-icm"
    sweeps: int = 128
    groups: int = 8
    temperatures: int = 8
    copies: int = 2
    beta_min: float = 0.1
    beta_max: float = 3.0
    exchange_interval: int = 1
    icm_interval: int = 1
    icm_beta_min: float = 1.0
    seed: int = 20261013
    device: str = "cpu"
    max_seconds: float = 60.0
    execution: str = "baseline"
    anneal_cycle_sweeps: int | None = None
    population_resampling: str = "multinomial"

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if self.population_resampling not in {"multinomial", "systematic"}:
            raise ValueError("population_resampling must be multinomial or systematic")
        if self.population_resampling != "multinomial" and self.algorithm != "metropolis-pa":
            raise ValueError("Systematic resampling requires metropolis-pa")
        for name in (
            "sweeps",
            "groups",
            "temperatures",
            "copies",
            "exchange_interval",
            "icm_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperatures < 2:
            raise ValueError("At least two temperature slots are required")
        if self.algorithm == "pbit-pt-icm" and self.copies != 2:
            raise ValueError("The current ICM implementation requires exactly two copies")
        if not (0 < self.beta_min < self.beta_max < math.inf):
            raise ValueError("Require finite 0 < beta_min < beta_max")
        if not (0 <= self.icm_beta_min < math.inf and 0 < self.max_seconds < math.inf):
            raise ValueError("Invalid ICM threshold or time limit")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA backends are supported")
        if self.execution not in {
            "baseline",
            "lattice-v1",
            "lattice-v2",
            "lattice-v3",
            "lattice-v4",
            "lattice-v5",
        }:
            raise ValueError("execution must be baseline or lattice-v1 through lattice-v5")
        if (
            self.execution in {"lattice-v2", "lattice-v3", "lattice-v4", "lattice-v5"}
            and torch.device(self.device).type != "cuda"
        ):
            raise ValueError("Fused lattice execution requires CUDA")
        if self.anneal_cycle_sweeps is not None and (
            isinstance(self.anneal_cycle_sweeps, bool)
            or not isinstance(self.anneal_cycle_sweeps, int)
            or self.anneal_cycle_sweeps < 2
        ):
            raise ValueError("anneal_cycle_sweeps must be an integer >= 2")


def role_seeds(instance: CubicInstance, seed: int, *, extra_roles=()) -> dict[str, int]:
    """Use distinct 32-bit keys; matching arms deliberately share initial/local words."""
    identity = ["ea3d-v1", instance.length, instance.boundary, instance.disorder_seed, seed]
    if getattr(instance, "coupling_id", None) is not None:
        identity = [
            "ea3d-explicit-v1",
            instance.length,
            instance.boundary,
            instance.coupling_id,
            seed,
        ]
    keys = {
        role: int.from_bytes(
            hashlib.sha256(json.dumps([*identity, role]).encode()).digest()[:4], "little"
        )
        for role in ("initial", "local", "exchange", "cluster", *extra_roles)
    }
    if len(set(keys.values())) != len(keys):
        raise ValueError("Random role seed collision")
    return keys


def solver_role_seeds(instance, seed, algorithm):
    return role_seeds(
        instance, seed, extra_roles=("resample",) if algorithm == "metropolis-pa" else ()
    )


def colored_sweep(
    spins,
    sparse,
    colors,
    beta,
    generator,
    *,
    metropolis=False,
    cached=None,
    fused_fields=None,
    fused_updates=None,
    beta_index=0,
):
    """Update an independent color, then read its new values for the next color."""
    for color, indices in enumerate(colors):
        if fused_updates is not None:
            uniform = torch.rand(
                (*spins.shape[:-1], indices.numel()),
                generator=generator,
                device=spins.device,
                dtype=spins.dtype,
            )
            fused_updates(spins, color, uniform, beta_index)
            continue
        if fused_fields is not None:
            field = fused_fields(spins, color)
        elif cached is None:
            field = sparse.local_fields(spins, indices)
        else:
            neighbors, couplings, bias = cached[color]
            field = (spins[..., neighbors] * couplings).sum(dim=-1) + bias
        uniform = torch.rand(
            field.shape, generator=generator, device=spins.device, dtype=spins.dtype
        )
        if metropolis:
            old = spins[..., indices]
            probability = torch.exp(torch.clamp(-2 * beta * old * field, max=0))
            spins[..., indices] = torch.where(uniform < probability, -old, old)
        else:
            spins[..., indices] = (
                2 * (uniform < torch.sigmoid(2 * beta * field)).to(spins.dtype) - 1
            )


def exchange_replicas(spins, energies, betas, parity, generator):
    """Disjoint odd/even neighboring-temperature exchanges, independently per copy."""
    left = torch.arange(parity, len(betas) - 1, 2, device=spins.device)
    right = left + 1
    log_ratio = (betas[left] - betas[right])[None, :, None] * (
        energies[:, left] - energies[:, right]
    )
    uniform = torch.rand(
        log_ratio.shape, generator=generator, device=spins.device, dtype=spins.dtype
    )
    accept = uniform < torch.exp(torch.clamp(log_ratio, max=0))
    old_left, old_right = spins[:, left].clone(), spins[:, right].clone()
    spins[:, left] = torch.where(accept[..., None], old_right, old_left)
    spins[:, right] = torch.where(accept[..., None], old_left, old_right)
    e_left, e_right = energies[:, left].clone(), energies[:, right].clone()
    energies[:, left] = torch.where(accept, e_right, e_left)
    energies[:, right] = torch.where(accept, e_left, e_right)
    return left, accept


def icm_pair(first, second, neighbors, rng):
    """Zhu 2015 zero-field variant: symmetry reduction, then a random-site cluster.

    Conserves the sum of the two energies, not generally each individual energy.
    A uniform disagreement site size-weights the connected-component selection.
    """
    disagreement = first != second
    global_flip = int(np.count_nonzero(disagreement) > first.size // 2)
    if global_flip:
        first *= -1
        disagreement = first != second
    candidates = np.flatnonzero(disagreement)
    if not candidates.size:
        return 0, global_flip, 0
    seed = int(rng.choice(candidates))
    visited = {seed}
    pending = [seed]
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if disagreement[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    cluster = list(visited)
    first[cluster] *= -1
    second[cluster] *= -1
    return len(cluster), global_flip, int(len(cluster) == candidates.size)


def icm_pair_lattice(first, second, length, rng):
    """Same random-site ICM on a complete open cubic graph, using compiled labeling."""
    disagreement = first != second
    global_flip = int(np.count_nonzero(disagreement) > first.size // 2)
    if global_flip:
        first *= -1
        disagreement = first != second
    candidates = np.flatnonzero(disagreement)
    if not candidates.size:
        return 0, global_flip, 0
    seed = int(rng.choice(candidates))
    labels, _count = label(disagreement.reshape(length, length, length))
    component = labels.ravel() == labels.ravel()[seed]
    first[component] *= -1
    second[component] *= -1
    size = int(np.count_nonzero(component))
    return size, global_flip, int(size == candidates.size)


def solve(
    instance: CubicInstance, config: SolverConfig, *, sampling_deadline: float | None = None
) -> tuple[dict, dict, dict]:
    """Fixed-horizon optimization; exact solutions are never passed to this function."""
    if sampling_deadline is not None and not math.isfinite(sampling_deadline):
        raise ValueError("The optional monotonic sampling deadline must be finite")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    lattice = config.execution in {
        "lattice-v1",
        "lattice-v2",
        "lattice-v3",
        "lattice-v4",
        "lattice-v5",
    }
    if lattice and instance.boundary != "open":
        raise ValueError("Lattice execution requires a complete open cubic graph")
    started = perf_counter()
    sparse = PaddedSparseIsing.from_model(instance.model, device=device, dtype=torch.float64)
    colors = tuple(color.to(device) for color in instance.colors)
    cached = (
        [(sparse.neighbors[c], sparse.couplings[c], sparse.bias[c]) for c in colors]
        if lattice
        else None
    )
    energy = sparse.energy
    if config.execution in {"lattice-v2", "lattice-v3"}:
        from .triton_energy import CubicEnergy

        energy = CubicEnergy(sparse, instance)
    elif config.execution in {"lattice-v4", "lattice-v5"}:
        from .triton_tiled_energy import TiledCubicEnergy

        energy = TiledCubicEnergy(sparse, instance)
    fused_fields = None
    if config.execution in {"lattice-v3", "lattice-v4"}:
        from .triton_fields import CubicFields

        fused_fields = CubicFields(sparse, colors)
    population_mode = config.algorithm in {"metropolis-rsa", "metropolis-pa"}
    keys = solver_role_seeds(instance, config.seed, config.algorithm)
    generators = {
        role: torch.Generator(device=device).manual_seed(keys[role])
        for role in (
            "initial",
            "local",
            "exchange",
            *(("resample",) if config.algorithm == "metropolis-pa" else ()),
        )
    }
    cluster_rng = np.random.Generator(np.random.PCG64(keys["cluster"]))
    rng_audit = {
        "seeds": keys,
        "cluster_initial": cluster_rng.bit_generator.state,
        "local_random_layout": "color-major [group, temperature, copy, color_site] float64",
        "paired_arms": "Initial/local keys are shared across algorithms within a seed block",
    }
    arrays = {
        f"rng_{role}_initial": generator.get_state().cpu().numpy().copy()
        for role, generator in generators.items()
    }
    shape = (config.groups, config.temperatures, config.copies, instance.model.size)
    spins = (2 * torch.randint(0, 2, shape, device=device, generator=generators["initial"]) - 1).to(
        torch.float64
    )
    arrays["initial_spins"] = spins.cpu().numpy().astype(np.int8)
    if fused_fields is not None:
        # Compile both color shapes inside preparation without consuming randomness.
        for color in range(len(colors)):
            fused_fields(spins, color)
    betas = torch.tensor(
        np.geomspace(config.beta_min, config.beta_max, config.temperatures),
        dtype=spins.dtype,
        device=device,
    )
    anneal = np.geomspace(
        config.beta_min, config.beta_max, config.anneal_cycle_sweeps or config.sweeps
    )
    if len(anneal) == 1:
        anneal[0] = config.beta_max
    population = None
    if population_mode:
        from .population import PopulationEvidence, stepped_schedule

        anneal = stepped_schedule(
            np.geomspace(config.beta_min, config.beta_max, config.temperatures), len(anneal)
        )
        population = PopulationEvidence(
            config.groups,
            config.temperatures * config.copies,
            arrays["rng_initial_initial"].size,
            resampling_scheme=config.population_resampling,
        )
        rng_audit["population_protocol"] = (
            "Independent Rademacher restart each cycle; first temperature uses local warmup; "
            "PA alone resamples on cooling boundaries"
        )
        rng_audit["population_resampling"] = (
            "Dedicated generator, torch.multinomial with replacement"
            if config.algorithm == "metropolis-pa"
            else "none"
        )
        if config.population_resampling == "systematic":
            rng_audit["population_resampling"] = (
                "Dedicated generator, systematic CDF with one FP64 offset per group; "
                "fixed particle order and strict upper-bound search"
            )

    fused_updates = None
    if config.execution == "lattice-v5":
        from .triton_updates import CubicUpdates

        temperature_ladder = config.algorithm.startswith("pbit-pt")
        fused_updates = CubicUpdates(
            sparse,
            colors,
            betas if temperature_ladder else anneal,
            temperatures=config.temperatures,
            copies=config.copies,
            metropolis=config.algorithm.startswith("metropolis"),
            temperature_ladder=temperature_ladder,
        )
        fused_updates.warmup(spins)

    neighbors = [[] for _ in range(instance.model.size)]
    for a, b in zip(
        instance.model.edge_left.tolist(), instance.model.edge_right.tolist(), strict=True
    ):
        neighbors[a].append(b)
        neighbors[b].append(a)
    best_energy = np.full(config.groups, np.inf)
    best_spins = np.empty((config.groups, instance.model.size), dtype=np.int8)
    improvements, improvement_spins, trace = [], [], []
    improvement_seconds = []
    sampling_start = None

    def observe(energies, sweep, phase):
        energy = energies.detach().cpu().numpy().reshape(config.groups, -1)
        slots = energy.argmin(axis=1)
        minima = energy[np.arange(config.groups), slots]
        improved = np.flatnonzero(minima < best_energy)
        if improved.size:
            if lattice:
                states = (
                    spins.reshape(config.groups, -1, instance.model.size)[
                        torch.as_tensor(improved, device=device),
                        torch.as_tensor(slots[improved], device=device),
                    ]
                    .cpu()
                    .numpy()
                )
            else:
                states = (
                    spins.detach().cpu().numpy().reshape(config.groups, -1, instance.model.size)
                )
            observed_seconds = 0.0 if sampling_start is None else perf_counter() - sampling_start
            for offset, group in enumerate(improved):
                best_energy[group] = minima[group]
                best_spins[group] = states[offset] if lattice else states[group, slots[group]]
                improvements.append([sweep, phase, group, slots[group], int(minima[group])])
                improvement_spins.append(best_spins[group].copy())
                improvement_seconds.append(observed_seconds)
        return minima

    with _phase("energy"):
        energies = energy(spins)
    with _phase("observe"):
        minima = observe(energies, 0, 0)
    trace.append(np.stack((minima.copy(), best_energy.copy())))
    accepted = np.zeros((config.groups, config.temperatures - 1, config.copies), dtype=np.int64)
    attempts = np.zeros_like(accepted)
    cluster_events = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = perf_counter() - started
    sampling_start = perf_counter()
    completed, exchange_round = 0, 0
    status = "completed"
    stop_reason = "sweep_horizon"
    failure = None
    host = None
    try:
        for sweep in range(1, config.sweeps + 1):
            now = perf_counter()
            if now - sampling_start >= config.max_seconds:
                status = "timeout"
                stop_reason = "sampling_window"
                break
            if sampling_deadline is not None and now >= sampling_deadline:
                status = "timeout"
                stop_reason = "call_deadline_reserve"
                break
            beta = (
                betas[None, :, None, None]
                if config.algorithm.startswith("pbit-pt")
                else float(anneal[(sweep - 1) % len(anneal)])
            )
            if population is not None:
                position = (sweep - 1) % len(anneal)
                if position == 0 and sweep > 1:
                    with _phase("population_restart"):
                        before_rng = generators["initial"].get_state().clone()
                        spins.copy_(
                            (
                                2
                                * torch.randint(
                                    0, 2, shape, device=device, generator=generators["initial"]
                                )
                                - 1
                            ).to(spins.dtype)
                        )
                        energies = energy(spins)
                        population.record_restart(sweep, before_rng, energies)
                        minima = observe(energies, sweep, 3)
                elif (
                    config.algorithm == "metropolis-pa"
                    and position > 0
                    and beta > float(anneal[position - 1])
                ):
                    with _phase("population_resample"):
                        population.resample(
                            sweep,
                            float(anneal[position - 1]),
                            beta,
                            spins,
                            energies,
                            generators["resample"],
                        )
            with _phase("local_updates"):
                colored_sweep(
                    spins,
                    sparse,
                    colors,
                    beta,
                    generators["local"],
                    metropolis=config.algorithm.startswith("metropolis"),
                    cached=cached,
                    fused_fields=fused_fields,
                    fused_updates=fused_updates,
                    beta_index=(
                        0 if config.algorithm.startswith("pbit-pt") else (sweep - 1) % len(anneal)
                    ),
                )
            with _phase("energy"):
                energies = energy(spins)
            with _phase("observe"):
                minima = observe(energies, sweep, 1)
            if config.algorithm == "pbit-pt-icm" and sweep % config.icm_interval == 0:
                with _phase("icm_d2h"):
                    host = spins.cpu().numpy().astype(np.int8)
                with _phase("icm_host"):
                    for temperature, beta_value in enumerate(betas.cpu().tolist()):
                        if beta_value < config.icm_beta_min:
                            continue
                        for group in range(config.groups):
                            pair = host[group, temperature]
                            if lattice:
                                size, flipped, full = icm_pair_lattice(
                                    pair[0], pair[1], instance.length, cluster_rng
                                )
                            else:
                                size, flipped, full = icm_pair(
                                    pair[0], pair[1], neighbors, cluster_rng
                                )
                            cluster_events.append([sweep, group, temperature, size, flipped, full])
                with _phase("icm_h2d"):
                    spins.copy_(torch.as_tensor(host, dtype=spins.dtype, device=device))
                host = None
                with _phase("energy"):
                    energies = energy(spins)
                with _phase("observe"):
                    minima = observe(energies, sweep, 2)
            if config.algorithm.startswith("pbit-pt") and sweep % config.exchange_interval == 0:
                with _phase("exchange"):
                    edges, mask = exchange_replicas(
                        spins, energies, betas, exchange_round % 2, generators["exchange"]
                    )
                with _phase("exchange_statistics"):
                    accepted[:, edges.cpu().numpy()] += mask.cpu().numpy()
                    attempts[:, edges.cpu().numpy()] += 1
                exchange_round += 1
            trace.append(np.stack((minima.copy(), best_energy.copy())))
            completed = sweep
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        failure = {"error": repr(error), "traceback": traceback.format_exc()}
        # Keep the partially updated physical state and any uncommitted CPU ICM work.
        if host is not None:
            arrays["interrupted_icm_working_spins"] = host
        energies = energy(spins)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_end = perf_counter()
    sampling_seconds = sampling_end - sampling_start
    arrays.update(
        initial_spins=arrays["initial_spins"],
        final_spins=spins.cpu().numpy().astype(np.int8),
        final_energy=energies.cpu().numpy(),
        best_spins=best_spins,
        best_energy=best_energy.astype(np.int64),
        trace=np.asarray(trace, dtype=np.int64),
        improvements=np.asarray(improvements, dtype=np.int64).reshape(-1, 5),
        improvement_spins=np.asarray(improvement_spins, dtype=np.int8),
        exchange_accepted=accepted,
        exchange_attempts=attempts,
        cluster_events=np.asarray(cluster_events, dtype=np.int64).reshape(-1, 6),
        betas=betas.cpu().numpy(),
        anneal_betas=anneal,
    )
    if population is not None:
        arrays.update(population.arrays())
    for role, generator in generators.items():
        arrays[f"rng_{role}_final"] = generator.get_state().cpu().numpy()
    rng_audit["cluster_final"] = cluster_rng.bit_generator.state
    summary = {
        "status": status,
        "solver": asdict(config),
        "completed_sweeps": completed,
        "failure": failure,
        "improvement_seconds": improvement_seconds,
        "spin_update_count_is_lower_bound": failure is not None,
        "spin_update_attempts": completed * math.prod(shape),
        "independent_units": config.groups,
        "unit_definition": "Complete paired ladders for PT+ICM; equally sized groups for controls",
        "preparation_seconds": preparation_seconds,
        "sampling_seconds": sampling_seconds,
        "sampling_started_perf_counter": sampling_start,
        "sampling_finished_perf_counter": sampling_end,
        "state_export_seconds": perf_counter() - sampling_end,
        "sampling_stop_reason": stop_reason if failure is None else status,
        "sampling_deadline": sampling_deadline,
        "sampling_deadline_overrun_seconds": (
            max(0.0, sampling_end - sampling_deadline) if sampling_deadline is not None else None
        ),
        "wall_limit_overrun_seconds": max(0.0, sampling_seconds - config.max_seconds),
        "sampling_includes": "updates, ICM transfers/BFS, PT, observations, synchronization",
        "observation_phases": {"0": "initial", "1": "complete local sweep", "2": "ICM"},
        "trace_axes": ["sweep", "current_min_or_best", "group"],
        "improvement_columns": ["sweep", "phase", "group", "slot", "energy"],
        "cluster_columns": [
            "sweep",
            "group",
            "temperature",
            "size",
            "global_flip",
            "full_disagreement",
        ],
    }
    if population is not None:
        summary["unit_definition"] = (
            "Independent population groups; particles and within-call restarts "
            "are not extra independent trials"
        )
        summary["population_particles_per_group"] = config.temperatures * config.copies
        summary["population_restarts"] = len(population.restart_sweeps)
        summary["population_resampling_events"] = len(population.resample_sweeps)
        summary["population_initialization_spin_draws"] = (
            len(population.restart_sweeps) + 1
        ) * math.prod(shape)
        summary["population_resampling_spin_copies"] = len(population.resample_sweeps) * math.prod(
            shape
        )
        if config.population_resampling == "systematic":
            summary["population_resampling_scheme"] = "systematic"
            summary["population_resampling_offset_values"] = (
                len(population.resample_sweeps) * config.groups
            )
        summary["sampling_includes"] += (
            ", independent restarts, population resampling and diagnostics"
        )
        summary["observation_phases"]["3"] = "independent population restart"
    return arrays, summary, rng_audit
