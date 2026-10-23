"""Importable configuration runner for tsp experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from combinatorial_solvers.core.cache import get_or_build_ising_model
from combinatorial_solvers.core.results import write_anneal_experiment
from combinatorial_solvers.core.sampler import run_colored_gibbs
from combinatorial_solvers.tsp import (
    FixedStartOneHotQUBO,
    get_or_build_multistart_population,
    load_tsplib,
    perturbed_tour_population,
    run_batched_first_improvement_two_opt,
    run_permutation_mcmc,
    write_permutation_experiment,
)
from combinatorial_solvers.tsp.configuration import (
    _file_sha256,
    _permutation_beta_schedule,
    _target_length,
)
from combinatorial_solvers.tsp.experiment_artifacts import (
    _baseline_payload,
    _write_one_hot_candidates,
    _write_one_hot_first_hits,
    _write_tuning,
)
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    dump_yaml,
    experiment_output,
    json_safe,
    linear_schedule,
    require_mapping,
    resolve_device,
    resolve_float_dtype,
    torch_cuda_memory,
    tune_cuda_replicas,
    validate_cuda_resource_snapshot,
    validate_experiment_output,
)


def _strict_one_hot_success(
    encoder: FixedStartOneHotQUBO,
    spins: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Vectorized certificate mask; invalid matrices are never projected or repaired."""

    return _make_one_hot_success(encoder, target_length, spins.device)(spins)


def _make_one_hot_success(
    encoder: FixedStartOneHotQUBO,
    target_length: int,
    device: torch.device,
):
    """Prebuild device tensors used by the sweep-boundary certificate kernel."""

    certify = _make_one_hot_certificate_kernel(encoder, device)

    def success(spins: torch.Tensor) -> torch.Tensor:
        feasible, _, lengths = certify(spins)
        return feasible & (lengths <= int(target_length))

    return success


def _make_one_hot_certificate_kernel(
    encoder: FixedStartOneHotQUBO,
    device: torch.device,
):
    """Return strict feasibility, decoded tours, and exact int64 lengths."""

    free_cities = torch.tensor(encoder.free_cities, dtype=torch.int64, device=device)
    distance = torch.as_tensor(
        encoder.instance.distance.copy(),
        dtype=torch.int64,
        device=device,
    )

    def certify(spins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bits = spins.to(torch.int64).add(1).div(2).reshape(-1, encoder.order, encoder.order)
        row_valid = torch.all(bits.sum(dim=2) == 1, dim=1)
        column_valid = torch.all(bits.sum(dim=1) == 1, dim=1)
        feasible = row_valid & column_valid
        free_indices = torch.argmax(bits, dim=1)
        ordered = free_cities[free_indices]
        fixed = torch.full(
            (spins.shape[0], 1),
            encoder.fixed_city,
            dtype=torch.int64,
            device=device,
        )
        tours = torch.cat((fixed, ordered), dim=1)
        successors = torch.roll(tours, shifts=-1, dims=1)
        lengths = distance[tours, successors].sum(dim=1)
        return feasible, tours, lengths

    return certify


def _run_one_hot_batch(
    model,
    colors,
    schedule: tuple[float, ...],
    solver: Mapping[str, Any],
    replicas: int,
    seed: int,
    dimension: int,
    target_length: int,
    certify,
):
    batch_tours = torch.full(
        (replicas, dimension),
        -1,
        dtype=torch.int64,
        device=model.coupling.device,
    )
    batch_lengths = torch.full(
        (replicas,),
        -1,
        dtype=torch.int64,
        device=model.coupling.device,
    )
    latest_tours: torch.Tensor | None = None
    latest_lengths: torch.Tensor | None = None

    def success(spins: torch.Tensor) -> torch.Tensor:
        nonlocal latest_tours, latest_lengths
        feasible, latest_tours, latest_lengths = certify(spins)
        return feasible & (latest_lengths <= int(target_length))

    def record_first_hit(
        spins: torch.Tensor,
        energies: torch.Tensor,
        new_hits: torch.Tensor,
    ) -> None:
        del spins, energies
        if latest_tours is None or latest_lengths is None:
            raise RuntimeError("One-hot certificates were not decoded before recording")
        batch_tours[new_hits] = latest_tours[new_hits]
        batch_lengths[new_hits] = latest_lengths[new_hits]

    batch = run_colored_gibbs(
        model,
        colors,
        schedule,
        int(solver.get("sweeps_per_beta", 1)),
        replicas,
        seed,
        success=success,
        record_every=int(solver.get("record_every", 1)),
        randomize_color_order=bool(solver.get("randomize_color_order", False)),
        on_first_hit=record_first_hit,
    )
    return batch, batch_tours.cpu(), batch_lengths.cpu()


def _one_hot_replica_count(
    configured: int | str,
    model,
    colors,
    encoder: FixedStartOneHotQUBO,
    target_length: int,
    schedule: tuple[float, ...],
    solver: Mapping[str, Any],
    device: torch.device,
):
    if not isinstance(configured, str):
        count = int(configured)
        if count <= 0:
            raise ValueError("replicas must be positive")
        return count, None
    if configured.lower() != "auto":
        raise ValueError("replicas must be a positive integer or 'auto'")
    tuning = dict(solver.get("autotune", {}))
    probe_steps = int(tuning.get("probe_beta_steps", min(4, len(schedule))))
    probe_schedule = tuple(torch.linspace(schedule[0], schedule[-1], probe_steps).tolist())
    probe_sweeps = int(tuning.get("probe_sweeps_per_beta", 1))

    success = _make_one_hot_success(encoder, target_length, device)

    def runner(replicas: int) -> tuple[float, int, int]:
        result = run_colored_gibbs(
            model,
            colors,
            probe_schedule,
            probe_sweeps,
            replicas,
            int(solver.get("seed", 0)) - 1,
            success=success,
            record_every=max(1, probe_steps * probe_sweeps),
        )
        return (
            result.kernel_seconds,
            int(result.peak_allocated_bytes or 0),
            int(result.peak_reserved_bytes or 0),
        )

    result = tune_cuda_replicas(runner, tuning, device)
    return result.selected_replicas, result


def _permutation_replica_count(
    configured: int | str,
    distance: torch.Tensor,
    target_length: int,
    schedule: tuple[float, ...],
    solver: Mapping[str, Any],
    device: torch.device,
    probability_dtype: torch.dtype,
):
    if not isinstance(configured, str):
        count = int(configured)
        if count <= 0:
            raise ValueError("replicas must be positive")
        return count, None
    if configured.lower() != "auto":
        raise ValueError("replicas must be a positive integer or 'auto'")
    tuning = dict(solver.get("autotune", {}))
    probe_steps = int(tuning.get("probe_beta_steps", min(4, len(schedule))))
    probe_schedule = tuple(torch.linspace(schedule[0], schedule[-1], probe_steps).tolist())
    probe_steps_per_beta = int(tuning.get("probe_steps_per_beta", 4))

    def runner(replicas: int) -> tuple[float, int, int]:
        result = run_permutation_mcmc(
            distance,
            probe_schedule,
            probe_steps_per_beta,
            replicas,
            int(solver.get("seed", 0)) - 1,
            proposal=str(solver.get("proposal", "two_opt")),
            acceptance=str(solver.get("acceptance", "metropolis")),
            target_length=target_length,
            record_every=max(1, probe_steps * probe_steps_per_beta),
            probability_dtype=probability_dtype,
            tour_update_mode=str(solver.get("tour_update_mode", "masked_full")),
            swap_interval=int(solver.get("swap_interval", 16)),
        )
        return (
            result.kernel_seconds,
            int(result.peak_allocated_bytes or 0),
            int(result.peak_reserved_bytes or 0),
        )

    result = tune_cuda_replicas(runner, tuning, device)
    return result.selected_replicas, result


def _run_one_hot(
    config: Mapping[str, Any],
    problem: Mapping[str, Any],
    solver: Mapping[str, Any],
    instance,
    instance_path: Path,
    target_length: int,
    output: Path,
    *,
    overwrite: bool,
    repository: Path | None,
) -> Path:
    dtype = resolve_float_dtype(str(solver.get("dtype", "float32")))
    device = resolve_device(str(solver.get("device", "auto")))
    resources = require_mapping(config, "resources") if "resources" in config else {}
    preflight_cuda_memory = None
    if device.type == "cuda":
        preflight_cuda_memory = torch_cuda_memory(device)
        validate_cuda_resource_snapshot(preflight_cuda_memory, resources, check_peak=False)
    encoder = FixedStartOneHotQUBO(
        instance,
        constraint_weight=float(problem.get("constraint_weight", 1.0)),
        objective_weight=float(problem.get("objective_weight", 0.5)),
        fixed_city=int(problem.get("fixed_city", 0)),
        normalize_distances=bool(problem.get("normalize_distances", True)),
    )
    if not encoder.satisfies_lucas_bound:
        raise ValueError("The configured one-hot weights violate the declared Lucas bound")
    specification = {
        "instance_sha256": _file_sha256(instance_path),
        "instance": instance.name,
        "dimension": instance.dimension,
        "constraint_weight": encoder.constraint_weight,
        "objective_weight": encoder.objective_weight,
        "fixed_city": encoder.fixed_city,
        "normalize_distances": encoder.normalize_distances,
        "dtype": str(dtype),
    }
    cache = dict(config.get("cache", {}))
    model, cache_path, cache_hit = get_or_build_ising_model(
        Path(cache.get("directory", "cache")),
        "tsp-one-hot",
        specification,
        lambda: encoder.to_qubo(dtype=dtype).to_ising(),
        dtype=dtype,
    )
    model = model.to(device=device, dtype=dtype)
    colors = encoder.color_blocks()
    schedule = linear_schedule(solver)
    replicas, tuning = _one_hot_replica_count(
        solver.get("replicas", 1024),
        model,
        colors,
        encoder,
        target_length,
        schedule,
        solver,
        device,
    )
    batch_count = int(solver.get("batches", 1))
    if batch_count <= 0:
        raise ValueError("batches must be positive")

    certify = _make_one_hot_certificate_kernel(encoder, device)
    batches = []
    first_hit_tours: list[torch.Tensor] = []
    first_hit_lengths: list[torch.Tensor] = []
    for batch_index in range(batch_count):
        batch, batch_tours, batch_lengths = _run_one_hot_batch(
            model,
            colors,
            schedule,
            solver,
            replicas,
            int(solver.get("seed", 0)) + batch_index,
            instance.dimension,
            target_length,
            certify,
        )
        batches.append(batch)
        first_hit_tours.append(batch_tours.cpu())
        first_hit_lengths.append(batch_lengths.cpu())
        if device.type == "cuda":
            validate_cuda_resource_snapshot(torch_cuda_memory(device), resources, check_peak=True)
    resolved = {
        **json_safe(config),
        "resolved": {
            "solver_kind": "one_hot_ising",
            "instance": instance.name,
            "instance_sha256": specification["instance_sha256"],
            "edge_weight_type": instance.edge_weight_type,
            "optimum": instance.optimum,
            "target_length": target_length,
            "replicas": replicas,
            "batches": batch_count,
            "total_chains": replicas * batch_count,
            "device": str(device),
            "dtype": str(dtype),
            "cache_hit": cache_hit,
        },
    }
    write_anneal_experiment(
        output,
        config=resolved,
        batches=batches,
        repository=repository,
        extra_environment={
            "resource_guard": json_safe(resources),
            "preflight_cuda_memory": json_safe(preflight_cuda_memory),
        },
        overwrite=overwrite,
    )
    _write_one_hot_candidates(output, encoder, batches, target_length)
    _write_one_hot_first_hits(
        output,
        batches,
        first_hit_tours,
        first_hit_lengths,
        target_length,
    )
    replay = json_safe(config)
    replay["experiment"]["output_dir"] = output.as_posix()
    replay["problem"]["target_length"] = target_length
    replay["solver"]["replicas"] = replicas
    dump_yaml(output / "replay_config.yaml", replay)
    dump_json(
        output / "model.json",
        {
            "cache_file": cache_path.as_posix(),
            "cache_hit": cache_hit,
            "variables": model.size,
            "colors": len(colors),
            "dense_coupling_bytes": model.coupling.numel() * model.coupling.element_size(),
            "metadata": model.metadata,
        },
    )
    _write_tuning(output, tuning)
    return output


def _run_permutation(
    config: Mapping[str, Any],
    problem: Mapping[str, Any],
    solver: Mapping[str, Any],
    instance,
    instance_path: Path,
    target_length: int,
    output: Path,
    baseline_population,
    *,
    overwrite: bool,
    repository: Path | None,
) -> Path:
    device = resolve_device(str(solver.get("device", "auto")))
    probability_dtype = resolve_float_dtype(str(solver.get("probability_dtype", "float32")))
    resources = require_mapping(config, "resources") if "resources" in config else {}
    preflight_cuda_memory = None
    if device.type == "cuda":
        preflight_cuda_memory = torch_cuda_memory(device)
        validate_cuda_resource_snapshot(preflight_cuda_memory, resources, check_peak=False)
    distance = torch.as_tensor(instance.distance.copy(), dtype=torch.int64, device=device)
    proposal = str(solver.get("proposal", "two_opt"))
    first_descent = proposal == "two_opt_first_descent"
    if first_descent:
        schedule = (1.0,)
        schedule_metadata: dict[str, float | str] = {"beta_scale_mode": "not_applicable"}
    else:
        schedule, schedule_metadata = _permutation_beta_schedule(solver, instance.distance)
    if first_descent and str(solver.get("acceptance", "greedy")) != "greedy":
        raise ValueError("two_opt_first_descent requires greedy acceptance")
    initialization = str(solver.get("initialization", "random"))
    perturb_swaps = int(solver.get("warm_start_perturb_swaps", 0))
    perturbation = str(solver.get("warm_start_perturbation", "swap"))
    top_k_value = solver.get("warm_start_top_k")
    warm_start_top_k = None if top_k_value is None else int(top_k_value)
    if initialization not in {"random", "multistart_two_opt"}:
        raise ValueError("solver.initialization must be 'random' or 'multistart_two_opt'")
    if perturb_swaps < 0:
        raise ValueError("warm_start_perturb_swaps must be nonnegative")
    if initialization == "random" and perturb_swaps != 0:
        raise ValueError("warm_start_perturb_swaps requires multistart_two_opt initialization")
    if initialization == "random" and "warm_start_perturbation" in solver:
        raise ValueError("warm_start_perturbation requires multistart_two_opt initialization")
    if perturbation not in {"swap", "double_bridge"}:
        raise ValueError("warm_start_perturbation must be 'swap' or 'double_bridge'")
    if initialization == "random" and warm_start_top_k is not None:
        raise ValueError("warm_start_top_k requires multistart_two_opt initialization")
    if warm_start_top_k is not None and warm_start_top_k <= 0:
        raise ValueError("warm_start_top_k must be positive when provided")
    warm_seed_tours: torch.Tensor | None = None
    if initialization == "multistart_two_opt":
        warm_results = baseline_population
        if warm_start_top_k is not None:
            if warm_start_top_k > len(warm_results):
                raise ValueError("warm_start_top_k exceeds the unique warm-seed population")
            warm_results = warm_results[:warm_start_top_k]
        warm_seed_tours = torch.tensor(
            [candidate.tour for candidate in warm_results],
            dtype=torch.long,
        )
    replicas, tuning = _permutation_replica_count(
        solver.get("replicas", 1024),
        distance,
        target_length,
        schedule,
        solver,
        device,
        probability_dtype,
    )
    batch_count = int(solver.get("batches", 1))
    if batch_count <= 0:
        raise ValueError("batches must be positive")
    batches = []
    initial_tour_sha256: list[str] = []
    for batch_index in range(batch_count):
        batch_seed = int(solver.get("seed", 0)) + batch_index
        initial_tours = None
        if warm_seed_tours is not None:
            initial_tours = perturbed_tour_population(
                warm_seed_tours,
                replicas,
                batch_seed,
                perturb_swaps=perturb_swaps,
                perturbation=perturbation,
                device="cpu",
            )
            initial_tour_sha256.append(
                hashlib.sha256(initial_tours.numpy().tobytes(order="C")).hexdigest()
            )
        if first_descent:
            if initial_tours is None:
                raise ValueError("two_opt_first_descent requires an explicit warm initialization")
            batches.append(
                run_batched_first_improvement_two_opt(
                    distance,
                    initial_tours,
                    int(solver.get("max_passes", solver.get("steps_per_beta", 1))),
                    batch_seed,
                    target_length=target_length,
                    record_every=int(solver.get("record_every", 1)),
                    initialization_label=initialization,
                )
            )
        else:
            batches.append(
                run_permutation_mcmc(
                    distance,
                    schedule,
                    int(solver.get("steps_per_beta", 1)),
                    replicas,
                    batch_seed,
                    proposal=proposal,
                    acceptance=str(solver.get("acceptance", "metropolis")),
                    initial_tours=initial_tours,
                    target_length=target_length,
                    record_every=int(solver.get("record_every", 1)),
                    probability_dtype=probability_dtype,
                    tour_update_mode=str(solver.get("tour_update_mode", "masked_full")),
                    swap_interval=int(solver.get("swap_interval", 16)),
                    initialization_label=initialization,
                )
            )
        if device.type == "cuda":
            validate_cuda_resource_snapshot(torch_cuda_memory(device), resources, check_peak=True)
    resolved = {
        **json_safe(config),
        "resolved": {
            "solver_kind": (
                "permutation_first_improvement_descent"
                if first_descent
                else (
                    "permutation_greedy_search"
                    if batches[0].acceptance == "greedy"
                    else "permutation_mcmc"
                )
            ),
            "instance": instance.name,
            "instance_sha256": _file_sha256(instance_path),
            "edge_weight_type": instance.edge_weight_type,
            "optimum": instance.optimum,
            "target_length": target_length,
            "replicas": replicas,
            "batches": batch_count,
            "total_chains": replicas * batch_count,
            "device": str(device),
            "distance_dtype": str(distance.dtype),
            "acceptance_probability_dtype": batches[0].probability_dtype,
            "tour_update_mode": batches[0].update_mode,
            "two_opt_pair_count": batches[0].two_opt_pair_count,
            "proposal": batches[0].proposal,
            "swap_interval": batches[0].swap_interval,
            "initialization": initialization,
            "warm_start_seed_count": (
                int(warm_seed_tours.shape[0]) if warm_seed_tours is not None else 0
            ),
            "warm_start_perturb_swaps": perturb_swaps,
            "warm_start_perturbation": perturbation,
            "warm_start_top_k": warm_start_top_k,
            "initial_tour_sha256": initial_tour_sha256,
            **schedule_metadata,
        },
    }
    write_permutation_experiment(
        output,
        config=resolved,
        batches=batches,
        distance=distance,
        repository=repository,
        extra_environment={
            "resource_guard": json_safe(resources),
            "preflight_cuda_memory": json_safe(preflight_cuda_memory),
        },
        overwrite=overwrite,
    )
    replay = json_safe(config)
    replay["experiment"]["output_dir"] = output.as_posix()
    replay["problem"]["target_length"] = target_length
    replay["solver"]["replicas"] = replicas
    dump_yaml(output / "replay_config.yaml", replay)
    _write_tuning(output, tuning)
    return output


def run_tsp_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    problem = require_mapping(config, "problem")
    solver = require_mapping(config, "solver")
    if str(solver.get("kind", "permutation")) == "edge_assignment":
        from combinatorial_solvers.tsp.edge_runner import run_config

        return run_config(
            config, output_override=output_override, overwrite=overwrite, repository=repository
        )
    if str(solver.get("kind", "permutation")) == "eax":
        from combinatorial_solvers.tsp.eax_runner import run_config

        return run_config(
            config, output_override=output_override, overwrite=overwrite, repository=repository
        )
    if str(solver.get("kind", "permutation")) == "lkh":
        from combinatorial_solvers.tsp.lkh_runner import run_config

        return run_config(
            config, output_override=output_override, overwrite=overwrite, repository=repository
        )
    if str(solver.get("kind", "permutation")) == "hybrid_local_benchmark":
        from combinatorial_solvers.tsp.hybrid_benchmark import run_config

        return run_config(
            config, output_override=output_override, overwrite=overwrite, repository=repository
        )
    if str(solver.get("kind", "permutation")) == "hybrid":
        from combinatorial_solvers.tsp.hybrid_runner import run_config

        return run_config(
            config, output_override=output_override, overwrite=overwrite, repository=repository
        )
    output = experiment_output(config, output_override)
    validate_experiment_output(output, overwrite)

    instance_path = Path(problem["instance"])
    optimum_value = problem.get("optimum")
    optimum = None if optimum_value is None else int(optimum_value)
    instance = load_tsplib(instance_path, optimum=optimum)
    target_length = _target_length(problem, optimum)
    instance_sha256 = _file_sha256(instance_path)
    cache = dict(config.get("cache", {}))
    baseline_population, baseline_cache_path, baseline_cache_hit = (
        get_or_build_multistart_population(
            instance,
            instance_sha256,
            Path(cache.get("directory", "cache")) / "tsp-baselines",
        )
    )
    baselines = _baseline_payload(instance, problem, baseline_population)
    baselines["cache"] = {
        "path": baseline_cache_path.as_posix(),
        "hit": baseline_cache_hit,
        "instance_sha256": instance_sha256,
    }
    solver_kind = str(solver.get("kind", "permutation"))
    if solver_kind == "one_hot":
        destination = _run_one_hot(
            config,
            problem,
            solver,
            instance,
            instance_path,
            target_length,
            output,
            overwrite=overwrite,
            repository=repository,
        )
    elif solver_kind == "permutation":
        destination = _run_permutation(
            config,
            problem,
            solver,
            instance,
            instance_path,
            target_length,
            output,
            baseline_population,
            overwrite=overwrite,
            repository=repository,
        )
    else:
        raise ValueError("solver.kind must be 'one_hot', 'permutation', or 'hybrid'")
    dump_json(destination / "baselines.json", baselines)
    return destination
