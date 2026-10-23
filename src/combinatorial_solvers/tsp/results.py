"""Portable N-by-M result bundles for feasible-space TSP experiments."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from combinatorial_solvers.core.metrics import (
    empirical_success_curve,
    fit_censored_weibull_its,
    time_to_solution,
)
from combinatorial_solvers.core.provenance import capture_provenance

from .permutation import PermutationMCMCResult, tour_lengths, validate_batched_tours

PERMUTATION_RESULT_SCHEMA_VERSION = 1
_ITS_TARGETS = (0.90, 0.95, 0.99)


def _prepare_empty_output(directory: Path, overwrite: bool) -> None:
    if directory.exists():
        if not directory.is_dir() or (any(directory.iterdir()) and not overwrite):
            raise FileExistsError(f"Output directory is not empty: {directory}")
    else:
        directory.mkdir(parents=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def _empirical_its(
    first_hits: np.ndarray,
    target_probability: float,
    censor_iteration: int,
) -> int | None:
    required_rank = math.ceil(target_probability * first_hits.size)
    observed = np.sort(first_hits[first_hits >= 0])
    if observed.size < required_rank:
        return None
    estimate = int(observed[required_rank - 1])
    return estimate if estimate <= censor_iteration else None


def _fixed_cutoff_restart_effort(
    success_probability: float,
    target_probability: float,
    censor_iteration: int,
) -> dict[str, int] | None:
    """Return plug-in independent-restart work at one fixed censor horizon."""

    if success_probability <= 0.0:
        return None
    if success_probability >= 1.0:
        runs = 1
    else:
        runs = math.ceil(math.log1p(-target_probability) / math.log1p(-success_probability))
    return {
        "independent_chain_runs": runs,
        "total_chain_iterations": runs * censor_iteration,
    }


def _validate_batches(
    batches: Sequence[PermutationMCMCResult],
    distance: torch.Tensor,
) -> tuple[int, int, int, int, tuple[torch.Tensor, ...]]:
    if not batches:
        raise ValueError("At least one permutation MCMC batch is required")
    first = batches[0]
    if first.target_length is None:
        raise ValueError("All batches must declare a target_length")
    if first.first_hit_iterations.ndim != 1 or first.first_hit_iterations.numel() == 0:
        raise ValueError("first_hit_iterations must be a non-empty vector")
    chains = int(first.first_hit_iterations.numel())
    if first.best_tours.ndim != 2:
        raise ValueError("best_tours must have shape [chains, cities]")
    cities = int(first.best_tours.shape[1])
    if (
        distance.ndim != 2
        or distance.shape != (cities, cities)
        or distance.dtype.is_floating_point
        or distance.dtype == torch.bool
    ):
        raise ValueError("distance must be a square integer matrix matching the tour dimension")
    verification_distance = distance.detach().cpu().to(torch.int64)
    if not torch.equal(verification_distance, verification_distance.mT) or bool(
        torch.any(torch.diagonal(verification_distance) != 0)
    ):
        raise ValueError("distance must be symmetric with a zero diagonal")
    censor_iteration = first.total_iterations
    target_length = first.target_length
    if censor_iteration <= 0:
        raise ValueError("total_iterations must be positive")

    recomputed_batches: list[torch.Tensor] = []
    for batch in batches:
        if batch.total_iterations != censor_iteration:
            raise ValueError("All batches must use the same censor iteration")
        if batch.target_length != target_length:
            raise ValueError("All batches must use the same target length")
        if batch.proposal != first.proposal or batch.acceptance != first.acceptance:
            raise ValueError("All batches must use the same proposal and acceptance rule")
        if batch.probability_dtype != first.probability_dtype:
            raise ValueError("All batches must use the same acceptance-probability dtype")
        if batch.update_mode != first.update_mode:
            raise ValueError("All batches must use the same tour update mode")
        if batch.two_opt_pair_count != first.two_opt_pair_count:
            raise ValueError("All batches must use the same 2-opt pair table")
        if batch.swap_interval != first.swap_interval:
            raise ValueError("All batches must use the same mixed-proposal swap interval")
        if batch.initialization != first.initialization:
            raise ValueError("All batches must use the same initialization policy")
        if batch.device != first.device:
            raise ValueError("All batches must use the same device")
        expected_tour_shape = (chains, cities)
        if batch.final_tours.shape != expected_tour_shape:
            raise ValueError("All final_tours arrays must have the same N-by-city shape")
        if batch.best_tours.shape != expected_tour_shape:
            raise ValueError("All best_tours arrays must have the same N-by-city shape")
        if batch.final_lengths.shape != (chains,) or batch.best_lengths.shape != (chains,):
            raise ValueError("All length arrays must have shape [chains]")
        if batch.first_hit_iterations.shape != (chains,):
            raise ValueError("All first-hit arrays must have shape [chains]")
        if batch.first_hit_tours.shape != expected_tour_shape:
            raise ValueError("All first_hit_tours arrays must have the same N-by-city shape")
        if batch.first_hit_lengths.shape != (chains,):
            raise ValueError("All first_hit_lengths arrays must have shape [chains]")
        if batch.proposed_moves != chains * censor_iteration:
            raise ValueError("proposed_moves does not match chains times total_iterations")
        if not 0 <= batch.accepted_moves <= batch.proposed_moves:
            raise ValueError("accepted_moves is outside its valid range")
        if not math.isfinite(batch.kernel_seconds) or batch.kernel_seconds < 0:
            raise ValueError("kernel_seconds must be finite and nonnegative")
        if not batch.trace:
            raise ValueError("Every batch must contain trace points")
        trace_iterations = [point.iteration for point in batch.trace]
        if trace_iterations[0] != 0 or trace_iterations[-1] != censor_iteration:
            raise ValueError("Every trace must contain iteration zero and the censor endpoint")
        if trace_iterations != sorted(set(trace_iterations)):
            raise ValueError("Trace iterations must be strictly increasing")

        validate_batched_tours(batch.final_tours)
        validate_batched_tours(batch.best_tours)
        hits = batch.first_hit_iterations.detach().cpu().numpy()
        if not np.issubdtype(hits.dtype, np.integer):
            raise TypeError("first_hit_iterations must use an integer dtype")
        if np.any(hits < -1) or np.any(hits > censor_iteration):
            raise ValueError("First hits must use -1 or an observed iteration through censoring")
        best_lengths = batch.best_lengths.detach().cpu().numpy()
        if not np.array_equal(hits >= 0, best_lengths <= target_length):
            raise ValueError("First-hit status and final best lengths are inconsistent")
        hit_mask = batch.first_hit_iterations >= 0
        recomputed_lengths = torch.full((chains,), -1, dtype=torch.int64)
        if bool(torch.any(hit_mask)):
            validate_batched_tours(batch.first_hit_tours[hit_mask])
            if bool(torch.any(batch.first_hit_lengths[hit_mask] > target_length)):
                raise ValueError("A first-hit tour exceeds the target length")
            recomputed_lengths[hit_mask.cpu()] = tour_lengths(
                verification_distance,
                batch.first_hit_tours[hit_mask].detach().cpu(),
            )
            if not torch.equal(
                recomputed_lengths[hit_mask.cpu()],
                batch.first_hit_lengths[hit_mask].detach().cpu().to(torch.int64),
            ):
                raise ValueError("Stored first-hit lengths do not match their tours")
        if bool(torch.any(batch.first_hit_tours[~hit_mask] != -1)) or bool(
            torch.any(batch.first_hit_lengths[~hit_mask] != -1)
        ):
            raise ValueError("Censored first-hit states must use the -1 sentinel")
        recomputed_batches.append(recomputed_lengths)

    return len(batches), chains, cities, target_length, tuple(recomputed_batches)


def _length_summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    numeric = values.detach().cpu().numpy().astype(np.float64, copy=False)
    quantiles = np.quantile(numeric, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_std": float(numeric.std()),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
    }


def write_permutation_experiment(
    output_directory: str | Path,
    *,
    config: Mapping[str, Any],
    batches: Sequence[PermutationMCMCResult],
    distance: torch.Tensor,
    repository: str | Path | None = None,
    extra_environment: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write an auditable portable bundle for M batches of N independent chains."""

    batch_count, chains, cities, target_length, recomputed_batches = _validate_batches(
        batches, distance
    )
    censor_iteration = batches[0].total_iterations
    destination = Path(output_directory)

    config_payload = {
        **dict(config),
        "schema_version": PERMUTATION_RESULT_SCHEMA_VERSION,
        "problem_kind": "tsp",
        "solver_kind": (
            "permutation_first_improvement_descent"
            if batches[0].proposal == "two_opt_first_descent"
            else (
                "permutation_greedy_search"
                if batches[0].acceptance == "greedy"
                else "permutation_mcmc"
            )
        ),
        "batches": batch_count,
        "chains_per_batch": chains,
        "total_chains": batch_count * chains,
        "cities": cities,
        "censor_iteration": censor_iteration,
        "target_length": target_length,
        "proposal": batches[0].proposal,
        "acceptance": batches[0].acceptance,
        "probability_dtype": batches[0].probability_dtype,
        "update_mode": batches[0].update_mode,
        "two_opt_pair_count": batches[0].two_opt_pair_count,
        "swap_interval": batches[0].swap_interval,
        "initialization": batches[0].initialization,
    }
    environment = capture_provenance(repository)
    if extra_environment:
        environment["experiment"] = dict(extra_environment)
    json.dumps(config_payload, allow_nan=False)
    json.dumps(environment, allow_nan=False)
    _prepare_empty_output(destination, overwrite)
    _write_json(destination / "config.json", config_payload)
    _write_json(destination / "environment.json", environment)

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    hit_rows: list[np.ndarray] = []
    first_hit_tour_rows: list[np.ndarray] = []
    first_hit_length_rows: list[np.ndarray] = []
    recomputed_first_hit_length_rows: list[np.ndarray] = []
    best_tour_rows: list[np.ndarray] = []
    best_length_rows: list[np.ndarray] = []
    for batch_index, batch in enumerate(batches):
        hits = batch.first_hit_iterations.detach().cpu().numpy().astype(np.int64, copy=False)
        successes = int(np.count_nonzero(hits >= 0))
        hit_rows.append(hits)
        first_hit_tour_rows.append(
            batch.first_hit_tours.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        first_hit_length_rows.append(
            batch.first_hit_lengths.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        recomputed_first_hit_length_rows.append(
            recomputed_batches[batch_index].numpy().astype(np.int64, copy=False)
        )
        best_tour_rows.append(batch.best_tours.detach().cpu().numpy().astype(np.int64, copy=False))
        best_length_rows.append(batch.best_lengths.detach().cpu().numpy())
        summary_rows.append(
            {
                "batch_index": batch_index,
                "seed": batch.seed,
                "chains": chains,
                "successes": successes,
                "success_rate": successes / chains,
                "batch_succeeded": int(successes > 0),
                "target_length": target_length,
                "total_iterations": censor_iteration,
                "kernel_seconds": batch.kernel_seconds,
                "seconds_per_chain": batch.kernel_seconds / chains,
                "accepted_moves": batch.accepted_moves,
                "proposed_moves": batch.proposed_moves,
                "acceptance_rate": batch.acceptance_rate,
                **_length_summary(batch.final_lengths, "final_length"),
                **_length_summary(batch.best_lengths, "best_length"),
                "peak_allocated_bytes": batch.peak_allocated_bytes,
                "peak_reserved_bytes": batch.peak_reserved_bytes,
                "proposal": batch.proposal,
                "acceptance": batch.acceptance,
                "probability_dtype": batch.probability_dtype,
                "update_mode": batch.update_mode,
                "two_opt_pair_count": batch.two_opt_pair_count,
                "swap_interval": batch.swap_interval,
                "initialization": batch.initialization,
                "device": batch.device,
            }
        )
        for point in batch.trace:
            trace_rows.append({"batch_index": batch_index, **asdict(point)})

    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (destination / "trace_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    hit_matrix = np.stack(hit_rows)
    batch_index = np.broadcast_to(
        np.arange(batch_count, dtype=np.int64)[:, None], hit_matrix.shape
    ).copy()
    chain_index = np.broadcast_to(
        np.arange(chains, dtype=np.int64)[None, :], hit_matrix.shape
    ).copy()
    np.savez_compressed(
        destination / "first_hits.npz",
        first_hit_iterations=hit_matrix,
        batch_index=batch_index,
        chain_index=chain_index,
        censored=hit_matrix < 0,
        censor_iteration=np.asarray(censor_iteration, dtype=np.int64),
    )
    first_hit_tour_matrix = np.stack(first_hit_tour_rows)
    first_hit_length_matrix = np.stack(first_hit_length_rows)
    recomputed_first_hit_length_matrix = np.stack(recomputed_first_hit_length_rows)
    hit_certificate = hit_matrix >= 0
    integer_length_certificate = hit_certificate & (
        first_hit_length_matrix == recomputed_first_hit_length_matrix
    )
    target_certificate = hit_certificate & (recomputed_first_hit_length_matrix <= target_length)
    certified = hit_certificate & integer_length_certificate & target_certificate
    np.savez_compressed(
        destination / "first_hit_tours.npz",
        first_hit_tours=first_hit_tour_matrix,
        first_hit_lengths=first_hit_length_matrix,
        recomputed_first_hit_lengths=recomputed_first_hit_length_matrix,
        first_hit_iterations=hit_matrix,
        batch_index=batch_index,
        chain_index=chain_index,
        target_length=np.asarray(target_length, dtype=np.int64),
        permutation_certificate=hit_certificate,
        integer_length_certificate=integer_length_certificate,
        target_certificate=target_certificate,
        certified=certified,
    )
    np.savez_compressed(
        destination / "best_tours.npz",
        best_tours=np.stack(best_tour_rows),
        best_lengths=np.stack(best_length_rows),
        batch_index=batch_index,
        chain_index=chain_index,
    )

    combined_hits = hit_matrix.reshape(-1)
    checkpoints = np.unique(
        np.asarray(
            [point.iteration for batch in batches for point in batch.trace],
            dtype=np.int64,
        )
    )
    curve = empirical_success_curve(
        combined_hits,
        checkpoints,
        censor_iteration=censor_iteration,
    )
    curve_rows = [asdict(point) for point in curve]
    with (destination / "success_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    successful_batches = sum(bool(np.any(hits >= 0)) for hits in hit_rows)
    observed_chain_probability = float(np.count_nonzero(combined_hits >= 0) / combined_hits.size)
    observed_batch_probability = successful_batches / batch_count
    mean_batch_kernel_seconds = float(
        np.mean([batch.kernel_seconds for batch in batches], dtype=np.float64)
    )
    restart_tts: dict[str, float | None] = {}
    for target in _ITS_TARGETS:
        estimate = time_to_solution(
            mean_batch_kernel_seconds,
            observed_batch_probability,
            target,
        )
        restart_tts[f"tts{int(target * 100)}_seconds_at_censor"] = (
            estimate if math.isfinite(estimate) else None
        )
    weibull = fit_censored_weibull_its(
        combined_hits,
        censor_iteration,
        _ITS_TARGETS,
    )
    fixed_cutoff_restart_work = {
        f"target_{int(target * 100)}": _fixed_cutoff_restart_effort(
            observed_chain_probability,
            target,
            censor_iteration,
        )
        for target in _ITS_TARGETS
    }
    _write_json(
        destination / "its.json",
        {
            "censor_iteration": censor_iteration,
            "target_length": target_length,
            "batches": batch_count,
            "chains_per_batch": chains,
            "chains": int(combined_hits.size),
            "observed_success_probability": observed_chain_probability,
            "observed_batch_success_probability": observed_batch_probability,
            "empirical": {
                f"its{int(target * 100)}": _empirical_its(
                    combined_hits,
                    target,
                    censor_iteration,
                )
                for target in _ITS_TARGETS
            },
            "empirical_unreached_means": f"> {censor_iteration}",
            "fixed_cutoff_restart_work": fixed_cutoff_restart_work,
            "fixed_cutoff_restart_work_note": (
                "Plug-in estimate for independent full-horizon chain restarts: "
                "ceil(log(1-target)/log(1-observed_success_probability)) times the "
                "censor horizon. Null means no successful chain was observed; sampling "
                "uncertainty is represented separately by the endpoint Wilson interval."
            ),
            "mean_batch_kernel_seconds": mean_batch_kernel_seconds,
            "parallel_batch_restart_tts": restart_tts,
            "parallel_batch_restart_tts_note": (
                "Uses measured whole-batch success probability and mean batch kernel time. "
                "Null means no successful batch was observed."
            ),
            "weibull_exploratory": asdict(weibull),
        },
    )
    return destination
