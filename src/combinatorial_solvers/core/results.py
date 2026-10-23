"""Portable experiment bundles for multiple independent chain batches."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    bootstrap_restart_cutoff,
    clopper_pearson_interval,
    discrete_survival_curve,
    empirical_success_curve,
    fit_censored_weibull_its,
    grouped_success_curve,
    time_to_solution,
)
from .provenance import capture_provenance
from .rng import audit_stream_partition
from .sampler import AnnealResult

RESULT_SCHEMA_VERSION = 5


def _prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empirical_its(first_hits: np.ndarray, target: float, censor_iteration: int) -> int | None:
    required_rank = math.ceil(target * first_hits.size)
    observed = np.sort(first_hits[first_hits >= 0])
    if observed.size < required_rank:
        return None
    estimate = int(observed[required_rank - 1])
    return estimate if estimate <= censor_iteration else None


def _analysis_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    study = config.get("study")
    if isinstance(study, Mapping) and isinstance(study.get("analysis"), Mapping):
        result.update(dict(study["analysis"]))
    direct = config.get("analysis")
    if isinstance(direct, Mapping):
        result.update(dict(direct))
    return result


def write_anneal_experiment(
    output_directory: str | Path,
    *,
    config: Mapping[str, Any],
    batches: Sequence[AnnealResult],
    repository: str | Path | None = None,
    extra_environment: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write exact first hits and compact trace summaries for N-by-M chain runs."""

    if not batches:
        raise ValueError("At least one annealing batch is required")
    analysis = _analysis_settings(config)
    cutoff_config = analysis.get("cutoff_selection")
    if cutoff_config is not None:
        if not isinstance(cutoff_config, Mapping):
            raise ValueError("analysis.cutoff_selection must be a mapping")
        if str(cutoff_config.get("dataset_role")) != "tuning":
            raise ValueError("Cutoff selection is permitted only for dataset_role='tuning'")
        raw_cutoffs = cutoff_config.get("candidate_cutoffs")
        if not isinstance(raw_cutoffs, Sequence) or isinstance(raw_cutoffs, (str, bytes)):
            raise ValueError("cutoff_selection.candidate_cutoffs must be a sequence")
    censor_iterations = {batch.total_sweeps for batch in batches}
    if len(censor_iterations) != 1:
        raise ValueError("All batches must use the same censor iteration")
    censor_iteration = censor_iterations.pop()
    destination = Path(output_directory)
    _prepare_output(destination, overwrite)

    _write_json(
        destination / "config.json",
        {"schema_version": RESULT_SCHEMA_VERSION, **dict(config)},
    )
    environment = capture_provenance(repository)
    if extra_environment:
        environment["experiment"] = dict(extra_environment)
    _write_json(destination / "environment.json", environment)

    summary_rows: list[dict[str, Any]] = []
    first_hits: list[np.ndarray] = []
    batch_indices: list[np.ndarray] = []
    chain_indices: list[np.ndarray] = []
    independent_unit_indices: list[np.ndarray] = []
    initial_temperature_indices: list[np.ndarray] = []
    independent_first_hits: list[np.ndarray] = []
    independent_batch_indices: list[np.ndarray] = []
    endpoint_success_masks: list[np.ndarray] = []
    endpoint_observed_masks: list[np.ndarray] = []
    final_energy_values: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    sampler_diagnostics: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        hits = batch.first_hit_sweeps.numpy().astype(np.int64, copy=False)
        endpoint_observed = batch.final_success_mask is not None
        endpoint_success = (
            np.zeros(hits.shape, dtype=np.bool_)
            if batch.final_success_mask is None
            else batch.final_success_mask.numpy().astype(np.bool_, copy=False)
        )
        final_energies = (
            np.full(hits.shape, np.nan, dtype=np.float64)
            if batch.final_energies is None
            else batch.final_energies.numpy().astype(np.float64, copy=False)
        )
        if endpoint_success.shape != hits.shape or final_energies.shape != hits.shape:
            raise ValueError("Endpoint evidence must contain one value per trajectory")
        endpoint_success_masks.append(endpoint_success)
        endpoint_observed_masks.append(np.full(hits.shape, endpoint_observed, dtype=np.bool_))
        final_energy_values.append(final_energies)
        independent_units = int(batch.independent_units or hits.size)
        if independent_units <= 0 or hits.size % independent_units:
            raise ValueError("AnnealResult independent units do not partition trajectories")
        dependency_size = hits.size // independent_units
        unit_hits = hits.reshape(independent_units, dependency_size)
        unit_first_hits = np.where(
            np.any(unit_hits >= 0, axis=1),
            np.min(np.where(unit_hits >= 0, unit_hits, censor_iteration + 1), axis=1),
            -1,
        ).astype(np.int64, copy=False)
        first_hits.append(hits)
        batch_indices.append(np.full(hits.shape, batch_index, dtype=np.int64))
        chain_indices.append(np.arange(hits.size, dtype=np.int64))
        independent_unit_indices.append(np.arange(hits.size, dtype=np.int64) // dependency_size)
        initial_temperature_indices.append(np.arange(hits.size, dtype=np.int64) % dependency_size)
        independent_first_hits.append(unit_first_hits)
        independent_batch_indices.append(
            np.full(unit_first_hits.shape, batch_index, dtype=np.int64)
        )
        successes = int(np.count_nonzero(hits >= 0))
        independent_successes = int(np.count_nonzero(unit_first_hits >= 0))
        endpoint_successes = int(np.count_nonzero(endpoint_success)) if endpoint_observed else None
        slot_sweeps = int(hits.size * batch.total_sweeps)
        summary_rows.append(
            {
                "batch_index": batch_index,
                "seed": batch.seed,
                "chains": hits.size,
                "independent_units": independent_units,
                "dependency_unit": batch.dependency_unit,
                "trajectories_per_independent_unit": dependency_size,
                "successes": successes,
                "success_rate": successes / hits.size,
                "endpoint_successes": endpoint_successes,
                "endpoint_success_rate": (
                    None if endpoint_successes is None else endpoint_successes / hits.size
                ),
                "endpoint_success_observed": endpoint_observed,
                "independent_unit_successes": independent_successes,
                "independent_unit_success_rate": independent_successes / independent_units,
                "total_sweeps": batch.total_sweeps,
                "kernel_seconds": batch.kernel_seconds,
                "seconds_per_chain": batch.kernel_seconds / hits.size,
                "seconds_per_chain_sweep": batch.kernel_seconds / slot_sweeps,
                "seconds_per_independent_unit": batch.kernel_seconds / independent_units,
                "slot_sweeps": slot_sweeps,
                "local_gibbs_node_updates": batch.local_gibbs_node_updates,
                "beta_stage_count": len(batch.beta_stage_sweeps),
                "beta_stage_sweeps_min": (
                    None if not batch.beta_stage_sweeps else min(batch.beta_stage_sweeps)
                ),
                "beta_stage_sweeps_max": (
                    None if not batch.beta_stage_sweeps else max(batch.beta_stage_sweeps)
                ),
                "local_gibbs_node_updates_per_second": (
                    None
                    if batch.local_gibbs_node_updates == 0
                    else batch.local_gibbs_node_updates / batch.kernel_seconds
                ),
                "best_energy_min": float(batch.best_energies.min().item()),
                "best_energy_median": float(batch.best_energies.median().item()),
                "final_energy_min": (
                    None
                    if not np.all(np.isfinite(final_energies))
                    else float(np.min(final_energies))
                ),
                "final_energy_median": (
                    None
                    if not np.all(np.isfinite(final_energies))
                    else float(np.median(final_energies))
                ),
                "peak_allocated_bytes": batch.peak_allocated_bytes,
                "peak_reserved_bytes": batch.peak_reserved_bytes,
                "device": batch.device,
                "field_backend": batch.field_backend,
                "gibbs_kernel": batch.gibbs_kernel,
                "rng_protocol": batch.rng_protocol,
                "fused_gibbs_calls": batch.fused_gibbs_calls,
                "energy_tracking": batch.energy_tracking,
                "backend_storage_bytes": batch.backend_storage_bytes,
                "backend_prepare_seconds": batch.backend_prepare_seconds,
                "metropolis_block_count": batch.metropolis_block_count,
                "metropolis_block_color_count": batch.metropolis_block_color_count,
                "metropolis_block_every": batch.metropolis_block_every,
                "metropolis_block_rule": batch.metropolis_block_rule,
                "exact_block_count": batch.exact_block_count,
                "exact_block_color_count": batch.exact_block_color_count,
                "exact_block_max_size": batch.exact_block_max_size,
                "exact_block_every": batch.exact_block_every,
                "exact_block_color_mode": batch.exact_block_color_mode,
                "exact_block_update_events": batch.exact_block_update_events,
                "exact_block_candidate_evaluations": (batch.exact_block_candidate_evaluations),
                "exact_block_plan_group_count": batch.exact_block_plan_group_count,
                "exact_block_field_evaluation_calls": (batch.exact_block_field_evaluation_calls),
                "exact_block_field_node_evaluations": (batch.exact_block_field_node_evaluations),
                "exact_block_workspace_bytes": batch.exact_block_workspace_bytes,
                "junction_tree_node_count": batch.junction_tree_node_count,
                "junction_tree_boundary_count": batch.junction_tree_boundary_count,
                "junction_tree_treewidth_upper_bound": (batch.junction_tree_treewidth_upper_bound),
                "junction_tree_conditional_table_states_per_replica": (
                    batch.junction_tree_conditional_table_states_per_replica
                ),
                "junction_tree_every": batch.junction_tree_every,
                "junction_tree_start_sweep": batch.junction_tree_start_sweep,
                "junction_tree_stop_sweep": batch.junction_tree_stop_sweep,
                "junction_tree_update_events": batch.junction_tree_update_events,
                "junction_tree_random_draws": batch.junction_tree_random_draws,
                "junction_tree_maximum_forward_table_states_per_replica": (
                    batch.junction_tree_maximum_forward_table_states_per_replica
                ),
                "junction_tree_replica_chunk_size": batch.junction_tree_replica_chunk_size,
                "junction_tree_chunk_count_per_event": (batch.junction_tree_chunk_count_per_event),
                "junction_tree_rng_stream": batch.junction_tree_rng_stream,
                "junction_tree_plan_count": batch.junction_tree_plan_count,
                "junction_tree_plan_node_counts": list(batch.junction_tree_plan_node_counts),
                "junction_tree_plan_boundary_counts": list(
                    batch.junction_tree_plan_boundary_counts
                ),
                "junction_tree_plan_treewidth_upper_bounds": list(
                    batch.junction_tree_plan_treewidth_upper_bounds
                ),
                "junction_tree_plan_conditional_table_states_per_replica": list(
                    batch.junction_tree_plan_conditional_table_states_per_replica
                ),
                "junction_tree_plan_update_events": list(batch.junction_tree_plan_update_events),
                "junction_tree_explicit_event_sweeps": list(
                    batch.junction_tree_explicit_event_sweeps
                ),
                "junction_tree_explicit_event_plan_indices": list(
                    batch.junction_tree_explicit_event_plan_indices
                ),
                "junction_tree_explicit_event_replica_ranges": [
                    list(value) for value in batch.junction_tree_explicit_event_replica_ranges
                ],
                "junction_tree_beta_override": batch.junction_tree_beta_override,
                "junction_tree_explicit_event_beta_overrides": list(
                    batch.junction_tree_explicit_event_beta_overrides
                ),
                "treewidth_corridor_count": batch.treewidth_corridor_count,
                "treewidth_corridor_color_count": batch.treewidth_corridor_color_count,
                "treewidth_corridor_max_nodes": batch.treewidth_corridor_max_nodes,
                "treewidth_corridor_max_treewidth": (batch.treewidth_corridor_max_treewidth),
                "treewidth_corridor_every": batch.treewidth_corridor_every,
                "treewidth_corridor_color_mode": batch.treewidth_corridor_color_mode,
                "treewidth_corridor_message_mode": batch.treewidth_corridor_message_mode,
                "treewidth_corridor_update_events": (batch.treewidth_corridor_update_events),
                "treewidth_corridor_gate_update_events": (
                    batch.treewidth_corridor_gate_update_events
                ),
                "treewidth_corridor_candidate_evaluations": (
                    batch.treewidth_corridor_candidate_evaluations
                ),
                "treewidth_corridor_field_evaluation_calls": (
                    batch.treewidth_corridor_field_evaluation_calls
                ),
                "treewidth_corridor_field_node_evaluations": (
                    batch.treewidth_corridor_field_node_evaluations
                ),
                "hastings_block_count": batch.hastings_block_count,
                "hastings_every": batch.hastings_every,
                "hastings_block_mode": batch.hastings_block_mode,
                "hastings_proposal_logit_clip": batch.hastings_proposal_logit_clip,
                "hastings_update_events": batch.hastings_update_events,
                "hastings_proposals": batch.hastings_proposals,
                "hastings_accepts": batch.hastings_accepts,
                "hastings_proposed_changes": batch.hastings_proposed_changes,
                "hastings_accepted_changes": batch.hastings_accepted_changes,
                "codeword_macro_move_count": batch.codeword_macro_move_count,
                "codeword_macro_move_names": list(batch.codeword_macro_move_names),
                "codeword_macro_plan_kind": batch.codeword_macro_plan_kind,
                "codeword_macro_support": batch.codeword_macro_support,
                "codeword_macro_rule": batch.codeword_macro_rule,
                "codeword_macro_uniform_mixture": batch.codeword_macro_uniform_mixture,
                "codeword_macro_every": batch.codeword_macro_every,
                "codeword_macro_start_sweep": batch.codeword_macro_start_sweep,
                "codeword_macro_stop_sweep": batch.codeword_macro_stop_sweep,
                "codeword_macro_update_events": batch.codeword_macro_update_events,
                "codeword_macro_proposals": batch.codeword_macro_proposals,
                "codeword_macro_accepts": batch.codeword_macro_accepts,
                "codeword_macro_candidate_evaluations": (
                    batch.codeword_macro_candidate_evaluations
                ),
                "codeword_macro_downhill_proposals": batch.codeword_macro_downhill_proposals,
                "codeword_macro_downhill_accepts": batch.codeword_macro_downhill_accepts,
                "codeword_macro_selected_move_counts": list(
                    batch.codeword_macro_selected_move_counts
                ),
                "codeword_macro_accepted_move_counts": list(
                    batch.codeword_macro_accepted_move_counts
                ),
                "codeword_macro_storage_bytes": batch.codeword_macro_storage_bytes,
                "local_field_proposal_clip": batch.local_field_proposal_clip,
                "gibbs_uniform_dtype": batch.gibbs_uniform_dtype,
                "local_field_clip_proposals": batch.local_field_clip_proposals,
                "local_field_clip_accepts": batch.local_field_clip_accepts,
                "local_field_clip_proposed_flips": batch.local_field_clip_proposed_flips,
                "local_field_clip_accepted_flips": batch.local_field_clip_accepted_flips,
                "gibbs_sweep_every": batch.gibbs_sweep_every,
                "restart_every_sweeps": batch.restart_every_sweeps,
                "restart_count": batch.restart_count,
                "parallel_tempering_temperatures": len(batch.parallel_tempering_beta_ladder),
                "parallel_tempering_exchange_every": (batch.parallel_tempering_exchange_every),
                "parallel_tempering_walkers_visited_both_endpoints": (
                    batch.parallel_tempering_walkers_visited_both_endpoints
                ),
                "hamiltonian_component_name": batch.hamiltonian_component_name,
                "hamiltonian_target_weight": batch.hamiltonian_target_weight,
                "hamiltonian_weight_schedule_steps": len(batch.hamiltonian_weight_schedule),
                "hamiltonian_tempering_weights": len(batch.hamiltonian_tempering_weights),
                "hamiltonian_tempering_target_slot": batch.hamiltonian_tempering_target_slot,
                "hamiltonian_success_policy": batch.hamiltonian_success_policy,
                "hamiltonian_tempering_exchange_every": (
                    batch.hamiltonian_tempering_exchange_every
                ),
                "hamiltonian_tempering_walkers_visited_both_endpoints": (
                    batch.hamiltonian_tempering_walkers_visited_both_endpoints
                ),
            }
        )
        attempts = np.asarray(batch.parallel_tempering_exchange_attempts, dtype=np.int64)
        accepts = np.asarray(batch.parallel_tempering_exchange_accepts, dtype=np.int64)
        if attempts.shape != accepts.shape or np.any(accepts < 0) or np.any(accepts > attempts):
            raise ValueError("Parallel-tempering exchange diagnostics are inconsistent")
        hamiltonian_attempts = np.asarray(
            batch.hamiltonian_tempering_exchange_attempts, dtype=np.int64
        )
        hamiltonian_accepts = np.asarray(
            batch.hamiltonian_tempering_exchange_accepts, dtype=np.int64
        )
        if (
            hamiltonian_attempts.shape != hamiltonian_accepts.shape
            or np.any(hamiltonian_accepts < 0)
            or np.any(hamiltonian_accepts > hamiltonian_attempts)
        ):
            raise ValueError("Hamiltonian-tempering exchange diagnostics are inconsistent")
        sampler_diagnostics.append(
            {
                "batch_index": batch_index,
                "seed": batch.seed,
                "dependency_unit": batch.dependency_unit,
                "independent_units": independent_units,
                "trajectories": int(hits.size),
                "trajectories_per_independent_unit": dependency_size,
                "beta_stage_sweeps": list(batch.beta_stage_sweeps),
                "independent_unit_successes": independent_successes,
                "restart_every_sweeps": batch.restart_every_sweeps,
                "restart_count": batch.restart_count,
                "exact_blocks": {
                    "count": batch.exact_block_count,
                    "colors": batch.exact_block_color_count,
                    "maximum_size": batch.exact_block_max_size,
                    "every": batch.exact_block_every,
                    "color_mode": batch.exact_block_color_mode,
                    "update_events": batch.exact_block_update_events,
                    "candidate_evaluations": batch.exact_block_candidate_evaluations,
                    "plan_group_count": batch.exact_block_plan_group_count,
                    "field_evaluation_calls": batch.exact_block_field_evaluation_calls,
                    "field_node_evaluations": batch.exact_block_field_node_evaluations,
                    "workspace_bytes": batch.exact_block_workspace_bytes,
                    "rng_stream": batch.exact_block_rng_stream,
                },
                "junction_tree": {
                    "node_count": batch.junction_tree_node_count,
                    "boundary_count": batch.junction_tree_boundary_count,
                    "treewidth_upper_bound": batch.junction_tree_treewidth_upper_bound,
                    "conditional_table_states_per_replica": (
                        batch.junction_tree_conditional_table_states_per_replica
                    ),
                    "every": batch.junction_tree_every,
                    "start_sweep": batch.junction_tree_start_sweep,
                    "stop_sweep": batch.junction_tree_stop_sweep,
                    "update_events": batch.junction_tree_update_events,
                    "random_draws": batch.junction_tree_random_draws,
                    "maximum_forward_table_states_per_replica": (
                        batch.junction_tree_maximum_forward_table_states_per_replica
                    ),
                    "replica_chunk_size": batch.junction_tree_replica_chunk_size,
                    "chunk_count_per_event": batch.junction_tree_chunk_count_per_event,
                    "rng_stream": batch.junction_tree_rng_stream,
                    "plan_count": batch.junction_tree_plan_count,
                    "plan_node_counts": list(batch.junction_tree_plan_node_counts),
                    "plan_boundary_counts": list(batch.junction_tree_plan_boundary_counts),
                    "plan_treewidth_upper_bounds": list(
                        batch.junction_tree_plan_treewidth_upper_bounds
                    ),
                    "plan_conditional_table_states_per_replica": list(
                        batch.junction_tree_plan_conditional_table_states_per_replica
                    ),
                    "plan_update_events": list(batch.junction_tree_plan_update_events),
                    "explicit_event_sweeps": list(batch.junction_tree_explicit_event_sweeps),
                    "explicit_event_plan_indices": list(
                        batch.junction_tree_explicit_event_plan_indices
                    ),
                    "explicit_event_replica_ranges": [
                        list(value) for value in batch.junction_tree_explicit_event_replica_ranges
                    ],
                    "beta_override": batch.junction_tree_beta_override,
                    "explicit_event_beta_overrides": list(
                        batch.junction_tree_explicit_event_beta_overrides
                    ),
                },
                "treewidth_corridors": {
                    "count": batch.treewidth_corridor_count,
                    "colors": batch.treewidth_corridor_color_count,
                    "maximum_nodes": batch.treewidth_corridor_max_nodes,
                    "maximum_treewidth": batch.treewidth_corridor_max_treewidth,
                    "every": batch.treewidth_corridor_every,
                    "color_mode": batch.treewidth_corridor_color_mode,
                    "message_mode": batch.treewidth_corridor_message_mode,
                    "update_events": batch.treewidth_corridor_update_events,
                    "gate_update_events": batch.treewidth_corridor_gate_update_events,
                    "candidate_evaluations": (batch.treewidth_corridor_candidate_evaluations),
                    "field_evaluation_calls": (batch.treewidth_corridor_field_evaluation_calls),
                    "field_node_evaluations": (batch.treewidth_corridor_field_node_evaluations),
                },
                "hastings_repair": {
                    "block_count": batch.hastings_block_count,
                    "every": batch.hastings_every,
                    "block_mode": batch.hastings_block_mode,
                    "proposal_logit_clip": batch.hastings_proposal_logit_clip,
                    "update_events": batch.hastings_update_events,
                    "proposals": batch.hastings_proposals,
                    "accepts": batch.hastings_accepts,
                    "proposed_changes": batch.hastings_proposed_changes,
                    "accepted_changes": batch.hastings_accepted_changes,
                    "acceptance": (
                        None
                        if batch.hastings_proposals == 0
                        else batch.hastings_accepts / batch.hastings_proposals
                    ),
                    "change_acceptance": (
                        None
                        if batch.hastings_proposed_changes == 0
                        else batch.hastings_accepted_changes / batch.hastings_proposed_changes
                    ),
                },
                "codeword_macro": {
                    "move_count": batch.codeword_macro_move_count,
                    "move_names": list(batch.codeword_macro_move_names),
                    "plan_kind": batch.codeword_macro_plan_kind,
                    "support": batch.codeword_macro_support,
                    "rule": batch.codeword_macro_rule,
                    "uniform_mixture": batch.codeword_macro_uniform_mixture,
                    "every": batch.codeword_macro_every,
                    "start_sweep": batch.codeword_macro_start_sweep,
                    "stop_sweep": batch.codeword_macro_stop_sweep,
                    "update_events": batch.codeword_macro_update_events,
                    "proposals": batch.codeword_macro_proposals,
                    "accepts": batch.codeword_macro_accepts,
                    "candidate_evaluations": batch.codeword_macro_candidate_evaluations,
                    "downhill_proposals": batch.codeword_macro_downhill_proposals,
                    "downhill_accepts": batch.codeword_macro_downhill_accepts,
                    "selected_move_counts": list(batch.codeword_macro_selected_move_counts),
                    "accepted_move_counts": list(batch.codeword_macro_accepted_move_counts),
                    "acceptance": (
                        None
                        if batch.codeword_macro_proposals == 0
                        else batch.codeword_macro_accepts / batch.codeword_macro_proposals
                    ),
                    "storage_bytes": batch.codeword_macro_storage_bytes,
                },
                "gibbs_engineering": {
                    "kernel": batch.gibbs_kernel,
                    "rng_protocol": batch.rng_protocol,
                    "fused_calls": batch.fused_gibbs_calls,
                },
                "stabilization": {
                    "gibbs_uniform_dtype": batch.gibbs_uniform_dtype,
                    "local_field_proposal_clip": batch.local_field_proposal_clip,
                    "clip_proposals": batch.local_field_clip_proposals,
                    "clip_accepts": batch.local_field_clip_accepts,
                    "clip_proposed_flips": batch.local_field_clip_proposed_flips,
                    "clip_accepted_flips": batch.local_field_clip_accepted_flips,
                    "clip_acceptance": (
                        None
                        if batch.local_field_clip_proposals == 0
                        else batch.local_field_clip_accepts / batch.local_field_clip_proposals
                    ),
                    "clip_flip_acceptance": (
                        None
                        if batch.local_field_clip_proposed_flips == 0
                        else batch.local_field_clip_accepted_flips
                        / batch.local_field_clip_proposed_flips
                    ),
                },
                "parallel_tempering": {
                    "beta_ladder": list(batch.parallel_tempering_beta_ladder),
                    "exchange_every": batch.parallel_tempering_exchange_every,
                    "exchange_attempts_by_edge": attempts.tolist(),
                    "exchange_accepts_by_edge": accepts.tolist(),
                    "exchange_acceptance_by_edge": [
                        (None if attempted == 0 else accepted / attempted)
                        for attempted, accepted in zip(
                            attempts.tolist(), accepts.tolist(), strict=True
                        )
                    ],
                    "walkers_visited_both_endpoints": (
                        batch.parallel_tempering_walkers_visited_both_endpoints
                    ),
                },
                "hamiltonian": {
                    "component_name": batch.hamiltonian_component_name,
                    "target_weight": batch.hamiltonian_target_weight,
                    "common_weight_schedule": list(batch.hamiltonian_weight_schedule),
                },
                "hamiltonian_tempering": {
                    "weight_ladder": list(batch.hamiltonian_tempering_weights),
                    "target_slot": batch.hamiltonian_tempering_target_slot,
                    "exchange_every": batch.hamiltonian_tempering_exchange_every,
                    "exchange_attempts_by_edge": hamiltonian_attempts.tolist(),
                    "exchange_accepts_by_edge": hamiltonian_accepts.tolist(),
                    "exchange_acceptance_by_edge": [
                        (None if attempted == 0 else accepted / attempted)
                        for attempted, accepted in zip(
                            hamiltonian_attempts.tolist(),
                            hamiltonian_accepts.tolist(),
                            strict=True,
                        )
                    ],
                    "walkers_visited_both_endpoints": (
                        batch.hamiltonian_tempering_walkers_visited_both_endpoints
                    ),
                    "success_policy": batch.hamiltonian_success_policy,
                },
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
    _write_json(
        destination / "sampler_diagnostics.json",
        {
            "schema_version": 1,
            "batches": sampler_diagnostics,
            "inference_note": (
                "Tempering slots inside one ladder are dependent. Batch-level inference "
                "remains primary; slot-level success is descriptive."
            ),
        },
    )

    combined_hits = np.concatenate(first_hits)
    combined_batch_indices = np.concatenate(batch_indices)
    np.savez_compressed(
        destination / "first_hits.npz",
        first_hit_sweeps=combined_hits,
        batch_index=combined_batch_indices,
        chain_index=np.concatenate(chain_indices),
        independent_unit_index=np.concatenate(independent_unit_indices),
        initial_temperature_index=np.concatenate(initial_temperature_indices),
        independent_first_hit_sweeps=np.concatenate(independent_first_hits),
        independent_unit_batch_index=np.concatenate(independent_batch_indices),
        independent_unit_local_index=np.concatenate(
            [np.arange(values.size, dtype=np.int64) for values in independent_first_hits]
        ),
        censor_sweep=np.asarray(censor_iteration, dtype=np.int64),
    )
    combined_endpoint_success = np.concatenate(endpoint_success_masks)
    combined_endpoint_observed = np.concatenate(endpoint_observed_masks)
    combined_final_energies = np.concatenate(final_energy_values)
    np.savez_compressed(
        destination / "endpoint_success.npz",
        endpoint_success=combined_endpoint_success,
        endpoint_observed=combined_endpoint_observed,
        final_energies=combined_final_energies,
        batch_index=combined_batch_indices,
        chain_index=np.concatenate(chain_indices),
        censor_sweep=np.asarray(censor_iteration, dtype=np.int64),
    )

    checkpoints = np.unique(
        np.asarray([point.sweep for batch in batches for point in batch.trace], dtype=np.int64)
    )
    curve = empirical_success_curve(
        combined_hits,
        checkpoints,
        censor_iteration=censor_iteration,
    )
    bootstrap_seed = int(analysis.get("bootstrap_seed", 0))
    bootstrap_resamples = int(analysis.get("bootstrap_resamples", 10_000))
    confidence = float(analysis.get("confidence", 0.95))
    grouped_curve = grouped_success_curve(
        combined_hits,
        combined_batch_indices,
        checkpoints,
        censor_iteration=censor_iteration,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    grouped_by_iteration = {point.iterations: point for point in grouped_curve}
    curve_rows = []
    for point in curve:
        grouped = grouped_by_iteration[point.iterations]
        curve_rows.append(
            {
                **asdict(point),
                "batch_groups": grouped.groups,
                "batch_group_mean_probability": grouped.group_mean_probability,
                "batch_group_standard_deviation": grouped.group_standard_deviation,
                "batch_group_confidence_low": grouped.confidence_low,
                "batch_group_confidence_high": grouped.confidence_high,
                "batch_group_interval_method": grouped.interval_method,
                "batch_group_interval_degenerate": grouped.interval_degenerate,
            }
        )
    with (destination / "success_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    survival = discrete_survival_curve(
        combined_hits,
        censor_iteration=censor_iteration,
        checkpoints=checkpoints,
    )
    survival_rows = [asdict(point) for point in survival]
    with (destination / "discrete_survival.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(survival_rows[0]))
        writer.writeheader()
        writer.writerows(survival_rows)

    rng_audits = [audit for batch in batches for audit in batch.rng_audits]
    if rng_audits:
        sampler_seeds = {audit.seed for audit in rng_audits if audit.stream_name == "sampler"}
        if sampler_seeds != {int(batch.seed) for batch in batches}:
            raise ValueError("Sampler RNG audit seeds do not match batch seeds")
        rng_payload = audit_stream_partition(rng_audits)
        rng_payload["status"] = "audited"
    else:
        rng_payload = {
            "schema_version": 1,
            "status": "unavailable_legacy_result_object",
            "governance_passed": False,
            "statistical_independence_proven": False,
            "batches": [],
            "limitations": "No generator state audit was attached to the supplied results.",
        }
    _write_json(destination / "rng_audit.json", rng_payload)

    targets = (0.90, 0.95, 0.99)
    mean_batch_seconds = sum(batch.kernel_seconds for batch in batches) / len(batches)
    observed_probability = float(np.count_nonzero(combined_hits >= 0) / combined_hits.size)
    endpoint_evidence_complete = bool(np.all(combined_endpoint_observed))
    endpoint_successes = (
        int(np.count_nonzero(combined_endpoint_success)) if endpoint_evidence_complete else None
    )
    combined_independent_hits = np.concatenate(independent_first_hits)
    independent_observed_probability = float(
        np.count_nonzero(combined_independent_hits >= 0) / combined_independent_hits.size
    )
    dependency_units = {batch.dependency_unit for batch in batches}
    if len(dependency_units) != 1:
        raise ValueError("All batches must use the same dependency unit")
    dependency_unit = dependency_units.pop()
    independent_unit_counts = [
        int(batch.independent_units or batch.first_hit_sweeps.numel()) for batch in batches
    ]
    batch_success_probability = sum(bool(np.any(hits >= 0)) for hits in first_hits) / len(batches)
    successful_batches = sum(bool(np.any(hits >= 0)) for hits in first_hits)
    batch_confidence_low, batch_confidence_high = clopper_pearson_interval(
        successful_batches,
        len(batches),
        confidence,
    )
    restart_tts: dict[str, float | None] = {}
    for target in targets:
        estimate = time_to_solution(mean_batch_seconds, batch_success_probability, target)
        restart_tts[f"tts{int(target * 100)}_seconds_at_censor"] = (
            estimate if math.isfinite(estimate) else None
        )
    weibull = fit_censored_weibull_its(combined_hits, censor_iteration, targets)
    grouped_censor = grouped_by_iteration[censor_iteration]
    cutoff_payload = None
    if cutoff_config is not None:
        raw_cutoffs = cutoff_config.get("candidate_cutoffs")
        selection = bootstrap_restart_cutoff(
            combined_hits,
            combined_batch_indices,
            np.asarray(raw_cutoffs),
            censor_iteration=censor_iteration,
            target_probability=float(cutoff_config.get("target_probability", 0.99)),
            confidence=confidence,
            resamples=int(cutoff_config.get("bootstrap_resamples", bootstrap_resamples)),
            seed=int(cutoff_config.get("bootstrap_seed", bootstrap_seed)),
        )
        cutoff_payload = asdict(selection)
    _write_json(
        destination / "its.json",
        {
            "censor_sweep": censor_iteration,
            "chains": int(combined_hits.size),
            "trajectory_units": int(combined_hits.size),
            "independent_units": int(sum(independent_unit_counts)),
            "independent_units_per_batch": independent_unit_counts,
            "dependency_unit": dependency_unit,
            "trajectory_success_inference": (
                "descriptive_correlated_slots"
                if dependency_unit in {"parallel_tempering_ladder", "hamiltonian_tempering_ladder"}
                else "independent_chain_estimate_subject_to_rng_audit_limits"
            ),
            "observed_success_probability": observed_probability,
            "endpoint_only": {
                "evidence_complete": endpoint_evidence_complete,
                "successes": endpoint_successes,
                "trials": int(combined_endpoint_success.size),
                "probability": (
                    None
                    if endpoint_successes is None
                    else endpoint_successes / combined_endpoint_success.size
                ),
                "evaluation_sweep": censor_iteration,
                "predicate": "same_certificate_and_energy_predicate_as_first_hit",
                "file": "endpoint_success.npz",
            },
            "independent_unit_observed_success_probability": (independent_observed_probability),
            "observed_batch_success_probability": batch_success_probability,
            "empirical": {
                f"its{int(target * 100)}": _empirical_its(combined_hits, target, censor_iteration)
                for target in targets
            },
            "independent_unit_empirical": {
                f"its{int(target * 100)}": _empirical_its(
                    combined_independent_hits, target, censor_iteration
                )
                for target in targets
            },
            "independent_unit_effort_note": (
                "A tempering ladder consumes one update per ladder slot per sweep; multiply "
                "ladder-sweeps by its slot count for slot-sweep work."
            ),
            "empirical_unreached_means": f"> {censor_iteration}",
            "parallel_batch_restart_tts": restart_tts,
            "parallel_batch_restart_tts_note": (
                "Uses measured batch success probability and mean batch kernel time. "
                "Null means no successful batch was observed."
            ),
            "batch_group_inference": {
                "groups": grouped_censor.groups,
                "inference_unit": "batch",
                "group_mean_success_probability": grouped_censor.group_mean_probability,
                "group_standard_deviation": grouped_censor.group_standard_deviation,
                "confidence": confidence,
                "confidence_low": grouped_censor.confidence_low,
                "confidence_high": grouped_censor.confidence_high,
                "interval_method": grouped_censor.interval_method,
                "interval_degenerate": grouped_censor.interval_degenerate,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_resamples": bootstrap_resamples,
                "small_group_warning": grouped_censor.groups < 8,
                "batch_any_success": {
                    "successes": successful_batches,
                    "trials": len(batches),
                    "probability": batch_success_probability,
                    "clopper_pearson_low": batch_confidence_low,
                    "clopper_pearson_high": batch_confidence_high,
                },
            },
            "discrete_survival": {
                "file": "discrete_survival.csv",
                "method": "integer_boundary_product_limit",
                "administrative_censor_iteration": censor_iteration,
            },
            "cutoff_selection_bootstrap": cutoff_payload,
            "weibull_exploratory": asdict(weibull),
        },
    )
    return destination
