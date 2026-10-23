"""Auditable hybrid runner with CPU or CUDA deterministic local search.

Use the existing scripts/tsp/solve.py entry point with solver.kind=hybrid.
Required settings are problem.instance, solver.max_seconds, and
experiment.output_dir. Optional prefix.kind is ils, lkh, or provided. All
remaining solver/prefix settings are the fields of HybridConfig/PrefixConfig.

For a shared LKH/EAX prefix, prefix.tours_file accepts a JSON object with schema
"tsp-hybrid-prefix-v1", instance_sha256, and elites=[{"tour": [zero-based city
IDs], "length": integer}]. Optional prefix_cost_seconds records the declared
upstream generation cost. Saved prefix.json files use this same format.

Outputs use a separate hybrid schema. Subtour repair is an explicit algorithm
stage, and target hits refer to certified pipeline tours, never raw auxiliary
bit states. No solver, test, benchmark, dependency installation, or download is
performed by importing this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.core.source_archive import pack
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    dump_yaml,
    experiment_output,
    json_safe,
    require_mapping,
    validate_experiment_output,
)

from .configuration import _target_length
from .hybrid import Deadline, HybridConfig, search
from .hybrid_backend import LocalSearchBackend
from .hybrid_classical import PrefixConfig, build_prefix
from .instance import certify_tour
from .tsplib import load_tsplib

SCHEMA = "tsp-ab-cycle-hybrid-v2"


def run_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    """Run a bounded hybrid pipeline and retain partial evidence on interruption."""
    del overwrite  # Hybrid evidence is never replaced, including failed runs.
    problem = require_mapping(config, "problem")
    solver = HybridConfig(**require_mapping(config, "solver"))
    prefix = PrefixConfig(**(require_mapping(config, "prefix") if "prefix" in config else {}))
    output = experiment_output(config, output_override).resolve()
    validate_experiment_output(output, False)
    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    deadline = Deadline(started + solver.max_seconds)
    best_observation: dict | None = None
    first_hit: dict | None = None
    events = 0
    dump_json(output / "config.json", {"schema_version": 2, "schema": SCHEMA, "input": config})
    try:
        source_path = Path(problem["instance"]).resolve()
        source_bytes = source_path.read_bytes()
        input_path = output / "input.tsp"
        input_path.write_bytes(source_bytes)
        instance_sha256 = hashlib.sha256(source_bytes).hexdigest()
        optimum = problem.get("optimum")
        instance = load_tsplib(input_path, optimum=None if optimum is None else int(optimum))
        if instance.dimension < 4:
            raise ValueError("The AB-cycle hybrid requires at least four cities")
        if int(instance.distance.max()) * instance.dimension > np.iinfo(np.int64).max:
            raise ValueError(
                "Closed-tour lengths could overflow the int64 certificate representation"
            )
        target = (
            _target_length(problem, instance.optimum)
            if instance.optimum is not None
            or "target_length" in problem
            or "target_relative_gap" in problem
            else None
        )
        source_manifest = pack(root, output / "source")
        environment = capture_provenance(root)
        backend = LocalSearchBackend(
            instance,
            device=solver.device,
            method=solver.local_search,
            max_moves=solver.local_search_passes,
            capacity=solver.local_search_batch_size,
            tile=solver.cuda_tile,
        )
        environment.update(
            command=sys.argv,
            backend="cuda_local_descent_cpu_graph" if solver.device == "cuda" else "cpu_reference",
            local_search_backend=backend.metadata(),
            source_bundle="source",
            source_archive_sha256=source_manifest["archive_sha256"],
        )
        dump_json(output / "environment.json", environment)
        resolved = {
            "schema_version": 2,
            "schema": SCHEMA,
            "input": config,
            "problem": {**problem, "instance": str(input_path)},
            "solver": asdict(solver),
            "prefix": asdict(prefix),
            "resolved": {
                "instance_sha256": instance_sha256,
                "cities": instance.dimension,
                "edge_weight_type": instance.edge_weight_type,
                "target_length": target,
                "distance_dtype": "int64",
                "probability_dtype": "float64",
                "hybrid_random_source": "Python random.Random (MT19937)",
                "external_rng": "snapshotted LKH binary" if prefix.kind == "lkh" else None,
                "rng_seeds": {
                    "prefix": solver.seed,
                    "classical": solver.seed + 1,
                    "cycles": solver.seed + 2,
                    "bits": solver.seed + 3,
                },
                "energy": "two_factor_length / positive_edge_median + penalty * (subtours - 1)",
                "equilibrium_claim": False,
                "budget_boundary": "from preparation; CPU move or bounded CUDA batch boundary",
                "batch_schedule": "repair each group, then polish it as a batch on either device",
                "declared_optimum_used_for_search_guidance": False,
                "reference_validation": "declared optimum; no independent optimality proof",
                "journal_elite_pool_encoding": "inline or reference to a prior full pool event",
            },
        }
        dump_json(output / "config.json", resolved)
        replay = {
            "problem": resolved["problem"],
            "solver": asdict(solver),
            "prefix": asdict(prefix),
            "experiment": {"output_dir": str(output)},
        }
        dump_yaml(output / "replay_config.yaml", replay)
        preparation_seconds = perf_counter() - started

        with (output / "events.jsonl").open("x", encoding="utf-8") as journal:
            last_pool = None
            last_pool_event = None

            def emit(event: dict) -> None:
                nonlocal best_observation, first_hit, events, last_pool, last_pool_event
                event = {
                    "event_index": events,
                    "elapsed_seconds": perf_counter() - started,
                    **event,
                }
                journal_event = dict(event)
                if "elite_pool" in event:
                    if event["elite_pool"] == last_pool:
                        del journal_event["elite_pool"]
                        journal_event["elite_pool_ref_event"] = last_pool_event
                    else:
                        last_pool, last_pool_event = event["elite_pool"], events
                journal.write(json.dumps(json_safe(journal_event), allow_nan=False) + "\n")
                journal.flush()
                events += 1
                if "tour" not in event:
                    return
                certificate = certify_tour(instance, event["tour"])
                if (
                    not certificate.is_permutation
                    or certificate.length is None
                    or certificate.length != event["length"]
                ):
                    raise AssertionError("A pipeline tour failed independent integer certification")
                observation = {
                    "event_index": event["event_index"],
                    "stage": event["stage"],
                    "certified_elapsed_seconds": perf_counter() - started,
                    "certified_monotonic_seconds": perf_counter(),
                    "tour": certificate.tour,
                    "length": certificate.length,
                }
                checkpoint_changed = False
                if best_observation is None or certificate.length < best_observation["length"]:
                    best_observation = observation
                    checkpoint_changed = True
                if first_hit is None and target is not None and certificate.length <= target:
                    first_hit = observation
                    checkpoint_changed = True
                # Every candidate remains in the durable journal. Rewrite the recovery
                # checkpoint only when its best or first-hit certificate changes.
                if not checkpoint_changed:
                    return
                temporary = output / "checkpoint.json.tmp"
                dump_json(
                    temporary,
                    {
                        "best_observation": best_observation,
                        "first_observed_target_hit": first_hit,
                        "elite_pool": event["elite_pool"],
                    },
                )
                os.replace(temporary, output / "checkpoint.json")

            pool, prefix_metadata = build_prefix(
                instance, instance_sha256, prefix, solver, output, deadline, emit, backend
            )
            if prefix.kind == "lkh":
                replay["prefix"]["lkh_executable"] = str(output / "lkh" / "solver")
            elif prefix.kind == "provided":
                replay["prefix"]["tours_file"] = str(output / "provided-prefix.json")
            dump_yaml(output / "replay_config.yaml", replay)
            upstream = prefix_metadata.get("upstream_prefix_seconds")
            if first_hit is not None:
                first_hit["elapsed_plus_declared_upstream_seconds"] = (
                    None
                    if prefix.kind == "provided" and upstream is None
                    else first_hit["certified_elapsed_seconds"] + (upstream or 0)
                )
            search_started = perf_counter()
            result = search(instance, pool, solver, deadline, emit, backend)
            search_seconds = perf_counter() - search_started
            emit({"stage": "search_complete", **result})

        certification_started = perf_counter()
        upstream = prefix_metadata.get("upstream_prefix_seconds")
        if first_hit is not None:
            first_hit["elapsed_plus_declared_upstream_seconds"] = (
                None
                if prefix.kind == "provided" and upstream is None
                else first_hit["certified_elapsed_seconds"] + (upstream or 0)
            )
        certificates = [asdict(certify_tour(instance, tour)) for tour in pool.tours]
        if any(not certificate["is_permutation"] for certificate in certificates):
            raise AssertionError("A final elite is not a Hamiltonian tour")
        dump_json(
            output / "certificates.json",
            {
                "elites": certificates,
                "target_length": target,
                "first_observed_target_hit": first_hit,
                "certificate_scope": "valid integer-cost closed tour; no new lower bound",
            },
        )
        certification_seconds = perf_counter() - certification_started
        elapsed = perf_counter() - started
        dump_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": result["stop_reason"],
                "selector": solver.selector,
                "device": solver.device,
                "local_search_backend": backend.metadata(),
                "best_tour": pool.best,
                "best_length": pool.lengths[pool.best],
                "first_observed_target_hit": first_hit,
                "target_length": target,
                "observed_target_met": None if target is None else first_hit is not None,
                "prefix": prefix_metadata,
                "search": result,
                "events": events,
                "preparation_seconds": preparation_seconds,
                "search_seconds": search_seconds,
                "certification_seconds": certification_seconds,
                "elapsed_seconds_before_summary_write": elapsed,
                "budget_overrun_seconds": max(0.0, elapsed - solver.max_seconds),
                "elapsed_plus_declared_upstream_seconds": (
                    None
                    if prefix.kind == "provided" and upstream is None
                    else elapsed + (upstream or 0)
                ),
                "observation_boundary": "certified tours after explicit repair/polishing",
                "timing_boundary": "input, source snapshot, prefix, search, IO and certificates",
                "replay_note": "Wall-time cutoffs vary work; events record actual boundaries.",
                "independent_problem_instances": 1,
            },
        )
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
                "events": events,
                "best_observation": best_observation,
                "first_observed_target_hit": first_hit,
            },
        )
        raise
    return output
