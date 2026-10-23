"""QAP ConfigRunner with independently checked original integer witnesses."""

from __future__ import annotations

import signal
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    validate_experiment_output,
)

from .model import read_qaplib, read_solution

SPECIFICATION = SolverSpec(
    "qap",
    "combinatorial_solvers.qap.runner:run_config",
    "Permutation-constrained QAP block heat baths",
    input_fields=(
        "problem.instance_file",
        "problem.reference_file",
        "baseline.executable",
        "baseline.source",
        "baseline.build_metadata",
        "baseline.build_log",
        "baseline.instrumented_source",
        "baseline.adapter_source",
        "baseline.upstream_conditions",
    ),
)


def registry():
    return SolverRegistry([SPECIFICATION])


def run_config(config, *, output_override=None, overwrite=False, repository=None):
    output = experiment_output(config, output_override)
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    dump_json(output / "config.json", config)
    dump_json(
        output / "environment.json",
        capture_provenance(Path(repository or Path(__file__).resolve().parents[3])),
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("QAP runner received SIGTERM; partial evidence is retained")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        instance = read_qaplib(Path(config["problem"]["instance_file"]))
        method = config.get("method", "native")
        if method in {"attribute-tabu", "taillard-rots"}:
            if method == "taillard-rots":
                from .rots import rots_reference as reference_solver
            else:
                from .baseline import tabu_reference as reference_solver

            arrays, summary = reference_solver(
                instance, config["problem"]["instance_file"], config["baseline"], output
            )
        else:
            from .solver import Settings, solve

            arrays, summary = solve(
                instance, Settings(**config["solver"]), evidence_directory=output
            )
            certificate_started = perf_counter()
            certificates = {}
            for key, cost_key in (
                ("permutation", "cost"),
                ("best_permutation", "best_cost"),
                ("quality_best_permutation", "quality_best_cost"),
            ):
                certificates[key] = [instance.certificate(p) for p in arrays[key]]
                if any(
                    not c["feasible"] or c["cost"] != int(cost)
                    for c, cost in zip(certificates[key], arrays[cost_key], strict=True)
                ):
                    raise AssertionError(
                        "Original QAP integer certificate differs from the saved state"
                    )
            expected = np.eye(instance.size, dtype=np.int8)[arrays["permutation"]]
            if not np.array_equal(expected, arrays["bits"]):
                raise AssertionError("Persistent binary assignment differs from permutation cache")
            summary["certificates"] = certificates
            summary["state_certificate_seconds"] = perf_counter() - certificate_started
        winner_started = perf_counter()
        quality_permutations = np.atleast_2d(arrays["quality_best_permutation"])
        winner = quality_permutations[int(np.argmin(np.atleast_1d(arrays["quality_best_cost"])))]
        python_cost = sum(
            int(instance.flow[i, j]) * int(instance.distance[winner[i], winner[j]])
            for i in range(instance.size)
            for j in range(instance.size)
        )
        if python_cost != summary["best_feasible_cost"]:
            raise AssertionError("Unbounded Python integer winner certificate differs")
        summary["python_integer_best_cost_certificate"] = python_cost
        summary["winner_certificate_seconds"] = perf_counter() - winner_started
        archive_started = perf_counter()
        np.savez_compressed(output / "states.npz", **arrays)
        summary["state_archive_seconds"] = perf_counter() - archive_started
        if config["problem"].get("reference_file"):
            reference_started = perf_counter()
            _, reference = read_solution(
                Path(config["problem"]["reference_file"]),
                instance,
                index_base=config["problem"].get("reference_index_base", 1),
                inverse=config["problem"].get("reference_inverse", False),
            )
            summary["reference_witness"] = reference
            summary["reference_status"] = config["problem"].get(
                "reference_status", "feasible_reference"
            )
            summary["relative_gap"] = (
                (summary["best_feasible_cost"] - reference["cost"]) / reference["cost"]
                if reference["cost"]
                else None
            )
            quality = np.atleast_1d(arrays["quality_best_cost"])
            initial = np.atleast_1d(arrays.get("initial_cost", summary["initial_best_cost"]))
            curves = (
                arrays["curve_best"] if "curve_best" in arrays else arrays["curve_cost"][:, None]
            )
            for label, quality_hits, initial_hits, observations in (
                (
                    "reference",
                    quality == reference["cost"],
                    initial == reference["cost"],
                    curves == reference["cost"],
                ),
                (
                    "reference_or_better",
                    quality <= reference["cost"],
                    initial <= reference["cost"],
                    curves <= reference["cost"],
                ),
            ):
                summary[label + "_hits"] = int(quality_hits.sum())
                summary["initial_" + label + "_hits"] = int(initial_hits.sum())
                summary["new_" + label + "_hits_after_initialization"] = int(
                    (quality_hits & ~initial_hits).sum()
                )
                observed = np.any(observations, axis=1)
                observed &= arrays["curve_eligible"]
                summary["ensemble_first_" + label + "_observation_seconds"] = (
                    0.0
                    if initial_hits.any()
                    else float(arrays["curve_seconds"][observed][0])
                    if observed.any()
                    else None
                )
            summary["first_hit_time_resolution"] = (
                "Recorded improvement time for the CPU comparison, or the upper bound from "
                "the completed GPU chunk; no unobserved transition-level time is inferred."
            )
            summary["reference_certificate_seconds"] = perf_counter() - reference_started
        summary.update(status="completed", complete_runner_seconds=perf_counter() - started)
        dump_json(output / "summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return output
