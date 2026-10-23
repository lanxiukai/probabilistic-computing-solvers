"""Job-shop ConfigRunner preserving complete original schedule certificates."""

from __future__ import annotations

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

from .model import read_instance

SPECIFICATION = SolverSpec(
    "jssp",
    "combinatorial_solvers.jssp.runner:run_config",
    "Job-shop priority-block heat baths",
    input_fields=("problem.instance_file",),
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
    try:
        instance = read_instance(Path(config["problem"]["instance_file"]))
        if config.get("method", "native") == "cp-sat":
            from .baseline import cp_sat

            arrays, summary = cp_sat(
                instance, **config["baseline"], evidence_directory=output / "recovery"
            )
        else:
            from .solver import Settings, solve

            arrays, summary = solve(
                instance, Settings(**config["solver"]), evidence_directory=output / "recovery"
            )
            certification_started = perf_counter()
            certificates = {}
            for key, cost_key in (
                ("sequence", "cost"),
                ("best_sequence", "best_cost"),
                ("quality_best_sequence", "quality_best_cost"),
            ):
                schedules = np.array(
                    [
                        instance.decode(
                            s, schedule_decoder=config["solver"].get("schedule_decoder", "append")
                        )
                        for s in arrays[key]
                    ]
                )
                certificates[key] = [instance.certificate(s) for s in schedules]
                if any(
                    not c["feasible"] or c["makespan"] != int(value)
                    for c, value in zip(certificates[key], arrays[cost_key], strict=True)
                ):
                    raise AssertionError("Original schedule differs from the GPU makespan cache")
                arrays[key + "_schedules"] = schedules
            if not np.array_equal(
                arrays["bits"], np.eye(instance.jobs, dtype=np.int8)[arrays["sequence"]]
            ):
                raise AssertionError("One-hot priority bits differ from the sequence cache")
            summary["certificates"] = certificates
            summary["reference_hits"] = int(
                (arrays["quality_best_cost"] == config["problem"].get("reference", -1)).sum()
            )
            reference = config["problem"].get("reference", -1)
            initial_hits = arrays["initial_cost"] == reference
            first_hit = np.full(len(initial_hits), np.nan)
            first_hit[initial_hits] = 0.0
            for seconds, costs, eligible in zip(
                arrays["curve_seconds"], arrays["curve_best"], arrays["curve_eligible"], strict=True
            ):
                if eligible:
                    first_hit[np.isnan(first_hit) & (costs == reference)] = seconds
            arrays["first_reference_hit_seconds"] = first_hit
            summary["initial_reference_hits"] = int(initial_hits.sum())
            summary["first_reference_hit_seconds"] = (
                float(np.nanmin(first_hit)) if np.isfinite(first_hit).any() else None
            )
            summary["certification_seconds"] = perf_counter() - certification_started
        np.savez_compressed(output / "states.npz", **arrays)
        reference = config["problem"].get("reference")
        summary.update(
            jobs=instance.jobs,
            machines=instance.machine_count,
            operations=instance.size,
            elementary_lower_bound=instance.lower_bound(),
            reference=reference,
            reference_status=config["problem"].get("reference_status", "not_supplied"),
            relative_gap=(summary["best_feasible_cost"] - reference) / reference
            if reference and summary["best_feasible_cost"] is not None
            else None,
            status="completed",
            complete_runner_seconds=perf_counter() - started,
        )
        dump_json(output / "summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    return output
