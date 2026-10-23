"""Max-Cut runner registered through the existing isolated-experiment extension."""

from __future__ import annotations

import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    validate_experiment_output,
)

from .baseline import run as run_baseline
from .model import exact_optimum, instance_from_problem
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "maxcut",
    "combinatorial_solvers.maxcut.runner:run_config",
    "Weighted Max-Cut",
    input_fields=(
        "problem.instance_file",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.upstream_archive",
        "baseline.patch",
        "baseline.probe",
        "baseline.probe_upstream",
        "baseline.probe_corrected",
        "baseline.driver",
        "baseline.license",
        "baseline.metadata",
        "baseline.build_log",
    ),
)


def registry():
    return SolverRegistry([SPECIFICATION])


def run_config(config, *, output_override=None, overwrite=False, repository=None):
    output = experiment_output(config, output_override)
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    torch.set_num_threads(1)
    root = Path(repository or Path(__file__).resolve().parents[3])
    dump_json(output / "config.json", config)
    try:
        dump_json(output / "environment.json", capture_provenance(root))
        instance = instance_from_problem(config["problem"])
        instance.save(output / "instance.npz")
        if config["method"] == "native":
            arrays, summary = solve(instance, Settings(**config["solver"]))
            np.savez(output / "states.npz", **arrays)
            dump_json(output / "benchmark_summary.json", summary)
            if summary["failure"]:
                raise RuntimeError(f"Native search failed: {summary['failure']}")
            before = perf_counter()
            actual = instance.cut(arrays["best_bits"])
            if not np.array_equal(actual, arrays["best_cut"]):
                raise AssertionError("Native integer certificate mismatch")
            summary.update(
                certified_cuts=actual.tolist(),
                final_cuts=instance.cut(arrays["final_bits"]).tolist(),
                certification_seconds=perf_counter() - before,
            )
        elif config["method"] == "mqlib":
            arrays, summary = run_baseline(instance, config["baseline"], output)
            np.savez(output / "states.npz", **arrays)
        else:
            raise ValueError("Unknown solver family")
        if config.get("exact_reference", False):
            before = perf_counter()
            reference = exact_optimum(instance)
            if summary["best_cut"] > reference["optimum_cut"]:
                raise AssertionError("Cut exceeds the exhaustive optimum")
            summary.update(
                exact_reference=reference, exact_reference_seconds=perf_counter() - before
            )
        summary.update(
            status="completed",
            problem=config["problem"],
            model_binary_variables=instance.size,
            physical_update_pbits_per_model=instance.size if config["method"] == "native" else 0,
            objective_active_variables=instance.active_variables,
            complete_runner_seconds=perf_counter() - started,
        )
        dump_json(output / "benchmark_summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
            },
        )
        raise
    return output
