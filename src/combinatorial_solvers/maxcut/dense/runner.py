"""Dense-only runner with explicit packed inputs and separate numerical kernels."""

from __future__ import annotations

import resource
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.maxcut.baseline import run as run_baseline
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    validate_experiment_output,
)

from .model import exact_optimum, instance_from_problem
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "maxcut",
    "combinatorial_solvers.maxcut.dense.runner:run_config",
    "Dense packed Max-Cut",
    input_fields=(
        "problem.instance_file",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.driver",
        "baseline.header",
        "baseline.license",
        "baseline.metadata",
        "baseline.build_log",
        "reference.manifest",
        "reference.original_graph",
        "reference.anechoic_methods",
        "reference.anechoic_ablation",
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
    root = Path(repository or Path(__file__).resolve().parents[4])
    dump_json(output / "config.json", config)
    try:
        dump_json(output / "environment.json", capture_provenance(root))
        before = perf_counter()
        instance = instance_from_problem(config["problem"])
        instance.save(output / "instance.npz")
        model_seconds = perf_counter() - before
        dump_json(
            output / "model.json",
            {
                "size": instance.size,
                "family": instance.family,
                "edges": instance.edge_count,
                "total_weight": instance.total_weight,
                "packed_bytes": len(instance.packed),
                "active_variables": instance.active_variables,
                "objective": "maximize integer weighted cut",
                "density": 2 * instance.edge_count / (instance.size * (instance.size - 1)),
            },
        )
        if config["method"] == "native":
            arrays, summary = solve(instance, Settings(**config["solver"]))
            np.savez(output / "states.npz", **arrays)
            dump_json(output / "benchmark_summary.json", summary)
            if summary["failure"]:
                raise RuntimeError(f"Dense search failed: {summary['failure']}")
            before = perf_counter()
            actual = instance.cut(arrays["best_bits"])
            if not np.array_equal(actual, arrays["best_cut"]):
                raise AssertionError("Dense integer bitset certificate failed")
            summary.update(
                certified_cuts=actual.tolist(),
                final_cuts=instance.cut(arrays["final_bits"]).tolist(),
                certification_seconds=perf_counter() - before,
            )
        elif config["method"] == "mqlib":
            arrays, summary = run_baseline(instance, config["baseline"], output)
            np.savez(output / "states.npz", **arrays)
            summary.update(physical_update_pbits_per_model=0, continuous_states_per_replica=0)
        else:
            raise ValueError("Unknown dense solver family")
        if config.get("exact_reference", False):
            before = perf_counter()
            reference = exact_optimum(instance)
            if summary["best_cut"] > reference["optimum_cut"]:
                raise AssertionError("Dense result exceeds exhaustive optimum")
            summary.update(
                exact_reference=reference, exact_reference_seconds=perf_counter() - before
            )
        summary.update(
            status="completed",
            problem=config["problem"],
            model_binary_variables=instance.size,
            objective_active_variables=instance.active_variables,
            model_seconds=model_seconds,
            complete_runner_seconds=perf_counter() - started,
            peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
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
