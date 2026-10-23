"""Custom registry and isolated graph-bisection runner."""

from __future__ import annotations

import resource
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
from .model import read_metis
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "partition",
    "combinatorial_solvers.partition.runner:run_config",
    "Strictly balanced graph bisection",
    input_fields=(
        "problem.graph",
        "initial.partition",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.license",
        "baseline.metadata",
        "baseline.build_log",
        "baseline.configure_log",
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
        graph = read_metis(config["problem"]["graph"])
        np.savez(output / "instance.npz", size=graph.size, edges=graph.edges, weights=graph.weights)
        initial = (
            np.loadtxt(config["initial"]["partition"], dtype=np.int8)
            if "initial" in config
            else None
        )
        if config["method"] == "native":
            arrays, summary = solve(graph, Settings(**config["solver"]), initial)
        elif config["method"] == "kaffpa":
            arrays, summary = run_baseline(
                graph, config["problem"]["graph"], config["baseline"], output, initial
            )
        else:
            raise ValueError("Unknown bisection method")
        np.savez_compressed(output / "states.npz", **arrays)
        np.savetxt(
            output / "best_partition.txt",
            arrays["best_bits"][int(np.argmin(arrays["best_cut"]))],
            fmt="%d",
        )
        before = perf_counter()
        for name in ("initial", "best", "final", "trace"):
            if not np.array_equal(graph.certify(arrays[f"{name}_bits"]), arrays[f"{name}_cut"]):
                raise AssertionError(f"Cut certificate failed: {name}")
        summary["certification_seconds"] = perf_counter() - before
        if config.get("exact_reference", False):
            summary["exact_reference"] = graph.exact()
            if summary["best_cut"] < summary["exact_reference"]["cut"]:
                raise AssertionError("Cut below the exact optimum")
        summary.update(
            status="completed",
            model_binary_variables=graph.size,
            objective_active_variables=len(np.unique(graph.edges)),
            edge_count=len(graph.edges),
            balance_tolerance=0,
            common_initial_runner_seconds=config.get("initial", {}).get("runner_seconds", 0.0),
            complete_runner_seconds=perf_counter() - started,
            peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        dump_json(output / "benchmark_summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    return output
