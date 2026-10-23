"""The MWIS ConfigRunner and its isolated registry extension."""

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

from .baselines import chils, greedy
from .model import exact_optimum, instance_from_problem
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "mwis",
    "combinatorial_solvers.mwis.runner:run_config",
    "Maximum-weight independent set with integer certificates",
    input_fields=(
        "problem.instance_file",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.metadata",
        "baseline.license",
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
    root = Path(repository or Path(__file__).resolve().parents[3])
    torch.set_num_threads(1)
    dump_json(output / "config.json", config)
    try:
        dump_json(output / "environment.json", capture_provenance(root))
        instance = instance_from_problem(config["problem"])
        dump_json(output / "instance.json", instance.payload())
        if config.get("method") == "chils":
            bits, summary = chils(instance, config["baseline"], output)
            np.savez(output / "states.npz", best_bits=bits)
        elif config.get("method") == "greedy":
            before = perf_counter()
            bits = greedy(instance)
            summary = {
                "method": "greedy",
                "search_seconds": perf_counter() - before,
                "certificate": instance.certificate(bits),
            }
            np.savez(output / "states.npz", best_bits=bits)
        elif config.get("method") == "native":
            arrays, summary = solve(instance, Settings(**config["solver"]))
            np.savez(output / "states.npz", **arrays)
            dump_json(output / "benchmark_summary.json", summary)
            if summary["failure"]:
                raise RuntimeError(f"Native search interrupted: {summary['failure']}")
            before = perf_counter()
            certificates = []
            for row, weight in zip(arrays["best_bits"], arrays["best_weight"], strict=True):
                certificate = instance.certificate(row)
                if weight >= 0 and (not certificate["feasible"] or certificate["weight"] != weight):
                    raise AssertionError("Native saved-state certificate mismatch")
                certificates.append(certificate if weight >= 0 else None)
            final = [instance.certificate(row) for row in arrays["final_bits"]]
            summary.update(
                certificates=certificates,
                final_certificates=final,
                certification_seconds=perf_counter() - before,
            )
        else:
            raise ValueError("method must be native, chils, or greedy")
        if config.get("exact_reference", False):
            before = perf_counter()
            reference = exact_optimum(instance)
            summary.update(
                exact_reference=reference, exact_reference_seconds=perf_counter() - before
            )
        summary.update(
            status="completed",
            problem=config["problem"],
            model_binary_variables=instance.size,
            physical_update_pbits_per_model=instance.size if config["method"] == "native" else 0,
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
