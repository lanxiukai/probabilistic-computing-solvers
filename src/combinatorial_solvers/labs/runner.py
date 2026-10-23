"""Isolated LABS runner; no input or solver dependency on another problem."""

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
from .model import Instance, correlations, legendre
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "labs",
    "combinatorial_solvers.labs.runner:run_config",
    "Aperiodic unrestricted native LABS",
    input_fields=(
        "baseline.executable",
        "baseline.source_archive",
        "baseline.driver",
        "baseline.portable_flip",
        "baseline.correction",
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
        if set(config["problem"]) != {"size"}:
            raise ValueError("LABS instance has only its length; a new seed is not a new instance")
        instance = Instance(config["problem"]["size"])
        np.savez(output / "instance.npz", size=instance.size)
        dump_json(
            output / "model.json",
            {
                "size": instance.size,
                "boundary": "aperiodic",
                "domain": "all binary sequences",
                "objective": "minimize sum of squared nonzero-lag correlations",
                "auxiliary_pbits": 0,
            },
        )
        if config["method"] == "native":
            arrays, summary = solve(instance, Settings(**config["solver"]))
            np.savez(output / "states.npz", **arrays)
            dump_json(output / "benchmark_summary.json", summary)
            if summary["failure"]:
                raise RuntimeError(f"LABS kernel failed: {summary['failure']}")
            before = perf_counter()
            for name in ("best", "final", "pending"):
                actual = instance.energy(arrays[f"{name}_bits"])
                if not np.array_equal(actual, arrays[f"{name}_energy"]):
                    raise AssertionError(f"Integer certificate failed: {name}")
            if not np.array_equal(correlations(arrays["final_bits"]), arrays["final_correlations"]):
                raise AssertionError("Integer final correlation certificate failed")
            summary["certification_seconds"] = perf_counter() - before
        elif config["method"] == "lmats":
            arrays, summary = run_baseline(instance, config["baseline"], output)
            np.savez(output / "states.npz", **arrays)
        elif config["method"] == "constructive":
            before = perf_counter()
            bits, construction = legendre(instance.size, appended=config["appended"])
            build_seconds = perf_counter() - before
            before = perf_counter()
            corr = correlations(bits)
            energy = int(np.dot(corr, corr))
            arrays = {"best_bits": bits, "best_energy": energy, "correlations": corr}
            summary = {
                "method": "constructive",
                "best_energy": energy,
                "merit_factor": instance.size**2 / (2 * energy),
                "construction": construction,
                "construction_seconds": build_seconds,
                "certification_seconds": perf_counter() - before,
                "physical_update_pbits_per_model": 0,
                "replicas": 1,
            }
            np.savez(output / "states.npz", **arrays)
        else:
            raise ValueError("Unknown LABS runner method")
        if config.get("exact_reference", False):
            before = perf_counter()
            reference = instance.exact_optimum()
            if summary["best_energy"] < reference["optimum_energy"]:
                raise AssertionError("Search result is below the exact optimum")
            summary.update(
                exact_reference=reference, exact_reference_seconds=perf_counter() - before
            )
        summary.update(
            status="completed",
            model_binary_variables=instance.size,
            objective_active_variables=instance.active_variables,
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
