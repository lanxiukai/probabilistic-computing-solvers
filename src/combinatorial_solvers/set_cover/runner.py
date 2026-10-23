"""Covering ConfigRunner with explicit inputs and immutable run outputs."""

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

from .baseline import cft_reference, fast_greedy, mip_reference
from .model import read_json, read_orlib
from .reference import exact_optimum

SPECIFICATION = SolverSpec(
    "set-cover",
    "combinatorial_solvers.set_cover.runner:run_config",
    "Weighted covering with original integer certificates",
    input_fields=(
        "problem.instance_file",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.source_manifest",
        "baseline.fmt_archive",
        "baseline.fmt_manifest",
        "baseline.metadata",
        "baseline.license",
        "baseline.fmt_license",
        "baseline.adapter",
        "baseline.build_log",
        "baseline.clock_trace",
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
    dump_json(output / "config.json", config)
    try:
        root = Path(repository or Path(__file__).resolve().parents[3])
        dump_json(output / "environment.json", capture_provenance(root))
        problem = config["problem"]
        path = Path(problem["instance_file"])
        instance = (
            read_json(path)
            if problem.get("format") == "json"
            else read_orlib(path, problem.get("format", "scp"))
        )
        dump_json(output / "instance-statistics.json", instance.statistics())
        before = perf_counter()
        if config["method"] == "native":
            from .solver import Settings, solve

            arrays, summary = solve(
                instance, Settings(**config["solver"]), evidence_directory=output
            )
        elif config["method"] == "greedy":
            bits = fast_greedy(instance)
            arrays = {"best_bits": bits[None, :]}
            summary = {"method": "greedy", "search_seconds": perf_counter() - before}
        elif config["method"] == "cft":
            arrays, summary = cft_reference(instance, config["baseline"], output)
        elif config["method"] == "milp":
            arrays, summary = mip_reference(instance, config["baseline"].get("max_seconds", 60))
        else:
            raise ValueError("Method must be native, greedy, cft, or milp")
        np.savez_compressed(output / "states.npz", **arrays)
        certification = perf_counter()
        certificates = []
        for i, row in enumerate(arrays.get("best_bits", [])):
            cert = instance.certificate(row)
            marked = arrays.get("best_cost", np.zeros(len(arrays["best_bits"]), dtype=int))[i] >= 0
            if marked and not cert["feasible"]:
                raise AssertionError("Stored best cover is infeasible")
            if "best_cost" in arrays and marked and cert["cost"] != arrays["best_cost"][i]:
                raise AssertionError("Stored best cover cost differs")
            certificates.append(cert)
        summary["certificates"] = certificates
        if config["method"] != "native":
            summary["best_feasible_cost"] = min(
                (c["cost"] for c in certificates if c["feasible"]), default=None
            )
            summary["physical_update_pbits_per_model"] = 0
            summary["pbit_heatbath_participation"] = False
        if config.get("exact_reference"):
            summary["exact_reference"] = exact_optimum(instance)
        summary.update(
            status="failed" if summary.get("failure") else "completed",
            model_binary_variables=instance.size,
            rows=len(instance.rows),
            incidences=sum(map(len, instance.rows)),
            certification_seconds=perf_counter() - certification,
            peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            complete_runner_seconds=perf_counter() - started,
        )
        dump_json(output / "summary.json", summary)
        if summary.get("failure"):
            raise RuntimeError(summary["failure"])
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
