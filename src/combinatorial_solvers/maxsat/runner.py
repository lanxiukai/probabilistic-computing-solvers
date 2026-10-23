"""Weighted Max-SAT ConfigRunner using the shared isolated-execution contract."""

from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combinatorial_solvers.core.evidence import write_new_json
from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    validate_experiment_output,
)

from .baseline import portfolio
from .baseline import run as run_baseline
from .model import exact_optimum, instance_from_problem
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "maxsat",
    "combinatorial_solvers.maxsat.runner:run_config",
    "Weighted soft-clause Max-SAT",
    input_fields=(
        "problem.instance_file",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.upstream_archive",
        "baseline.patch",
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
    experiment = config.get("experiment", {})
    target = experiment.get("target_cost")
    budget = experiment.get("runner_seconds")
    reserve = float(experiment.get("return_reserve_seconds", 2))
    deadline = None if budget is None else started + float(budget) - reserve
    first_certificate = None

    def record_target(cost, packet, index):
        nonlocal first_certificate
        if first_certificate is not None or target is None or cost > target:
            return
        elapsed = perf_counter() - started
        first_certificate = {
            "target_cost": target,
            "certified_cost": int(cost),
            "elapsed_runner_seconds": elapsed,
            "within_budget": budget is None or elapsed <= budget,
            "witness_file": packet.relative_to(output).as_posix(),
            "witness_key": "best_bits",
            "witness_index": index,
        }
        write_new_json(output / "first_certificate.json", first_certificate)

    try:
        dump_json(output / "environment.json", capture_provenance(root))
        instance = instance_from_problem(config["problem"])
        instance.save(output / "instance.npz")
        dump_json(
            output / "model.json",
            {
                "variables": instance.variables,
                "clauses": len(instance.weights),
                "active_variables": instance.active_variables,
                "hard_clauses": 0,
                "total_weight": sum(map(int, instance.weights)),
                "objective": "minimize unsatisfied soft-clause weight",
            },
        )
        if config["method"] == "native":
            settings = Settings(**config["solver"])
            if target is not None:
                settings = replace(settings, target_cost=target)
            arrays, summary = solve(instance, settings, sampling_deadline=deadline)
            np.savez(output / "states.npz", **arrays)
            dump_json(output / "benchmark_summary.json", summary)
            if summary["failure"]:
                raise RuntimeError(f"Native search failed: {summary['failure']}")
            before = perf_counter()
            costs = instance.cost(arrays["best_bits"])
            if not np.array_equal(costs, arrays["best_cost"]):
                raise AssertionError("Best-state integer certificate mismatch")
            record_target(int(costs.min()), output / "states.npz", int(costs.argmin()))
            summary.update(
                certified_costs=costs.tolist(),
                final_costs=instance.cost(arrays["final_bits"]).tolist(),
                certification_seconds=perf_counter() - before,
            )
        elif config["method"] == "usw-ls":
            baseline = dict(config["baseline"])
            if target is not None:
                baseline["target_cost"] = target
            if baseline.get("workers", 1) > 1:
                arrays, summary = portfolio(
                    instance,
                    baseline,
                    output,
                    absolute_deadline=deadline,
                    on_certificate=record_target,
                )
                np.savez(output / "states.npz", **arrays)
            else:
                bits, summary = run_baseline(instance, baseline, output, absolute_deadline=deadline)
                np.savez(output / "states.npz", best_bits=bits)
                record_target(summary["best_cost"], output / "states.npz", None)
        else:
            raise ValueError("method must be native or usw-ls")
        if config.get("exact_reference", False):
            before = perf_counter()
            reference = exact_optimum(instance)
            if summary["best_cost"] < reference["optimum_cost"]:
                raise AssertionError("Observed objective below the exact optimum")
            summary.update(
                exact_reference=reference, exact_reference_seconds=perf_counter() - before
            )
        summary.update(
            status="completed",
            problem=config["problem"],
            model_binary_variables=instance.variables,
            physical_update_pbits_per_model=(
                instance.variables if config["method"] == "native" else 0
            ),
            objective_active_variables=instance.active_variables,
            complete_runner_seconds=perf_counter() - started,
            first_certificate=first_certificate,
            target_cost=target,
            runner_budget_seconds=budget,
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
