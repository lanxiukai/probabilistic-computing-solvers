"""Covering-array ConfigRunner with complete original pair-coverage certificates."""

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

from .model import read_instance

SPECIFICATION = SolverSpec(
    "covering-array",
    "combinatorial_solvers.covering_array.runner:run_config",
    "Pairwise covering-array conditional heat baths",
    input_fields=("problem.instance_file", "baseline.jar"),
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

    def terminate(signum, frame):
        raise InterruptedError("Covering-array runner interrupted by signal")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        instance = read_instance(Path(config["problem"]["instance_file"]))
        if config.get("method", "native") == "acts-ipog":
            from .baseline import acts_reference

            arrays, summary = acts_reference(instance, config["baseline"], output)
        else:
            from .solver import Settings, solve

            arrays, summary = solve(
                instance, Settings(**config["solver"]), evidence_directory=output
            )
            certification_started = perf_counter()
            certificates = {}
            for key, cost_key in (
                ("values", "cost"),
                ("best_values", "best_cost"),
                ("quality_best_values", "quality_best_cost"),
            ):
                certificates[key] = [instance.certificate(a) for a in arrays[key]]
                if any(
                    not c["valid_assignment"] or c["missing_interactions"] != int(value)
                    for c, value in zip(certificates[key], arrays[cost_key], strict=True)
                ):
                    raise AssertionError(
                        "Independent pair coverage differs from the incremental cache"
                    )
            for r, a in enumerate(arrays["values"]):
                np.testing.assert_array_equal(instance.counts(a), arrays["counts"][r])
            if not np.array_equal(
                arrays["bits"], np.eye(instance.levels, dtype=np.int8)[arrays["values"]]
            ):
                raise AssertionError("One-hot bits differ from the array entry cache")
            summary.update(
                certificates=certificates,
                zero_missing_hits=int((arrays["quality_best_cost"] == 0).sum()),
                certification_seconds=perf_counter() - certification_started,
            )
        output_started = perf_counter()
        np.savez_compressed(output / "states.npz", **arrays)
        summary.update(
            rows=instance.rows,
            columns=instance.columns,
            levels=instance.levels,
            strength=2,
            total_interactions=instance.pairs * instance.levels**2,
            optimality_scope=(
                "Zero missing interactions certifies this row count; "
                "it does not prove the minimum possible number of rows."
            ),
            status="completed",
            final_array_write_seconds=perf_counter() - output_started,
            complete_runner_seconds=perf_counter() - started,
        )
        dump_json(output / "summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return output
