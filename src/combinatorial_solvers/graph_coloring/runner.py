"""Graph coloring ConfigRunner with complete edge and palette certificates."""

from __future__ import annotations

import hashlib
import json
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

from .model import read_dimacs

SPECIFICATION = SolverSpec(
    "graph-coloring",
    "combinatorial_solvers.graph_coloring.runner:run_config",
    "Fixed-palette graph and list coloring",
    input_fields=(
        "problem.instance_file",
        "problem.allowed_file",
        "problem.symmetry_clique_file",
        "baseline.executable",
        "baseline.source",
        "baseline.build_metadata",
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
    dump_json(output / "config.json", config)
    dump_json(
        output / "environment.json",
        capture_provenance(Path(repository or Path(__file__).resolve().parents[3])),
    )
    try:
        problem = config["problem"]
        allowed = (
            np.load(problem["allowed_file"], allow_pickle=False)
            if problem.get("allowed_file")
            else None
        )
        instance = read_dimacs(Path(problem["instance_file"]), problem["colors"], allowed)
        clique = None
        clique_preparation_seconds = 0.0
        if problem.get("symmetry_clique_file"):
            if problem.get("allowed_file"):
                raise ValueError("Clique color symmetry is only supported for standard coloring")
            payload = json.loads(Path(problem["symmetry_clique_file"]).read_text())
            actual_hash = hashlib.sha256(Path(problem["instance_file"]).read_bytes()).hexdigest()
            if payload["input_sha256"] != actual_hash:
                raise ValueError("Clique preprocessing certificate belongs to another graph")
            clique = instance.clique_certificate(payload["vertices"])
            instance = instance.with_clique_symmetry(payload["vertices"])
            clique_preparation_seconds = payload["elapsed_seconds"]
        certificates = {}
        if config.get("method", "native") == "tabucol-style":
            from .baseline import tabu_reference

            arrays, summary = tabu_reference(
                instance, problem["instance_file"], config["baseline"], output
            )
        else:
            from .solver import Settings, solve

            arrays, summary = solve(
                instance, Settings(**config["solver"]), evidence_directory=output
            )
            certificates = {}
            for key, cost_key in (
                ("colors", "conflicts"),
                ("best_colors", "best_conflicts"),
                ("quality_best_colors", "quality_best_cost"),
            ):
                certificates[key] = [instance.certificate(c) for c in arrays[key]]
                if any(
                    not c["valid_assignment"] or c["conflicts"] != int(value)
                    for c, value in zip(certificates[key], arrays[cost_key], strict=True)
                ):
                    raise AssertionError(
                        "Saved conflict count differs from independent edge enumeration"
                    )
            if not np.array_equal(
                arrays["bits"], np.eye(instance.colors, dtype=np.int8)[arrays["colors"]]
            ):
                raise AssertionError("One-hot state differs from the color cache")
            for r, c in enumerate(arrays["colors"]):
                expected = np.array(
                    [np.bincount(c[row], minlength=instance.colors) for row in instance.adjacency()]
                )
                np.testing.assert_array_equal(expected, arrays["counts"][r])
        if clique is not None:
            arrays["symmetry_clique_vertices"] = np.asarray(clique["vertices"], dtype=np.int64)
        np.savez_compressed(output / "states.npz", **arrays)
        nonisolated = np.array([bool(row) for row in instance.adjacency()])
        summary.update(
            certificates=certificates,
            vertices=instance.size,
            edges=len(instance.edges),
            palette=instance.colors,
            allowed_binary_variables=int(instance.allowed.sum()),
            fixed_zero_variables=int((~instance.allowed).sum()),
            isolated_vertices=int((~nonisolated).sum()),
            nonisolated_vertices=int(nonisolated.sum()),
            interacting_model_binary_variables=int(nonisolated.sum()) * instance.colors,
            interacting_allowed_binary_variables=int(instance.allowed[nonisolated].sum()),
            singleton_color_vertices=int((instance.allowed.sum(axis=1) == 1).sum()),
            unclamped_allowed_binary_variables=int(
                instance.allowed[instance.allowed.sum(axis=1) > 1].sum()
            ),
            interacting_unclamped_allowed_binary_variables=int(
                instance.allowed[nonisolated & (instance.allowed.sum(axis=1) > 1)].sum()
            ),
            classical_clique_preprocessing=clique,
            one_time_clique_preprocessing_seconds=clique_preparation_seconds,
            chromatic_lower_bound=clique["size"] if clique is not None else None,
            chromatic_optimum_certified=bool(
                clique is not None
                and clique["size"] == instance.colors
                and summary["best_feasible_cost"] == 0
            ),
            zero_conflict_hits=int(
                (
                    np.atleast_1d(arrays.get("quality_best_cost", summary["best_feasible_cost"]))
                    == 0
                ).sum()
            ),
            optimality_scope=(
                "Zero conflicts is the ground state of this fixed-palette objective; "
                "a chromatic lower bound requires the separately verified clique certificate."
            ),
            status="completed",
            complete_runner_seconds=perf_counter() - started,
        )
        dump_json(output / "summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {"error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    return output
