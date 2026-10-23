"""Frozen-input CPU/CUDA local-search cost and exact-output comparisons."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import numpy as np
import torch

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.core.source_archive import pack
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    require_mapping,
    validate_experiment_output,
)

from .baselines import nearest_neighbor_tsp
from .hybrid import Deadline, perturb_tour
from .hybrid_backend import LocalSearchBackend
from .hybrid_local import first_descent
from .instance import certify_tour
from .tsplib import load_tsplib


def run_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    del overwrite
    settings = require_mapping(config, "solver")
    problem = require_mapping(config, "problem")
    output = experiment_output(config, output_override).resolve()
    validate_experiment_output(output, False)
    batches = settings.get("batches", [1, 8, 32, 128])
    tiles = settings.get("tiles", [128, 256, 512])
    repeats = settings.get("repeats", 3)
    moves = settings.get("local_search_passes", 200)
    method = settings.get("local_search", "two_opt_or_opt")
    seed = settings["seed"]
    if not batches or any(type(value) is not int or value <= 0 for value in batches):
        raise ValueError("Benchmark batches must be positive integers")
    if not tiles or any(tile not in {128, 256, 512, 1024} for tile in tiles):
        raise ValueError("Unsupported CUDA benchmark tile")
    if type(repeats) is not int or repeats < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Benchmark requires positive repeats and a nonnegative seed")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    started = perf_counter()
    deadline = Deadline(started + float(settings["max_seconds"]))
    dump_json(output / "config.json", config)
    try:
        pack(root, output / "source")
        input_bytes = Path(problem["instance"]).read_bytes()
        (output / "input.tsp").write_bytes(input_bytes)
        instance = load_tsplib(output / "input.tsp")
        torch.set_num_threads(1)
        dump_json(output / "environment.json", {**capture_provenance(root), "command": sys.argv})
        rng = Random(seed)
        generated = perf_counter()
        bases = [
            first_descent(
                instance,
                nearest_neighbor_tsp(instance, start_city=city).tour,
                moves,
                lambda: 1.0,
                or_opt=method == "two_opt_or_opt",
            )
            for city in rng.sample(range(instance.dimension), min(8, instance.dimension))
        ]
        tours = []
        for _ in range(max(batches)):
            tour = rng.choice(bases)
            for _ in range(2):
                tour = perturb_tour(tour, rng)
            tours.append(tour)
        expected = tuple(
            first_descent(instance, tour, moves, lambda: 1.0, or_opt=method == "two_opt_or_opt")
            for tour in tours
        )
        fixture_seconds = perf_counter() - generated
        np.savez_compressed(
            output / "fixtures.npz",
            input_tours=np.asarray(tours),
            expected_tours=np.asarray(expected),
            base_tours=np.asarray(bases),
        )
        dump_json(
            output / "rng.json",
            {"seed": seed, "state_after_fixture_generation": rng.getstate(), "problem_trials": 0},
        )
        rows = []
        with (output / "measurements.jsonl").open("x") as journal:
            for device, tile in [("cpu", 256), *[("cuda", tile) for tile in tiles]]:
                deadline.check()
                created = perf_counter()
                backend = LocalSearchBackend(
                    instance,
                    device=device,
                    method=method,
                    max_moves=moves,
                    capacity=max(batches),
                    tile=tile,
                )
                setup = perf_counter() - created
                for batch in batches:
                    for repeat in range(repeats + 1):
                        deadline.check()
                        before = backend.metadata()
                        call_started = perf_counter()
                        result = backend.solve(tours[:batch], lambda: 1.0)
                        seconds = perf_counter() - call_started
                        after = backend.metadata()
                        matched = result == expected[:batch]
                        if not all(certify_tour(instance, tour).is_permutation for tour in result):
                            raise AssertionError("A benchmark result is not a complete tour")
                        row = {
                            "device": device,
                            "tile": tile if device == "cuda" else None,
                            "batch": batch,
                            "repeat": repeat,
                            "first_for_batch": repeat == 0,
                            "solve_wall_seconds": seconds,
                            "setup_seconds": setup,
                            "exact_cpu_match": matched,
                            "kernel_event_seconds": (
                                after["phase_seconds"]["kernel"] - before["phase_seconds"]["kernel"]
                                if device == "cuda"
                                else None
                            ),
                        }
                        rows.append(row)
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                        np.savez_compressed(
                            output / f"outputs-{device}-{tile}-{batch}-{repeat}.npz",
                            tours=np.asarray(result),
                        )
                        if not matched:
                            raise AssertionError("CUDA and CPU descent differ on a frozen tour")
                    dump_json(output / f"backend-{device}-{tile}-{batch}.json", backend.metadata())
        dump_json(
            output / "benchmark_summary.json",
            {
                "schema": "tsp-hybrid-local-benchmark-v1",
                "cities": instance.dimension,
                "method": method,
                "fixture_generation_and_reference_seconds": fixture_seconds,
                "measurements": rows,
                "elapsed_seconds": perf_counter() - started,
                "timing_scope": (
                    "solve wall includes validation, transfers, device work and certificates; "
                    "setup and compile are separate"
                ),
                "kernel_event_note": (
                    "CUDA event interval includes first-launch dispatch/loading gaps; "
                    "it is not an isolated instruction-throughput measurement"
                ),
                "independent_solver_trials": 0,
            },
        )
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
