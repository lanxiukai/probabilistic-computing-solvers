"""Audited native GA-EAX with explicit seeds and returned-tour certificates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.core.source_archive import pack
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    require_mapping,
    validate_experiment_output,
)

from .configuration import _target_length
from .instance import TSPInstance, certify_tour
from .tsplib import load_tsplib

SCHEMA = "tsp-standalone-eax-v1"


@dataclass(frozen=True)
class EAXConfig:
    max_seconds: float
    executable_sha256: str
    kind: str = "eax"
    device: str = "cpu"
    seed: int = 1
    population: int = 100
    offspring: int = 30
    restarts: int = 1

    def __post_init__(self):
        if self.kind != "eax" or self.device != "cpu":
            raise ValueError("Native GA-EAX requires kind=eax and device=cpu")
        if (
            isinstance(self.max_seconds, bool)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be positive and finite")
        for name in ("seed", "population", "offspring", "restarts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid positive integer {name}")
        if self.population < 2 or self.offspring > 1500 or self.seed + self.restarts > 65536:
            raise ValueError("Require population >= 2, offspring <= 1500, seeds <= 65535")
        if len(self.executable_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.executable_sha256
        ):
            raise ValueError("Expected lowercase SHA256 executable identity")


def read_eax_tours(instance: TSPInstance, path: Path) -> list[dict]:
    """Require complete original N/cost plus N-city records; never repair output."""
    values = [int(token) for token in path.read_text().split()]
    width = instance.dimension + 2
    if not values or len(values) % width:
        raise ValueError("Truncated GA-EAX tour output")
    result = []
    for start in range(0, len(values), width):
        n, claimed = values[start : start + 2]
        tour = tuple(city - 1 for city in values[start + 2 : start + width])
        certificate = certify_tour(instance, tour)
        if (
            n != instance.dimension
            or not certificate.is_permutation
            or certificate.length != claimed
        ):
            raise ValueError("GA-EAX returned an invalid tour or inconsistent integer cost")
        result.append({"tour": tour, "length": claimed})
    return result


def run_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    del overwrite
    solver = EAXConfig(**require_mapping(config, "solver"))
    problem = require_mapping(config, "problem")
    output = experiment_output(config, output_override).resolve()
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    started = perf_counter()
    stop = started + solver.max_seconds
    best = first_hit = None
    events = []
    dump_json(output / "config.json", {"schema": SCHEMA, "input": config})
    try:
        input_path = output / "input.tsp"
        shutil.copy2(problem["instance"], input_path)
        instance = load_tsplib(input_path, optimum=problem.get("optimum"))
        if instance.dimension <= 50 or instance.edge_weight_type not in {"EUC_2D", "ATT"}:
            raise ValueError("Upstream GA-EAX bridge supports EUC_2D/ATT with > 50 cities")
        target = (
            _target_length(problem, instance.optimum)
            if (
                instance.optimum is not None
                or "target_length" in problem
                or "target_relative_gap" in problem
            )
            else None
        )
        work = output / "eax"
        work.mkdir()
        executable = Path(require_mapping(config, "prefix")["eax_executable"]).resolve()
        identity = hashlib.sha256(executable.read_bytes()).hexdigest()
        if identity != solver.executable_sha256 or not os.access(executable, os.X_OK):
            raise ValueError("GA-EAX executable identity or permissions do not match")
        shutil.copy2(executable, work / "solver")
        # The upstream parser requires separate colon tokens in headers.
        lines = input_path.read_text().splitlines()
        (work / "input.tsp").write_text(
            "\n".join(" : ".join(line.split(":", 1)) if ":" in line else line for line in lines)
            + "\n"
        )
        source = pack(root, output / "source")
        dump_json(
            output / "environment.json",
            {
                **capture_provenance(root),
                "source_archive_sha256": source["archive_sha256"],
                "executable_sha256": identity,
            },
        )
        dump_json(
            output / "config.json",
            {
                "schema": SCHEMA,
                "input": config,
                "resolved": {
                    "solver": asdict(solver),
                    "target_length": target,
                    "reference_optimum_sent_to_solver": False,
                    "reference_tour_loaded": False,
                    "instance_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                },
            },
        )
        prepared = perf_counter()
        status = "restart_limit"
        with (output / "events.jsonl").open("x", buffering=1) as journal:
            for index in range(solver.restarts):
                remaining = stop - perf_counter()
                if remaining <= 0:
                    status = "wall_clock_limit"
                    break
                name = f"run-{index}"
                command = [
                    str(work / "solver"),
                    "1",
                    name,
                    str(solver.population),
                    str(solver.offspring),
                    "input.tsp",
                ]
                seed = solver.seed + index
                dump_json(work / f"{name}.command.json", {"command": command, "TSP_EAX_SEED": seed})
                native_started = perf_counter()
                timed_out, code = False, None
                with (work / f"{name}.log").open("xb") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=work,
                            env={**os.environ, "TSP_EAX_SEED": str(seed)},
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=remaining,
                            check=False,
                        )
                        code = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                event = {
                    "event_index": index,
                    "stage": "eax_returned_tour",
                    "seed": seed,
                    "return_code": code,
                    "timed_out": timed_out,
                    "native_process_seconds": perf_counter() - native_started,
                }
                tour_path = work / f"{name}_BestSol"
                if tour_path.exists():
                    tours = read_eax_tours(instance, tour_path)
                    if len(tours) != 1:
                        raise ValueError("Expected one native GA-EAX trial per process")
                    event.update(tours[0])
                    if best is None or event["length"] < best["length"]:
                        best = tours[0]
                    if first_hit is None and target is not None and event["length"] <= target:
                        now = perf_counter()
                        first_hit = {
                            **tours[0],
                            "event_index": index,
                            "stage": event["stage"],
                            "certified_elapsed_seconds": now - started,
                            "certified_monotonic_seconds": now,
                            "search_to_certificate_seconds": now - prepared,
                        }
                event["elapsed_seconds"] = perf_counter() - started
                events.append(event)
                journal.write(json.dumps(event) + "\n")
                if code not in (None, 0):
                    raise RuntimeError("GA-EAX exited with failure; original output is preserved")
                if timed_out:
                    status = "wall_clock_limit"
                    break
                if not tour_path.exists():
                    raise RuntimeError("GA-EAX completed without a complete tour")
        elapsed = perf_counter() - started
        dump_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": status,
                "device": "cpu",
                "best_tour": None if best is None else best["tour"],
                "best_length": None if best is None else best["length"],
                "target_length": target,
                "first_observed_target_hit": first_hit,
                "observed_target_met": first_hit is not None,
                "within_budget_hit": first_hit is not None
                and first_hit["certified_elapsed_seconds"] <= solver.max_seconds,
                "preparation_seconds": prepared - started,
                "search_seconds": elapsed - (prepared - started),
                "elapsed_seconds_before_summary_write": elapsed,
                "budget_overrun_seconds": max(0, elapsed - solver.max_seconds),
                "completed_restarts": len(events),
                "executable_sha256": identity,
                "observation_boundary": (
                    "complete tours after process exit; internal first passage unobserved"
                ),
            },
        )
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
            },
        )
        raise
    return output
