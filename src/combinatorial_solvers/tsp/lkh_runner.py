"""Bounded standalone LKH execution with original inputs and strict certificates.

The executable uses the existing explicit input field prefix.lkh_executable.
Targets are observed only by this wrapper; no reference optimum or tour is sent
to LKH. A returned-tour certificate upper-bounds the native first-passage time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
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
from .hybrid_classical import _read_lkh_tour
from .instance import certify_tour
from .tsplib import load_tsplib

SCHEMA = "tsp-standalone-lkh-v1"


@dataclass(frozen=True)
class LKHConfig:
    max_seconds: float
    kind: str = "lkh"
    device: str = "cpu"
    seed: int = 0
    restarts: int = 8
    max_trials: int | None = None
    precision: int = 100
    executable_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind != "lkh" or self.device != "cpu":
            raise ValueError("Standalone LKH requires kind=lkh and device=cpu")
        if (
            isinstance(self.max_seconds, bool)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("solver.max_seconds must be finite and positive")
        for name in ("seed", "restarts", "max_trials", "precision"):
            value = getattr(self, name)
            if value is None and name == "max_trials":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (name != "seed"):
                raise ValueError(f"Invalid integer solver.{name}")
        if self.executable_sha256 is not None and (
            len(self.executable_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.executable_sha256)
        ):
            raise ValueError("Expected a lowercase SHA256 executable identity")


def run_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    """Run independent native restarts, preserving failures and timed-out outputs."""
    del overwrite
    problem = require_mapping(config, "problem")
    solver = LKHConfig(**require_mapping(config, "solver"))
    executable = Path(require_mapping(config, "prefix")["lkh_executable"]).resolve()
    output = experiment_output(config, output_override).resolve()
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    started = perf_counter()
    stop = started + solver.max_seconds
    first_hit = best = None
    observations = []
    dump_json(output / "config.json", {"schema": SCHEMA, "input": config})
    try:
        input_path = output / "input.tsp"
        input_path.write_bytes(Path(problem["instance"]).read_bytes())
        instance = load_tsplib(input_path, optimum=problem.get("optimum"))
        target = (
            _target_length(problem, instance.optimum)
            if instance.optimum is not None
            or "target_length" in problem
            or "target_relative_gap" in problem
            else None
        )
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("prefix.lkh_executable must be an executable file")
        identity = hashlib.sha256(executable.read_bytes()).hexdigest()
        if solver.executable_sha256 is not None and identity != solver.executable_sha256:
            raise ValueError("The LKH executable does not match its frozen identity")
        work = output / "lkh"
        work.mkdir()
        binary = work / "solver"
        shutil.copy2(executable, binary)
        source = pack(root, output / "source")
        environment = capture_provenance(root)
        environment.update(
            command=sys.argv,
            backend="standalone_lkh_cpu",
            executable_sha256=identity,
            source_archive_sha256=source["archive_sha256"],
        )
        dump_json(output / "environment.json", environment)
        dump_json(
            output / "config.json",
            {
                "schema": SCHEMA,
                "input": config,
                "resolved": {
                    "solver": asdict(solver),
                    "max_trials": solver.max_trials or instance.dimension,
                    "instance_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "executable_sha256": identity,
                    "target_length": target,
                    "reference_optimum_sent_to_solver": False,
                    "reference_tour_loaded": False,
                    "observation_boundary": "returned complete tours after native process exit",
                },
            },
        )
        prepared = perf_counter()
        stop_reason = "restart_limit"
        with (output / "events.jsonl").open("x", buffering=1) as journal:
            for index in range(solver.restarts):
                remaining = stop - perf_counter()
                if remaining <= 0:
                    stop_reason = "wall_clock_limit"
                    break
                seed = 1 + (solver.seed + index) % (2**31 - 2)
                parameter = work / f"run-{index}.par"
                tour_path = work / f"run-{index}.tour"
                parameter.write_text(
                    "PROBLEM_FILE = ../input.tsp\n"
                    f"OUTPUT_TOUR_FILE = {tour_path.name}\n"
                    f"RUNS = 1\nMAX_TRIALS = {solver.max_trials or instance.dimension}\n"
                    f"SEED = {seed}\nPRECISION = {solver.precision}\n"
                    f"TIME_LIMIT = {remaining * 0.9:.9f}\nTRACE_LEVEL = 1\n",
                    encoding="utf-8",
                )
                native_started = perf_counter()
                timed_out, return_code = False, None
                with (work / f"run-{index}.log").open("xb") as log:
                    try:
                        completed = subprocess.run(
                            [str(binary), parameter.name],
                            cwd=work,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=max(0.001, stop - perf_counter()),
                            check=False,
                        )
                        return_code = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                event = {
                    "event_index": index,
                    "stage": "lkh_returned_tour",
                    "seed": seed,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "native_process_seconds": perf_counter() - native_started,
                }
                if tour_path.exists():
                    tour = _read_lkh_tour(instance, tour_path)
                    certificate = certify_tour(instance, tour)
                    if not certificate.is_permutation or certificate.length is None:
                        raise ValueError("LKH returned an invalid Hamiltonian tour")
                    event.update(tour=tour, length=certificate.length)
                    if best is None or certificate.length < best["length"]:
                        best = {"tour": tour, "length": certificate.length}
                    if first_hit is None and target is not None and certificate.length <= target:
                        now = perf_counter()
                        first_hit = {
                            **best,
                            "event_index": index,
                            "stage": event["stage"],
                            "certified_elapsed_seconds": now - started,
                            "certified_monotonic_seconds": now,
                            "search_to_certificate_seconds": now - prepared,
                        }
                event["elapsed_seconds"] = perf_counter() - started
                observations.append(event)
                journal.write(json.dumps(event) + "\n")
                if return_code not in (None, 0):
                    raise RuntimeError(f"LKH restart {index} failed; its log is preserved")
                if timed_out:
                    stop_reason = "wall_clock_limit"
                    break
                if not tour_path.exists():
                    raise RuntimeError("LKH completed without producing a complete tour")
        elapsed = perf_counter() - started
        dump_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": stop_reason,
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
                "budget_overrun_seconds": max(0.0, elapsed - solver.max_seconds),
                "completed_restarts": len(observations),
                "native_process_seconds": sum(
                    row["native_process_seconds"] for row in observations
                ),
                "observation_boundary": "returned complete tour; not native internal first passage",
                "offline_compilation_included": False,
                "executable_sha256": identity,
            },
        )
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
                "first_observed_target_hit": first_hit,
                "best_observation": best,
                "completed_observations": observations,
            },
        )
        raise
    return output
