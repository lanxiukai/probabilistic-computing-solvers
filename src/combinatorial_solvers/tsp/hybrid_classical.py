"""Bounded classical prefixes for the TSP hybrid, including an optional LKH adapter.

LKH parameters follow SRC/ReadParameters.c (PROBLEM_FILE, OUTPUT_TOUR_FILE,
RUNS, MAX_TRIALS, SEED, PRECISION, and TIME_LIMIT). No reference optimum or
reference solution is passed to the external solver. EAX or other solvers can
provide the same instance-bound JSON elite packet used by saved prefixes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter

from .baselines import nearest_neighbor_tsp
from .hybrid import (
    BudgetExpired,
    Deadline,
    ElitePool,
    EventSink,
    HybridConfig,
    perturb_tour,
    polish_tour,
)
from .hybrid_backend import LocalSearchBackend
from .hybrid_moves import Tour, canonical_tour
from .instance import TSPInstance, tour_length

PREFIX_SCHEMA = "tsp-hybrid-prefix-v1"


@dataclass(frozen=True)
class PrefixConfig:
    kind: str = "ils"
    starts: int = 8
    iterations: int = 32
    time_fraction: float = 0.25
    lkh_executable: str | None = None
    lkh_max_trials: int = 1000
    lkh_precision: int = 1
    tours_file: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"ils", "lkh", "provided"}:
            raise ValueError("prefix.kind must be ils, lkh, or provided")
        for name in ("starts", "lkh_max_trials", "lkh_precision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"prefix.{name} must be a positive integer")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise ValueError("prefix.iterations must be a nonnegative integer")
        if not math.isfinite(self.time_fraction) or not 0 < self.time_fraction < 1:
            raise ValueError("prefix.time_fraction must lie strictly between zero and one")
        if (self.kind == "lkh") != (self.lkh_executable is not None):
            raise ValueError("Set lkh_executable exactly when prefix.kind is lkh")
        if (self.kind == "provided") != (self.tours_file is not None):
            raise ValueError("Set tours_file exactly when prefix.kind is provided")


def _read_lkh_tour(instance: TSPInstance, path: Path) -> Tour:
    """Read a complete one-based TSPLIB TOUR, never an argmax or partial path."""
    dimension = None
    active = ended = False
    cities = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.replace(":", " ").split()[-1])
        elif line == "TOUR_SECTION":
            active = True
        elif active and not ended:
            for token in line.split():
                if token == "-1":
                    ended = True
                    break
                if token == "EOF":
                    raise ValueError("An LKH tour ended without the TOUR_SECTION terminator")
                cities.append(int(token) - 1)
    if not active or not ended or dimension != instance.dimension:
        raise ValueError("LKH did not write a complete tour with the expected dimension")
    return canonical_tour(instance, cities)


def _write_explicit_tsp(instance: TSPInstance, path: Path, deadline: Deadline) -> None:
    if int(instance.distance.max()) > 2**31 - 1:
        raise ValueError("LKH explicit edge weights must fit its signed 32-bit input format")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "NAME: hybrid_input\nTYPE: TSP\n"
            f"DIMENSION: {instance.dimension}\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n"
        )
        for row in instance.distance:
            deadline.check()
            stream.write(" ".join(str(int(value)) for value in row) + "\n")
        stream.write("EOF\n")


def build_prefix(
    instance: TSPInstance,
    instance_sha256: str,
    prefix: PrefixConfig,
    solver: HybridConfig,
    output: Path,
    deadline: Deadline,
    emit: EventSink,
    backend: LocalSearchBackend | None = None,
) -> tuple[ElitePool, dict]:
    """Create or load elites, with independent random state and measured prefix cost."""
    started = perf_counter()
    local = Deadline(min(deadline.stop, started + deadline.remaining() * prefix.time_fraction))
    pool = ElitePool(instance, solver.elite_size)
    rng = Random(solver.seed)
    metadata: dict = {"kind": prefix.kind, "seed": solver.seed, "stop_reason": "prefix_complete"}

    def observe(tour: Tour, **details) -> None:
        canonical, length = pool.add(tour)
        emit(
            {
                "stage": "prefix_candidate",
                "tour": canonical,
                "length": length,
                "elite_pool": pool.payload(),
                **details,
            }
        )

    try:
        local.check()
        if prefix.kind == "provided":
            source = Path(str(prefix.tours_file)).resolve()
            data = source.read_bytes()
            (output / "provided-prefix.json").write_bytes(data)
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("Provided prefix must be a JSON object")
            if (
                payload.get("schema") != PREFIX_SCHEMA
                or payload.get("instance_sha256") != instance_sha256
            ):
                raise ValueError(
                    "Provided elites must use the prefix schema and match the exact instance"
                )
            rows = payload.get("elites")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Provided prefix contains no elite tours")
            upstream = payload.get("prefix_cost_seconds")
            if upstream is not None and (
                isinstance(upstream, bool) or not math.isfinite(upstream) or upstream < 0
            ):
                raise ValueError("Provided prefix cost must be finite and nonnegative")
            metadata.update(
                source_path=str(source),
                source_sha256=hashlib.sha256(data).hexdigest(),
                upstream_prefix_seconds=upstream,
                upstream_cost_is_declared=True,
            )
            for row in rows:
                local.check()
                canonical = canonical_tour(instance, row["tour"])
                claimed = row["length"]
                if isinstance(claimed, bool) or not isinstance(claimed, int):
                    raise ValueError("Provided tour lengths must be integers")
                if tour_length(instance, canonical) != claimed:
                    raise ValueError("Provided prefix contains an incorrect claimed tour length")
                observe(canonical, origin="provided")
        elif prefix.kind == "lkh":
            executable = Path(str(prefix.lkh_executable)).expanduser().resolve()
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError("prefix.lkh_executable must be an existing executable file")
            work = output / "lkh"
            work.mkdir()
            binary = work / "solver"
            shutil.copy2(executable, binary)
            metadata.update(
                executable_path=str(executable),
                executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                executable_snapshot="lkh/solver",
            )
            _write_explicit_tsp(instance, work / "instance.tsp", local)
            for index in range(prefix.starts):
                local.check()
                allotted = local.remaining() / (prefix.starts - index)
                seed = 1 + (solver.seed + index) % (2**31 - 2)
                parameter = work / f"run-{index}.par"
                tour_path = work / f"run-{index}.tour"
                parameter.write_text(
                    "PROBLEM_FILE = instance.tsp\n"
                    f"OUTPUT_TOUR_FILE = {tour_path.name}\n"
                    f"RUNS = 1\nMAX_TRIALS = {prefix.lkh_max_trials}\n"
                    f"SEED = {seed}\nPRECISION = {prefix.lkh_precision}\n"
                    f"TIME_LIMIT = {allotted * 0.9:.9f}\n"
                    "TRACE_LEVEL = 1\n",
                    encoding="utf-8",
                )
                run_started = perf_counter()
                timed_out = False
                return_code = None
                with (work / f"run-{index}.log").open("wb") as log:
                    try:
                        completed = subprocess.run(
                            [str(binary.resolve()), parameter.name],
                            cwd=work,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=min(allotted, local.remaining()),
                            check=False,
                        )
                        return_code = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                emit(
                    {
                        "stage": "lkh_run",
                        "run": index,
                        "seed": seed,
                        "return_code": return_code,
                        "timed_out": timed_out,
                        "seconds": perf_counter() - run_started,
                    }
                )
                if return_code not in (None, 0):
                    raise RuntimeError(f"LKH run {index} failed; its log is preserved")
                if tour_path.exists():
                    observe(_read_lkh_tour(instance, tour_path), origin="lkh", run=index)
                elif not timed_out:
                    raise RuntimeError("LKH completed without producing a tour")
        else:
            starts = rng.sample(range(instance.dimension), min(prefix.starts, instance.dimension))
            for offset in range(0, len(starts), solver.local_search_batch_size):
                cities = starts[offset : offset + solver.local_search_batch_size]
                initials = []
                for city in cities:
                    local.check()
                    initial = nearest_neighbor_tsp(instance, start_city=city).tour
                    observe(initial, origin="nearest_neighbor", start_city=city)
                    initials.append(initial)
                local.check()
                optimized = (
                    backend.solve(initials, local.remaining)
                    if backend is not None
                    else tuple(
                        polish_tour(
                            instance, tour, solver.local_search_passes, local, solver.local_search
                        )
                        for tour in initials
                    )
                )
                for city, tour in zip(cities, optimized, strict=True):
                    observe(
                        tour,
                        origin="prefix_local_search",
                        method=solver.local_search,
                        start_city=city,
                    )
            for iteration in range(prefix.iterations):
                local.check()
                parent = rng.choice(pool.tours)
                candidate = perturb_tour(parent, rng)
                observe(
                    polish_tour(
                        instance,
                        candidate,
                        solver.local_search_passes,
                        local,
                        solver.local_search,
                        backend,
                    ),
                    origin="prefix_ils",
                    iteration=iteration,
                )
    except BudgetExpired:
        if prefix.kind == "provided":
            raise
        metadata["stop_reason"] = "prefix_wall_clock_limit"
    if not pool.tours:
        raise BudgetExpired("The prefix budget produced no certified elite tour")
    seconds = perf_counter() - started
    upstream = metadata.get("upstream_prefix_seconds")
    prefix_cost = (
        None if prefix.kind == "provided" and upstream is None else seconds + (upstream or 0.0)
    )
    metadata.update(
        prefix_seconds=seconds, prefix_cost_seconds=prefix_cost, rng_state=rng.getstate()
    )
    packet = {
        "schema": PREFIX_SCHEMA,
        "instance_sha256": instance_sha256,
        "elites": pool.payload(),
        **metadata,
    }
    (output / "prefix.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    return pool, metadata
