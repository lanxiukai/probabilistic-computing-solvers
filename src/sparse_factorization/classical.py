"""Isolated fixed-seed YAFU CPU controls, including timeout evidence."""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .solver.primes import is_prime
from .solver.publication_io import atomic_json, digest, read_json


def parse_factors(text: str, target: int) -> list[int]:
    values = [
        int(value)
        for value in re.findall(r"^\s*(?:P|PRP)\d+\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    ]
    if (
        not values
        or math.prod(values) != target
        or not all(1 < v < 2**64 and is_prime(v) for v in values)
    ):
        raise ValueError("YAFU output does not certify the complete expected factorization")
    return sorted(values)


def run_yafu(
    binary: Path,
    target: int,
    output: Path,
    *,
    seed: int,
    max_seconds=300,
    library_directory: Path | None = None,
) -> dict:
    binary = binary.resolve(strict=True)
    build = read_json(binary.parent / "build.json")
    if build["revision"] != "c7d90283972820f99b3b4920bfc1eff05c1f6342" or build[
        "binary_sha256"
    ] != digest(binary):
        raise ValueError("Classical binary differs from its pinned build receipt")
    library_directory = library_directory or binary.parent / "lib"
    if any(digest(library_directory / name) != sha for name, sha in build["libraries"].items()):
        raise ValueError("Classical runtime libraries differ from the build receipt")
    if not isinstance(target, int) or target <= 1 or not 0 < seed < 2**64 or max_seconds <= 0:
        raise ValueError("Invalid classical control input or budget")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary),
        f"factor({target})",
        "-threads",
        "1",
        "-plan",
        "noecm",
        "-seed",
        str(seed),
    ]
    atomic_json(
        output / "config.json",
        {
            "target": target,
            "seed": seed,
            "threads": 1,
            "plan": "noecm",
            "wall_cap_seconds": max_seconds,
            "binary_sha256": digest(binary),
            "source_revision": "c7d90283972820f99b3b4920bfc1eff05c1f6342",
        },
    )
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if library_directory is not None:
        environment["LD_LIBRARY_PATH"] = str(library_directory.resolve())
    start = time.perf_counter()
    timed_out = False
    (output / "input.txt").write_text(f"factor({target})\n")
    with (output / "stdout.log").open("x") as log, (output / "input.txt").open() as input_stream:
        with subprocess.Popen(
            command,
            cwd=output,
            stdin=input_stream,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        ) as process:
            try:
                code = process.wait(timeout=max_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
    elapsed = time.perf_counter() - start
    factors = []
    parse_error = None
    if not timed_out and code == 0:
        try:
            text = (output / "stdout.log").read_text(errors="replace")
            accepted_seed = re.search(r"^read:\s*(\d+)\s*$", text, flags=re.MULTILINE)
            if accepted_seed is None or int(accepted_seed[1]) != seed:
                raise ValueError("YAFU did not consume the frozen uint64 seed")
            factors = parse_factors(text, target)
        except ValueError as error:
            parse_error = str(error)
    result = {
        "target": target,
        "arm": "yafu_cpu_noecm",
        "success": bool(factors),
        "status": "timeout" if timed_out else "completed" if factors else "failed",
        "returncode": code,
        "wall_seconds": elapsed,
        "factors": factors,
        "success_criterion": "complete integer factorization; no Ising certificate",
        "independent_physical_trials": 0,
        "certificate_error": parse_error,
    }
    atomic_json(output / "summary.json", result)
    return result
