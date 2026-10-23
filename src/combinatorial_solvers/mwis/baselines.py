"""Deterministic greedy and an isolated, externally versioned CHILS baseline."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from .model import Instance


def greedy(instance: Instance):
    adjacency = instance.adjacency()
    order = sorted(
        range(instance.size), key=lambda i: (-int(instance.weights[i]) / (1 + len(adjacency[i])), i)
    )
    bits = np.zeros(instance.size, dtype=np.uint8)
    blocked = np.zeros(instance.size, dtype=bool)
    for node in order:
        if not blocked[node]:
            bits[node] = 1
            blocked[node] = True
            blocked[adjacency[node]] = True
    return bits


def chils(instance: Instance, config: dict, output: Path):
    binary = Path(config["executable"]).resolve(strict=True)
    seconds = float(config["max_seconds"])
    seed, threads, population = (int(config[k]) for k in ("seed", "threads", "population"))
    if not 0 < seconds <= 600 or min(threads, population) < 1 or not 0 <= seed < 2**31:
        raise ValueError("Invalid external baseline budget or settings")
    input_path, solution_path = output / "instance.graph", output / "solution.txt"
    instance.write_metis(input_path)
    command = [
        str(binary),
        "-g",
        str(input_path),
        "-o",
        str(solution_path),
        "-t",
        str(seconds),
        "-r",
        str(seed),
        "-c",
        str(threads),
        "-p",
        str(population),
        "-b",
    ]
    if "iterations" in config:
        command.extend(["-m", str(int(config["iterations"]))])
    if "exchange_seconds" in config:
        command.extend(["-s", str(float(config["exchange_seconds"]))])
    (output / "baseline-command.json").write_text(
        json.dumps({"command": command, "threads": threads}, indent=2) + "\n"
    )
    started = perf_counter()
    with (output / "stdout.txt").open("x") as stdout, (output / "stderr.txt").open("x") as stderr:
        result = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
            timeout=seconds + 30,
            check=False,
        )
    elapsed = perf_counter() - started
    if result.returncode != 0 or not solution_path.is_file():
        raise RuntimeError(f"CHILS did not produce a solution: exit={result.returncode}")
    chosen = [int(x) - 1 for x in solution_path.read_text().split()]
    if len(set(chosen)) != len(chosen) or any(x < 0 or x >= instance.size for x in chosen):
        raise ValueError("CHILS output contains invalid vertices")
    bits = np.zeros(instance.size, dtype=np.uint8)
    bits[chosen] = 1
    certificate = instance.certificate(bits)
    if not certificate["feasible"]:
        raise AssertionError("CHILS returned a conflicting assignment")
    return bits, {
        "method": "chils",
        "certificate": certificate,
        "process_seconds": elapsed,
        "requested_search_seconds": seconds,
        "threads": threads,
        "population": population,
        "timing_scope": (
            "Complete external process, including input parsing and output; "
            "requested search budget separately recorded"
        ),
    }
