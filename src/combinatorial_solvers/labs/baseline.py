"""Bounded single-thread lMAts process with explicit integer witnesses."""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np


def run(instance, config, output):
    binary = Path(config["executable"]).resolve(strict=True)
    seconds, seed = float(config["max_seconds"]), int(config["seed"])
    if not 0 < seconds <= 600 or not 0 <= seed < 2**32 or instance.size > 512:
        raise ValueError("Invalid CPU reference length, budget or seed")
    command = [
        str(binary),
        str(instance.size),
        str(seed),
        str(seconds),
        str(config.get("fixed_calls", -1)),
        str(config.get("quality_calls", -1)),
    ]
    (output / "baseline-command.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
    started, ready, tail = perf_counter(), None, b""
    with (
        (output / "stdout.txt").open("xb", buffering=0) as stdout,
        (output / "stderr.txt").open("xb", buffering=0) as stderr,
    ):
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
        ) as process:
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        if perf_counter() - started > seconds + 30:
                            raise TimeoutError("CPU memetic solver exceeded its return allowance")
                        for event, _ in selector.select(timeout=0.05):
                            chunk = os.read(event.fd, 65536)
                            if not chunk:
                                selector.unregister(event.fileobj)
                                continue
                            stdout.write(chunk)
                            tail = (tail + chunk)[-128:]
                            if ready is None and b"c benchmark-search-ready\n" in tail:
                                ready = perf_counter()
                                process.stdin.write(b"g\n")
                                process.stdin.flush()
                    code = process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.stdin.close()
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    ended = perf_counter()
    if code != 0 or ready is None:
        raise RuntimeError(f"CPU memetic solver failed with exit {code}")
    raw = (output / "stdout.txt").read_text()
    arrays = {}
    for name, label in (("best_bits", "v"), ("final_bits", "f")):
        models = re.findall(rf"^{label} ([01]+)$", raw, re.M)
        if len(models) != 1:
            raise ValueError("Ambiguous CPU witness")
        arrays[name] = np.asarray([int(c) for c in models[0]], dtype=np.uint8)

    def integer(label):
        return int(re.findall(rf"^c {label} (-?\d+)$", raw, re.M)[-1])

    before = perf_counter()
    best, final = (int(instance.energy(arrays[name])) for name in ("best_bits", "final_bits"))
    if best != integer("best-energy") or final != integer("final-energy"):
        raise AssertionError("CPU memetic integer certificate disagrees")
    return arrays, {
        "method": "lMAts-corrected",
        "best_energy": best,
        "final_energy": final,
        "merit_factor": instance.size**2 / (2 * best),
        "completed_calls": integer("benchmark-calls"),
        "quality_calls": integer("quality-calls"),
        "iterations": integer("iterations"),
        "evaluations": integer("evaluations"),
        "process_seconds": ended - started,
        "preparation_seconds": ready - started,
        "search_seconds": ended - ready,
        "requested_search_seconds": seconds,
        "certification_seconds": perf_counter() - before,
        "physical_update_pbits_per_model": 0,
        "replicas": 1,
    }
