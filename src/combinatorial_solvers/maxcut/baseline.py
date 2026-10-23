"""MQLib callback executable, immutable raw output, and integer certification."""

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
    seconds = float(config["max_seconds"])
    seed = int(config["seed"])
    if not 0 < seconds <= 600 or not 0 <= seed < 2**32:
        raise ValueError("Invalid baseline budget or seed")
    filename = output / "instance.mc"
    instance.write_mqlib(filename)
    command = [
        str(binary),
        config["algorithm"],
        str(filename),
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
                        if perf_counter() - started > seconds + 90:
                            raise TimeoutError("MQLib exceeded its callback return allowance")
                        for event, _ in selector.select(timeout=0.05):
                            chunk = os.read(event.fd, 65536)
                            if not chunk:
                                selector.unregister(event.fileobj)
                                continue
                            stdout.write(chunk)
                            tail += chunk
                            if ready is None and b"c benchmark-search-ready\n" in tail:
                                ready = perf_counter()
                                process.stdin.write(b"g\n")
                                process.stdin.flush()
                            tail = tail[-128:]
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
    raw = (output / "stdout.txt").read_text()
    if code != 0 or ready is None:
        raise RuntimeError(f"MQLib returned no certificate: exit={code}")
    arrays = {}
    for key, label in (("best_bits", "v"), ("final_bits", "f")):
        models = re.findall(rf"^{label} ([01]+)$", raw, re.M)
        if len(models) != 1:
            raise ValueError("Missing or ambiguous baseline witness")
        arrays[key] = np.asarray([int(c) for c in models[0]], dtype=np.uint8)

    def integer(label):
        return int(re.findall(rf"^c {label} (-?\d+)$", raw, re.M)[-1])

    cut, final = int(instance.cut(arrays["best_bits"])), int(instance.cut(arrays["final_bits"]))
    if cut != integer("best-cut") or final != integer("final-cut"):
        raise AssertionError("MQLib objective differs from its integer witness")
    return arrays, {
        "method": config["algorithm"],
        "best_cut": cut,
        "final_cut": final,
        "completed_calls": integer("benchmark-calls"),
        "first_stop_call": integer("first-stop-call"),
        "quality_calls": integer("quality-calls"),
        "process_seconds": ended - started,
        "preparation_seconds": ready - started,
        "search_seconds": ended - ready,
        "requested_search_seconds": seconds,
        "timing_scope": (
            "Graph read excluded from search; callback accepts only in-budget reports; "
            "return and certificate costs separate"
        ),
    }
