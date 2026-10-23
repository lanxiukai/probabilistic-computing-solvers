"""File protocol for archived reference decoders; no access to transmitted truth."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from time import perf_counter

import numpy as np

from combinatorial_solvers.workflows.runtime import dump_json

from .model import SCALE


def run(instance, config, output):
    started = perf_counter()
    path = output / "reference-input.bin"
    with path.open("xb") as handle:
        handle.write(b"LDPCCS01")
        handle.write(
            struct.pack("<4I", instance.code.size, instance.code.checks, instance.frames, SCALE)
        )
        for row in instance.code.rows:
            handle.write(struct.pack("<I", len(row)))
            handle.write(row.astype("<u4").tobytes())
        for costs, target in zip(instance.costs, instance.targets, strict=True):
            handle.write(costs.astype("<u4").tobytes())
            handle.write(target.astype(np.uint8).tobytes())
    records_path = output / "reference-records.jsonl"
    command = [
        str(config["executable"]),
        str(path),
        str(records_path),
        config["mode"],
        str(config["iterations"]),
        str(config["alpha"]),
        config["schedule"],
    ]
    dump_json(output / "reference-command.json", command)
    before = perf_counter()
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        subprocess.run(
            command,
            cwd=output,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            timeout=config["max_seconds"],
            check=True,
        )
    child_seconds = perf_counter() - before
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if [r["frame"] for r in records] != list(range(instance.frames)):
        raise AssertionError("Reference frame coverage differs")

    def bits(key):
        return np.asarray([[int(x) for x in r[key]] for r in records], dtype=np.int8)

    selected, bp = bits("selected_bits"), bits("bp_bits")
    cost, violations = instance.cost(selected), instance.violations(selected)
    if not np.array_equal(cost, [r["cost"] for r in records]) or not np.array_equal(
        violations, [r["violations"] for r in records]
    ):
        raise AssertionError("Reference decoder certificate differs")
    arrays = {
        "initial_bits": np.zeros((instance.frames, 1, instance.code.size), dtype=np.int8),
        "best_bits": selected[:, None, :],
        "final_bits": selected[:, None, :],
        "best_cost": cost[:, None],
        "best_violations": violations[:, None],
        "final_cost": cost[:, None],
        "final_violations": violations[:, None],
        "decoded_error": selected,
        "bp_bits": bp,
        "bp_cost": instance.cost(bp),
        "bp_violations": instance.violations(bp),
        "posterior": np.asarray([r["posterior"] for r in records]),
        "iterations": np.asarray([r["iterations"] for r in records]),
        "post_called": np.asarray([r["post_called"] for r in records]),
        "frame_decoder_seconds": np.asarray([r["decoder_seconds"] for r in records]),
    }
    summary = {
        "method": f"reference-{config['mode']}-{config['schedule']}",
        "frames": instance.frames,
        "replicas": 1,
        "cpu_threads": 1,
        "valid_frames": int(np.count_nonzero(violations == 0)),
        "feasible_cost_sum": int(cost[violations == 0].sum()),
        "physical_update_pbits_per_model": 0,
        "auxiliary_pbits": 0,
        "reference_child_seconds": child_seconds,
        "search_seconds": sum(r["decoder_seconds"] for r in records),
        "adapter_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": 0,
    }
    return arrays, summary
