"""Execute the separately budgeted RTX 5090 correctness and capture-cost gate."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sparse_factorization.replay import compare_arrays
from sparse_factorization.repro import execute, prepare, source_identity
from sparse_factorization.solver.publication_io import atomic_json, canonical_digest, read_json
from sparse_factorization.stream import arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    plan, spec = read_json(args.plan), read_json(args.spec)
    if spec["plan_sha256"] != canonical_digest(plan):
        raise ValueError("Preflight specification and frozen plan differ")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != plan["hardware"]["gpu"]:
        raise RuntimeError("The final device gate must run on an actual RTX 5090")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "definition.py")
    atomic_json(args.output / "plan.json", plan)
    atomic_json(args.output / "spec.json", spec)
    start = time.perf_counter()
    code = source_identity(project)
    prepare(project, spec["capture_runs"][0], args.output / "source-audit")
    with (args.output / "software-checks.log").open("x") as log:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=project,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=spec["software_wall_cap_seconds"],
        )
    results = {}
    tasks = [*spec["capture_runs"], *plan["cost_runs"]]
    for config in tasks:
        maximum = config["maximum_sampling_seconds"] + config["task_overhead_seconds"]
        if time.perf_counter() - start + maximum > spec["maximum_total_wall_seconds"]:
            raise TimeoutError("Insufficient remaining preflight budget for the next frozen run")
        if source_identity(project) != code:
            raise ValueError("Source changed during the device gate")
        destination = args.output / config["run_id"]
        prepare(project, config, destination)
        if execute(destination):
            raise RuntimeError("A preflight worker failed; preserve evidence and close the gate")
        result = read_json(destination / "result/summary.json")
        if result["events"] != config["maximum_events"] or result["status"] != "event_limit":
            raise ValueError("Fixed-work preflight did not complete its frozen event count")
        results[config["run_id"]] = result
    for flat, chunked in spec["capture_pairs"]:
        checkpoints = []
        histories = []
        for name in (flat, chunked):
            paths = sorted((args.output / name / "result/stream/chunks").glob("chunk-*"))
            checkpoints.append(arrays(paths[-1] / "checkpoint.npz"))
            histories.append(
                np.concatenate([arrays(p / "metrics.npz")["cumulative_by_position"] for p in paths])
            )
        compare_arrays(*checkpoints)
        np.testing.assert_array_equal(*histories)
    flat_seconds = sum(results[a]["sampling_seconds"] for a, _ in spec["capture_pairs"])
    chunk_seconds = sum(results[b]["sampling_seconds"] for _, b in spec["capture_pairs"])
    ratio = chunk_seconds / flat_seconds
    result = {
        "passed": ratio <= spec["maximum_capture_ratio"],
        "capture_sampling_ratio": ratio,
        "source_sha256": code["sha256"],
        "plan_sha256": canonical_digest(plan),
        "device": torch.cuda.get_device_name(0),
        "wall_seconds": time.perf_counter() - start,
        "new_independent_trials": 0,
        "results": results,
    }
    atomic_json(args.output / "summary.json", result)
    if not result["passed"]:
        raise RuntimeError("Capture overhead gate failed; do not start the main panel")


if __name__ == "__main__":
    main()
