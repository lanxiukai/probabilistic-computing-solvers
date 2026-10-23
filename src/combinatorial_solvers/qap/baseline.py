"""Independent single-thread attribute-tabu comparison with original-state checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np


def tabu_reference(instance, instance_file, config, output: Path):
    seed = config.get("seed", 2026102301)
    initial = np.random.Generator(np.random.PCG64(seed)).permutation(instance.size)
    initial_path = output / "tabu-initial.txt"
    initial_path.write_text(" ".join(map(str, initial)) + "\n")
    max_seconds = config.get("max_seconds", 10)
    if not 0 < max_seconds < float("inf"):
        raise ValueError("Reference search time must be positive and finite")
    command = [
        str(Path(config["executable"]).resolve()),
        str(Path(instance_file).resolve()),
        str(max_seconds),
        str(seed),
        str(config.get("max_iterations", 100000000)),
        str(int(config.get("verify_caches", False))),
        str(initial_path.resolve()),
    ]
    before = perf_counter()
    raw = output / "tabu-result.json"
    with raw.open("w") as stdout, (output / "tabu-stderr.log").open("w") as stderr:
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=config.get("process_limit_seconds", 120),
            check=False,
        )
    if process.returncode:
        raise RuntimeError("Attribute-tabu reference failed; process evidence is retained")
    record = json.loads(raw.read_text())
    arrays = {
        key: np.asarray(record[key], dtype=np.int64)
        for key in (
            "initial_permutation",
            "best_permutation",
            "final_permutation",
            "curve_cost",
            "curve_iteration",
            "curve_permutations",
        )
    }
    arrays["curve_seconds"] = np.asarray(record["curve_seconds"])
    arrays["initial_cost"] = np.array(record["curve_cost"][0], dtype=np.int64)
    arrays["best_cost"] = np.array(record["best_cost"], dtype=np.int64)
    arrays["final_cost"] = np.array(record["final_cost"], dtype=np.int64)
    arrays["final_rng"] = np.frombuffer(record["final_rng"].encode(), dtype=np.uint8)
    if not np.array_equal(initial, arrays["initial_permutation"]):
        raise AssertionError("Reference changed the declared seeded initial permutation")
    for key, cost in (
        ("best_permutation", record["best_cost"]),
        ("final_permutation", record["final_cost"]),
    ):
        cert = instance.certificate(arrays[key])
        if not cert["feasible"] or cert["cost"] != cost:
            raise AssertionError("Reference permutation failed original integer certification")
    for p, cost in zip(arrays["curve_permutations"], arrays["curve_cost"], strict=True):
        if instance.certificate(p)["cost"] != int(cost):
            raise AssertionError("Reference improvement history failed independent certification")
    eligible = arrays["curve_seconds"] <= max_seconds
    if config.get("quality_max_iteration") is not None:
        eligible &= arrays["curve_iteration"] <= config["quality_max_iteration"]
    if config.get("quality_max_curve_index") is not None:
        eligible &= np.arange(len(eligible)) <= config["quality_max_curve_index"]
    eligible_index = int(np.flatnonzero(eligible)[-1])
    quality_cost = int(arrays["curve_cost"][eligible_index])
    arrays["quality_best_permutation"] = arrays["curve_permutations"][eligible_index].copy()
    arrays["quality_best_cost"] = np.array(quality_cost)
    arrays["curve_eligible"] = eligible
    return arrays, {
        "method": "attribute-tabu",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "initial_best_cost": int(arrays["curve_cost"][0]),
        "best_feasible_cost": quality_cost,
        "returned_best_cost": record["best_cost"],
        "quality_max_iteration": int(arrays["curve_iteration"][eligible_index]),
        "quality_max_curve_index": eligible_index,
        "search_seconds": record["search_seconds"],
        "deadline_overshoot_seconds": max(0, record["search_seconds"] - max_seconds),
        "baseline_complete_seconds": perf_counter() - before,
        "iterations": record["iterations"],
        "restarts": record["restarts"],
        "parallel_workers": 1,
        "independent_trial_unit": "one seeded single-thread search",
        "observation": (
            "Every improved original integer state; improvements observed after the search "
            "deadline or replay quality fence are excluded."
        ),
        "implementation_scope": (
            "Independent attribute-tabu implementation inspired by Taillard; "
            "not an upstream Ro-TS reproduction"
        ),
    }
