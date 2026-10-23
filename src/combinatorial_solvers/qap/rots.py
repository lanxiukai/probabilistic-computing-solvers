"""Measured external Taillard Ro-TS, separate from this project's own solver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

UPSTREAM_DEFAULT_RNG = (12345, 67890, 13579, 24680, 98765, 43210)


def rng_words(seed):
    generator = np.random.Generator(np.random.PCG64(seed))
    return [int(generator.integers(1, 2147483647 if i < 3 else 2145483479)) for i in range(6)]


def arithmetic_bounds(instance):
    first = sorted(map(int, instance.flow.ravel()))
    second = sorted(map(int, instance.distance.ravel()))
    product = max(first) * max(second)
    upper = sum(a * b for a, b in zip(first, second, strict=True))
    direct_delta = 2 * (instance.size - 1) * product
    if (
        max(max(first), max(second)) >= 2**30
        or max(upper, upper + 8 * product) >= 2**31
        or direct_delta >= 999999999
        or instance.size > 2048
    ):
        raise ValueError("Input exceeds the external upstream int32 or move-sentinel envelope")
    return {
        "objective_rearrangement_upper_bound": upper,
        "delta_direct_term_abs_sum_bound": direct_delta,
        "delta_increment_intermediate_abs_bound": upper + 8 * product,
    }


def rots_reference(instance, instance_file, config, output: Path):
    bounds = arithmetic_bounds(instance)
    seed = config.get("seed", 2026102301)
    words = config.get("rng_words", rng_words(seed))
    seconds = config.get("max_seconds", 30)
    iterations = config.get("max_iterations", 100000000)
    if not 0 < seconds < float("inf") or not 1 <= iterations <= 100000000:
        raise ValueError("External reference limits are invalid")
    command = [
        str(Path(config["executable"]).resolve()),
        str(Path(instance_file).resolve()),
        str(seconds),
        str(iterations),
        ",".join(map(str, words)),
        str(output.resolve()),
        str(int(config.get("verify_caches", False))),
    ]
    started = perf_counter()
    raw_path = output / "rots-result.json"
    with raw_path.open("w") as stdout, (output / "rots-stderr.log").open("w") as stderr:
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=config.get("process_limit_seconds", seconds + 30),
            check=False,
        )
    if process.returncode:
        raise RuntimeError("External Ro-TS process failed; all partial evidence is retained")
    raw = json.loads(raw_path.read_text())
    with (output / "observations.jsonl").open() as stream:
        observations = [json.loads(line) for line in stream]
    arrays = {
        key: np.asarray(raw[key], dtype=np.int64)
        for key in (
            "initial_permutation",
            "final_permutation",
            "best_permutation",
            "initial_cost",
            "final_cost",
            "best_cost",
            "initial_rng",
            "search_initial_rng",
            "final_rng",
            "rng_draws",
        )
    }
    arrays["next_rng_values"] = np.asarray(raw["next_rng_values"], dtype=np.float64)
    arrays["curve_permutations"] = np.asarray(
        [record["permutation"] for record in observations], dtype=np.int64
    )
    arrays["curve_cost"] = np.asarray([record["cost"] for record in observations], dtype=np.int64)
    arrays["curve_iteration"] = np.asarray(
        [record["iteration"] for record in observations], dtype=np.int64
    )
    arrays["curve_seconds"] = np.asarray([record["seconds"] for record in observations])
    if not np.array_equal(arrays["initial_rng"], words):
        raise AssertionError("External reference changed the declared initial RNG words")
    for permutation, cost in zip(arrays["curve_permutations"], arrays["curve_cost"], strict=True):
        if instance.certificate(permutation) != {"feasible": True, "cost": int(cost)}:
            raise AssertionError("External reference improvement failed original certification")
    for key, cost_key in (
        ("initial_permutation", "initial_cost"),
        ("final_permutation", "final_cost"),
        ("best_permutation", "best_cost"),
    ):
        if instance.certificate(arrays[key]) != {"feasible": True, "cost": int(arrays[cost_key])}:
            raise AssertionError("External reference endpoint failed original certification")
    eligible = arrays["curve_seconds"] <= seconds
    if config.get("quality_max_iteration") is not None:
        eligible &= arrays["curve_iteration"] <= config["quality_max_iteration"]
    if config.get("quality_max_curve_index") is not None:
        eligible &= np.arange(len(eligible)) <= config["quality_max_curve_index"]
    index = int(np.flatnonzero(eligible)[-1])
    arrays["curve_eligible"] = eligible
    arrays["quality_best_permutation"] = arrays["curve_permutations"][index].copy()
    arrays["quality_best_cost"] = arrays["curve_cost"][index].copy()
    if raw["signal"]:
        np.savez_compressed(output / "partial-states.npz", **arrays)
        raise InterruptedError("External Ro-TS stopped at a committed iteration after a signal")
    return arrays, {
        "method": "taillard-rots",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "initial_best_cost": int(arrays["initial_cost"]),
        "best_feasible_cost": int(arrays["quality_best_cost"]),
        "returned_best_cost": int(arrays["best_cost"]),
        "quality_max_iteration": int(arrays["curve_iteration"][index]),
        "quality_max_curve_index": index,
        "iterations": raw["iterations"],
        "search_seconds": raw["search_seconds"],
        "preparation_seconds": raw["preparation_seconds"],
        "deadline_overshoot_seconds": max(0, raw["search_seconds"] - seconds),
        "baseline_complete_seconds": perf_counter() - started,
        "parallel_workers": 1,
        "independent_trial_unit": "one complete seeded external single-thread search",
        "reference_used_for_stopping_or_initialization": False,
        "arithmetic_bounds": bounds,
        "upstream_version": "Taillard standard C Ro-TS dated 2006-03-14",
        "upstream_conditions": (
            "Non-commercial use; acknowledge E. Taillard; not this project's MIT"
        ),
        "observation": (
            "Original integer best states after completed iterations; the initial state is "
            "recorded at time zero. Deadline-crossing improvements are retained and excluded "
            "from quality. Periodic full tabu/RNG checkpoints are included in search time."
        ),
    }
