"""Independent single-thread TabuCol-style comparison with full edge verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np


def tabu_reference(instance, instance_file, config, output: Path):
    seed = config.get("seed", 2026102301)
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = np.array([rng.choice(np.flatnonzero(row)) for row in instance.allowed])
    initial_path = output / "tabu-initial.txt"
    initial_path.write_text(" ".join(map(str, initial)) + "\n")
    allowed_path = output / "tabu-allowed.txt"
    allowed_path.write_text(" ".join(map(str, instance.allowed.astype(int).ravel())) + "\n")
    command = [
        str(Path(config["executable"]).resolve()),
        str(Path(instance_file).resolve()),
        str(instance.colors),
        str(config.get("max_seconds", 10)),
        str(seed),
        str(config.get("max_iterations", 100000000)),
        str(int(config.get("verify_caches", False))),
        str(initial_path.resolve()),
        str(allowed_path.resolve()),
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
        raise RuntimeError("TabuCol-style reference failed; process evidence is retained")
    record = json.loads(raw.read_text())
    arrays = {
        key: np.asarray(record[key], dtype=np.int64)
        for key in (
            "initial_colors",
            "best_colors",
            "returned_best_colors",
            "final_colors",
            "curve_cost",
            "curve_iteration",
            "curve_colors",
            "curve_eligible",
        )
    }
    arrays["curve_seconds"] = np.asarray(record["curve_seconds"])
    arrays["final_rng"] = np.frombuffer(record["final_rng"].encode(), dtype=np.uint8)
    for key, cost in (
        ("best_colors", record["best_cost"]),
        ("returned_best_colors", record["returned_best_cost"]),
        ("final_colors", record["final_cost"]),
    ):
        cert = instance.certificate(arrays[key])
        if not cert["valid_assignment"] or cert["conflicts"] != cost:
            raise AssertionError("Reference coloring failed independent edge certification")
    for c, cost in zip(arrays["curve_colors"], arrays["curve_cost"], strict=True):
        if instance.certificate(c)["conflicts"] != int(cost):
            raise AssertionError("Reference improvement history failed independent certification")
    return arrays, {
        "method": "tabucol-style",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "best_feasible_cost": record["best_cost"],
        "returned_best_cost": record["returned_best_cost"],
        "search_seconds": record["search_seconds"],
        "baseline_complete_seconds": perf_counter() - before,
        "iterations": record["iterations"],
        "parallel_workers": 1,
        "quality_iteration": record["quality_iteration"],
        "deadline_overshoot_seconds": max(
            0, record["search_seconds"] - config.get("max_seconds", 10)
        ),
        "observation": (
            "Every completed move; a move observed beyond the deadline is excluded "
            "from best_feasible_cost and retained in returned_best_cost."
        ),
        "implementation_scope": (
            "Independent TabuCol-style conflict search; not an upstream HEAD reproduction"
        ),
    }
