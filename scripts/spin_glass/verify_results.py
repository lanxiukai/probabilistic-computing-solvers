"""Verify public bytes, configurations, denominators, statistics and integer witnesses."""

import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from cubic_spin_glass.certificates import verify_witnesses


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_attempts(rows, panel):
    definitions = {d["id"]: d for d in panel["cells"]}
    if len(rows) != 272 or len({r["cell"] for r in rows}) != 272 or len(definitions) != 271:
        raise ValueError("Attempt or logical-call denominator is incorrect")
    if sorted(r["execution_order"] for r in rows) != list(range(272)):
        raise ValueError("Attempt order is missing or duplicated")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["logical_cell"]].append(row)
        definition = definitions.get(row["logical_cell"])
        if definition is None:
            raise ValueError("An unplanned logical call entered the table")
        for key in (
            "phase",
            "length",
            "instance_id",
            "family",
            "candidate",
            "seed",
            "solver",
            "instance_file",
        ):
            if row[key] != definition[key]:
                raise ValueError("Observed configuration differs from the frozen panel")
    if set(groups) != set(definitions):
        raise ValueError("A planned logical call is missing")
    successful = {}
    for logical, attempts in groups.items():
        accepted = [r for r in attempts if r["status"] == "verified"]
        if len(accepted) != 1:
            raise ValueError("Each logical call requires exactly one usable result")
        successful[logical] = accepted[0]
        expected = 2 if logical == "confirm-l32-i9-s1-pa" else 1
        if len(attempts) != expected or sorted(r["attempt"] for r in attempts) != list(
            range(1, expected + 1)
        ):
            raise ValueError("Unexpected retry or missing failed attempt")
    failures = [r for r in rows if r["status"] != "verified"]
    if len(failures) != 1 or failures[0]["status"] != "failed":
        raise ValueError("The operational failure must be retained")
    if failures[0]["sampling_seconds"] is not None or failures[0]["usable_state_available"]:
        raise ValueError("Unknown failed sampling progress must not be presented as measured")
    if collections.Counter(r["phase"] for r in successful.values()) != {
        "preflight": 16,
        "calibration": 36,
        "confirmation": 216,
        "recovery": 3,
    }:
        raise ValueError("A required phase is incomplete")
    return successful


def verify(data, project):
    panel_path = project / "configs/spin-glass/s23-panel.json"
    panel = read(panel_path)
    provenance = read(project / "provenance/spin-glass/release-v1.json")
    if digest(panel_path) != provenance["public_panel_sha256"]:
        raise ValueError("Frozen panel changed")
    for row in provenance["numerical_files_byte_identical_to_executed_source"]:
        if digest(project / row["path"]) != row["sha256"]:
            raise ValueError("Runtime source differs from the measured implementation")
    manifest = read(data / "manifest.json")
    for name, expected in manifest["files"].items():
        path = data / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Published evidence bytes changed")
    rows = read(data / "calls.json")["calls"]
    successful = check_attempts(rows, panel)
    selected = {}
    for length in panel["lengths"]:
        for family in panel["families"]:
            scores = []
            for candidate in range(2):
                chosen = [
                    r
                    for r in rows
                    if r["phase"] == "calibration"
                    and (r["length"], r["family"], r["candidate"]) == (length, family, candidate)
                ]
                if len(chosen) != 2:
                    raise ValueError("Calibration coverage is incorrect")
                means = [sum(r["checkpoints"]["60"]) / r["groups"] for r in chosen]
                scores.append((sum(means) / len(means), max(means), candidate))
            selected[f"{length}:{family}"] = min(scores)[2]
    if selected != panel["selected_candidates"]:
        raise ValueError("Development-only selection differs from the frozen winner")
    for row in successful.values():
        if row["phase"] == "confirmation":
            if row["candidate"] != selected[f"{row['length']}:{row['family']}"]:
                raise ValueError("Confirmation used a different candidate")
    receipt = read(data / "verification-receipt.json")
    compared = 0
    for comparison in receipt["full_comparisons"]:
        a, b = successful[comparison["cell"]], successful[comparison["reference"]]
        if a["raw_arrays"] != b["raw_arrays"] or a["rng"] != b["rng"]:
            raise ValueError("Recorded array fingerprints or RNG differ for a required replay")
        compared += len(a["raw_arrays"])
    if compared != 226 or sum(len(r["raw_arrays"]) for r in successful.values()) != 6229:
        raise ValueError("Array fingerprint coverage differs from the recorded audit")
    spec = importlib.util.spec_from_file_location(
        "analysis", Path(__file__).with_name("analyze_results.py")
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    observed, expected = analysis.analyze(data, panel), read(data / "summary.json")
    for key in (
        "panels",
        "calls",
        "bootstrap_seed",
        "bootstrap_samples",
        "sensitivity_l32_without_retried_instance",
    ):
        if observed[key] != expected[key]:
            raise ValueError("Recomputed instance-level statistics differ from the report")
    if not math.isclose(
        observed["all_attempt_complete_call_seconds"],
        expected["operational_accounting"]["all_attempt_complete_call_seconds"],
        abs_tol=1e-8,
    ):
        raise ValueError("Failed-attempt cost is missing from the total")
    solutions = read(data / "best-observed.json")["solutions"]
    if (
        len(solutions) != 108
        or len({(r["length"], r["instance_id"], r["method"]) for r in solutions}) != 108
    ):
        raise ValueError("Portable witness panel is incomplete")
    for row in solutions:
        instance, document = read(data / row["instance_file"]), read(data / row["witness_file"])
        verify_witnesses(instance, document)
        witness = document["witnesses"][0]
        if witness["energy"] != row["energy"] or witness["observed_seconds"] > 60:
            raise ValueError("Portable witness disagrees with the strict-budget index")
        candidates = [
            r
            for r in successful.values()
            if r["phase"] == "confirmation"
            and (r["length"], r["instance_id"], r["family"])
            == (row["length"], row["instance_id"], row["method"])
        ]
        if len(candidates) != 2:
            raise ValueError("Witness selection requires exactly two successful calls")
        origin = next((r for r in candidates if r["cell"] == witness["source_cell"]), None)
        if (
            origin is None
            or witness["energy"] != origin["checkpoints"]["60"][witness["group"]]
            or witness["energy"] != min(e for r in candidates for e in r["checkpoints"]["60"])
        ):
            raise ValueError(
                "Witness is not the recorded best across its two calls and four groups"
            )
    return {
        "passed": True,
        "attempts": 272,
        "successful_logical_calls": 271,
        "failed_attempts_preserved": 1,
        "confirmation_calls": 216,
        "integer_witnesses_checked": 108,
        "recorded_array_fingerprints": 6229,
        "matched_replay_fingerprints": compared,
        "statistics_recomputed": True,
        "new_gpu_runs": 0,
        "ground_state_optimality_proven": False,
        "raw_trajectory_bytes_rechecked": False,
        "scope": (
            "Public core bytes, fingerprints, statistics and integer witnesses; "
            "full raw trajectories are not bundled."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/spin-glass/s23"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Verification output already exists")
    result = verify(args.data.resolve(), Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
