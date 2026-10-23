"""Verify published core bytes, source bindings, and strict physical states on CPU."""

import argparse
import json
import math
from pathlib import Path

import torch

from sparse_factorization.runner import certificate_verifier
from sparse_factorization.solver.publication_io import atomic_json, digest, read_json
from sparse_factorization.stream import arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/factorization/release-v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[2]
    provenance = read_json(project / "provenance/factorization/release-v3.json")
    for row in provenance["numerical_files_byte_identical_to_executed_source"]:
        if digest(project / row["path"]) != row["sha256"]:
            raise ValueError("Published numerical source differs from its recorded identity")
    inventory = read_json(args.data / "inventory.json")
    for row in inventory["files"]:
        path = args.data / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Published core evidence changed")
    index = read_json(args.data / "index.json")
    if len(index["rows"]) != 54 or len({r["run_id"] for r in index["rows"]}) != 54:
        raise ValueError("Unexpected physical-run denominator")
    checked, failures, verifiers = [], [], {}
    for row in index["rows"]:
        root = args.data / "physical" / row["run_id"]
        config = read_json(root / "config.json")
        recorded = read_json(root / "result/summary.json")
        if not row["success"]:
            if (
                recorded["status"] != "event_limit"
                or recorded["events"] != config["maximum_events"]
            ):
                raise ValueError("Censored result does not match the event limit")
            failures.append(row["run_id"])
            continue
        if digest(root / "witnesses.npz") != row["witness_sha256"]:
            raise ValueError("Witness packet changed")
        if config["target"] not in verifiers:
            verifiers[config["target"]] = certificate_verifier(config)
        certificates = verifiers[config["target"]](arrays(root / "witnesses.npz"))
        if [c["state_sha256"] for c in certificates] != [
            c["state_sha256"] for c in recorded["witnesses"]
        ]:
            raise ValueError("Certified state differs from the reported accepted state")
        checked.append({"run_id": row["run_id"], "certificates": certificates})
    replays = [read_json(p) for p in sorted((args.data / "replays").glob("*.json"))]
    if len(replays) != 6 or not all(r["full_origin_replay"] for r in replays):
        raise ValueError("Required recorded origin-replay receipts are missing")
    classical = [read_json(p) for p in (args.data / "classical").glob("*/summary.json")]
    if len(classical) != 54 or not all(
        r["success"] and math.prod(r["factors"]) == r["target"] for r in classical
    ):
        raise ValueError("Classical-control integer certificates failed")
    integer = [read_json(p) for p in (args.data / "integer").glob("*/result/summary.json")]
    if len(integer) != 18:
        raise ValueError("Unexpected selected-root control denominator")
    for row in integer:
        if row["success"] and not all(
            c["factor_a"] * c["factor_b"] == row["target"] for c in row["certificates"]
        ):
            raise ValueError("Selected-root integer certificate failed")
    result = {
        "passed": True,
        "physical_runs": len(index["rows"]),
        "strict_successful_runs_verified": len(checked),
        "event_limit_runs": failures,
        "recorded_origin_replays": len(replays),
        "classical_integer_certificates": len(classical),
        "selected_root_control_runs": len(integer),
        "new_gpu_replays": 0,
        "new_independent_trials": 0,
        "rows": checked,
    }
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
