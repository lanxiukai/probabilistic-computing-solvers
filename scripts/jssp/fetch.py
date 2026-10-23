"""Fetch six pinned public job-shop inputs and verify their original byte hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from combinatorial_solvers.jssp.model import read_instance

REVISION = "8a6e36ef31263d36975e56fb481e0ba04e188d6f"
BASE_URL = (
    f"https://raw.githubusercontent.com/ScheduleOpt/benchmarks/{REVISION}/jobshop/instances/text"
)
INPUTS = {
    "ft06": (
        "FisherThompson1963",
        "0cc5ca77665766f7829c20722a9439800ee2b34e8271ed9f0ca98c41b0c3ab3b",
        55,
    ),
    "ft10": (
        "FisherThompson1963",
        "e10bbc52f7a7bb7604bdadfbe322af84fc7726dd9937512b89759e7ac2d9aac3",
        930,
    ),
    "la01": (
        "Lawrence1984",
        "813573d84c1c42ef073ee056048dfcf460b748530e28c14fe3cc39cd11884b53",
        666,
    ),
    "la16": (
        "Lawrence1984",
        "99c2944e82fb9452545e19fdabe18ae5f9f5353984544dce85645bcda0923292",
        945,
    ),
    "ta01js": (
        "Taillard1993",
        "eb6efca2f3b22f29609abbbc15acc3ec33cac91eb2b75097aad343fc6930623d",
        1231,
    ),
    "ta51js": (
        "Taillard1993",
        "80be589030a44dbd2bd317e30478559a866394d34526b81d54d4bf62026e7c01",
        2760,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", choices=tuple(INPUTS), nargs="+", default=list(INPUTS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_revision": REVISION,
        "records": [],
        "reference_source": (
            f"https://github.com/ScheduleOpt/benchmarks/blob/{REVISION}/jobshop/README.md"
        ),
        "raw_redistribution_authorized": False,
    }
    for name in dict.fromkeys(args.instances):
        family, expected, reference = INPUTS[name]
        url = f"{BASE_URL}/{family}/{name}.txt"
        request = urllib.request.Request(url, headers={"User-Agent": "JobShopResearch/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read()
        path = args.output / (name + ".txt")
        path.write_bytes(raw)
        actual = hashlib.sha256(raw).hexdigest()
        record = {
            "file": path.name,
            "url": url,
            "sha256": actual,
            "expected_sha256": expected,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "bytes": len(raw),
            "status": "hash_verified" if actual == expected else "hash_mismatch",
            "reference_makespan": reference,
            "reference_status": "catalog_equal_lower_and_upper_bound",
        }
        manifest["records"].append(record)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if actual != expected:
            raise ValueError("Pinned input hash mismatch: " + name)
        instance = read_instance(path)
        record.update(
            jobs=instance.jobs,
            machines=instance.machine_count,
            model_binary_variables=instance.size * instance.jobs,
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps({"instance": name, "status": record["status"], "bytes": len(raw)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
