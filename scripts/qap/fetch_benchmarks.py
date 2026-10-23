"""Fetch pinned QAPLIB input and reference files, then certify every reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from combinatorial_solvers.qap.model import read_qaplib, read_solution


def fetch(manifest, output):
    catalog = json.loads(manifest.read_text())
    if catalog.get("schema") != "qap-benchmarks-v1":
        raise ValueError("Unsupported benchmark catalog")
    output.mkdir(parents=True, exist_ok=True)
    for record in catalog["records"]:
        for item in record["files"]:
            path = output / item["filename"]
            if path.exists():
                raw = path.read_bytes()
            else:
                request = urllib.request.Request(
                    item["url"], headers={"User-Agent": "QAP-research-reproduction/1.0"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Benchmark bytes differ from the pinned source: " + path.name)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(raw)
        instance = read_qaplib(output / (record["name"] + ".dat"))
        _, certificate = read_solution(
            output / (record["name"] + ".sln"),
            instance,
            index_base=record["reference_index_base"],
            inverse=record["reference_inverse"],
        )
        if certificate["cost"] != record["reference_cost"]:
            raise AssertionError("Pinned catalog and original reference witness disagree")
        print(json.dumps({"instance": record["name"], "reference_cost": certificate["cost"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/qap/benchmarks.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fetch(args.manifest, args.output)
