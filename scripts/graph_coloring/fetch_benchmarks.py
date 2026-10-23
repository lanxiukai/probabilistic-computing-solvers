"""Fetch original DIMACS coloring bytes pinned by the published input catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch(catalog, output, names=None):
    records = json.loads(Path(catalog).read_text())["records"]
    requested = set(names) if names else {row["name"] for row in records}
    if requested - {row["name"] for row in records}:
        raise ValueError("Requested instance is absent from the pinned catalog")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for row in records:
        if row["name"] not in requested:
            continue
        source = row["file"]
        filename = source["filename"]
        if Path(filename).name != filename or not filename.endswith(".col"):
            raise ValueError("Catalog filename must be a local DIMACS graph filename")
        path = output / filename
        if path.exists():
            raw = path.read_bytes()
            status = "already-present-and-verified"
        else:
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "graph-coloring-reproduction/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read(source["bytes"] + 1)
            status = "downloaded-and-verified"
        if len(raw) != source["bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Original graph differs from the pinned bytes: " + filename)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)
        receipts.append({"name": row["name"], "status": status, "sha256": source["sha256"]})
    return {"status": "verified", "solver_executed": False, "records": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("configs/graph-coloring/benchmarks.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/graph-coloring/dimacs"))
    parser.add_argument("--instance", action="append", dest="instances")
    args = parser.parse_args()
    print(json.dumps(fetch(args.catalog, args.output, args.instances), indent=2))


if __name__ == "__main__":
    main()
