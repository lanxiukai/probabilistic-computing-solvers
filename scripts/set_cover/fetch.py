"""Fetch OR-Library originals by fixed decompressed identities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from combinatorial_solvers.set_cover.model import read_orlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/set-cover/benchmarks.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.manifest.read_text())
    if catalog.get("schema") != "set-cover-benchmarks-v1":
        raise ValueError("Unsupported benchmark catalog")
    args.output.mkdir(parents=True, exist_ok=True)
    for record in catalog["records"]:
        destination = args.output / record["file"]
        if destination.exists():
            raw = gzip.decompress(destination.read_bytes())
        else:
            with urllib.request.urlopen(record["url"], timeout=60) as response:
                raw = response.read()
        if (
            len(raw) != record["raw_bytes"]
            or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
        ):
            raise ValueError("Original benchmark differs from its pinned numerical input")
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(gzip.compress(raw, mtime=0))
        instance = read_orlib(destination, record["format"])
        if instance.statistics() != record["statistics"]:
            raise AssertionError("Parsed incidence structure differs from the catalog")
        print(
            json.dumps({"instance": record["id"], "columns": instance.size, "status": "verified"})
        )


if __name__ == "__main__":
    main()
