"""Fetch the official NIST ACTS Basic jar with a fixed byte identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch(manifest: Path, output: Path):
    """Restore pinned upstream bytes or verify an existing file without replacing it."""
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    if catalog.get("schema") != "covering-array-external-baselines-v1":
        raise ValueError("Unsupported external baseline catalog")
    record = catalog["acts"]
    if output.exists():
        raw = output.read_bytes()
    else:
        request = urllib.request.Request(
            record["url"], headers={"User-Agent": "ResearchBenchmarkAcquisition/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(record["bytes"] + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != record["bytes"] or digest != record["sha256"]:
        raise ValueError("ACTS jar differs from the pinned official download")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(raw)
    return {
        "status": "verified",
        "file": str(output),
        "version": catalog["version"],
        "bytes": len(raw),
        "sha256": digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/covering-array/external-baselines.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.manifest, args.output)))


if __name__ == "__main__":
    main()
