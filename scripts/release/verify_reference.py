"""Compare locally built reference source members with the measured source identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def archive_members(path):
    with tarfile.open(path) as archive:
        return {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive
            if member.isfile()
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.root / "baselines/source-identities.json").read_text())[
        args.reference
    ]
    for filename, expected in record["archives"].items():
        if archive_members(args.directory / filename) != expected:
            raise ValueError(f"Reference source members differ: {filename}")
    for filename, expected in record.get("files", {}).items():
        actual = hashlib.sha256((args.directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference adapter source differs: {filename}")
    print(
        json.dumps(
            {
                "reference": args.reference,
                "source_members": "verified",
                "binary_or_performance_equivalence_asserted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
