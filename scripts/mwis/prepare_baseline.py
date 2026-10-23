"""Archive a reviewed CHILS checkout, compile it, and preserve build provenance."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from combinatorial_solvers.core.evidence import file_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != "515952724cd3dcc6c4365a340ecf0f1da782119a":
        raise ValueError("Unexpected frozen reference commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        text=True,
    ).strip():
        raise ValueError("Build from a clean pinned upstream checkout")
    compiler = subprocess.check_output(["gcc", "--version"], text=True)
    archive = output / "source.tar.gz"
    with archive.open("xb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "HEAD"], cwd=source, stdout=handle, check=True
        )
    command = ["make", "-j2", "CC=gcc"]
    metadata = {
        "url": "https://github.com/KarlsruheMIS/CHILS",
        "commit": commit,
        "compiler": compiler,
        "build_command": command,
        "source_archive": file_identity(archive),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(source / "LICENSE", output / "LICENSE")
    with (output / "build.log").open("x") as handle:
        subprocess.run(
            command, cwd=source, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=120
        )
    shutil.copy2(source / "CHILS", output / "CHILS")
    metadata["executable"] = file_identity(output / "CHILS")
    metadata["license"] = file_identity(output / "LICENSE")
    metadata["build_log"] = file_identity(output / "build.log")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"commit": commit, "output": str(output)}))


if __name__ == "__main__":
    main()
