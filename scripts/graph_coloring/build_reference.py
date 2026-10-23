"""Build the owned single-thread TabuCol-style comparison with a source receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("tabu_reference.cpp")
    copied = output / source.name
    shutil.copy2(source, copied)
    command = [
        "c++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        str(copied),
        "-o",
        str(output / "reference"),
    ]
    with (output / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {
        "command": command,
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "source_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "executable_sha256": hashlib.sha256((output / "reference").read_bytes()).hexdigest(),
        "implementation": "Independent owned TabuCol-style source; not upstream HEAD",
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "status": "built",
        "solver_executed": False,
        "source_sha256": receipt["source_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/graph-coloring/tabu-reference"))
    args = parser.parse_args()
    print(json.dumps(build(args.output)))


if __name__ == "__main__":
    main()
