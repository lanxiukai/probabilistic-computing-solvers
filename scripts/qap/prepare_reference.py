"""Build and inventory the independent QAP attribute-tabu CPU comparison."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

from combinatorial_solvers.core.evidence import file_identity


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "tabu_reference.cpp"
    shutil.copyfile(Path(__file__).with_name("tabu_reference.cpp"), source)
    command = ["c++", "-O3", "-DNDEBUG", "-std=c++17", str(source), "-o", str(output / "tabu")]
    with (output / "build.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    metadata = {
        "command": command,
        "platform": platform.platform(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "source": file_identity(source),
        "executable": file_identity(output / "tabu"),
        "method": (
            "Independent attribute-tabu implementation inspired by Taillard; "
            "not an upstream Ro-TS reproduction"
        ),
    }
    (output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
