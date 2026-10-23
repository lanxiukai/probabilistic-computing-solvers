"""Preserve an unmodified, pinned KaHIP CLI and its two 32K example graphs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from combinatorial_solvers.core.evidence import file_identity, write_new_json

COMMIT = "34e1d0a00afeae2ae2ebfe625826e9c8f8f40ac4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--commit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if json.loads(args.commit_json.read_text())["sha"] != COMMIT:
        raise ValueError("Unexpected upstream commit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.archive, output / "upstream.tar.gz")
    shutil.copy2(args.commit_json, output / "upstream_commit.json")
    with tempfile.TemporaryDirectory(prefix="partition-kahip-") as temporary:
        work = Path(temporary)
        with tarfile.open(args.archive) as archive:
            names = archive.getnames()
            prefix = f"KaHIP-KaHIP-{COMMIT[:7]}"
            if any(name.split("/")[0] != prefix for name in names):
                raise ValueError("Archive root differs from the pinned commit")
            archive.extractall(work, filter="data")
        source, build = work / prefix, work / "build"
        commands = [
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DNOMPI=ON",
                "-DPARHIP=OFF",
                "-DNONATIVEOPTIMIZATIONS=ON",
            ],
            ["cmake", "--build", str(build), "--target", "kaffpa", "-j", "6"],
        ]
        for command, log in zip(commands, ("configure.log", "build.log"), strict=True):
            with (output / log).open("w") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(build / "kaffpa", output / "kaffpa")
        shutil.copy2(source / "LICENSE", output / "LICENSE")
        for graph in ("delaunay_n15.graph", "rgg_n_2_15_s0.graph"):
            shutil.copy2(source / "examples" / graph, output / graph)
    write_new_json(
        output / "metadata.json",
        {
            "url": "https://github.com/KaHIP/KaHIP",
            "commit": COMMIT,
            "source_changes": [],
            "algorithm": "KaFFPa strong, zero imbalance and final exact-balance refinement",
            "commands": commands,
            "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
            "files": {p.name: file_identity(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps({"output": str(output), "status": "built"}))


if __name__ == "__main__":
    main()
