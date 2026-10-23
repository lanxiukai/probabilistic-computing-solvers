"""Rebuild research-only TSP references from explicitly supplied pinned archives.

This recipe is project-authored. It does not redistribute or relicense upstream
sources, download files, or execute a solver. Obtain the archives under their
authors' terms before invoking this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from time import perf_counter

LKH_SHA256 = "693bb5e6b048e58f11584f32ac56325020f6d65fbabbec5d6da95528bcad3f04"
EAX_SHA256 = "d00186a9281a77ee940dd0be6b876821aab2ff0ca51264523fa7fcc1967a5885"
EAX_COMMIT = "015dfbe9f267230f78787bd244af393ffc018900"


def identity(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "executable": bool(path.stat().st_mode & 0o111),
    }


def extract(archive: Path, output: Path, expected: str):
    if identity(archive)["sha256"] != expected:
        raise ValueError("Upstream source archive differs from its pinned identity")
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if any(
            not (p.isdir() or p.isfile())
            or Path(p.name).is_absolute()
            or ".." in Path(p.name).parts
            for p in members
        ):
            raise ValueError("Only regular relative upstream files/directories are supported")
        handle.extractall(output, filter="data")


def seed_bridge(source: Path):
    target = source / "main.cpp"
    before = target.read_text()
    original = "  InitURandom();"
    if before.count(original) != 1:
        raise ValueError("Pinned GA-EAX seed call changed")
    target.write_text(
        before.replace(
            original,
            '  const char* seed = getenv("TSP_EAX_SEED");\n'
            "  if (seed) InitURandom(atoi(seed)); else InitURandom();",
        )
    )


def build(lkh_archive: Path, eax_archive: Path | None, output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    try:
        originals = output / "upstream"
        originals.mkdir()
        extract(lkh_archive, originals, LKH_SHA256)
        if eax_archive is not None:
            extract(eax_archive, originals, EAX_SHA256)
        work = output / "work"
        work.mkdir()
        lkh = work / "lkh"
        eax = work / "eax"
        shutil.copytree(originals / "LKH-3.0.13", lkh)
        if eax_archive is not None:
            shutil.copytree(originals / f"GA-EAX-{EAX_COMMIT}/GA_EAX_1.0/Normal", eax)
            seed_bridge(eax)
        binaries = output / "bin"
        binaries.mkdir()
        specifications = [
            ("lkh", lkh / "SRC", ["make", "-j2", "CC=gcc"]),
            (
                "eax",
                eax,
                [
                    "g++",
                    "-O3",
                    "-o",
                    str(binaries / "GA-EAX"),
                    "main.cpp",
                    "env.cpp",
                    "cross.cpp",
                    "evaluator.cpp",
                    "indi.cpp",
                    "rand.cpp",
                    "kopt.cpp",
                    "sort.cpp",
                    "-lm",
                ],
            ),
        ]
        if eax_archive is None:
            specifications = specifications[:1]
        for name, cwd, command in specifications:
            started = perf_counter()
            with (output / f"{name}.log").open("xb") as log:
                result = subprocess.run(
                    command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
                )
            commands.append(
                {
                    "name": name,
                    "command": command,
                    "cwd": str(cwd.relative_to(output)),
                    "return_code": result.returncode,
                    "seconds": perf_counter() - started,
                }
            )
            if result.returncode:
                raise RuntimeError(f"{name} compilation failed; original log retained")
        shutil.copy2(lkh / "LKH", binaries / "LKH")
        record = {
            "schema": "tsp-native-build-v1",
            "source": {"lkh": LKH_SHA256, **({"eax": EAX_SHA256} if eax_archive else {})},
            "eax_commit": EAX_COMMIT if eax_archive else None,
            "commands": commands,
            "compiler": {
                tool: subprocess.check_output([tool, "--version"], text=True)
                for tool in (("gcc", "g++") if eax_archive else ("gcc",))
            },
            "binaries": {
                name: identity(binaries / name)
                for name in (("LKH", "GA-EAX") if eax_archive else ("LKH",))
            },
            "seed_bridge": "Call the existing InitURandom(int) overload via TSP_EAX_SEED"
            if eax_archive
            else None,
            "solver_executions": 0,
            "upstream_license": "Research use; authors reserve rights",
        }
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps({"error": repr(error), "commands": commands}, indent=2) + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lkh-archive", type=Path, required=True)
    parser.add_argument("--eax-archive", type=Path, help="Omit when only the LKH prefix is needed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.lkh_archive, args.eax_archive, args.output)
    print(json.dumps({"binaries": result["binaries"], "solver_executions": 0}))


if __name__ == "__main__":
    main()
