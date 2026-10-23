"""Build MQLib with a certified partition correction and an external callback."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from combinatorial_solvers.core.evidence import file_identity, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != "585496274af5abb0849d0d47e135496b4688680b":
        raise ValueError("Unexpected frozen reference commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source,
        text=True,
    ).strip():
        raise ValueError("Build from a clean pinned upstream checkout")
    driver = Path(__file__).with_name("mqlib_driver.cpp").resolve()
    shutil.copy2(driver, output / "driver.cpp")
    shutil.copy2(source / "LICENSE", output / "LICENSE")
    with tarfile.open(output / "upstream.tar.gz", "w:gz") as archive:
        for name in ("include", "src", "Makefile", "README.md", "LICENSE"):
            archive.add(source / name, arcname=name)
    before = subprocess.check_output(["git", "diff", "--", "src", "include"], cwd=source, text=True)
    if before:
        raise ValueError("Expected an unmodified upstream implementation")
    commands, checks = [], []
    probe = Path(__file__).with_name("mqlib_partition_probe.cpp").resolve()
    shutil.copy2(probe, output / "partition_probe.cpp")
    anchor = "if (assignments_[i]) {"
    replacement = "if (assignments_[i] == 1) {"
    invariant_anchor = "\n}\n\nvoid Festa2002Solution::VNS(int kMax)"
    invariant = """
  for (int v : S_) if (assignments_[v] != 1) {
    std::cerr << "partition invariant failed in S" << std::endl; std::exit(13);
  }
  for (int v : Sbar_) if (assignments_[v] != -1) {
    std::cerr << "partition invariant failed in Sbar" << std::endl; std::exit(13);
  }
"""
    with tempfile.TemporaryDirectory(prefix="maxcut-mqlib-build-") as temporary:
        work = Path(temporary) / "source"
        work.mkdir()
        for name in ("include", "src"):
            shutil.copytree(source / name, work / name)
        for name in ("Makefile", "README.md", "LICENSE"):
            shutil.copy2(source / name, work / name)
        target = work / "src/heuristics/maxcut/festa2002.cpp"
        original = target.read_text()
        if original.count(anchor) != 1 or original.count(invariant_anchor) != 1:
            raise ValueError("The audited upstream correction anchor changed")
        corrected = original.replace(anchor, replacement)
        (output / "correction.patch").write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    corrected.splitlines(keepends=True),
                    fromfile="upstream/src/heuristics/maxcut/festa2002.cpp",
                    tofile="corrected/src/heuristics/maxcut/festa2002.cpp",
                )
            )
        )
        target.write_text(corrected)
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for name in ("include", "src", "Makefile", "README.md", "LICENSE"):
                archive.add(work / name, arcname=name)

        def compile_program(entry, destination, log):
            command = [
                "g++",
                "-O3",
                "-std=c++17",
                "-I",
                str(work / "include"),
                str(entry),
                *[str(p) for p in sorted((work / "src").rglob("*.cpp")) if p.name != "main.cpp"],
                "-o",
                str(destination),
            ]
            commands.append(command)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError("Baseline build failed; preserve build.log")

        with (output / "build.log").open("x") as log:
            compile_program(driver, output / "MQLib-callback", log)
            for name, text, expected in (("upstream", original, 13), ("corrected", corrected, 0)):
                target.write_text(text.replace(invariant_anchor, invariant + invariant_anchor))
                executable = Path(temporary) / f"probe-{name}"
                compile_program(probe, executable, log)
                result = subprocess.run(
                    [str(executable)], capture_output=True, text=True, check=False
                )
                checks.append(
                    {
                        "version": name,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "expected": expected,
                    }
                )
                write_new_json(output / f"probe-{name}.json", checks[-1])
                if result.returncode != expected:
                    raise AssertionError(
                        "The original failure and corrected invariant were not reproduced"
                    )
    write_new_json(
        output / "metadata.json",
        {
            "upstream": "https://github.com/MQLib/MQLib",
            "commit": commit,
            "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
            "commands": commands,
            "returncode": 0,
            "algorithm_changes": "FESTA neighborhood membership tests spin == +1, not nonzero",
            "correction_checks": checks,
            "probe_invariant_instrumentation": invariant,
            "termination": "public MaxCutCallback; fixed report count replay",
            "driver": file_identity(output / "driver.cpp"),
        },
    )
    print(json.dumps({"commit": commit, "output": str(output), "status": "prepared"}))


if __name__ == "__main__":
    main()
