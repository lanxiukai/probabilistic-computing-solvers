"""Build pinned AC-CFT with original search code and an audited timer observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = args.source_directory.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((sources / "source.json").read_text())
    if source["revision"] != "6cb63d0036223b8be9c4d01f95b296d641209f95":
        raise ValueError("Unexpected CFT revision")
    fmt = json.loads((sources / "fmt-source.json").read_text())
    if digest(sources / "fmt-10.2.1.tar.gz") != fmt["sha256"]:
        raise ValueError("fmt archive changed")
    for name in ("source-only.tar.gz", "source.json", "fmt-10.2.1.tar.gz", "fmt-source.json"):
        shutil.copy2(sources / name, output / name)
    shutil.copy2(Path(__file__).with_name("reference_driver.cpp"), output / "reference_driver.cpp")
    commands = []
    with tempfile.TemporaryDirectory(prefix="set-cover-baseline-build-") as temporary:
        work = Path(temporary)
        with tarfile.open(output / "source-only.tar.gz") as archive:
            archive.extractall(work, filter="data")
        with tarfile.open(output / "fmt-10.2.1.tar.gz") as archive:
            archive.extractall(work, filter="data")
        upstream = work / "cft"
        for item in source["files"]:
            if digest(upstream / item["path"]) != item["sha256"]:
                raise ValueError("Pinned upstream source differs")
        shutil.copy2(upstream / "LICENSE", output / "LICENSE")
        shutil.copy2(work / "fmt-10.2.1/LICENSE", output / "fmt-LICENSE")
        pristine = work / "pristine"
        shutil.copytree(upstream, pristine)
        chrono = upstream / "src/utils/Chrono.hpp"
        old = "return std::chrono::duration_cast<UnitT>(now - start).count();"
        new = (
            "return ::research_clock(this, std::chrono::duration_cast<UnitT>(now - start).count());"
        )
        content = chrono.read_text()
        if content.count(old) != 1:
            raise ValueError("Timer observation point differs")
        chrono.write_text(content.replace(old, new))
        (output / "clock-instrumentation.json").write_text(
            json.dumps(
                {
                    "file": "src/utils/Chrono.hpp",
                    "before": old,
                    "after": new,
                    "purpose": (
                        "Record unchanged native elapsed values; replay the same values only "
                        "in explicitly marked non-performance replays."
                    ),
                    "search_score_proposal_or_stopping_comparison_changes": False,
                },
                indent=2,
            )
            + "\n"
        )
        with (output / "build.log").open("w") as log:
            for directory, name in ((upstream, "reference"), (pristine, "reference-pristine")):
                command = [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-march=native",
                    "-DFMT_HEADER_ONLY",
                    f"-I{directory / 'src'}",
                    f"-I{work / 'fmt-10.2.1/include'}",
                    str(output / "reference_driver.cpp"),
                    "-o",
                    str(output / name),
                ]
                commands.append(command)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = {
        "upstream_revision": source["revision"],
        "fmt_version": fmt["version"],
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "build_commands": commands,
        "native_search_headers_unchanged": True,
        "timer_observer_disclosed": True,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "built", "output": str(output)}))


if __name__ == "__main__":
    main()
