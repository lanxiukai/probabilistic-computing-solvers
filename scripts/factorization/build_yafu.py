"""Build the pinned CPU baseline in an isolated directory without system installs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from sparse_factorization.solver.publication_io import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("baselines/factorization/yafu.lock.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        raise ValueError("Build parallelism must be in [1,16]")
    spec = json.loads(args.lock.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    def acquire(url, path, expected):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        if digest(path) != expected:
            raise ValueError(f"Baseline source checksum differs: {path.name}")
        return path

    source = acquire(spec["source_url"], cache / "yafu-source.tar.gz", spec["source_sha256"])
    deps = output / "deps"
    deps.mkdir()
    for row in spec["packages"]:
        package = acquire(row["url"], cache / "debs" / row["file"], row["sha256"])
        subprocess.run(["dpkg-deb", "-x", str(package), str(deps)], check=True)
    with tarfile.open(source) as archive:
        archive.extractall(output, filter="data")
    tree = output / ("yafu-" + spec["revision"])
    command = [
        "make",
        f"-j{args.jobs}",
        "yafu",
        *spec["make_flags"],
        f"GMP_INCDIR={deps}/usr/include/x86_64-linux-gnu",
        f"GMP_LIBDIR={deps}/usr/lib/x86_64-linux-gnu",
        f"ECM_INCDIR={deps}/usr/include",
        f"ECM_LIBDIR={deps}/usr/lib/x86_64-linux-gnu",
    ]
    with (output / "build.log").open("x") as log:
        subprocess.run(command, cwd=tree, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(tree / "yafu", output / "yafu")
    library = output / "lib"
    library.mkdir()
    for name in ("libgmp.so.10", "libecm.so.1"):
        shutil.copy2(deps / "usr/lib/x86_64-linux-gnu" / name, library / name)
    atomic_json(
        output / "build.json",
        {
            "schema": spec["schema"],
            "revision": spec["revision"],
            "lock_sha256": digest(args.lock),
            "source_sha256": spec["source_sha256"],
            "flags": spec["make_flags"],
            "binary_sha256": digest(output / "yafu"),
            "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
            "libraries": {p.name: digest(p) for p in library.iterdir()},
        },
    )
    print(json.dumps({"status": "built", "binary": str(output / "yafu")}))


if __name__ == "__main__":
    main()
