"""Fetch pinned AC-CFT source files and fmt without the upstream instance collection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "research-reproduction"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/set-cover/cft-source.json"))
    parser.add_argument(
        "--fmt-manifest", type=Path, default=Path("configs/set-cover/fmt-source.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    fmt = json.loads(args.fmt_manifest.read_text())
    if source["revision"] != "6cb63d0036223b8be9c4d01f95b296d641209f95":
        raise ValueError("Unexpected AC-CFT revision")
    args.output.mkdir(parents=True, exist_ok=False)

    def fetch(record):
        path = Path(record["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe upstream source path")
        url = (
            "https://raw.githubusercontent.com/c4v4/cft/"
            + source["revision"]
            + "/"
            + path.as_posix()
        )
        data = download(url)
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Pinned AC-CFT source file differs: " + path.as_posix())
        return path.as_posix(), data

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch, source["files"]))
    with (args.output / "source-only.tar.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, content in sorted(files.items()):
                    member = tarfile.TarInfo("cft/" + name)
                    member.size, member.mode, member.mtime = len(content), 0o644, 0
                    archive.addfile(member, io.BytesIO(content))
    dependency = download(fmt["url"])
    if len(dependency) != fmt["bytes"] or hashlib.sha256(dependency).hexdigest() != fmt["sha256"]:
        raise ValueError("Pinned fmt archive differs")
    (args.output / "fmt-10.2.1.tar.gz").write_bytes(dependency)
    (args.output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    (args.output / "fmt-source.json").write_text(json.dumps(fmt, indent=2) + "\n")
    print(json.dumps({"status": "verified", "upstream_source_files": len(files)}))


if __name__ == "__main__":
    main()
