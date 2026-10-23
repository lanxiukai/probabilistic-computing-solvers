"""Portable executable-source archives with verified, non-overwriting restoration."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from .provenance import source_tree_fingerprint

SCHEMA = "executable-source-archive-v1"


def _files(root: Path) -> list[Path]:
    paths = [root / name for name in (".python-version", "pyproject.toml", "uv.lock")]
    for name in ("README.md", "snapshot.json"):
        if (root / name).is_file():
            paths.append(root / name)
    for directory in ("src", "scripts", "tests", "configs"):
        paths.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml"}
        )
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("Archive sources must be existing regular files")
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def pack(source: Path, output: Path, *, expected: str | None = None) -> dict:
    """Freeze source and dependency locks; do not include environments or caches."""
    source = source.resolve()
    fingerprint = source_tree_fingerprint(source)
    if expected is not None and fingerprint["sha256"] != expected:
        raise ValueError("Source fingerprint does not match the recorded execution")
    payload = {p.relative_to(source).as_posix(): p.read_bytes() for p in _files(source)}
    manifest = {
        "schema": SCHEMA,
        "source_tree": fingerprint,
        "scope": [
            "runtime locks",
            "README",
            "snapshot metadata",
            "src",
            "scripts",
            "tests",
            "configs",
        ],
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payload.items()
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name, data in payload.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    manifest["archive_bytes"] = archive.stat().st_size
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore(bundle: Path, output: Path) -> dict:
    """Verify every stored file before extracting into a new independent directory."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    archive = bundle / "source.tar.gz"
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown executable archive schema")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive integrity check failed")
    expected = {row["path"]: row for row in manifest["files"]}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive members do not match the source manifest")
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Archive must contain only relative regular source files")
            content = tar.extractfile(member)
            assert content is not None
            data = content.read()
            row = expected[member.name]
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Source member integrity check failed")
        output.mkdir(parents=True, exist_ok=False)
        tar.extractall(output, filter="data")
    actual = source_tree_fingerprint(output)
    if actual != manifest["source_tree"]:
        raise ValueError("Restored execution fingerprint differs from the source")
    return {"source_tree": actual, "files_verified": len(expected), "restored": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("pack")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--expected-source-sha256")
    extract = sub.add_parser("restore")
    extract.add_argument("--bundle", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        pack(args.source, args.output, expected=args.expected_source_sha256)
        if args.action == "pack"
        else restore(args.bundle, args.output)
    )
    print(json.dumps(result, indent=2))
