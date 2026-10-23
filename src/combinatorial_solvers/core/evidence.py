"""Problem-independent inventories for immutable runs and explicit catalog aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "experiment-manifest-v1"
FINAL_STATUSES = {"completed", "failed", "interrupted", "preparation_failed"}


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "executable": bool(path.stat().st_mode & 0o111),
    }


def write_new_json(path: Path, payload: Any) -> None:
    """Claim a new file exclusively; never replace a completed record."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def validate_identity(problem: str, run_id: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", problem):
        raise ValueError("Problem must be a lowercase slug")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_id):
        raise ValueError("Run ID must be one lowercase path component")


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence must not contain symlinks: {path}")
        if path.is_dir() or path == root / "manifest.json":
            continue
        if not path.is_file():
            raise ValueError(f"Evidence must contain regular files: {path}")
        records.append({"path": path.relative_to(root).as_posix(), **file_identity(path)})
    return records


def write_manifest(root: Path, *, problem: str, run_id: str, status: str) -> Path:
    """Finalize a single run, including partial outputs of failed or interrupted runs."""
    validate_identity(problem, run_id)
    if status not in FINAL_STATUSES:
        raise ValueError("Only finished runs can be inventoried")
    if not root.is_dir():
        raise FileNotFoundError(root)
    destination = root / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    records = _inventory(root)
    if not records:
        raise ValueError("Evidence inventory cannot be empty")
    write_new_json(
        destination,
        {
            "schema": MANIFEST_SCHEMA,
            "problem": problem,
            "run_id": run_id,
            "status": status,
            "files": records,
        },
    )
    return destination


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") not in FINAL_STATUSES:
        raise ValueError("Unsupported experiment manifest")
    validate_identity(manifest["problem"], manifest["run_id"])
    if not manifest.get("files") or manifest["files"] != _inventory(root):
        raise ValueError("Experiment manifest differs from saved evidence")
    return manifest


def write_catalog(manifests: list[Path], output: Path) -> Path:
    """Aggregate explicit finalized manifests without scanning or modifying other runs."""
    if not manifests:
        raise ValueError("At least one experiment manifest is required")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    identities = set()
    for path in manifests:
        path = path.resolve()
        if path.name != "manifest.json":
            raise ValueError("Each catalog input must be a run's manifest.json")
        if output.resolve().is_relative_to(path.parent):
            raise ValueError("A catalog must be outside its immutable input runs")
        manifest = verify_manifest(path.parent)
        identity = (manifest["problem"], manifest["run_id"])
        if identity in identities:
            raise ValueError(f"Duplicate experiment identity: {identity}")
        identities.add(identity)
        rows.append(
            {
                "problem": identity[0],
                "run_id": identity[1],
                "status": manifest["status"],
                "manifest": os.path.relpath(path, output.resolve().parent),
                "manifest_sha256": file_identity(path)["sha256"],
                "files": len(manifest["files"]),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(
        output,
        {
            "schema": "experiment-catalog-v1",
            "runs": sorted(rows, key=lambda r: (r["problem"], r["run_id"])),
        },
    )
    return output
