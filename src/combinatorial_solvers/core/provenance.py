"""Reproducibility metadata that deliberately excludes machine-specific paths."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .telemetry import query_nvidia_smi


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_value(repository: Path, arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def source_tree_fingerprint(repository: str | Path) -> dict[str, Any]:
    """Hash the locked runtime and executable source without absolute paths."""

    root = Path(repository)
    candidates = [
        root / ".python-version",
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "src").rglob("*.json")),
        *sorted((root / "scripts").rglob("*.py")),
    ]
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise ValueError("No pyproject.toml or Python package sources were found")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "algorithm": "sha256-path-null-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "scope": [
            ".python-version",
            "pyproject.toml",
            "uv.lock",
            "src/**/*.py",
            "src/**/*.json",
            "scripts/**/*.py",
        ],
    }


def capture_provenance(repository: str | Path | None = None) -> dict[str, Any]:
    """Capture package, CUDA, GPU, and Git metadata without absolute paths."""

    result: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {
            name: _package_version(name)
            for name in (
                "ising-machine-solvers",
                "matplotlib",
                "numpy",
                "pandas",
                "scipy",
                "torch",
                "triton",
            )
        },
        "torch": {
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
        },
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        result["torch"]["device_name"] = torch.cuda.get_device_name(device)
        result["torch"]["device_capability"] = list(torch.cuda.get_device_capability(device))
    snapshot = query_nvidia_smi()
    if snapshot is not None:
        result["nvidia_smi"] = {
            "name": snapshot.name,
            "total_bytes": snapshot.total_bytes,
            "used_bytes": snapshot.used_bytes,
            "free_bytes": snapshot.free_bytes,
            "utilization_percent": snapshot.utilization_percent,
        }
    if repository is not None:
        root = Path(repository)
        result["git"] = {
            "commit": _git_value(root, ["rev-parse", "HEAD"]),
            "branch": _git_value(root, ["branch", "--show-current"]),
            "dirty": bool(_git_value(root, ["status", "--porcelain"])),
        }
        try:
            result["source_tree"] = source_tree_fingerprint(root)
        except ValueError:
            result["source_tree"] = None
    result["byte_order"] = sys.byteorder
    return result
