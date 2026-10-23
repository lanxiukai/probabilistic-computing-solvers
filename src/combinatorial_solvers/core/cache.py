"""Portable caches for compiled Ising coefficients and their provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import IsingModel

CACHE_SCHEMA_VERSION = 1


def canonical_cache_key(namespace: str, specification: Mapping[str, Any]) -> str:
    """Return a stable content key for a model-building specification."""

    payload = json.dumps(
        {
            "namespace": namespace,
            "schema_version": CACHE_SCHEMA_VERSION,
            "specification": specification,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_ising_model(
    path: str | Path,
    model: IsingModel,
    *,
    cache_key: str,
    specification: Mapping[str, Any],
) -> Path:
    """Atomically save coefficients without using pickle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_key": cache_key,
        "model_metadata": dict(model.metadata),
        "schema_version": CACHE_SCHEMA_VERSION,
        "specification": dict(specification),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".npz",
        prefix=f".{destination.stem}-",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            coupling=model.coupling.detach().cpu().numpy(),
            bias=model.bias.detach().cpu().numpy(),
            constant=np.asarray(model.constant, dtype=np.float64),
            names=np.asarray(model.names, dtype=np.str_),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, destination)
    return destination


def load_ising_model(
    path: str | Path,
    *,
    expected_cache_key: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[IsingModel, dict[str, Any]]:
    """Load and validate a cache created by :func:`save_ising_model`."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported Ising cache schema")
        if expected_cache_key is not None and metadata.get("cache_key") != expected_cache_key:
            raise ValueError("Ising cache key does not match the requested specification")
        coupling = torch.from_numpy(np.asarray(archive["coupling"]).copy())
        bias = torch.from_numpy(np.asarray(archive["bias"]).copy())
        if dtype is not None:
            coupling = coupling.to(dtype=dtype)
            bias = bias.to(dtype=dtype)
        model = IsingModel(
            coupling=coupling,
            bias=bias,
            constant=float(archive["constant"].item()),
            names=tuple(str(name) for name in archive["names"].tolist()),
            metadata=metadata.get("model_metadata", {}),
        )
    return model, metadata


def get_or_build_ising_model(
    cache_directory: str | Path,
    namespace: str,
    specification: Mapping[str, Any],
    builder: Callable[[], IsingModel],
    *,
    dtype: torch.dtype | None = None,
) -> tuple[IsingModel, Path, bool]:
    """Load a matching model or build and cache it.

    Returns ``(model, path, cache_hit)``.
    """

    key = canonical_cache_key(namespace, specification)
    path = Path(cache_directory) / namespace / f"{key}.npz"
    if path.exists():
        model, _ = load_ising_model(path, expected_cache_key=key, dtype=dtype)
        return model, path, True
    model = builder()
    save_ising_model(path, model, cache_key=key, specification=specification)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model, path, False
