"""Bind a frozen TSP candidate case to explicit benchmark and baseline inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_case(
    case: dict, instances: list[dict], instance_directory: Path, baseline_directory: Path
) -> dict:
    config = copy.deepcopy(case["configuration"])
    instance = next(row for row in instances if row["name"] == case["instance"])
    source = (instance_directory / instance["filename"]).resolve()
    if digest(source) != instance["sha256"]:
        raise ValueError("The supplied TSPLIB input does not match the frozen instance")
    build = json.loads((baseline_directory / "build.json").read_text())
    from prepare_native import EAX_SHA256, LKH_SHA256

    kind = config["solver"]["kind"]
    required_source = "eax" if kind == "eax" else "lkh"
    expected_sources = {"lkh": LKH_SHA256, "eax": EAX_SHA256}
    if required_source not in build["source"] or any(
        expected_sources.get(name) != identity for name, identity in build["source"].items()
    ):
        raise ValueError("Baseline build does not identify the pinned upstream sources")
    config["problem"]["instance"] = str(source)
    name = "GA-EAX" if kind == "eax" else "LKH"
    binary = (baseline_directory / "bin" / name).resolve()
    identity = digest(binary)
    if identity != build["binaries"][name]["sha256"]:
        raise ValueError("Native binary changed after its recorded build")
    field = "eax_executable" if kind == "eax" else "lkh_executable"
    config["prefix"] = {field: str(binary)}
    config["solver"]["lkh_sha256" if kind == "edge_assignment" else "executable_sha256"] = identity
    config["experiment"] = {}
    return config
