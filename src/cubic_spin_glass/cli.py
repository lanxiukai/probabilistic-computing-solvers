"""Portable single-instance entry point for the extracted cubic-spin-glass package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cubic_spin_glass.certificates import budget_witnesses, canonical_instance
from cubic_spin_glass.model import explicit_instance
from cubic_spin_glass.solver import SolverConfig, solve


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unreported"


def run(config_path: Path, output: Path, environment_project: Path | None = None):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = "failed"
    try:
        config_bytes = config_path.read_bytes()
        (output / "input_config.json").write_bytes(config_bytes)
        config = json.loads(config_bytes)
        if set(config) != {"instance_file", "solver"}:
            raise ValueError("Configuration requires exactly instance_file and solver")
        instance_path = config_path.parent / config["instance_file"]
        payload = canonical_instance(json.loads(instance_path.read_bytes()))
        write_json(output / "instance.json", payload)
        solver_config = SolverConfig(**config["solver"])
        environment_project = (environment_project or Path.cwd()).resolve()
        environment_snapshot = output / "source-environment"
        environment_snapshot.mkdir()
        # Preserve the owning lock and build metadata, not only installed versions.
        for name in ("uv.lock", "pyproject.toml"):
            shutil.copyfile(environment_project / name, environment_snapshot / name)
        if (environment_project / ".python-version").is_file():
            shutil.copyfile(
                environment_project / ".python-version", environment_snapshot / ".python-version"
            )
        # Package snapshots contain only extracted runtime files, never private logs.
        package = Path(__file__).resolve().parent
        for source in sorted(package.rglob("*.py")):
            if source.is_symlink():
                raise ValueError("Runtime source symlinks are not supported")
            target = output / "source" / "cubic_spin_glass" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        write_json(
            output / "config.json",
            {"instance_file": "instance.json", "solver": asdict(solver_config)},
        )
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy")}
        if solver_config.device.startswith("cuda"):
            versions["triton"] = importlib.metadata.version("triton")
        write_json(
            output / "environment.json",
            {
                "python": sys.version,
                "platform": platform.system(),
                "machine": platform.machine(),
                "cpu_model": cpu_model(),
                "packages": versions,
                "torch_threads": torch.get_num_threads(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(solver_config.device)
                if solver_config.device.startswith("cuda")
                else None,
            },
        )
        instance = explicit_instance(**payload)
        if solver_config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(solver_config.device)
        arrays, summary, rng = solve(instance, solver_config)
        if solver_config.device.startswith("cuda"):
            summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                solver_config.device
            )
        # Save partial raw states before certification; failed calls remain inspectable.
        np.savez(output / "states.npz", **arrays)
        write_json(output / "rng.json", rng)
        write_json(output / "summary.json", summary)
        if summary["failure"] is not None:
            raise RuntimeError("Solver failed; partial evidence was saved")
        for state_key, expected in (
            ("best_spins", arrays["best_energy"]),
            ("final_spins", arrays["final_energy"]),
            ("improvement_spins", arrays["improvements"][:, 4]),
        ):
            states = arrays[state_key].reshape(-1, instance.model.size)
            energies = np.asarray(expected).reshape(-1)
            for start in range(0, len(states), 16):
                if not np.array_equal(
                    instance.energy_numpy(states[start : start + 16]), energies[start : start + 16]
                ):
                    raise ValueError(f"Saved energy certificate failed: {state_key}")
        witnesses = budget_witnesses(
            payload,
            arrays["improvements"].tolist(),
            arrays["improvement_spins"],
            summary["improvement_seconds"],
            solver_config.groups,
            solver_config.max_seconds,
        )
        write_json(output / "witnesses.json", witnesses)
        summary["strict_budget_energy_by_group"] = [
            item["energy"] for item in witnesses["witnesses"]
        ]
        summary["all_saved_energies_verified"] = True
        summary["optimality_proven"] = False
        summary["elapsed_before_manifest_seconds"] = perf_counter() - started
        write_json(output / "summary.json", summary)
        status = "completed"
    except BaseException as error:
        write_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        files = {
            str(path.relative_to(output)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(
            output / "manifest.json",
            {"schema": "ea3d-portable-run-v1", "status": status, "files": files},
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--environment-project", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    print(run(args.config.resolve(), args.output.resolve(), args.environment_project))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
