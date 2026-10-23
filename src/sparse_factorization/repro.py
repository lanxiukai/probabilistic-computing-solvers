"""Portable source snapshots and isolated, one-shot run execution."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .solver.publication_io import atomic_json, canonical_digest, digest, read_json


def source_files(project: Path) -> list[Path]:
    files = [
        project / name for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md")
    ]
    files += [
        project / name for name in ("README.zh-CN.md", "LICENSE") if (project / name).is_file()
    ]
    for directory in ("src/sparse_factorization", "scripts/factorization", "tests/factorization"):
        files += sorted((project / directory).rglob("*.py"))
    files += sorted((project / "src/sparse_factorization/templates").glob("*.md"))
    for directory in ("configs/factorization", "baselines/factorization"):
        files += sorted((project / directory).glob("*.json"))
    return files


def source_identity(project: Path) -> dict:
    rows = [
        {"path": p.relative_to(project).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
        for p in source_files(project)
    ]
    return {"files": rows, "sha256": canonical_digest(rows)}


def prepare(project: Path, config: dict, output: Path, plan: dict | None = None) -> dict:
    """Freeze inputs and exact source without starting a solver."""
    from .runner import validate_config

    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity(project)
    atomic_json(output / "config.json", config)
    if "plan_sha256" in config:
        if plan is None or canonical_digest(plan) != config["plan_sha256"]:
            raise ValueError("A protocol-bound run needs its complete matching frozen plan")
        atomic_json(output / "plan.json", plan)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for row in identity["files"]:
            file = project / row["path"]
            if file.is_symlink():
                raise ValueError("Execution source cannot be a symlink")
            archive.add(file, arcname=row["path"], recursive=False)
    prepared = {
        "schema": "factor-prepared-run-v1",
        "source": identity,
        "config_sha256": canonical_digest(config),
        "source_archive_sha256": digest(output / "source.tar.gz"),
    }
    atomic_json(output / "prepared.json", prepared)
    return prepared


def restore(output: Path, destination: Path) -> dict:
    prepared = read_json(output / "prepared.json")
    if canonical_digest(read_json(output / "config.json")) != prepared["config_sha256"]:
        raise ValueError("Prepared input configuration changed")
    config = read_json(output / "config.json")
    if (
        "plan_sha256" in config
        and canonical_digest(read_json(output / "plan.json")) != config["plan_sha256"]
    ):
        raise ValueError("Prepared protocol changed")
    if digest(output / "source.tar.gz") != prepared["source_archive_sha256"]:
        raise ValueError("Prepared source archive changed")
    destination.mkdir(parents=True, exist_ok=False)
    expected = {r["path"]: r for r in prepared["source"]["files"]}
    with tarfile.open(output / "source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Source archive inventory differs")
        for member in members:
            target = destination / member.name
            if not member.isfile() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe source archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as sink:
                sink.write(source.read())
            if digest(target) != expected[member.name]["sha256"]:
                raise ValueError("Restored source member differs")
    return prepared


def environment() -> dict:
    import torch

    result = {
        "python": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "triton", "numpy", "scipy")
        },
    }
    cpu = Path("/proc/cpuinfo").read_text()
    result["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpu.splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    result["system_memtotal"] = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if path.is_file():
            result["memory_cgroup_limit"] = path.read_text().strip()
            break
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    result["gpu"] = query.stdout.strip()
    result["gpu_query_returncode"] = query.returncode
    result["torch_gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result["torch_gpu_count"] = torch.cuda.device_count()
    return result


def execute(output: Path) -> int:
    """Run from restored source; never import the live editing tree in the worker."""
    output = output.resolve()
    if (output / "started.json").exists():
        raise FileExistsError("A prepared run is one-shot; prepare a new identity for another run")
    prepared = restore(output, output / "execution-source")
    atomic_json(output / "started.json", {"source_sha256": prepared["source"]["sha256"]})
    config = read_json(output / "config.json")
    maximum = config["maximum_sampling_seconds"] + config.get("task_overhead_seconds", 900)
    source = output / "execution-source"
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from sparse_factorization.repro import worker; "
        "from pathlib import Path; worker(Path(sys.argv[2]))"
    )
    command = [sys.executable, "-I", "-c", script, str(source / "src"), str(output)]
    started = time.perf_counter()
    with (output / "run.log").open("x") as log:
        with subprocess.Popen(
            command,
            cwd=source,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            start_new_session=True,
        ) as process:
            try:
                code = process.wait(timeout=maximum)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
    atomic_json(
        output / "execution.json",
        {
            "returncode": code,
            "launcher_wall_seconds": time.perf_counter() - started,
            "task_wall_cap_seconds": maximum,
            "source_sha256": prepared["source"]["sha256"],
        },
    )
    files = [
        {"path": p.relative_to(output).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.is_relative_to(source)
    ]
    atomic_json(
        output / "manifest.json",
        {
            "schema": "factor-run-manifest-v1",
            "status": "completed" if code == 0 else "failed",
            "files": files,
        },
    )
    return code


def worker(output: Path) -> None:
    source = output / "execution-source"
    if not Path(__file__).resolve().is_relative_to(source / "src"):
        raise RuntimeError("Worker imported live code instead of its immutable snapshot")
    prepared = read_json(output / "prepared.json")
    if source_identity(source) != prepared["source"]:
        raise ValueError("Restored execution source differs")
    observed = environment()
    atomic_json(output / "environment.json", observed)
    config = read_json(output / "config.json")
    try:
        if "plan_sha256" in config:
            plan = read_json(output / "plan.json")
            if (
                observed["torch_gpu_name"] != plan["hardware"]["gpu"]
                or observed["torch_gpu_count"] != plan["hardware"]["gpu_count"]
            ):
                raise RuntimeError("Actual GPU allocation differs from the frozen release plan")
        if config.get("arm", "physical") == "selected_root_integer":
            from .arithmetic import run_arithmetic

            run_arithmetic(config, output / "result", prepared["source"]["sha256"])
        else:
            from .runner import run_physical

            run_physical(config, output / "result", prepared["source"]["sha256"])
    except BaseException as error:
        atomic_json(output / "worker_failure.json", {"error": repr(error)})
        raise
    if source_identity(source) != prepared["source"]:
        raise ValueError("Execution modified its source snapshot")
