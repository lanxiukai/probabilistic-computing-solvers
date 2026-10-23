"""Prepare portable inputs, execute archived source, and finalize one isolated run."""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from combinatorial_solvers.core.evidence import (
    file_identity,
    validate_identity,
    write_manifest,
    write_new_json,
)
from combinatorial_solvers.core.provenance import source_tree_fingerprint
from combinatorial_solvers.core.source_archive import pack, restore
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import load_yaml_mapping

PREPARED_SCHEMA = "prepared-experiment-v1"


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _field(config: dict, key: str) -> Any:
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _set_field(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
        if not isinstance(config, dict):
            raise ValueError(f"Configuration field requires mapping parents: {key}")
    config[parts[-1]] = value


def prepare_experiment(
    specification: SolverSpec,
    config_path: Path,
    *,
    repository: Path,
    run_id: str,
    output_root: Path | None = None,
) -> Path:
    """Freeze a new run without executing any solver or allocating GPU resources.

    Relative input paths are resolved from the owning project, as for the existing
    CLI. Only fields declared by the problem are interpreted as input files.
    """
    SolverRegistry([specification])
    validate_identity(specification.name, run_id)
    repository = repository.resolve()
    config_path = config_path if config_path.is_absolute() else repository / config_path
    config = load_yaml_mapping(config_path)
    base = output_root if output_root is not None else repository / "artifacts/runs"
    if not base.is_absolute():
        base = repository / base
    output = (base / specification.name / run_id).resolve()
    artifacts = repository / "artifacts"
    if output.is_relative_to(artifacts) and not output.is_relative_to(artifacts / "runs"):
        raise ValueError("New experiments under artifacts must use the independent runs namespace")
    if repository.is_relative_to(output) or any(
        output.is_relative_to(repository / directory)
        for directory in ("src", "scripts", "configs", "tests")
    ):
        raise ValueError("Experiment outputs must be outside executable source directories")
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_new_json(output / "request.json", config)
        frozen = copy.deepcopy(config)
        inputs = []
        for field in specification.input_fields:
            value = _field(frozen, field)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"Input file must be a nonempty path: {field}")
            original = Path(value).expanduser()
            original = original if original.is_absolute() else repository / original
            original = original.resolve(strict=True)
            if not original.is_file():
                raise ValueError(f"Input must be a regular file: {field}")
            relative = Path("inputs", *field.split("."), original.name)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            inputs.append({"field": field, "path": relative.as_posix(), **file_identity(target)})
            _set_field(frozen, field, relative.as_posix())
        for field in specification.working_directory_fields:
            _set_field(frozen, field, Path("work", *field.split(".")).as_posix())
        _set_field(frozen, "experiment.output_dir", "result")
        write_new_json(output / "config.json", frozen)
        source = pack(repository, output / "source")
        # Validate the archive itself; hashing a changing worktree alone is insufficient.
        with TemporaryDirectory(prefix="ising-prepare-") as temporary:
            restore(output / "source", Path(temporary) / "execution")
        write_new_json(
            output / "prepared.json",
            {
                "schema": PREPARED_SCHEMA,
                "problem": specification.name,
                "run_id": run_id,
                "prepared_at_utc": _utc(),
                "runner": asdict(specification),
                "config": file_identity(output / "config.json"),
                "inputs": inputs,
                "source_tree": source["source_tree"],
                "source_archive_sha256": source["archive_sha256"],
            },
        )
    except BaseException as error:
        write_new_json(output / "failure.json", {"phase": "prepare", "error": repr(error)})
        write_manifest(
            output, problem=specification.name, run_id=run_id, status="preparation_failed"
        )
        raise
    return output


def _prepared(output: Path) -> dict:
    payload = json.loads((output / "prepared.json").read_text())
    if payload.get("schema") != PREPARED_SCHEMA:
        raise ValueError("Unsupported prepared experiment")
    validate_identity(payload["problem"], payload["run_id"])
    specification = SolverSpec(**payload["runner"])
    SolverRegistry([specification])
    if specification.name != payload["problem"]:
        raise ValueError("Runner and experiment problem identities differ")
    if file_identity(output / "config.json") != payload["config"]:
        raise ValueError("Frozen configuration changed after preparation")
    for item in payload["inputs"]:
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(output / "inputs"):
            raise ValueError("Frozen inputs must stay inside the experiment")
        if file_identity(path) != {key: item[key] for key in ("bytes", "sha256", "executable")}:
            raise ValueError(f"Frozen input changed after preparation: {item['field']}")
    if file_identity(output / "source/source.tar.gz")["sha256"] != payload["source_archive_sha256"]:
        raise ValueError("Frozen source archive changed after preparation")
    return payload


def execute_experiment(output: Path) -> int:
    """Run a prepared directory once; each child imports its own restored source."""
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Finalized experiment cannot be executed again")
    prepared = _prepared(output)
    write_new_json(output / "started.json", {"started_at_utc": _utc()})
    started = perf_counter()
    status, returncode = "failed", None
    try:
        with TemporaryDirectory(prefix="ising-execution-") as temporary:
            workspace = Path(temporary)
            source = workspace / "source"
            restored = restore(output / "source", source)
            if restored["source_tree"] != prepared["source_tree"]:
                raise ValueError("Restored source differs from prepared source")
            scratch = workspace / "scratch"
            scratch.mkdir()
            overrides = {
                "PYTHONPATH": str(source / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(scratch),
                "XDG_CACHE_HOME": str(scratch / "cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(scratch / "torchinductor"),
            }
            command = [
                sys.executable,
                "-m",
                "combinatorial_solvers.cli.experiment_worker",
                "--run-dir",
                str(output),
                "--source-root",
                str(source),
            ]
            write_new_json(
                output / "command.json",
                {"command": command, "cwd": str(source), "environment_overrides": overrides},
            )
            with (output / "run.log").open("x") as log:
                with subprocess.Popen(
                    command,
                    cwd=source,
                    env={**os.environ, **overrides},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    try:
                        returncode = process.wait()
                    except BaseException:
                        # The session belongs only to this run, including external solvers.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
            _prepared(output)
            if returncode == 0:
                status = "completed"
            elif returncode < 0:
                status = "interrupted"
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        write_new_json(output / "failure.json", {"phase": "execute", "error": repr(error)})
        raise
    finally:
        write_new_json(
            output / "run.json",
            {
                "status": status,
                "returncode": returncode,
                "finished_at_utc": _utc(),
                "launcher_wall_seconds": perf_counter() - started,
                "timing_scope": "source restoration, child process and validation; not solver time",
            },
        )
        write_manifest(
            output, problem=prepared["problem"], run_id=prepared["run_id"], status=status
        )
    return returncode


def execute_worker(output: Path, source: Path) -> None:
    """Internal child entry; domain validation and certificate semantics stay in the runner."""
    output, source = output.resolve(), source.resolve()
    if not Path(__file__).resolve().is_relative_to(source / "src"):
        raise RuntimeError("Worker imported live code instead of the restored source")
    prepared = _prepared(output)
    if source_tree_fingerprint(source) != prepared["source_tree"]:
        raise ValueError("Worker source differs from the frozen experiment")
    specification = SolverSpec(**prepared["runner"])
    config = json.loads((output / "config.json").read_text())
    for item in prepared["inputs"]:
        _set_field(config, item["field"], str(output / item["path"]))
    runner = specification.load()
    module = sys.modules[runner.__module__]
    if not Path(module.__file__).resolve().is_relative_to(source / "src"):
        raise RuntimeError("Problem runner must be part of the archived source")
    write_new_json(
        output / "execution.json",
        {
            "source_tree": prepared["source_tree"],
            "runner_module": runner.__module__,
            "runner_source": Path(module.__file__).resolve().relative_to(source).as_posix(),
            "python": sys.version,
        },
    )
    result = runner(config, output_override=output / "result", overwrite=False, repository=source)
    if Path(result).resolve() != output / "result" or not (output / "result").is_dir():
        raise ValueError("Runner did not return the assigned experiment output directory")
    if source_tree_fingerprint(source) != prepared["source_tree"]:
        raise ValueError("Executing source was modified during the experiment")
