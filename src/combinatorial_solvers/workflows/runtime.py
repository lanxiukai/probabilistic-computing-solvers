"""Shared validation and configuration helpers for command-line experiments."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from combinatorial_solvers.core.telemetry import (
    GIB,
    ReplicaTuningResult,
    TrialRunner,
    autotune_replica_count,
    torch_cuda_memory,
)


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override experiment.output_dir from the config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The configuration root must be a mapping")
    return dict(payload)


def require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return dict(value)


def resolve_device(name: str) -> torch.device:
    selected = str(name).lower()
    if selected == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_float_dtype(name: str) -> torch.dtype:
    selected = str(name).lower()
    if selected in {"float32", "fp32"}:
        return torch.float32
    if selected in {"float64", "fp64"}:
        return torch.float64
    raise ValueError("dtype must be float32/fp32 or float64/fp64")


def linear_schedule(
    config: Mapping[str, Any], *, step_key: str = "beta_steps"
) -> tuple[float, ...]:
    """Build a reproducible beta schedule, retaining the legacy function name.

    Supported kinds are ``linear`` (the default), ``geometric``, ``power``, and
    ``cyclic_linear``. Prefix experiments always slice a fully constructed source
    schedule so paired runs consume exactly the same beta values.
    """

    start = float(config["beta_start"])
    stop = float(config["beta_stop"])
    steps = int(config[step_key])
    source_steps = int(config.get("beta_schedule_steps", steps))
    if not math.isfinite(start) or not math.isfinite(stop) or start <= 0 or stop <= 0:
        raise ValueError("Beta endpoints must be finite and positive")
    if steps <= 0:
        raise ValueError(f"{step_key} must be positive")
    if source_steps <= 0:
        raise ValueError("beta_schedule_steps must be positive")
    if steps > source_steps:
        raise ValueError(f"{step_key} cannot exceed beta_schedule_steps")
    kind = str(config.get("beta_schedule_kind", "linear"))
    if source_steps == 1:
        schedule = torch.tensor([stop], dtype=torch.float64)
    elif kind == "linear":
        schedule = torch.linspace(start, stop, source_steps, dtype=torch.float64)
    elif kind == "geometric":
        schedule = torch.logspace(
            math.log10(start),
            math.log10(stop),
            source_steps,
            dtype=torch.float64,
        )
    elif kind == "power":
        power = float(config.get("beta_schedule_power", 1.0))
        if not math.isfinite(power) or power <= 0:
            raise ValueError("beta_schedule_power must be finite and positive")
        progress = torch.linspace(0.0, 1.0, source_steps, dtype=torch.float64)
        schedule = start + (stop - start) * progress.pow(power)
    elif kind == "cyclic_linear":
        cycles = int(config.get("beta_cycles", 1))
        if cycles <= 0 or cycles > source_steps:
            raise ValueError("beta_cycles must be between one and beta_schedule_steps")
        base, remainder = divmod(source_steps, cycles)
        segments = [
            torch.linspace(
                start,
                stop,
                base + (cycle < remainder),
                dtype=torch.float64,
            )
            for cycle in range(cycles)
        ]
        schedule = torch.cat(segments)
    else:
        raise ValueError("beta_schedule_kind must be linear, geometric, power, or cyclic_linear")
    return tuple(float(value) for value in schedule[:steps].tolist())


def resolve_beta_stage_sweeps(
    config: Mapping[str, Any],
    *,
    stage_count: int,
) -> tuple[int, ...]:
    """Resolve an auditable, possibly non-uniform dwell count for each beta stage."""

    if stage_count <= 0:
        raise ValueError("Beta stage count must be positive")
    default_dwell = int(config.get("sweeps_per_beta", 1))
    if default_dwell <= 0:
        raise ValueError("sweeps_per_beta must be positive")
    raw_profile = config.get("beta_dwell_profile")
    if raw_profile is None:
        return (default_dwell,) * stage_count
    if not isinstance(raw_profile, Mapping):
        raise ValueError("beta_dwell_profile must be a mapping")
    profile = dict(raw_profile)
    kind = str(profile.get("kind", ""))
    total_sweeps = int(profile.get("total_sweeps", 0))
    base_dwell = int(profile.get("base_sweeps_per_beta", default_dwell))
    if total_sweeps <= 0 or base_dwell <= 0:
        raise ValueError("Beta dwell totals and base dwell must be positive")

    if kind == "uniform":
        dwell, remainder = divmod(total_sweeps, stage_count)
        if dwell <= 0 or remainder:
            raise ValueError("A uniform beta dwell total must divide the stage count")
        return (dwell,) * stage_count

    base_total = base_dwell * stage_count
    if total_sweeps < base_total:
        raise ValueError("Beta dwell total cannot be smaller than the registered base ramp")
    extra = total_sweeps - base_total
    resolved = [base_dwell] * stage_count
    if kind == "terminal_hold":
        resolved[-1] += extra
        return tuple(resolved)
    if kind != "stage_window_hold":
        raise ValueError(
            "beta_dwell_profile.kind must be uniform, terminal_hold, or stage_window_hold"
        )

    start = int(profile.get("window_stage_start", -1))
    stop = int(profile.get("window_stage_stop", -1))
    if start < 0 or stop < start or stop >= stage_count:
        raise ValueError("Beta dwell stage window must be an inclusive in-range interval")
    window_size = stop - start + 1
    dwell_extra, remainder = divmod(extra, window_size)
    for offset, index in enumerate(range(start, stop + 1)):
        resolved[index] += dwell_extra + (offset < remainder)
    return tuple(resolved)


def validate_cuda_resource_snapshot(
    snapshot: Mapping[str, int],
    resources: Mapping[str, Any],
    *,
    check_peak: bool,
) -> None:
    """Validate a CUDA memory snapshot against explicit experiment guardrails."""

    minimum_free = int(float(resources.get("minimum_free_vram_gib", 0.0)) * GIB)
    maximum_reserved = int(float(resources.get("maximum_reserved_vram_gib", 8.0)) * GIB)
    if minimum_free < 0 or maximum_reserved <= 0:
        raise ValueError("CUDA resource guard thresholds are invalid")
    free_bytes = int(snapshot["device_free_bytes"])
    reserved_bytes = int(snapshot["process_peak_reserved_bytes"])
    if free_bytes < minimum_free:
        raise RuntimeError("CUDA device free memory is below the configured reserve")
    if check_peak and reserved_bytes > maximum_reserved:
        raise RuntimeError("CUDA process peak reserved memory exceeded the configured budget")


def experiment_output(
    config: Mapping[str, Any],
    override: Path | None,
) -> Path:
    experiment = require_mapping(config, "experiment")
    value = override if override is not None else experiment.get("output_dir")
    if value is None:
        raise ValueError("experiment.output_dir is required")
    return Path(value)


def validate_experiment_output(output: Path, overwrite: bool) -> None:
    """Refuse unknown non-empty directories, even when overwrite was requested."""

    if not output.exists() or not any(output.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    marker = output / "config.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("Overwrite requires an existing owned experiment bundle") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("Overwrite requires an existing owned experiment bundle")


def json_safe(value: Any) -> Any:
    """Convert paths, dtypes, devices, and nested mappings to JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a portable, human-editable replay configuration."""

    Path(path).write_text(
        yaml.safe_dump(json_safe(payload), sort_keys=False),
        encoding="utf-8",
    )


def tune_cuda_replicas(
    runner: TrialRunner,
    tuning: Mapping[str, Any],
    device: torch.device,
) -> ReplicaTuningResult:
    """Apply the repository-wide throughput and 8-GiB-style memory guardrails."""

    if device.type != "cuda":
        raise ValueError("Automatic replica tuning requires a CUDA device")
    result = autotune_replica_count(
        runner,
        initial_replicas=int(tuning.get("initial_replicas", 128)),
        maximum_replicas=int(tuning.get("maximum_replicas", 65_536)),
        maximum_reserved_bytes=int(float(tuning.get("maximum_vram_gib", 8.0)) * GIB),
        minimum_device_free_bytes=int(float(tuning.get("minimum_free_vram_gib", 4.0)) * GIB),
        minimum_per_chain_improvement=float(tuning.get("minimum_per_chain_improvement", 0.03)),
        repeats=int(tuning.get("repeats", 3)),
        warmup_repeats=int(tuning.get("warmup_repeats", 0)),
        non_improvement_patience=int(tuning.get("non_improvement_patience", 1)),
        free_memory_probe=lambda: torch_cuda_memory(device)["device_free_bytes"],
    )
    if result.selected_replicas is None:
        raise RuntimeError("Replica autotuning found no configuration inside the resource budget")
    return result
