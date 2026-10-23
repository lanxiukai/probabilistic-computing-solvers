"""GPU snapshots and bounded replica-count autotuning."""

from __future__ import annotations

import csv
import io
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import torch

GIB = 1024**3


@dataclass(frozen=True)
class GpuSnapshot:
    name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization_percent: float


@dataclass(frozen=True)
class ReplicaTrial:
    replicas: int
    seconds: float | None
    seconds_per_chain: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    device_free_bytes: int | None
    accepted: bool
    stop_reason: str | None


@dataclass(frozen=True)
class ReplicaTuningResult:
    selected_replicas: int | None
    trials: tuple[ReplicaTrial, ...]


def query_nvidia_smi(timeout_seconds: float = 5.0) -> GpuSnapshot | None:
    """Return a single-GPU snapshot, or ``None`` when NVML is unavailable."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    rows = list(csv.reader(io.StringIO(completed.stdout.strip())))
    if not rows:
        return None
    name, total_mib, used_mib, free_mib, utilization = [value.strip() for value in rows[0]]
    return GpuSnapshot(
        name=name,
        total_bytes=int(float(total_mib) * 1024**2),
        used_bytes=int(float(used_mib) * 1024**2),
        free_bytes=int(float(free_mib) * 1024**2),
        utilization_percent=float(utilization),
    )


def torch_cuda_memory(device: torch.device | str = "cuda") -> dict[str, int]:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA memory is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
    return {
        "device_total_bytes": int(total_bytes),
        "device_free_bytes": int(free_bytes),
        "process_allocated_bytes": int(torch.cuda.memory_allocated(selected)),
        "process_reserved_bytes": int(torch.cuda.memory_reserved(selected)),
        "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
        "process_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
    }


TrialRunner = Callable[[int], tuple[float, int, int]]
FreeMemoryProbe = Callable[[], int]


def autotune_replica_count(
    runner: TrialRunner,
    *,
    initial_replicas: int = 128,
    maximum_replicas: int = 65536,
    maximum_reserved_bytes: int = 8 * GIB,
    minimum_device_free_bytes: int = 4 * GIB,
    minimum_per_chain_improvement: float = 0.03,
    repeats: int = 3,
    warmup_repeats: int = 0,
    non_improvement_patience: int = 1,
    free_memory_probe: FreeMemoryProbe | None = None,
) -> ReplicaTuningResult:
    """Double replicas until memory or average per-chain time stops improving.

    ``runner`` returns ``(seconds, peak_allocated_bytes, peak_reserved_bytes)`` for one
    bounded probe. Warm-up runs are excluded from timing but included in peak-memory checks;
    measured probes use median time. ``non_improvement_patience`` permits noisy intermediate
    sizes while still selecting the best observed per-chain time. A trial that exceeds a
    memory budget or raises CUDA OOM is recorded but never selected.
    """

    if initial_replicas <= 0 or maximum_replicas < initial_replicas:
        raise ValueError("Replica-count bounds are invalid")
    if (
        maximum_reserved_bytes <= 0
        or minimum_device_free_bytes < 0
        or not 0 <= minimum_per_chain_improvement < 1
        or repeats <= 0
        or warmup_repeats < 0
        or non_improvement_patience <= 0
    ):
        raise ValueError("Replica tuning thresholds are invalid")

    trials: list[ReplicaTrial] = []
    selected: int | None = None
    best_per_chain: float | None = None
    consecutive_non_improvements = 0
    replicas = initial_replicas
    while replicas <= maximum_replicas:
        device_free = free_memory_probe() if free_memory_probe is not None else None
        if device_free is not None and device_free < minimum_device_free_bytes:
            trials.append(
                ReplicaTrial(
                    replicas=replicas,
                    seconds=None,
                    seconds_per_chain=None,
                    peak_allocated_bytes=None,
                    peak_reserved_bytes=None,
                    device_free_bytes=device_free,
                    accepted=False,
                    stop_reason="device_free_memory_below_reserve",
                )
            )
            break

        timings: list[float] = []
        allocated_values: list[int] = []
        reserved_values: list[int] = []
        out_of_memory = False
        for run_index in range(warmup_repeats + repeats):
            try:
                seconds_value, allocated_value, reserved_value = runner(replicas)
            except torch.cuda.OutOfMemoryError:
                out_of_memory = True
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                out_of_memory = True
                break
            if (
                not math.isfinite(float(seconds_value))
                or float(seconds_value) <= 0
                or int(allocated_value) < 0
                or int(reserved_value) < 0
                or int(allocated_value) > int(reserved_value)
            ):
                raise ValueError("Replica runner returned invalid timing or memory data")
            if run_index >= warmup_repeats:
                timings.append(float(seconds_value))
            allocated_values.append(int(allocated_value))
            reserved_values.append(int(reserved_value))
        if out_of_memory:
            trials.append(
                ReplicaTrial(
                    replicas=replicas,
                    seconds=None,
                    seconds_per_chain=None,
                    peak_allocated_bytes=max(allocated_values, default=None),
                    peak_reserved_bytes=max(reserved_values, default=None),
                    device_free_bytes=device_free,
                    accepted=False,
                    stop_reason="cuda_out_of_memory",
                )
            )
            break

        seconds = float(median(timings))
        peak_allocated = max(allocated_values)
        peak_reserved = max(reserved_values)
        per_chain = seconds / replicas
        inside_memory_budget = peak_reserved <= maximum_reserved_bytes
        accepted = inside_memory_budget
        stop_reason = None
        if not inside_memory_budget:
            stop_reason = "memory_budget_exceeded"
        elif best_per_chain is not None:
            improvement = (best_per_chain - per_chain) / best_per_chain
            if improvement <= minimum_per_chain_improvement:
                accepted = False
                stop_reason = "per_chain_time_did_not_improve"
        trials.append(
            ReplicaTrial(
                replicas=replicas,
                seconds=float(seconds),
                seconds_per_chain=float(per_chain),
                peak_allocated_bytes=int(peak_allocated),
                peak_reserved_bytes=int(peak_reserved),
                device_free_bytes=device_free,
                accepted=accepted,
                stop_reason=stop_reason,
            )
        )
        if not inside_memory_budget:
            break
        if accepted:
            selected = replicas
            best_per_chain = per_chain
            consecutive_non_improvements = 0
        else:
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= non_improvement_patience:
                break
        if replicas > maximum_replicas // 2:
            break
        replicas *= 2
    return ReplicaTuningResult(selected_replicas=selected, trials=tuple(trials))
