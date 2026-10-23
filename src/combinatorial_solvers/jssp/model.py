"""Classic job-shop instances, serial decoding and independent schedule checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    machines: np.ndarray
    durations: np.ndarray

    def __post_init__(self):
        m, d = np.asarray(self.machines), np.asarray(self.durations)
        if (
            m.ndim != 2
            or m.shape != d.shape
            or min(m.shape) < 2
            or m.dtype.kind not in "iu"
            or d.dtype.kind not in "iu"
        ):
            raise ValueError("Rectangular integer job-operation arrays are required")
        if not (d > 0).all() or int(d.max()) * d.size >= 2**31:
            raise ValueError("Durations exceed the positive exact-int32 scheduling envelope")
        if not all(np.array_equal(np.sort(row), np.arange(m.shape[1])) for row in m):
            raise ValueError("This model requires each job to visit every machine exactly once")
        for key, value in (("machines", m), ("durations", d)):
            value = value.astype(np.int64, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def jobs(self):
        return len(self.machines)

    @property
    def machine_count(self):
        return self.machines.shape[1]

    @property
    def size(self):
        return self.machines.size

    def validate_sequence(self, sequence):
        s = np.asarray(sequence)
        if (
            s.shape != (self.size,)
            or s.dtype.kind not in "iu"
            or not ((s >= 0) & (s < self.jobs)).all()
            or not np.all(np.bincount(s, minlength=self.jobs) == self.machine_count)
        ):
            raise ValueError("Priority sequence must contain each job exactly once per operation")
        return s

    def decode(self, sequence, *, schedule_decoder="append"):
        if schedule_decoder not in {"append", "active"}:
            raise ValueError("Unknown schedule decoder")
        s = self.validate_sequence(sequence)
        next_operation = [0] * self.jobs
        job_ready = [0] * self.jobs
        machine_ready = [0] * self.machine_count
        intervals = [[] for _ in range(self.machine_count)]
        starts = np.zeros_like(self.machines)
        for job_raw in s:
            job = int(job_raw)
            op = next_operation[job]
            machine = int(self.machines[job, op])
            duration = int(self.durations[job, op])
            if schedule_decoder == "active":
                start = job_ready[job]
                insertion = 0
                for before, after in intervals[machine]:
                    if start + duration <= before:
                        break
                    start = max(start, after)
                    insertion += 1
                end = start + duration
                intervals[machine].insert(insertion, (start, end))
            else:
                start = max(job_ready[job], machine_ready[machine])
                end = start + duration
            starts[job, op] = start
            job_ready[job] = end
            machine_ready[machine] = max(machine_ready[machine], end)
            next_operation[job] += 1
        return starts

    def certificate(self, starts):
        s = np.asarray(starts)
        valid = s.shape == self.machines.shape and s.dtype.kind in "iu" and bool((s >= 0).all())
        if not valid:
            return {"feasible": False, "makespan": None}
        for job in range(self.jobs):
            if any(
                int(s[job, op]) + int(self.durations[job, op]) > int(s[job, op + 1])
                for op in range(self.machine_count - 1)
            ):
                return {"feasible": False, "makespan": None}
        for machine in range(self.machine_count):
            operations = sorted(
                (int(s[j, o]), int(s[j, o]) + int(self.durations[j, o]))
                for j, o in zip(*np.where(self.machines == machine), strict=True)
            )
            if any(
                end > next_start
                for (_, end), (next_start, _) in zip(operations, operations[1:], strict=False)
            ):
                return {"feasible": False, "makespan": None}
        return {
            "feasible": True,
            "makespan": max(
                int(s[j, o]) + int(self.durations[j, o])
                for j in range(self.jobs)
                for o in range(self.machine_count)
            ),
        }

    def lower_bound(self):
        return max(
            int(self.durations.sum(axis=1).max()),
            max(int(self.durations[self.machines == m].sum()) for m in range(self.machine_count)),
        )


def read_instance(path: Path):
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if "machines" in payload and "durations" in payload:
            return Instance(np.asarray(payload["machines"]), np.asarray(payload["durations"]))
        raise ValueError("JSON requires explicit machines and durations fields")
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and all(token.lstrip("-").isdigit() for token in fields):
            rows.append([int(x) for x in fields])
    jobs, machines = rows[0]
    values = np.asarray(rows[1:])
    if values.shape != (jobs, 2 * machines):
        raise ValueError("Expected one machine-duration pair per operation")
    return Instance(values[:, ::2], values[:, 1::2])
