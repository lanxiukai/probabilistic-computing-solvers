"""Exact earliest-gap insertion decoding for two-position conditional updates."""

import triton
import triton.language as tl


@triton.jit
def active_energy(
    Sequence,
    Machines,
    Durations,
    Patterns,
    replica,
    rows,
    old_jobs,
    J: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    JJ: tl.constexpr,
    MM: tl.constexpr,
):
    patterns = tl.arange(0, PP)
    block_slots = tl.arange(0, B)
    jobs = tl.arange(0, JJ)
    machines = tl.arange(0, MM)
    interval_slots = tl.arange(0, JJ)
    operation = tl.full((PP, JJ), 0, tl.int32)
    job_ready = tl.full((PP, JJ), 0, tl.int32)
    machine_counts = tl.full((PP, MM), 0, tl.int32)
    interval_starts = tl.full((PP, MM, JJ), 2147483647, tl.int32)
    interval_ends = tl.full((PP, MM, JJ), 0, tl.int32)
    for position in range(L):
        job = tl.full((PP,), 0, tl.int32) + tl.load(Sequence + replica * L + position)
        for a in tl.static_range(B):
            row = tl.sum(tl.where(block_slots == a, rows, 0), 0)
            index = tl.load(Patterns + patterns * B + a, patterns < P, 0)
            proposed_job = tl.gather(old_jobs, index, 0)
            job = tl.where(position == row, proposed_job, job)
        job_index = job[:, None].to(tl.int32)
        op = tl.gather(operation, job_index, 1).reshape(PP)
        machine = tl.load(Machines + job * M + op, patterns < P, 0).to(tl.int32)
        duration = tl.load(Durations + job * M + op, patterns < P, 0).to(tl.int32)
        ready = tl.gather(job_ready, job_index, 1).reshape(PP)
        machine_index = tl.broadcast_to(machine[:, None, None], (PP, 1, JJ))
        starts = tl.gather(interval_starts, machine_index, 1).reshape(PP, JJ)
        ends = tl.gather(interval_ends, machine_index, 1).reshape(PP, JJ)
        count = tl.gather(machine_counts, machine[:, None], 1).reshape(PP)
        previous_index = tl.broadcast_to(tl.maximum(interval_slots - 1, 0)[None, :], (PP, JJ))
        previous_ends = tl.gather(ends, previous_index, 1)
        previous_ends = tl.where(interval_slots[None, :] == 0, 0, previous_ends)
        candidate_start = tl.maximum(ready[:, None], previous_ends)
        valid_gap = (interval_slots[None, :] <= count[:, None]) & (
            candidate_start + duration[:, None] <= starts
        )
        insertion = tl.min(tl.where(valid_gap, interval_slots[None, :], JJ), 1)
        start = tl.gather(candidate_start, insertion[:, None], 1).reshape(PP)
        end = start + duration
        shifted_starts = tl.gather(starts, previous_index, 1)
        shifted_ends = tl.gather(ends, previous_index, 1)
        new_starts = tl.where(interval_slots[None, :] > insertion[:, None], shifted_starts, starts)
        new_ends = tl.where(interval_slots[None, :] > insertion[:, None], shifted_ends, ends)
        new_starts = tl.where(
            interval_slots[None, :] == insertion[:, None], start[:, None], new_starts
        )
        new_ends = tl.where(interval_slots[None, :] == insertion[:, None], end[:, None], new_ends)
        selected_machine = machines[None, :, None] == machine[:, None, None]
        interval_starts = tl.where(selected_machine, new_starts[:, None, :], interval_starts)
        interval_ends = tl.where(selected_machine, new_ends[:, None, :], interval_ends)
        operation += (jobs[None, :] == job[:, None]).to(tl.int32)
        job_ready = tl.where(jobs[None, :] == job[:, None], end[:, None], job_ready)
        machine_counts += (machines[None, :] == machine[:, None]).to(tl.int32)
    return tl.max(job_ready, 1)
