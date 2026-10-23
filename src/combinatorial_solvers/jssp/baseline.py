"""Single-worker OR-Tools CP-SAT scheduling reference with primal/bound separation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import ortools
from ortools.sat.python import cp_model


def cp_sat(instance, *, seed=2026102301, max_seconds=10, evidence_directory=None):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("The reference seed must be a nonnegative integer")
    if not 0 < max_seconds < float("inf"):
        raise ValueError("The reference time limit must be positive and finite")
    started = perf_counter()
    model = cp_model.CpModel()
    horizon = int(instance.durations.sum())
    starts = {}
    ends = {}
    by_machine = [[] for _ in range(instance.machine_count)]
    for j in range(instance.jobs):
        for o in range(instance.machine_count):
            name = f"j{j}o{o}"
            starts[j, o] = model.new_int_var(0, horizon, "start_" + name)
            ends[j, o] = model.new_int_var(0, horizon, "end_" + name)
            interval = model.new_interval_var(
                starts[j, o], int(instance.durations[j, o]), ends[j, o], name
            )
            by_machine[int(instance.machines[j, o])].append(interval)
            if o:
                model.add(starts[j, o] >= ends[j, o - 1])
    for intervals in by_machine:
        model.add_no_overlap(intervals)
    makespan = model.new_int_var(instance.lower_bound(), horizon, "makespan")
    model.add_max_equality(
        makespan, [ends[j, instance.machine_count - 1] for j in range(instance.jobs)]
    )
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed % 2147483647)
    solver.parameters.max_time_in_seconds = float(max_seconds)
    evidence = Path(evidence_directory) if evidence_directory is not None else None
    if evidence is not None:
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "model.pbtxt").write_text(str(model.proto))
        (evidence / "parameters.pbtxt").write_text(str(solver.parameters))

    class Capture(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.objectives = []
            self.seconds = []
            self.schedules = []

        def on_solution_callback(self):
            self.objectives.append(int(self.objective_value))
            self.seconds.append(self.wall_time)
            self.schedules.append(
                [
                    [self.value(starts[j, o]) for o in range(instance.machine_count)]
                    for j in range(instance.jobs)
                ]
            )
            if evidence is not None:
                number = len(self.objectives)
                np.savez_compressed(
                    evidence / f"solution-{number}.npz",
                    starts=np.array(self.schedules[-1], dtype=np.int64),
                    makespan=np.array(self.objectives[-1]),
                    seconds=np.array(self.seconds[-1]),
                )
                with (evidence / "curve.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "number": number,
                                "makespan": self.objectives[-1],
                                "seconds": self.seconds[-1],
                                "bound": self.best_objective_bound,
                            }
                        )
                        + "\n"
                    )

    capture = Capture()
    preparation_seconds = perf_counter() - started
    status = solver.solve(model, capture)
    certification_started = perf_counter()
    if evidence is not None:
        (evidence / "response.pbtxt").write_text(str(solver.response_proto))
    arrays = {
        "curve_objective": np.array(capture.objectives, dtype=np.int64),
        "curve_seconds": np.array(capture.seconds, dtype=np.float64),
        "schedules": np.array(capture.schedules, dtype=np.int64).reshape(
            -1, instance.jobs, instance.machine_count
        ),
    }
    eligible = arrays["curve_seconds"] <= max_seconds
    arrays["curve_eligible"] = eligible
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    returned_best = None
    if feasible:
        schedule = np.array(
            [
                [solver.value(starts[j, o]) for o in range(instance.machine_count)]
                for j in range(instance.jobs)
            ]
        )
        certificate = instance.certificate(schedule)
        if not certificate["feasible"] or certificate["makespan"] != round(solver.objective_value):
            raise AssertionError("CP-SAT returned an invalid original schedule")
        arrays["best_schedule"] = schedule[None, ...]
        returned_best = certificate["makespan"]
    best = None
    if eligible.any():
        candidates = np.flatnonzero(eligible)
        selected = candidates[np.argmin(arrays["curve_objective"][eligible])]
        schedule = arrays["schedules"][selected]
        certificate = instance.certificate(schedule)
        if not certificate["feasible"] or certificate["makespan"] != int(
            arrays["curve_objective"][selected]
        ):
            raise AssertionError("CP-SAT supplied an invalid eligible incumbent")
        arrays["quality_best_schedule"] = schedule[None, ...]
        best = certificate["makespan"]
    returned_proved = status == cp_model.OPTIMAL
    proof_within_deadline = returned_proved and solver.wall_time <= max_seconds
    return arrays, {
        "method": "cp-sat",
        "solver_version": ortools.__version__,
        "solver_status": solver.status_name(status),
        "best_feasible_cost": best,
        "returned_best_cost": returned_best,
        "lower_bound": float(solver.best_objective_bound),
        "returned_optimality_proved": returned_proved,
        "optimality_proved": proof_within_deadline and best == returned_best,
        "optimality_scope": (
            "optimality_proved requires the final response inside the search deadline; "
            "lower_bound and returned_optimality_proved describe the final solver response"
        ),
        "eligible_incumbents": int(eligible.sum()),
        "deadline_overshoot_seconds": max(0.0, solver.wall_time - max_seconds),
        "preparation_seconds": preparation_seconds,
        "search_seconds": solver.wall_time,
        "certification_seconds": perf_counter() - certification_started,
        "baseline_complete_seconds": perf_counter() - started,
        "parallel_workers": 1,
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "seed": seed,
        "max_seconds": max_seconds,
        "observation": (
            "Only incumbent callback observations at or before the deadline count toward quality; "
            "all callbacks and the final returned schedule are retained"
        ),
        "recovery": (
            "Every incumbent schedule is saved; CP-SAT internal search state is not exposed"
        ),
    }
