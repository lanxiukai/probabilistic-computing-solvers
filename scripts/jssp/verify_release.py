"""Check portable JSSP witnesses and published claims without running a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from combinatorial_solvers.jssp.model import read_instance

KNOWN_REFERENCES = {
    "0cc5ca77665766f7829c20722a9439800ee2b34e8271ed9f0ca98c41b0c3ab3b": ("ft06", 55),
    "e10bbc52f7a7bb7604bdadfbe322af84fc7726dd9937512b89759e7ac2d9aac3": ("ft10", 930),
    "813573d84c1c42ef073ee056048dfcf460b748530e28c14fe3cc39cd11884b53": ("la01", 666),
    "99c2944e82fb9452545e19fdabe18ae5f9f5353984544dce85645bcda0923292": ("la16", 945),
    "eb6efca2f3b22f29609abbbc15acc3ec33cac91eb2b75097aad343fc6930623d": ("ta01js", 1231),
    "80be589030a44dbd2bd317e30478559a866394d34526b81d54d4bf62026e7c01": ("ta51js", 2760),
}
THINNING_RULE = "all_replica_best_improvements_and_eligibility_boundaries"


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Release paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release paths must stay inside their declared root")
    return path


def checked_file(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise AssertionError("Released file digest mismatch: " + path.name)
    return actual


def require_equal(metrics, name, expected):
    if name in metrics and metrics[name] != expected:
        raise AssertionError("Published claim differs from its witness: " + name)


def require_fields(mapping, fields):
    if not set(fields).issubset(mapping):
        raise AssertionError("Required published fields are missing")


def original_schedule_cost(instance, starts):
    """Check original precedence/capacity constraints independently of the decoder."""
    starts = np.asarray(starts)
    if starts.shape != instance.machines.shape or starts.dtype.kind not in "iu":
        raise AssertionError("A schedule must be a job-operation integer matrix")
    if (starts < 0).any():
        raise AssertionError("A schedule has a negative start time")
    intervals = [[] for _ in range(instance.machine_count)]
    makespan = 0
    for job in range(instance.jobs):
        previous_end = 0
        for operation in range(instance.machine_count):
            start = int(starts[job, operation])
            end = start + int(instance.durations[job, operation])
            if start < previous_end:
                raise AssertionError("A schedule violates original job precedence")
            machine = int(instance.machines[job, operation])
            intervals[machine].append((start, end))
            previous_end = end
            makespan = max(makespan, end)
    for machine in intervals:
        ordered = sorted(machine)
        for previous, following in zip(ordered, ordered[1:], strict=False):
            if previous[1] > following[0]:
                raise AssertionError("A schedule violates original machine capacity")
    return makespan


def sequence_costs(instance, sequences, schedule_decoder):
    values = np.asarray(sequences)
    if values.ndim != 2 or values.shape[1] != instance.size or values.dtype.kind not in "iu":
        raise AssertionError("Priority witnesses must have shape replicas by operations")
    costs, schedules = [], []
    for sequence in values:
        instance.validate_sequence(sequence)
        schedule = instance.decode(sequence, schedule_decoder=schedule_decoder)
        costs.append(original_schedule_cost(instance, schedule))
        schedules.append(schedule)
    return np.array(costs, dtype=np.int64), np.array(schedules, dtype=np.int64)


def verified_reference(configuration, instance, digest, record):
    reference = configuration["problem"].get("reference")
    if reference is None:
        return None
    if digest in KNOWN_REFERENCES:
        name, known = KNOWN_REFERENCES[digest]
        if record.get("instance", name) != name or reference != known:
            raise AssertionError("Published benchmark/reference differs from the pinned catalog")
    elif (
        configuration["problem"].get("reference_status") != "elementary_lower_bound"
        or reference != instance.lower_bound()
    ):
        raise AssertionError(
            "Reference is neither pinned nor an independently computed lower bound"
        )
    return reference


def observations(states, settings, metrics, record, replicas):
    times = np.asarray(states["curve_seconds"])
    steps = np.asarray(states["curve_steps"])
    if (
        times.ndim != 1
        or steps.shape != times.shape
        or steps.dtype.kind not in "iu"
        or not np.isfinite(times).all()
        or (times < 0).any()
        or (np.diff(times) < 0).any()
        or (steps <= 0).any()
        or (np.diff(steps) <= 0).any()
    ):
        raise AssertionError("Invalid observation times or update indices")
    count = len(times)
    chunk = settings["chunk_steps"]
    completed = metrics["completed_steps"]
    if not isinstance(chunk, int) or chunk <= 0 or not isinstance(completed, int) or completed < 0:
        raise AssertionError("Invalid declared update counts")
    total = math.ceil(completed / chunk)
    if "curve_indices" in states:
        indices = np.asarray(states["curve_indices"])
        if indices.shape != times.shape or indices.dtype.kind not in "iu":
            raise AssertionError("Invalid original curve indices")
    else:
        indices = np.arange(count)
    thinning = record.get("curve_thinning", record.get("curve_curation", {}).get("curve_"))
    if thinning is None:
        np.testing.assert_array_equal(indices, np.arange(total))
    else:
        if (
            thinning.get("rule") != THINNING_RULE
            or thinning.get("original_observations") != total
            or thinning.get("retained_observations") != count
            or thinning.get("first_reference_hit_preserved") is not True
            or "curve_indices" not in states
        ):
            raise AssertionError("Thinning must declare preserved first hits and original indices")
    if count:
        if indices[0] != 0 or indices[-1] != total - 1 or (np.diff(indices) <= 0).any():
            raise AssertionError("Thinning omitted an endpoint or reordered observations")
    elif total:
        raise AssertionError("Completed updates have no observation evidence")
    np.testing.assert_array_equal(steps, np.minimum((indices + 1) * chunk, completed))
    maximum = settings["max_seconds"]
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum <= 0:
        raise AssertionError("Invalid search deadline")
    eligible = times <= maximum
    if settings.get("quality_max_step") is not None:
        eligible &= steps <= settings["quality_max_step"]
    np.testing.assert_array_equal(eligible, states["curve_eligible"])
    changes = np.flatnonzero(eligible[1:] != eligible[:-1])
    if any(indices[i + 1] != indices[i] + 1 for i in changes):
        raise AssertionError("Thinning omitted an eligibility boundary")
    for key in ("curve_cost", "curve_best"):
        if states[key].shape != (count, replicas) or states[key].dtype.kind not in "iu":
            raise AssertionError("Curve values do not cover every declared replica")
    expected_step = int(steps[eligible][-1]) if eligible.any() else 0
    require_equal(metrics, "quality_max_step", expected_step)
    return times, eligible


def native_witness(instance, states, configuration, metrics, record, reference):
    settings = configuration["solver"]
    decoder = settings.get("schedule_decoder", "append")
    method = settings.get("method", "heatbath")
    require_fields(
        metrics,
        (
            "method",
            "pbit_heatbath_participation",
            "initial_best_cost",
            "best_feasible_cost",
            "returned_best_cost",
            "replicas_improved",
            "pbit_incremental_best_reduction",
            "parallel_replicas",
            "model_binary_variables",
            "physical_update_pbits_per_model",
            "visited_assignment_bits_min",
            "completed_steps",
            "quality_max_step",
        ),
    )
    if method not in {"heatbath", "metropolis", "plateau"}:
        raise AssertionError("Unknown native update rule")
    require_equal(metrics, "method", method)
    require_equal(metrics, "pbit_heatbath_participation", method == "heatbath")
    actual = {}
    checked = 0
    for key, cost_key in (
        ("initial_sequence", "initial_cost"),
        ("sequence", "cost"),
        ("best_sequence", "best_cost"),
        ("quality_best_sequence", "quality_best_cost"),
    ):
        costs, schedules = sequence_costs(instance, states[key], decoder)
        if states[cost_key].dtype.kind not in "iu":
            raise AssertionError("Makespan witnesses must contain exact integer values")
        np.testing.assert_array_equal(costs, states[cost_key])
        actual[key] = costs
        checked += len(costs)
        supplied = key + "_schedules"
        if supplied in states:
            for schedule in states[supplied]:
                original_schedule_cost(instance, schedule)
            np.testing.assert_array_equal(states[supplied], schedules)
    replicas = len(actual["initial_sequence"])
    if any(len(costs) != replicas for costs in actual.values()):
        raise AssertionError("Priority states disagree on the replica count")
    if settings["replicas"] != replicas or not replicas:
        raise AssertionError("Configured replica count differs from the witnesses")
    if "seed" in settings:
        generator = np.random.Generator(np.random.PCG64(settings["seed"]))
        expected = np.array(
            [
                generator.permutation(np.repeat(np.arange(instance.jobs), instance.machine_count))
                for _ in range(replicas)
            ]
        )
        np.testing.assert_array_equal(states["initial_sequence"], expected)
    initial = actual["initial_sequence"]
    quality = actual["quality_best_sequence"]
    returned = actual["best_sequence"]
    if (
        (quality > initial).any()
        or (returned > quality).any()
        or (returned > actual["sequence"]).any()
    ):
        raise AssertionError("Best-state witnesses violate incumbent ordering")
    require_equal(metrics, "initial_best_cost", int(initial.min()))
    require_equal(metrics, "best_feasible_cost", int(quality.min()))
    require_equal(metrics, "returned_best_cost", int(returned.min()))
    require_equal(metrics, "replicas_improved", int((quality < initial).sum()))
    reduction = int(initial.min() - quality.min()) if method == "heatbath" else None
    require_equal(metrics, "pbit_incremental_best_reduction", reduction)
    require_equal(metrics, "parallel_replicas", replicas)
    binary = instance.size * instance.jobs
    require_equal(metrics, "model_binary_variables", binary)
    require_equal(metrics, "physical_update_pbits_per_model", binary if method == "heatbath" else 0)
    np.testing.assert_array_equal(
        states["bits"], np.eye(instance.jobs, dtype=np.int8)[states["sequence"]]
    )
    visited = states["visited"]
    if (
        visited.shape != states["bits"].shape
        or visited.dtype.kind not in "biu"
        or not np.isin(visited, (0, 1)).all()
    ):
        raise AssertionError("Invalid participation mask")
    covered = visited.sum(axis=(1, 2))
    require_equal(metrics, "visited_assignment_bits_min", int(covered.min()))
    require_equal(metrics, "visited_assignment_bits_max", int(covered.max()))
    for name in ("replicas", "parallel_replicas"):
        require_equal(record.get("resources", {}), name, replicas)
    times, eligible = observations(states, settings, metrics, record, replicas)
    best_curve = states["curve_best"]
    lower = reference if reference is not None else instance.lower_bound()
    if (best_curve < lower).any() or (states["curve_cost"] < lower).any():
        raise AssertionError("Recorded curve contradicts a verified lower bound")
    if len(best_curve):
        if (
            (best_curve[0] > initial).any()
            or (np.diff(best_curve, axis=0) > 0).any()
            or (best_curve > states["curve_cost"]).any()
        ):
            raise AssertionError("Invalid incumbent curve")
        np.testing.assert_array_equal(best_curve[-1], returned)
    expected = best_curve[eligible][-1] if eligible.any() else initial
    np.testing.assert_array_equal(expected, quality)
    if reference is not None:
        require_fields(
            metrics,
            (
                "reference_hits",
                "initial_reference_hits",
                "first_reference_hit_seconds",
            ),
        )
        if (quality < reference).any() or (initial < reference).any():
            raise AssertionError("A witness contradicts the verified reference optimum")
        hit = quality == reference
        initial_hit = initial == reference
        first = np.full(replicas, np.nan)
        first[initial_hit] = 0.0
        for seconds, costs, admitted in zip(times, best_curve, eligible, strict=True):
            if admitted:
                first[np.isnan(first) & (costs == reference)] = seconds
        np.testing.assert_array_equal(first, states["first_reference_hit_seconds"])
        require_equal(metrics, "reference_hits", int(hit.sum()))
        require_equal(metrics, "initial_reference_hits", int(initial_hit.sum()))
        require_equal(
            metrics, "new_reference_hits_after_initialization", int((hit & ~initial_hit).sum())
        )
        require_equal(
            metrics, "first_reference_hit_seconds", float(np.nanmin(first)) if hit.any() else None
        )
    return checked, int(quality.min())


def cp_sat_witness(instance, states, configuration, metrics, reference):
    settings = configuration["baseline"]
    require_fields(
        metrics,
        (
            "method",
            "best_feasible_cost",
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
            "parallel_workers",
        ),
    )
    require_equal(metrics, "method", "cp-sat")
    maximum = settings["max_seconds"]
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum <= 0:
        raise AssertionError("Invalid CP-SAT search deadline")
    times = states["curve_seconds"]
    costs = states["curve_objective"]
    schedules = states["schedules"]
    if (
        times.ndim != 1
        or costs.shape != times.shape
        or len(schedules) != len(times)
        or costs.dtype.kind not in "iu"
    ):
        raise AssertionError("Invalid CP-SAT incumbent trace")
    if not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) < 0).any():
        raise AssertionError("Invalid CP-SAT observation times")
    for schedule, cost in zip(schedules, costs, strict=True):
        if original_schedule_cost(instance, schedule) != int(cost):
            raise AssertionError("A CP-SAT incumbent has a forged objective")
    if (np.diff(costs) > 0).any():
        raise AssertionError("CP-SAT incumbents are not monotone")
    eligible = times <= maximum
    if "curve_eligible" in states:
        np.testing.assert_array_equal(states["curve_eligible"], eligible)
    best = int(costs[eligible].min()) if eligible.any() else None
    require_equal(metrics, "best_feasible_cost", best)
    require_equal(metrics, "pbit_heatbath_participation", False)
    require_equal(metrics, "physical_update_pbits_per_model", 0)
    require_equal(metrics, "parallel_workers", 1)
    if best is not None:
        schedule = states[
            "quality_best_schedule" if "quality_best_schedule" in states else "best_schedule"
        ]
        if schedule.shape != (1, instance.jobs, instance.machine_count):
            raise AssertionError("Invalid best CP-SAT schedule shape")
        if original_schedule_cost(instance, schedule[0]) != best:
            raise AssertionError("Best CP-SAT schedule is not the eligible incumbent")
    if "best_schedule" in states:
        returned = states["best_schedule"]
        if returned.shape != (1, instance.jobs, instance.machine_count):
            raise AssertionError("Invalid returned CP-SAT schedule shape")
        returned_cost = original_schedule_cost(instance, returned[0])
        if not len(costs) or returned_cost != int(costs.min()):
            raise AssertionError("Returned CP-SAT schedule differs from its complete trace")
        require_equal(metrics, "returned_best_cost", returned_cost)
    else:
        require_equal(metrics, "returned_best_cost", None)
    if metrics.get("optimality_proved"):
        if "search_seconds" in metrics and metrics["search_seconds"] > maximum:
            raise AssertionError("CP-SAT proof completion is outside its declared deadline")
        known = reference if reference is not None else instance.lower_bound()
        if best != known:
            raise AssertionError("The public witness does not independently establish this optimum")
    return len(schedules) + int(best is not None), best


def verify(calls_file, repository):
    records = json.loads(calls_file.read_text())
    if not isinstance(records, list):
        raise ValueError("Released calls must be a list")
    results, witnesses = [], {}
    for record in records:
        if record["id"] in witnesses:
            raise AssertionError("Duplicate public call ID")
        configuration = record["configuration"]
        if isinstance(configuration, str):
            configuration = json.loads(contained(repository, configuration).read_text())
        if record.get("configuration_file"):
            executable = json.loads(contained(repository, record["configuration_file"]).read_text())
            if executable != configuration:
                raise AssertionError(
                    "Executable configuration differs from the recorded configuration"
                )
        inputs = {}
        for entry in record["benchmark_inputs"]:
            if entry["field"] in inputs:
                raise AssertionError("Duplicate benchmark input field")
            path = contained(repository, entry["path"])
            digest = checked_file(path, entry["sha256"])
            inputs[entry["field"]] = (path, digest)
        path, digest = inputs["problem.instance_file"]
        configured = configuration["problem"].get("instance_file")
        if configured and contained(repository, configured) != path:
            raise AssertionError("Configured benchmark differs from the hashed input")
        instance = read_instance(path)
        reference = verified_reference(configuration, instance, digest, record)
        witness = contained(calls_file.parent, record["witness_file"])
        checked_file(witness, record["witness_sha256"])
        with np.load(witness, allow_pickle=False) as archive:
            states = {key: archive[key] for key in archive.files}
        metrics = record["metrics"]
        for key, value in (
            ("jobs", instance.jobs),
            ("machines", instance.machine_count),
            ("operations", instance.size),
            ("reference", reference),
            ("elementary_lower_bound", instance.lower_bound()),
        ):
            require_equal(metrics, key, value)
        method = configuration.get("method", "native")
        if method == "cp-sat":
            checked, best = cp_sat_witness(instance, states, configuration, metrics, reference)
        elif method == "native":
            checked, best = native_witness(
                instance, states, configuration, metrics, record, reference
            )
        else:
            raise AssertionError("Unknown configured solver family")
        expected_gap = (
            (best - reference) / reference if reference is not None and best is not None else None
        )
        require_equal(metrics, "relative_gap", expected_gap)
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            for source in (metrics, record.get("resources", {})):
                if name in source and (not isinstance(source[name], int) or source[name] < 0):
                    raise AssertionError("Invalid reported memory count")
        results.append(
            {
                "id": record["id"],
                "witnesses_checked": checked,
                "best_feasible_cost": best,
                "verified_reference": reference,
                "curve_thinning_declared": (
                    record.get("curve_thinning") is not None
                    or "curve_" in record.get("curve_curation", {})
                ),
            }
        )
        witnesses[record["id"]] = witness
    replays = 0
    for record in records:
        origin = record.get("replay_of")
        if origin is not None:
            with (
                np.load(witnesses[origin], allow_pickle=False) as a,
                np.load(witnesses[record["id"]], allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise AssertionError("Replay witness fields differ")
                for key in a.files:
                    if key not in {"curve_seconds", "first_reference_hit_seconds"}:
                        np.testing.assert_array_equal(a[key], b[key])
            replays += 1
    return {
        "status": "verified",
        "solver_executed": False,
        "calls": len(results),
        "witnesses_checked": sum(row["witnesses_checked"] for row in results),
        "work_replays_checked": replays,
        "records": results,
        "scope": (
            "Original witnesses and recorded-claim consistency, "
            "not independent GPU timing attestation"
        ),
        "thinning_scope": (
            "First hits are recomputed from retained observations and cross-checked "
            "against saved first-hit arrays; omitted events require the separately "
            "audited full source trace"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.calls.resolve(), args.repository.resolve())
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as output:
            output.write(encoded)
    print(encoded, end="")
