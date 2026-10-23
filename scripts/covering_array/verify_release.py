"""Verify portable covering-array witnesses without importing or running a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

CURATION_RULE = "all_replica_best_improvements_and_eligibility_boundaries"
STATE_PAIRS = (
    ("initial_values", "initial_cost"),
    ("values", "cost"),
    ("best_values", "best_cost"),
    ("quality_best_values", "quality_best_cost"),
)
DEFAULTS = {
    "replicas": 64,
    "seed": 2026102301,
    "width": 1,
    "orientation": "row",
    "method": "heatbath",
    "chunk_steps": 256,
    "max_steps": 100000000,
    "max_seconds": 10,
    "quality_max_step": None,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite_number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def read_json(path):
    def invalid_constant(token):
        raise ValueError("Non-finite JSON number: " + token)

    return json.loads(path.read_text(), parse_constant=invalid_constant)


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Release paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release paths must remain inside their declared root")
    return path


def checked_file(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    require(actual == expected, "Released file digest mismatch: " + path.name)
    return actual


def claims(record, expected, required=()):
    metrics, resources = record["metrics"], record.get("resources", {})
    require(set(required).issubset(metrics), "Required published metrics are missing")
    for name, value in expected.items():
        for source in (metrics, resources):
            if name not in source:
                continue
            observed = source[name]
            same_type = (
                type(observed) is type(value)
                if type(value) in (bool, int) or value is None
                else True
            )
            require(
                same_type and observed == value,
                "Published claim differs from its witness: " + name,
            )


def dimensions(path):
    payload = read_json(path)
    rows, columns, levels = (payload[name] for name in ("rows", "columns", "levels"))
    require(
        integer(rows, 1) and integer(columns, 2) and integer(levels, 2) and levels <= 4,
        "Invalid declared covering-array dimensions",
    )
    require(
        type(payload.get("strength", 2)) is int and payload.get("strength", 2) == 2,
        "Only strength-two arrays are supported",
    )
    return rows, columns, levels


def missing_pairs(values, rows, columns, levels):
    """Enumerate all original column/symbol pairs using integer set unions."""
    require(
        values.ndim == 3
        and values.shape[1:] == (rows, columns)
        and values.shape[0] > 0
        and values.dtype.kind in "iu"
        and ((values >= 0) & (values < levels)).all(),
        "Invalid array dimensions or alphabet entries",
    )
    result = np.zeros(len(values), dtype=np.int64)
    # Each bit denotes an original symbol pair, independent of any count cache.
    for start in range(0, len(values), 16):
        block = values[start : start + 16].astype(np.uint32)
        for left in range(columns - 1):
            codes = levels * block[:, :, left, None] + block[:, :, left + 1 :]
            present = np.bitwise_or.reduce(np.left_shift(np.uint32(1), codes), axis=1)
            result[start : start + len(block)] += (levels**2 - np.bitwise_count(present)).sum(
                axis=1, dtype=np.int64
            )
    return result


def integer_array(value, shape, name, maximum=None):
    require(
        value.shape == shape and value.dtype.kind in "iu" and (value >= 0).all(),
        "Invalid nonnegative integer array: " + name,
    )
    if maximum is not None:
        require((value <= maximum).all(), "Array exceeds its legal bound: " + name)


def check_observations(states, settings, record, replicas, total_interactions):
    completed = record["metrics"]["completed_steps"]
    chunk, maximum_steps = settings["chunk_steps"], settings["max_steps"]
    require(integer(completed) and completed <= maximum_steps, "Invalid completed step count")
    require(
        completed == maximum_steps or completed % chunk == 0,
        "A completed partial chunk must end at the configured step limit",
    )
    total = (completed + chunk - 1) // chunk
    times, steps = states["curve_seconds"], states["curve_steps"]
    require(
        times.ndim == 1
        and times.dtype.kind in "iuf"
        and np.isfinite(times).all()
        and (times >= 0).all()
        and (times[1:] >= times[:-1]).all(),
        "Invalid observation times",
    )
    count = len(times)
    if count:
        integer_array(steps, (count,), "curve_steps", completed)
        require(
            (steps > 0).all() and (steps[1:] > steps[:-1]).all(),
            "Observation steps must strictly increase",
        )
    else:
        require(steps.shape == (0,), "Invalid empty observation steps")
    indices = states.get("curve_indices", np.arange(count, dtype=np.int64))
    integer_array(indices, (count,), "curve_indices")
    curation = record.get("curve_curation", {}).get("curve_")
    if curation is None:
        np.testing.assert_array_equal(indices, np.arange(total))
    else:
        require(
            curation.get("rule") == CURATION_RULE
            and type(curation.get("original_observations")) is int
            and curation["original_observations"] == total
            and type(curation.get("retained_observations")) is int
            and curation["retained_observations"] == count
            and curation.get("first_reference_hit_preserved") is True
            and "curve_indices" in states,
            "Invalid curve curation declaration",
        )
    if count:
        require(
            indices[0] == 0 and indices[-1] == total - 1 and (indices[1:] > indices[:-1]).all(),
            "Curve curation omitted an endpoint or reordered observations",
        )
        np.testing.assert_array_equal(steps, np.minimum((indices + 1) * chunk, completed))
    else:
        require(total == 0, "Completed updates have no observation evidence")
    eligible = times <= settings["max_seconds"]
    if settings["quality_max_step"] is not None:
        eligible &= steps <= settings["quality_max_step"]
    supplied = states["curve_eligible"]
    require(
        supplied.shape == (count,)
        and (not count or supplied.dtype.kind in "biu")
        and np.isin(supplied, (0, 1)).all(),
        "Invalid eligibility flags",
    )
    np.testing.assert_array_equal(supplied, eligible)
    transitions = np.flatnonzero(eligible[1:] != eligible[:-1])
    require(
        all(indices[index + 1] == indices[index] + 1 for index in transitions),
        "Curve curation omitted an eligibility boundary",
    )
    require(
        not count or (indices[times > settings["max_seconds"]] == total - 1).all(),
        "Updates continued after an observed deadline crossing",
    )
    for key in ("curve_cost", "curve_best"):
        if count:
            integer_array(states[key], (count, replicas), key, total_interactions)
        else:
            require(
                states[key].shape in ((0,), (0, replicas)),
                "Invalid cost dimensions for an empty trace",
            )
    search_seconds = record["metrics"]["search_seconds"]
    require(finite_number(search_seconds), "Invalid reported search duration")
    require(not count or times[-1] <= search_seconds, "An observation exceeds total search time")
    require(
        completed == maximum_steps or search_seconds >= settings["max_seconds"],
        "Search ended before either recorded stopping limit",
    )
    quality_step = int(steps[eligible][-1]) if eligible.any() else 0
    claims(
        record,
        {
            "quality_max_step": quality_step,
            "deadline_overshoot_seconds": max(0.0, search_seconds - settings["max_seconds"]),
        },
        required=("quality_max_step",),
    )
    return eligible, quality_step


def native_witness(record, states, configuration, shape):
    rows, columns, levels = shape
    settings = DEFAULTS | configuration["solver"]
    for name in ("replicas", "width", "chunk_steps", "max_steps"):
        require(integer(settings[name], 1), "Invalid integer setting: " + name)
    require(integer(settings["seed"]), "Invalid initialization seed")
    require(
        settings["replicas"] <= 1024 and settings["width"] in (1, 2),
        "Unsupported replica count or block width",
    )
    require(
        settings["orientation"] in ("row", "column")
        and (settings["orientation"] != "column" or (settings["width"] == 2 and rows >= 2)),
        "Invalid block orientation",
    )
    require(
        finite_number(settings["max_seconds"]) and settings["max_seconds"] > 0,
        "Invalid search deadline",
    )
    fence = settings["quality_max_step"]
    require(
        fence is None or (integer(fence) and fence <= settings["max_steps"]),
        "Invalid replay quality boundary",
    )
    method = settings["method"]
    require(method in ("heatbath", "metropolis", "plateau"), "Unknown native update rule")
    replicas, model_size = settings["replicas"], rows * columns * levels
    total_interactions = columns * (columns - 1) // 2 * levels**2
    actual = {}
    for key, cost_key in STATE_PAIRS:
        require(states[key].shape == (replicas, rows, columns), "Replica/array dimensions differ")
        actual[key] = missing_pairs(states[key], rows, columns, levels)
        integer_array(states[cost_key], (replicas,), cost_key, total_interactions)
        np.testing.assert_array_equal(actual[key], states[cost_key], err_msg=cost_key)
    generator = np.random.Generator(np.random.PCG64(settings["seed"]))
    cold = generator.integers(0, levels, size=(replicas, rows, columns), dtype=np.int64)
    np.testing.assert_array_equal(states["initial_values"], cold, err_msg="Cold initialization")
    initial, quality, returned = (
        actual[key] for key in ("initial_values", "quality_best_values", "best_values")
    )
    require(
        (quality <= initial).all()
        and (returned <= quality).all()
        and (returned <= actual["values"]).all(),
        "Best-state witnesses violate incumbent ordering",
    )
    for key in ("quality_best_values", "best_values"):
        unchanged = actual[key] == initial
        np.testing.assert_array_equal(states[key][unchanged], states["initial_values"][unchanged])
    unchanged = returned == quality
    np.testing.assert_array_equal(
        states["best_values"][unchanged], states["quality_best_values"][unchanged]
    )
    bit_shape = (replicas, rows, columns, levels)
    for name in ("bits", "visited"):
        require(states[name].dtype.kind in "biu", "Binary state has a nonintegral dtype")
        require(
            states[name].shape == bit_shape and np.isin(states[name], (0, 1)).all(),
            "Invalid binary state: " + name,
        )
    np.testing.assert_array_equal(states["bits"], np.eye(levels, dtype=np.int8)[states["values"]])
    visited_cells = states["visited"][..., 0].astype(bool)
    np.testing.assert_array_equal(
        states["visited"], np.repeat(visited_cells[..., None], levels, -1)
    )
    completed = record["metrics"]["completed_steps"]
    require(integer(completed), "Invalid completed step count")
    integer_array(states["changed"], (replicas,), "changed", completed)
    scheduled = min(completed, rows * columns)
    require(
        visited_cells.reshape(replicas, -1)[:, :scheduled].all(),
        "Visited state omits the deterministic initial sweep",
    )
    require(
        (visited_cells.sum(axis=(1, 2)) <= settings["width"] * completed).all(),
        "Visited cells exceed the number of possible updates",
    )
    if settings["width"] == 1 and completed <= rows * columns:
        require(
            not visited_cells.reshape(replicas, -1)[:, scheduled:].any(),
            "Visited state invents unscheduled single-cell updates",
        )
    for key, _ in STATE_PAIRS[1:]:
        difference = states[key] != states["initial_values"]
        require(not (difference & ~visited_cells).any(), "An unvisited cell changed value")
        require(
            (difference.sum(axis=(1, 2)) <= settings["width"] * states["changed"]).all(),
            "State differences exceed the changed-block count",
        )
    for key in ("initial_rng", "final_rng"):
        rng = states[key]
        require(
            rng.ndim == 1 and rng.size > 0 and rng.dtype == np.uint8,
            "Invalid opaque GPU RNG evidence",
        )
    require(
        states["initial_rng"].shape == states["final_rng"].shape,
        "Initial and final RNG state lengths differ",
    )
    if completed == 0:
        np.testing.assert_array_equal(states["initial_rng"], states["final_rng"])
    eligible, quality_step = check_observations(
        states, settings, record, replicas, total_interactions
    )
    if len(eligible):
        best, current = states["curve_best"], states["curve_cost"]
        require(
            (best <= current).all() and (best <= initial).all() and (best[1:] <= best[:-1]).all(),
            "Observation incumbent ordering is invalid",
        )
        np.testing.assert_array_equal(returned, best[-1])
        np.testing.assert_array_equal(actual["values"], current[-1])
        unchanged = states["changed"] == 0
        for curve in (best, current):
            require(
                (curve[:, unchanged] == initial[unchanged]).all(),
                "A zero-change replica reports a changing trajectory",
            )
    else:
        np.testing.assert_array_equal(returned, initial)
    expected_quality = states["curve_best"][eligible][-1] if eligible.any() else initial
    np.testing.assert_array_equal(quality, expected_quality, err_msg="Strict-deadline quality")
    if not eligible.any():
        np.testing.assert_array_equal(states["quality_best_values"], states["initial_values"])
    heatbath = method == "heatbath"
    hits, initial_hits = quality == 0, initial == 0
    visited = states["visited"].sum(axis=(1, 2, 3))
    expected = {
        "method": method,
        "pbit_heatbath_participation": heatbath,
        "physical_update_pbits_per_model": model_size if heatbath else 0,
        "model_binary_variables": model_size,
        "parallel_replicas": replicas,
        "zero_missing_hits": int(hits.sum()),
        "initial_zero_missing_hits": int(initial_hits.sum()),
        "new_zero_missing_hits_after_initialization": int((hits & ~initial_hits).sum()),
        "initial_best_cost": int(initial.min()),
        "best_feasible_cost": int(quality.min()),
        "returned_best_cost": int(returned.min()),
        "replicas_improved": int((quality < initial).sum()),
        "pbit_incremental_best_reduction": int(initial.min() - quality.min()) if heatbath else None,
        "visited_assignment_bits_min": int(visited.min()),
        "visited_assignment_bits_max": int(visited.max()),
        "initialization": "independent_uniform_array_entries",
        "fixed_row_target": rows,
        "fixed_row_target_met": bool(hits.any()),
    }
    claims(
        record,
        expected,
        required=(
            "method",
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
            "model_binary_variables",
            "parallel_replicas",
            "zero_missing_hits",
            "initial_best_cost",
            "best_feasible_cost",
            "returned_best_cost",
            "replicas_improved",
            "visited_assignment_bits_min",
        ),
    )
    if record["metrics"].get("optimality_proved"):
        require(rows == levels**2 and hits.any(), "No minimum-row optimality certificate")
    return {
        "witnesses_checked": 4 * replicas,
        "method": method,
        "parallel_replicas": replicas,
        "model_binary_variables": model_size,
        "physical_update_pbits_per_model": expected["physical_update_pbits_per_model"],
        "zero_missing_hits": int(hits.sum()),
        "initial_zero_missing_hits": int(initial_hits.sum()),
        "new_zero_missing_hits_after_initialization": int((hits & ~initial_hits).sum()),
        "best_feasible_cost": int(quality.min()),
        "returned_best_cost": int(returned.min()),
        "initial_best_cost": int(initial.min()),
        "pbit_incremental_best_reduction": expected["pbit_incremental_best_reduction"],
        "fixed_row_target_met": bool(hits.any()),
        "quality_max_step": quality_step,
        "visited_assignment_bits_min": int(visited.min()),
    }


def classical_witness(record, states, configuration, shape):
    rows, columns, levels = shape
    values = states["constructed_array"]
    require(values.ndim == 2 and len(values) > 0, "Invalid constructed-array dimensions")
    constructed = len(values)
    missing = int(missing_pairs(values[None, ...], constructed, columns, levels)[0])
    require(missing == 0, "ACTS output is not a complete covering array")
    target_met = constructed <= rows
    expected = {
        "method": "acts-ipog",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "model_binary_variables": 0,
        "parallel_replicas": 0,
        "parallel_workers": 1,
        "constructed_rows": constructed,
        "fixed_row_target": rows,
        "fixed_row_target_met": target_met,
        "best_feasible_cost": missing,
        "zero_missing_hits": int(target_met),
        "pbit_incremental_best_reduction": None,
    }
    claims(
        record,
        expected,
        required=(
            "method",
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
            "parallel_workers",
            "constructed_rows",
            "fixed_row_target",
            "fixed_row_target_met",
            "best_feasible_cost",
            "search_seconds",
        ),
    )
    limit = configuration["baseline"].get("process_limit_seconds", 120)
    require(finite_number(limit) and limit > 0, "Invalid ACTS process limit")
    if record["metrics"].get("optimality_proved"):
        require(constructed == levels**2, "No minimum-row optimality certificate")
    return {
        "witnesses_checked": 1,
        "method": "acts-ipog",
        "constructed_rows": constructed,
        "constructed_array_missing_interactions": missing,
        "fixed_row_target_met": target_met,
        "zero_missing_hits": int(target_met),
        "physical_update_pbits_per_model": 0,
        "pbit_incremental_best_reduction": None,
        "best_feasible_cost": missing,
        "cost_scope": "Missing interactions in the constructed array, with its own row count",
    }


def prepare_record(record, calls_file, repository):
    require(isinstance(record, dict), "Each call must be a JSON object")
    require(isinstance(record.get("id"), str) and record["id"], "Missing call ID")
    configuration = record["configuration"]
    require(isinstance(configuration, dict), "Configuration must be an embedded object")
    if "configuration_file" in record:
        path = contained(repository, record["configuration_file"])
        if "configuration_sha256" in record:
            checked_file(path, record["configuration_sha256"])
        require(
            read_json(path) == configuration, "Executable configuration differs from its record"
        )
    inputs = {}
    for group in ("benchmark_inputs", "baseline_inputs"):
        entries = record.get(group, [])
        require(isinstance(entries, list), "Input bindings must be a list")
        for entry in entries:
            field = entry["field"]
            require(isinstance(field, str) and field not in inputs, "Duplicate input field")
            path = contained(repository, entry["path"])
            checked_file(path, entry["sha256"])
            if "recorded_original_sha256" in entry:
                require(
                    entry["recorded_original_sha256"] == entry["sha256"],
                    "Released input differs from its recorded original bytes",
                )
            configured = configuration
            for part in field.split("."):
                configured = configured[part]
            require(
                contained(repository, configured) == path,
                "Configured input differs from its hashed binding",
            )
            inputs[field] = path
    require("problem.instance_file" in inputs, "Missing original instance binding")
    if configuration.get("method", "native") == "acts-ipog":
        require("baseline.jar" in inputs, "Missing ACTS executable input binding")
    witness = contained(calls_file.parent, record["witness_file"])
    checked_file(witness, record["witness_sha256"])
    require(isinstance(record.get("metrics"), dict), "Metrics must be an object")
    require(isinstance(record.get("resources", {}), dict), "Resources must be an object")
    role = record.get("evaluation_role")
    require(isinstance(role, str) and role, "Missing evaluation role")
    configured_role = configuration.get("experiment", {}).get("role")
    require(
        configured_role is None or configured_role == role,
        "Evaluation role differs from the executable configuration",
    )
    kind, origin = record.get("trial_kind"), record.get("replay_of")
    require(kind in ("natural", "work-replay"), "Unknown trial kind")
    require(
        (kind == "natural" and origin is None)
        or (kind == "work-replay" and isinstance(origin, str) and origin != record["id"]),
        "Replay linkage disagrees with the trial kind",
    )
    return configuration, inputs["problem.instance_file"], witness


def verify_replays(records, prepared):
    by_id = {record["id"]: record for record in records}
    replay_count = 0
    for record in records:
        if record["trial_kind"] != "work-replay":
            continue
        origin = record["replay_of"]
        require(
            origin in by_id and by_id[origin]["trial_kind"] == "natural",
            "A work replay must identify a natural call in the same release",
        )
        source = by_id[origin]
        original_config, original_input, original_witness = prepared[origin]
        replay_config, replay_input, replay_witness = prepared[record["id"]]
        require(original_input == replay_input, "Replay original problem differs")
        require(
            "solver" in original_config and "solver" in replay_config,
            "Only native calls have a fixed-work replay contract",
        )
        original_settings, replay_settings = (
            DEFAULTS | config["solver"] for config in (original_config, replay_config)
        )
        work_fields = {"max_steps", "max_seconds", "quality_max_step"}
        require(
            {k: v for k, v in original_settings.items() if k not in work_fields}
            == {k: v for k, v in replay_settings.items() if k not in work_fields},
            "Replay changed the method, seed, or update settings",
        )
        require(
            replay_settings["max_steps"] == source["metrics"]["completed_steps"]
            and replay_settings["quality_max_step"] == source["metrics"]["quality_max_step"],
            "Replay does not preserve original work and quality boundaries",
        )
        with (
            np.load(original_witness, allow_pickle=False) as a,
            np.load(replay_witness, allow_pickle=False) as b,
        ):
            require(set(a.files) == set(b.files), "Replay witness fields differ")
            for name in a.files:
                if name != "curve_seconds":
                    np.testing.assert_array_equal(a[name], b[name], err_msg="Work replay: " + name)
        replay_count += 1
    return replay_count


def verify_panel(panel, records, prepared, replays):
    natural = [record for record in records if record["trial_kind"] == "natural"]
    expected = {
        "problem": "covering-array",
        "status": "independently-verified",
        "planned_natural_calls": len(natural),
        "completed_natural_calls": len(natural),
        "failed_natural_calls": 0,
        "unexecuted_natural_calls": 0,
        "work_replays": replays,
        "replays_add_independent_trials": False,
    }
    require(
        isinstance(panel, dict)
        and all(type(panel.get(k)) is type(v) and panel[k] == v for k, v in expected.items()),
        "Panel denominators or replay accounting differ from released calls",
    )
    natural_ids = panel.get("expected_natural_ids")
    replay_ids = panel.get("expected_replays")
    require(
        isinstance(natural_ids, list)
        and all(isinstance(value, str) and value for value in natural_ids)
        and len(set(natural_ids)) == len(natural_ids)
        and set(natural_ids) == {record["id"] for record in natural},
        "Complete frozen natural-call roster differs",
    )
    require(
        isinstance(replay_ids, list)
        and all(
            isinstance(value, dict)
            and all(isinstance(value.get(key), str) and value[key] for key in ("id", "origin"))
            for value in replay_ids
        )
        and len(replay_ids) == replays
        and {value["id"]: value["origin"] for value in replay_ids}
        == {
            record["id"]: record["replay_of"]
            for record in records
            if record["trial_kind"] == "work-replay"
        },
        "Complete frozen replay roster differs",
    )
    require(
        all(isinstance(record.get("instance"), str) and record["instance"] for record in records),
        "Missing frozen target instance identifier",
    )
    targets = panel.get("frozen_targets")
    require(
        isinstance(targets, dict) and set(targets) == {record["instance"] for record in records},
        "Frozen target instances differ",
    )
    identities = {}
    for record in records:
        instance = record["instance"]
        target = targets[instance]
        require(
            isinstance(target, dict)
            and type(target.get("cost")) is int
            and target["cost"] == 0
            and target.get("status") == "fixed-row-feasibility"
            and isinstance(target.get("source"), str)
            and bool(target["source"].strip())
            and all(integer(target.get(key), 1) for key in ("rows", "columns", "levels"))
            and type(target.get("strength")) is int
            and target["strength"] == 2,
            "Frozen target lacks fixed-row feasibility dimensions, status or source",
        )
        original = prepared[record["id"]][1]
        require(
            tuple(target[key] for key in ("rows", "columns", "levels")) == dimensions(original),
            "Frozen target dimensions differ from the original input",
        )
        identity = hashlib.sha256(original.read_bytes()).hexdigest()
        require(
            identities.setdefault(instance, identity) == identity,
            "Calls disagree on original instance bytes",
        )
    return len(targets)


def verify(calls_file, repository):
    calls_file, repository = Path(calls_file).resolve(), Path(repository).resolve()
    records = read_json(calls_file)
    require(isinstance(records, list) and records, "Released calls must be a nonempty list")
    prepared = {}
    # Authenticate every bound input and witness before interpreting array contents.
    for record in records:
        item = prepare_record(record, calls_file, repository)
        require(record["id"] not in prepared, "Duplicate public call ID")
        prepared[record["id"]] = item
    results = []
    for record in records:
        configuration, original, witness = prepared[record["id"]]
        shape = dimensions(original)
        rows, columns, levels = shape
        claims(
            record,
            {
                "rows": rows,
                "columns": columns,
                "levels": levels,
                "strength": 2,
                "total_interactions": columns * (columns - 1) // 2 * levels**2,
                "status": "completed",
            },
            required=("rows", "columns", "levels", "strength", "total_interactions", "status"),
        )
        with np.load(witness, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ member")
            states = {name: archive[name] for name in archive.files}
        family = configuration.get("method", "native")
        if family == "native":
            result = native_witness(record, states, configuration, shape)
        elif family == "acts-ipog":
            result = classical_witness(record, states, configuration, shape)
        else:
            raise AssertionError("Unknown configured method family")
        for source in (record["metrics"], record.get("resources", {})):
            for name, value in source.items():
                if name.endswith("_seconds") and value is not None:
                    require(finite_number(value), "Invalid reported duration: " + name)
                if name.endswith("_bytes"):
                    require(integer(value), "Invalid reported memory count: " + name)
        for name, value in record.get("resources", {}).items():
            if name in record["metrics"]:
                require(value == record["metrics"][name], "Metrics/resources disagree: " + name)
        result.update(
            id=record["id"],
            trial_kind=record["trial_kind"],
            evaluation_role=record["evaluation_role"],
            fixed_row_target=rows,
            row_lower_bound=levels**2,
            curve_curation_declared="curve_" in record.get("curve_curation", {}),
        )
        results.append(result)
    replays = verify_replays(records, prepared)
    natural = [record for record in records if record["trial_kind"] == "natural"]
    targets = verify_panel(read_json(calls_file.parent / "panel.json"), records, prepared, replays)
    return {
        "status": "verified",
        "solver_executed": False,
        "calls": len(records),
        "natural_calls": len(natural),
        "work_replays_checked": replays,
        "frozen_targets_checked": targets,
        "replays_add_independent_trials": False,
        "natural_calls_by_evaluation_role": dict(Counter(r["evaluation_role"] for r in natural)),
        "work_replays_by_evaluation_role": dict(
            Counter(r["evaluation_role"] for r in records if r["trial_kind"] == "work-replay")
        ),
        "witnesses_checked": sum(result["witnesses_checked"] for result in results),
        "records": results,
        "scope": (
            "Original pair coverage and recorded-claim consistency; no solver execution, "
            "independent GPU timing attestation, or proof of heat-bath benefit over controls"
        ),
        "curve_scope": (
            "Retained incumbent points, endpoints and eligibility boundaries are checked. "
            "Omitted current-state trajectories and preservation of every improvement require "
            "the separately audited complete trace. RNG bytes are compared on work replay."
        ),
        "cost_scope": (
            "best_feasible_cost is the historical field name for missing-interaction count. "
            "Only zero certifies coverage; ACTS own-row coverage and the fixed-row target "
            "are reported separately. Coverage alone does not prove minimum row count."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.calls, args.repository)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
