"""Verify released covering witnesses with NumPy and integer arithmetic, without solving."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Release paths must remain relative to their declared root")
    return path


def checked_file(path, expected):
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected,
        "Released file digest mismatch: " + path.name,
    )


def read_instance(path, format):
    """Parse the original row-oriented SCP or column-oriented railway text."""
    require(format in {"scp", "rail"}, "Unsupported original covering format")
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    tokens = iter(raw.decode("ascii").split())

    def take():
        try:
            return int(next(tokens))
        except StopIteration as error:
            raise ValueError("Truncated original covering instance") from error

    m, n = take(), take()
    require(m > 0 and n > 0, "Invalid original covering dimensions")
    rows = [[] for _ in range(m)]
    if format == "scp":
        costs = [take() for _ in range(n)]
        for row in rows:
            degree = take()
            require(0 < degree <= n, "Invalid original row degree")
            row.extend(take() - 1 for _ in range(degree))
    else:
        costs = []
        for column in range(n):
            costs.append(take())
            degree = take()
            require(0 <= degree <= m, "Invalid original column degree")
            for _ in range(degree):
                row = take() - 1
                require(0 <= row < m, "Original row index is out of range")
                rows[row].append(column)
    require(next(tokens, None) is None, "Trailing original instance tokens")
    require(all(c > 0 for c in costs), "Original costs must be positive integers")
    require(sum(costs) <= np.iinfo(np.int64).max, "Original costs exceed exact integer range")
    require(
        all(row and len(set(row)) == len(row) and all(0 <= j < n for j in row) for row in rows),
        "Original incidence rows are empty, duplicated or out of range",
    )
    return np.asarray(costs, dtype=np.int64), rows, hashlib.sha256(raw).hexdigest()


def original_values(bits, costs, rows):
    """Recompute original integer costs and every row's coverage in bounded batches."""
    values = np.asarray(bits)
    n = len(costs)
    require(
        values.ndim >= 1
        and values.shape[-1] == n
        and values.dtype.kind in "biu"
        and np.isin(values, (0, 1)).all(),
        "Cover witness is not a binary array of the original width",
    )
    flat = values.reshape(-1, n)
    columns = np.asarray([j for row in rows for j in row], dtype=np.int64)
    starts = np.r_[0, np.cumsum([len(row) for row in rows])[:-1]]
    result = flat.astype(np.int64) @ costs
    coverage = np.empty((len(flat), len(rows)), dtype=np.int64)
    for start in range(0, len(flat), 16):
        block = flat[start : start + 16]
        coverage[start : start + len(block)] = np.add.reduceat(
            block[:, columns], starts, axis=1, dtype=np.int64
        )
    return result.reshape(values.shape[:-1]), coverage.reshape(values.shape[:-1] + (len(rows),))


def check_claims(record, expected, required=()):
    metrics = record["metrics"]
    require(all(key in metrics for key in required), "A required public metric is missing")
    for source in (metrics, record.get("resources", {})):
        for key, value in expected.items():
            if key in source:
                require(source[key] == value, "Published metric differs: " + key)


def verify_dual(states, certificate, costs, rows):
    """Use Python integers for all rational inequalities, without a floating tolerance."""
    numerators = states["dual_numerators"]
    denominator = certificate["denominator"]
    require(
        numerators.shape == (len(rows),)
        and numerators.dtype.kind in "iu"
        and (numerators >= 0).all()
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
        and denominator > 0,
        "Invalid rational dual witness",
    )
    column_sums = [0] * len(costs)
    for value, row in zip(numerators, rows, strict=True):
        for column in row:
            column_sums[column] += int(value)
    require(
        all(
            total <= denominator * int(cost) for total, cost in zip(column_sums, costs, strict=True)
        ),
        "Rational dual violates an original column inequality",
    )
    numerator = sum(map(int, numerators))
    lower = (numerator + denominator - 1) // denominator
    require(
        certificate
        == {
            "numerator": numerator,
            "denominator": denominator,
            "integer_cost_lower_bound": lower,
            "method": "rational-cover-dual-verification",
        },
        "Published rational lower bound differs from its witness",
    )
    return lower


def verify_observations(record, states, initial, final, best):
    settings, metrics = record["configuration"]["solver"], record["metrics"]
    indices = states["observation_indices"]
    steps, times = states["observation_steps"], states["observation_seconds"]
    count, chunk = metrics["completed_chunks"], settings["chunk_steps"]
    completed = metrics["completed_steps"]
    require(
        count > 0 and chunk > 0 and 0 < completed <= settings["max_steps"], "Invalid work count"
    )
    require(count == (completed + chunk - 1) // chunk, "Completed chunk count differs")
    require(
        indices.ndim == 1
        and indices.dtype.kind in "iu"
        and len(indices) > 0
        and indices[0] == 0
        and indices[-1] == count - 1
        and (np.diff(indices) > 0).all(),
        "Retained observation indices omit or reorder original endpoints",
    )
    np.testing.assert_array_equal(steps, np.minimum((indices + 1) * chunk, completed))
    require(
        times.shape == indices.shape
        and np.isfinite(times).all()
        and (times >= 0).all()
        and (np.diff(times) >= 0).all(),
        "Invalid observation times",
    )
    curation = record["curve_curation"]["observation_"]
    require(
        curation["original_observations"] == count
        and curation["retained_observations"] == len(indices)
        and curation["rule"] == "all_replica_best_improvements_and_eligibility_boundaries"
        and curation["first_reference_hit_preserved"] is True,
        "Observation curation contract differs",
    )
    fence = settings.get("quality_max_step")
    if fence is None:
        eligible = times <= settings["max_seconds"]
    else:
        require(0 <= fence <= settings["max_steps"], "Invalid fixed-work quality fence")
        eligible = steps <= fence
    saved = states["observation_eligible"]
    require(saved.shape == indices.shape and np.isin(saved, (0, 1)).all(), "Invalid eligibility")
    np.testing.assert_array_equal(saved.astype(bool), eligible)
    changes = np.flatnonzero(eligible[1:] != eligible[:-1])
    require(
        (indices[changes + 1] == indices[changes] + 1).all(),
        "Curated observations omit an eligibility boundary neighbor",
    )
    last_eligible = int(steps[eligible][-1]) if eligible.any() else 0
    require(last_eligible == metrics["quality_max_step"], "Published quality fence differs")
    current, running = states["observation_cost"], states["observation_best_cost"]
    violations = states["observation_violations"]
    shape = (len(indices), len(initial))
    for values in (current, running, violations):
        require(values.shape == shape and values.dtype.kind in "iu", "Invalid observation matrix")
    require(
        (current > 0).all() and (violations == 0).all(), "Feasible trace contains an invalid state"
    )
    previous = initial.copy()
    for cost, allowed, recorded in zip(current, eligible, running, strict=True):
        if allowed:
            previous = np.minimum(previous, cost)
        np.testing.assert_array_equal(recorded, previous)
    np.testing.assert_array_equal(previous, best)
    np.testing.assert_array_equal(current[-1], final)
    return eligible


def verify_call(record, repository, calls_directory):
    configuration = record["configuration"]
    if isinstance(configuration, str):
        configuration = json.loads(contained(repository, configuration).read_text())
        record = {**record, "configuration": configuration}
    if "configuration_file" in record:
        saved = json.loads(contained(repository, record["configuration_file"]).read_text())
        require(saved == configuration, "Published configuration copies differ")
    problem, metrics = configuration["problem"], record["metrics"]
    entries = record["benchmark_inputs"]
    require(
        len(entries) == 1 and entries[0]["field"] == "problem.instance_file",
        "Input binding differs",
    )
    entry = entries[0]
    path = contained(repository, entry["path"])
    require(path == contained(repository, problem["instance_file"]), "Configuration input differs")
    checked_file(path, entry["sha256"])
    costs, rows, raw_digest = read_instance(path, problem.get("format", "scp"))
    if "raw_sha256" in entry:
        require(raw_digest == entry["raw_sha256"], "Original decompressed bytes differ")
    require(problem["benchmark_id"] == record["instance"], "Published instance identity differs")
    witness = contained(calls_directory, record["witness_file"])
    checked_file(witness, record["witness_sha256"])
    with np.load(witness, allow_pickle=False) as archive:
        states = {key: archive[key] for key in archive.files}
    native = configuration["method"] == "native"
    method = configuration["solver"]["method"] if native else configuration["method"]
    require(
        method in {"heatbath", "metropolis", "plateau", "cft", "milp"}, "Unknown release method"
    )
    participation = method == "heatbath"
    replicas = configuration["solver"]["replicas"] if native else 1
    expected = {
        "method": {"cft": "ac-cft", "milp": "scipy-highs-milp-and-lp"}.get(method, method),
        "model_binary_variables": len(costs),
        "physical_update_pbits_per_model": len(costs) if participation else 0,
        "pbit_heatbath_participation": participation,
        "rows": len(rows),
        "incidences": sum(map(len, rows)),
        "replicas": replicas,
        "status": "completed",
    }
    actual, checked = {}, 0
    keys = (
        ("initial_bits", "final_bits", "best_bits", "best_energy_bits")
        if native
        else ("best_bits",)
    )
    for key in keys:
        require(states[key].shape == (replicas, len(costs)), "Witness replica dimensions differ")
        values, coverage = original_values(states[key], costs, rows)
        require((coverage > 0).all(), "Published cover leaves an original row uncovered")
        actual[key] = values
        checked += values.size
        prefix = key.removesuffix("_bits")
        for name, computed in ((prefix + "_cost", values), (prefix + "_coverage", coverage)):
            if name in states:
                np.testing.assert_array_equal(states[name], computed)
    best = actual["best_bits"]
    expected["best_feasible_cost"] = int(best.min())
    initial = None
    if native:
        settings = configuration["solver"]
        require(settings["representation"] == "feasible", "Release requires the feasible route")
        initial = actual["initial_bits"]
        require((best <= initial).all(), "Quality witnesses are worse than initialization")
        np.testing.assert_array_equal(states["best_cost"], best)
        np.testing.assert_array_equal(states["final_cost"], actual["final_bits"])
        np.testing.assert_array_equal(states["best_energy"], actual["best_energy_bits"])
        np.testing.assert_array_equal(actual["best_energy_bits"], best)
        visited = states["visited"]
        require(
            visited.shape == (replicas, len(costs)) and np.isin(visited, (0, 1)).all(),
            "Invalid binary visitation mask",
        )
        for key in ("final_violations", "final_cardinality"):
            if key in states:
                value = (
                    np.zeros(replicas, dtype=np.int64)
                    if key.endswith("violations")
                    else states["final_bits"].sum(axis=1)
                )
                np.testing.assert_array_equal(states[key], value)
        expected.update(
            initial_best_cost=int(initial.min()),
            feasible_replicas=replicas,
            minimum_visited_variables_per_replica=int(visited.sum(axis=1).min()),
            auxiliary_pbits=0,
            representation="feasible",
            seed=settings["seed"],
        )
        eligible = verify_observations(record, states, initial, actual["final_bits"], best)
    else:
        eligible = None
    dual = metrics.get("rational_dual_certificate")
    require(
        (dual is not None) == ("dual_numerators" in states), "Dual certificate and witness differ"
    )
    lower = verify_dual(states, dual, costs, rows) if dual is not None else None
    if lower is not None:
        require(lower <= int(best.min()), "Certified lower bound exceeds a feasible witness")
        expected["independent_optimality_proven"] = lower == int(best.min())
    check_claims(
        record,
        expected,
        (
            "best_feasible_cost",
            "model_binary_variables",
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
        ),
    )
    return {
        "id": record["id"],
        "instance": record["instance"],
        "method": method,
        "trial_kind": record["trial_kind"],
        "input_raw_sha256": raw_digest,
        "replicas": replicas,
        "witnesses_checked": checked,
        "best_feasible_cost": int(best.min()),
        "initial_best_cost": int(initial.min()) if initial is not None else None,
        "replicas_improved_after_initialization": int((best < initial).sum()) if native else None,
        "rational_lower_bound": lower,
        "pbit_heatbath_participation": participation,
    }, (costs, rows, best, initial, states, eligible)


def verify(calls_file, repository):
    calls_file, repository = Path(calls_file), Path(repository)
    records = json.loads(calls_file.read_text())
    panel = json.loads((calls_file.parent / "panel.json").read_text())
    require(isinstance(records, list) and records, "Released calls must be a nonempty list")
    by_id = {record["id"]: record for record in records}
    require(len(by_id) == len(records), "Released call identifiers are duplicated")
    natural = {r["id"] for r in records if r["trial_kind"] == "natural"}
    replay = {r["id"]: r["replay_of"] for r in records if r["trial_kind"] == "work-replay"}
    require(len(natural) + len(replay) == len(records), "Unknown trial kind")
    expected_natural = panel["expected_natural_ids"]
    expected_replays = panel["expected_replays"]
    require(
        len(set(expected_natural)) == len(expected_natural)
        and natural == set(expected_natural)
        and len(expected_replays) == len(replay)
        and replay == {r["id"]: r["origin"] for r in expected_replays},
        "Complete frozen call roster differs",
    )
    require(
        panel["planned_natural_calls"] == panel["completed_natural_calls"] == len(natural)
        and panel["failed_natural_calls"] == panel["unexecuted_natural_calls"] == 0
        and panel["work_replays"] == len(replay)
        and panel["replays_add_independent_trials"] is False,
        "Published denominators do not preserve the complete panel",
    )
    targets = panel["frozen_targets"]
    require(set(targets) == {r["instance"] for r in records}, "Frozen target instances differ")
    results, problems, replay_arrays, input_identities = [], {}, {}, {}
    for record in records:
        result, (costs, rows, best, initial, states, eligible) = verify_call(
            record, repository, calls_file.parent
        )
        previous_identity = input_identities.setdefault(
            record["instance"], result["input_raw_sha256"]
        )
        require(
            previous_identity == result["input_raw_sha256"],
            "Calls disagree on original instance bytes",
        )
        target = targets[record["instance"]]
        cost = target["cost"]
        require(
            isinstance(cost, int)
            and not isinstance(cost, bool)
            and cost > 0
            and isinstance(target["status"], str)
            and bool(target["status"])
            and isinstance(target["source"], str)
            and bool(target["source"]),
            "Frozen target lacks its cost, status or source",
        )
        result["target_hit_replicas"] = int((best <= cost).sum())
        result["initial_target_hit_replicas"] = (
            int((initial <= cost).sum()) if initial is not None else None
        )
        result["new_target_hit_replicas_after_initialization"] = (
            int(((best <= cost) & (initial > cost)).sum()) if initial is not None else None
        )
        result["pbit_new_target_hit_replicas"] = (
            result["new_target_hit_replicas_after_initialization"]
            if result["pbit_heatbath_participation"]
            else 0
        )
        check_claims(record, {key: result[key] for key in result if key.endswith("replicas")})
        if initial is not None:
            observed = (states["observation_best_cost"] <= cost).any(axis=1) & eligible
            result["first_target_observation_seconds"] = (
                0.0
                if (initial <= cost).any()
                else float(states["observation_seconds"][observed][0])
                if observed.any()
                else None
            )
        problems.setdefault(record["instance"], (costs, rows))
        if record["id"] in replay or record["id"] in replay.values():
            replay_arrays[record["id"]] = states
        results.append(result)
    for identifier, origin in replay.items():
        require(origin in natural, "Replay origin is not a natural call")
        left, right = replay_arrays[identifier], replay_arrays[origin]
        require(set(left) == set(right), "Replay witness array sets differ")
        for key in left:
            if key != "observation_seconds":
                np.testing.assert_array_equal(
                    left[key], right[key], err_msg="Replay differs: " + key
                )
    for instance, target in targets.items():
        if "source_witness" in target:
            entry = target["source_witness"]
            path = contained(repository, entry["path"])
            checked_file(path, entry["sha256"])
            loaded = np.load(path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                with loaded:
                    bits = loaded["best_bits"]
            else:
                bits = loaded
            cost, coverage = original_values(bits, *problems[instance])
            require(
                (coverage > 0).all() and int(cost.min()) == target["cost"],
                "Historical target witness differs",
            )
    natural_results = [r for r in results if r["trial_kind"] == "natural"]
    pbit = [r for r in natural_results if r["pbit_heatbath_participation"]]
    return {
        "status": "verified",
        "solver_executed": False,
        "natural_calls": len(natural),
        "work_replays": len(replay),
        "new_independent_trials_from_replays": 0,
        "method_natural_calls": dict(Counter(r["method"] for r in natural_results)),
        "witnesses_checked": sum(r["witnesses_checked"] for r in results),
        "pbit_natural_calls": len(pbit),
        "pbit_target_hit_calls": sum(r["target_hit_replicas"] > 0 for r in pbit),
        "pbit_calls_improving_initialization": sum(
            r["replicas_improved_after_initialization"] > 0 for r in pbit
        ),
        "records": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, default=Path("data/set-cover/rtx5090/calls.json"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.calls, args.repository)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
