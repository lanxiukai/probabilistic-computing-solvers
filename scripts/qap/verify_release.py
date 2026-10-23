"""Verify portable QAP result witnesses and fetched benchmark bytes without solving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from combinatorial_solvers.qap.model import read_qaplib, read_solution


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release paths must remain inside their declared root")
    return path


def checked_file(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise AssertionError("Released file digest mismatch: " + path.name)


def original_costs(instance, permutations):
    values = np.asarray(permutations)
    shape = values.shape[:-1]
    flat = values.reshape(-1, instance.size)
    if values.dtype.kind not in "iu" or not np.all(
        np.sort(flat, axis=1) == np.arange(instance.size)
    ):
        raise AssertionError("Released witness violates the original permutation constraints")
    return np.asarray(
        [
            np.sum(instance.flow * instance.distance[p[:, None], p[None, :]], dtype=np.int64)
            for p in flat
        ]
    ).reshape(shape)


def verify(calls_file, repository):
    records = json.loads(calls_file.read_text())
    if not isinstance(records, list):
        raise ValueError("Released calls must be a list")
    results = []
    witnesses = {}
    for record in records:
        configuration = record["configuration"]
        if isinstance(configuration, str):
            configuration = json.loads(contained(repository, configuration).read_text())
        inputs = {}
        for entry in record["benchmark_inputs"]:
            path = contained(repository, entry["path"])
            checked_file(path, entry["sha256"])
            inputs[entry["field"]] = path
        instance = read_qaplib(inputs["problem.instance_file"])
        path = contained(calls_file.parent, record["witness_file"])
        checked_file(path, record["witness_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            states = {key: archive[key] for key in archive.files}
        metrics = record["metrics"]
        if (
            "model_binary_variables" in metrics
            and metrics["model_binary_variables"] != instance.size**2
        ):
            raise AssertionError("Published binary model size differs from the input order")
        if "physical_update_pbits_per_model" in metrics:
            expected_pbits = instance.size**2 if metrics["pbit_heatbath_participation"] else 0
            if metrics["physical_update_pbits_per_model"] != expected_pbits:
                raise AssertionError("Published p-bit count includes unrepresented variables")
        checked = 0
        actual = {}
        for key, cost_key in (
            ("initial_permutation", "initial_cost"),
            ("permutation", "cost"),
            ("final_permutation", "final_cost"),
            ("best_permutation", "best_cost"),
            ("quality_best_permutation", "quality_best_cost"),
        ):
            if key in states:
                actual[key] = original_costs(instance, states[key])
                checked += actual[key].size
                if cost_key in states:
                    np.testing.assert_array_equal(actual[key], states[cost_key])
        for state_key, metric_key in (
            ("initial_permutation", "initial_best_cost"),
            ("quality_best_permutation", "best_feasible_cost"),
            ("best_permutation", "returned_best_cost"),
        ):
            if state_key in actual and metric_key in metrics:
                if int(np.min(actual[state_key])) != metrics[metric_key]:
                    raise AssertionError("Published objective differs from its original witness")
        if "parallel_replicas" in metrics:
            if (
                len(np.atleast_1d(actual["quality_best_permutation"]))
                != metrics["parallel_replicas"]
            ):
                raise AssertionError("Published replica count differs from saved witnesses")
        quality_states = np.atleast_2d(states["quality_best_permutation"])
        winner = quality_states[int(np.argmin(np.atleast_1d(actual["quality_best_permutation"])))]
        python_cost = sum(
            int(instance.flow[i, j]) * int(instance.distance[winner[i], winner[j]])
            for i in range(instance.size)
            for j in range(instance.size)
        )
        if python_cost != metrics["best_feasible_cost"]:
            raise AssertionError("Unbounded integer certificate differs from the public result")
        if "bits" in states:
            np.testing.assert_array_equal(
                states["bits"], np.eye(instance.size, dtype=np.int8)[states["permutation"]]
            )
        if "visited" in states:
            if not np.isin(states["visited"], (0, 1)).all():
                raise AssertionError("Visited record is not binary")
            if "visited_assignment_bits_min" in metrics:
                if (
                    int(states["visited"].sum(axis=(1, 2)).min())
                    != metrics["visited_assignment_bits_min"]
                ):
                    raise AssertionError("Published visitation count differs from its saved mask")
        settings = configuration.get("solver", configuration.get("baseline"))
        times = states["curve_seconds"]
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise AssertionError("Released observation times are invalid")
        indices = states.get("curve_indices", np.arange(len(times), dtype=np.int64))
        curation = record.get("curve_curation", {}).get("curve_")
        if curation is not None:
            count = curation["original_observations"]
            if (
                len(indices) != len(times)
                or curation["retained_observations"] != len(times)
                or np.any(np.diff(indices) <= 0)
                or (len(indices) and (indices[0] != 0 or indices[-1] != count - 1))
            ):
                raise AssertionError("Released curve indices omit its declared original endpoints")
            if "curve_best" in states:
                chunk = settings["chunk_steps"]
                completed = metrics["completed_steps"]
                if count != (completed + chunk - 1) // chunk:
                    raise AssertionError("Original observation count differs from completed work")
                np.testing.assert_array_equal(
                    states["curve_steps"], np.minimum((indices + 1) * chunk, completed)
                )
            else:
                np.testing.assert_array_equal(indices, np.arange(count))
        eligible = times <= settings["max_seconds"]
        for setting, array in (
            ("quality_max_step", "curve_steps"),
            ("quality_max_iteration", "curve_iteration"),
        ):
            if settings.get(setting) is not None:
                eligible &= states[array] <= settings[setting]
        if settings.get("quality_max_curve_index") is not None:
            eligible &= indices <= settings["quality_max_curve_index"]
        np.testing.assert_array_equal(eligible, states["curve_eligible"])
        if curation is not None and len(eligible) > 1:
            changes = np.flatnonzero(eligible[1:] != eligible[:-1])
            if np.any(indices[changes + 1] != indices[changes] + 1):
                raise AssertionError("Released curve omitted an original eligibility neighbor")
        if "curve_best" in states:
            expected = (
                states["curve_best"][eligible][-1] if eligible.any() else states["initial_cost"]
            )
            np.testing.assert_array_equal(expected, states["quality_best_cost"])
        else:
            original_curve = original_costs(instance, states["curve_permutations"])
            checked += original_curve.size
            np.testing.assert_array_equal(original_curve, states["curve_cost"])
            if np.any(np.diff(original_curve) >= 0):
                raise AssertionError("Released classical improvements do not strictly decrease")
            if np.any(np.diff(states["curve_iteration"]) <= 0):
                raise AssertionError("Released classical iterations do not strictly increase")
            np.testing.assert_array_equal(
                states["curve_permutations"][0], states["initial_permutation"]
            )
            np.testing.assert_array_equal(
                states["curve_permutations"][-1], states["best_permutation"]
            )
            last_eligible = int(np.flatnonzero(eligible)[-1])
            np.testing.assert_array_equal(
                states["curve_permutations"][last_eligible], states["quality_best_permutation"]
            )
            np.testing.assert_array_equal(
                states["curve_cost"][last_eligible], states["quality_best_cost"]
            )
        reference_cost = None
        if "problem.reference_file" in inputs:
            problem = configuration["problem"]
            _, certificate = read_solution(
                inputs["problem.reference_file"],
                instance,
                index_base=problem.get("reference_index_base", 1),
                inverse=problem.get("reference_inverse", False),
            )
            reference_cost = certificate["cost"]
            if problem.get("reference_status") == "proven_optimum" and np.any(
                actual["quality_best_permutation"] < reference_cost
            ):
                raise AssertionError("Witness contradicts the claimed proven optimum")
            current = np.atleast_1d(actual["quality_best_permutation"])
            initial = np.atleast_1d(actual["initial_permutation"])
            curves = (
                states["curve_best"] if "curve_best" in states else states["curve_cost"][:, None]
            )
            for name, target in (("reference", np.equal), ("reference_or_better", np.less_equal)):
                hit = target(current, reference_cost)
                initial_hit = target(initial, reference_cost)
                observed = np.any(target(curves, reference_cost), axis=1) & eligible
                expected_metrics = {
                    name + "_hits": int(hit.sum()),
                    "initial_" + name + "_hits": int(initial_hit.sum()),
                    "new_" + name + "_hits_after_initialization": int((hit & ~initial_hit).sum()),
                    "ensemble_first_" + name + "_observation_seconds": (
                        0.0
                        if initial_hit.any()
                        else float(times[observed][0])
                        if observed.any()
                        else None
                    ),
                }
                for key, value in expected_metrics.items():
                    if key in metrics and metrics[key] != value:
                        raise AssertionError("Published reference outcome differs: " + key)
        results.append(
            {
                "id": record["id"],
                "witnesses_checked": checked,
                "best_feasible_cost": int(np.min(actual["quality_best_permutation"])),
                "reference_cost": reference_cost,
            }
        )
        witnesses[record["id"]] = path
    replay_count = 0
    for record in records:
        origin = record.get("replay_of")
        if origin is None:
            continue
        with np.load(witnesses[origin]) as first, np.load(witnesses[record["id"]]) as second:
            if set(first.files) != set(second.files):
                raise AssertionError("Public replay arrays differ")
            for key in first.files:
                if key != "curve_seconds":
                    np.testing.assert_array_equal(first[key], second[key])
        replay_count += 1
    return {
        "status": "verified",
        "solver_executed": False,
        "calls": len(results),
        "witnesses_checked": sum(item["witnesses_checked"] for item in results),
        "work_replays_checked": replay_count,
        "records": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.calls.resolve(), args.repository.resolve())
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
