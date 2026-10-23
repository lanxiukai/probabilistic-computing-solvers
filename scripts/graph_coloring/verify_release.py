"""Verify portable coloring witnesses on the CPU without importing a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
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
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == expected, "Released file digest mismatch: " + path.name)
    return actual


def read_graph(path):
    """Read original edge bytes independently of the solver's model parser."""
    header, edges = None, []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] in {"c", "%"}:
            continue
        if fields[0] == "p" and len(fields) == 4 and fields[1] in {"edge", "edges", "col"}:
            require(header is None, "Graph has multiple headers")
            header = tuple(map(int, fields[2:]))
        elif fields[0] == "e" and len(fields) == 3:
            edges.append(tuple(sorted((int(fields[1]) - 1, int(fields[2]) - 1))))
        else:
            raise ValueError("Unsupported original DIMACS record")
    require(header is not None, "Graph header is missing")
    size, count = header
    require(size >= 2 and count == len(edges), "Original graph size or edge count differs")
    require(len(set(edges)) == count, "Original graph repeats an edge")
    require(all(0 <= u < v < size for u, v in edges), "Original graph is not simple")
    return size, tuple(edges)


def load_problem(configuration, entries, repository):
    problem = configuration["problem"]
    inputs = {}
    for entry in entries:
        field = entry["field"]
        require(field not in inputs, "A benchmark input field is duplicated")
        path = contained(repository, entry["path"])
        checked_file(path, entry["sha256"])
        require(field.startswith("problem."), "Unexpected benchmark input field")
        key = field.removeprefix("problem.")
        require(
            contained(repository, problem[key]) == path,
            "Configuration and benchmark input binding differ",
        )
        inputs[field] = path
    for key in ("instance_file", "allowed_file", "symmetry_clique_file"):
        require(
            bool(problem.get(key)) == ("problem." + key in inputs),
            "A configured graph, list or clique input lacks its digest binding",
        )
    size, edges = read_graph(inputs["problem.instance_file"])
    palette = problem["colors"]
    require(
        isinstance(palette, int) and not isinstance(palette, bool) and palette >= 2,
        "Invalid palette size",
    )
    allowed = np.ones((size, palette), dtype=bool)
    clique = None
    if "problem.allowed_file" in inputs:
        values = np.load(inputs["problem.allowed_file"], allow_pickle=False)
        require(
            values.shape == allowed.shape and np.isin(values, (0, 1)).all(),
            "Allowed-color input is not a binary vertex-palette matrix",
        )
        allowed = values.astype(bool)
        require(allowed.any(axis=1).all(), "An allowed-color list is empty")
    if "problem.symmetry_clique_file" in inputs:
        require(
            "problem.allowed_file" not in inputs,
            "Clique symmetry cannot override lists",
        )
        payload = json.loads(inputs["problem.symmetry_clique_file"].read_text())
        checked_file(inputs["problem.instance_file"], payload["input_sha256"])
        clique = payload["vertices"]
        require(
            isinstance(clique, list)
            and 0 < len(clique) <= palette
            and all(isinstance(v, int) and not isinstance(v, bool) for v in clique)
            and len(set(clique)) == len(clique)
            and all(0 <= v < size for v in clique),
            "Clique certificate has invalid vertices",
        )
        edge_set = set(edges)
        require(
            all(tuple(sorted(pair)) in edge_set for pair in combinations(clique, 2)),
            "Claimed clique contains a nonedge",
        )
        for key, expected in (
            ("size", len(clique)),
            ("pair_checks", len(clique) * (len(clique) - 1) // 2),
            ("valid_clique", True),
        ):
            if key in payload:
                require(
                    payload[key] == expected,
                    "Clique metadata differs from its certificate",
                )
        for color, vertex in enumerate(clique):
            allowed[vertex] = False
            allowed[vertex, color] = True
    return size, palette, edges, allowed, clique


def original_costs(values, size, palette, edges, allowed):
    values = np.asarray(values)
    require(
        values.ndim >= 1 and values.shape[-1] == size and values.dtype.kind in "iu",
        "Color witness has an invalid shape or integer type",
    )
    rows = values.reshape(-1, size)
    require(((rows >= 0) & (rows < palette)).all(), "Color witness is outside the palette")
    require(allowed[np.arange(size), rows].all(), "Color witness violates a list or clamp")
    u, v = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    result = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(rows), 256):
        block = rows[start : start + 256]
        result[start : start + len(block)] = (block[:, u] == block[:, v]).sum(axis=1)
    return result.reshape(values.shape[:-1])


def check_claims(record, expected, required=()):
    metrics, resources = record["metrics"], record.get("resources", {})
    for key in required:
        require(key in metrics, "Required public metric is missing: " + key)
    for key, value in expected.items():
        for source in (metrics, resources):
            if key in source:
                require(source[key] == value, "Published metric differs: " + key)


def curve_eligibility(states, settings, coordinate):
    times, steps = states["curve_seconds"], states[coordinate]
    require(
        times.ndim == steps.ndim == 1
        and times.shape == steps.shape
        and np.isfinite(times).all()
        and (times >= 0).all()
        and (np.diff(times) >= 0).all()
        and steps.dtype.kind in "iu"
        and (steps >= 0).all()
        and (np.diff(steps) > 0).all(),
        "Released observation coordinates are invalid",
    )
    limit = settings["max_seconds"]
    require(np.isfinite(limit) and limit > 0, "Invalid search deadline")
    eligible = times <= limit
    fence = "quality_max_step" if coordinate == "curve_steps" else "quality_max_iteration"
    if settings.get(fence) is not None:
        eligible &= steps <= settings[fence]
    if settings.get("quality_max_curve_index") is not None:
        require("curve_indices" in states, "A curve-index fence needs original indices")
        eligible &= states["curve_indices"] <= settings["quality_max_curve_index"]
    saved = states["curve_eligible"]
    require(
        saved.shape == eligible.shape and np.isin(saved, (0, 1)).all(),
        "Invalid eligibility",
    )
    np.testing.assert_array_equal(saved.astype(bool), eligible)
    return eligible


def verify_curve_curation(record, states, observations, indices, rule):
    curation = record.get("curve_curation", {}).get("curve_")
    if "curve_indices" in states:
        saved = states["curve_indices"]
        require(
            saved.shape == indices.shape and saved.dtype.kind in "iu",
            "Original curve indices have an invalid shape or type",
        )
        np.testing.assert_array_equal(saved, indices)
    if curation is None:
        require(
            len(indices) == observations,
            "A sparse curve needs explicit curation metadata",
        )
        return
    require("curve_indices" in states, "Curated observations lack original curve indices")
    require(
        curation["original_observations"] == observations
        and curation["retained_observations"] == len(indices),
        "Declared curve observation counts differ",
    )
    require(
        curation["rule"] == rule and curation["first_reference_hit_preserved"] is True,
        "Declared curve selection rule differs",
    )


def verify_native(record, states, problem, configuration):
    size, palette, edges, allowed, _ = problem
    settings = configuration["solver"]
    replicas = settings.get("replicas", 64)
    ladder = settings.get("tempering_ladder_size", 1)
    require(replicas > 0 and ladder > 0 and replicas % ladder == 0, "Invalid replica groups")
    method = settings.get("method", "heatbath")
    require(method in {"heatbath", "metropolis", "plateau"}, "Unknown transition method")
    costs = {}
    for key, cost_key in (
        ("initial_colors", "initial_cost"),
        ("colors", "conflicts"),
        ("best_colors", "best_conflicts"),
        ("quality_best_colors", "quality_best_cost"),
    ):
        require(
            states[key].shape == (replicas, size),
            "Native witness replica count differs",
        )
        costs[key] = original_costs(states[key], size, palette, edges, allowed)
        np.testing.assert_array_equal(costs[key], states[cost_key])
    onehot = np.eye(palette, dtype=np.int8)
    np.testing.assert_array_equal(states["bits"], onehot[states["colors"]])
    visited = states["visited"]
    require(
        visited.shape == (replicas, size, palette)
        and np.isin(visited, (0, 1)).all()
        and (visited <= allowed).all(),
        "Visited record violates its binary shape or allowed-color mask",
    )
    considered = visited.sum(axis=2)
    require(
        ((considered == 0) | (considered == allowed.sum(axis=1))).all(),
        "Visited vertex mask does not contain its complete allowed block",
    )
    for key in costs:
        require(
            ((considered > 0) | (states[key] == states["initial_colors"])).all(),
            "An unvisited vertex changed its assignment",
        )
    if "counts" in states:
        u, v = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
        expected = np.zeros((replicas, size, palette), dtype=np.int64)
        for r, colors in enumerate(states["colors"]):
            np.add.at(expected[r], (u, colors[v]), 1)
            np.add.at(expected[r], (v, colors[u]), 1)
        np.testing.assert_array_equal(expected, states["counts"])
    eligible = curve_eligibility(states, settings, "curve_steps")
    best = states["curve_best"]
    require(
        best.shape == (len(eligible), replicas)
        and best.dtype.kind in "iu"
        and (best >= 0).all()
        and (best <= len(edges)).all()
        and (np.diff(best, axis=0) <= 0).all(),
        "Retained per-replica best curve is invalid",
    )
    initial, quality, returned = (
        costs[key] for key in ("initial_colors", "quality_best_colors", "best_colors")
    )
    expected_quality = best[eligible][-1] if eligible.any() else initial
    np.testing.assert_array_equal(expected_quality, quality)
    np.testing.assert_array_equal(best[-1] if len(best) else initial, returned)
    require(
        (returned <= quality).all() and (quality <= initial).all(),
        "Best costs increase",
    )
    require((returned <= costs["colors"]).all(), "Returned best exceeds the final cost")
    if len(best):
        require((best[0] <= initial).all(), "First retained best exceeds initialization")
    if "curve_cost" in states:
        current = states["curve_cost"]
        require(
            current.shape == best.shape
            and current.dtype.kind in "iu"
            and (current >= best).all()
            and (current <= len(edges)).all(),
            "Retained current-state costs are invalid",
        )
        if len(current):
            np.testing.assert_array_equal(current[-1], costs["colors"])
    completed = record["metrics"]["completed_steps"]
    steps = states["curve_steps"]
    chunk, max_steps = (
        settings.get("chunk_steps", 256),
        settings.get("max_steps", 100000000),
    )
    require(0 <= completed <= max_steps and chunk > 0, "Invalid completed work count")
    if completed >= size:
        np.testing.assert_array_equal(visited, np.broadcast_to(allowed, visited.shape))
    observations = (completed + chunk - 1) // chunk
    require(bool(len(steps)) == bool(completed), "Final work count lacks its endpoint")
    indices = (steps + chunk - 1) // chunk - 1
    if len(steps):
        require(
            steps[0] == min(chunk, max_steps) and steps[-1] == completed,
            "Sparse curve omits a required endpoint",
        )
        require(
            ((steps % chunk == 0) | (steps == max_steps)).all(),
            "Invalid chunk boundary",
        )
        transitions = np.flatnonzero(eligible[:-1] != eligible[1:])
        require(
            (np.diff(indices)[transitions] == 1).all(),
            "Sparse curve omits a time boundary",
        )
    verify_curve_curation(
        record,
        states,
        observations,
        indices,
        "all_replica_best_improvements_and_eligibility_boundaries",
    )
    free = allowed.sum(axis=1) > 1
    active = np.zeros(size, dtype=bool)
    for u, v in edges:
        active[u] = active[v] = True
    heatbath = method == "heatbath" and bool(free.any())
    expected = {
        "method": method,
        "pbit_heatbath_participation": heatbath,
        "physical_update_pbits_per_model": int(allowed[free].sum()) if heatbath else 0,
        "parallel_replicas": replicas,
        "independent_replica_groups": replicas // ladder,
        "tempering_ladder_size": ladder,
        "initial_best_cost": int(initial.min()),
        "initial_zero_conflict_hits": int((initial == 0).sum()),
        "zero_conflict_hits": int((quality == 0).sum()),
        "pbit_new_zero_conflict_hits": int(((initial > 0) & (quality == 0)).sum())
        if heatbath
        else 0,
        "zero_conflict_groups": int((quality.reshape(-1, ladder).min(axis=1) == 0).sum()),
        "replicas_improved": int((quality < initial).sum()),
        "best_feasible_cost": int(quality.min()),
        "returned_best_cost": int(returned.min()),
        "quality_max_step": int(steps[eligible][-1]) if eligible.any() else 0,
        "visited_assignment_bits_min": int(visited.sum(axis=(1, 2)).min()),
        "visited_unclamped_assignment_bits_min": int(visited[:, free].sum(axis=(1, 2)).min()),
        "visited_interacting_assignment_bits_min": int(visited[:, active].sum(axis=(1, 2)).min()),
        "visited_interacting_unclamped_assignment_bits_min": int(
            visited[:, active & free].sum(axis=(1, 2)).min()
        ),
    }
    check_claims(
        record,
        expected,
        (
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
            "parallel_replicas",
            "best_feasible_cost",
            "returned_best_cost",
            "zero_conflict_hits",
            "pbit_new_zero_conflict_hits",
            "independent_replica_groups",
            "zero_conflict_groups",
        ),
    )
    return expected, 4 * replicas


def verify_classical(record, states, problem, configuration):
    size, palette, edges, allowed, _ = problem
    costs = {}
    for key in (
        "initial_colors",
        "best_colors",
        "returned_best_colors",
        "final_colors",
    ):
        require(states[key].shape == (size,), "Classical witness must have one assignment")
        costs[key] = int(original_costs(states[key], size, palette, edges, allowed))
    curve = original_costs(states["curve_colors"], size, palette, edges, allowed)
    np.testing.assert_array_equal(curve, states["curve_cost"])
    eligible = curve_eligibility(states, configuration["baseline"], "curve_iteration")
    require(
        len(curve) > 0
        and curve.shape == eligible.shape
        and states["curve_iteration"][0] == 0
        and states["curve_seconds"][0] == 0
        and curve[0] == costs["initial_colors"]
        and (np.diff(curve) < 0).all(),
        "Classical improvement curve is invalid",
    )
    require(eligible[0], "The classical initial observation must be eligible")
    require(
        costs["best_colors"] == int(curve[eligible][-1]),
        "Classical eligible best differs",
    )
    require(
        costs["returned_best_colors"] == int(curve[-1]),
        "Classical returned best differs",
    )
    require(costs["final_colors"] >= int(curve[-1]), "Classical final cost beats its best")
    verify_curve_curation(
        record, states, len(curve), np.arange(len(curve)), "all_original_observations"
    )
    expected = {
        "method": "tabucol-style",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "parallel_workers": 1,
        "best_feasible_cost": costs["best_colors"],
        "returned_best_cost": costs["returned_best_colors"],
        "zero_conflict_hits": int(costs["best_colors"] == 0),
        "pbit_new_zero_conflict_hits": 0,
    }
    check_claims(
        record,
        expected,
        (
            "pbit_heatbath_participation",
            "physical_update_pbits_per_model",
            "best_feasible_cost",
            "returned_best_cost",
            "zero_conflict_hits",
        ),
    )
    return expected, 4 + len(curve)


def verify(calls_file, repository):
    calls_file, repository = Path(calls_file).resolve(), Path(repository).resolve()
    records = json.loads(calls_file.read_text())
    require(
        isinstance(records, list) and len(records) > 0,
        "Released calls must be a nonempty list",
    )
    results, paths = [], {}
    for record in records:
        require(record["id"] not in paths, "A call identifier is duplicated")
        configuration = record["configuration"]
        if isinstance(configuration, str):
            configuration = json.loads(contained(repository, configuration).read_text())
        if "configuration_file" in record:
            saved_configuration = json.loads(
                contained(repository, record["configuration_file"]).read_text()
            )
            require(
                saved_configuration == configuration,
                "Reproduction configuration file differs from the embedded configuration",
            )
        problem = load_problem(configuration, record["benchmark_inputs"], repository)
        size, palette, edges, allowed, clique = problem
        path = contained(calls_file.parent, record["witness_file"])
        checked_file(path, record["witness_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            states = dict(archive)
        classical = configuration.get("method", "native") == "tabucol-style"
        expected, checked = (verify_classical if classical else verify_native)(
            record, states, problem, configuration
        )
        if clique is not None:
            np.testing.assert_array_equal(states["symmetry_clique_vertices"], clique)
        else:
            require("symmetry_clique_vertices" not in states, "Unbound clique witness")
        active = np.zeros(size, dtype=bool)
        for u, v in edges:
            active[u] = active[v] = True
        free = allowed.sum(axis=1) > 1
        counts = {
            "vertices": size,
            "edges": len(edges),
            "palette": palette,
            "model_binary_variables": size * palette,
            "allowed_binary_variables": int(allowed.sum()),
            "fixed_zero_variables": int((~allowed).sum()),
            "isolated_vertices": int((~active).sum()),
            "nonisolated_vertices": int(active.sum()),
            "interacting_model_binary_variables": int(active.sum()) * palette,
            "interacting_allowed_binary_variables": int(allowed[active].sum()),
            "singleton_color_vertices": int((~free).sum()),
            "unclamped_allowed_binary_variables": int(allowed[free].sum()),
            "interacting_unclamped_allowed_binary_variables": int(allowed[active & free].sum()),
            "chromatic_lower_bound": len(clique) if clique is not None else None,
            "chromatic_optimum_certified": bool(
                clique is not None
                and len(clique) == palette
                and expected["best_feasible_cost"] == 0
            ),
        }
        check_claims(record, counts, ("chromatic_lower_bound", "chromatic_optimum_certified"))
        results.append(
            {
                "id": record["id"],
                "witnesses_checked": checked,
                **counts,
                **expected,
                "retained_observations_checked": len(states["curve_seconds"]),
            }
        )
        paths[record["id"]] = path
    replays = 0
    for record in records:
        origin = record.get("replay_of")
        if origin is None:
            continue
        require(
            origin in paths and origin != record["id"],
            "Replay origin is missing or cyclic",
        )
        with np.load(paths[origin], allow_pickle=False) as left:
            first = dict(left)
        with np.load(paths[record["id"]], allow_pickle=False) as right:
            second = dict(right)
        require(first.keys() == second.keys(), "Replay witness fields differ")
        for key in first:
            if key != "curve_seconds":
                np.testing.assert_array_equal(first[key], second[key])
        replays += 1
    return {
        "status": "verified",
        "solver_executed": False,
        "calls": len(results),
        "work_replays_checked": replays,
        "witnesses_checked": sum(row["witnesses_checked"] for row in results),
        "scope": (
            "Original graph/list/clique witnesses, retained observations and declared counts. "
            "Sparse curves do not recover omitted current-state trajectories. Timing and RNG "
            "execution authenticity require the separately retained original experiment evidence."
        ),
        "records": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.calls, args.repository)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
