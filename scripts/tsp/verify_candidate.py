"""Verify curated TSP candidate results using explicitly supplied TSPLIB inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from combinatorial_solvers.tsp.edge_model import build_assignment, successor_cycles
from combinatorial_solvers.tsp.instance import certify_tour
from combinatorial_solvers.tsp.tsplib import load_tsplib


def verify_certificate(path: Path, instance_directory: Path) -> dict:
    packet = json.loads(path.read_text())
    if packet["schema"] != "tsp-native-tour-certificate-v1" or packet["index_base"] != 0:
        raise ValueError("Unsupported native tour certificate")
    description = packet["instance"]
    source = instance_directory / description["filename"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != description["sha256"]:
        raise ValueError("Supplied TSPLIB bytes differ from the certificate input")
    instance = load_tsplib(source, optimum=description["known_optimum"])
    if instance.dimension != description["dimension"]:
        raise ValueError("Certificate dimension differs from its input")
    selected = packet["selected_certificate"]
    observations = packet["certificates"]
    if type(selected) is not int or not 0 <= selected < len(observations):
        raise ValueError("Selected certificate index is invalid")
    for row in observations:
        if row["role"] not in {"discovery", "confirmation", "replay"}:
            raise ValueError("Unexpected certificate role")
        certificate = certify_tour(instance, row["tour"])
        if not certificate.is_permutation or certificate.length != row["length"]:
            raise ValueError("Native tour certificate is invalid")
    if observations[selected]["length"] != instance.optimum:
        raise ValueError("Selected tour does not match the declared reference optimum")
    natural = [row for row in observations if row["role"] != "replay"]
    confirmation = [row for row in natural if row["role"] == "confirmation"]
    return {
        "cities": instance.dimension,
        "certified_length": observations[selected]["length"],
        "certified_tours": len(observations),
        "natural_calls": len(natural),
        "natural_reference_hits": sum(row["length"] == instance.optimum for row in natural),
        "confirmation_calls": len(confirmation),
        "confirmation_reference_hits": sum(
            row["length"] == instance.optimum for row in confirmation
        ),
        "independent_lower_bound_proved": False,
        "new_solver_calls": 0,
    }


def verify(core: Path, instance_directory: Path) -> dict:
    manifest = json.loads((core / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = core / name
        if (
            path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("A curated core member changed")
    data = json.loads((core / "calls.json").read_text())
    if "5090" not in data["hardware"]["gpu"]:
        raise ValueError("These public GPU results require actual RTX 5090 measurements")
    inputs = {}
    for description in data["instances"]:
        path = instance_directory / description["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != description["sha256"]:
            raise ValueError("Supplied TSPLIB bytes differ from the measurement input")
        inputs[description["name"]] = load_tsplib(path, optimum=description["optimum"])
    tours = states = pbit_boundaries = 0
    pbit_cycle_counts = []
    for row in data["calls"]:
        instance = inputs[row["instance"]]
        packet = json.loads((core / row["events_file"]).read_text())
        lengths = []
        for event in packet:
            if "tour" not in event:
                continue
            certificate = certify_tour(instance, event["tour"])
            if not certificate.is_permutation or certificate.length != event["length"]:
                raise AssertionError("Curated tour certificate is invalid")
            lengths.append(event["length"])
            tours += 1
        if min(lengths, default=None) != row["summary"]["best_length"]:
            raise AssertionError("Best-tour summary differs from its observations")
        if row["method"] not in {"pbit", "random", "assignment", "ils"}:
            continue
        prefix = next(event["tour"] for event in packet if event["stage"] == "prefix")
        solver = row["configuration"]["solver"]
        model = build_assignment(
            instance, [prefix], neighbors=solver["neighbors"], penalty=solver["penalty"]
        )
        if model.bits != row["summary"]["model"]["binary_nodes_per_replica"]:
            raise AssertionError("Rebuilt model size differs")
        for relative in row["state_files"]:
            with np.load(core / relative, allow_pickle=False) as saved:
                outgoing, incoming = model.counts(saved["states"])
                if "outgoing" in saved:
                    np.testing.assert_array_equal(outgoing, saved["outgoing"])
                    np.testing.assert_array_equal(incoming, saved["incoming"])
                    np.testing.assert_allclose(
                        model.energies(saved["states"]), saved["energies"], rtol=1e-12
                    )
                states += len(saved["states"])
                if (
                    row["method"] == "pbit"
                    and row["trial_kind"] == "natural"
                    and Path(relative).name != "initial.npz"
                ):
                    pbit_boundaries += len(saved["states"])
                    feasible = np.all(outgoing == 1, axis=1) & np.all(incoming == 1, axis=1)
                    for state in saved["states"][feasible]:
                        selected = state.astype(bool)
                        successor = np.empty(instance.dimension, dtype=np.int32)
                        successor[model.source[selected]] = model.destination[selected]
                        pbit_cycle_counts.append(len(successor_cycles(successor)))
    return {
        "calls": len(data["calls"]),
        "tour_certificates": tours,
        "state_certificates": states,
        "pbit_boundary_states": pbit_boundaries,
        "pbit_degree_feasible_boundary_states": len(pbit_cycle_counts),
        "pbit_connected_boundary_states": pbit_cycle_counts.count(1),
        "pbit_feasible_state_cycle_counts": pbit_cycle_counts,
        "new_solver_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--core", type=Path)
    selection.add_argument("--certificate", type=Path)
    parser.add_argument("--instances-directory", type=Path, required=True)
    args = parser.parse_args()
    checked = (
        verify_certificate(args.certificate, args.instances_directory)
        if args.certificate is not None
        else verify(args.core, args.instances_directory)
    )
    print(json.dumps(checked))


if __name__ == "__main__":
    main()
