"""Standalone integer witness checks; no solver, NumPy, Torch or GPU imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def canonical_instance(payload: dict) -> dict:
    """Validate the complete open +/-1 cubic domain independently of model.py."""
    if not isinstance(payload, dict) or set(payload) != {"length", "boundary", "bonds"}:
        raise ValueError("Instance requires exactly length, boundary and bonds")
    length = payload["length"]
    if type(length) is not int or not 2 <= length <= 32 or payload["boundary"] != "open":
        raise ValueError("Supported domain: open, zero-field +/-1 cubes, 2 <= L <= 32")
    expected = set()
    for site in range(length**3):
        x, y, z = site % length, (site // length) % length, site // length**2
        for coordinate, stride in ((x, 1), (y, length), (z, length**2)):
            if coordinate + 1 < length:
                expected.add((site, site + stride))
    bonds = payload["bonds"]
    if not isinstance(bonds, list) or len(bonds) != len(expected):
        raise ValueError("Missing or extra lattice bonds")
    canonical = {}
    for bond in bonds:
        if not isinstance(bond, list) or len(bond) != 3 or any(type(v) is not int for v in bond):
            raise ValueError("Bonds must contain three integers")
        a, b, coupling = bond
        edge = tuple(sorted((a, b)))
        if edge not in expected or edge in canonical or coupling not in (-1, 1):
            raise ValueError("Invalid, duplicate, non-neighbor or non-binary bond")
        canonical[edge] = coupling
    return {
        "length": length,
        "boundary": "open",
        "bonds": [[a, b, canonical[a, b]] for a, b in sorted(canonical)],
    }


def instance_digest(payload: dict) -> str:
    canonical = canonical_instance(payload)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def energy_certificate(payload: dict, spins: list) -> dict:
    model = canonical_instance(payload)
    if (
        not isinstance(spins, list)
        or len(spins) != model["length"] ** 3
        or any(type(s) is not int or s not in (-1, 1) for s in spins)
    ):
        raise ValueError("Witness must contain exactly L^3 integer bipolar spins")
    bonds = model["bonds"]
    energy = -sum(coupling * spins[a] * spins[b] for a, b, coupling in bonds)
    unsatisfied = sum(coupling * spins[a] * spins[b] < 0 for a, b, coupling in bonds)
    if energy != 2 * unsatisfied - len(bonds):
        raise AssertionError("Energy and unsatisfied-bond counts disagree")
    return {"energy": energy, "unsatisfied_bonds": unsatisfied, "bonds": len(bonds)}


def verify_witnesses(payload: dict, document: dict) -> dict:
    """Certify energies only; timestamps/provenance are not optimality proofs."""
    if document.get("schema") != "ea3d-witness-v1":
        raise ValueError("Unsupported witness schema")
    if document.get("instance_sha256") != instance_digest(payload):
        raise ValueError("Instance digest mismatch")
    witnesses = document.get("witnesses")
    if not isinstance(witnesses, list) or not witnesses:
        raise ValueError("At least one witness is required")
    results = []
    for witness in witnesses:
        check = energy_certificate(payload, witness["spins"])
        if type(witness.get("energy")) is not int or witness["energy"] != check["energy"]:
            raise ValueError("Witness energy mismatch")
        results.append(check)
    return {"valid": True, "witnesses": results, "optimality_proven": False}


def budget_witnesses(payload, events, states, seconds, groups, budget):
    """Select strict-window incumbents, including the observed initialization."""
    if (
        type(groups) is not int
        or groups < 1
        or not isinstance(budget, (float, int))
        or not math.isfinite(budget)
        or budget <= 0
        or len(events) != len(states)
        or len(events) != len(seconds)
    ):
        raise ValueError("Invalid budget, group count or event lengths")
    selected = [None] * groups
    last_time = -1.0
    for row, (event, state, elapsed) in enumerate(zip(events, states, seconds, strict=True)):
        if len(event) != 5 or any(type(v) is not int for v in event):
            raise ValueError("Expected five integer improvement columns")
        if (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < last_time
        ):
            raise ValueError("Observation times must be finite, nonnegative and ordered")
        if elapsed < 0:
            raise ValueError("Negative observation time")
        last_time = elapsed
        sweep, phase, group, slot, energy = event
        if not 0 <= group < groups or sweep < 0 or slot < 0 or phase not in (0, 1, 2):
            raise ValueError("Invalid improvement coordinates")
        if elapsed <= budget and (selected[group] is None or energy < selected[group]["energy"]):
            selected[group] = {
                "group": group,
                "event_row": row,
                "sweep": sweep,
                "phase": phase,
                "slot": slot,
                "observed_seconds": elapsed,
                "energy": energy,
                "spins": state,
            }
    if any(item is None for item in selected):
        raise ValueError("Missing in-budget witness for a group")
    for item in selected:
        if hasattr(item["spins"], "tolist"):
            item["spins"] = item["spins"].tolist()
    document = {
        "schema": "ea3d-witness-v1",
        "instance_sha256": instance_digest(payload),
        "sampling_budget_seconds": budget,
        "optimality_proven": False,
        "witnesses": selected,
    }
    verify_witnesses(payload, document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("witnesses", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_witnesses(
                json.loads(args.instance.read_text()), json.loads(args.witnesses.read_text())
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
