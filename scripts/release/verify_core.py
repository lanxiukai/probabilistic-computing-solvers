"""Verify public RTX 5090 certificates on CPU, including unsuccessful decoded states."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays_identity(packet):
    h = hashlib.sha256()
    for key in sorted(packet):
        value = np.asarray(packet[key])
        h.update(key.encode() + b"\0" + value.dtype.str.encode() + b"\0")
        h.update(
            json.dumps(list(value.shape)).encode() + b"\0" + np.ascontiguousarray(value).tobytes()
        )
    return h.hexdigest()


def external_input(root, description, fetch):
    path = root / description["cache_file"]
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"Fetch the pinned external input first: {path}")
        with urllib.request.urlopen(description["url"], timeout=30) as response:
            payload = response.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("Unexpected external benchmark size")
        expected_download = description.get("download_sha256", description["sha256"])
        if hashlib.sha256(payload).hexdigest() != expected_download:
            raise ValueError("Downloaded benchmark checksum differs")
        if description.get("compression") == "gzip":
            payload = gzip.decompress(payload)
        if hashlib.sha256(payload).hexdigest() != description["sha256"]:
            raise ValueError("Decoded benchmark checksum differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    if digest(path) != description["sha256"]:
        raise ValueError("Cached external input changed")
    return path


def load_instance(root, directory, description, fetch):
    if "file" in description:
        path = directory / description["file"]
        if digest(path) != description["file_sha256"]:
            raise ValueError("Published instance bytes changed")
        if path.suffix == ".json":
            packet = json.loads(path.read_text())
            actual = hashlib.sha256(
                json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        else:
            with np.load(path, allow_pickle=False) as data:
                packet = dict(data)
            actual = arrays_identity(packet)
    elif "generator" in description:
        from combinatorial_solvers.maxcut.dense.model import instance_from_problem

        model = instance_from_problem(description["generator"])
        packet = {
            "size": np.asarray(model.size),
            "family": np.asarray(model.family),
            "packed": model.packed,
        }
        actual = arrays_identity(packet)
    else:
        path = external_input(root, description["external"], fetch)
        if description["kind"] == "partition":
            from combinatorial_solvers.partition.model import read_metis

            model = read_metis(path)
        else:
            from combinatorial_solvers.maxcut.model import from_mqlib

            model = from_mqlib(path)
        packet = {
            "size": np.asarray(model.size),
            "edges": model.edges,
            "weights": model.weights,
        }
        actual = arrays_identity(packet)
    if actual != description["canonical_identity"]:
        raise ValueError("Reconstructed mathematical instance differs")
    return packet


def binary(values):
    values = np.asarray(values)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("A certificate contains a nonbinary state")
    return values


def graph_cut(bits, packet):
    bits = binary(bits)
    flat = bits.reshape(-1, int(packet["size"]))
    edges, weights = packet["edges"], packet["weights"]
    result = []
    for state in flat:
        result.append(np.sum((state[edges[:, 0]] != state[edges[:, 1]]) * weights, dtype=np.int64))
    return np.asarray(result).reshape(bits.shape[:-1])


def dense_cuts(bits, packet):
    """Independent big-integer signed-spin check, without a floating matrix."""
    bits = binary(bits)
    n, family = int(packet["size"]), str(packet["family"])
    payload, offset, rows, positive = packet["packed"], 0, [], 0
    for i in range(n - 1):
        width = n - i - 1
        length = (width + 7) // 8
        row = np.unpackbits(payload[offset : offset + length], count=width)
        mask = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        rows.append((width, mask, mask.bit_count()))
        positive += mask.bit_count()
        offset += length
    total = 2 * positive - n * (n - 1) // 2 if family == "sk" else positive
    cuts = []
    for state in bits.reshape(-1, n):
        value = int.from_bytes(np.packbits(state, bitorder="little").tobytes(), "little")
        energy = 0
        for width, mask, count in rows:
            spin, value = 2 * (value & 1) - 1, value >> 1
            field = (
                width - 2 * (mask ^ value).bit_count()
                if family == "sk"
                else 2 * (mask & value).bit_count() - count
            )
            energy += spin * field
        if (total - energy) % 2:
            raise ValueError("Dense energy parity differs")
        cuts.append((total - energy) // 2)
    return np.asarray(cuts).reshape(bits.shape[:-1])


def ldpc_check(packet, state, record, evaluation):
    bits = binary(state["best_bits"])
    n, costs, targets = int(packet["size"]), packet["costs"], packet["targets"]
    pointers, columns = packet["pointers"], packet["columns"]
    counts = np.add.reduceat(bits[..., columns], pointers[:-1], axis=-1, dtype=np.int64)
    original = np.count_nonzero(counts % 2 != targets[:, None, :], axis=-1)
    model = original
    if bits.shape[-1] != n:
        assert np.all(np.diff(pointers) == 6)
        aux = bits[..., n:].reshape(*bits.shape[:-1], len(pointers) - 1, 2)
        residual = counts - 2 * aux[..., 0] - 4 * aux[..., 1] - targets[:, None, :]
        model = np.count_nonzero(residual, axis=-1)
    cost = np.sum(bits[..., :n] * costs[:, None, :], axis=-1, dtype=np.int64)
    for key, actual in (
        ("best_cost", cost),
        ("best_violations", model),
        ("best_original_violations", original),
    ):
        np.testing.assert_array_equal(state[key], actual)
    winners = state.get("winner_replica", np.zeros(len(costs), dtype=np.int64))
    selected = bits[np.arange(len(costs)), winners, :n]
    np.testing.assert_array_equal(selected, state["decoded_error"])
    valid = model[np.arange(len(costs)), winners] == 0
    code_valid = original[np.arange(len(costs)), winners] == 0
    expected_q = np.rint(256 * np.minimum(np.abs(evaluation["llr"]), 32)).astype(np.int64)
    np.testing.assert_array_equal(costs, expected_q)
    np.testing.assert_array_equal(evaluation["hard_decision"], evaluation["true_error"])
    hard_syndrome = (
        np.add.reduceat(
            evaluation["hard_decision"][:, columns], pointers[:-1], axis=-1, dtype=np.int64
        )
        % 2
    )
    np.testing.assert_array_equal(hard_syndrome, targets)
    bit_errors = np.count_nonzero(selected != evaluation["true_error"], axis=-1)
    metrics = record["metrics"]
    assert int(valid.sum()) == metrics["valid_frames"]
    assert int(code_valid.sum()) == metrics["codeword_valid_frames"]
    assert int(np.count_nonzero(bit_errors)) == metrics["frame_errors"]
    if "certified_correct_frames" in metrics:
        assert (
            int(np.count_nonzero(valid & (bit_errors == 0))) == metrics["certified_correct_frames"]
        )
    np.testing.assert_array_equal(bit_errors, metrics["frame_bit_errors"])
    return int(np.prod(bits.shape[:-1]))


def verify_panel(root, panel, fetch=False):
    directory = root / "data" / panel / "rtx5090"
    calls = json.loads((directory / "calls.json").read_text())
    descriptions = json.loads((directory / "instances.json").read_text())
    if panel == "tsp":
        from combinatorial_solvers.tsp.instance import certify_tour
        from combinatorial_solvers.tsp.tsplib import load_tsplib

        model = load_tsplib(external_input(root, descriptions[0]["external"], fetch))
        for call in calls:
            witness_path = directory / call["witness_file"]
            assert digest(witness_path) == call["witness_sha256"]
            witness = json.loads(witness_path.read_text())
            assert certify_tour(model, witness["best_tour"]).length == witness["best_length"]
            first = witness["first_certificate"]
            if first:
                assert certify_tour(model, first["tour"]).length == first["length"] == 26524
        return {"panel": panel, "calls": len(calls), "status": "verified"}
    instances = {item["id"]: load_instance(root, directory, item, fetch) for item in descriptions}
    verified = 0
    for call in calls:
        path = directory / call["witness_file"]
        if digest(path) != call["witness_sha256"]:
            raise ValueError("Published witness bytes changed")
        with np.load(path, allow_pickle=False) as data:
            state = dict(data)
        bits = binary(state["best_bits"])
        packet = instances[call["instance"]]
        metrics = call["metrics"]
        if panel == "ldpc":
            assert digest(directory / call["evaluation_file"]) == call["evaluation_sha256"]
            with np.load(directory / call["evaluation_file"], allow_pickle=False) as data:
                verified += ldpc_check(packet, state, call, dict(data))
            continue
        if panel == "mwis":
            weights, edges = np.asarray(packet["weights"]), np.asarray(packet["edges"])
            objective = np.atleast_1d(np.sum(bits * weights, axis=-1, dtype=np.int64))
            feasible = np.atleast_1d(
                ~np.any(bits[..., edges[:, 0]] & bits[..., edges[:, 1]], axis=-1)
            )
            stored = np.atleast_1d(state.get("best_weight", objective))
            marked = stored >= 0
            assert np.all(feasible[marked])
            np.testing.assert_array_equal(stored[marked], objective[marked])
            assert (int(np.max(objective[marked])) if marked.any() else -1) == metrics["weight"]
        elif panel == "maxsat":
            from combinatorial_solvers.maxsat.model import Instance

            model = Instance(int(packet["variables"]), packet["literals"], packet["weights"])
            objective = model.cost(bits)
            assert int(np.min(objective)) == metrics["cost"]
            if "best_cost" in state:
                np.testing.assert_array_equal(state["best_cost"], objective)
            if call["first_certificate"]:
                first = call["first_certificate"]
                assert int(model.cost(state["first_certificate_bits"])) == first["certified_cost"]
                assert first["certified_cost"] <= first["target_cost"]
        elif panel == "labs":
            from combinatorial_solvers.labs.model import correlations

            correlation = correlations(bits)
            objective = (correlation * correlation).sum(-1, dtype=np.int64)
            assert int(np.min(objective)) == metrics["energy"]
        else:
            objective = (
                dense_cuts(bits, packet) if panel == "maxcut-dense" else graph_cut(bits, packet)
            )
            if panel == "partition":
                assert np.all(bits.sum(-1) == int(packet["size"]) // 2)
                assert int(np.min(objective)) == metrics["best_cut"]
            else:
                assert int(np.max(objective)) == metrics["cut"]
            if "best_cut" in state:
                np.testing.assert_array_equal(state["best_cut"], objective)
        verified += len(np.atleast_2d(bits))
    return {"panel": panel, "calls": len(calls), "best_states": verified, "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch missing pinned public benchmarks"
    )
    args = parser.parse_args()
    for panel in args.panel:
        print(json.dumps(verify_panel(args.root, panel, args.fetch)), flush=True)


if __name__ == "__main__":
    main()
