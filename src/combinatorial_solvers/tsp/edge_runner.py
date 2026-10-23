"""Sparse candidate-arc p-bit optimization with explicit classical controls.

LKH builds one common prefix. The binary network solves a restricted assignment
relaxation; sparse matching, subtour splicing and GPU 2-opt are classical readout.
Every produced readout tour is certified, with prefix and component provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import numpy as np
import torch

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.core.source_archive import pack
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    require_mapping,
    validate_experiment_output,
)

from .configuration import _target_length
from .edge_model import assignment_readout, build_assignment, splice_cycles
from .hybrid import perturb_tour
from .hybrid_backend import LocalSearchBackend
from .hybrid_classical import _read_lkh_tour
from .instance import certify_tour
from .tsplib import load_tsplib

SCHEMA = "tsp-candidate-edge-pbit-v1"


@dataclass(frozen=True)
class EdgeConfig:
    max_seconds: float
    lkh_sha256: str
    kind: str = "edge_assignment"
    device: str = "cuda"
    method: str = "pbit"
    seed: int = 41001
    neighbors: int = 16
    replicas: int = 8
    rounds: int = 4
    sweeps: int = 128
    readouts: int = 8
    beta_start: float = 0.05
    beta_stop: float = 8
    penalty: float = 2
    readout_bias: float = 4
    local_moves: int = 128
    prefix_seconds: float = 5
    prefix_trials: int = 1
    max_bits: int = 36000

    def __post_init__(self):
        if self.kind != "edge_assignment" or self.device != "cuda":
            raise ValueError("The candidate-edge runner requires kind=edge_assignment, device=cuda")
        if self.method not in {"pbit", "random", "assignment", "ils"}:
            raise ValueError("Unknown candidate-edge method")
        for name in (
            "seed",
            "neighbors",
            "replicas",
            "rounds",
            "sweeps",
            "readouts",
            "local_moves",
            "prefix_trials",
            "max_bits",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid positive integer {name}")
        for name in (
            "max_seconds",
            "beta_start",
            "beta_stop",
            "penalty",
            "readout_bias",
            "prefix_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive finite {name}")
        if self.readouts > self.replicas or self.replicas > 64 or self.max_bits > 36000:
            raise ValueError("Require readouts <= replicas <= 64 and max_bits <= 36000")
        if len(self.lkh_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.lkh_sha256):
            raise ValueError("Expected a pinned LKH executable identity")


def run_config(
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
    overwrite: bool = False,
    repository: Path | None = None,
) -> Path:
    del overwrite
    solver = EdgeConfig(**require_mapping(config, "solver"))
    problem = require_mapping(config, "problem")
    output = experiment_output(config, output_override).resolve()
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository or Path(__file__).resolve().parents[3]).resolve()
    started = perf_counter()
    stop = started + solver.max_seconds
    prepared = started
    best = first_hit = None
    event_index, completed_rounds, sampled_sweeps = 0, 0, 0
    phases = {key: 0.0 for key in ("prefix", "network_setup", "sampling", "readout", "polishing")}
    dump_json(output / "config.json", {"schema": SCHEMA, "input": config})
    try:
        input_path = output / "input.tsp"
        shutil.copy2(problem["instance"], input_path)
        instance = load_tsplib(input_path, optimum=problem.get("optimum"))
        target = (
            _target_length(problem, instance.optimum)
            if (
                instance.optimum is not None
                or "target_length" in problem
                or "target_relative_gap" in problem
            )
            else None
        )
        source = pack(root, output / "source")
        dump_json(
            output / "environment.json",
            {
                **capture_provenance(root),
                "source_archive_sha256": source["archive_sha256"],
                "gpu": torch.cuda.get_device_name(0),
            },
        )
        torch.cuda.reset_peak_memory_stats()
        with (output / "events.jsonl").open("x", buffering=1) as journal:

            def observe(tour, stage, **details):
                nonlocal best, first_hit, event_index
                certificate = certify_tour(instance, tour)
                if not certificate.is_permutation or certificate.length is None:
                    raise AssertionError("A candidate is not a Hamiltonian tour")
                now = perf_counter()
                event = {
                    "event_index": event_index,
                    "stage": stage,
                    "tour": tour,
                    "length": certificate.length,
                    "elapsed_seconds": now - started,
                    **details,
                }
                journal.write(json.dumps(event) + "\n")
                if best is None or certificate.length < best["length"]:
                    best = {
                        "tour": tour,
                        "length": certificate.length,
                        "stage": stage,
                        "event_index": event_index,
                    }
                if first_hit is None and target is not None and certificate.length <= target:
                    first_hit = {
                        **best,
                        "certified_elapsed_seconds": now - started,
                        "certified_monotonic_seconds": now,
                        "search_to_certificate_seconds": None
                        if stage == "prefix"
                        else now - prepared,
                    }
                event_index += 1

            prefix_started = perf_counter()
            work = output / "prefix"
            work.mkdir()
            executable = Path(require_mapping(config, "prefix")["lkh_executable"]).resolve()
            if hashlib.sha256(executable.read_bytes()).hexdigest() != solver.lkh_sha256:
                raise ValueError("LKH prefix binary identity mismatch")
            shutil.copy2(executable, work / "solver")
            remaining = min(solver.prefix_seconds, stop - perf_counter())
            if remaining <= 0:
                raise TimeoutError("Preparation exhausted the runner budget before its prefix")
            (work / "run.par").write_text(
                "PROBLEM_FILE = ../input.tsp\nOUTPUT_TOUR_FILE = prefix.tour\nRUNS = 1\n"
                f"MAX_TRIALS = {solver.prefix_trials}\nSEED = {solver.seed}\nPRECISION = 100\n"
                f"TIME_LIMIT = {remaining * 0.8:.9f}\nTRACE_LEVEL = 1\n"
            )
            prefix_timeout = False
            with (work / "run.log").open("xb") as log:
                try:
                    native = subprocess.run(
                        [str(work / "solver"), "run.par"],
                        cwd=work,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        timeout=remaining,
                        check=False,
                    )
                    if native.returncode:
                        raise RuntimeError("LKH prefix failed; original output preserved")
                except subprocess.TimeoutExpired:
                    prefix_timeout = True
            if not (work / "prefix.tour").exists():
                raise RuntimeError("LKH prefix produced no complete tour within its cap")
            prefix = _read_lkh_tour(instance, work / "prefix.tour")
            observe(prefix, "prefix", timed_out=prefix_timeout)
            phases["prefix"] = perf_counter() - prefix_started

            setup_started = perf_counter()
            model = build_assignment(
                instance, [prefix], neighbors=solver.neighbors, penalty=solver.penalty
            )
            if model.bits > solver.max_bits:
                raise ValueError("Candidate network exceeds this campaign's physical-node cap")
            metadata = model.metadata()
            dump_json(output / "model.json", metadata)
            color_indices = np.concatenate(model.colors)
            color_offsets = np.cumsum([0] + [len(color) for color in model.colors])
            np.savez_compressed(
                output / "model.npz",
                source=model.source,
                destination=model.destination,
                cost=model.cost,
                colors=color_indices,
                color_offsets=color_offsets,
            )
            dump_json(
                output / "config.json",
                {
                    "schema": SCHEMA,
                    "input": config,
                    "resolved": {
                        "solver": asdict(solver),
                        "model": metadata,
                        "target_length": target,
                        "reference_optimum_sent_to_solver": False,
                        "reference_tour_loaded": False,
                        "instance_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "network_scope": "restricted assignment relaxation, not full TSP",
                        "capture": ("initial/final state of every anneal; all tours; seeded RNG"),
                    },
                },
            )
            rng, classical_rng = np.random.default_rng(solver.seed), Random(solver.seed)
            initial = (rng.random((solver.replicas, model.bits)) < 1 / solver.neighbors).astype(
                np.int8
            )
            initial[0] = model.encode(prefix)
            sampler = None
            if solver.method == "pbit":
                from .edge_cuda import CudaAssignment

                sampler = CudaAssignment(model, initial, seed=solver.seed)
            backend = LocalSearchBackend(
                instance,
                device="cuda",
                method="two_opt",
                max_moves=solver.local_moves,
                capacity=solver.readouts,
                tile=256,
            )
            np.savez_compressed(output / "initial.npz", states=initial)
            phases["network_setup"] = perf_counter() - setup_started
            prepared = perf_counter()
            stopped = "round_limit"
            for round_id in range(solver.rounds):
                if perf_counter() >= stop:
                    stopped = "wall_clock_limit"
                    break
                phase_started = perf_counter()
                if sampler is not None:
                    sampler.advance(solver.sweeps, solver.beta_start, solver.beta_stop)
                    sampled_sweeps += solver.sweeps
                    snapshot = sampler.snapshot()
                    states = snapshot["states"]
                else:
                    states = (
                        initial
                        if round_id == 0
                        else (rng.random(initial.shape) < 1 / solver.neighbors).astype(np.int8)
                    )
                    snapshot = {"states": states}
                energies = model.energies(states)
                outgoing, incoming = model.counts(states)
                snapshot.update(energies=energies, outgoing=outgoing, incoming=incoming)
                np.savez_compressed(
                    output / f"states-{round_id}.npz",
                    states=states,
                    energies=energies,
                    outgoing=outgoing,
                    incoming=incoming,
                    flips=snapshot.get("flips", np.zeros(states.shape, dtype=np.int32)),
                )
                phases["sampling"] += perf_counter() - phase_started
                phase_started = perf_counter()
                tours = []
                chosen = np.argsort(energies, kind="stable")[: solver.readouts]
                for replica in chosen:
                    if perf_counter() >= stop:
                        stopped = "wall_clock_limit"
                        break
                    if solver.method == "ils":
                        assert best is not None
                        tour = perturb_tour(best["tour"], classical_rng)
                        cycles = None
                    else:
                        successor = assignment_readout(
                            model,
                            None if solver.method == "assignment" else states[replica],
                            bias=solver.readout_bias,
                        )
                        tour, cycles = splice_cycles(instance, successor)
                    feasible_assignment = bool(
                        np.all(outgoing[replica] == 1) and np.all(incoming[replica] == 1)
                    )
                    observe(
                        tour,
                        "classical_readout",
                        round=round_id,
                        replica=int(replica),
                        raw_assignment_feasible=feasible_assignment,
                        readout_cycles=cycles,
                        raw_energy=float(energies[replica]),
                    )
                    tours.append(tour)
                    if solver.method == "assignment":
                        break
                phases["readout"] += perf_counter() - phase_started
                phase_started = perf_counter()
                if tours and perf_counter() < stop:
                    for index, tour in enumerate(
                        backend.solve(tours, lambda: stop - perf_counter())
                    ):
                        observe(tour, "polished_readout", round=round_id, candidate=index)
                phases["polishing"] += perf_counter() - phase_started
                completed_rounds += 1
                if solver.method == "assignment":
                    break
            elapsed = perf_counter() - started
        final_snapshot = sampler.snapshot() if sampler is not None else None
        assert best is not None
        dump_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": stopped,
                "method": solver.method,
                "device": "cuda",
                "best_tour": best["tour"],
                "best_length": best["length"],
                "best_stage": best["stage"],
                "prefix_length": certify_tour(instance, prefix).length,
                "target_length": target,
                "first_observed_target_hit": first_hit,
                "observed_target_met": first_hit is not None,
                "within_budget_hit": first_hit is not None
                and first_hit["certified_elapsed_seconds"] <= solver.max_seconds,
                "preparation_seconds": prepared - started,
                "search_seconds": elapsed - (prepared - started),
                "elapsed_seconds_before_summary_write": elapsed,
                "budget_overrun_seconds": max(0, elapsed - solver.max_seconds),
                "phase_seconds": phases,
                "search": {"completed_rounds": completed_rounds},
                "model": metadata,
                "sampled_sweeps_per_replica": sampled_sweeps,
                "attempted_bit_updates": sampled_sweeps * model.bits * solver.replicas,
                "distinct_nodes_flipped": 0
                if final_snapshot is None
                else int(np.count_nonzero(final_snapshot["flips"].sum(axis=0))),
                "total_flips": 0 if final_snapshot is None else int(final_snapshot["flips"].sum()),
                "simultaneous_replicas": solver.replicas,
                "total_simulated_binary_nodes": model.bits * solver.replicas if sampler else 0,
                "local_search_backend": backend.metadata(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "rng": (
                    "NumPy PCG64 initialization, Triton Philox sweep-seeded heat bath; "
                    "Python Random for ILS"
                ),
                "raw_sampler_tour_claimed": False,
            },
        )
    except BaseException as error:
        dump_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
                "best": best,
                "first_observed_target_hit": first_hit,
            },
        )
        raise
    return output
