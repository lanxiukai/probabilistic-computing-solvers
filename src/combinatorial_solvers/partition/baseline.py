"""Unmodified KaFFPa strong calls with strict balance and complete witnesses."""

from __future__ import annotations

import os
import subprocess
from time import perf_counter

import numpy as np

from combinatorial_solvers.workflows.runtime import dump_json


def run(graph, graph_path, config, output, initial=None):
    started = perf_counter()
    if initial is None:
        initial = np.random.default_rng(config["seed"]).permutation(
            np.repeat(np.arange(2, dtype=np.int8), graph.size // 2)
        )
    best = initial.copy()
    best_cut = int(graph.certify(best))
    labels, cuts, times, eligible = [best.copy()], [best_cut], [0.0], [True]
    commands, completed, quality_calls = [], 0, 0
    fixed = config.get("fixed_calls")
    one_pass = config.get("one_pass", False)
    while fixed is None or completed < fixed:
        if fixed is None and (
            (one_pass and completed == 1) or perf_counter() - started >= config["max_seconds"]
        ):
            break
        call_dir = output / f"call-{completed:04d}"
        call_dir.mkdir()
        candidate_path = call_dir / "partition.txt"
        command = [
            str(config["executable"]),
            str(graph_path),
            "--k=2",
            "--imbalance=0",
            "--enforce_balance",
            "--preconfiguration=strong",
            f"--seed={(config['seed'] + completed) % (2**31 - 1)}",
            f"--output_filename={candidate_path}",
        ]
        commands.append(command)
        with (
            (call_dir / "stdout.txt").open("w") as stdout,
            (call_dir / "stderr.txt").open("w") as stderr,
        ):
            subprocess.run(
                command,
                cwd=call_dir,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                timeout=config.get("call_timeout_seconds", 180),
                check=True,
            )
        candidate = np.loadtxt(candidate_path, dtype=np.int8)
        value = int(graph.certify(candidate))
        elapsed = perf_counter() - started
        completed += 1
        in_budget = (
            completed <= config["quality_calls"]
            if "quality_calls" in config
            else elapsed <= config["max_seconds"]
        )
        labels.append(candidate)
        cuts.append(value)
        times.append(elapsed)
        eligible.append(in_budget)
        if in_budget:
            quality_calls = completed
            if value < best_cut:
                best, best_cut = candidate.copy(), value
    dump_json(output / "commands.json", commands)
    return {
        "initial_bits": initial[None, :],
        "initial_cut": np.asarray([graph.cut(initial)]),
        "best_bits": best[None, :],
        "best_cut": np.asarray([best_cut]),
        "final_bits": labels[-1][None, :],
        "final_cut": np.asarray([cuts[-1]]),
        "trace_bits": np.asarray(labels)[:, None, :],
        "trace_cut": np.asarray(cuts)[:, None],
        "trace_seconds": np.asarray(times),
        "trace_eligible": np.asarray(eligible),
        "trace_call": np.arange(len(labels)),
    }, {
        "method": "kaffpa-strong",
        "best_cut": best_cut,
        "replicas": 1,
        "cpu_threads": 1,
        "physical_update_pbits_per_model": 0,
        "completed_calls": completed,
        "quality_calls": quality_calls,
        "search_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": 0,
    }
