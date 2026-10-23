"""Isolated LDPC experiment entry point with no transmitted-word input."""

from __future__ import annotations

import resource
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combinatorial_solvers.core.provenance import capture_provenance
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec
from combinatorial_solvers.workflows.runtime import (
    dump_json,
    experiment_output,
    validate_experiment_output,
)

from .baseline import run as reference
from .belief import solve as belief
from .model import SCALE, Instance, quadratic_residual
from .solver import Settings, solve

SPECIFICATION = SolverSpec(
    "ldpc",
    "combinatorial_solvers.ldpc.runner:run_config",
    "Likelihood-ranked LDPC syndrome search",
    input_fields=(
        "problem.instance",
        "initial.errors",
        "baseline.executable",
        "baseline.source_archive",
        "baseline.driver",
        "baseline.license",
        "baseline.metadata",
        "baseline.build_log",
    ),
)


def registry():
    return SolverRegistry([SPECIFICATION])


def run_config(config, *, output_override=None, overwrite=False, repository=None):
    output = experiment_output(config, output_override)
    validate_experiment_output(output, False)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    torch.set_num_threads(1)
    root = Path(repository or Path(__file__).resolve().parents[3])
    dump_json(output / "config.json", config)
    try:
        dump_json(output / "environment.json", capture_provenance(root))
        instance = Instance.load(config["problem"]["instance"])
        instance.save(output / "instance.npz")
        representation = "native"
        if config["method"] == "native":
            initial = None
            if "initial" in config:
                with np.load(config["initial"]["errors"], allow_pickle=False) as packet:
                    if set(packet.files) != {"errors"}:
                        raise ValueError(
                            "Warm inputs may contain only the decoder's error estimate"
                        )
                    initial = packet["errors"]
            settings = Settings(**config["solver"])
            representation = settings.representation
            arrays, summary = solve(instance, settings, initial)
            summary["penalty_q"] = settings.penalty * SCALE
            summary["sufficient_exact_penalty_bound_met"] = bool(
                settings.penalty * SCALE > instance.costs.sum(-1).max()
            )
        elif config["method"] == "belief":
            arrays, summary = belief(instance, **config["belief"])
        elif config["method"] == "reference":
            arrays, summary = reference(instance, config["baseline"], output)
        else:
            raise ValueError("Unknown LDPC method")
        before = perf_counter()
        for name in ("initial", "best", "final"):
            bits = arrays[f"{name}_bits"]
            data = bits[..., : instance.code.size]
            cost = instance.cost(data)
            original_v = instance.violations(data)
            model_v = (
                (quadratic_residual(instance, bits) != 0).sum(-1)
                if representation == "quadratic"
                else original_v
            )
            for field, actual in (("cost", cost), ("violations", model_v)):
                key = f"{name}_{field}"
                if key in arrays and not np.array_equal(arrays[key], actual):
                    raise AssertionError(f"LDPC state certificate differs: {key}")
                arrays[key] = actual
            arrays[f"{name}_original_violations"] = original_v
        winners = arrays.get("winner_replica", np.zeros(instance.frames, dtype=np.int64))
        selected = arrays["best_bits"][np.arange(instance.frames), winners]
        if not np.array_equal(selected[:, : instance.code.size], arrays["decoded_error"]):
            raise AssertionError("Decoder output is not a saved physical state")
        strict_v = arrays["best_violations"][np.arange(instance.frames), winners]
        code_v = instance.violations(arrays["decoded_error"])
        if np.any((strict_v == 0) & (code_v != 0)):
            raise AssertionError("Feasible encoding did not decode to a valid codeword")
        summary["certification_seconds"] = perf_counter() - before
        summary["codeword_valid_frames"] = int(np.count_nonzero(code_v == 0))
        summary["valid_frames"] = int(np.count_nonzero(strict_v == 0))
        np.savez_compressed(output / "states.npz", **arrays)
        np.savez_compressed(output / "warm-start.npz", errors=arrays["decoded_error"])
        if config.get("exact_reference", False):
            before = perf_counter()
            errors, optimum, rank = instance.exact()
            np.savez_compressed(
                output / "exact-reference.npz", errors=errors, costs=optimum, rank=rank
            )
            actual = instance.cost(arrays["decoded_error"])
            if np.any((strict_v == 0) & (actual < optimum)):
                raise AssertionError("Feasible decoder cost is below exhaustive ML reference")
            summary["exact_ml_frames"] = int(
                np.count_nonzero((strict_v == 0) & (actual == optimum))
            )
            summary["exact_reference_seconds"] = perf_counter() - before
            summary["exact_rank"] = rank
        summary.update(
            status="completed",
            original_variables=instance.code.size,
            parity_checks=instance.code.checks,
            parity_edges=len(instance.code.columns),
            receiver_scale=SCALE,
            complete_runner_seconds=perf_counter() - started,
            common_initial_runner_seconds=config.get("initial", {}).get("runner_seconds", 0.0),
            peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        dump_json(
            output / "model.json",
            {
                "objective": "minimize sum(cost_i * error_i) subject to H error = target mod 2",
                "receiver_scale": SCALE,
                "original_variables": instance.code.size,
                "representation": representation,
                "auxiliary_pbits": summary["auxiliary_pbits"],
                "frames": instance.frames,
                "replicas": summary["replicas"],
                "transmitted_word_available_to_solver": False,
                "native_order_range": [
                    int(np.diff(instance.code.pointers).min()),
                    int(np.diff(instance.code.pointers).max()),
                ],
            },
        )
        dump_json(output / "benchmark_summary.json", summary)
    except BaseException as error:
        dump_json(
            output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    return output
