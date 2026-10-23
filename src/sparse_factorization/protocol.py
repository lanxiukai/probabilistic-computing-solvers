"""Deterministic, answer-separated preparation of the release validation panel."""

from __future__ import annotations

import random

from .core.rng import derive_stream_seed
from .solver.primes import is_prime
from .solver.publication_io import canonical_digest

PROTOCOL = "sparse-factorization-release-v2"
SEED_NAMESPACE = "sparse-factorization-release-v1"
WIDTHS = (64, 80, 96, 112, 120, 128)
SETTINGS = {
    64: ("quotient_filter256_v1", 8192, 4096, 120),
    80: ("fermat_crt_filter256_v1", 4096, 16384, 120),
    96: ("fermat_crt128_filter256_v1", 2048, 65536, 600),
    112: ("fermat_deferred262144_v1", 32, 131072, 1800),
    120: ("fermat_deferred262144_v1", 32, 524288, 7200),
    128: ("fermat_deferred262144_v1", 32, 1048576, 14400),
}


def solver_config(width: int, target: int, name: str, repetition: int) -> dict:
    mode, replicas, events, seconds = SETTINGS[width]
    config = {
        "run_id": name,
        "protocol_id": PROTOCOL,
        "algorithm": "circuit_only",
        "target": target,
        "product_bits": width,
        "topology": "carry_chain",
        "model_storage": "edge_v1",
        "replicas": replicas,
        "temperature_slots": 32,
        "beta": [12.0, 12.0],
        "exchange_every": 0,
        "modular_root_mode": mode,
        "root_guidance": "partition_v1",
        "guidance_execution": "skip_unused_v1",
        "root_policy": "uniform_0_1_2_4_v2",
        "proposal_beta_scale": 1.0,
        "logit_clip": 16,
        "lookahead_strength": 1,
        "circuit_backend": "triton_graph_v1",
        "history_backend": "triton_device_v1",
        "old_state_cache": "validated_v1",
        "cdf_layout": "beta_group_context",
        "cdf_search": "scan32_v1",
        "maximum_events": events,
        "maximum_sampling_seconds": seconds,
        "maximum_reserved_vram_gib": 12,
        "chunk_events": 8192,
        "stop_on_first_native_hit": True,
        "repetition": repetition,
    }
    for role in ("initialization", "sampler", "circuit", "exchange"):
        config[role + "_seed"] = derive_stream_seed(20261017, f"{SEED_NAMESPACE}:{name}:{role}")
    return config


def generate_panel(excluded: set[int] | None = None) -> tuple[dict, dict]:
    """Generate fixed public N values and a separately stored private draw trace.

    Rejection conditions are frozen before generation. No solver is run and no
    generated factor is used to pick solver settings or search schedules.
    """
    excluded = set() if excluded is None else set(excluded)
    public, traces = [], []
    for width in WIDTHS:
        bits = width // 2
        for stratum in range(3):
            name = f"w{width}-holdout-{stratum}"
            seed = derive_stream_seed(20261017, f"{SEED_NAMESPACE}:generation:{name}")
            rng = random.Random(seed)
            trace = {"instance_id": name, "seed": seed, "raw_candidates": [], "pairs": []}
            traces.append(trace)

            def prime(trace=trace, rng=rng, bits=bits):
                while len(trace["raw_candidates"]) < 1_000_000:
                    value = rng.randrange(2 ** (bits - 2), 2 ** (bits - 1)) * 2 + 1
                    trace["raw_candidates"].append(value)
                    if is_prime(value):
                        return value
                raise RuntimeError("Frozen instance-generation budget exhausted")

            while True:
                a, b = prime(), prime()
                n = a * b
                reason = (
                    "equal"
                    if a == b
                    else "gap"
                    if 10 * abs(a - b) < max(a, b)
                    else "stratum"
                    if not (18 + stratum) * 2**width <= 24 * n < (19 + stratum) * 2**width
                    else "duplicate"
                    if n in excluded
                    else "accepted"
                )
                trace["pairs"].append({"a": a, "b": b, "reason": reason})
                if reason == "accepted":
                    break
            excluded.add(n)
            public.append(
                {
                    "instance_id": name,
                    "target": n,
                    "product_bits": width,
                    "factor_bits": [bits, bits],
                    "stratum": stratum,
                }
            )
    configs = []
    for instance in public:
        for repetition in range(3):
            config = solver_config(
                instance["product_bits"],
                instance["target"],
                instance["instance_id"] + f"-s{repetition}",
                repetition,
            )
            config["instance_id"] = instance["instance_id"]
            configs.append(config)
    initializer_keys = [
        derive_stream_seed(c["initialization_seed"], "p7_independent_physical_initialization_v1")
        & 0xFFFFFFFF
        for c in configs
    ]
    if len(set(initializer_keys)) != len(initializer_keys):
        raise ValueError("Effective CPU initialization stream collision")
    role_seeds = [
        c[role + "_seed"]
        for c in configs
        for role in ("initialization", "sampler", "circuit", "exchange")
    ]
    if len(set(role_seeds)) != len(role_seeds):
        raise ValueError("Role seed collision")
    physical_seconds = sum(c["maximum_sampling_seconds"] for c in configs)
    arithmetic = [c["run_id"] for c in configs if c["repetition"] == 0]
    replays = [
        c["run_id"] for c in configs if c["repetition"] == 0 and c["instance_id"].endswith("-0")
    ]
    cost_runs = []
    for repetition in range(5):
        for label, mode, events in (
            ("crt", "fermat_crt128_filter256_v1", 32768),
            ("deferred", "fermat_deferred262144_v1", 32),
        ):
            name = f"w128-cost-{label}-s{repetition}"
            cell = solver_config(128, 260329436338324092485864776144305861893, name, repetition)
            cell.update(
                modular_root_mode=mode,
                maximum_events=events,
                maximum_sampling_seconds=300,
                stop_on_first_native_hit=False,
                cost_only=True,
                task_overhead_seconds=120,
            )
            cost_runs.append(cell)
    plan = {
        "schema": "factor-release-plan-v2",
        "protocol_id": PROTOCOL,
        "frozen_date": "2026-10-18",
        "status": "prepared_not_executed",
        "purpose": "Independent final-version validation, not tuning or a general-input guarantee",
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "minimum_system_ram_gb": 48,
            "recommended_disk_gb": 80,
        },
        "population": {
            "balanced_factor_widths": True,
            "distinct_primes": True,
            "minimum_relative_factor_gap": 0.1,
            "normalized_product_interval": [0.75, 0.875],
            "strata": "Three equal public-N intervals within [0.75,0.875)",
            "scope": "Restricted semiprime panel; not all integers of each width",
        },
        "generation": {
            "algorithm": "MT19937 odd-candidate rejection, deterministic u64 primality",
            "seed_master": 20261017,
            "seed_namespace": SEED_NAMESPACE,
            "max_raw_candidates_per_instance": 1_000_000,
            "answer_visibility": "Private draw trace excluded from solver inputs",
        },
        "instances": public,
        "physical_runs": configs,
        "cost_runs": cost_runs,
        "cost_comparison": {
            "candidates_per_run": 268435456,
            "repetitions_per_arm": 5,
            "same_replicas": 32,
            "limits": "Different kernels/events; candidate cost only, not success-rate speedup",
        },
        "controls": {
            "selected_root_integer": {
                "paired_runs": arithmetic,
                "semantics": "Same full noise layout and selected proposals; no body/MH",
                "scope": "Selected-candidate control; not an exhaustive bank or tuned factorizer",
            },
            "classical": {
                "implementation": "YAFU",
                "source_revision": "c7d90283972820f99b3b4920bfc1eff05c1f6342",
                "source_url": "https://github.com/bbuhrow/yafu",
                "algorithm": "factor with noecm pretest plan; trial division, rho and SIQS",
                "build": "GNU C; AVX2/BMI2; ECM=1 OMP=0 NO_ZLIB=1; capture library versions",
                "threads": 1,
                "repetitions": 3,
                "wall_cap_seconds_per_run": 300,
                "targets": [i["instance_id"] for i in public],
                "runs": [
                    {
                        "run_id": f"{i['instance_id']}-yafu-s{repeat}",
                        "instance_id": i["instance_id"],
                        "target": i["target"],
                        "seed": derive_stream_seed(
                            20261017, f"{SEED_NAMESPACE}:yafu:{i['instance_id']}:{repeat}"
                        ),
                        "repetition": repeat,
                    }
                    for i in public
                    for repeat in range(3)
                ],
            },
        },
        "origin_replays": {
            "runs": replays,
            "max_sampling_multiplier": 1.5,
            "counts_as_independent_trial": False,
        },
        "budget": {
            "physical_sampling_cap_seconds": physical_seconds,
            "arithmetic_sampling_cap_seconds": physical_seconds // 3,
            "origin_sampling_cap_seconds": sum(SETTINGS[w][3] * 1.5 for w in WIDTHS),
            "task_overhead_cap_seconds": 900,
            "preflight_wall_cap_seconds": 7200,
            "maximum_campaign_wall_seconds": 432000,
            "unused_budget_recycled": False,
            "cloud_execution": "Requires the later supplied instance and session budget",
        },
        "stop_policy": {
            "per_run": "First strict hit or declared event/time limit",
            "late_hits": "Retain separately, count as censored within-budget failure",
            "failed_hits": "Continue other preregistered runs; no reseeding or retuning",
            "software_or_certificate_failure": "Stop campaign and retain partial evidence",
        },
        "statistics": {
            "independent_unit": "Entire run, not replicas",
            "primary": ["successes/attempts", "censoring", "first passage", "end-to-end wall"],
            "no_tts99": True,
            "all_frozen_outcomes_reported": True,
        },
        "preflight": [
            "CPU/GPU integer and certificate tests",
            "chunked/unsegmented trajectory equality",
            "source isolation",
            "capture overhead <=5% on matched long enough work",
            "runtime versions, actual RAM/disk/GPU, scalar baseline build identity",
        ],
    }
    return plan, {
        "schema": "private-generation-v1",
        "contains_known_factors": True,
        "plan_sha256": canonical_digest(plan),
        "traces": traces,
        "new_solver_trials": 0,
    }
