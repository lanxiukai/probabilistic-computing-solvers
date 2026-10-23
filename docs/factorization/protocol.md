# Frozen evaluation protocol

English | [简体中文](protocol.zh-CN.md)

The accepted [release-v3.json](../../configs/factorization/release-v3.json)
contains every target, role seed and final numerical configuration. It retains
the executed physical/control definitions; operational and intermediate-study
metadata are omitted. [Provenance](../../provenance/factorization/release-v3.json)
records the original and public plan hashes and the metadata-only rebinding.

The panel contains six widths × three targets × three seeds = 54 physical runs,
18 selected-root integer controls, 54 single-thread YAFU controls, and six
predesignated origin replays. Replays and cost measurements add no trials.
All 132 tasks completed. Two physical runs and one selected-root control
reached their event limits without success; none were omitted or replaced by
new seeds. Identical-tuple infrastructure recovery counts once.

| Bits | Replicas | Event cap | Sampling cap (s) | Root proposal |
|---|---|---|---|---|
| 64 | 8192 | 4096 | 120 | quotient_filter256_v1 |
| 80 | 4096 | 16384 | 120 | fermat_crt_filter256_v1 |
| 96 | 2048 | 65536 | 600 | fermat_crt128_filter256_v1 |
| 112 | 32 | 131072 | 720 | fermat_deferred262144_v1 |
| 120 | 32 | 524288 | 2880 | fermat_deferred262144_v1 |
| 128 | 32 | 1048576 | 6060 | fermat_deferred262144_v1 |

All physical runs use beta=12, no PT, no local Gibbs sweep, and
`buffer_sqrt_qr_v1`. Temperature-slot-shaped storage is retained for compatibility;
the replicas do not exchange temperatures. Stop at the first certified in-budget
hit, the event limit, or the sampling cap. A complete run has 300 seconds of
additional task allowance; YAFU has a 300-second process limit. JSON is the
authority for complete settings. These are limits, not predicted durations.

Targets are two distinct equal-width primes, with normalized product in
[0.75,0.875) and relative factor gap at least 0.1. Each stratum supplies one
target. Target/role-seed tuples define runs; replicas are not independent
benchmark instances. Compatibility/cap-calibration reuse of target-0 fixtures
after profile selection is disclosed in the report.

Only product and structural-zero pins are fixed. Initialization uses independent
physical Rademacher spins. Strict success needs accepted-state nontrivial
integer factors and full/conditioned zero energy, with transition checks.
Ordinary event/time limits retain censored outcomes; infrastructure interruptions
and new-seed trials must not be silently pooled. Six designated runs preserve
completed origin-replay receipts. Full source restoration and fresh run directories
are required for reproduction; single-GPU timings use exclusive serial execution.

Source, software, actual hardware, all timing fields, integer-control distinctions,
and small-sample limitations are described in the [report](experiment-report.md).
The public source contains no private generation-answer trace.
