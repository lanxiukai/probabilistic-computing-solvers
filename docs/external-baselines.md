# External references and build boundaries

English | [简体中文](external-baselines.zh-CN.md)

The root MIT license covers project-authored code. It does not relicense
upstream software, benchmark collections, or locally generated combinations.
The release includes project-owned adapters and required notices; it does not
redistribute upstream source archives or reference binaries.
The [version inventory](../data/release/external-baselines.json) records the exact
versions reviewed, rather than making a claim about a later upstream release.

| Reference | Pinned source | Observed terms / boundary |
|---|---|---|
| CHILS | `515952724cd3dcc6c4365a340ecf0f1da782119a` | MIT; original native-CPU build options |
| USW-LS | `1b1d216f91e7da81460a21f5cfffab5eed40d3c9` | No LICENSE observed at this revision; obtain and use source under applicable terms |
| MQLib | `585496274af5abb0849d0d47e135496b4688680b` | MIT; unmodified BURER2002 plus disclosed FESTA correction |
| KaHIP | `34e1d0a00afeae2ae2ebfe625826e9c8f8f40ac4` | MIT; unmodified KaFFPa |
| quantumgizmos/ldpc | `d3429964cd4ffe1abfc041c6ec8b8425cb174f40` | MIT; retain bundled dependency notices |
| tavildar/LDPC | `05ee7f4af36ed5dacf52861315af68b8a17e71e0` | MIT; WiFi matrix source |
| LKH | `3.0.13` | Author distribution terms; obtain separately |

No Anechoic author code is distributed: the dense dynamics implement paper
equations, while private author-function comparisons remain private. No pSA_LDPC
author code or matrices were executed/reused. The GPLv2 lMAts development control
is not part of the new 5090 L2 panel and is not bundled.

## Build project adapters

From this repository, put obtained sources/builds under ignored `external/`.
The builders preserve source, compilation logs, binaries, input identities, and
instrumentation locally. Start from a clean checkout at the exact revision.
Rebuilding can change the executable hash and must be disclosed when reproducing
the recorded reference; `--allow-rebuilt-baseline` records those differences.

```bash
git clone https://github.com/KarlsruheMIS/CHILS external/sources/CHILS
git -C external/sources/CHILS checkout --detach 515952724cd3dcc6c4365a340ecf0f1da782119a
uv run --frozen python scripts/mwis/prepare_baseline.py \
  --source external/sources/CHILS --output external/builds/chils

git clone https://github.com/MQLib/MQLib external/sources/MQLib
git -C external/sources/MQLib checkout --detach 585496274af5abb0849d0d47e135496b4688680b
uv run --frozen python scripts/maxcut/prepare_baseline.py \
  --source external/sources/MQLib --output external/builds/mqlib
```

The MQLib builder preserves and tests the sign/boolean correction
`assignments_[i] == 1`; its FESTA variant still has the documented visitation-order
limitation. The main sparse CPU comparison is BURER2002.

After building, compare source-member contents with the measured version:

```bash
uv run --frozen python scripts/release/verify_reference.py \
  --reference mqlib --directory external/builds/mqlib
```

The same check accepts `chils`, `usw-ls`, `kahip`, and `ldpc`. It compares source
and adapter contents independently of archive timestamps. Matching source does
not imply identical binaries or performance on different hosts.

For a legitimately obtained USW-LS checkout at the listed revision:

```bash
uv run --frozen python scripts/maxsat/prepare_baseline.py \
  --source external/sources/USW-LS --output external/builds/usw-ls
```

This script provides project-authored stopping/readiness/certificate observation
logic and generates the adapted source locally. Neither generated patches nor
adapted upstream source/binaries are redistributed by this repository. Access to
the external source remains a reproduction requirement; the adapter grants no
additional rights. Do not substitute an uninstrumented binary and call its timing
the same measurement.

KaHIP and LDPC builders consume the original GitHub API tarballs and matching
commit metadata, obtainable from `https://api.github.com/repos/OWNER/REPO/tarball/COMMIT`
and `https://api.github.com/repos/OWNER/REPO/commits/COMMIT`:

```bash
uv run --frozen python scripts/partition/prepare_baseline.py \
  --archive external/sources/kahip.tar.gz \
  --commit-json external/sources/kahip-commit.json --output external/builds/kahip

uv run --frozen python scripts/ldpc/prepare_references.py \
  --decoder-archive external/sources/decoder.tar.gz \
  --decoder-commit external/sources/decoder-commit.json \
  --wifi-archive external/sources/wifi.tar.gz \
  --wifi-commit external/sources/wifi-commit.json --output external/builds/ldpc
```

Use `KaHIP/KaHIP`, `quantumgizmos/ldpc`, and `tavildar/LDPC` with the revisions
above. The LDPC observer preserves unmodified author headers; it gives BP the
proper error probabilities and OSD the distinct scoring vector needed for full
log-odds ranking. This adaptation and the OSD size/iteration bounds are part of
the measured method.

Build LKH from the [author's distribution](http://webhotel4.ruc.dk/~keld/research/LKH-3/)
and place its executable as `LKH` in the supplied baseline directory. The runner
records the executable identity; the release does not supply a universal binary hash
for different compilers or hosts.

## Set cover: AC-CFT and fmt

AC-CFT revision `6cb63d0036223b8be9c4d01f95b296d641209f95` is MIT licensed;
fmt 10.2.1 retains its own license. The [AC-CFT manifest](../configs/set-cover/cft-source.json)
and [fmt manifest](../configs/set-cover/fmt-source.json) pin separately fetched sources.
Only project-authored adapters, fetchers and builders are distributed, without
original benchmarks, upstream source archives or binaries. The timer observer
preserves upstream search; replayed clock traces reproduce work and add no new
performance measurements. Build hashes can change with compiler, host and paths;
compare original source-member identities. Commands are in the
[set-cover source guide](set-cover/source-guide.md).

## JSSP

OR-Tools CP-SAT 9.15.6755 is an independent single-worker CPU reference under Apache-2.0. Original ScheduleOpt inputs are fetched from upstream with fixed hashes, not redistributed. The [report](jssp/report.md) records sources, strict deadlines and proof scope.

## Covering arrays

[NIST ACTS Basic/IPOG](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools) is an independent CPU construction reference. Its pinned jar is fetched separately from NIST and is excluded from the MIT source distribution. It chooses its own row count: complete coverage above the fixed row target remains a miss for that target. See the [report](covering-array/report.md) and [source guide](covering-array/source-guide.md).

Coloring keeps final confirmation, development comparisons and fixed-work implementation panels separate. Original DIMACS inputs are fetched by pinned hashes. The owned TabuCol-style program is an independent CPU reference, not an upstream HEAD reproduction. See the [report](graph-coloring/experiment-report.md) and [source guide](graph-coloring/source-guide.md).

## QAP

Owned attribute tabu and official Taillard Ro-TS are separate CPU references. The official 2006 source has non-commercial-use and acknowledgement conditions and is fetched/built separately through the owned adapter after explicit acceptance. Original/instrumented upstream source and binaries are excluded from the MIT distribution. The [report](qap/experiment-report.md) retains different seeds, resources, budgets and misses; the [reproduction guide](qap/README.md) pins sources and build steps.
