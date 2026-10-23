# Accepted benchmark overview

English | [简体中文](benchmark-overview.zh-CN.md)

These are accepted **RTX 5090 measurements**, not new experiments or estimates
for other devices. The fourteen topics include separate sparse and dense Max-Cut
panels. Problem sizes, targets, methods, and stopping rules differ, so elapsed
times across rows are not a ranking of problem difficulty or solver superiority.
For background, see [probabilistic computing and p-bits](probabilistic-computing.md).

## Scope, resources, and findings

The table selects the largest published model scale for each topic. Time is
normally the complete runner mean, including preparation, search, output, and
certification; exceptions are identified. Memory is the maximum recorded
per-call allocator peak for the selected method/scale panel, including its
parallel states.

**A** = PyTorch peak allocated memory. **R** = PyTorch peak reserved memory,
including allocator cache. They measure different quantities; neither is total
GPU occupancy or a theoretical minimum-memory requirement.

| Topic and selected method | Single-model scale | RTX 5090 time, parallel states, and peak memory | Main finding and boundary |
|---|---|---|---|
| [Integer factorization](factorization/experiment-report.md) | 128-bit product; 32448 total / 32256 free binary states | About 34.5 min median complete call; 32 replicas; 2.086 GiB R | 7/9 certified successes at 128 bits; failures remain in the denominator |
| [TSP, candidate-network p-bit workflow](tsp/candidate-report.md) | 2103 cities; 33,662 candidate-arc variables | 17.047 s preparation + process exit, one fixed-seed call; 8 replicas; 36.30 MiB A | 0/7 prefix improvements; terminal gap 0.2548%; connectivity is repaired classically, not embedded |
| [3D EA spin glass](spin-glass/experiment-report.md) | 32768 spins, 32³ | About 70–80 s; PT/SA: 768 states, PA: 2048 particles; 396–1547 MiB A | SA has the best mean at 32³ in this panel; large ground states are unknown |
| [MWIS, pair Gibbs](mwis/experiment-report.md) | 32768 variables | About 15.3 s; 16 trajectories; 37.9 MiB A | Mean quality improves 0.855% over single-site p-bit updates, but remains 3.489% below single-thread CHILS |
| [Weighted Max-SAT, p-bit](maxsat/experiment-report.md) | 32768 original variables; native soft clauses | About 8.8 s to the prescribed quality target and return; 16 trajectories; 169.2 MiB A | A target-time benefit against the measured CPU portfolios; no distinct heat-bath advantage established |
| [Sparse Max-Cut, heat-bath annealing](maxcut/experiment-report.md) | 32768 variables | About 16 s; 16 trajectories; 36.5 MiB A | Ordinary heat-bath means do not exceed BURER2002 on any of the four 32K graphs |
| [Dense Max-Cut, matched panel](maxcut/experiment-report.md) | Full 32768-variable models | About 121–132 s across condition means; 16 trajectories; 4.546 GiB A | The tested stochastic routes do not exceed same-GPU dSB; results are graph-family dependent |
| [LABS, p-bit](labs/experiment-report.md) | 32768 original variables; native high-order model | About 123 s; 16 trajectories; 76.8 MiB A | A 0.000750% energy improvement costs much more than cheap descent; no practical general advantage |
| [Balanced bisection, four-site cold heat bath](partition/experiment-report.md) | 32768 vertices; strict cardinality | About 128–251 s across graph means; 8 replicas; 173.9 MiB A | Behind KaFFPa; 0/12 warm improvements in the wider panel |
| [LDPC, cold likelihood search](ldpc/experiment-report.md) | Native: 32766 bits; quadratic: 32760 bits | Native: about 3.94 s / 88.6 MiB A; quadratic: about 4.60 s / 50.1 MiB A; each call has 8 frames × 4 replicas = 32 states | Native cold search returns some correct words; quadratic cold search returns no feasible outputs; overall behind BP/LSD |
| [Weighted set cover, feasible eight-bit heat bath](set-cover/report.md) | RAIL516: 47,311 original bits, no auxiliaries | 38.3254 s complete mean; 128 replicas; 228.638 MiB A / 236 MiB R | Heat bath 184/185/185 versus certified optimum 182; AC-CFT reaches 182 in all three calls, 2.2234 s complete mean |
| [JSSP, feasible priority-sequence heat bath](jssp/report.md) | ta51js: 37,500 redundant one-hot entries | 50.687 s complete mean; 256 replicas; 105.880 MiB A / 170 MiB R | 2950–2960 versus reference 2760; 12/12 calls hit optima on four smaller inputs; CP-SAT has lower complete mean time on every input |
| [Covering arrays, two-row column heat bath](covering-array/report.md) | 35×300×3: 31,500 one-hot entries | 48.7488 s complete mean; 128 replicas; 454.903 MiB A / 494.000 MiB R | Heat bath succeeds in 15/15 calls; plateau also succeeds in 15/15 and has more hits at 30 rows; no minimum-row proof |
| [Coloring, clique-supported single-vertex heat bath](graph-coloring/experiment-report.md) | inithx.i.1: 46,656 stored / 28,026 original factor-attached bits | 7.1736 s complete mean; 128 chains; 76.399 MiB A / 92.000 MiB R | Heat bath and plateau both achieve 128/128 zero conflicts in both calls; two 25-color inputs remain unsolved; no universal thermal advantage |
| [QAP, four-position conditional heat bath](qap/experiment-report.md) | tai150b: 150² = 22,500 one-hot entries | 35.7474 s complete mean; 512 replicas; 123.073 MiB A / 192.000 MiB R | Best 499,088,558, 0.0384679% above historical feasible reference; Nugent optima reached in 9/9 calls, Taillard references in 0/9 |

The scientific and practical motivations for these topics are summarized in the
[root README](../README.md). Application relevance does not imply that a measured
model already delivers a deployable advantage.

## How to interpret the numbers

- **One model versus parallel work.** Variable and auxiliary-bit counts belong
  to one encoded model. Replicas, temperature slots, particles, and channel
  frames are separate resource dimensions and are not independent new trials.
  These are software state counts, not fabricated p-bit-device counts.
- **Complete time versus search time.** The selected MWIS, sparse Max-Cut, dense
  Max-Cut, LABS, and partition search windows are 12, 15, 90, 120, and 45 s.
  Preparation, state output, repair, and certification can materially increase
  the complete time. Max-SAT stops at a declared quality target; the large LDPC
  calls use fixed work limits.
- **Factorization.** The 34.5 min median includes all nine 128-bit calls, including
  two event-limit failures of about 49–50 min. The successful-only sampling
  first-hit median is about 15.8 min, a different statistic. Total state includes
  clamped ports.
- **TSP.** The current candidate network runs binary heat-bath updates on GPU.
  Its 33,662 variables describe a restricted assignment relaxation, with eight
  replicas; global connectivity requires classical readout/repair. The terminal
  17.047 s is one complete call, not a target-hit time or a median. The earlier
  [150-city study](tsp/experiment-report.md) used a CPU selector and GPU local
  search; its 4.1 s median first certificate and 0.18 MiB peak remain historical.
- **Spin glass.** PT/SA use 4 groups × 96 temperature slots × 2 copies; PA uses
  4 groups × 128 × 4 particles. Allocated peaks are respectively 490.6, 396.2,
  and 1547.0 MiB. Successful complete means are 70.6, 74.1, and 69.4 s. Including
  one failed PA call gives 79.7 s per completed logical call, retained in the
  table's approximate 70–80 s range.
- **LDPC.** Four replicas are used per frame. The quadratic model contains
  16380 data bits and 16380 auxiliary bits, so its original code length differs
  from the native model's. A feasible codeword, the transmitted word, and a
  globally most-likely word are different claims.

The allocated/reserved figures exclude complete driver/context costs and other
processes. Cold and warm starts, classical CPU controls, and different methods
are kept separate in the linked reports. Earlier factorization/spin-glass panels
and the additional combinatorial campaign have their own environment records;
sharing a GPU model does not make every measurement an identical environment.

## Data and reproduction

The [resource ledger](../data/release/benchmark-resources.json) identifies selected
calls, aggregation rules, parallel counts, and memory fields. It summarizes
the earlier accepted measurements and adds no trials. The historical TSP selector counts and
allocator peaks are additional metadata extracted from those same archived
calls; their source-summary and call-manifest fingerprints are retained.

The current TSP row uses its separate [metrics](../data/tsp/candidate-network/analysis/metrics.json)
and [complete calls](../data/tsp/candidate-network/rtx5090/calls.json).

Most values can be traced directly to:

- [Factorization aggregates](../data/factorization/release-v3/summary/aggregates.json)
  and the corresponding per-call records linked by its report.
- [Spin-glass calls](../data/spin-glass/s23/calls.json).
- The existing per-panel calls linked by the reports in the table, including
  [TSP](../data/tsp/rtx5090/calls.json),
  [MWIS](../data/mwis/rtx5090/calls.json),
  [Max-SAT](../data/maxsat/rtx5090/calls.json),
  [sparse Max-Cut](../data/maxcut-sparse/rtx5090/calls.json),
  [dense Max-Cut](../data/maxcut-dense/rtx5090/calls.json),
  [LABS](../data/labs/rtx5090/calls.json),
  [partition](../data/partition/rtx5090/calls.json), and
  [LDPC](../data/ldpc/rtx5090/calls.json).

The earlier additional combinatorial campaign remains **365 natural calls and 35
separate replays**; the new TSP study adds 28 natural calls and one separate replay.
Factorization and spin glass retain their original panels.
The [reproduction guide](reproduction.md) explains CPU certificate verification,
fresh execution, replay, and external baseline requirements. All numerical
results here come from these repository sources; external research references
and their limits remain in the complete project reports.

Set cover has separate [calls](../data/set-cover/rtx5090/calls.json),
[complete denominators and frozen targets](../data/set-cover/rtx5090/panel.json),
and [environment](../data/set-cover/rtx5090/environment.json). Its trials are
kept separate from the historical campaign.
