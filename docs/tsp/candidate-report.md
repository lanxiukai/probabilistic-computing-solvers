# RTX 5090 candidate-network p-bit workflow: measured results

English | [简体中文](candidate-report.zh-CN.md)

**The p-bit hybrid workflow was measured on seven instances from 200 to 2103
cities. It did not improve the LKH prefix on any instance.** The one optimum hit,
kroA200, was already present in the classical prefix. This release provides the
GPU implementation, complete measured outcomes and reproducible certificates;
it establishes no independent p-bit quality or speed advantage.

## Method, device and frozen evaluation

The [method](candidate-network.md) combines classical LKH initialization,
candidate-arc GPU p-bit heat-bath updates, classical matching and subtour repair,
CUDA integer 2-opt, and exact integer tour certification. The binary network
encodes a restricted assignment relaxation; it does not encode global TSP
connectivity. Classical supporting stages and the retained prefix are attributed
separately from p-bit sampling.

Measurements used **NVIDIA GeForce RTX 5090**, compute capability 12.0, with
33,766,572,032 reported device bytes, and **AMD EPYC 9654 96-Core Emb Processor**.
The container exposed 384 logical CPUs and affinity entries; this is not a claim
of exclusive ownership of those CPUs. A unified cgroup CPU quota was not exposed.
LKH and numerical libraries were limited to one thread. Python 3.14.6,
PyTorch 2.13.0+cu130, Triton 3.7.1 and driver 615.71.09 were recorded.
The fixed LKH 3.0.13 source was rebuilt on this CPU.

The [frozen protocol](../../data/tsp/candidate-network/protocol.json) contains
**28 natural calls: 7 p-bit workflow calls and 21 attribution controls**, plus
**one separate d2103 p-bit origin replay**. Each instance uses one fixed seed.
All instances and seeds were seen during private development; this is device
confirmation, not unseen-instance generalization or a success-probability study.
The p-bit numerical parameters were retained. Standalone LKH and GA-EAX
performance calls from the earlier six-method plan were explicitly omitted
before execution under the selected resource scope. LKH still supplies the
shared initialization and its software correctness fixture.

## Primary results

The known optima are reference targets from [TSPLIB](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html).
“Post-prefix” is the best newly produced candidate after readout and polishing;
“retained” includes the option of keeping the initial tour. Complete time includes
source preparation, process startup and execution through process exit. It is
not pure GPU sampling time.

| Instance | Nodes | Optimum | Prefix = retained | Post-prefix | Gap (%) | Complete (s) |
|---|---:|---:|---:|---:|---:|---:|
| kroA200 | 3200 | 29368 | 29368 | 31084 | 0.0000 | 4.203 |
| pcb442 | 7073 | 50778 | 50785 | 59893 | 0.0138 | 4.624 |
| pr1002 | 16037 | 259045 | 260520 | 340972 | 0.5694 | 6.395 |
| nrw1379 | 22065 | 56638 | 56692 | 76257 | 0.0953 | 7.968 |
| fl1577 | 25270 | 22249 | 23535 | 31069 | 5.7800 | 10.154 |
| u1817 | 29086 | 57201 | 57504 | 82885 | 0.5297 | 11.122 |
| d2103 | 33662 | 80450 | 80655 | 116060 | 0.2548 | 17.047 |

The prefix and retained lengths coincide in all seven rows. The larger
post-prefix lengths show why returning the retained tour alone would hide the
lack of p-bit improvement. Seven evaluated scales do not mean seven p-bit optima.
The maximum single network has **33,662 binary nodes**, no stochastic auxiliary
bits, and **8 replicas reported separately**. Peak Torch allocated memory was
**36.30 MiB** across those replicas; this excludes complete driver/context memory.
No physical p-bit device, chip area or energy use was measured.

## Attribution controls and complete costs

All four methods received the same prefix length for each instance. Compared
by final retained length:

| Control | p-bit better | Equal | p-bit worse |
|---|---:|---:|---:|
| Random bits + identical readout | 0 | 5 | 2 |
| Exact assignment + repair | 0 | 7 | 0 |
| Same-prefix ILS | 0 | 5 | 2 |

![Measured quality and complete cost](../../data/tsp/candidate-network/analysis/quality-and-cost.png)

The p-bit and assignment quality curves coincide. Assignment returned the same
quality faster in every observed pair; this is a single-seed observation, not a
general timing distribution. Random readout and ILS improved the retained tour
on fl1577 and d2103 while the p-bit route did not. The protocol uses fixed work
and upper time limits, not equal consumed computation or equal tuning effort.

The 29 calls, including replay, accumulated **227.794 seconds** of preparation
plus execution to exit; the seven natural p-bit calls accumulated **61.512 seconds**.
Their phase totals were **3.836 seconds sampling**, **12.743 seconds readout**,
**2.175 seconds polishing**, **10.064 seconds prefix** and **8.400 seconds network
setup**. These phases are not the full elapsed cost. Sampling includes state
capture, CPU energy calculations and output; it is not a CUDA-event kernel
measurement. Environment setup, native compilation, independent auditing and
transfer are outside the 227.794-second call total. The declared runner and
controller caps were 1860 and 3600 seconds.

Of **224 natural p-bit annealing-boundary states**, **4** satisfied the degree
constraints. They contained **23, 28, 30 and 65 subtours**, respectively; **none**
was a complete tour. These are correlated boundary observations, not independent
trials. Repairing a state into a tour does not establish raw-state feasibility or
a complete TSP ground state.

## Evidence and reproduction

All 29 calls completed; there were no failed, interrupted or unexecuted entries
in this revised panel. The verifier checked **1,451 tour observations and 992
binary states**, including control/initial states and replay. The origin replay
matched **65 non-timing events and 26 numerical arrays** and adds no trial.
Independent restoration of archived source and outputs re-certified the 200-city
and 2103-city p-bit calls.

The [core calls](../../data/tsp/candidate-network/rtx5090/calls.json) preserve every
declared call, returned tour and captured state; [metrics](../../data/tsp/candidate-network/analysis/metrics.json)
retain per-instance gaps, post-prefix quality and component costs. Coordinates,
native source archives/binaries and private operational records are excluded.
Acquire the exact external inputs specified in the protocol, then run:

```bash
uv run --frozen python scripts/tsp/verify_candidate.py \
  --core data/tsp/candidate-network/rtx5090 \
  --instances-directory data/external/tsp
uv run --frozen python scripts/tsp/summarize_candidate.py \
  --core data/tsp/candidate-network/rtx5090 --output /tmp/tsp-analysis-NEW
```

These commands verify data and reproduce the figure without executing a solver.
Fresh GPU execution uses the p-bit example in the [method guide](candidate-network.md).

## Classical references and limits

Standalone classical algorithms are references. The separate
[u1817 CPU certificate](../../data/tsp/candidate-network/u1817-optimum.json) records
GA-EAX reaching 57,201 on an AMD Ryzen 9 5950X: three hits in four natural calls,
plus one replay. It used population 500 and a 180-second cap after targeted
classical tuning. It is neither a p-bit result nor a same-host timing comparison.
Its missing confirmation hit remains in the packet.

The earlier [150-city RTX 5090 report](experiment-report.md) retains its different
CPU selector/GPU local-search method and original denominator. This new evidence
supports the implemented workflow's scale and reproducibility, but no superiority
over strong TSP heuristics, no general optimum probability and no hardware energy
claim. Used sources and the complete inherited research bibliography are linked
from the [method bibliography](candidate-network.md).
