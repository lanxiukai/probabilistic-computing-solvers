# Graph and list coloring with conditional p-bit blocks

English | [简体中文](experiment-report.zh-CN.md)

## Problem and relevance

For an undirected simple graph, a fixed palette of $K$ colors and nonempty allowed lists $L_v$, minimize the original number of conflicting edges,

$$H(c)=\sum_{\{u,v\}\in E}[c_u=c_v],\qquad c_v\in L_v.$$

Zero is a certified ground state of this fixed-palette objective. A proper $K$-coloring establishes an upper bound on the chromatic number. Failure to find a $(K-1)$-coloring does not establish a matching lower bound. When a separately verified $K$-clique accompanies a proper $K$-coloring, the two certificates establish the exact chromatic number.

Coloring represents conflict-based resource assignment and is important in discrete mathematics and computing. The selected `fpsol2` and `inithx` DIMACS families originate in register allocation for real code; the `le450` family contains constructed Leighton graphs with guaranteed palette targets. These experiments address the published graph subproblems, rather than performance of a complete compiler or scheduling system. [R1–R2]

## Representation and p-bit stage

Each GPU chain maintains an explicit one-hot binary array $x_{v,a}=[c_v=a]$ with $|V|K$ entries. The color labels and neighbor-color counts are caches, and independent checks compare those caches with the saved bits and the original edges. Forbidden list entries are fixed to zero. The objective is quadratic on this constrained binary domain,

$$H(x)=\sum_{\{u,v\}\in E}\sum_{a=0}^{K-1}x_{u,a}x_{v,a}.$$

The implementation preserves one-hot and list constraints directly. It does not instantiate a penalty coupling for every one-hot constraint. A conditional p-bit block is drawn from all legal colors of one selected vertex, or all legal color combinations of several selected vertices, with weights proportional to $\exp(-\beta H)$. This is a GPU simulation of a constrained binary heat bath.

Single-vertex and adjacent-pair blocks support arbitrary palettes. The additional three/four-vertex route is restricted to $K\le8$ and enumerates $K^B$ assignments. Its vertex walk and duplicate-vertex fallback depend only on the input graph and independent random draws. Each internal edge contributes once. The implementation uses integer local costs and FP64 conditional weights. At fixed temperature, an input-dependent mixture of these conditional updates preserves the corresponding constrained Boltzmann measure; a cyclic optimization schedule does not claim an equilibrium output distribution.

Each chain starts uniformly within its declared color lists. No reference coloring is read into the initial state. Every accepted transition can update the best-state cache, while host observations occur at chunk boundaries. A deadline-crossing chunk is retained in the raw record and excluded from the deadline-eligible quality result.

## Classical support and comparisons

A structural clique can remove global color-label symmetry in standard coloring. The clique is found from the graph alone, checked against every required original edge, and assigned distinct fixed colors. Every proper coloring admits a global renaming satisfying this constraint. The reduction preserves the existence of a zero-conflict solution; it need not preserve the minimum positive conflict count of an uncolorable graph. Restricted lists do not admit this global symmetry in general, so this preprocessing is disabled for list-coloring inputs. Clique construction is classical support, with its time, pinned bits and initial objective reported separately. An initial zero-conflict state earns no new p-bit hit. This is a standard reduction also discussed in modern exact-coloring work. [R5]

The optional exchange-Monte-Carlo route uses fixed inverse-temperature ladders. Adjacent labels exchange with acceptance

$$\min\{1,\exp[(\beta_i-\beta_j)(H_i-H_j)]\}.$$

Temperature labels move while binary states stay in their persistent chains. This is equivalent to swapping states between fixed-temperature slots. Chains within a ladder are coupled; the independent experimental unit is the ladder, not each temperature slot. The exchange step is classical support for the conditional p-bit updates. [R6–R7]

Matched Metropolis and random-tie minimum-energy descent (`plateau`) use the same binary representation and permitted blocks. An independent, single-threaded C++ TabuCol-style implementation supplies a broader classical reference, using conflict vertices, incremental counts, randomized tabu tenure and aspiration. It is an owned implementation inspired by the TabuCol method, not a reproduction of upstream HEAD. [R3–R4] Equal search seconds on one GPU with many chains and one CPU thread do not imply matched resources or a speed advantage.

The PT and plateau comparisons evaluate complete routes: plateau does not use temperature exchange. A difference between them cannot by itself isolate the contribution of the conditional p-bit kernel from the exchange support.

## Timing, scale and evidence scope

Preparation, bounded search and complete runner time are separate measurements. Search includes in-search checkpoint and curve-writing overhead; complete runner time also includes final certification and output writing. First-zero values are observations at eligible chunk boundaries, not exact within-chunk hit times. GPU allocator peaks exclude the full CUDA context and driver memory. Device and software versions are bound to each measurement record.

The original `fpsol2.i.1` has 496 vertices, including 227 isolated vertices. At 65 colors it materializes 32,240 bits per chain, of which 17,485 connect to original conflict factors. Fixing its independently verified 65-clique leaves 28,015 free allowed coordinates, including 13,260 attached to original conflict factors. The original `inithx.i.1` has 864 vertices, 18,707 edges and 345 isolated vertices: its 54-color representation materializes 46,656 bits, with 28,026 attached to original conflict factors. Fixing its verified 54-clique leaves 43,740 free coordinates, including 25,110 attached to those factors. These are audited input/model counts, not evidence that a particular search succeeded.

Connection to conflict factors refers to adjacency in the original graph, rather than the number of remaining free–free couplings after clique fixing. The visited mask marks all allowed one-hot coordinates of each vertex considered by an update; a marked coordinate need not have flipped. The separate `changed` counter counts changed joint block assignments, not distinct flipped bits. None of these model sizes alone establishes successful solution of a fully interacting 32K-bit instance.

The public data are separated by purpose; each directory has its own `calls.json`, configurations and witnesses:

| Ledger under `data/graph-coloring/` | Purpose | Natural calls | Work replays | Status |
|---|---|---:|---:|---|
| `rtx5090` | Frozen new-seed and held-out G3 confirmation | 56 | 4 | Complete original and public-core verification passed |
| `development-g2b` | G2B route, population and classical-control comparison | 13 | 2 | Accepted development evidence |
| `implementation-g2a` | G2A fixed-work implementation comparison | 8 implementation calls | 0 separate replays | Accepted equivalence pilot |

The five G2A equivalence pairs are comparisons among its eight calls; they do not add trials. G2B is development evidence used before the G3 freeze. Its calls and replays do not enlarge the final confirmation denominator. The [source and reproduction guide](source-guide.md) gives separate verification commands for the three ledgers.

## G3 independent confirmation

The frozen confirmation uses new seeds 2026102351 and 2026102352 and 128 chains per native call: 56 natural calls (18 heat bath, 18 plateau, 20 single-thread Tabu) plus four replays that add no trials. No call failed or remained unexecuted. Development graphs are le450_5a, le450_25c and fpsol2.i.1; le450_5b, le450_25d, fpsol2.i.2, inithx.i.1 and the independent list fixture are held out. Policies were fixed before seeing outcomes. The second seed reverses method/input order; all outcomes remain reported.

Leighton inputs use unfixed PT: four-vertex blocks and inverse temperatures 0.4–12 for five colors, two-vertex blocks and 1–16 for 25 colors, each with eight groups of 16 temperatures and a 60 s search limit. Register graphs use one-vertex blocks at inverse temperature 16: 20 s with a fixed clique, and 60 s for the extra unfixed fpsol2.i.1 calls. The list fixture uses four-vertex blocks, eight groups of 16 temperatures, 0.4–12 and 30 s. Chunks contain 512 steps; early stopping is allowed after every group has an eligible zero. Plateau has no exchanges and treats each chain as a group. Chains or temperature groups do not enlarge the two-seed confirmation denominator.

| Graph / route | Heat-bath eligible conflicts, two seeds | Heat-bath zero hits, each /128 | Plateau conflicts | Single-thread Tabu conflicts |
|---|---|---|---|---|
| `le450_5a` / unfixed | 0/0 | 29/22 | 20/22 | 0/0 |
| `le450_5b` / unfixed | 0/0 | 15/11 | 17/20 | 0/0 |
| `le450_25c` / unfixed | 7/6 | 0/0 | 10/9 | 3/2 |
| `le450_25d` / unfixed | 6/7 | 0/0 | 8/10 | 2/3 |
| `fpsol2.i.1` / clique | 0/0 | 128/128 | 0/0 | 0/0 |
| `fpsol2.i.2` / clique | 0/0 | 7/5 | 0/0 | 6/8 |
| `fpsol2.i.1` / unfixed | 0/0 | 7/8 | 0/0 | 14/13 |
| `inithx.i.1` / clique | 0/0 | 128/128 | 0/0 | 10/11 |
| `list4-256` / lists | 0/0 | 62/69 | 0/0 | 0/0 |

Heat bath reaches zero in 14/18 calls. None of the 18 initial ensembles contains a zero, and all 18 calls improve their initial best conflict count. Paired best-conflict outcomes are 8 wins, 10 ties and 0 losses against plateau, and 6 wins, 8 ties and 4 losses against matching-route single-thread Tabu. Extra unfixed inithx.i.1 Tabu calls leave 31/30 conflicts and are outside that matched denominator. Heat bath improves on this plateau route for five-color graphs, while CPU Tabu reaches zero faster; the 25-color heat baths retain 6–7 conflicts versus 2–3 for Tabu.

Unfixed fpsol2.i.1 heat bath has 7/8 zero-hit chains versus 35/31 for plateau; with clique fixing both reach 128/128. On fpsol2.i.2, both have 7/5 hits; on inithx.i.1, both reach 128/128. The list heat bath solves 8/8 PT groups in each call with 62/69 solved chains, versus 126/127 chains for plateau. Different group-stopping conditions prevent short search time from establishing a kernel speed advantage. These are scoped capability and route comparisons, not a universal thermal-update advantage.

The three clique-fixed register graphs have matching clique certificates, so zero-conflict witnesses close their chromatic lower bounds. A separate [five-clique proof](../../data/graph-coloring/proofs/le450_5b.json) certifies le450_5b after search; it was not a solver input, and raw runtime lower-bound fields remain null. The [25-clique proof](../../data/graph-coloring/proofs/le450_25d.json) is only a lower bound because no zero-conflict upper witness was found here. The list planting witness is certification-only, never a search input.

### Confirmation cost and resources

Complete runner seconds are means of two natural calls; memory is the maximum heat-bath allocated/reserved MiB. These include preparation, search, certification and result writing, excluding outer process startup. One-time clique preparation is also outside runner time and has a separate retained receipt.

| Graph / route | Heat bath s | Plateau s | Tabu s | Heat-bath A/R MiB |
|---|---:|---:|---:|---:|
| `le450_5a` / unfixed | 44.3641 | 64.0854 | 1.0388 | 8.897/24.000 |
| `le450_5b` / unfixed | 56.9493 | 66.1349 | 1.7531 | 8.897/24.000 |
| `le450_25c` / unfixed | 69.4157 | 66.2143 | 61.6163 | 21.585/36.000 |
| `le450_25d` / unfixed | 67.0700 | 65.5723 | 61.6007 | 21.585/36.000 |
| `fpsol2.i.1` / clique | 4.5261 | 4.3146 | 0.5731 | 52.261/56.000 |
| `fpsol2.i.2` / clique | 24.1061 | 23.7513 | 20.7354 | 25.423/38.000 |
| `fpsol2.i.1` / unfixed | 68.3794 | 65.7511 | 60.5114 | 52.261/56.000 |
| `inithx.i.1` / clique | 7.1736 | 7.2311 | 21.2619 | 76.399/92.000 |
| `list4-256` / lists | 3.3272 | 35.7104 | 0.4741 | 7.414/24.000 |

The measured environment is RTX 5090, AMD EPYC 9354 single-thread CPU reference, Python 3.14.6, PyTorch 2.13.0+cu130, Triton 3.7.1, NumPy 2.4.4 and CUDA 13.0; see the [environment record](../../data/graph-coloring/rtx5090/environment.json). Complete confirmation evidence contains 298,050 checked coloring states. The [full ledger](../../data/graph-coloring/rtx5090/calls.json) retains preparation/search clocks, initialization, participation, deadline overshoot and memory per call. The largest successful storage model has 46,656 bits but only 28,026 attach to original conflict factors; this is not a fully interacting 32K-bit success.

## Audited G2B resource and control comparison

G2B retained the accepted numerical implementation and used development seed `2026102342`. The 13 natural calls contain six p-bit configurations, three matched plateau controls and four independent CPU references; two additional work replays add no quality trials. Chunks contain 512 updates, with temperature exchanges after each chunk. Unfixed calls have 60-second search caps; the clique `fpsol2.i.1` calls have 20-second caps. The native route may exit after every independent group has an eligible zero.

| Graph / route | Chains | Eligible conflicts | Zero-hit independent groups | Complete runner (s) |
|---|---:|---:|---:|---:|
| `le450_5a`, unfixed four-vertex PT | 128 | **0** | 7/8 | 65.2844 |
| `le450_5a`, unfixed four-vertex PT | 512 | **0** | 27/32 | 70.8703 |
| `le450_5a`, unfixed four-vertex plateau | 512 | 13 | 0/512 | 68.4280 |
| `le450_5a`, unfixed CPU Tabu | 1 | **0** | 1/1 | 0.9869 |
| `fpsol2.i.1`, unfixed one-vertex heat bath | 512 | **0** | 12/512 | 74.7858 |
| `fpsol2.i.1`, unfixed one-vertex plateau | 512 | **0** | 34/512 | 74.5654 |
| `fpsol2.i.1`, clique one-vertex heat bath | 512 | **0** | 512/512 | 13.1638 |
| `fpsol2.i.1`, clique one-vertex plateau | 512 | **0** | 512/512 | 12.6914 |
| `fpsol2.i.1`, unfixed CPU Tabu | 1 | 12 | 0/1 | 60.5434 |
| `fpsol2.i.1`, clique CPU Tabu | 1 | 2 | 0/1 | 20.6529 |
| `le450_25c`, unfixed two-vertex PT | 128 | 7 | 0/8 | 67.0946 |
| `le450_25c`, unfixed two-vertex PT | 512 | 8 | 0/32 | 79.8761 |
| `le450_25c`, unfixed CPU Tabu | 1 | **4** | 0/1 | 61.6226 |

On `le450_5a`, 24/128 and 70/512 individual chains reached zero, distributed across the groups shown above. The 512-chain run observed its first zero at 3.1253 seconds, versus 11.6394 seconds with 128 chains; it also uses four times as many chains and has a larger complete cost. The strong unfixed CPU reference observed zero at 0.01457 seconds. These measurements do not establish a p-bit speed advantage over that reference.

On `fpsol2.i.1`, unfixed plateau reached zero in more chains than the heat bath, and both clique-supported routes reached zero in every chain. Clique heat bath and plateau observed their first zero at 0.003602 and 0.003526 seconds, but their complete costs were 13.1638 and 12.6914 seconds. Their short search portions cannot stand in for the complete workflow. The standalone Tabu misses are retained without treating them as evidence of an isolated heat-bath advantage. No independent heat-bath benefit is established. `le450_25c` remains a miss for both PT populations, with worse conflict counts than the CPU reference.

Independent audits of the complete original evidence checked 276,324 saved color assignments, 15 reconstructed initializations, 210 recovery checkpoints and two exact replays containing 31 and 32 non-time arrays. All natural calls appear in the table. The final confirmation fixes 128 chains as a common resource budget, with lower complete costs in these PT comparisons; this is not a claim that 128 is optimal for every metric. The earlier zero and larger absolute count of solved groups at 512 chains remain part of the evidence. The complete frozen confirmation is reported above.

## Audited implementation-equivalence pilot

G2A compared a bounded integer-energy FP64 Boltzmann-weight cache with direct exponentials, and preparation reuse with fresh preparation. The eight calls used seed `2026102341`, 128 chains, chunks of 512 steps and 65,536 steps per chain. Every call completed this work before its 120-second safety cap. The five declared pairs matched all 31 or 32 non-time state arrays exactly; initial-state and initial/final checkpoint arrays also matched. The independent full-state audit checked 10,240 saved color assignments. These implementation comparisons add no independent quality-confirmation trials.

| Cache comparison | Search seconds, off → on | Complete runner seconds, off → on | Conflicts in both calls |
|---|---:|---:|---:|
| `le450_5a`, four-vertex block / temperature exchange | 0.8403 → 0.7029 | 4.6808 → 4.7363 | 7 |
| `le450_25c`, two-vertex block / temperature exchange | 0.5529 → 0.4468 | 5.9349 → 5.7532 | 20 |
| `fpsol2.i.1`, 65-clique / single-vertex fixed temperature | 0.2217 → 0.1392 | 4.4939 → 4.4762 | 0 |

Preparation reuse reduced observed preparation time from 3.0836 to 2.8576 seconds on `le450_5a`, and from 2.7701 to 2.2326 seconds on `fpsol2.i.1`, with the same deterministic states. These observations support using the two options in subsequent development. Each comparison is a single sequential timing observation; the inconsistent complete-run changes do not establish a stable end-to-end speed advantage.

## Evidence boundaries

Original DIMACS graphs and the generated list-coloring illustration are separate input categories. The synthetic illustration has 256 vertices, 2,245 edges, four palette colors and three allowed colors at every vertex. Its 1,024 materialized coordinates contain 768 free allowed coordinates. A supplied planting witness is reserved for certification and is not a solver input. This illustration does not establish general list-coloring hardness.

The release must bind every accepted measurement to its source, configuration, original-input identity, initial state, RNG endpoints and independently verified witness. Full initial/final states, eligible and returned best states, periodic checkpoints, flushed curves and every temperature-exchange draw are retained. Checkpoint and curve overhead inside the search budget is included in its wall time. A recovered checkpoint is evidence; automatic resume is not claimed.

The current refinements address structural symmetry, low-palette block size, temperature exchange, replica utilization and checkpoint overhead. Their frozen new-seed and held-out-instance results, including misses, are reported above. Historical palette references are not a current solver ranking, and software feasibility results are not physical p-bit hardware measurements.

## References

- **R1.** [DIMACS original coloring instances](https://archive.dimacs.rutgers.edu/pub/challenge/graph/benchmarks/color/). Original ASCII data and file-header provenance. Access does not independently establish redistribution permission.
- **R2.** Michael Trick. [Graph Coloring Instances](https://mat.tepper.cmu.edu/COLOR/instances.html). Historical palette targets and register-allocation/Leighton sources; the original vertex count is retained separately from reduced graphs.
- **R3.** A. Hertz and D. de Werra. [Using tabu search techniques for graph coloring](https://doi.org/10.1007/BF02239976). *Computing* 39, 345–351 (1987). Reference method for the independently implemented classical comparison.
- **R4.** L. Moalic and A. Gondran. [Variations on Memetic Algorithms for Graph Coloring Problems](https://arxiv.org/abs/1401.2184v2), revised 2016. HEAD background; its external implementation and published performance have not been reproduced here.
- **R5.** D. Faber, A. Jabrayilov and P. Mutzel. [SAT Encoding of Partial Ordering Models for Graph Coloring Problems](https://doi.org/10.4230/LIPIcs.SAT.2024.12). SAT 2024, 12:1–12:20. Exact-method context, standard clique precoloring and benchmark preprocessing; its SAT experiments are not reproduced here.
- **R6.** K. Hukushima and K. Nemoto. [Exchange Monte Carlo Method and Application to Spin Glass Simulations](https://doi.org/10.1143/JPSJ.65.1604). *JPSJ* 65(6), 1604–1608 (1996); [author preprint](https://arxiv.org/abs/cond-mat/9512035). Exchange-method background, without reproducing the spin-glass experiments.
- **R7.** C. J. Geyer. [Simulated Tempering and Umbrella Sampling](https://www.stat.umn.edu/geyer/mcmc/library/mcmc/html/temper.html). Product-measure and coupled-chain interpretation.
- **Software.** [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/) and [Triton](https://triton-lang.org/). Exact execution versions and dependency identities belong to the frozen lock and measurement records.

Release checks: 742 tests passed in the assembled public tree; two unrelated covering-array ACTS checks passed separately after correcting fixture paths. The first formatting check used a default line width in the isolated directory; formatting under the public project settings then passed. All seven numerical module ASTs match G3, and four complete-work replays passed.
