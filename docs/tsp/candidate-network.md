# Candidate-arc p-bit network for symmetric TSP

English | [简体中文](candidate-network.zh-CN.md)

The subject of this extension is the **TSP workflow with p-bit participation**:
classical initialization, a GPU simulation of binary p-bit heat-bath updates for
a restricted assignment relaxation, classical readout, and tour certification.
Standalone LKH and GA-EAX are reference solvers used to assess this workflow.
Their optimal tours are classical reference results and do not count as p-bit
achievements. The [RTX 5090 measurements](candidate-report.md) cover seven
instances and show no improvement over the prefix. The
[earlier 150-city measurements](experiment-report.md) retain their original CPU
selector and GPU local-search scope.

## Workflow and attribution

| Stage | Implementation | Contribution to report |
|---|---|---|
| Initialization | Classical LKH prefix and candidate-arc construction | Starting tour and its cost |
| p-bit updates | GPU binary heat-bath annealing | Raw bit states, energy, degree feasibility and sampling cost |
| Readout | Classical matching, subtour repair and CUDA integer 2-opt | Complete candidate tours and readout cost |
| Retention and certification | Compare with the prefix and verify integer tour costs | Final quality, improvement over the prefix and complete cost |

The released method is a hybrid workflow. LKH initialization and classical
readout are supporting components; standalone LKH and GA-EAX calls are controls.
Report the final hybrid result separately from the measured contribution of the
p-bit stage. Keeping an already optimal prefix establishes no p-bit improvement.

## Model and physical-node accounting

Keep 16 nearest outgoing neighbors per city and both directions of a tour produced
by an actual LKH prefix. Each directed candidate arc has one binary variable.
With outgoing/incoming selected counts $r_i,q_j$, normalized integer distance $c_e$
and penalty $\lambda$, the model is

$$E(x)=\sum_e c_ex_e+\lambda\sum_i(r_i-1)^2+\lambda\sum_j(q_j-1)^2.$$

The constant is $2n\lambda$, the linear coefficient is $c_e-2\lambda$, and two
arcs sharing a source or destination have a quadratic coefficient $2\lambda$.
There are no stochastic auxiliary variables. One model's node count is the
number of candidate arcs; eight replicas are reported separately. Mapping one
node to one physical p-bit assumes native support for these sparse couplings.
Controller arithmetic, stored weights and routing are additional resources;
fabricated devices, embedding overhead and energy consumption are not measured.

Degree-one constraints still allow subtours. The candidate graph need not contain
the full-graph optimum, and the chosen finite penalty is not a proof of feasibility.
This model is not a complete TSP Ising encoding.

## GPU updates and classical readout

The binary conditional uses

$$\Delta_e E=c_e+2\lambda(r_i+q_j-2x_e-1),\qquad
P(x_e\leftarrow1)=\sigma(-\beta\Delta_e E).$$

Arcs of one color share neither endpoint. A persistent Triton block owns one
replica and uses barriers between colors. Degree counts are int32; fields and
probabilities are float32. A sweep-specific Philox seed and replica/arc counter
identify each draw. Annealing is an optimization procedure, not a claim of
equilibrium TSP samples.

Readout solves a sparse classical matching problem with state-biased costs,
splices subtours, and applies CUDA integer 2-opt. Repair may leave the candidate
graph. Raw state feasibility, readout quality and the best retained prefix are
separate observations. Matching, repair, polishing, certification and output are
all charged. Controls use original-cost matching, random bits with the same
readout, same-prefix ILS, standalone LKH and standalone GA-EAX.

## Reproduction inputs and tools

The [frozen protocol](../../data/tsp/candidate-network/protocol.json) lists all
instances, raw-input hashes, reference optima, seeds and parameters. Obtain
TSPLIB files from their listed upstream URLs and verify the uncompressed bytes.
No benchmark coordinates, native source archives or native executables are
redistributed here. LKH and GA-EAX reserve their authors' rights and are supplied
for research use; the project's MIT license does not relicense them.

The project-authored build recipe accepts explicitly obtained pinned archives.
It preserves originals and logs the compiler and output identities. Omit
`--eax-archive` for the p-bit workflow; supply it only for a standalone classical
reference build. GA-EAX uses
its existing explicit seed function through `TSP_EAX_SEED`; its search and
termination rules remain unchanged. Rebuild on the target CPU and record that
build before measuring.

```bash
uv run --frozen python scripts/tsp/prepare_native.py \
  --lkh-archive downloads/LKH-3.0.13.tgz --output /tmp/tsp-native-NEW
uv run --frozen python scripts/tsp/run_candidate.py \
  --plan data/tsp/candidate-network/protocol.json \
  --case pbit-5090-pcb442-pbit-42001 \
  --instances-directory data/external/tsp \
  --baseline-directory /tmp/tsp-native-NEW --output /tmp/tsp-case-NEW
```

The second command creates a fresh isolated run. It does not load a published
best tour or reuse a completed output directory. For the accepted core bundle,
`scripts/tsp/verify_candidate.py --core PATH --instances-directory PATH` recomputes
integer tour certificates and state energies without running a solver. The measured core is
`data/tsp/candidate-network/rtx5090`. Complete private operational archives are
not public core data.

## Classical reference certificate

The following certificate documents the strength of a classical control. It is
not a result of the p-bit workflow and is not its main reproduction target.
A [CPU-generated u1817 certificate packet](../../data/tsp/candidate-network/u1817-optimum.json)
contains the computed 57,201-length tour and all four natural outcomes plus one
replay, including a confirmation miss. It provides a feasible tour matching the
published optimal length, not a newly computed lower-bound proof. The packet is
never loaded as a solver starting tour and contains no benchmark coordinates.

```bash
uv run --frozen python scripts/tsp/verify_candidate.py \
  --certificate data/tsp/candidate-network/u1817-optimum.json \
  --instances-directory data/external/tsp
```

## References

- [GA-EAX 1.0 author code, pinned commit](https://github.com/nagata-yuichi/GA-EAX/tree/015dfbe9f267230f78787bd244af393ffc018900), implementing Nagata and Kobayashi's 2013 algorithm.
- [LKH-3 author distribution](http://webhotel4.ruc.dk/~keld/research/LKH-3/), pinned version 3.0.13.
- [TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and [proven optimal lengths](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html).
- [SciPy sparse minimum-weight full bipartite matching](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html).
- [Triton](https://triton-lang.org/main/index.html) and [PyTorch](https://pytorch.org/docs/stable/index.html), GPU execution software.
- [NumPy](https://numpy.org/doc/stable/) and [Matplotlib](https://matplotlib.org/stable/), array verification and reproducible figures.
- [Existing TSP source inventory](README.md), including Ising encodings, constrained tempering, strong classical methods and hardware directions that were not replicated by this extension.

Citation does not imply that an alternative method or a hardware claim was reproduced.
