# LDPC maximum-likelihood codeword search

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). This is classical binary linear
LDPC decoding in error coordinates. With hard decision $h$, codeword $c$, and
$e=c\oplus h$, minimize $\sum_i q_i e_i$ subject to $He=Hh$ over GF(2).
The receiver receives only $H$, the syndrome, and
$q_i=\operatorname{round}(256\min(|LLR_i|,32))$.
Signed LLRs, hard decisions, and transmitted truth stay outside the solver.
Quantized likelihood and original continuous likelihood are distinct objectives.

Source: `src/combinatorial_solvers/ldpc/`. Native XOR factors and colored Gibbs update
the original bits. The degree-six quadratic encoding adds two slack bits per
check: $\Gamma(\sum e_i-2u-4v-\sigma)^2$.
The large native panel has 32766 data p-bits and no auxiliaries.
The quadratic panel has 16380 data plus 16380 auxiliary p-bits, total 32760.
The separate WiFi-648 panel uses 648 original bits.
Replicas, cache, update arithmetic, and classical BP state are counted separately.

Finite penalty weights do not certify a globally equivalent ML optimization.
Only actually feasible physical states compete on the original likelihood.
Joint slack updates and a parity-penalty ramp are explicit variants;
BP warm starts must report their own cost and whether the final state improved
on the initial decoder output. Include same-GPU min-sum/sum-product and external
BP/OSD/LSD controls. OSD ranking must use full log-odds, not a one-sided surrogate.

Verify every syndrome, auxiliary residual, original cost, and reconstructed
codeword independently. Report feasibility, ML evidence, FER/BER, and runtime
separately: even an exact ML word can differ from the transmitted word.
Use the declared $E_s/N_0$ convention; do not infer $E_b/N_0$ from an unverified
design rate. All-zero transmission requires the tested gauge symmetry.
Eight frames per large code and repeated sampler seeds cannot establish rare
error rates or large-code ML optimality. Retain the earlier OSD timeout as a
limitation, and bound OSD to its declared small-code scope.

Run `ldpc` via the [common instructions](../reproduction.md). The example creates
a synthetic code and saves only receiver input. See [external baselines](../external-baselines.md)
for author-library and WiFi matrix provenance.

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [Error-Rate Reduction in LDPC Decoding via Bit-Aligned Temporal Reinforcement in Parallel Probabilistic-Bit Dynamics](https://arxiv.org/html/2609.12389v1)
- R2: [github.com/nonizawa/pSA_LDPC/blob/99bde4cd173ce90bdde098873925589fb9d4a05d/README.md](https://github.com/nonizawa/pSA_LDPC/blob/99bde4cd173ce90bdde098873925589fb9d4a05d/README.md)
- R3: [quantumgizmos/ldpc](https://github.com/quantumgizmos/ldpc)
- R4: [Localized statistics decoding for quantum low-density parity-check codes](https://arxiv.org/abs/2406.18655)
- R5: [Tavildar LDPC](https://github.com/tavildar/LDPC)
