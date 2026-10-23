# Pairwise covering arrays

[简体中文](README.zh-CN.md)

GPU-simulated p-bit block heat bath searches for a covering array at a fixed number of rows. The model covers all symbol pairs between every pair of columns, uses a homogeneous alphabet and has no forbidden combinations. Its objective is the original number of missing interactions; zero certifies the declared row count.

The frozen confirmation evaluates five dimensions, three fresh seeds and 128 replicas per native call. It includes 15 heat-bath calls, 30 matched classical-control calls and five independent ACTS/IPOG constructions. Three fixed-work replays add no independent trials. A separate development panel retains 10 natural calls and one replay; it is not fresh-seed confirmation. The largest evaluated model has 31,500 persistent one-hot binary entries per replica. Evaluation size alone does not establish a successful array or a benefit from thermal updates.

- [Model, update rule and attribution](method.md)
- [Frozen protocol, results and limitations](report.md)
- [Source map and reproduction](source-guide.md)

The implementation simulates constrained p-bit states on an RTX 5090. It does not demonstrate a physical p-bit device, a hardware speed advantage, or a current best-known row count.

All 15 heat-bath and 15 plateau confirmation calls construct valid arrays, through 31,500 encoded bits. Plateau has more successful replicas at the 30-row node; no independent thermal-rule advantage is established.
