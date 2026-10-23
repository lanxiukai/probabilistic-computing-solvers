# Arithmetic-Guided Sparse Ising Factorization

English | [简体中文](README.zh-CN.md)

A GPU implementation combining arithmetic-guided proposals with
Metropolis–Hastings sampling of a sparse Ising multiplier. Strict success requires
correct nontrivial integer factors in the actual accepted state and zero FP64
energy in both full and conditioned models.

The accepted final RTX 5090 panel has 52/54 strict successes, including 7/9
at 128 bits. Both event-limited outcomes and all controls are included. GPU simulation is distinct from p-bit device
measurements; the restricted instance family does not establish general
128-bit coverage or a speed advantage over classical factorization.

- [Experiment report and complete references](experiment-report.md)
- [Method](method.md) and [frozen protocol](protocol.md)
- [Reproduction](reproduction.md) and [source guide](source-guide.md)
- [Release status](readiness.md)

The Python package remains `sparse_factorization`; its command is `factor-bench`.
See the [repository guide](../repository-guide.md) for the multi-project layout.
