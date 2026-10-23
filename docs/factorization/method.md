# Method and scope

English | [简体中文](method.zh-CN.md)

This is a GPU software implementation. The multiplier uses compact sparse
storage, carry_chain topology, Aadit gate penalties, and unit COPY penalties.
Only product and structural-zero ports are pinned. Known factors do not enter
initialization, candidate support, or scheduling. Foundations include
[Camsari et al.](experiment-report.md#ref-2017-camsari-stochastic-p-bits-invertible-logic),
[Aadit et al.](experiment-report.md#ref-2022-aadit-sparse-ising-machines), and
[Lucas](experiment-report.md#ref-2014-lucas-ising-formulations-of-many-np-problems).

The energy convention is

$$H(s)=C-\sum_i h_i s_i-\sum_{i<j}J_{ij}s_i s_j,$$

where $s_i=2x_i-1$. Conditioning on the product preserves the exact constant.
An n×n multiplier has $8n^2-5n$ full spins and $8n(n-1)$ free spins; product
bit width is distinct from the physical variable count.

The diagram shows the deferred square-root path used at 112/120/128 product bits.

```mermaid
flowchart LR
    N[Public N and register width] --> C[Arithmetic root candidates]
    C --> P[Positive-weight prescreen]
    P --> R[Exact radius for first survivor]
    R --> F[Low-bit product filter]
    F --> B[Complete random physical body]
    B --> M[MH using actual finite proposal mass]
    M --> V[Accepted-state integer and dual zero-energy certificates]
```

The 64-bit configuration uses filtered quotient proposals; 80-bit uses CRT/Fermat;
96-bit uses four-word CRT; 112/120/128-bit uses deferred square roots. These are
different proposal kernels. A curve combining them is not a complexity curve
for one fixed algorithm. Frozen JSON supplies the complete parameters.

For larger widths, public $N$ defines necessary CRT quadratic-residue conditions.
Deferred square roots prescreen candidates using 24 additional small primes,
compute the exact radius only for the first survivor, then independently check
the low 24 product bits. Each replica uses K=262144 candidates and eight uint32
words per candidate; all words are consumed. Null candidates, an empty surviving
bank, and postfilter rejection retain the complete previous state.

Write a complete state as $y=(y_r,y_b)$, with factor-register roots $y_r$
and circuit body $y_b$. Let $Q_*(r)$ be the actual mass of a non-null root;
its missing mass belongs to the null outcome. The prescreen passes with
$u(r)=1$ or $2^{-27}$; the independent product postfilter passes with
$v(r)=1$ or $2^{-24}$. Define
$Z_{pre}=\sum_r Q_*(r)u(r)$ and $S_K(Z)=\sum_{j=0}^{K-1}(1-Z)^j$.
The body distribution $h_\beta(y_b\mid y_r)$ uses the implemented integer-CDF
probabilities. Only the first prescreen survivor is postfiltered; a postfilter
rejection does not try the next candidate. The off-diagonal full proposal is

$$q_K(y\mid x)=S_K(Z_{pre})Q_*(y_r)u(y_r)v(y_r)h_\beta(y_b\mid y_r).$$

The state-independent $S_K$ cancels in the forward/reverse MH ratio. Overlapping
branches contribute summed marginal mass. The body uses the actual integer-CDF
mass. Full support, balance, and small-model correctness do not establish rapid
mixing in large models or a classical speed advantage.

The configuration uses beta=12 without PT. Root proposals are independent of the
physical state, so a matched-random-stream selected-root integer control is
reported separately. It omits body/MH while preserving every noise request; it
measures the selected candidate's contribution, not a full-bank scan or a tuned
classical factorizer. YAFU provides a separate end-to-end CPU reference.

The large-width implementation requires public $N\ge3(2^n-1)^2/4$, where n is
factor register width. The evaluation family is narrower:
$N/2^{2n}\in[0.75,0.875)$, two distinct n-bit primes, and minimum relative gap 0.1.
It is not an arbitrary-integer interface or all balanced semiprimes. Generation
strata and rejected draws are retained in private provenance.

## Method references

| Topic | Sources |
|---|---|
| Acceptance, reverse support, and diagonal mass | [Metropolis](experiment-report.md#ref-1953-metropolis), [Hastings](experiment-report.md#ref-1970-hastings), [Tierney](experiment-report.md#ref-1998-tierney-metropolis-hastings-kernels) |
| Informed discrete proposals | [Zanella](experiment-report.md#ref-2020-zanella-locally-balanced-proposals), [Grathwohl et al.](experiment-report.md#ref-2021-grathwohl-gwg) |
| Conditional trees, constructive bodies, and lookahead | [Hamze/de Freitas](experiment-report.md#ref-2004-hamze-trees), [Siepmann/Frenkel](experiment-report.md#ref-1992-siepmann-frenkel), [Naesseth et al.](experiment-report.md#ref-2014-naesseth-smc) |
| Difference of squares, CRT, and modular sieves | [Erra/Grenier](experiment-report.md#ref-2009-erra-fermat-revisited), [Blake](experiment-report.md#ref-2023-blake-fermat-machine-learning), [Hittmeir](experiment-report.md#ref-2022-hittmeir-factorization-subset-sum) |
| Coloring and scheduling | [Gonzalez et al.](experiment-report.md#ref-2011-gonzalez-parallel-gibbs-sampling), [Brélaz](experiment-report.md#ref-1979-brelaz) |
| Temperature-exchange background | [Hukushima/Nemoto](experiment-report.md#ref-1996-hukushima-nemoto), [Katzgraber et al.](experiment-report.md#ref-2006-katzgraber-feedback-optimized-parallel-tempering) |
| Classical controls and complexity | [YAFU](experiment-report.md#ref-project-bbuhrow-yafu), [Pollard](experiment-report.md#ref-1975-pollard-rho-factorization), [Brent](experiment-report.md#ref-1980-brent-monte-carlo-factorization), [Stevenhagen](experiment-report.md#ref-2008-stevenhagen-number-field-sieve) |

These references support principles or comparisons; the concrete finite-precision
kernel, update order, and certificates are defined by this method and its source.
The [report bibliography](experiment-report.md#references) also retains excluded
directions, code comparisons, and abstract-only sources. CRT and difference of
squares have classical precedents; the research questions concern the specific
combination, complete cost, strict state certification, and reuse across targets.
