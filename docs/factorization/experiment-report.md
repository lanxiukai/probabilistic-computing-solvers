# Arithmetic-Guided Sparse Ising Factorization: Experiment Report

English | [简体中文](experiment-report.zh-CN.md)

Accepted benchmark, 2026-10-19. **52 of 54 frozen runs reached a certified
zero-energy physical state on one RTX 5090.** The 128-bit panel reached 7/9;
its other two runs exhausted their event limits. All 45 runs at 64, 80, 96,
112, and 120 bits succeeded. These are 18 restricted targets with three seeds
each, not 54 different instances or a general factoring benchmark.

## 1. Problem and method

Given public product $N$ and two $n$-bit factor registers, the solver searches
the complete sparse multiplier state. It uses compact `carry_chain` storage,
nonnegative gate/COPY penalties, and

$$H(s)=C-\sum_i h_i s_i-\sum_{i<j}J_{ij}s_i s_j,\qquad s_i=2x_i-1.$$

Only product and structural-zero ports are pinned. Every run starts with
independent physical Rademacher spins; known factors are used only for
subsequent certification. The model has $8n^2-5n$ full and $8n(n-1)$ free
spins. At 128 product bits this is **32,448 full / 32,256 free spins**.
Foundations include [invertible p-bit logic](#ref-2017-camsari-stochastic-p-bits-invertible-logic)
and [COPY-sparsified multipliers](#ref-2022-aadit-sparse-ising-machines).

Root candidates use public arithmetic: quotient guidance at 64 bits,
CRT/Fermat at 80 bits, four-word CRT at 96 bits, and deferred square roots
at 112/120/128 bits. The last proposal samples K=262144 candidates per replica,
uses positive quadratic-residue screening weights, computes an exact radius
for the first survivor, and performs an independent low-product-bit screen.
It then generates a complete random circuit body. The MH step uses the
actual root marginal mass and quantized integer-CDF body mass:

$$\alpha(x,y)=\min\{1,e^{-\beta[H(y)-H(x)]}q(x\mid y)/q(y\mid x)\}.$$

The finite-bank factor cancels only because its support/weights are
state-independent. Uniform-tail overlap is added to the biased mass;
failed screening keeps the entire old state. Accepted states, not temporary
DAG assignments, are checked for hits. See the [method](method.md) for the
complete probability convention and [Hastings](#ref-1970-hastings).

The choice of MH enables coordinated factor/COPY/carry changes and corrects
non-symmetric arithmetic proposals. It is not a claim that MH universally
beats Gibbs: exact Gibbs updates are an acceptance-one MH special case
([Chib and Greenberg](#ref-1995-chib-greenberg)). Sequential local gate draws
are not exact draws from the complete multiplier's conditional distribution.
This final panel has no matched 128-bit Gibbs baseline. The frozen configuration
uses beta=12, no local Gibbs sweeps, and no parallel tempering. Correctness and
full support do not establish rapid equilibrium mixing.

The accepted execution profile is `buffer_sqrt_qr_v1`: reusable random buffers,
integer square-root kernels, and fused residue checks. Its numerical/runtime
source matches the measured source file-for-file; packaging and reporting
changes are identified in [provenance](../../provenance/factorization/release-v3.json).
No intermediate optimization measurements are pooled into this report.

## 2. Equipment, frozen panel, and measurement boundaries

| Equipment | Recorded setting |
|---|---|
| GPU | One NVIDIA GeForce RTX 5090, 32607 MiB, recorded power limit 500 W |
| Driver / CUDA | 595.91.07 / PyTorch CUDA 13.0 build |
| CPU | AMD Ryzen 9 7945HX with Radeon Graphics |
| Host memory | MemTotal 98056316 kB; process cgroup limit 96392445952 bytes |
| CPU allocation | GPU worker affinity: 30 logical CPUs; YAFU: one pinned CPU thread |
| Software | Python 3.14.6; PyTorch 2.13.0+cu130; Triton 3.7.1; NumPy 2.4.4; SciPy 1.18.0 |

Raw device/software fields accompany every physical result. The
[locked environment](../../uv.lock) and [frozen public protocol](protocol.md)
provide the reproduction inputs. GPU experiments and origin replays ran
serially; a CPU baseline could overlap on its separately assigned core.
Completed evidence was transferred and verified while later computation ran.

For each width, three public-$N$ strata cover
$N/2^{2n}\in[0.75,0.875)$. Factors are distinct $n$-bit primes with relative
gap at least 0.1. Parameters never use factor answers. This is a restricted
balanced-semiprime family, not arbitrary integers. The panel was frozen before
final execution. Target-0 fixtures were subsequently used for compatibility
and cap calibration after execution-profile selection; they are not presented
as wholly unseen performance tests. Repeated measurements of the same
target/role-seed tuple are not pooled as additional trials.

Strict success requires an actual accepted state with a correct nontrivial
integer product, legal spins, and **zero full and conditioned FP64 energy**,
plus transition verification. Six designated runs also have completed full
origin replays. The 132 completed tasks comprise 54 physical runs, 18 paired
selected-root integer controls, 54 single-thread YAFU controls, and six replays;
controls and replays add no independent physical trials.

Sampling first passage is the recorded observation upper bound. Setup includes
model preparation and warmup; certification and chunk capture are reported
separately. Cell wall time includes the run's setup, sampling, evidence output,
and certification, but excludes cold installation, later transfer/audits, and
the separate origin replay. Capture overlaps sampling and must not be added
again. YAFU wall time is the timed factorizer process; its versioned build is
excluded. Raw per-run fields remain available.

## 3. Final results and figures

All nine frozen runs per width appear below. First-passage medians/ranges are
**conditional on success**; cell medians include the event-limited runs.
Small sample sizes and restricted instances do not support TTS99 estimates.
Different widths use different proposal kernels and populations, so this is
not a scaling law for one fixed algorithm.

| Bits | Strict successes | Successful first passage median (s) | Successful range (s) | All-run cell median (s) | Peak reserved GiB |
|---|---|---|---|---|---|
| 64 | 9/9 | 1.333271 | 0.249215–3.070311 | 21.134627 | 3.713 |
| 80 | 9/9 | 0.044377 | 0.014388–2.807065 | 18.692184 | 2.855 |
| 96 | 9/9 | 5.997098 | 1.100490–12.796780 | 20.488252 | 1.943 |
| 112 | 9/9 | 5.662224 | 1.260776–15.552503 | 11.456806 | 2.074 |
| 120 | 9/9 | 128.035249 | 23.915363–1067.315636 | 134.730049 | 2.080 |
| 128 | 7/9 | 948.686157 | 381.258808–2307.776375 | 2070.297489 | 2.086 |

![Final RTX 5090 timing and censoring](../../data/factorization/release-v3/summary/runtime.png)

Circles are strict successes; crosses show event-limited observations. The
complete [CSV](../../data/factorization/release-v3/summary/results.csv) and
[aggregates](../../data/factorization/release-v3/summary/aggregates.json) retain
all denominators. Peak reserved VRAM is the allocator measurement, not the
model's theoretical minimum memory or a device-energy measurement.

### Complete 128-bit panel

| Run | Status | Events | First passage (s) | Sampling (s) | Cell (s) |
|---|---|---|---|---|---|
| w128-holdout-0-s0 | strict_hit | 812155 | 2307.776375 | 2307.784397 | 2315.276440 |
| w128-holdout-0-s1 | strict_hit | 725809 | 2062.702358 | 2062.722354 | 2070.297489 |
| w128-holdout-0-s2 | event_limit | 1048576 | — | 2981.062412 | 2985.194211 |
| w128-holdout-1-s0 | strict_hit | 795245 | 2261.345760 | 2261.352639 | 2269.507280 |
| w128-holdout-1-s1 | strict_hit | 201167 | 571.597949 | 571.617555 | 579.183858 |
| w128-holdout-1-s2 | strict_hit | 134080 | 381.258808 | 381.272809 | 388.884611 |
| w128-holdout-2-s0 | event_limit | 1048576 | — | 2942.713694 | 2947.193040 |
| w128-holdout-2-s1 | strict_hit | 337761 | 948.686157 | 948.695869 | 956.241805 |
| w128-holdout-2-s2 | strict_hit | 299182 | 840.289897 | 840.309406 | 847.852709 |

The fastest recorded 128-bit strict hit was **381.258808 sampling seconds**;
its complete cell took **388.884611 seconds**. This is one successful run,
not an expected solve time. The two unsuccessful runs each exhausted
1,048,576 events; they remain in the denominator. An infrastructure replacement
restarted one identical tuple from its original seed and matched the saved
prefix. The logical run is counted once; this is not a new-seed retry.

### Arithmetic attribution and classical controls

| Bits | Integer-control successes | Integer-control cell median (s) | YAFU successes | YAFU process median (s) |
|---|---|---|---|---|
| 64 | 3/3 | 1.100020 | 9/9 | 0.033192 |
| 80 | 3/3 | 0.809065 | 9/9 | 0.065921 |
| 96 | 3/3 | 2.696634 | 9/9 | 0.116040 |
| 112 | 3/3 | 5.565105 | 9/9 | 0.115901 |
| 120 | 3/3 | 32.330447 | 9/9 | 0.165467 |
| 128 | 2/3 | 921.598952 | 9/9 | 0.165715 |

![Separate physical-state and integer controls](../../data/factorization/release-v3/summary/controls.png)

The selected-root arm consumes the same random layout and tests the selected
arithmetic candidate, omitting the circuit body and MH. It is not a scan of
every candidate in the bank or an optimized classical factorizer. Its 17/18
successes include 2/3 at 128 bits. [YAFU](#ref-project-bbuhrow-yafu), revision
`c7d90283972820f99b3b4920bfc1eff05c1f6342`, completed **54/54** integer
factorizations using `-plan noecm -threads 1`; the [baseline lock](../../baselines/factorization/yafu.lock.json)
records the build. The three arms have different success criteria and timing
scopes. YAFU is substantially faster on the integer task; these data establish
no classical factoring speed advantage.

Correct roots originate in the arithmetic proposals, which are independent
of the current physical state. Full-state generation adds work and can lose
integer candidates. A classical solver can also construct a legal multiplier
state after finding factors; having a certificate is not an exclusive capability
of this method. The additional research constraint here is the declared random
initialization, proposal, and accepted-transition path.

## 4. Comparison with representative research

Literature scope was reviewed on 2026-10-18. Device counts, factor-register
success, complete physical states, and intermediate sieve relations are
different outcomes. This table is not a cross-hardware speed or energy ranking.

| Work | Reported scope | Implementation / success | Comparison boundary |
|---|---|---|---|
| [Aadit et al., 2022](#ref-2022-aadit-sparse-ising-machines) | Exact to 32 bits; 50-bit near-ground-state evaluation | FPGA sparse multiplier | Additional arithmetic guidance and GPU simulation here |
| [Jung et al., 2023](#ref-2023-jung-virtual-boltzmann-factorization) | 10–64 bits | FPGA VCBM, annealing and divisibility decisions | Different physical-state representation |
| [Chen et al., 2023](#ref-2023-chen-espin-qfactor) | QFactor to 64 bits | Gradient/noise; scaling measured in MATLAB | Simulation is not a 64-bit chip measurement |
| [Lee et al., 2025](#ref-2025-lee-chaotic-oscillator-pbit) | 32/48/64 bits | Chaotic p-bit devices with FPGA/VCBM | Device randomness differs from GPU pseudorandom simulation |
| [Duffee et al., 2025](#ref-2025-duffee-integrated-circuit-probabilistic-computer) | 6-bit V-MTJ; 24-bit modified ASIC | On-chip pseudorandom source for 24 bits | 40/80-bit synthesis is not measured factorization |
| [250-MTJ system, 2026](#ref-2026-yang-250-mtj-probabilistic-ising-machine) | Exact 24 bits; 444 logical p-bits | MTJ/FPGA; energy and product checks | Device count and logical variables differ |
| [IsingFormer v2](#ref-2025-bunaiyan-isingformer) | 20-bit learned model, 440 instances; separate 16–24-bit proposal study | RTX6000 Ada; correct product, no full-state zero-energy requirement | Training and offline proposal bank; optimization timing excludes bank construction |
| [TNSS, 2026](#ref-2026-tesoro-tensor-network-schnorr-sieving) | Complete factorization to 100 bits | Tensor networks, lattice sieving, classical postprocessing | 130-bit relation/resource studies are not complete 130-bit factorization |
| [Photonic lattice sieve, 2026](#ref-2026-yang-photonic-lattice-sieve) | Abstract: hardware 32-bit, simulation 120-bit complete factorization | Hybrid relation search and lattice sieving | 100-bit relation pairs are intermediate; abstract-only access |
| [Gao et al., ISCAS 2026](#ref-2026-gao-48bit) | Abstract claims 48-bit results | 40 nm CMOS and MTJ design | Full text unavailable; no quantitative ranking from abstract alone |
| [CUDA-MPQS v1, 2026](#ref-2026-januszewski-cuda-mpqs) | Author-reported 512-bit RSA-155 | 64 H100s for sieving; one H100 for linear algebra | Classical quadratic sieve; not independently reproduced here |
| This project | 128-bit complete physical-state certificates; 7/9 runs | One RTX 5090; arithmetic proposals, random body and MH | Restricted family; all failures retained; no classical speed advantage |

Two supported distinctions are useful: **complete sparse-state certification
at 128 bits** on the declared family, and **no neural-network pretraining**
relative to learned-proposal configurations such as IsingFormer. The latter
avoids training data/weights and offline MCMC proposal banks, while arithmetic
tables, compilation, and sampling still cost time. Neither distinction implies
a world record, hardware advantage, superior equilibrium sampling, or better
generalization. The larger classical GPU results in the table also preclude
presenting 128 bits as a general GPU factorization record.

## 5. Reproduction and evidence

The [reproduction guide](reproduction.md) provides CPU certificate checks,
figure regeneration, isolated GPU runs, and serial origin replay. Core data
include every physical outcome/configuration/environment, accepted-state packets,
both control groups, and the six recorded origin-replay receipts. The
[inventory](../../data/factorization/release-v3/inventory.json) binds their bytes.
Full intermediate trajectories and operational logs remain in the private
research archive. CPU core verification checks stored accepted states and
receipts; it does not claim to rerun a GPU trajectory.

The linked bibliography below retains all used papers, projects, websites, and
software, including background and excluded directions. Citations do not imply
replication. Paper PDFs and retrieval audits are not bundled. Code and original
project materials are provided under the [MIT license](../../LICENSE);
external dependencies retain their own licenses.

<a id="references"></a>

## 6. References

Citation metadata reviewed on 2026-10-18: 80 research works, 15 projects, and 21 technical sources. Publication titles retain their original wording.

### Research literature

<a id="ref-1953-metropolis"></a>

- **[1]** N. Metropolis; Arianna W. Rosenbluth; M. N. Rosenbluth; Augusta H. Teller; Edward Teller. Equation of State Calculations by Fast Computing Machines. 1953. [DOI](https://doi.org/10.1063/1.1699114) · [Link 2](https://bayes.wustl.edu/Manual/Metropolis.pdf)

<a id="ref-1970-hastings"></a>

- **[2]** W. K. Hastings. Monte Carlo sampling methods using Markov chains and their applications. 1970. [DOI](https://doi.org/10.1093/biomet/57.1.97)

<a id="ref-1974-lehman"></a>

- **[3]** R. Sherman Lehman. Factoring Large Integers. 1974. Full text was not obtained during this research. [DOI](https://doi.org/10.1090/S0025-5718-1974-0340163-2) · [Link 2](https://www.ams.org/journals/mcom/1974-28-126/S0025-5718-1974-0340163-2/S0025-5718-1974-0340163-2.pdf)

<a id="ref-1975-pollard-rho-factorization"></a>

- **[4]** J. Pollard. A Monte Carlo method for factorization. 1975. [DOI](https://doi.org/10.1007/BF01933667) · [Link 2](https://pages.cs.wisc.edu/~cs812-1/pollardrho.pdf)

<a id="ref-1979-brelaz"></a>

- **[5]** Daniel Brélaz. New methods to color the vertices of a graph. 1979. [DOI](https://doi.org/10.1145/359094.359101) · [Link 2](https://cse.unl.edu/~choueiry/Documents/brelaz.pdf)

<a id="ref-1980-brent-monte-carlo-factorization"></a>

- **[6]** R. Brent. An improved Monte Carlo factorization algorithm. 1980. [DOI](https://doi.org/10.1007/BF01933190) · [Link 2](https://maths-people.anu.edu.au/~brent/pd/rpb051i.pdf)

<a id="ref-1983-kirkpatrick-optimization-by-simulated-annealing"></a>

- **[7]** S. Kirkpatrick; C. Gelatt; M. Vecchi. Optimization by Simulated Annealing. 1983. [DOI](https://doi.org/10.1126/science.220.4598.671)

<a id="ref-1988-hajek-cooling-schedules-optimal-annealing"></a>

- **[8]** B. Hajek. Cooling Schedules for Optimal Annealing. 1988. [DOI](https://doi.org/10.1287/moor.13.2.311) · [Link 2](https://web.mit.edu/6.435/www/Hajek88.pdf)

<a id="ref-1992-siepmann-frenkel"></a>

- **[9]** Jörn Ilja Siepmann; Daan Frenkel. Configurational bias Monte Carlo: a new sampling scheme for flexible chains. 1992. [DOI](https://doi.org/10.1080/00268979200100061) · [Link 2](https://siepmann.chem.umn.edu/publications/3)

<a id="ref-1995-eldershaw-brent-parallel-factorization"></a>

- **[10]** Craig Eldershaw; Richard P. Brent. Integer Factorisation on the AP1000. 1995. Revised author manuscript, 1995. [Link 1](https://maths-people.anu.edu.au/brent/pd/rpb156.pdf) · [Link 2](https://maths-people.anu.edu.au/brent/pub/pub156.html)

<a id="ref-1996-hukushima-nemoto"></a>

- **[11]** Koji Hukushima; Koji Nemoto. Exchange Monte Carlo Method and Application to Spin Glass Simulations. 1996. [DOI](https://doi.org/10.1143/JPSJ.65.1604)

<a id="ref-1998-tierney-metropolis-hastings-kernels"></a>

- **[12]** L. Tierney. A Note on Metropolis-Hastings Kernels for General State Spaces. 1998. [DOI](https://doi.org/10.1214/aoap/1027961031) · [Link 2](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TransdimMCMC/TierneyAAP1998.pdf)

<a id="ref-1999-consta-recoil"></a>

- **[13]** Styliani Consta; N. B. Wilding; Daan Frenkel; Z. Alexandrowicz. Recoil growth: An efficient simulation method for multi-polymer systems. 1999. 1998 preprint; J. Chem. Phys. 110, 3220–3228 (1999). [DOI](https://doi.org/10.1063/1.477844) · [Link 2](https://arxiv.org/abs/cond-mat/9811071) · [Link 3](https://arxiv.org/pdf/cond-mat/9811071)

<a id="ref-2001-houdayer-cluster-monte-carlo-spin-glass"></a>

- **[14]** J. Houdayer. A Cluster Monte Carlo Algorithm for 2-Dimensional Spin Glasses. 2001. arXiv:cond-mat/0101116. [DOI](https://doi.org/10.1007/PL00011151) · [Link 2](https://arxiv.org/pdf/cond-mat/0101116)

<a id="ref-2002-burges-factoring-as-optimization"></a>

- **[15]** Christopher J. C. Burges. Factoring as Optimization. 2002. Microsoft Research Technical Report MSR-TR-2002-83. [Link 1](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2002-83.pdf)

<a id="ref-2004-hamze-trees"></a>

- **[16]** Firas Hamze; Nando de Freitas. From Fields to Trees. 2004. UAI 2004; uploaded to arXiv in 2012. [Link 1](https://arxiv.org/abs/1207.4149)

<a id="ref-2006-katzgraber-feedback-optimized-parallel-tempering"></a>

- **[17]** Helmut G. Katzgraber; Simon Trebst; David A. Huse; Matthias Troyer. Feedback-optimized parallel tempering Monte Carlo. 2006. arXiv:cond-mat/0602085. [DOI](https://doi.org/10.1088/1742-5468/2006/03/P03018) · [Link 2](https://arxiv.org/pdf/cond-mat/0602085) · [Link 3](https://arxiv.org/html/cond-mat/0602085v3)

<a id="ref-2007-roberts-adaptive-mcmc"></a>

- **[18]** G. Roberts; J. Rosenthal. Coupling and Ergodicity of Adaptive Markov Chain Monte Carlo Algorithms. 2007. [DOI](https://doi.org/10.1239/jap/1183667414) · [Link 2](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/AdaptiveMC/RobRosCouplingErgod2007JAP.pdf)

<a id="ref-2008-stevenhagen-number-field-sieve"></a>

- **[19]** Peter Stevenhagen. The number field sieve. 2008. MSRI Publications 44 chapter; distinct from the 1990 work with the same title. [Link 1](https://pub.math.leidenuniv.nl/~stevenhagenp/ANTproc/04psh.pdf)

<a id="ref-2009-bernstein-ecm-graphics-cards"></a>

- **[20]** D. Bernstein; Tien-Ren Chen; Chen-Mou Cheng; T. Lange; Bo-Yin Yang. ECM on Graphics Cards. 2009. [DOI](https://doi.org/10.1007/978-3-642-01001-9_28) · [Link 2](https://eecm.cr.yp.to/gpuecm-20090127.pdf)

<a id="ref-2009-erra-fermat-revisited"></a>

- **[21]** Robert Erra; Christophe Grenier. The Fermat factorization method revisited. 2009. Author manuscript, 2009-06-30; separate IACR 2009/318 revision, 2009-07-24. Versions differ in complexity statements; only the square-difference identity and close-factor limits are used. [Link 1](https://hsbp.org/tiki-download_wiki_attachment.php?attId=174) · [Link 2](https://eprint.iacr.org/2009/318)

<a id="ref-2009-metzner-transition-path-theory-markov-jump-processes"></a>

- **[22]** Philipp Metzner; C. Schütte; E. Vanden-Eijnden. Transition Path Theory for Markov Jump Processes. 2009. [DOI](https://doi.org/10.1137/070699500) · [Link 2](https://publications.imp.fu-berlin.de/43/1/MeScVE09.pdf) · [Link 3](https://publications.imp.fu-berlin.de/43/)

<a id="ref-2009-zimmermann-parallel-fpga-ecm-thesis"></a>

- **[23]** Ralf Zimmermann. Optimized Implementation of the Elliptic Curve Factorization Method on a Highly Parallelized Hardware Cluster. 2009. Thesis, November 2009; distinct from the 2010 conference paper. [Link 1](https://informatik.rub.de/wp-content/uploads/2021/11/ralf_zimmermann_diplomarbeit.pdf)

<a id="ref-2010-zimmermann-reconfigurable-factoring"></a>

- **[24]** Ralf Zimmermann; Tim Güneysu; Christof Paar. High-Performance Integer Factoring with Reconfigurable Devices. 2010. Author one-page abstract only. [DOI](https://doi.org/10.1109/FPL.2010.26) · [Link 2](https://labh-curien.univ-st-etienne.fr/cryptarchi/workshop11/abstracts/zimmermann.pdf)

<a id="ref-2011-gonzalez-parallel-gibbs-sampling"></a>

- **[25]** Joseph E. Gonzalez; Yucheng Low; Arthur Gretton; Carlos Guestrin. Parallel Gibbs Sampling: From Colored Fields to Thin Junction Trees. 2011. AISTATS 2011, PMLR 15:324–332. [Link 1](https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf)

<a id="ref-2011-salmon-philox"></a>

- **[26]** John K. Salmon; Mark A. Moraes; Ron O. Dror; David Elliot Shaw. Parallel Random Numbers: As Easy as 1, 2, 3. 2011. [DOI](https://doi.org/10.1145/2063384.2063405) · [Link 2](https://www.deshawresearch.com/resources_random123.html)

<a id="ref-2013-pascanu-gradient-clipping"></a>

- **[27]** Razvan Pascanu; Tomas Mikolov; Yoshua Bengio. On the Difficulty of Training Recurrent Neural Networks. 2013. ICML 2013, PMLR 28. Stability analogy only, not a Gibbs update rule. [Link 1](https://proceedings.mlr.press/v28/pascanu13.pdf)

<a id="ref-2014-decelle-bp"></a>

- **[28]** Aurélien Decelle; Florent Krząkała. Belief-propagation-guided Monte-Carlo sampling. 2014. [DOI](https://doi.org/10.1103/PhysRevB.89.214421) · [Link 2](https://www.lri.fr/~adecelle/content/sc_articles/prb_89_214421.pdf)

<a id="ref-2014-lucas-ising-formulations-of-many-np-problems"></a>

- **[29]** Andrew Lucas. Ising formulations of many NP problems. 2014. [DOI](https://doi.org/10.3389/fphy.2014.00005)

<a id="ref-2014-mansinghka-venture"></a>

- **[30]** Vikash Mansinghka; Daniel Selsam; Yura Perov. Venture: a higher-order probabilistic programming platform with programmable inference. 2014. [Link 1](https://arxiv.org/pdf/1404.0099)

<a id="ref-2014-miele-gpu-cofactorization"></a>

- **[31]** Andrea Miele; Joppe W. Bos; T. Kleinjung; A. Lenstra. Cofactorization on Graphics Processing Units. 2014. [DOI](https://doi.org/10.1007/978-3-662-44709-3_19) · [Link 2](https://eprint.iacr.org/2014/397.pdf) · [Link 3](https://eprint.iacr.org/2014/397)

<a id="ref-2014-naesseth-smc"></a>

- **[32]** Christian A. Naesseth; Fredrik Lindsten; Thomas B. Schön. Sequential Monte Carlo for Graphical Models. 2014. [Link 1](https://proceedings.neurips.cc/paper/5570-sequential-monte-carlo-for-graphical-models.pdf)

<a id="ref-2015-ioffe-batch-normalization"></a>

- **[33]** Sergey Ioffe; Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 2015. arXiv:1502.03167. [DOI](https://doi.org/10.48550/arXiv.1502.03167) · [Link 2](https://arxiv.org/pdf/1502.03167)

<a id="ref-2015-isakov-siman"></a>

- **[34]** Sergei V. Isakov; Ilia N. Zintchenko; Troels F. Rønnow; Matthias Troyer. Optimized simulated annealing for Ising spin glasses. 2015. Computer Physics Communications 192, 265–271 (2015); SimAn v1.0 ancillary source. [Link 1](https://arxiv.org/abs/1401.1084) · [Link 2](https://arxiv.org/abs/1401.1084v4)

<a id="ref-2015-zhu-isoenergetic-cluster-moves-spin-glass"></a>

- **[35]** Zheng Zhu; Andrew J. Ochoa; Helmut G. Katzgraber. Efficient Cluster Algorithm for Spin Glasses in Any Space Dimension. 2015. arXiv:1501.05630. [DOI](https://doi.org/10.1103/PhysRevLett.115.077201) · [Link 2](https://arxiv.org/pdf/1501.05630)

<a id="ref-2016-ba-layer-normalization"></a>

- **[36]** Jimmy Lei Ba; Jamie Kiros; Geoffrey E. Hinton. Layer Normalization. 2016. arXiv:1607.06450. [DOI](https://doi.org/10.48550/arXiv.1607.06450) · [Link 2](https://arxiv.org/pdf/1607.06450)

<a id="ref-2017-camsari-stochastic-p-bits-invertible-logic"></a>

- **[37]** Kerem Y Çamsarı; Rafatul Faria; Brian M. Sutton; S. Datta. Stochastic p-Bits for Invertible Logic. 2017. [DOI](https://doi.org/10.1103/PhysRevX.7.031014)

<a id="ref-2018-jiang-quantum-annealing-prime-factorization"></a>

- **[38]** Shuxian Jiang; Keith A. Britt; A. McCaskey; T. Humble; S. Kais. Quantum Annealing for Prime Factorization. 2018. [DOI](https://doi.org/10.1038/s41598-018-36058-z) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-018-36058-z/MediaObjects/41598_2018_36058_MOESM1_ESM.pdf)

<a id="ref-2018-miyato-spectral-normalization"></a>

- **[39]** Takeru Miyato; Toshiki Kataoka; Masanori Koyama; Yuichi Yoshida. Spectral Normalization for Generative Adversarial Networks. 2018. arXiv:1802.05957. [DOI](https://doi.org/10.48550/arXiv.1802.05957) · [Link 2](https://arxiv.org/pdf/1802.05957)

<a id="ref-2018-titsias-auxiliary-gradient-sampling"></a>

- **[40]** Michalis K. Titsias; Omiros Papaspiliopoulos. Auxiliary gradient-based sampling algorithms. 2018. arXiv:1610.09641v3, 2018-01-25. Auxiliary conditional MH and marginal MH are distinguished. [DOI](https://doi.org/10.48550/arXiv.1610.09641) · [Link 2](https://arxiv.org/pdf/1610.09641v3)

<a id="ref-2019-borders-integer-factorization-stochastic-mtjs"></a>

- **[41]** W. A. Borders; A. Pervaiz; S. Fukami; Kerem Y Çamsarı; H. Ohno; S. Datta. Integer factorization using stochastic magnetic tunnel junctions. 2019. [DOI](https://doi.org/10.1038/s41586-019-1557-9)

<a id="ref-2019-camsari-sign-problem-free-hamiltonians-pbits"></a>

- **[42]** Kerem Y Çamsarı; Shuvro Chowdhury; S. Datta. Scalable Emulation of Sign-Problem-Free Hamiltonians with Room-Temperature p-bits. 2019. [DOI](https://doi.org/10.1103/PhysRevApplied.12.034061)

<a id="ref-2020-neklyudov-imcmc"></a>

- **[43]** Kirill Neklyudov; Max Welling; Evgenii Egorov; Dmitry Vetrov. Involutive MCMC: a Unifying Framework. 2020. [Link 1](https://proceedings.mlr.press/v119/neklyudov20a.html) · [Link 2](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf)

<a id="ref-2020-wang-prime-factorization-parameter-optimized-ising"></a>

- **[44]** Bao-Nan Wang; Feng Hu; Haonan Yao; Chao Wang. Prime factorization algorithm based on parameter optimization of Ising model. 2020. [DOI](https://doi.org/10.1038/s41598-020-62802-5) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-020-62802-5/MediaObjects/41598_2020_62802_MOESM1_ESM.pdf)

<a id="ref-2020-wloka-ecm-on-gpus"></a>

- **[45]** Jonas Wloka; Jan Richter-Brockmann; C. Stahlke; T. Kleinjung; C. Priplata; T. Güneysu. Revisiting ECM on GPUs. 2020. [DOI](https://doi.org/10.1007/978-3-030-65411-5_15) · [Link 2](https://codeload.github.com/Chair-for-Security-Engineering/ecmongpu/tar.gz/b81f7ff5385c640f729453842286943503d6a18a) · [Link 3](https://eprint.iacr.org/2020/1265.pdf) · [Link 4](https://eprint.iacr.org/2020/1265)

<a id="ref-2020-zanella-locally-balanced-proposals"></a>

- **[46]** Giacomo Zanella. Informed Proposals for Local MCMC in Discrete Spaces. 2020. arXiv:1711.07424. [DOI](https://doi.org/10.1080/01621459.2019.1585255) · [Link 2](https://arxiv.org/pdf/1711.07424)

<a id="ref-2021-grathwohl-gwg"></a>

- **[47]** Will Grathwohl; Kevin Swersky; Milad Hashemi; David Duvenaud; Chris Maddison. Oops I Took A Gradient: Scalable Sampling for Discrete Distributions. 2021. [Link 1](https://proceedings.mlr.press/v139/grathwohl21a.html) · [Link 2](https://proceedings.mlr.press/v139/grathwohl21a/grathwohl21a.pdf)

<a id="ref-2021-leleu-chaotic-amplitude-control"></a>

- **[48]** T. Leleu; F. Khoyratee; T. Levi; R. Hamerly; T. Kohno; Kazuyuki Aihara. Scaling Advantage of Chaotic Amplitude Control for High-Performance Combinatorial Optimization. 2021. [DOI](https://doi.org/10.1038/s42005-021-00768-0) · [Link 2](https://www.nature.com/articles/s42005-021-00768-0.pdf)

<a id="ref-2021-onizawa-design-framework-invertible-logic"></a>

- **[49]** N. Onizawa; Kaito Nishino; Sean C. Smithson; Brett H. Meyer; W. Gross; Hitoshi Yamagata; Hiroyuki Fujita; T. Hanyu. A Design Framework for Invertible Logic. 2021. [DOI](https://doi.org/10.1109/TCAD.2020.3003906)

<a id="ref-2022-aadit-sparse-ising-machines"></a>

- **[50]** Navid Anjum Aadit; A. Grimaldi; M. Carpentieri; L. Theogarajan; J. Martinis; G. Finocchio; Kerem Y Çamsarı. Massively parallel probabilistic computing with sparse Ising machines. 2022. [DOI](https://doi.org/10.1038/s41928-022-00774-2) · [Link 2](https://www.nature.com/articles/s41928-022-00774-2)

<a id="ref-2022-hittmeir-factorization-subset-sum"></a>

- **[51]** Markus Hittmeir. Integer factorization as subset-sum problem. 2022. arXiv:2205.10074v1, 2022-05-20; DOI links the associated journal version. CRT/modular-sieve discussion in sections 2–4; the complete subset-sum algorithm was not replicated. [DOI](https://doi.org/10.1016/j.jnt.2023.02.010) · [Link 2](https://arxiv.org/pdf/2205.10074v1) · [Link 3](https://arxiv.org/abs/2205.10074)

<a id="ref-2022-inui-amplitude-homogeneity"></a>

- **[52]** Yoshitaka Inui; Mastiyage Don Sudeera Hasaranga Gunathilaka; Satoshi Kako; T. Aonishi; Yoshihisa Yamamoto. Control of Amplitude Homogeneity in Coherent Ising Machines with Artificial Zeeman Terms. 2022. [DOI](https://doi.org/10.1038/s42005-022-00927-x) · [Link 2](https://www.nature.com/articles/s42005-022-00927-x.pdf)

<a id="ref-2022-patel-logically-synthesized-rbms"></a>

- **[53]** Saavan Patel; Philip Canoza; Sayeef Salahuddin. Logically synthesized and hardware-accelerated restricted Boltzmann machines for combinatorial optimization and integer factorization. 2022. [DOI](https://doi.org/10.1038/s41928-022-00714-0)

<a id="ref-2023-blake-fermat-machine-learning"></a>

- **[54]** Samuel T. Blake. Integer Factorisation, Fermat & Machine Learning on a Classical Computer. 2023. arXiv:2308.12290v1. [DOI](https://doi.org/10.48550/arXiv.2308.12290) · [Link 2](https://arxiv.org/pdf/2308.12290v1)

<a id="ref-2023-chen-espin-qfactor"></a>

- **[55]** Wenhan Chen; Haodi Tang; Yu Wang; Xianwu Hu; Yu-Ming Lin; Tai Min; Yufeng Xie. E-Spin: A Stochastic Ising Spin Based on Electrically-Controlled MTJ for Constructing Large-Scale Ising Annealing Systems. 2023. [DOI](https://doi.org/10.3390/mi14020258) · [Link 2](https://mdpi-res.com/d_attachment/micromachines/micromachines-14-00258/article_deploy/micromachines-14-00258.pdf)

<a id="ref-2023-he-many-body-invertible-logic"></a>

- **[56]** Yi-Han He; Chao Fang; Sheng Luo; Gengchiau Liang. Many-Body Effects-Based Invertible Logic With a Simple Energy Landscape and High Accuracy. 2023. [DOI](https://doi.org/10.1109/JXCDC.2023.3320230)

<a id="ref-2023-jung-virtual-boltzmann-factorization"></a>

- **[57]** Hyundo Jung; Hyunjin Kim; Woojin Lee; Jinwoog Jeon; Yohan Choi; Taehyeong Park; Chulwoo Kim. A quantum-inspired probabilistic prime factorization based on virtually connected Boltzmann machine and probabilistic annealing. 2023. [DOI](https://doi.org/10.1038/s41598-023-43054-5) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-023-43054-5/MediaObjects/41598_2023_43054_MOESM1_ESM.pdf) · [Link 3](https://www.nature.com/articles/s41598-023-43054-5)

<a id="ref-2023-khattar-schnorr-factorization-comment"></a>

- **[58]** Tanuj Khattar; Noureldin Yosri. A comment on "Factoring integers with sublinear resources on a superconducting quantum processor". 2023. arXiv:2307.09651v1. [DOI](https://doi.org/10.48550/arXiv.2307.09651) · [Link 2](https://arxiv.org/pdf/2307.09651v1) · [Link 3](https://arxiv.org/abs/2307.09651v1)

<a id="ref-2023-shao-voltage-controlled-mtj-factorization"></a>

- **[59]** Yixin Shao; Christian Duffee; E. Raimondo; Noraica Davila; V. Lopez-Dominguez; J. Katine; G. Finocchio; P. Khalili. Probabilistic computing with voltage-controlled dynamics in magnetic tunnel junctions. 2023. Main text read; Supplementary Notes 1–3 unavailable. [DOI](https://doi.org/10.1088/1361-6528/acf6c7)

<a id="ref-2024-fleiser-cuda-metal-factorization-thesis"></a>

- **[60]** Jan-Jakub Fleišer. Parallel factorization on GPU using CUDA and Metal APIs. 2024. CTU Prague thesis, 2024. Thesis read; accompanying CD source/data unavailable. [Link 1](https://dspace.cvut.cz/bitstreams/604d1a07-a17c-4202-8296-d50a4bbd48ce/download) · [Link 2](https://dspace.cvut.cz/entities/publication/30f57e6a-dbc3-425e-b1af-4965902de291)

<a id="ref-2024-hales-hiary"></a>

- **[61]** Jonathon Hales; Ghaith A. Hiary. A generalization of Lehman’s method. 2024. [DOI](https://doi.org/10.1007/s11139-024-00959-7) · [Link 2](https://link.springer.com/article/10.1007/s11139-024-00959-7)

<a id="ref-2024-he-direct-ground-state-probabilistic-logic"></a>

- **[62]** Yi-Han He; Sheng Luo; Chao Fang; Gengchiau Liang. Direct design of ground-state probabilistic logic using many-body interactions for probabilistic computing. 2024. [DOI](https://doi.org/10.1038/s41598-024-65676-z) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-024-65676-z/MediaObjects/41598_2024_65676_MOESM1_ESM.docx)

<a id="ref-2025-bunaiyan-isingformer"></a>

- **[63]** Saleh Bunaiyan; Corentin Delacour; Shuvro Chowdhury; Kyle Lee; Abdelrahman S. Abdelrahman; K. Çamsari. IsingFormer: Augmenting Parallel Tempering With Learned Proposals. 2025. arXiv:2509.23043v2, revised 2026-09-11. 24-page preprint with supplementary sections; distinct from v1 and device measurements. [DOI](https://doi.org/10.48550/arXiv.2509.23043) · [Link 2](https://arxiv.org/pdf/2509.23043v2) · [Link 3](https://arxiv.org/abs/2509.23043v2) · [Link 4](https://arxiv.org/html/2509.23043v2)

<a id="ref-2025-chowdhury-boundary"></a>

- **[64]** Shuvro Chowdhury; Navid Anjum Aadit; Andrea Grimaldi; +12 more. Pushing the boundary of quantum advantage in hard combinatorial optimization with probabilistic computers. 2025. [DOI](https://doi.org/10.1038/s41467-025-64235-y) · [Link 2](https://www.nature.com/articles/s41467-025-64235-y)

<a id="ref-2025-duffee-integrated-circuit-probabilistic-computer"></a>

- **[65]** Christian Duffee; J. Athas; Yixin Shao; Noraica Davila Melendez; E. Raimondo; J. Katine; Kerem Y Çamsarı; G. Finocchio; Pedram Khalili Amiri. An integrated-circuit-based probabilistic computer that uses voltage-controlled magnetic tunnel junctions as its entropy source. 2025. [DOI](https://doi.org/10.1038/s41928-025-01439-6)

<a id="ref-2025-efe-circuit-sat"></a>

- **[66]** Ahmet Efe; Abhimanyu Kumar; Nafisa Sadaf Prova; Hüsrev Cılasun; Ziqing Zeng; Tahmida Islam; Ruihong Yin; Chaohui Li; Peter Kreye; Chris Kim; Sachin S. Sapatnekar; Ulya R. Karpuzcu. On the Limits of Ising Machines for Circuit-Derived SAT. 2025. Historical unversioned arXiv entry; v2 was available on 2026-10-18. [Link 1](https://arxiv.org/pdf/2511.21046)

<a id="ref-2025-lee-chaotic-oscillator-pbit"></a>

- **[67]** Woojin Lee; Hyunjin Kim; Hyundo Jung; Yohan Choi; Jinwoog Jeon; Chulwoo Kim. Correlation free large-scale probabilistic computing using a true-random chaotic oscillator p-bit. 2025. [DOI](https://doi.org/10.1038/s41598-025-93218-8) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-025-93218-8/MediaObjects/41598_2025_93218_MOESM1_ESM.docx) · [Link 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-025-93218-8/MediaObjects/41598_2025_93218_MOESM2_ESM.docx) · [Link 4](https://www.nature.com/articles/s41598-025-93218-8.pdf)

<a id="ref-2025-lee-noise-augmented-chaotic-ising"></a>

- **[68]** Kyle Lee; Shuvro Chowdhury; Kerem Y Çamsarı. Noise-augmented chaotic Ising machines for combinatorial optimization and sampling. 2025. [DOI](https://doi.org/10.1038/s42005-025-01945-1) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42005-025-01945-1/MediaObjects/42005_2025_1945_MOESM1_ESM.pdf) · [Link 3](https://www.nature.com/articles/s42005-025-01945-1.pdf)

<a id="ref-2025-onizawa-gpu-pbit-simulated-annealing"></a>

- **[69]** N. Onizawa; T. Hanyu. GPU-accelerated simulated annealing based on p-bits with real-world device-variability modeling. 2025. [DOI](https://doi.org/10.1038/s41598-025-90520-3) · [Link 2](https://www.nature.com/articles/s41598-025-90520-3.pdf)

<a id="ref-2025-qian-van"></a>

- **[70]** Moxian Qian; Shiyang Chen. Variational Autoregressive Networks Applied to φ⁴ Field Theory Systems. 2025. arXiv:2512.19575v1; citation uses the read version, before later title changes. [Link 1](https://arxiv.org/abs/2512.19575v1) · [Link 2](https://arxiv.org/html/2512.19575v1)

<a id="ref-2025-sajeeb-scalable-connectivity-ising-machines"></a>

- **[71]** M. H. Sajeeb; Navid Anjum Aadit; Shuvro Chowdhury; Tong Wu; Cesely Smith; Dhruv Chinmay; Atharva Raut; K. Çamsari; Corentin Delacour; T. Srimani. Scalable Connectivity for Ising Machines: Dense to Sparse. 2025. arXiv:2503.01177v3, 2025-06-02; Phys. Rev. Applied 24, 014005 (2025). [DOI](https://doi.org/10.1103/kx8m-5h3h) · [Link 2](https://arxiv.org/pdf/2503.01177v3) · [Link 3](https://arxiv.org/abs/2503.01177v3) · [Link 4](https://journals.aps.org/prapplied/abstract/10.1103/kx8m-5h3h)

<a id="ref-2025-xu-spike-adaptation"></a>

- **[72]** Haijie Xu; Cen Wang; Yajun Zhang; +2 more. Probabilistic computing accelerated by spike-frequency adaptation: Applications in integer factorization. 2025. Abstract and topic checked; algorithm not implemented or replicated here. [DOI](https://doi.org/10.1103/dvxz-w9ts) · [Link 2](https://journals.aps.org/prapplied/abstract/10.1103/dvxz-w9ts)

<a id="ref-2026-del-bono-global-annealing"></a>

- **[73]** Luca Maria Del Bono; F. Ricci-Tersenghi; F. Zamponi. Demonstrating Real Advantage of Machine-Learning-Enhanced Monte Carlo for Combinatorial Optimization. 2026. arXiv:2510.19544v2, 2026-04-19; PNAS 123, e2534768123 (2026). [DOI](https://doi.org/10.1073/pnas.2534768123) · [Link 2](https://arxiv.org/pdf/2510.19544v2) · [Link 3](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/README.md) · [Link 4](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/data_loads.py) · [Link 5](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/geometry.py) · [Link 6](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/global_steps.py) · [Link 7](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/made.py) · [Link 8](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/utilities.py) · [Link 9](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/global_annealing.py) · [Link 10](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/monte_carlo.py) · [Link 11](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/population_annealing.py) · [Link 12](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/simulated_annealing.py) · [Link 13](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/README.md) · [Link 14](https://arxiv.org/abs/2510.19544v2)

<a id="ref-2026-gao-48bit"></a>

- **[74]** Kun-Peng Gao; Jiadong Chen; Yi-Fan Wang; Shao-Hao Wang; Tai Min; Yu-Feng Xie. A 0.562 mm² MTJ-Based Ising Machine for 48-bit Integer Factorization in 40 nm CMOS. 2026. Official conference abstract only; full text not purchased. [DOI](https://doi.org/10.1109/ISCAS66217.2026.11562081) · [Link 2](https://epapers2.org/iscas2026/ESR/paper_details.php?paper_id=2503) · [Link 3](https://ieeexplore.ieee.org/document/11562081) · [Link 4](https://openalex.org/W7165127850) · [Link 5](https://www.researchgate.net/publication/407285877_A_0562_mm_2_MTJ-Based_Ising_Machine_for_48-bit_Integer_Factorization_in_40_nm_CMOS)

<a id="ref-2026-januszewski-cuda-mpqs"></a>

- **[75]** Fabian Januszewski; Christoph Heinrichs. CUDA-MPQS: A GPU-Resident Self-Initializing Quadratic Sieve, and the Factorization of RSA-155. 2026. arXiv:2610.07126v1, 2026-10-05. Preprint and pinned source; author performance claims not independently replicated here. [DOI](https://doi.org/10.48550/arXiv.2610.07126) · [Link 2](https://arxiv.org/pdf/2610.07126v1) · [Link 3](https://codeload.github.com/drjanosch42/cuda-mpqs/tar.gz/99c61967585b97a71b28fbaf19690af83ac88c17) · [Link 4](https://arxiv.org/abs/2610.07126v1)

<a id="ref-2026-luo-general-ising-factorization"></a>

- **[76]** Zhang Luo; Yifan Yuan; Zhicheng Wang; Ming-Che Lai; P. Lu. General integer factorization algorithm based on Ising machine. 2026. EPJ Quantum Technology 13:1 (2026). Encoding and capacity extrapolation are not solved-instance results. [DOI](https://doi.org/10.1140/epjqt/s40507-025-00449-9) · [Link 2](https://inspirehep.net/files/5ac3f3f28b725eae5550ab2c92177a8b) · [Link 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1140%2Fepjqt%2Fs40507-025-00449-9/MediaObjects/40507_2025_449_MOESM1_ESM.pdf)

<a id="ref-2026-tesoro-tensor-network-schnorr-sieving"></a>

- **[77]** Marco Tesoro; Ilaria Siloi; Daniel Jaschke; G. Magnifico; Simone Montangero. Integer Factorization via Tensor Network Schnorr's Sieving. 2026. arXiv:2410.16355v3, 2026-01-26; PRA 113, 032418 (2026). Accepted manuscript read; complete TN/Schnorr pipeline not replicated here. [DOI](https://doi.org/10.1103/d9dl-ctt4) · [Link 2](https://arxiv.org/pdf/2410.16355v3)

<a id="ref-2026-yang-250-mtj-probabilistic-ising-machine"></a>

- **[78]** Shu-Han Yang; A. Grimaldi; You-Wei Bao; E. Raimondo; Jia Si; G. Finocchio; Hyunsoo Yang. 250 magnetic tunnel junctions-based probabilistic Ising machine. 2026. [DOI](https://doi.org/10.1038/s41467-026-72020-8) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-72020-8/MediaObjects/41467_2026_72020_MOESM1_ESM.pdf) · [Link 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-72020-8/MediaObjects/41467_2026_72020_MOESM2_ESM.pdf) · [Link 4](https://www.nature.com/articles/s41467-026-72020-8.pdf) · [Link 5](https://www.nature.com/articles/s41467-026-72020-8)

<a id="ref-2026-yang-photonic-lattice-sieve"></a>

- **[79]** Jin-Min Yang; Wen-Jia Zhang; Junjie Yu; Songsong Li; Zu-Yuan He. Integer factorization via spatial photonic Ising machine using a lattice sieve method. 2026. Optics Express 34(14), 25016–25027 (2026); PMID 42596112. Official PubMed metadata and abstract only. [DOI](https://doi.org/10.1364/OE.591485) · [Link 2](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=42596112&retmode=xml)

<a id="ref-2026-yao-natively-sparse-ising-machine"></a>

- **[80]** Bai-Jian Yao; Daniel Ebler; Xu Shi; Jun-Tao Wang; Fan Zhang; Wei Zhang; Jie Sun. Precision meets speed through an FPGA-based natively sparse Ising machine for combinatorial optimization. 2026. [DOI](https://doi.org/10.1038/s41467-026-75119-0) · [Link 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-75119-0/MediaObjects/41467_2026_75119_MOESM1_ESM.pdf) · [Link 3](https://www.nature.com/articles/s41467-026-75119-0_reference.pdf)

### Projects and implementations

<a id="ref-project-bbuhrow-yafu"></a>

- **[81]** bbuhrow. bbuhrow/yafu. n.d.. c7d90283972820f99b3b4920bfc1eff05c1f6342; build and dependencies pinned by the baseline lock. [Link 1](https://github.com/bbuhrow/yafu) · [Link 2](https://codeload.github.com/bbuhrow/yafu/tar.gz/c7d90283972820f99b3b4920bfc1eff05c1f6342)

<a id="ref-project-chair-for-security-engineering-ecmongpu"></a>

- **[82]** Chair-for-Security-Engineering. Chair-for-Security-Engineering/ecmongpu. n.d.. b81f7ff5385c640f729453842286943503d6a18a. [Link 1](https://github.com/Chair-for-Security-Engineering/ecmongpu) · [Link 2](https://codeload.github.com/Chair-for-Security-Engineering/ecmongpu/tar.gz/b81f7ff5385c640f729453842286943503d6a18a)

<a id="ref-project-drjanosch42-cuda-mpqs"></a>

- **[83]** drjanosch42. drjanosch42/cuda-mpqs. n.d.. 99c61967585b97a71b28fbaf19690af83ac88c17. [Link 1](https://github.com/drjanosch42/cuda-mpqs) · [Link 2](https://codeload.github.com/drjanosch42/cuda-mpqs/tar.gz/99c61967585b97a71b28fbaf19690af83ac88c17)

<a id="ref-project-flintlib-flint"></a>

- **[84]** flintlib. flintlib/flint. n.d.. 12dd4de456cdaa22e052750c9839f45dc7507f3e. [Link 1](https://github.com/flintlib/flint) · [Link 2](https://github.com/flintlib/flint/blob/12dd4de456cdaa22e052750c9839f45dc7507f3e/src/ulong_extras/factor.c)

<a id="ref-project-gomc-wsu-gomc"></a>

- **[85]** GOMC-WSU. GOMC-WSU/GOMC. n.d.. 785c2ceb6dc402b165906928452e4938b86eef04. [Link 1](https://github.com/GOMC-WSU/GOMC) · [Link 2](https://github.com/GOMC-WSU/GOMC/blob/785c2ceb6dc402b165906928452e4938b86eef04/src/moves/Regrowth.h)

<a id="ref-project-hyundojung-vcbm_pa"></a>

- **[86]** HyundoJung. HyundoJung/VCBM_PA. n.d.. Historical access returned 404; source retrieval or replication is not claimed. [Link 1](https://github.com/HyundoJung/VCBM_PA)

<a id="ref-project-kyleaskine-cuda-sieve"></a>

- **[87]** kyleaskine. kyleaskine/cuda-sieve. n.d.. [Link 1](https://github.com/kyleaskine/cuda-sieve) · [Link 2](https://github.com/kyleaskine/cuda-sieve/blob/main/README.md)

<a id="ref-project-laplaxe-mlmc_optimization"></a>

- **[88]** Laplaxe. Laplaxe/MLMC_optimization. n.d.. 394d47ab9d0514b6cb1c7f764113c579ab8f3eb6. [Link 1](https://github.com/Laplaxe/MLMC_optimization) · [Link 2](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/README.md) · [Link 3](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/global_annealing.py) · [Link 4](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/monte_carlo.py) · [Link 5](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/population_annealing.py) · [Link 6](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Modern/optimization/simulated_annealing.py) · [Link 7](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/README.md) · [Link 8](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/data_loads.py) · [Link 9](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/geometry.py) · [Link 10](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/global_steps.py) · [Link 11](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/made.py) · [Link 12](https://raw.githubusercontent.com/Laplaxe/MLMC_optimization/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6/Code/Legacy/packages/utilities.py)

<a id="ref-project-necludov-imcmc"></a>

- **[89]** necludov. necludov/iMCMC. n.d.. a8d14cd37899520e4883356f8175eef1fb64c507. [Link 1](https://github.com/necludov/iMCMC) · [Link 2](https://github.com/necludov/iMCMC/blob/a8d14cd37899520e4883356f8175eef1fb64c507/irr-MALA/core/iMCMC.py)

<a id="ref-project-nonizawa-gpu-psav"></a>

- **[90]** nonizawa. nonizawa/GPU-pSAv. n.d.. 61c8e1cf577c3871c9b089f0fc6e8b53851475db. [Link 1](https://github.com/nonizawa/GPU-pSAv)

<a id="ref-project-nonizawa-psa"></a>

- **[91]** nonizawa. nonizawa/pSA. n.d.. 551845188d79f4b6a8cc2b2c956927b526f7ed3c. [Link 1](https://github.com/nonizawa/pSA)

<a id="ref-project-opuslab-3dspinglasswithpbits"></a>

- **[92]** OPUSLab. OPUSLab/3DSpinGlassWithPbits. n.d.. e91cf095a2a92a8a90d516a59520e3298270f281. [Link 1](https://github.com/OPUSLab/3DSpinGlassWithPbits) · [Link 2](https://github.com/OPUSLab/3DSpinGlassWithPbits/blob/e91cf095a2a92a8a90d516a59520e3298270f281/APT.m)

<a id="ref-project-saavan-logic_rbm"></a>

- **[93]** Saavan. Saavan/Logic_RBM. n.d.. 9954ad4071209c713bd45e0c4e141e6be14f7fe8. [Link 1](https://github.com/Saavan/Logic_RBM) · [Link 2](https://github.com/Saavan/Logic_RBM/blob/9954ad4071209c713bd45e0c4e141e6be14f7fe8/Software/RBM/rbm.py) · [Link 3](https://github.com/Saavan/Logic_RBM/blob/9954ad4071209c713bd45e0c4e141e6be14f7fe8/Software/LogicRBM/rbm.py) · [Link 4](https://github.com/Saavan/Logic_RBM/blob/9954ad4071209c713bd45e0c4e141e6be14f7fe8/Software/RBM/rbm.py#L349)

<a id="ref-project-tensorflow-probability"></a>

- **[94]** tensorflow. tensorflow/probability. n.d.. [Link 1](https://github.com/tensorflow/probability) · [Link 2](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/mcmc/metropolis_hastings.py)

<a id="ref-project-wgrathwohl-gwg_release"></a>

- **[95]** wgrathwohl. wgrathwohl/GWG_release. n.d.. 6441499c388e5e9f1a6b306fcf832593d1811d8f. [Link 1](https://github.com/wgrathwohl/GWG_release) · [Link 2](https://github.com/wgrathwohl/GWG_release/blob/6441499c388e5e9f1a6b306fcf832593d1811d8f/samplers.py)

### Technical documentation and websites

<a id="ref-context7"></a>

- **[96]** Context7. n.d.. Technical conclusions refer to the corresponding official documentation. [Link 1](https://github.com/upstash/context7)

<a id="ref-distribution-indices"></a>

- **[97]** PyPI and PyTorch distribution indices. n.d.. Package versions and hashes are in uv.lock. [Link 1](https://pypi.org/) · [Link 2](https://download.pytorch.org/whl/cu130)

<a id="ref-ecm"></a>

- **[98]** GMP-ECM. n.d.. GMP-ECM 7.0.5; execution follows the baseline configuration. [Link 1](https://gitlab.inria.fr/zimmerma/ecm)

<a id="ref-git"></a>

- **[99]** Git. n.d.. Reproduction also requires source snapshots and dependency locks. [Link 1](https://git-scm.com/docs)

<a id="ref-gmp"></a>

- **[100]** GNU Multiple Precision Arithmetic Library. n.d.. GMP 6.3.0; distribution packages pinned by the baseline lock. [Link 1](https://gmplib.org/)

<a id="ref-metadata-services"></a>

- **[101]** Crossref, OpenAlex, and Semantic Scholar. n.d.. Bibliographic metadata sources, not performance evidence. [Link 1](https://www.crossref.org/) · [Link 2](https://openalex.org/) · [Link 3](https://www.semanticscholar.org/)

<a id="ref-numpy-rng"></a>

- **[102]** NumPy random generator documentation. n.d.. Equal seeds alone do not establish identical trajectories. [Link 1](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html) · [Link 2](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html) · [Link 3](https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html)

<a id="ref-nvidia-cuda"></a>

- **[103]** NVIDIA CUDA Best Practices Guide. n.d.. Citation review date: 2026-10-18. [Link 1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

<a id="ref-openssh"></a>

- **[104]** OpenSSH. n.d.. Public documentation. [Link 1](https://www.openssh.com/manual.html)

<a id="ref-python"></a>

- **[105]** Python. n.d.. Runtime pinned by .python-version and environment records. [Link 1](https://docs.python.org/3/)

<a id="ref-pytorch-cuda"></a>

- **[106]** PyTorch CUDA semantics, CUDA Graphs, and installation index. n.d.. PyTorch 2.13; main documentation is mutable. [Link 1](https://docs.pytorch.org/docs/2.13/notes/cuda.html) · [Link 2](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs) · [Link 3](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) · [Link 4](https://download.pytorch.org/whl/cu130)

<a id="ref-pytorch-hidden-sync"></a>

- **[107]** PyTorch eager execution: hidden host-to-device synchronization. n.d.. Development note, 2026-08-11. [Link 1](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/)

<a id="ref-pytorch-nonzero"></a>

- **[108]** PyTorch torch.nonzero. n.d.. main documentation consulted during research. [Link 1](https://docs.pytorch.org/docs/main/generated/torch.nonzero.html)

<a id="ref-pytorch-profiler"></a>

- **[109]** PyTorch Profiler. n.d.. 2.13. [Link 1](https://docs.pytorch.org/docs/2.13/profiler.html) · [Link 2](https://docs.pytorch.org/docs/2.13/generated/torch.autograd.profiler_util.EventList.html)

<a id="ref-pytorch-randint"></a>

- **[110]** PyTorch torch.randint. n.d.. 2.13. [Link 1](https://docs.pytorch.org/docs/2.13/generated/torch.randint.html)

<a id="ref-rsync"></a>

- **[111]** rsync. n.d.. Completion markers and independent verification used by this project. [Link 1](https://rsync.samba.org/)

<a id="ref-supervisor"></a>

- **[112]** Supervisor. n.d.. Infrastructure reference. [Link 1](https://supervisord.org/)

<a id="ref-triton-docs"></a>

- **[113]** Triton tutorials and language documentation. n.d.. Installed version pinned in uv.lock; main documentation is mutable. [Link 1](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) · [Link 2](https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html) · [Link 3](https://triton-lang.org/main/python-api/generated/triton.language.store.html) · [Link 4](https://triton-lang.org/main/python-api/generated/triton.language.sum.html) · [Link 5](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html) · [Link 6](https://triton-lang.org/main/python-api/generated/triton.language.range.html) · [Link 7](https://triton-lang.org/main/python-api/triton-semantics.html) · [Link 8](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.static_range.html) · [Link 9](https://triton-lang.org/main/_sources/python-api/triton-semantics.rst.txt) · [Link 10](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.where.html)

<a id="ref-ubuntu-packages"></a>

- **[114]** Ubuntu package archive. n.d.. Ubuntu 24.04 amd64; baselines/factorization/yafu.lock.json. [Link 1](https://archive.ubuntu.com/ubuntu/pool/universe/g/gmp-ecm/libecm1_7.0.5%2bds-1build1_amd64.deb) · [Link 2](https://archive.ubuntu.com/ubuntu/pool/universe/g/gmp-ecm/libecm1-dev_7.0.5%2bds-1build1_amd64.deb) · [Link 3](https://archive.ubuntu.com/ubuntu/pool/universe/g/gmp-ecm/libecm1-dev-common_7.0.5%2bds-1build1_all.deb) · [Link 4](https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmp-dev_6.3.0%2bdfsg-2ubuntu6.1_amd64.deb) · [Link 5](https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmp10_6.3.0%2bdfsg-2ubuntu6.1_amd64.deb)

<a id="ref-uv"></a>

- **[115]** uv. n.d.. Runtime version recorded with each result. [Link 1](https://docs.astral.sh/uv/)

<a id="ref-vast-ai"></a>

- **[116]** Vast.ai connection, storage, and instance documentation. n.d.. Public documentation. [Link 1](https://docs.vast.ai/guides/instances/connect/ssh) · [Link 2](https://docs.vast.ai/guides/instances/storage/data-movement) · [Link 3](https://docs.vast.ai/guides/instances/storage) · [Link 4](https://docs.vast.ai/api-reference/hello-world) · [Link 5](https://docs.vast.ai/api-reference/instances/destroy-instance)

### Additional methodological source

<a id="ref-1995-chib-greenberg"></a>

- **[117]** Siddhartha Chib; Edward Greenberg. Understanding the Metropolis–Hastings Algorithm. The American Statistician 49(4), 327–335 (1995). Section 6.2 checked on 2026-10-19 for Gibbs updates and kernel composition. [DOI](https://doi.org/10.1080/00031305.1995.10476177)

<a id="software-references"></a>

### Locked software sources

Runtime software is pinned by [uv.lock](../../uv.lock). These 50 third-party package pages are software-source citations, not additional experiments.

| Package | Version | Source |
|---|---|---|
| contourpy | 1.3.3 | [Project](https://pypi.org/project/contourpy/) |
| cuda-bindings | 13.3.1 | [Project](https://pypi.org/project/cuda-bindings/) |
| cuda-pathfinder | 1.6.0 | [Project](https://pypi.org/project/cuda-pathfinder/) |
| cuda-toolkit | 13.0.3.0 | [Project](https://pypi.org/project/cuda-toolkit/) |
| cycler | 0.12.1 | [Project](https://pypi.org/project/cycler/) |
| filelock | 3.32.3 | [Project](https://pypi.org/project/filelock/) |
| fonttools | 4.63.0 | [Project](https://pypi.org/project/fonttools/) |
| fsspec | 2026.7.0 | [Project](https://pypi.org/project/fsspec/) |
| iniconfig | 2.3.0 | [Project](https://pypi.org/project/iniconfig/) |
| jinja2 | 3.1.6 | [Project](https://pypi.org/project/jinja2/) |
| kiwisolver | 1.5.0 | [Project](https://pypi.org/project/kiwisolver/) |
| markupsafe | 3.0.3 | [Project](https://pypi.org/project/markupsafe/) |
| matplotlib | 3.11.1 | [Project](https://pypi.org/project/matplotlib/) |
| mpmath | 1.3.0 | [Project](https://pypi.org/project/mpmath/) |
| networkx | 3.6.1 | [Project](https://pypi.org/project/networkx/) |
| nodeenv | 1.10.0 | [Project](https://pypi.org/project/nodeenv/) |
| numpy | 2.4.4 | [Project](https://pypi.org/project/numpy/) |
| nvidia-cublas | 13.1.1.3 | [Project](https://pypi.org/project/nvidia-cublas/) |
| nvidia-cuda-cupti | 13.0.85 | [Project](https://pypi.org/project/nvidia-cuda-cupti/) |
| nvidia-cuda-nvrtc | 13.0.88 | [Project](https://pypi.org/project/nvidia-cuda-nvrtc/) |
| nvidia-cuda-runtime | 13.0.96 | [Project](https://pypi.org/project/nvidia-cuda-runtime/) |
| nvidia-cudnn-cu13 | 9.20.0.48 | [Project](https://pypi.org/project/nvidia-cudnn-cu13/) |
| nvidia-cufft | 12.0.0.61 | [Project](https://pypi.org/project/nvidia-cufft/) |
| nvidia-cufile | 1.15.1.6 | [Project](https://pypi.org/project/nvidia-cufile/) |
| nvidia-curand | 10.4.0.35 | [Project](https://pypi.org/project/nvidia-curand/) |
| nvidia-cusolver | 12.0.4.66 | [Project](https://pypi.org/project/nvidia-cusolver/) |
| nvidia-cusparse | 12.6.3.3 | [Project](https://pypi.org/project/nvidia-cusparse/) |
| nvidia-cusparselt-cu13 | 0.8.1 | [Project](https://pypi.org/project/nvidia-cusparselt-cu13/) |
| nvidia-nccl-cu13 | 2.29.7 | [Project](https://pypi.org/project/nvidia-nccl-cu13/) |
| nvidia-nvjitlink | 13.3.33 | [Project](https://pypi.org/project/nvidia-nvjitlink/) |
| nvidia-nvshmem-cu13 | 3.4.5 | [Project](https://pypi.org/project/nvidia-nvshmem-cu13/) |
| nvidia-nvtx | 13.0.85 | [Project](https://pypi.org/project/nvidia-nvtx/) |
| packaging | 26.3 | [Project](https://pypi.org/project/packaging/) |
| pandas | 3.0.5 | [Project](https://pypi.org/project/pandas/) |
| pillow | 12.3.0 | [Project](https://pypi.org/project/pillow/) |
| pluggy | 1.6.0 | [Project](https://pypi.org/project/pluggy/) |
| pygments | 2.20.0 | [Project](https://pypi.org/project/pygments/) |
| pyparsing | 3.3.2 | [Project](https://pypi.org/project/pyparsing/) |
| pyright | 1.1.411 | [Project](https://pypi.org/project/pyright/) |
| pytest | 9.0.2 | [Project](https://pypi.org/project/pytest/) |
| python-dateutil | 2.9.0.post0 | [Project](https://pypi.org/project/python-dateutil/) |
| pyyaml | 6.0.3 | [Project](https://pypi.org/project/pyyaml/) |
| ruff | 0.14.14 | [Project](https://pypi.org/project/ruff/) |
| scipy | 1.18.0 | [Project](https://pypi.org/project/scipy/) |
| setuptools | 84.0.0 | [Project](https://pypi.org/project/setuptools/) |
| six | 1.17.0 | [Project](https://pypi.org/project/six/) |
| sympy | 1.14.0 | [Project](https://pypi.org/project/sympy/) |
| torch | 2.13.0+cu130 | [Project](https://pypi.org/project/torch/) |
| triton | 3.7.1 | [Project](https://pypi.org/project/triton/) |
| typing-extensions | 4.16.0 | [Project](https://pypi.org/project/typing-extensions/) |
