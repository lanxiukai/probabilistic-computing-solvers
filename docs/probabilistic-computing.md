# Probabilistic computing and p-bits

English | [简体中文](probabilistic-computing.zh-CN.md)

Probabilistic computing uses controlled randomness as part of computation.
Instead of representing an answer only by one deterministic state, a system can
explore a distribution of states and use its samples to estimate probabilities,
complete partially specified information, or search for low-cost solutions.
Here we focus on networks of **probabilistic bits, or p-bits**.

This is an introduction to the model and its history. The repository contains
GPU simulations and classical controls; its measured scope and limitations are
listed in the [benchmark overview](benchmark-overview.md).

## 1. A p-bit is a tunable random bit

A p-bit produces a binary output whose probability can be adjusted by an input.
At zero bias an ideal p-bit spends equal time in its two states. Positive or
negative bias favors one state without necessarily making it permanent.
Connecting p-bits makes their probabilities depend on other states, so the
network can express correlations and constraints. [P3](#p3)

For a bipolar state $m_i\in\{-1,+1\}$, a common idealized update uses a fresh,
independent uniform draw:

$$
m_i^{\mathrm{new}}=\operatorname{sign}\!\left[\tanh I_i-r_i\right],
\qquad r_i\sim\mathcal U(-1,1).
$$

Consequently,

$$
\Pr(m_i^{\mathrm{new}}=+1\mid I_i)
=\frac{1+\tanh I_i}{2}
=\frac{1}{1+\exp(-2I_i)}.
$$

Thus the dimensionless input $I_i$ controls a probability, while each output
is still one definite binary value. The alternative encoding is
$x_i=(m_i+1)/2\in\{0,1\}$. This is a mathematical reference model; an actual
device needs a measured input-response curve and may have temporal correlations.
[P6](#p6)

~~~mermaid
flowchart LR
    R["Random source"] --> P["Tunable p-bit"]
    B["Bias / local field"] --> P
    P --> S["Binary state"]
    S --> C["Coupling / constraint network"]
    C --> B
~~~

The diagram is a functional view, not a particular transistor circuit.
A software pseudorandom generator, digital logic, or a noisy physical device can
supply the random input.

## 2. From local updates to an energy model

One useful model assigns an energy to the joint state:

$$
E(\mathbf m)=-\sum_{i<j}J_{ij}m_i m_j-\sum_i h_i m_i,
\qquad J_{ij}=J_{ji},\quad J_{ii}=0.
$$

The local field for bit $i$ is

$$
H_i=h_i+\sum_{j\ne i}J_{ij}m_j,\qquad I_i=\beta H_i.
$$

Holding all other bits fixed, the energy difference between $m_i=+1$ and
$m_i=-1$ is $-2H_i$. Substituting this difference into the conditional
Boltzmann probability gives the sigmoid above. At a fixed finite
$\beta\ge0$, exact single-site heat-bath updates with an appropriate scan and
ergodic dynamics have stationary distribution

$$
\pi(\mathbf m)=Z^{-1}\exp[-\beta E(\mathbf m)].
$$

The constant $Z$ normalizes the probabilities. This connects stochastic binary
networks to statistical mechanics and Boltzmann machines. Here $\beta$ is an algorithmic inverse-temperature
parameter; it need not be the physical temperature of a device.
[P2](#p2), [P6](#p6)

**The update schedule is part of the algorithm.** Updating coupled bits
simultaneously from old fields does not generally implement the same Gibbs
sampler. Conditionally independent sets can be updated together; interacting
blocks need a valid joint update or a justified correction. Hardware delays,
finite precision, correlated random inputs, and stale boundary states can also
change the distribution. Parallel architecture must therefore be evaluated
together with its dynamics. [P5](#p5), [P8](#p8)

Not every solver in this repository uses this single-site rule. There are also
Metropolis-Hastings proposals, conditional blocks, tempering, population
methods, and classical repair stages. Their own method and protocol documents
define the actual process.

## 3. A small example: an AND constraint used in either direction

Let $x,y,z\in\{0,1\}$ and require $z=x\land y$. As an illustrative construction,
the penalty

$$
E_{\mathrm{AND}}(x,y,z)=\lambda(xy-2xz-2yz+3z),\qquad \lambda>0
$$

is zero on exactly the valid truth-table rows and positive on the other four
assignments:

| $x$ | $y$ | Valid $z$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Fixing $x$ and $y$ lets the remaining variable favor the valid output.
Fixing $z=0$ instead allows the three compatible input pairs to be explored;
fixing $z=1$ favors $(x,y)=(1,1)$. This is the idea of **invertible logic**:
conditioning different ports of a constraint network changes which variables
are inferred. The example above is a direct algebraic construction, rather
than a device measurement. [P3](#p3)

At finite temperature and finite penalty, invalid states can still appear.
A larger multiplier or scheduling model also introduces many interacting
constraints. A correct small gate therefore does not establish fast
factorization or a general optimization advantage.

## 4. Sampling, inference, and optimization ask different questions

| Task | Desired result | What must be checked |
|---|---|---|
| Sampling | States with a specified distribution | Stationarity, mixing, autocorrelation, and distributional error |
| Inference | Conditional probabilities or likely assignments given observations | Model, conditioning, estimator, and uncertainty |
| Optimization | A feasible assignment with a good objective | Original constraints, objective, quality target, and full cost |

For optimization, lowering the effective temperature can favor low-energy
states. Finite-time annealing can still become trapped. Tempering and collective
updates try to improve exploration, while strong classical methods may exploit
structure that a generic energy model misses. [P1](#p1), [P2](#p2)

For inference, the most likely complete assignment and the most likely value
of each individual variable are different quantities. For sampling, a good
minimum-energy state is not evidence that the distribution is correct.
The same network can support different tasks, but their success criteria must
be stated separately.

## 5. Why the idea developed, and how it is implemented

The algorithmic foundations predate the p-bit name: Monte Carlo methods use
random transitions to study distributions, and Boltzmann machines connect
stochastic networks with learning. Modern p-bit research asks whether making
these primitives native to an architecture can reduce the cost of repeated
random-number generation, local-field calculation, and state updates.
Energy efficiency is a hardware research objective that requires measurement,
not a consequence of using the word “probabilistic.” [P1](#p1), [P2](#p2), [P6](#p6)

| Period | Selected milestone | Significance |
|---|---|---|
| 1953 | Metropolis and colleagues' Monte Carlo work [P1](#p1) | Sampling configurations with energy-dependent stochastic transitions |
| 1985 | Ackley, Hinton, and Sejnowski's Boltzmann-machine learning [P2](#p2) | Learning and conditional completion in stochastic networks |
| 2017 | Stochastic p-bits and invertible logic [P3](#p3) | A tunable random-bit abstraction connected to logic constraints |
| 2019 | Stochastic magnetic-tunnel-junction factorization experiment [P4](#p4) | A physical proof of concept for a small coupled probabilistic system |
| 2022 | Sparse Ising-machine architecture [P5](#p5) | Studying parallel updates and the relationship between connectivity and computation |
| 2025 | Integrated circuit with a voltage-controlled MTJ entropy source [P7](#p7) | Combining a magnetic random source with digital computation |
| 2026 | Networked FPGA system reporting 1,000,000 p-bits [P8](#p8) | A preprint studying distributed scale and communication-induced accuracy tradeoffs |

This is a selected reading path, not an exhaustive chronology. The 2026 entry
is a preprint, and its scale or performance is not a result of this repository.
Device count, connectivity, auxiliary copies, replicas, and target accuracy
must be matched before comparing systems.

Common implementation routes include:

- **GPU/CPU software:** numerical probabilities, pseudorandom draws, and explicit
  update kernels. This is the implementation level measured here.
- **Digital FPGA/ASIC circuits:** programmable state, random sources, field
  accumulation, and update control. A digital p-bit can occupy many logic elements.
- **Stochastic devices with supporting circuits:** for example, fluctuating
  magnetic tunnel junctions with readout and bias control. Other designs use
  voltage-driven magnetic randomness as an entropy source for digital logic.
  These are different circuit architectures, not interchangeable device counts.
  [P4](#p4), [P7](#p7)

Coupling storage and communication, calibration, update timing, and readout may
dominate system cost even when the random element itself is small.

## 6. Bits, p-bits, and qubits

| Unit | Operational description |
|---|---|
| Digital bit | A classical binary value stored or manipulated by logic |
| p-bit | A classical random binary output with a controllable probability |
| Qubit | A quantum state with amplitudes and phase, manipulated coherently and measured probabilistically |

A p-bit network uses classical probabilities. Its fluctuations do not provide
quantum coherence, interference, or entanglement. Classical p-bit methods may
simulate some quantum models through a suitable mathematical mapping, but that
does not make their hardware a quantum computer. Likewise, software simulation
of p-bits does not reproduce the energy consumption of a magnetic circuit.
[P6](#p6)

## 7. What matters when reading a result

The useful comparison is the complete task at matched quality and resources:
instance preparation, encoding, search, transfer, repair, verification, and
any training or tuning. A high flip rate does not by itself imply a short time
to a good solution. A large encoded network may include auxiliary variables,
and many replicas of a small problem are not one larger coupled instance.

This repository retains cases where stochastic methods help and cases where
strong classical controls remain better. It does not establish universal
p-bit superiority, quantum advantage, or a general hardware energy advantage.
Start with the [benchmark overview](benchmark-overview.md), then read the
problem-specific report and [reproduction instructions](reproduction.md).

## References and reading scope

Sources checked on 2026-10-21. The equations and AND example above are
explanatory derivations. Historical claims use the sources below; publisher
abstracts and preprints support only the stated scope. Citing hardware work
does not mean it was reproduced here.

<a id="p1"></a>
**P1.** N. Metropolis et al. *Equation of State Calculations by Fast Computing
Machines*. Journal of Chemical Physics 21, 1087–1092 (1953).
[DOI](https://doi.org/10.1063/1.1699114) ·
[Paper copy](https://people.umass.edu/bvs/The_1953_paper.pdf).
Background on Monte Carlo sampling.

<a id="p2"></a>
**P2.** D. H. Ackley, G. E. Hinton, and T. J. Sejnowski.
*A Learning Algorithm for Boltzmann Machines*. Cognitive Science 9, 147–169 (1985).
[Author-hosted paper](https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf).
Stochastic-network energy, equilibrium, conditioning, and learning.

<a id="p3"></a>
**P3.** K. Y. Camsari, R. Faria, B. M. Sutton, and S. Datta.
*Stochastic p-Bits for Invertible Logic*. Physical Review X 7, 031014 (2017).
[Publisher](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.031014).
The p-bit abstraction and conditioning of logic networks.

<a id="p4"></a>
**P4.** W. A. Borders et al. *Integer factorization using stochastic magnetic
tunnel junctions*. Nature 573, 390–393 (2019).
[Publisher](https://www.nature.com/articles/s41586-019-1557-9).
Publisher abstract and extended-data descriptions; a physical proof of concept,
not a basis for a general speed or energy claim.

<a id="p5"></a>
**P5.** N. A. Aadit et al. *Massively parallel probabilistic computing with
sparse Ising machines*. Nature Electronics (2022).
[DOI](https://doi.org/10.1038/s41928-022-00774-2) ·
[Author preprint, v2](https://arxiv.org/abs/2110.02481v2).
Architecture and parallelism background; reported speedups are not adopted here.

<a id="p6"></a>
**P6.** S. Chowdhury et al. *A full-stack view of probabilistic computing with
p-bits: devices, architectures and algorithms*. IEEE Journal on Exploratory
Solid-State Computational Devices and Circuits (2023).
[DOI](https://doi.org/10.1109/JXCDC.2023.3256981) ·
[Author manuscript, v3](https://arxiv.org/abs/2302.06457v3).
Sections on the update model and implementation routes.

<a id="p7"></a>
**P7.** *An integrated-circuit-based probabilistic computer that uses
voltage-controlled magnetic tunnel junctions as its entropy source*.
Nature Electronics 8, 784–793 (2025).
[Publisher](https://www.nature.com/articles/s41928-025-01439-6).
Publisher abstract; used for the architecture example, without endorsing
performance comparisons.

<a id="p8"></a>
**P8.** N. A. Aadit et al. *Programmable Probabilistic Computer with
1,000,000 p-bits*. arXiv:2606.25313v1 (2026), preprint.
[Author preprint](https://arxiv.org/abs/2606.25313v1).
Abstract-level description of networked FPGAs and the communication/accuracy
tradeoff; no hardware reproduction or independent performance validation here.
