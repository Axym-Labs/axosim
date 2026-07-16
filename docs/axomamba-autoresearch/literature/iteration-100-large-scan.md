# Literature Scan After Iteration 100

Date: 2026-07-12
Decision target: select high-temperature structural AxoMamba experiments for
iterations 101-109 after the near-spike objective promotion.
Scope: biological-neuron surrogates, multicompartment computation, state-space
sequence models, learned dynamical-system simulation, neural operators, latent
state estimation, and high-order time-series dynamics.

## Executive Decision

The strongest next experiment is **shared branch-local temporal state before
cross-branch aggregation**. AxoMamba currently applies nonlinear routing to 118
branch currents, projects and mixes them into 58 channels, and only then gives
the sequence to Mamba. That forces all synapse groups to lose their separate
temporal state before the main recurrence. Recent multicompartment work instead
assigns dynamics to dendritic compartments and integrates their evolving states
at the soma. This is directly aligned with the task and is structurally distinct
from the rejected global fast/slow split, explicit fixed current ports, and
grouped hidden channels.

The second priority is an **official Mamba-3 recurrence/MIMO comparison**.
Mamba-3 changes the state update itself through improved discretization,
complex-valued dynamics, and a multi-input/multi-output formulation. Its
accepted ICLR 2026 evidence is much stronger than another hand-designed Mamba
side path, and official code now exists. It should be attempted after the local
compartment test, with matched parameter budgets and measured kernel latency.

The third priority is **high-order observable/state augmentation**, but only
after a premise audit. Delay-coordinate and high-order system-identification
work argues that a first-order latent model may hide true multi-lag structure in
an inflated state. AxoMamba already has recurrent state, so explicit voltage
derivative/curvature heads or tapped observables are warranted only if residual
conditional dependence remains after controlling for the current latent state.

Tiny metric movement is not a research result. Statistical significance remains
a guard against noise, but structural candidates ordinarily need at least 0.01
mV on voltage or response SERA, a similarly meaningful F1 change, or a clear
robustness/efficiency gain. The scan is used to change mechanisms, not to create
new scalar sweeps.

## Search Protocol

### Inclusion

- Primary papers, official proceedings, official repositories, and current
  accepted work available through 2026-07-12.
- Mechanisms that alter where temporal state lives, how it evolves, or how a
  partially observed dynamical system is represented.
- A controlled implementation path under AxoMamba's frozen 120k presentation,
  full-diagnostics, five-morphology protocol.
- Evidence relevant to sparse events, local compartments, intervention
  response, state tracking, or learned trajectory simulation.

### Exclusion and negative-evidence filter

- Generic spike smoothing and filtered event losses: rejected at I85 and I96.
- Generic parallel fast/slow Mamba: rejected at I86.
- Frequency-domain loss shaping: failed the I90 long gate.
- Unsigned load/timestep modulation: rejected at I91.
- Fixed fast/slow E/I proxies and readout injection: failed replication at I98.
- Hidden-state grouping and polarity ports: rejected at I99.
- Full unfused biophysical recurrence: failed the measured latency gate by 174x.
- Autoregressive rollout stabilization: AxoMamba maps exogenous currents to the
  trace and does not feed predicted voltage back as next-step state.
- Irregular-sampling machinery: AxoBench uses fixed 1 ms sampling.

### Search record

Searches were run on 2026-07-12 across arXiv, OpenReview/ICLR/ICML/NeurIPS,
PMLR, PubMed, PNAS, Nature-family journals, and official GitHub repositories.
Query families included:

- `2025 2026 multicompartment dendritic temporal processing sequence model`
- `Mamba-3 complex dynamics MIMO official implementation`
- `high-order state-space time series delay coordinate latent state`
- `neural operator dynamical system latent state observer sparse observation`
- `state tracking limits diagonal SSM 2026`
- `biological neuron surrogate local dendritic dynamics`

The local search screened all iteration-84 through iteration-100 records and
artifacts before ranking ideas, so the queue does not relabel a failed branch.

## Evidence Synthesis

### 1. Temporal state should exist before compartment integration

The most task-specific precedent remains the NeuronIO surrogate result: detailed
dendritic computation requires temporal nonlinear structure, and removing NMDA
conductances substantially reduces the network complexity needed to emulate the
cell. This establishes that branch-local temporal effects can be computationally
real rather than decorative.
[Neuron 2021](https://doi.org/10.1016/j.neuron.2021.07.002)

The Branch-ELM baseline gives each routed branch explicit trainable memory before
the soma combines branch outputs. This is the closest architectural control for
AxoMamba and a stronger motivation for branch-local state than generic
multiscale forecasting results.
[ICLR 2024](https://openreview.net/forum?id=vE1e1mLJ0U)

Spiking-WM uses multicompartment neurons for nonlinear integration of multiple
dendritic sources over long sequences and reports comparable control performance
to GRU world models plus gains over other SNNs on long speech sequences. The
work is now published in PNAS, strengthening the earlier preprint evidence. Its
specific SNN claims do not transfer, but the placement of dynamics before
compartment integration does.
[PNAS 2025](https://doi.org/10.1073/pnas.2513319122)

Related multicompartment reinforcement-learning work assigns apical, basal, and
somatic compartments distinct functions rather than merely partitioning an
already mixed hidden state. Again, the transferable principle is local evolving
state followed by integration.
[Neural Networks 2025](https://doi.org/10.1016/j.neunet.2024.106898)

**AxoMamba implication.** Apply one small Mamba block with shared weights to
each routed branch sequence, then integrate branch states and retain a reduced
global Mamba trunk. Sharing keeps the model from assigning a separate arbitrary
network to every branch while preserving local temporal processing. The control
must have the same parameter count and comparable recurrence compute; branch
identity and morphology conditioning stay unchanged.

### 2. Upgrade state dynamics, not just their scalar decay

Mamba-3 is an ICLR 2026 oral that introduces a recurrence from improved SSM
discretization, complex-valued state updates for richer tracking, and MIMO state
updates. The paper reports better state tracking and downstream quality while
using smaller state than Mamba-2 in some comparisons. These are changes to the
recurrence's representational class, unlike AxoMamba's failed decay
initialization, grouped channels, or external current ports.
[ICLR 2026 paper](https://arxiv.org/abs/2603.15569)

The authors' official repository exposes Mamba-3 and MIMO mode. This makes a
faithful extension preferable to reimplementing the equations locally, subject
to an installation/kernel feasibility audit on the current GPU.
[Official state-spaces/mamba repository](https://github.com/state-spaces/mamba)

An accepted ICLR 2026 analysis of diagonal SSMs identifies limits on state
tracking. It reinforces the possibility that current real diagonal dynamics are
the bottleneck, but it does not predict an AxoBench gain by itself.
[ICLR 2026](https://openreview.net/forum?id=5bg5Ru5OML)

Work on coefficient dynamics derives expressivity/stability tradeoffs across
SSMs, attention, and gated linear recurrences. Its useful lesson here is to
change coefficient dynamics only with a declared state-tracking premise, not to
sweep decay values without evidence.
[ICLR 2026 submission](https://openreview.net/forum?id=9uChwgWHX9)

**AxoMamba implication.** Benchmark official Mamba-3 SISO and MIMO step latency,
memory, and parameter formula first. If feasible, compare an exact-budget
Mamba-3 trunk against the promoted trunk. Do not mix this test with branch-local
state or new losses.

### 3. High-order and observable dynamics

High-order dynamics work argues that forcing a first-order transition onto a
multi-lag process can push memory into unnecessarily large latent states. It
uses conditional mutual information to estimate Markov order and discusses
Hankel/delay-coordinate reconstructions as observable-space alternatives. The
paper is under review, so it motivates a diagnostic rather than a wholesale
framework transplant.
[ICLR 2026 submission](https://openreview.net/pdf?id=RryrppAV70)

Latent time-series forecasting work argues that observation-only objectives can
produce temporally disordered latent states and adds explicit latent forecasting
structure. AxoMamba's current final representation has shown low effective rank,
but earlier linear-decodability audits did not identify a clean missing target.
[ICML 2026](https://openreview.net/forum?id=s49fw3BVU0)

Sparse-observation latent neural operators encode multiple observations before
evolving a latent operator and report robustness to noise and sparse sensing.
That setting differs from AxoBench's fully observed inputs, but it suggests a
state-observer audit when testing intervention or morphology transfer.
[ICLR 2026 submission](https://openreview.net/forum?id=bVyg58mjhF)

RTS smoother-guided neural differential modeling alternates latent-state
estimation and parameter learning under partial observation. AxoBench has no
privileged channel-state labels, so this is not an immediate architecture, but
it clarifies why inventing fixed current proxies at I97 was weaker than learning
an observable latent state from measured outputs.
[TMLR submission 2026](https://openreview.net/forum?id=GmMtDQ54iR)

**AxoMamba implication.** Estimate whether target voltage at t+1 retains
conditional information from delayed targets/inputs after a frozen model's
current representation. Only if the residual dependence is material should a
candidate add causal delay-coordinate observables or a joint level/derivative/
curvature state head. Direct derivative loss alone has already failed and is
not to be repeated.

### 4. Neural operators and predictor-corrector state

State-space neural operators report accurate, efficient mappings for dynamical
systems and motivate treating AxoMamba as a learned solution operator, not merely
a sequence classifier.
[Neural Networks 2025](https://doi.org/10.1016/j.neunet.2025.108496)

Mamba Neural Operator applies Mamba-style sequence processing to operator
learning. Its broad result supports SSMs in physical simulation, but its spatial
tokenization and PDE benchmarks do not identify the correct neuron-surrogate
architecture.
[arXiv 2024](https://arxiv.org/abs/2410.02113)

CoEvol-NO, an ICML 2026 spotlight, couples persistent latent state and coordinates
with an error-driven predictor-corrector update. AxoMamba lacks a spatial mesh
and does not autoregress on its voltage, so the full method does not transfer.
The potentially useful later idea is an internal predictor-corrector state whose
correction is trained from current-step output residuals without future leakage.
[ICML 2026](https://openreview.net/forum?id=pG9Wn3Aheq)

**AxoMamba implication.** Keep operator and predictor-corrector ideas behind the
branch-local and recurrence tests. Any correction must remain causal at
inference and must not use teacher voltage as an input.

### 5. What biological inspiration does and does not justify

Recent dual-compartment forecasting and multicompartment spiking work repeatedly
find value in functional separation across compartments and frequency ranges.
[TS-LIF 2025](https://arxiv.org/abs/2503.05108),
[MSFI 2026](https://doi.org/10.1016/j.neunet.2025.108024)

Iterations 85-99 show that AxoMamba does not benefit from importing the surface
form of those mechanisms: fixed smoothing, generic fast/slow paths, handcrafted
current states, and polarity grouping all failed. The surviving hypothesis is
narrower: **preserve local temporal state until after the biologically meaningful
branch partition has done work**. Biological language is not an acceptance
criterion; SERA, F1 over time, intervention SERA, and failure diagnostics are.

## Ranked Structural Queue

| rank | experiment | falsifiable mechanism | acceptance gate | risk |
| ---: | --- | --- | --- | --- |
| 1 | shared branch-local Mamba before aggregation | local synapse groups require temporal integration before cross-branch mixing | material improvement in response/voltage SERA or F1 with no significant core regression; must beat exact-budget global-only control | medium |
| 2 | official Mamba-3 SISO/MIMO trunk | complex/MIMO recurrence tracks neuron state better than current real diagonal selective SSM | feasible latency/memory, exact parameter budget, material core gain | medium-high |
| 3 | conditional Markov-order/observability audit | current latent state omits delayed information needed for voltage/event evolution | predeclared residual-information threshold before any model change | low audit |
| 4 | learned causal high-order observable state | explicit level/slope/curvature state prevents first-order latent compression | only after rank 3 passes; must improve dynamics plus another core axis | medium |
| 5 | branch-local predictor/global corrector | local state predicts and global state corrects interactions without teacher leakage | must outperform rank-1 simpler local-state model | high |
| 6 | Hankel-energy-guided state allocation | many current Mamba states are low-energy while short-delay observables are missing | state audit plus exact-budget reallocation, not state-size tuning | medium |
| 7 | intervention-equivariant latent correction | dropout response should be linear locally even when baseline dynamics are nonlinear | premise audit on paired teacher responses; no intervention label input | high |

## Iteration 101 Design

**Premise.** Cross-branch projection before recurrence irreversibly mixes
compartments whose local histories can have different causal effects. A shared
branch-local recurrence can form a temporally integrated feature for each routed
branch before the global soma-like trunk combines them.

**Candidates.** Implement a small shared local Mamba over every branch sequence,
then concatenate/project the local outputs into a reduced global Mamba trunk.
Use at least two independently attributable placements in parallel if exact
budgeting permits:

1. local state only before aggregation plus a reduced global trunk;
2. local residual state before aggregation plus the reduced global trunk;
3. an exact-parameter global-only control with the same reduced trunk and an
   inactive but computed local path where feasible.

Do not tune a scale grid. Use zero-initialized residual coupling for an initial
equivalence test, then one structurally active magnitude fixed by normalization.
Parameter counts must match exactly or within a predeclared negligible fraction;
recurrence calls, wall time, and GPU memory must be reported because branch-local
execution can become expensive even with shared parameters.

**Premise diagnostics.** Before the 120k panel, verify causality, branch
permutation behavior under shared weights, local-state non-degeneracy, exact
budget, and a short step-latency gate. After training, audit whether local states
remain branch-distinct and whether their use changes under sparse bursts and
interventions.

**Promotion gate.** Require no significant regression on any core metric and at
least one practically material improvement: 0.01 mV on voltage or response
SERA, 0.01 on F1 area/5 ms F1, or an equivalently compelling intervention/
efficiency result. A low-1e-3 voltage delta is explicitly insufficient.

## Iterations 102-109 Sketch

- I102-I103: replicate only a material branch-local lead; otherwise close the
  family after one architecture-level placement correction, not a coefficient
  sweep.
- I104: run the official Mamba-3 kernel/parameter feasibility audit and record
  the installed revision and hardware behavior.
- I105-I107: exact-budget Mamba-3 SISO/MIMO tests if feasible; otherwise move to
  the high-order observability audit without building a local approximation.
- I108: test a learned high-order state only if the audit passes its threshold.
- I109: replicate the strongest structural lead for the I110 long gate.
- I110: 300k gate, mandatory cleanup, full verification, and the next dedicated
  literature refresh.

## Limits

- Multicompartment gains in SNN control/speech do not estimate AxoBench effect
  size and may depend on spike-specific dynamics absent here.
- Mamba-3 evidence is primarily language/state-tracking; kernel feasibility and
  behavior at AxoMamba's small width must be measured.
- The high-order paper is under review, and numerical differentiation can be
  brittle around spikes. Any derivative state needs robust causal construction.
- A single 120k seed screens mechanisms; promotion still requires fresh-seed and
  300k evidence.
- Exact parameter matching does not equal compute matching. Both are reported,
  and a materially slower model needs a larger quality gain to be useful.
