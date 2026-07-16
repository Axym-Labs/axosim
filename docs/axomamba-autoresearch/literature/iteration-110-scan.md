# Literature Scan After Iteration 110

Date: 2026-07-12
Decision target: choose structural AxoMamba experiments after promoting the
rank-4 cooperative branch interaction at iteration 110.
Scope: nonlinear dynamical systems, dendritic integration, neural operators,
interventional state-space models, and explicit high-order sequence models.

## Executive Decision

Iteration 110 changes the evidence base. An instantaneous low-rank bilinear
interaction between the 118 routed branches improved 300k voltage SERA by
0.0475 mV and response SERA by 0.0059 mV, while all event-metric point estimates
were favorable. The next phase should therefore extend the successful
interaction mechanism, not resume parameter tuning or replace Mamba wholesale.

The highest-priority experiment is a **low-rank temporal Volterra branch
interaction**. One factor of the promoted bilinear path remains instantaneous;
the other receives a small causal state before multiplication. This tests
whether branch cooperation depends on relative temporal context, a mechanism
that neither the failed additive branch-memory bank nor the failed additive
delay coordinates could express. The matched control must instantiate and
execute the same state path but use the instantaneous factor, giving identical
parameters and essentially identical compute.

The second priority is **multiplicative modulation of the global hidden state**
rather than additive bilinear injection. Dendritic gating and feature-binding
work suggests that branch interactions can control which somatic computation is
active, rather than merely contribute another current. This is a distinct
placement test with the same low-rank factors and no scale sweep.

The third priority is an **intervention-aware latent transition**, motivated by
interventional state-space models. It should model baseline and perturbed
trajectories with a shared transition and an explicit perturbation response
state. Earlier paired-data mixing and direct response loss were unstable, so
this requires a structural equivariance premise and a paired inactive control,
not another data-fraction or loss-weight sweep.

Tiny statistically significant changes remain non-results. Ordinarily, a
candidate needs at least 0.01 mV on voltage or response SERA, a comparable F1
change, or a clear robustness/efficiency gain without a significant core
regression.

## Evidence Filter

### What iteration 101-110 established

- Branch-local scalar Mamba was too slow and functionally weak: 5.3x latency
  for sub-material changes.
- Additive fixed branch memory improved one seed and reversed on replication.
- Explicit lag coordinates worsened held-out residual prediction in all five
  morphologies.
- Official Mamba-2 materially harmed voltage and event fidelity.
- Mamba-GRU hybrids produced large but seed-unstable SERA/F1 tradeoffs in both
  orders.
- Instantaneous low-rank branch interaction survived a 300k fresh-seed gate and
  is now the promoted architecture.

This rules out generic local recurrence, fixed smoothing, additive lags, and
another broad recurrence replacement as immediate follow-ups. It does not rule
out memory *inside a nonlinear interaction*, which is a different function
class.

## Evidence Synthesis

### 1. Explicit nonlinear interactions with memory

The Discrete Volterra Network (DiVo) represents nonlinear time-series dynamics
through low-rank polynomial interaction kernels with memory and sparsification.
It is an ICLR 2026 submission rather than accepted evidence, but its central
construction is directly testable here: second-order interactions need not be
instantaneous, and low-rank factorization can keep them compact.
[OpenReview](https://openreview.net/forum?id=i4Mph3pvwr)

The promoted AxoMamba term is already a rank-4 second-order feature,
`silu(U b_t) * silu(V b_t)`, but both factors see only the current branch vector.
A causal state on one factor yields a low-rank temporal interaction without
expanding all branch-by-branch lag products. This differs from iteration 102,
where filtered branch currents were only concatenated into an additive linear
projection.

**AxoMamba implication.** Use four rank-specific causal exponential states,
initialized across event-to-membrane timescales, on one factor before the
product. Compare temporal versus instantaneous selection while both candidates
instantiate and execute the same filters. Do not sweep timescales or coupling
strength in the first test.

### 2. Dendritic computation is integration plus modulation

DendSN explicitly models multiple nonlinear dendritic branches and reports
better expressivity and robustness than point-neuron SNNs. Its revised 2025
version also uses dendritic branch gating to reduce task interference. The
classification results do not transfer to neuron simulation, but the structural
distinction between branch contribution and branch modulation does.
[arXiv](https://arxiv.org/abs/2412.06355)

Spiking-WM places nonlinear multi-compartment integration inside a temporal
world model and reports gains over other spiking models on long sequences, with
control performance comparable to a GRU world model. Again, the useful evidence
is architectural: evolving compartment signals cooperate before a global
prediction is formed.
[arXiv](https://arxiv.org/abs/2503.00713)

Single-neuron feature-binding work finds that spatially and temporally modulated
dendritic sublinearity changes which distributed input combinations can be
bound. This supports testing interaction-as-gate after temporal interaction,
while not implying that its Purkinje-cell conclusions transfer to the present
morphologies.
[arXiv](https://arxiv.org/abs/2405.12645)

**AxoMamba implication.** If temporal interaction fails, that result should not
invalidate the promoted interaction. The next clean test changes its role from
an additive hidden update to a bounded multiplicative modulation of the hidden
state, with identical factorization and parameter budget.

### 3. State-space neural operators favor structured interaction, damping, and
frequency selection

SS-NO, published in TMLR in March 2026, combines state-space memory with
adaptive damping and learnable frequency modulation for time-dependent PDE
operators, reporting competitive errors with fewer parameters. It supports
treating AxoMamba as a learned causal solution operator, but iterations 85-90
already rejected generic filtering and frequency-loss shaping. Its transferable
lesson is to place state where it controls a structured interaction, not to add
another global spectral objective.
[TMLR paper](https://openreview.net/forum?id=SwLxxz0x58)

Latent Mamba Operator combines a latent neural-operator kernel with Mamba for
PDEs. Its spatial geometry is not the neuron morphology, but it independently
supports separating interaction construction from temporal latent evolution.
[ICML 2025](https://openreview.net/forum?id=Y4BDcJmb8t)

SKOLR links lag-extended observables to structured Koopman linear recurrence.
Iteration 104's negative delay audit makes a direct lag stack low priority here;
the remaining relevant idea is that nonlinear measurement functions should be
chosen before a compact transition is fitted.
[ICML 2025](https://openreview.net/forum?id=Xg1BGlybfq)

### 4. Intervention response should be structural, not merely another loss

Interventional State Space Models explicitly distinguish observational dynamics
from responses to causal manipulations and report better recovery under novel
perturbations than ordinary SSMs on simulated and biological neural data. This
is closer to AxoBench's intervention SERA than generic forecasting work.
[ICML 2025](https://openreview.net/forum?id=n7qKt6gjl9)

Iterations 81-84 showed that paired intervention exposure can help one seed and
reverse on another, while direct response-SERA optimization overfit training
pairs. A future test should therefore constrain how perturbations act on latent
state, for example through a shared baseline transition plus a low-rank response
correction, and hold exposure fixed. It should not tune pairing fractions.

## Ranked Structural Queue

| rank | experiment | falsifiable mechanism | acceptance gate |
| ---: | --- | --- | --- |
| 1 | temporal low-rank branch interaction | branch cooperation depends on causal relative timing, not only simultaneous branch amplitudes | material core gain over an exact-parameter, matched-compute instantaneous control; no significant core regression |
| 2 | bilinear hidden modulation | cooperative dendritic features select or gate global processing rather than add another current | material SERA/F1 gain over additive promoted interaction with equal parameters and compute |
| 3 | response-equivariant latent state | perturbations should alter a shared transition through a constrained response coordinate | replicate intervention SERA gain across seeds without voltage/F1 failure |
| 4 | interaction-conditioned Mamba input gate | branch interaction should control selective recurrence before state update | only after ranks 1-2 isolate the useful interaction role |
| 5 | operator-style branch/state alternation | branch geometry and temporal state require repeated interaction, not one projection followed by two temporal blocks | exact-budget replacement, with latency reported and no hand-tuned depth sweep |
| 6 | official Mamba-3 recurrence | improved discretization/MIMO state may improve state tracking | remains kernel-gated until official SISO or MIMO executes reliably on this GPU |

## Iteration 111 Design

Implement one rank-4 causal state on the second factor of the promoted bilinear
interaction. Initialize its four timescales to 5, 20, 50, and 100 ms so the test
spans event-local through membrane-scale context without a coefficient sweep.
Run three candidates on one frozen 120k schedule:

1. promoted instantaneous bilinear baseline;
2. exact-parameter/matched-compute instantaneous control that computes but does
   not select the causal factors;
3. temporal-factor bilinear interaction.

The active-versus-compute-control contrast is causal for the structural change.
The promoted baseline detects any effect from adding dormant state parameters
or changing implementation. Continue only if the temporal model yields a
material core gain without a significant regression; otherwise remove the path
at the next cleanup and move to multiplicative hidden modulation.
