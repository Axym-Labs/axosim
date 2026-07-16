# Iteration 170 literature scan: aggressive dynamics interventions

Date: 2026-07-13

## Scope

This scan applies the revised practical bar: a candidate should plausibly move
voltage SERA by 0.1-1.0 mV, not merely produce a paired-significant change in
the third decimal place. I161-I169 already reject mature latent/output
distillation, isolated residual-refiner training, causal multigrid, explicit
residual assignment, local ConvSSM, affine output calibration, capacity scaling
from 35k to 207k, and 2,000 ms state continuity.

## Evidence

### Sharp events need an explicit operating mode

Hybrid Neural World Models trains one physical-state surrogate over continuous
prediction horizons and recovers a per-trajectory error map concentrated at
shocks and contacts. Its largest gains come from treating sharp-event failure
as a distinct operating mode rather than expecting a uniform residual metric
to repair it. The transferable idea is direct multi-horizon supervision and an
event-error coordinate; solver fallback is not relevant to a standalone neuron
surrogate.

Source: [Lakshmanan and Chopra, Hybrid Neural World Models, 2026](https://arxiv.org/abs/2605.28317).

PRISM-WM similarly argues that monolithic continuous latent models over-smooth
hybrid dynamics at discrete mode transitions. AxoMamba's high-dV/dt and
near-spike errors are an analogous diagnostic, although I153 already warns
that an unsupervised learned regime gate is insufficient.

Source: [Prismatic World Model, 2025](https://arxiv.org/abs/2512.08411).

### Spectral bias can require an operator change

Recent high-frequency-scaling work targets neural operators that systematically
miss high-frequency physical modes, matching AxoMamba's measured gain collapse
from 0.998 below 5 Hz to 0.312 at 80-250 Hz. The key claim is architectural
latent-frequency amplification, not a frequency-weighted loss. I170 is the
first direct test through a causal high-pass latent readout.

Source: [Mitigating spectral bias in neural operators via high-frequency scaling, 2025](https://doi.org/10.1016/j.neunet.2025.108027).

Iterative Refinement Neural Operators interpret repeated correction as a
learned fixed-point solver and report frequency-resolved error reductions. This
suggests shared-weight iterative refinement as the next spectral mechanism if
I170's one-shot head fails; it is distinct from I163's supervised assignment of
the teacher residual to one module.

Source: [Iterative Refinement Neural Operators, 2026](https://openreview.net/pdf?id=4lAS8tI23u).

### Partial observability should be treated as state estimation

Finite-sample identification of partially observed bilinear systems separates
latent realization from observed input-output fitting. AxoMamba currently
assumes a deterministic zero state for every training window even though
voltage and hidden channel state at the window boundary are unobserved. Longer
context did not help at I169, so the stronger test is an explicit observed
initial-state protocol or a learned posterior over initial state, not another
burn-in.

Source: [Sattar et al., L4DC 2025](https://proceedings.mlr.press/v283/sattar25a.html).

### Dendritic structure helps when it changes information access

Chavlis and Poirazi report that structured dendritic connectivity and
restricted input sampling can improve robustness and parameter efficiency by
orders of magnitude on their tasks. AxoMamba already has sparse physical
routing and route occupancy, while generic subunits and weak graph summaries
failed. The remaining high-impact translation is exact tree-neighborhood
receptive fields with verified SWC edges, not more generic branch capacity.

Source: [Chavlis and Poirazi, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56297-9).

Topology-preserving neural operators reinforce that useful geometric priors
must preserve algebraic structure rather than append a few graph statistics.
This supports postponing another graph experiment until exact channel-to-SWC
edge transport is available.

Source: [Topology-Preserving Neural Operator Learning via Hodge Decomposition, ICML 2026](https://openreview.net/forum?id=sLVaNS2DS3).

## Ranked experiments

| rank | experiment | exact control | aggressive pass condition |
| ---: | --- | --- | --- |
| 1 | Observed/posterior initial state: provide the true initial soma voltage when the deployment contract permits it; otherwise infer a stochastic initial latent from a short prefix | zero/mean initial state with identical encoder and parameters; evaluate after the same 500 ms exclusion | at least 0.1 mV voltage and response gain; otherwise initial-state uncertainty is not the dominant error |
| 2 | Shared-weight fixed-point refinement over the completed latent/output, two or three correction iterations | same number of repeated blocks with detached/no-feedback state and equal execution | at least 0.1 mV voltage gain plus reduced high-dV/dt or near-spike error |
| 3 | Multi-horizon physical-state prediction with an explicit horizon coordinate and shared trunk | same targets and forward count with constant/shuffled horizon | at least 0.1 mV voltage gain and better sharp-event diagnostics |
| 4 | Exact SWC tree-neighborhood receptive fields before Mamba | degree-matched rewired tree and route-occupancy-only control, exact parameter budget | at least 0.1 mV or 0.05 mV plus a large intervention-response gain |
| 5 | Event-error coordinate supervised from input/teacher change labels, without free learned regime routing | equal-capacity shared decoder and shuffled event labels | at least 0.1 mV with no F1 regression; do not repeat I153's unsupervised gate |

## Recommendation

If I170 clears 0.1 mV, replicate the frequency head unchanged before changing
its cutoff. If it fails, test observed/posterior initial state first because it
changes available state information rather than model capacity. The experiment
must state explicitly whether initial voltage is allowed by the simulator API;
do not silently leak a target into the existing spike-input-only benchmark.

Fixed-point refinement is the best architecture-only fallback. It should use
shared weights and an equal-execution detached control, and should be abandoned
after one two-versus-three-iteration bracket if neither clears 0.1 mV.

## Guardrails

- No follow-up tuning for a voltage effect below 0.1 mV.
- Do not revive generic regime gates, scalar output calibration, multigrid,
  residual-target assignment, or local ConvSSM under new names.
- Treat parameter or compute changes as declared interventions and match every
  other axis; report latency alongside accuracy.
- Keep the ordinary spike-input-only task separate from any observed-initial-
  state task so a large gain is scientifically interpretable.
