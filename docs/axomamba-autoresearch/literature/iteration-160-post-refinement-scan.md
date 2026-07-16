# Iteration 160 literature scan: post-operator refinement

Date: 2026-07-12

## Question

What should follow if I160 confirms that a causal local TCN applied after the
global AxoMamba state is the first robust, balanced structural improvement?

This scan starts from the observed mechanism rather than reopening failed
families. I151-I154 rejected explicit membrane states, branchwise decays,
trajectory-regime experts, and subtask curricula. I155-I159 instead show that a
post-trunk local operator can improve voltage, dynamics, intervention response,
and event F1 together.

## Evidence

### TCNs are appropriate finite-memory system correctors

Temporal convolutional networks can approximate causal input-output maps with
approximately finite memory, including maps with finite-dimensional state-space
realizations. This supplies a formal reason that a local TCN can correct the
finite-horizon errors of a global state-space trunk rather than merely add
generic width.

Source: [Hanson and Raginsky, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/39555391eb0624a439c5131b1bb8a2e0-Abstract.html).

The original NeuronIO result independently established temporal convolution as
a strong empirical surrogate for detailed single-neuron dynamics. AxoMamba's
result is complementary: Mamba models the global state, while the TCN corrects
its completed representation locally.

Source: [Beniaguev, Segev, and London, Neuron 2021](https://doi.org/10.1016/j.neuron.2021.07.002).

### Local convolution and global state-space structure are complementary

MambaNO explicitly combines global Mamba integration with local alias-free
processing and reports strong neural-operator accuracy, including multiscale
PDEs. ConvS5 similarly combines convolutional structure with state-space
dynamics and reports gains on long-horizon spatiotemporal prediction. These do
not determine AxoMamba's best fusion point, but they support the hybrid model
class now observed empirically.

Sources: [MambaNO, NeurIPS 2024](https://openreview.net/forum?id=gUEBXGV8JM),
[ConvS5, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ff9783ec29688387d44779d67d06ef66-Abstract-Conference.html).

### Refinement can be a distinct optimization stage

Spectral-Refiner reports that a dedicated refinement/fine-tuning stage can
substantially improve a trained spatiotemporal neural operator. The transferable
premise is to make the correction role explicit during optimization: first
establish the global operator, then train the local corrector on its residual,
then optionally joint-fine-tune. Its turbulence-specific spectral machinery is
not transferable.

Source: [Cao et al., ICLR 2025](https://openreview.net/forum?id=MKP1g8wU0P).

Adaptive Correction for Neural Operators offers newer but not yet
peer-reviewed support for a lightweight learned post-operator correction. Its
strict conservation projection has no justified neuron analogue, so only the
plug-and-play correction architecture is relevant.

Source: [Liu et al., ICML 2026 submission](https://openreview.net/forum?id=hXqkdhwV46).

### Multigrid structure is the next scale-separation test

Multigrid-inspired surrogates process coarse and fine resolutions explicitly
and report lower prediction errors than single-resolution U-Nets on fluid
dynamics. Multiscale Neural Operators likewise treat unresolved fine dynamics
as a learned correction to coarse dynamics. The AxoMamba translation is a
causal coarse-rate post-TCN alongside the current full-rate post-TCN, not a
larger dilation sweep.

Sources: [Le and Ooi, 2021](https://arxiv.org/abs/2105.03854),
[Lütjens et al., 2022](https://arxiv.org/abs/2207.11417).

### Recurrent-operator training is not directly applicable

Recent Recurrent Neural Operator work improves autoregressive PDE rollouts by
training on the model's own predictions. AxoMamba emits a sequence directly
from exogenous synaptic input and does not feed predicted voltage back as the
next input, so exposure-bias claims do not transfer. It becomes relevant only
if a future model is explicitly autoregressive.

Source: [Ye, Zhang, and Wang, 2025](https://arxiv.org/abs/2505.20721).

## Ranked experiments

| Rank | Experiment | Exact control | Falsifying result |
|---|---|---|---|
| 1 | Residual-refiner training: initialize from the promoted post-TCN model, freeze global trunk while training the TCN on the existing joint target, then joint-fine-tune within the same total presentation budget | End-to-end post-TCN training with identical initialization, presentations, optimizer steps, and phase boundaries; frozen random local path control | No material SERA/F1 improvement, or early freezing creates a persistent event/voltage tradeoff |
| 2 | Causal multigrid post refiner: full-rate TCN plus a pooled coarse-rate TCN, fused after Mamba | Two full-rate TCN paths with identical parameters and executed blocks; scale-zero | Coarse path does not beat full/full capacity or worsens dynamics/near-spike behavior |
| 3 | Residual-target auxiliary: train the post TCN against detached teacher-minus-trunk voltage residual while retaining the joint output objective | Same TCN and compute with a shuffled or zero-weight residual target | Auxiliary fits but core held-out metrics do not improve materially |
| 4 | Local ConvSSM refiner replacing TCN | Equal-parameter post-TCN and scale-zero | State-space local path cannot beat the simpler finite-memory correction |
| 5 | Morphology-conditioned correction scale after promotion | Shared post-TCN scale with equal parameter count | Morphology disparity and intervention SERA do not improve |

## Recommendation

If I160 promotes a post-TCN, I161 should test residual-refiner optimization.
This is the closest literature-backed extension of the successful mechanism and
does not alter inference cost. Use one declared split, selected before seeing
results, and preserve the 120k total presentation budget.

If I160 rejects both post paths at 300k, do not run these refinements. Revert to
I140 and use the next scan to reconsider the model class rather than tuning a
short-budget artifact.

## Guardrails

- Do not sweep dilation, kernel size, TCN depth, or residual scale.
- Continue exact parameter, presentation, optimizer-step, data-order, and
  executable-path controls.
- Treat post-dilated versus post-undilated as a balanced frontier unless the
  long gate resolves it; do not select one from a single metric.
- Retain high-dV/dt, near-spike, residual-lag, voltage-gain, intervention, and
  morphology diagnostics. The current post-TCN improvement must not be called
  AP-specific unless those localized ratios improve.
- Promotion requires a fresh 300k winner and exact output identity between the
  selected run checkpoint and the reusable AxoMamba path.
