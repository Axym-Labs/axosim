# Literature Scan After Iteration 140

Date: 2026-07-12

## Question and Scope

Which recent structural ideas from neural operators, dynamical-system
surrogates, graph simulation, and dendritic computation can exploit I131-I139's
robust route-occupancy result and plausibly improve AxoMamba by tenths of a mV,
rather than producing another small scalar tradeoff?

The scan prioritizes 2024-2026 primary papers with an architecture and an
empirical dynamical-system result. It excludes methods whose main advantage is
irregular sampling, autoregressive rollout stability, resolution transfer, or
uncertainty, because AxoBench uses fixed 1 ms sampling, direct 500-step windows,
and a fixed input discretization. Ideas already falsified in I101-I139 are not
reintroduced under new names.

## Search Method

Searched arXiv and OpenReview on 2026-07-12 using combinations of:

- `2025 2026 neural operator dynamical systems conditional geometry graph`
- `state space neural operator time dependent causal`
- `dual path local global neural operator dynamics`
- `neuron morphology surrogate graph neural network`
- `predictor corrector neural operator 2026`
- `graph simulator numerical integrator time dependent PDE`

The screen retained work that changes spatial/temporal factorization, local and
global fusion, explicit forcing/state evolution, or structured sparse sampling.
Claims below are limited to each paper's studied domain; transfer to a neuron
surrogate is a hypothesis, not evidence already established by the paper.

## Current Evidence Filter

The local experiments impose unusually strong constraints on transfer:

- Real route features beat shuffled and scale-zero controls on three seeds by
  roughly 0.20 mV voltage SERA and 0.08-0.10 tolerant F1.
- I139 shows that occupancy, multiplicity, and target length carry essentially
  the entire effect. Continuous source/target arclength behaves like shuffled
  features.
- Hard route masking fails, so disconnected sources cannot simply be deleted.
- Replacing generic routing with exact physical target routing is unstable
  across seeds, while shuffled physical routing fails catastrophically.
- A tiny additive graph event operator, branch-local recurrence, latent spatial
  tokens, generic extra depth, morphology LoRA, and simple output-state/change
  corrections have all failed balanced replication.

The next architecture must therefore preserve the generic path, add morphology
locally as a learnable residual, and include controls that distinguish real
route information from extra capacity.

## Selected Evidence

### Dual-path operators

[DPNO](https://arxiv.org/abs/2507.12719) organizes each neural-operator block
into parallel residual- and dense-style paths. It reports gains across DeepONet
and FNO implementations and improvements exceeding 30% on some PDE cases. The
important transferable claim is modest: parallel complementary operators can
be more effective than stacking another copy of one operator.

[DSO](https://arxiv.org/abs/2603.26800) explicitly separates local detail from
global evolution using a depthwise local path and an MLP-Mixer global path. The
authors motivate the split with perturbation experiments and report large
long-horizon gains on fluid dynamics. AxoMamba has an analogous empirical split:
real morphology correspondence is locally necessary, but generic global routing
is the more stable predictor.

**AxoMamba implication.** I135 replaced one route with the other. It did not
test a generic global route plus a separately projected physical-route residual.
A dual-route architecture is therefore a new intervention, not a softened
coefficient retest.

### Space-time factorization and graph integration

[State Space Neural Operator](https://arxiv.org/abs/2507.23428) extends SSMs to
joint spatiotemporal operators and reports strong PDE performance with adaptive
damping, frequency modulation, and a factorized spatial/temporal variant.
[PhyMPGN](https://arxiv.org/abs/2410.01337) places graph message passing inside
a numerical integrator and adds a learnable Laplace-Beltrami block on irregular
meshes. A 2025 controlled study of
[graph simulators versus neural operators](https://arxiv.org/abs/2509.06154)
reports substantially better low-data and rollout behavior when local graph
dynamics and explicit time stepping replace a purely global operator.

**AxoMamba implication.** These papers support preserving a local morphology
state through a temporal update before fusion. They do not rescue I132's
two-scalar event-message residual, which had neither a local state nor a
separate residual route. However, I101's branch-local Mamba and I117's latent
tokens warn against a large independent local trunk. Start with a small shared
causal local operator on physical-route branch currents and force it to prove
value against an equally sized generic-route local control.

### Predictor-corrector state

[CoEvol-NO](https://openreview.net/forum?id=pG9Wn3Aheq) uses persistent state and
an error-driven predictor-corrector update while co-evolving state and spatial
coordinates. It was selected as an ICML 2026 spotlight. A related 2026
[Neural CDE corrector](https://openreview.net/forum?id=biZBdFpOzu) learns errors
of an existing time-series predictor and reports improvements across several
predictor families.

**AxoMamba implication.** A second-pass correction conditioned on the first-pass
prediction error proxy is more expressive than I137's leaky additive voltage
forcing. It is still second priority because AxoBench supplies no intermediate
teacher state and a learned self-error proxy may collapse into generic depth.
The control must execute the same second pass without route conditioning.

### Explicit history transport

[HS-FNO](https://arxiv.org/abs/2605.09523) treats non-Markovian dynamics in a
history space and exactly shift-appends the known part of the next state rather
than relearning deterministic history transport. It reports a large aggregate
rollout-error reduction over current-state and unconstrained history baselines.

**AxoMamba implication.** This is elegant but low priority here. AxoMamba already
receives the complete 500 ms forcing history and emits the complete target
window without autoregressive state transport. A shift register would duplicate
known input coordinates unless a residual-information audit first shows that
the Mamba state forgets specific lags.

### Structured dendritic sampling

[Chavlis and Poirazi](https://www.nature.com/articles/s41467-025-56297-9)
show that sparse dendritic connectivity and restricted input sampling can
improve parameter efficiency and robustness in classification. Their masks
separate input-to-dendrite synapses from dendrite-to-soma cable weights.

**AxoMamba implication.** The biological result supports two-stage sparse
integration, but I133 shows a hard biological mask is wrong for this surrogate.
Use real routing as a soft residual branch and retain a learned cable fusion;
do not zero nominally unused source channels.

## Ranked Experiments

| Rank | Structural test | Discriminating controls | Pass condition |
| ---: | --- | --- | --- |
| 1 | Dual-route local/global operator: current generic branch path plus a separately projected exact physical-route residual, fused before the existing Mamba trunk | scale-zero physical path; shuffled physical route; equally sized generic-route residual; real route; common parameters and compute | real route improves voltage, response, and F1 materially over all three controls on two seeds; target at least 0.10 mV voltage or a comparably large multi-axis gain |
| 2 | Route-local causal operator before fusion: small shared depthwise temporal block over physical branch currents plus unchanged global Mamba | instantaneous dual-route control; generic-route local operator; shuffled route | temporal local path beats instantaneous real-route fusion, not merely shuffled routing |
| 3 | Route-conditioned predictor-corrector: first-pass global prediction and a second hidden correction pass gated by physical-route state | same two-pass unconditioned corrector; shuffled route; correction scale zero | correction improves dynamics/response and voltage without tolerant-F1 loss on two seeds |
| 4 | Learned local diffusion operator over the SWC tree inside the physical residual path | degree-matched rewired tree and no-message controls; use more than I132's two scalar terms | real graph beats rewired graph by at least 0.05 mV or a material intervention-response gain after the residual route itself passes |
| 5 | Exact history-state transport | residual-lag premise audit first; same parameter/compute lag-stack control | only run if past inputs retain conditional target information absent from current hidden state |

## Immediate Decision

After I140, implement rank 1 as a true 2 x 2-plus-shuffled premise test:

1. current generic path with physical residual scale zero;
2. duplicate generic route through the residual path, controlling capacity;
3. real physical residual route;
4. shuffled physical residual route;
5. optionally real occupancy-conditioned physical residual, testing whether the
   I139 feature prior and physical branch organization compose.

Initialize the residual projection to zero so all arms start output-identical,
but execute and optimize identical tensors in every arm. Do not tune fusion
scale before real routing beats both generic-capacity and shuffled controls.
If it fails on two seeds, close local/global route fusion and move to the
predictor-corrector premise rather than narrowing coefficients.

## Limits

Most selected evidence concerns PDE fields or classification, not detailed
single-neuron electrophysiology. Reported percentage gains are not expected
effect sizes for AxoMamba. The scan identifies architectural analogies and
controlled hypotheses; only AxoBench's paired diagnostics can establish value.
