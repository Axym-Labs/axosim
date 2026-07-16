# Literature Scan After Iteration 084

Date: 2026-07-12
Scope: neural dynamical-system surrogates, ODE/operator simulation,
biophysical and spiking models, intervention-aware identification, and
time-series/state-space models.
Decision target: select mechanism-changing AxoMamba experiments for iterations
85 onward, under the existing 120k-presentation and held-out-diagnostic protocol.

## Executive Decision

The best immediately transferable idea is **multi-timescale filtered spike-train
supervision**, commonly represented by van Rossum-style distances. AxoMamba's
current F1 surrogate pools event probabilities inside tolerance windows, whereas
filtered spike-train losses directly penalize timing displacement and count
mismatch at declared time constants. This targets the actual baseline failures:
exact F1 `0.054`, 5 ms F1 `0.261`, count error `0.470`, and a tolerance gap of
`0.207`. It is parameter-neutral, cheap, and attributable. Iteration 85 should
compare causal filtered losses at 1, 2, and 5 ms plus a small multiscale blend
against the unchanged current objective.

The strongest architecture idea is a **genuinely parallel fast/slow Mamba
pathway**, with an exact parameter-budget match. It differs from the failed
pre-Mamba short/long convolutional mixer and failed decay initialization: each
path gets its own state evolution and temporal resolution, then the paths fuse
before the voltage and spike heads. This is supported independently by
multi-scale Mamba, parallel multi-compartment spiking models, and dual-memory
neuromorphic models. It should follow the filtered-loss test, because the latter
can expose whether architecture changes improve event timing rather than merely
voltage.

The scan does **not** support immediately building the proposed six-state BSSM.
The current data expose weak gain and event fidelity, but not a uniquely
biophysical latent-state failure. A bounded hybrid-current experiment remains
worthwhile later, after a cheap premise audit derives candidate current traces
from known excitatory/inhibitory inputs. Likewise, PDE rollout methods are not
directly transferable because AxoMamba predicts an entire trace from exogenous
input rather than recursively feeding its own output back as state.

## Search Protocol

### Inclusion criteria

- Primary papers or official proceedings pages, with preference for accepted
  work and direct empirical evidence.
- A mechanism that could alter temporal state representation, sparse-event
  fidelity, intervention response, robustness, or sample efficiency.
- A clear path to a controlled AxoMamba experiment under fixed data and
  diagnostics.
- Recent work through July 2026 plus older foundational papers where necessary.

### Exclusion or down-ranking criteria

- Pure benchmark claims without a transferable mechanism.
- Autoregressive-rollout remedies whose premise is absent in AxoMamba.
- Irregular-sampling methods when all AxoBench traces are regularly sampled.
- Spatial neural operators that require a mesh or resolution-generalization
  problem not present in the current branch-routed representation.
- Changes already rejected by the loop: derivative-to-spike coupling, generic
  short/long convolutional mixing, uniform or layer-separated Mamba decay
  initialization, recurrent second-pass refinement, output smoothing/gain,
  refractory inhibition, direct paired-response SERA, and intervention-data
  replacement.

### Queries

Searches were run on 2026-07-12 over arXiv, OpenReview, NeurIPS proceedings,
PMLR, Nature-family journals, eLife, and PubMed. Query families included:

- `neural operator dynamical systems stable long rollout surrogate simulation`
- `controlled neural differential equations intervention system identification`
- `biophysical neuron surrogate state space model Mamba neuron simulation`
- `Mamba time series forecasting multiscale frequency 2025 2026`
- `causal representation learning dynamical systems interventions time series`
- `differentiable spike train distance temporal precision loss`
- `multi-compartment spiking neuron multi-scale temporal processing`
- `hybrid biophysical neuron neural ODE learned current`

The scan also searched AxoSim's specifications and all 84 iteration records to
avoid proposing a renamed version of a completed experiment.

## Current Failure Surface

Iteration 84's ordinary-training control provides the decision baseline:

| Diagnostic | Value | Implication |
| --- | ---: | --- |
| voltage SERA | 17.381 mV | large relevance-weighted amplitude error remains |
| dynamics SERA | 21.339 mV/ms | transitions remain difficult |
| F1 area, 0-5 ms | 0.178 | weak event fidelity across tolerance |
| exact F1 | 0.054 | exact timing is especially poor |
| 5 ms F1 | 0.261 | tolerance only partly recovers matches |
| response SERA | 17.633 mV | intervention effects remain inaccurate |
| voltage gain | 0.237 | predicted trace substantially compresses teacher variation |
| residual lag-1 | 0.402 | errors retain temporal structure |
| near-spike voltage error | 4.523 | errors localize strongly around events |
| spike-count error | 0.470 | event timing is not the only spike failure |
| refractory violations | 0.232 | event readout lacks realistic local structure |
| intervention response skill | 0.0008 | almost no skill over the response null |
| intervention gain | 0.0057 | predicted perturbation responses are severely attenuated |

These diagnostics favor objectives and architectures that preserve multiple
timescales and event structure. They do not yet identify a particular ion
channel, conductance state, or autoregressive stability failure.

## Evidence Map

### 1. Biological-neuron surrogates

1. Beniaguev, Segev, and London showed that a temporal convolutional network
   could emulate a detailed layer-5 pyramidal neuron at millisecond resolution,
   and that removing NMDA receptors sharply reduced the required network
   complexity. This supports the premise that temporal architecture should track
   the teacher's nonlinear biophysics, but it does not specify an SSM design.
   [Neuron 2021](https://doi.org/10.1016/j.neuron.2021.07.002)
2. The ELM/Branch-ELM paper is the closest task-specific baseline. Its explicit,
   trainable memory timescales and nonlinear input integration achieve strong
   parameter efficiency on NeuronIO. AxoMamba must therefore improve temporal
   fidelity, not merely claim long-context capacity.
   [ICLR 2024](https://openreview.net/forum?id=vE1e1mLJ0U)
3. A CNN-LSTM surrogate has also reproduced somatic and dendritic voltages and
   calcium-current dynamics in detailed neurons. The important transfer is that
   jointly predicting internal observables can constrain a surrogate, provided
   those targets actually exist. AxoBench v1 currently exposes soma voltage and
   spikes, not channel-state labels.
   [eLife 2022](https://elifesciences.org/articles/79535)
4. A new hybrid-neuron study parameterizes unknown conductance dynamics through
   voltage-dependent steady-state and time-constant functions, recovering ion
   kinetics and a learned axial current while preserving mechanistic structure.
   This is stronger support for a small learned-current correction than for a
   fully hand-specified BSSM.
   [arXiv 2026](https://arxiv.org/abs/2606.16693)

**Transfer:** retain branch routing and explicit morphology identity; consider a
small causal current-state side path only if input-derived current proxies are
measurably related to the residuals. Do not invent unobserved gating targets.

### 2. Precise event objectives

5. The van Rossum distance compares spike trains after temporal filtering, so
   its time constant controls the tradeoff between exact timing and rate. It is
   differentiable when applied to continuous spike probabilities and naturally
   penalizes count and displacement together.
   [Neural Computation 2001](https://www.research.ed.ac.uk/en/publications/a-novel-spike-distance/)
6. SuperSpike operationalized this idea with a smooth, causal filtered spike
   error and surrogate gradients to train precise multi-spike temporal patterns.
   AxoMamba already outputs differentiable logits, so it can use the filtered
   target/prediction loss without adopting an SNN or surrogate-gradient cell.
   [Neural Computation 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/)
7. Adaptive spiking recurrent networks combine tunable temporal dynamics with
   activity regularization and report strong sparse time-domain performance.
   This supports monitoring count and refractory failures alongside timing, but
   activity regularization itself is not directly applicable to AxoMamba's
   non-spiking hidden states.
   [Nature Machine Intelligence 2021](https://www.nature.com/articles/s42256-021-00397-w)

**Transfer:** add causal exponential or double-exponential filtering to predicted
probabilities and binary targets at 1-5 ms. Keep BCE and the existing F1
surrogate fixed in the first test so the filtered term is independently
attributable. Evaluate exact F1, 5 ms F1, F1 area, count, and refractory errors.

### 3. Multi-timescale state and memory

8. HiPPO derives online memory as optimal polynomial projections of continuous
   history and provides timescale robustness with bounded gradients. S4 turns
   structured state-space dynamics into an efficient long-sequence layer.
   Mamba then makes state transitions input-selective.
   [HiPPO, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html),
   [S4, ICLR 2022](https://openreview.net/forum?id=uYLFoz1vlAC),
   [Mamba](https://arxiv.org/abs/2312.00752)
9. Multi-scale Mamba processes time series with parallel Mamba blocks at
   different sampling rates and reports gains over its single-scale Mamba
   baseline. The evidence is a preprint rather than a mature consensus result,
   but the mechanism is simple and testable.
   [ms-Mamba 2025](https://arxiv.org/abs/2504.07654)
10. Parallel multi-compartment spiking neurons diagonalize interacting linear
    compartment dynamics and process multiple timescales in parallel. This is
    architecture-level support for distinct state paths rather than merely
    changing the initial decay spectrum of one path.
    [PMSN 2024](https://arxiv.org/abs/2408.14917)
11. A 2026 dual-memory neuromorphic model combines fast sparse activity with a
    compact slow state that modulates spiking dynamics, reporting competitive
    long-sequence accuracy with fewer parameters. The hardware claims do not
    transfer, but the fast/slow state decomposition does.
    [Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-026-01255-3)
12. Closed-form continuous-time networks approximate liquid time-constant
    dynamics with explicit time-dependent gates, avoiding expensive numerical
    ODE solves. Their strongest relevance is input-dependent time constants,
    not the biological language around the cell.
    [Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00556-7)

**Transfer:** create two parameter-budgeted Mamba state paths with distinct
effective temporal resolutions or constrained delta multipliers. This is not
the rejected pre-Mamba temporal convolution and not the rejected initialization
of the same Mamba blocks. Fuse paths once, before existing heads, and preserve
causality.

### 4. Latent and continuous-time dynamics

13. Latent ODEs and neural CDEs model continuous latent trajectories; CDEs let
    subsequent observations continuously drive state. These are important for
    irregularly sampled, partially observed series.
    [Latent ODE, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html),
    [Neural CDE, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)
14. Neural delay differential equations augment partially observed systems with
    history and analyze stability. AxoMamba's residual lag suggests unresolved
    history, but Mamba already has latent memory and the benchmark is regularly
    sampled, so a solver-based replacement is not justified.
    [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/6332a8f62e3a9d5831724f2ffe55cae0-Abstract.html)
15. Stabilized neural differential equations enforce known manifold constraints
    through an asymptotically stabilizing term. AxoBench currently has only
    output validity bounds, not a known latent-state manifold, making this a
    premise rather than an experiment.
    [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2a4179ef39846557e99f6bfac580ea2e-Abstract-Conference.html)
16. ICML 2026's latent time-series forecasting work argues that observation-only
    regression can produce temporally disordered latent states and adds latent
    forecasting structure. Its reported setting is forecasting, but the
    transferable question is whether neighboring AxoMamba states are smooth and
    intervention-sensitive in ways the readout currently ignores.
    [ICML 2026](https://openreview.net/forum?id=s49fw3BVU0)

**Transfer:** first audit latent-state temporal continuity, effective rank, and
linear decodability of teacher voltage/input currents. Add a latent objective
only if the audit reveals a concrete defect. Continuous-time solvers are low
priority because fixed 1 ms sampling removes their main advantage.

### 5. Neural operators and learned simulation

17. Fourier neural operators learn mappings between function spaces and can
    generalize across PDE instances and resolutions. AxoMamba already maps an
    input function to an output function, but its channels are routed synapses,
    not a translation-invariant spatial grid.
    [ICLR 2021](https://openreview.net/forum?id=c8P9NQVtmnO)
18. Graph-network simulators found that noise corruption and message-passing
    depth mattered for long rollouts. Message-passing PDE solvers formulate
    autoregressive stability as domain adaptation with a pushforward trick.
    [ICML 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html),
    [ICLR 2022](https://openreview.net/forum?id=vSix3HPYKSU)
19. PDE-Refiner attributes long-rollout failures to neglected non-dominant
    frequencies and uses iterative denoising/refinement. SGNO instead constrains
    a spectral generator to avoid amplifying modes.
    [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html),
    [SGNO 2026](https://arxiv.org/abs/2602.18801)

**Transfer:** frequency-aware error analysis is useful, especially separating
near-spike high-frequency errors from subthreshold low-frequency gain. Iterative
denoising, pushforward training, and generator-stability constraints should not
be imported now: AxoMamba is causal sequence-to-sequence and does not feed its
predicted voltage back into the next step.

### 6. Interventions and causal dynamics

20. Interventional causal representation learning shows that interventions can
    identify latent factors under assumptions about intervention structure.
    Work connecting causal representation learning to dynamical systems isolates
    trajectory-specific parameters to form controllable models.
    [Ahuja et al. 2022](https://arxiv.org/abs/2209.11924),
    [Yao et al. 2024](https://arxiv.org/abs/2405.13888)
21. Neural pharmacodynamic state-space models structure how treatments affect
    latent disease state, improving generalization and interpretability in their
    clinical setting. The transferable principle is explicit action/intervention
    conditioning, not paired-response regression alone.
    [ICML 2021](https://proceedings.mlr.press/v139/hussain21a.html)

**Transfer:** intervention type could become an explicit context variable only
when it is available at inference. In the current benchmark, interventions alter
the input itself, and the model should infer their effect from that input.
Iteration 81-84 showed that replacement and paired-response objectives are
seed-sensitive. Do not add intervention labels that would leak the evaluation
condition or weaken the intended input-output test.

### 7. Sparse biological computation and control

22. Neural circuit policies use sparse, structured continuous-time networks for
    compact and auditable control. Their robustness result concerns learned
    policies with a perception encoder, not single-neuron simulation.
    [Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-020-00237-3)
23. Multi-compartment spiking world models report benefits from dendritic
    integration on control and long-sequence tasks, but the work is presently a
    preprint and combines several changes. It supports compartmental state as a
    hypothesis, not as evidence that AxoMamba should become an SNN.
    [Spiking-WM 2025](https://arxiv.org/abs/2503.00713)

**Transfer:** sparse wiring and compartmental state should be judged by AxoBench
timing, intervention, and efficiency metrics. The earlier external-ML experiments
already failed to establish general ML utility for pretrained AxoMamba, so this
loop should optimize the biological-neuron surrogate rather than reopen control.

## Ranked Experiment Queue

| Rank | Experiment | Falsifiable prediction | Required controls | Cost/risk |
| ---: | --- | --- | --- | --- |
| 1 | Multi-timescale filtered spike-train loss | Improves F1 area and either exact or 5 ms F1 without significant voltage/response regression; count error should not worsen | same model, seed, schedule, updates; weight-zero control; profile loss time | low |
| 2 | Exact-budget parallel fast/slow Mamba | Reduces residual lag-1 and near-spike/high-dVdt error while improving at least one core SERA/F1 axis | equal parameters within 1%; equal presentations/steps; single-path width-matched controls | medium-high |
| 3 | Input-conditioned timescale modulation | Improves both ordinary and intervention SERA by adapting state decay to sparse input bursts | parameter-matched static-timescale control; inspect delta distributions | medium |
| 4 | Frequency-banded voltage objective | Improves voltage gain and near-spike error without the F1 loss seen under scalar output gain | fixed low/high-pass kernels; matched loss compute or measured overhead | low-medium |
| 5 | Latent-state audit, then continuity/decodability loss only if warranted | Audit reveals state disorder or missing E/I information; corrective loss improves residual lag and response | predeclared audit threshold; no objective if premise fails | low audit, medium experiment |
| 6 | Hybrid fast/slow E/I current side path | Improves intervention gain and response SERA because dropout effects act through separable causal currents | input-derived targets only; exact-budget Mamba control; state interpretability tests | high |
| 7 | Controlled spectral/delta parameterization | Constraining a subset of state decays prevents attenuation and improves voltage gain | compare to learned unconstrained and failed init-only paths | medium |
| 8 | Noise/denoising training | Improves robustness only if train-test state corruption or measurement noise is demonstrated | premise audit first; no autoregressive rationale | medium, low current priority |

## Iteration 85 Design

**Premise.** The current F1 surrogate is a set-overlap approximation over pooled
windows. It provides no graded penalty for a spike moving from 1 to 5 ms away
inside a tolerance and can reward broad probability mass. A filtered spike-train
distance gives dense gradients whose temporal scale is explicit and whose
integrated amplitude reflects spike count.

**Candidates.** Keep the promoted BCE/SERA/F1 objective unchanged and add a
normalized causal filtered-probability MSE using:

- single exponential time constant 1 ms;
- 2 ms;
- 5 ms;
- equal blend of 1, 2, and 5 ms.

Use one conservative coefficient selected by matching the initial gradient norm
to a small fraction of the existing F1 term, rather than sweeping arbitrary loss
weights in the same iteration. Include a weight-zero control through the same
compute path if filtering overhead is material; otherwise the unchanged control
is sufficient.

**Promotion gate.** Require no significant regression in voltage SERA or
response SERA and improvement in F1 area plus either exact or 5 ms F1. Treat
lower count error and fewer refractory violations as supporting evidence. A
candidate that improves only 5 ms F1 while further lowering exact F1 is not a
balanced promotion.

## Iterations 86-90 Sketch

- **I86:** replicate the best filtered loss if it produces a material lead;
  otherwise close it after one bounded coefficient correction justified by
  measured gradient scale.
- **I87-I89:** implement and test the exact-budget parallel fast/slow Mamba
  path. Start with two paths and fixed relative temporal scales; test adaptive
  scales only if the fixed decomposition works.
- **I90:** run 300k only on a replicated lead. In parallel, perform the mandatory
  cleanup. Delete rejected filtered-loss or multiscale paths, direct
  response-SERA code, and unused intervention replacement machinery. Remove
  duplicate assignments and stale configuration fields, then run full AxoSim,
  AxoBench evaluator, and stable-checkpoint-load verification.

The next literature scan is scheduled for iteration 100 and every tenth
iteration thereafter.

## Limits

- Several 2025-2026 sources are preprints; they motivate bounded tests rather
  than establish settled results.
- Published gains on forecasting, PDEs, control, and SNN classification do not
  estimate effect size on AxoBench.
- AxoBench's fixed 1 ms sampling weakens the case for ODE/CDE solvers.
- The current evaluator has only output diagnostics. Claims about interpretable
  conductance or compartment states require new, independently justified tests.
- The intervention dataset changes the input distribution but does not expose a
  privileged intervention variable to the model. Intervention-conditioned
  architectures would change the task and are excluded.
