# Iteration 150 literature scan: multiscale and hybrid dynamics

Date: 2026-07-12

## Question

Which structural temporal mechanisms remain plausible after I141-I149 rejected
route residuals, fixed forcing bases, predictor-correctors, route moments,
morphology readout experts, and task-specific temporal heads?

The scan emphasizes mechanisms that could move a visible fraction of roughly
20 mV SERA. It deliberately deprioritizes coefficient tuning, another static
route feature, and another additive one-layer MLP.

## Evidence

### Separate resolved and unresolved timescales

Learning Stochastic Multiscale Models represents a dynamical system with an
explicit coarse macroscale state and a latent microscale state. At matched
resolution it reports better predictive accuracy than under-resolved direct
simulation, closure models, and reduced-order alternatives. The transferable
idea is structural scale separation, not its stochastic variational objective.

Source: [Ilersich and Nair, 2025](https://arxiv.org/abs/2506.22655).

Temporal dendritic heterogeneity reaches the same design principle from
neuroscience: distinct dendritic branches learn distinct timing factors, then
their states are integrated. It reports gains in accuracy, robustness, and
compactness across temporal benchmarks. Our fixed trace families did not test
this premise because they filtered inputs before one common temporal state;
they did not maintain learned fast and slow latent systems.

Source: [Zheng et al., Nature Communications 2024](https://www.nature.com/articles/s41467-023-44614-z).

The 2025 observation that individual prefrontal neurons multiplex slow and fast
dynamics is biological support for coexisting timescales rather than a claim
about surrogate performance.

Source: [Hashemirad et al., Nature Communications 2025](https://www.nature.com/articles/s41467-025-66081-4).

### Local and global operator paths

MambaNO combines global Mamba integration with a local alias-free path and
reports strong accuracy/efficiency on PDEs including multiscale solutions. The
useful distinction is local versus global evolution before fusion. Our failed
post-hoc forcing and route residuals did not instantiate a full local temporal
operator with an equally expressive global control.

Source: [Zheng et al., NeurIPS 2024](https://openreview.net/forum?id=gUEBXGV8JM).

### Hybrid regimes rather than task-specific heads

Neural Hybrid Automata model continuous flows and discrete event-triggered
mode transitions separately. Graph Switching Dynamical Systems similarly uses
different dynamical modes and learns switching behavior. A neuron trajectory
has an unusually direct candidate partition: ordinary subthreshold evolution
versus rapid depolarization/spike/reset neighborhoods. I149 split by output task
after the shared dynamics; it did not split the dynamics by state regime.

Sources: [Poli et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html),
[Liu et al., ICML 2023](https://proceedings.mlr.press/v202/liu23z.html).

Latent Segmented ODEs further shows that a single smooth latent flow is a poor
fit to trajectories with jumps and changing modes. Its offline changepoint
procedure is not causal enough for AxoMamba, but its failure model is relevant.

Source: [Shi and Morris, ICML 2021](https://proceedings.mlr.press/v139/shi21c.html).

### Mechanistic backbone plus learned correction

Hybrid dynamical models can preserve a known low-dimensional mechanism while a
neural component learns the residual. Recent domain-adaptation evidence shows
that even partly specified physics can improve robustness and sample
efficiency. Hybrid causal ODE work warns that an unconstrained neural residual
can erase the mechanistic semantics, so intervention diagnostics are essential.

Sources: [Osaka et al., ACML 2025](https://proceedings.mlr.press/v304/osaka26a.html),
[Zou et al., ICML 2024](https://proceedings.mlr.press/v235/zou24b.html).

For AxoMamba, the concrete backbone is a causal leaky membrane state driven by
the learned branch current, with Mamba learning the nonlinear residual. This is
more faithful than I145's predictor-corrector, which predicted a hidden forcing
error but did not expose or integrate a voltage state.

### Spatial and temporal factorization

Multi-stage state-space motion forecasting separately models temporal and
spatial interactions before hierarchical decoding. Its venue status is weaker
than the evidence above, so it is supporting rather than decisive evidence.
The design nevertheless reinforces that simply widening one state stream is
not equivalent to factorizing spatial and temporal operators.

Source: [MS3M, 2025 submission](https://openreview.net/forum?id=MmOQY71YHw).

## Ranked experiments

| Rank | Experiment | Exact control | Falsifying result |
|---|---|---|---|
| 1 | Parallel fast/slow latent Mamba: full-rate path plus causally pooled slow path, fused before the stable heads | Two full-rate paths with identical blocks/parameters and matched execution; scale-zero path | No material gain over both controls, or improvement confined to dynamics SERA while voltage/response regress |
| 2 | Regime-gated Mamba experts: subthreshold and event-transition temporal blocks mixed by a causal gate from branch drive and shared state | Mean/shared mixture with identical experts and gate compute; cyclic or detached gate as identity control | Gate remains constant, modes do not align with high-dV/dt/near-spike diagnostics, or SERA/F1 trade off |
| 3 | Explicit leaky membrane backbone plus learned Mamba residual | Same state integrator with shuffled/constant drive and an equal-parameter generic residual | Intervention response or voltage gain does not improve materially, or learned residual overwhelms backbone |
| 4 | MambaNO-style local/global block inside the trunk | Duplicate global blocks with matched parameters and receptive-field compute | Local path does not preferentially reduce high-dV/dt and near-spike error |
| 5 | Morphology-conditioned timescale modulation of the winning multiscale model | Shared timescales with identical parameter budget | Morphology disparity and intervention SERA do not improve |

## Recommendation

I151 should test rank 1 if I150 confirms that temporal depth is useful. The
full-rate/full-rate control is mandatory: without it, a fast/slow win is
confounded by adding a second Mamba block. Use one aggressive downsampling
factor selected from the 500-step window and observed autocorrelation, not a
factor sweep.

If I150 rejects extra temporal depth, skip rank 1 and test rank 3. A failed
deep temporal gate would make another two-path Mamba experiment low-value,
whereas an explicit voltage-state backbone changes the model class.

Regime gating is second rather than first because I149 already exposes a
voltage-response frontier. It becomes attractive after a temporal-depth win,
when the gate can allocate demonstrated capacity by trajectory regime instead
of by output task.

## Guardrails

- Preserve I140's occupancy operator and objective unchanged for the first test.
- Match parameters, presentations, optimizer steps, data order, and executable
  paths; record the unavoidable latency difference of causal pooling.
- Require at least 0.05 mV voltage improvement or an equally strong balanced
  SERA/F1/intervention gain to replicate. The target remains substantially
  larger, but 0.05 mV is the minimum evidence threshold established by I149.
- Diagnose high-dV/dt, near-spike, residual lag, voltage gain, intervention
  direction/gain, and morphology disparity before attributing mechanism.
- Do not tune pooling factor, gate temperature, or membrane time constant after
  a clean premise failure.
