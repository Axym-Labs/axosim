# Aggressive Architecture Scan After Iteration 116

Date: 2026-07-12
Decision target: structural changes capable of reducing a visible fraction of
the roughly 20 mV voltage error after distillation and small residual/gating
attachments failed.

## Decision

The next experiment should stop treating the 118 dendritic branches as a vector
that must be collapsed before temporal processing. The strongest directly
transferable idea is a **latent branch neural operator**: learn several spatial
branch mixtures, evolve each mixture causally through a shared Mamba transition,
then decode the set into the soma-level stream. This changes the model's state
factorization rather than adding another scalar correction to the existing
single stream.

Run both a residual form, which adds the decoded operator to the promoted branch
projection, and a replacement form, which forces the operator to carry the
branch signal. Include the promoted model and a genuine widened/deeper global
Mamba with comparable trainable capacity. Report parameter, measured latency,
memory, and presentation mismatches explicitly; they are separate explanatory
axes.

## Primary Evidence

### Latent Mamba Operator (LaMO), ICML 2025

LaMO combines a learned kernel-integral spatial representation with Mamba in a
compact latent domain. Its irregular-mesh implementation learns soft assignments
from input points to latent slices, applies a shared state-space operator to the
slices, and projects the processed slices back to input points. Across regular
grids, meshes, and point clouds, the paper reports a 32.3% average improvement
over the previous second-best solution-operator baseline.

Source: [ICML/PMLR paper](https://proceedings.mlr.press/v267/tiwari25a.html) and
[official implementation](https://github.com/M3RG-IITD/LaMO).

**Transfer to AxoMamba.** Branches are the spatial points and their routed
synaptic sums are the forcing field. Unlike the paper's mostly spatial Mamba
application, AxoMamba must remain causal in physical time, so each learned
latent branch mixture becomes its own shared-Mamba temporal stream. The
adaptation preserves the paper's kernel-slice-process-decode factorization but
uses the axis that matches this problem's dynamics.

### State Space Neural Operator (SS-NO), TMLR 2026

SS-NO extends structured state-space models to joint spatiotemporal operator
learning and introduces adaptive damping and learnable frequency modulation.
Its main relevance is independent support for retaining spatial structure
inside the state-space operator instead of aggregating space first and asking a
single temporal stream to recover it.

Source: [TMLR/OpenReview paper](https://openreview.net/forum?id=SwLxxz0x58).

**Transfer boundary.** Earlier AxoMamba frequency-loss and fixed-memory tests
already make generic damping/frequency addons low priority. The useful new
claim is the joint spatial-temporal state factorization.

### Sequential Neural Operator Transformer (S-NOT), 2025/2026

S-NOT replaces DeepONet's dot-product branch/trunk merger with an attention
mechanism after a recurrent encoding of sequential loads, improving nonlinear,
path-dependent surrogate prediction and outlier robustness.

Source: [arXiv paper](https://arxiv.org/abs/2507.03272).

**Transfer to AxoMamba.** The model should not reduce separately evolving
branch states through one fixed sum. Learned decoding across latent branch
states is the relevant component. Full noncausal self-attention over time is
not appropriate for this benchmark.

### Khatri-Rao Neural Operator (KRNO), ICML 2025

KRNO learns a continuous time-shift operator through nonstationary integral
transforms with nearly linear scaling and reports competitive or state-of-the-
art temporal and spatiotemporal forecasting.

Source: [ICML/PMLR paper](https://proceedings.mlr.press/v267/dama25a.html).

**Transfer to AxoMamba.** Learned nonstationary mixtures are a stronger prior
than one fixed global branch projection. A full KRNO port would introduce too
many simultaneous changes; the latent kernel projection is the attributable
first test.

### Structured dendritic connectivity, Nature Communications 2025

Chavlis and Poirazi show that fixed sparse dendrite-to-soma organization plus
restricted input sampling can improve robustness and parameter efficiency over
dense networks on their classification benchmarks.

Source: [Nature Communications article](https://www.nature.com/articles/s41467-025-56297-9).

**Transfer boundary.** Their tasks are not dynamical neuron simulation, so the
reported accuracy gains are not expected here. The structural result supports
preserving multiple restricted branch subspaces through computation rather
than immediately densifying them.

## Ranked Queue

| rank | experiment | expected leverage | decisive control |
| ---: | --- | --- | --- |
| 1 | learned latent branch-token Mamba operator | multiple branch mixtures retain independent causal state before soma aggregation | widened/deeper global Mamba with comparable parameters plus promoted control |
| 2 | activity-dependent token assignment and decoding | nonstationary branch cooperation changes with the current input regime | static-assignment operator with the same token/state path |
| 3 | alternating latent spatial and global temporal blocks | repeated forcing/state interaction can represent path-dependent nonlinear dynamics | same-depth global-only trunk and latency accounting |
| 4 | adaptive-damped latent state | localizes each latent state's temporal receptive field | inactive damping path with equal parameters/compute |
| 5 | intervention-equivariant response state | separates baseline transition from perturbation response | exposure-matched unconstrained latent state |

## Stopping Rule

Do not tune token count, latent width, damping, or assignment temperature after
a clean failure. A first run must show a material multi-axis movement, ideally
at least 0.05-0.10 mV SERA or a comparably clear F1 gain, before replication.
Tiny significant deltas do not qualify. If the latent operator is active but
fails, move to a different state factorization or data representation rather
than polishing its scalars.
