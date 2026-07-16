# Iteration 180 literature scan: raw voltage and hybrid dynamics

Date: 2026-07-13

## Question and scope

The decisive I173-I179 result is not another architecture tweak: AxoMamba was
trained on a soma trace clipped above -55 mV while current diagnostics score
raw voltage. Correcting that target after biological pretraining improves raw
voltage SERA by about 0.79 mV. This scan asks which structural representations
could improve the remaining sharp AP transitions without reopening failed
soft waveform heads, generic extra capacity, or low-impact scalar tuning.

The search covered 2025-2026 primary papers on neural operators for sharp
transitions, hybrid dynamical-system embeddings, electrophysiology surrogates,
and target-domain adaptation, plus older primary work where it defined a
distinct relevant model class. Searches were run on 2026-07-13 across arXiv,
OpenReview, and PMLR. Papers were included only when their mechanism translated
to causal input-output simulation; generic PDE accuracy papers were excluded.

## Selected evidence

### Represent the event boundary explicitly

[Cut-DeepONet](https://arxiv.org/abs/2605.19823) argues that continuous neural
operators struggle with sharp transitions and instead lifts the problem into
smooth subregions separated by an input-dependent predicted discontinuity.
Its reported benefit comes from changing the representation rather than simply
adding resolution or capacity. For AxoMamba, the nearest faithful translation
is a causal hard event/phase coordinate from the spike head, followed by a
small raw-voltage jump decoder. I176's soft sigmoid waveform does not test this
because it never creates a discrete boundary coordinate.

[Shearlet Neural Operators](https://arxiv.org/abs/2604.25181) use localized,
multiscale atoms for shock-dominated fields and report their largest gains on
sharp, localized structures. Directional spatial shearlets do not transfer to
a one-dimensional neuron trace, but localized event atoms do: a compact bank
indexed by causal phase since an event is more justified than another global
frequency head.

### Hybrid systems can become smooth in a lifted latent space

[Embedding Hybrid Systems into Continuous Latent Vector Fields](https://openreview.net/forum?id=jk66MtkUS1)
shows that discontinuous hybrid flows can admit a continuous higher-dimensional
embedding and trains with consistency in latent and state spaces. The useful
implication is not to add a generic latent loss, already rejected by AxoMamba,
but to make the event/reset coordinate explicit before enforcing consistency.
An unconstrained latent distillation objective does not test this premise.

[LatSegODE](https://proceedings.mlr.press/v139/shi21c.html) likewise treats
smooth flows and changepoints separately. Offline PELT segmentation is not
causal and therefore unsuitable here, but its successful decomposition
supports using known spike-event supervision to define the transition during
training and a predicted event coordinate at inference.

### Pretraining then adapting the target is a principled regime

[Domain Adaptation with Hybrid Modeling for Learning Dynamical Systems](https://proceedings.mlr.press/v304/osaka26a.html)
reports that retaining invariant dynamics and fine-tuning to a shifted target
domain can improve adaptation efficiency. This matches I175/I179: the clipped
model supplies mature subthreshold/event features, then raw-target adaptation
changes the output contract. The paper does not establish AxoMamba's result,
but it supports treating adaptation as the primary recipe rather than
discarding the pretrained checkpoint and retraining raw from scratch.

### Transform-domain dynamics remain a distinct fallback

[Neural Laplace](https://proceedings.mlr.press/v162/holt22a.html) represents
history dependence, stiff dynamics, and discontinuities as sums of complex
exponentials in the Laplace domain. This is materially different from I170's
one-step high-pass readout. A bounded causal exponential-basis residual could
be tested after event lifting, but only with an equal-basis raw-feature control
and the 0.1 mV gate.

[Latent Dynamics Networks for cardiac electrophysiology](https://arxiv.org/abs/2605.01463)
provide current evidence that compact latent dynamics can serve as efficient
electrophysiology forward surrogates. Their inverse-problem setting and cardiac
scale differ substantially from single-neuron AP prediction, so they motivate
the broad latent-dynamics class but no direct architecture choice.

## Ranked experiments after I180

| rank | experiment | discriminating control | advancement rule |
| ---: | --- | --- | --- |
| 1 | Causal hard event lifting: straight-through event indicator plus phase-since-event atoms feeding a raw-voltage jump decoder | same decoder driven by soft probability and scale zero; equal parameters and execution | >=0.1 mV voltage gain with no F1 regression and a clear near-spike/high-dV/dt improvement |
| 2 | Event-lifted continuous latent consistency: predict a smooth clipped state and a jump state in a shared lifted representation | existing output decomposition with no event coordinate and shuffled event coordinate | >=0.1 mV or >=0.05 mV plus a replicated large F1/intervention gain |
| 3 | Causal Laplace/exponential residual basis over the completed raw-adapted latent | equal-size ordinary causal-convolution basis and scale zero | >=0.1 mV and improved residual lag/high-frequency gain |
| 4 | Target-adaptation schedule only if I180 exposes long-run F1 loss: clipped-to-raw continuation versus constant raw, same total updates | constant half/full raw from the same checkpoint | recover F1 while retaining at least 80% of full-raw SERA gain |
| 5 | Higher-dimensional hybrid latent flow | same-dimensional generic latent expansion and event-shuffled consistency | only after the small event-lifting premise passes |

## Guardrails

- Do not tune raw-target mix after the controlled half/full long gate; treat it
  as a declared task frontier.
- Do not revive soft spike waveforms, generic soma TCN capacity, residual target
  assignment, or latent distillation under new names. I176-I178 resolve those.
- A hard event coordinate must be causal at inference. Teacher spikes may
  supervise its estimator but may not enter the forward path used in
  diagnostics.
- Match event-decoder parameters and execution in soft, hard, shuffled, and
  scale-zero controls. Report refractory/count failures explicitly.
- Preserve the 0.1 mV practical gate. Event F1 alone can justify one long arm,
  not an open-ended architecture family.

## Decision

I180 should remain a confirmation and cleanup iteration, not absorb a new idea.
If full raw survives 300k, promote the raw-adaptation recipe. Start I181 with
hard event lifting because it most directly targets the newly visible AP
discontinuity and is not falsified by I176's soft waveform failure. If full raw
fails long, use half raw and test an adaptation schedule before adding event
structure.
