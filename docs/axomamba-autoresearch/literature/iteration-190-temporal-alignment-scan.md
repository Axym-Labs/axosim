# Iteration 190 literature scan: temporal alignment after raw adaptation

Date: 2026-07-13

## Question and evidence boundary

I180-I189 separate three failure modes. Raw-target adaptation supplies the only
large voltage gain (about 0.84 mV); stronger raw losses add a reproducible but
smaller 0.067-0.074 mV; a causal refractory state repeatedly improves tolerant
F1 but not exact timing. Hard event lifting, temporal bases, calibrated output
heads, onset sharpening, and direct dynamics-SERA training do not add practical
value. This scan therefore asks which structural mechanisms specifically align
sparse events in time without reopening rejected generic capacity or waveform
decoders.

The scan used primary 2024-2026 papers from arXiv, OpenReview, and PMLR. Work
was ranked only when its mechanism maps to causal 1 ms neuron simulation and
has a discriminating control available in AxoSim. Withdrawn or rejected 2026
submissions are treated as idea sources, not established evidence.

## Selected evidence

### Learn delays rather than only filtering amplitudes

[Learning Delays in Spiking Neural Networks using Dilated Convolutions with
Learnable Spacings](https://openreview.net/forum?id=4r2ybzJnmN) is an ICLR 2024
poster that learns sparse temporal offsets with DCLS and reports gains on SHD,
SSC, and speech commands. The mechanism differs from AxoSim's rejected dense
causal kernels: each channel learns when a signal arrives, not an arbitrary
waveform amplitude at every lag. AxoMamba's 0.28 timing-tolerance gap and low
exact F1 make a small fractional-delay bank at the spike readout a direct test.

[SpikingGamma](https://arxiv.org/abs/2602.01978) develops smoothed delay kernels
and recursive internal memory for temporally precise spiking without surrogate
gradients. Its broad SNN claims do not transfer automatically, but the smooth
delay parameterization is useful here: compare learned continuous delays with
fixed zero delay and an equal-size ordinary causal kernel. The event source
must remain predicted latent activity, never teacher spikes.

### Hybrid embeddings remain plausible, but explicit event coordinates failed

[Embedding Hybrid Systems into Continuous Latent Vector
Fields](https://openreview.net/forum?id=jk66MtkUS1) proves an extrinsic
continuous embedding result and uses latent/state consistency to learn hybrid
flows. I181 falsifies AxoSim's simplest explicit hard-event coordinate, so the
next faithful translation would be a higher-dimensional continuous quotient
embedding without thresholded mode labels. It is expensive and should follow,
not precede, a successful delay test.

[CHyLL](https://openreview.net/forum?id=y1G0X3cnPd) similarly glues the reset
surface into a continuous latent quotient without event segmentation. The
submission was withdrawn, so it motivates a falsifiable geometry rather than
supporting a performance expectation.

### Temporal self-prediction may stabilize event representations

[PredNext](https://openreview.net/forum?id=LjugJFmItY), an ICLR 2026 poster,
uses cross-view temporal prediction to improve deep SNN representation
consistency. Generic latent distillation already failed in AxoMamba. A relevant
variant would predict the next event-aligned latent from the current latent and
input, with shuffled-time and output-only controls; it should run only if
learned delays show that temporal alignment is a real bottleneck.

### Sparse-input identification suggests an upstream impulse view

[SpinSVAR](https://proceedings.mlr.press/v286/misiakos25a.html) gives a
structural autoregression estimator under sparse Laplacian inputs and reports
advantages when sparse shocks drive dynamics. AxoSim inputs are sparse signed
events, so a learned per-branch impulse-arrival operator before Mamba is more
justified than another post-readout correction. This is a fallback if output
delay alignment fails; controls must preserve the current route and compare
sparse delayed impulses with a dense temporal projection.

[System Identification of Neural Systems: If We Got It Right, Would We
Know?](https://proceedings.mlr.press/v202/han23d.html) cautions that functional
similarity often cannot identify architecture even in favorable synthetic
settings. For this loop, biological labels such as refractory or delay should
not count as evidence by themselves. They must change held-out timing,
intervention, or robustness metrics under controlled parameter/compute tests.

### Long-memory retrieval is not the present bottleneck

[Expansion Span](https://proceedings.mlr.press/v288/nunez25a.html) augments
hybrid SSMs with relevance-based retrieval beyond recent context. AxoSim's
I169 matched-compute continuity test and the fixed 500 ms window audit did not
support missing long context. Attention/retrieval is therefore explicitly
deprioritized despite its relevance to general SSM literature.

## Ranked experiments after I190

| rank | experiment | discriminating control | advancement rule |
| ---: | --- | --- | --- |
| 1 | Learned fractional delay bank on final spike features/logits | fixed zero delay, fixed integer delay, equal-size ordinary causal kernel | improve exact F1 and timing gap significantly with no SERA loss; no delay-count sweep |
| 2 | Sparse delayed branch-impulse operator before Mamba | current instantaneous route, dense temporal projection, shuffled delays | >=0.1 mV or large replicated intervention/timing gain with route identity preserved |
| 3 | Continuous higher-dimensional hybrid quotient embedding, no hard event label | equal-dimensional generic residual and shuffled consistency | >=0.1 mV or major multi-F1/intervention gain; only one dimension choice |
| 4 | Event-aligned next-latent prediction | output-only auxiliary and shuffled-time latent target | only after delay evidence; must beat generic distillation failures |
| 5 | Relevance retrieval/attention | none planned | deprioritized unless a new long-context audit reverses I169 |

## Guardrails

- Do not tune refractory duration, onset scale, raw-objective weights, dynamic-
  SERA weight, temporal-basis taus, or event threshold after I190.
- A learned delay must alter temporal alignment rather than emulate another
  dense waveform head. Match parameter count and executed interpolation.
- Use predicted states only at inference; teacher spikes may define losses but
  cannot select delays or modes in the scored forward pass.
- Preserve the 0.1 mV voltage gate. Smaller effects require a replicated,
  significant exact-F1, tolerant-F1, or intervention gain.
- Continue to treat biological resemblance as a hypothesis source, not a
  success metric independent of held-out behavior.

## Decision

I190 remains a confirmation and cleanup iteration. If its refractory effect
survives, the long winner may be retained while I191 tests learned delays as a
separate exact-timing mechanism. If refractory fails long, prune it and test
delays directly on the simpler raw-adapted stable model. Do not add attention,
retrieval, or another generic temporal head.
