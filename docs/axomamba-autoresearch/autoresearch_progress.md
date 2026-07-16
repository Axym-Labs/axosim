# AxoMamba Autoresearch Progress

The user-set hard stop is iteration 200. Iteration 200 includes the scheduled
300k gate, literature scan, dead-code cleanup, full verification, and final
result synthesis; no iteration 201 is launched.

This is the append-only human-readable experiment ledger. Machine-readable
diagnostics live under `artifacts/diagnostics/`; run checkpoints and training
metrics live under the ignored `runs/axomamba-autoresearch/` tree.

From iteration 164 onward, the practical advancement bar is at least 0.1 mV
voltage SERA unless a candidate supplies a major non-voltage or mechanistic
gain. Smaller statistically significant changes remain logged but do not earn
follow-up tuning or promotion on metrics alone; the search prioritizes
operator-class and capacity-allocation changes plausibly capable of 0.1-1 mV.

## Pre-loop Controls

### Batch-size comparison - 2026-07-11

- Regime: blind downstream comparison; batch size changes optimizer-step count
  and is judged directly on held-out core metrics.
- Controls: identical initialization, frozen 120,000-window schedule, exactly
  24,000 presentations per morphology, constant 3e-4 AdamW, and one shared
  loader/GPU transfer. Parameter count was 74,188 for every candidate.

| batch | updates | compute s | voltage SERA | dynamics SERA | F1 area | F1 exact | F1 5 ms | intervention SERA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7,500 | 26.42 | 17.8537 | 21.3380 | 0.10572 | 0.06146 | 0.13843 | 17.7023 |
| 32 | 3,750 | 12.40 | 18.0336 | 21.3401 | 0.09927 | 0.05003 | 0.12871 | 17.6995 |
| 64 | 1,875 | 10.59 | 17.8563 | 21.3414 | 0.08685 | 0.03529 | 0.12180 | 17.7353 |

- Interpretation: batch 64 saves little compute beyond 32 and significantly
  regresses all F1 metrics. Batch 32 halves batch-16 compute and update count;
  it regresses voltage SERA by 0.180 mV and exact F1 by 0.0114, while F1 area
  and 5 ms F1 are uncertain and intervention SERA improves by 0.00282 mV.
- Decision: use batch 32 as the iteration-latency compromise requested by the
  user. Keep the measured degradation visible; batch 16 remains the quality
  reference and batch 64 is rejected.
- Artifacts: `artifacts/diagnostics/iteration-000-preloop-batch-size/` and
  `runs/axomamba-autoresearch/iteration-000-preloop-batch-size/`.

### Current-stack addon ablation - 2026-07-11

- Regime: controlled one-at-a-time removal from the promoted AxoMamba stack.
- Controls: batch 32, 3,750 updates, same frozen 120k schedule, common matching
  parameter initialization, same objective and diagnostics.

| removal | params | delta voltage SERA | delta dynamics SERA | delta F1 area | delta F1 exact | delta F1 5 ms | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| branch gain | 74,143 | +0.00015 | -0.000002 | +0.00002 | +0.00028 | -0.00034 | +0.00006 | remove; inert |
| branch tanh | 74,098 | +0.00087 | +0.00004 | -0.00486 | -0.00119 | -0.00532 | +0.00084 | retain |
| integrative mixer | 69,132 | -0.01696 | -0.00101 | -0.00228 | +0.00007 | -0.00188 | -0.00643 | remove; SERA win |
| soma gate | 74,057 | +0.04839 | -0.00187 | +0.00660 | +0.00256 | +0.01153 | +0.00221 | retain; real tradeoff |

- Interpretation: branch gain has no measurable effect. The integrative mixer
  significantly worsens all three SERA metrics and does not significantly help
  F1. Branch tanh protects F1. The soma gate improves voltage/intervention SERA
  while hurting F1, so the SERA-first policy retains it for now.
- Decision: promote AxoMamba with branch tanh and soma gate, but without branch
  gain or the integrative mixer. Validate the combined stripped control inside
  iteration 1 before attributing objective gains.
- Artifacts: `artifacts/diagnostics/iteration-000-preloop-addon-ablation/` and
  `runs/axomamba-autoresearch/iteration-000-preloop-addon-ablation/`.

### Iteration 001 - 2026-07-12

- Regime: direct downstream objective alignment. The hypothesis was that
  differentiable root-SERA and tolerance-curve F1 losses would improve the
  exact AxoBench targets they approximate without adding failure triggers.
- Baseline: stripped AxoMamba control (branch tanh + soma gate; no branch gain
  or integrative mixer), standard 0.25 BCE + 0.75 soma MSE objective.
- Controls: identical 69,087-parameter model, initialization, batch 32, 3,750
  updates, frozen balanced 120k schedule, optimizer, and diagnostics.

| candidate | compute s | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SERA 0.005 | 16.80 | -0.00137 | +0.00093 | -0.01311 | +0.00017 | -0.01823 | -0.00808 | reject alone |
| SERA 0.01 | 16.68 | -0.01299 | +0.00242 | -0.02279 | -0.00481 | -0.03396 | -0.01063 | reject alone |
| F1 curve 0.05 | 16.10 | -0.04047 | -0.00066 | +0.01609 | -0.00617 | +0.02963 | -0.00300 | keep lead |
| SERA 0.005 + F1 0.05 | 21.89 | -0.05235 | +0.00029 | +0.02076 | -0.00277 | +0.04156 | -0.00963 | promote |

- Interpretation: direct SERA alone trades away the F1 curve and slightly
  worsens dynamics. F1-curve training unexpectedly helps voltage and response
  SERA as well as tolerant F1. The combined objective is the strongest point:
  four significant core improvements, with dynamics and exact F1 uncertain.
  It has the same 16 attention triggers as control and adds no failure mode.
- Structural confirmation: the stripped control versus the old full stack
  improves voltage SERA by 0.023 mV, dynamics by 0.004 mV/ms, F1 area by
  0.011, 5 ms F1 by 0.017, and intervention SERA by 0.007 mV; exact F1 is
  uncertain. This supports the combined addon decision rather than merely the
  one-at-a-time removals.
- Decision: promote the stripped architecture and combined SERA 0.005 + F1
  curve 0.05 objective. Iteration 2 will target the remaining exact-F1 and
  dynamics gaps directly rather than tuning unrelated knobs.
- Artifacts: `artifacts/diagnostics/iteration-001-metric-objectives/`, including
  `promoted-control-vs-old-full.json`.

### Iteration 002 - 2026-07-12

- Regime: direct objective alignment for the two iteration-1 gaps: dynamics
  SERA and exact F1.
- Baseline: iteration-1 promoted SERA 0.005 + F1-curve 0.05 objective.
- Controls: identical architecture, parameters, initialization, frozen 120k
  schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| candidate | compute s | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamics 0.002 | 25.32 | +0.00201 | -0.00044 | +0.00701 | +0.00101 | +0.00572 | +0.00053 | reject |
| dynamics 0.005 | 25.24 | +0.01044 | -0.00101 | +0.00252 | -0.00118 | +0.00327 | +0.00162 | reject |
| exact F1 0.02 | 22.22 | -0.00677 | -0.00033 | -0.00081 | +0.00031 | -0.00211 | -0.00101 | reject mechanism |
| F1 curve 0.1 | 21.24 | -0.01932 | -0.00124 | -0.00078 | -0.00211 | -0.00492 | -0.00087 | keep lead, no promote |
| dynamics 0.002 + exact 0.02 | 26.06 | -0.00318 | -0.00100 | +0.00055 | +0.00183 | +0.00007 | -0.00045 | reject cost |

- Interpretation: direct dynamics SERA roughly doubles control compute and
  does not significantly improve its target; it also regresses voltage and/or
  intervention SERA. The exact-F1 surrogate does not significantly improve
  exact F1. F1-curve 0.1 significantly improves only voltage SERA, with the
  other core metrics uncertain and slightly lower.
- Decision: retain iteration 1 as current best. Keep F1-curve 0.1 as a later
  low-temperature lead; mark direct dynamics and exact-F1 additions for pruning
  at iteration 10 unless a distinct mechanism requires them.
- Artifacts: `artifacts/diagnostics/iteration-002-dynamics-exact-objectives/`.

### Iteration 003 premise - E/I cancellation

- Scientific premise: signed branch summation destroys coincident excitatory
  and inhibitory load before temporal state construction.
- Frozen-stream diagnostic over 64 windows: 59.83% of branch-time bins are
  active; 25.80% of active bins contain both E and I; 15.22% fully cancel; and
  33.83% of total event magnitude is removed by cancellation.
- Decision: premise supported strongly enough to test a parameter-matched
  `ei_tanh` branch transform against current signed tanh.
- Artifact: `artifacts/ei-cancellation-premise.json`.

### Iteration 003 - 2026-07-12

- Regime: scientific state-construction test. The supported cancellation
  premise was intervened on with separate excitatory/inhibitory tanh transforms
  before recombination.
- Controls: exact parameter match (69,087), common initialization, objective,
  batch 32, 3,750 updates, frozen 120k schedule, and diagnostics.
- Result: E/I tanh versus signed tanh changes voltage SERA by +0.00356 mV
  (regressed), dynamics by +0.00016 mV/ms (uncertain), F1 area by +0.00552
  (uncertain), exact F1 by +0.00092 (uncertain), 5 ms F1 by +0.00534
  (uncertain), and intervention SERA by +0.00286 mV (regressed). Compute rises
  from 21.94 s to 26.49 s.
- Interpretation: cancellation is real, but preserving E/I through separate
  saturation does not improve the learned temporal state. The premise does not
  imply this particular intervention is useful.
- Decision: reject `ei_tanh`; retain signed tanh. Keep the failed option only
  until the scheduled iteration-10 cleanup.
- Artifacts: `artifacts/diagnostics/iteration-003-ei-state-construction/`.

### Iteration 004 - 2026-07-12

- Regime: blind but high-leverage batch-32 optimizer compensation. Higher
  learning rates test whether fewer updates can be offset without changing
  data, parameters, presentations, or model compute.
- Baseline: iteration-1 architecture/objective, constant learning rate 3e-4.
- Controls: identical 69,087 parameters, initialization, frozen 120k schedule,
  batch 32, 3,750 updates, objective, and diagnostics.

| schedule | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4.5e-4 constant | +0.00481 | -0.00118 | -0.00124 | -0.00001 | -0.00444 | -0.00661 | reject tradeoff |
| 6e-4 constant | +0.05278 | +0.00233 | +0.00248 | -0.00275 | +0.00103 | -0.00052 | reject |
| 9e-4 constant | +0.04284 | +0.00197 | +0.02037 | -0.00348 | +0.03452 | +0.00748 | reject tradeoff |
| 6e-4 cosine | -0.35505 | +0.00142 | +0.00196 | -0.00229 | +0.00223 | -0.00700 | promote |

- Interpretation: constant-rate scaling does not recover the batch-size loss.
  A 6e-4 cosine schedule is qualitatively better: large significant voltage
  and intervention SERA gains, no significant core regression, and it clears
  the voltage-bias and default-threshold-count attention triggers.
- Decision: promote 6e-4 cosine. Retain the iteration-1 SERA/F1 objective until
  it is rechecked under this changed optimizer.
- Artifacts: `artifacts/diagnostics/iteration-004-batch32-learning-rate/`.

### Iteration 005 - 2026-07-12

- Regime: controlled objective recheck under the promoted 6e-4 cosine schedule.
- Baseline: SERA 0.005 + F1-curve 0.05, iteration-4 optimizer, stripped
  69,087-parameter AxoMamba.
- Controls: common initialization, architecture, frozen 120k schedule, batch
  32, 3,750 updates, optimizer schedule, and diagnostics.

| objective change | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| remove direct SERA | +0.06339 | +0.00244 | +0.00614 | +0.00644 | +0.01037 | +0.01353 | reject |
| SERA 0.0025 | +0.02875 | +0.00172 | +0.00269 | +0.00035 | +0.00655 | +0.00599 | reject |
| F1 curve 0.075 | +0.00573 | +0.00401 | +0.00794 | +0.00008 | +0.01357 | +0.00282 | reject tradeoff |
| F1 curve 0.1 | +0.00888 | +0.00456 | +0.00730 | +0.00307 | +0.01021 | +0.00441 | reject tradeoff |

- Interpretation: the existing combined objective remains the balanced point.
  Weakening SERA significantly regresses voltage, dynamics, and intervention
  SERA. Heavier F1 weighting significantly improves tolerant F1 but also
  significantly regresses those three SERA axes. The objective conclusions are
  therefore stable after the optimizer change.
- Decision: retain SERA 0.005 + F1-curve 0.05. Stop local objective-weight
  tuning and test the paper-motivated architectural coupling between voltage
  and spike predictions.
- Artifacts: `artifacts/diagnostics/iteration-005-objective-under-cosine/`.

### Iteration 006 premise - voltage/event coupling

- Scientific premise: current spike and voltage outputs share a latent state
  but remain freely independent readout coordinates, despite the paper's
  prediction that faithful surrogates need intrinsic coupling.
- Frozen-stream diagnostic over 4,096 windows: spike prevalence is 0.347%; mean
  normalized soma is 1.209 at spike steps versus 0.020 elsewhere, and the
  spike/soma correlation peaks at 0.201 one millisecond before the event.
- Decision: test a fixed spike-logit residual from the predicted soma. This
  routes event-loss gradients through voltage without adding parameters.

### Iteration 006 - 2026-07-12

- Regime: paper-motivated architectural coupling between the continuous soma
  coordinate and discrete spike logit.
- Baseline: iteration-5 uncoupled model. Candidates add fixed soma residuals
  of 0.25, 0.5, 1.0, or 2.0 to the spike logit.
- Controls: exact 69,087-parameter match, common initialization, objective,
  6e-4 cosine optimizer, batch 32, 3,750 updates, frozen 120k schedule, and
  diagnostics. Candidate compute is 21.15-21.25 s versus 22.00 s control.

| coupling | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | -0.00049 | -0.00010 | +0.00359 | -0.00002 | +0.00723 | -0.00023 | positive, weaker |
| 0.5 | -0.00053 | +0.00011 | +0.00421 | -0.00025 | +0.00954 | -0.00011 | positive, weaker |
| 1.0 | -0.00035 | +0.00072 | +0.00493 | +0.00485 | +0.00849 | -0.00021 | positive, uncertain |
| 2.0 | -0.00348 | -0.00028 | +0.00819 | +0.00366 | +0.01425 | -0.00114 | promote |

- Interpretation: scale 2.0 significantly improves voltage SERA, F1 area,
  5 ms F1, and intervention SERA. Dynamics and exact F1 are uncertain, no
  core metric significantly regresses, and the candidate adds no attention
  trigger. Improvements increase broadly with coupling strength, so 2.0 is a
  lower bound rather than an established optimum.
- Decision: promote fixed voltage-to-spike coupling 2.0. Iteration 7 will map
  the upper range at 3, 4, 6, and 8 before changing mechanisms.
- Artifacts: `artifacts/diagnostics/iteration-006-voltage-spike-coupling/`.

### Iteration 007 - 2026-07-12

- Regime: upper-range mapping of the promoted voltage-to-spike coupling.
- Baseline: coupling 2.0. Candidates: 3.0, 4.0, 6.0, and 8.0.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| coupling | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3.0 | -0.00685 | -0.00085 | -0.00182 | -0.00106 | -0.00142 | -0.00397 | positive, weaker |
| 4.0 | -0.01132 | +0.00060 | +0.00049 | +0.00003 | -0.00466 | -0.00763 | promote |
| 6.0 | -0.00830 | +0.00132 | -0.00838 | -0.00120 | -0.01636 | -0.00858 | reject F1 regression |
| 8.0 | +0.00938 | +0.00097 | -0.00930 | -0.00847 | -0.01338 | -0.00007 | reject |

- Interpretation: coupling 4.0 significantly improves both voltage and
  intervention SERA while preserving F1 area and exact F1; its 5 ms F1 point
  estimate is lower but uncertain. Scale 6 significantly regresses tolerant
  F1, and scale 8 also regresses voltage SERA, establishing the boundary.
- Decision: promote coupling 4.0 as the SERA-first balanced point.
- Artifacts: `artifacts/diagnostics/iteration-007-coupling-range/`.

### Iteration 008 premise - positive voltage slope

- Frozen-stream diagnostic over 4,096 windows: normalized soma dV averages
  0.194 at spike steps versus -0.0007 elsewhere. Spike correlation is 0.103
  for signed dV and 0.262 for positive dV, exceeding the 0.195 level
  correlation that supported iterations 6-7.
- Decision: test a fixed positive-dV residual into the spike logit on top of
  the promoted level coupling. This adds no parameter and sends event-loss
  gradients through local voltage dynamics.

### Iteration 008 - 2026-07-12

- Regime: parameter-neutral positive predicted-dV coupling into the spike
  logit, on top of voltage-level coupling 4.0.
- Controls: identical 69,087 parameters, common initialization, objective,
  frozen 120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| dV coupling | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2.0 | +0.02515 | -0.00043 | -0.00815 | -0.00202 | -0.00819 | +0.00653 | reject |
| 4.0 | +0.02840 | -0.00017 | -0.00786 | -0.00291 | -0.01151 | +0.00637 | reject |
| 8.0 | +0.02611 | +0.00095 | -0.00967 | -0.00689 | -0.01476 | +0.00674 | reject |
| 16.0 | +0.02875 | +0.00459 | -0.00101 | -0.00596 | -0.00172 | +0.00443 | reject |

- Interpretation: teacher positive dV is event-correlated, but forcing the
  model's predicted positive dV into the event logit worsens voltage and
  intervention SERA at every scale. Scales 4-8 also regress tolerant F1, scale
  16 regresses dynamics, and scale 8 adds a default-threshold count trigger.
  The premise does not transfer to this learned-coordinate intervention.
- Decision: reject positive-dV coupling and keep derivative coupling disabled.
  Remove its failed code path in the iteration-10 prune.
- Artifacts: `artifacts/diagnostics/iteration-008-voltage-derivative-coupling/`.

### Iteration 009 - 2026-07-12

- Regime: low-temperature initialization panel after the architecture passes.
- Controls: exact 69,087 parameters, common state except the named initialized
  tensor, objective, frozen 120k schedule, batch 32, 3,750 updates, optimizer,
  and diagnostics.

| initialization | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| residual 0.03 | +0.00565 | +0.00050 | -0.00553 | -0.00661 | -0.00163 | +0.00170 | reject |
| residual 0.3 | -0.00443 | -0.00016 | -0.01074 | -0.00043 | -0.01576 | +0.00599 | reject |
| branch tanh 0.03 | +0.00054 | -0.00006 | -0.00054 | -0.00110 | +0.00067 | +0.00015 | reject |
| branch tanh 0.3 | -0.00312 | -0.00063 | -0.00716 | +0.00202 | -0.00989 | +0.00047 | reject |
| soma gate 0.03 | +0.00286 | -0.00012 | -0.00055 | +0.00154 | -0.00117 | +0.00051 | reject |
| soma gate 0.3 | -0.00834 | -0.00037 | -0.00422 | +0.00117 | -0.00307 | -0.00120 | reject tradeoff |

- Interpretation: larger residual and branch-tanh starts trade voltage gains
  for significant tolerant-F1 and/or intervention regressions. Soma-gate 0.3
  improves voltage and intervention SERA but significantly regresses F1 area.
  Smaller initializations do not improve the balanced core metric set.
- Decision: retain 0.1 for all three scales. Promote no candidate.
- Artifacts: `artifacts/diagnostics/iteration-009-initialization-scales/`.

### Iteration 010 maintenance - 2026-07-12

- Pruned failed loop modifications: E/I-separated tanh, positive-dV spike
  coupling, direct dynamics-SERA loss, and exact-F1-only loss. Historical
  manifests and diagnostics remain as the experiment record.
- Long-run question: does successful voltage-level coupling remain useful at
  300k presentations, or is it only an early-optimization accelerator?

### Iteration 010 - 2026-07-12

- Regime: scheduled 300k long-run persistence test and coupling ablation.
- Baseline: promoted voltage coupling 4.0. Candidates: coupling 2.0 and
  uncoupled.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  morphology-balanced 300k schedule, batch 32, 9,375 updates, optimizer, and
  diagnostics. Model compute is 53.75-55.97 s per candidate.

| coupling | voltage SERA | dynamics SERA | F1 area | exact F1 | 5 ms F1 | intervention SERA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 | 17.55336 | 21.33630 | 0.14879 | 0.04827 | 0.20915 | 17.66807 |
| 2.0 | 17.56026 | 21.33691 | 0.14280 | 0.04714 | 0.20581 | 17.67081 |
| 0.0 | 17.56403 | 21.33657 | 0.13972 | 0.04717 | 0.19730 | 17.66853 |

- Interpretation: coupling 4.0 remains useful beyond early training. Versus
  uncoupled it significantly improves voltage SERA by 0.01067 mV, F1 area by
  0.00908, and 5 ms F1 by 0.01186. Versus coupling 2.0 it significantly
  improves voltage SERA by 0.00691 mV, F1 area by 0.00599, and intervention
  SERA by 0.00274 mV. Dynamics/exact F1 remain uncertain. Uncoupled and 2.0
  each add a default-threshold count trigger relative to 4.0.
- Decision: retain coupling 4.0 as the long-run-supported AxoMamba default.
- Artifacts: `artifacts/diagnostics/iteration-010-long-coupling-ablation/`.

### Iteration 011 premise - depolarized-only coupling

- The 300k promoted model predicts normalized soma from -1.17 to 0.69, with
  90th/99th percentiles 0.37/0.53. Teacher voltage thresholding becomes much
  more event-selective toward its depolarized tail (exact event F1 rises from
  0.014 at threshold 0 to 0.518 at 1.0 while retaining nearly all events).
- Decision: compare linear coupling with parameter-neutral rectified coupling
  thresholds 0.0-0.4, targeting depolarized and near-spike failure modes.

### Iteration 011 - 2026-07-12

- Regime: parameter-neutral rectified voltage-to-spike coupling.
- Baseline: promoted signed linear coupling 4.0. Candidates use ReLU thresholds
  0.0, 0.1, 0.2, 0.3, or 0.4 at the same scale.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| threshold | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | +0.01011 | -0.00163 | -0.00081 | -0.00340 | +0.00166 | +0.00550 | reject |
| 0.1 | +0.01500 | -0.00070 | -0.00248 | -0.00153 | -0.00239 | +0.00746 | reject |
| 0.2 | +0.01747 | +0.00010 | -0.00299 | -0.00080 | -0.00274 | +0.00816 | reject |
| 0.3 | +0.01916 | +0.00059 | +0.00148 | -0.00010 | +0.00050 | +0.00836 | reject |
| 0.4 | +0.01762 | +0.00030 | -0.00033 | -0.00203 | +0.00345 | +0.00831 | reject |

- Interpretation: every rectified variant significantly regresses voltage and
  intervention SERA, and no F1 change is significant. Even threshold 0.0 is
  worse, showing that the useful mechanism includes signed suppression from
  hyperpolarized voltage rather than only depolarized excitation.
- Decision: retain full signed linear coupling. Queue thresholded coupling for
  iteration-20 pruning if it remains unused.
- Artifacts: `artifacts/diagnostics/iteration-011-thresholded-voltage-coupling/`.

### Iteration 012 premise - bidirectional event/voltage coupling

- Diagnostics still show voltage gain 0.197 and 4.47x near-spike error. Current
  coupling lets voltage shape the event logit and routes event gradients into
  voltage, but the base event head contributes no explicit AP component back
  to soma.
- Decision: add a fixed sigmoid(base-spike-logit) residual to soma before the
  promoted voltage-to-spike coupling. This is acyclic, parameter-neutral, and
  directly tests whether event predictions can restore peak voltage.

### Iteration 012 - 2026-07-12

- Regime: parameter-neutral base-spike-probability residual into soma before
  the promoted soma-to-spike coupling.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| spike-to-soma scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | -0.00393 | -0.00012 | -0.00872 | -0.00399 | -0.01093 | -0.00278 | reject tradeoff |
| 0.25 | +0.01731 | -0.00060 | -0.00797 | +0.00068 | -0.01242 | +0.00611 | reject |
| 0.5 | +0.01603 | +0.00085 | -0.01372 | -0.00189 | -0.01342 | +0.00627 | reject |
| 1.0 | -0.00523 | +0.00614 | -0.01234 | -0.00089 | -0.01594 | -0.00418 | reject |

- Interpretation: scale 0.1 improves voltage and intervention SERA, but its F1
  point estimates drop materially. Larger scales significantly regress F1
  area/5 ms F1 and often voltage, response, or dynamics. The event head is not
  temporally faithful enough to reconstruct the AP waveform without feeding
  timing errors back into soma.
- Decision: reject explicit spike-to-soma residuals; retain one-way signed
  voltage-to-spike coupling. Queue this failed path for iteration-20 pruning.
- Artifacts: `artifacts/diagnostics/iteration-012-bidirectional-output-coupling/`.

### Iteration 013 premise - refractory event dynamics

- The promoted model has a 0.221 fraction of predicted inter-spike intervals
  below the broad 2 ms refractory bound. The event output currently has no
  causal self-inhibition despite this directly measured duplicate-event mode.
- Decision: subtract a fixed multiple of the previous step's predicted event
  probability from the current spike logit. This is causal, differentiable,
  parameter-neutral, and directly targets refractory violations and F1.

### Iteration 013 - 2026-07-12

- Regime: parameter-neutral causal refractory inhibition of spike logits.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| inhibition | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | refractory fraction | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | -0.00114 | -0.00097 | -0.00587 | -0.00268 | -0.00644 | +0.00183 | 0.230 | reject |
| 2.0 | -0.00129 | -0.00003 | -0.00138 | -0.00776 | +0.00557 | +0.00240 | 0.193 | reject |
| 4.0 | +0.01107 | +0.00134 | -0.00261 | -0.01031 | +0.00198 | +0.01348 | 0.200 | reject |
| 8.0 | -0.00515 | +0.00090 | -0.01295 | -0.00179 | -0.01415 | +0.01020 | 0.253 | reject |

- Interpretation: strength 2 reduces refractory violations from 0.221 to
  0.193 and count error from 0.745 to 0.723, but does not improve F1 and
  significantly regresses intervention SERA. Stronger inhibition regresses
  exact F1, F1 area, voltage, or response. Duplicate suppression is not the
  missing event-recovery mechanism.
- Decision: reject causal refractory inhibition; queue for iteration-20 prune.
- Artifacts: `artifacts/diagnostics/iteration-013-refractory-event-head/`.

### Iteration 014 premise - soma output gain

- The promoted model's least-squares voltage gain is only 0.197, while voltage
  bias is acceptable. This points to amplitude compression rather than a
  constant offset and aligns with high depolarized/near-spike errors.
- Decision: test fixed soma output scales 1.25-4.0 before the successful
  voltage-to-spike coupling. The change is end-to-end trained, adds no
  parameters, and is guarded by voltage-bound/slew and state diagnostics.

### Iteration 014 - 2026-07-12

- Regime: parameter-neutral soma output scaling before the promoted
  voltage-to-spike coupling.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| soma scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.25 | +0.00359 | -0.00024 | -0.01282 | -0.00432 | -0.01407 | +0.00072 | reject |
| 1.5 | +0.00782 | -0.00008 | -0.01149 | -0.00619 | -0.01296 | +0.00200 | reject |
| 2.0 | +0.00234 | -0.00297 | -0.00953 | -0.00361 | -0.01124 | +0.00480 | reject |
| 3.0 | -0.00262 | -0.00043 | -0.01314 | -0.00030 | -0.01694 | +0.00736 | reject |
| 4.0 | -0.01627 | -0.00150 | -0.00907 | -0.00052 | -0.01041 | +0.00722 | reject |

- Interpretation: scale 4 significantly improves voltage SERA but regresses
  intervention SERA and lowers tolerant F1. Smaller scales mostly regress
  voltage/intervention and significantly regress F1. The low voltage-gain
  diagnostic is not repaired by scalar output calibration; it reflects learned
  state/shape error.
- Decision: retain soma scale 1.0; queue output scaling for iteration-20 prune.
- Artifacts: `artifacts/diagnostics/iteration-014-soma-output-scale/`.

### Iteration 015 premise - coupled-model optimization

- High-temperature passes have isolated one successful architecture mechanism
  and several falsified alternatives. The 6e-4 cosine schedule was selected
  before voltage coupling changed the gradient geometry, so optimizer settings
  are now a justified lower-temperature axis.
- Decision: compare nearby cosine rates, Adam versus AdamW, weight decay, and
  gradient clipping with architecture/data/parameters/updates fixed.

### Iteration 015 - 2026-07-12

- Regime: lower-temperature optimizer panel on the promoted coupled model.
- Controls: identical 69,087 parameters, initialization, objective, frozen
  120k schedule, batch 32, 3,750 updates, and diagnostics.

| optimizer change | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AdamW 4.5e-4 cosine | +0.01037 | -0.00073 | -0.00816 | +0.00117 | -0.01158 | +0.00508 | reject |
| AdamW 9e-4 cosine | -0.00104 | +0.00165 | -0.01199 | -0.00483 | -0.01529 | -0.00121 | reject |
| AdamW 1.2e-3 cosine | -0.01294 | +0.00222 | -0.01261 | -0.00398 | -0.01424 | -0.00253 | reject tradeoff |
| Adam 6e-4 cosine | +0.00237 | -0.00024 | -0.01296 | -0.00991 | -0.01515 | +0.00206 | reject |
| no weight decay | -0.000003 | +0.000002 | +0.00008 | +0.00003 | +0.00011 | +0.000001 | equivalent |
| clip norm 1 | -0.00266 | -0.00077 | -0.00379 | +0.00175 | -0.00309 | +0.00109 | reject tradeoff |

- Interpretation: higher rate 1.2e-3 improves both SERA metrics but
  significantly regresses F1 area. Lower/higher rates and Adam regress F1 or
  SERA. Weight decay is inert at this budget, and clipping offers no balanced
  improvement.
- Decision: retain AdamW 6e-4 cosine and weight decay 1e-4 for continuity;
  remove weight decay later only as cleanup, not a claimed metric gain.
- Artifacts: `artifacts/diagnostics/iteration-015-coupled-optimizer/`.

### Iteration 016 premise - F1 tolerance curriculum

- The total F1-curve weight is already at its balanced maximum: iteration 5
  showed larger weights trade away all SERA axes. The remaining degree of
  freedom is which timing tolerances contribute gradient.
- Decision: compare close-timing horizons and duplicate-tolerance weighting at
  the same F1 loss coefficient, parameters, and update budget.

### Iteration 016 - 2026-07-12

- Regime: lower-temperature timing-tolerance allocation at fixed F1 loss
  coefficient 0.05.
- Controls: identical 69,087 parameters, initialization, frozen 120k schedule,
  batch 32, 3,750 updates, optimizer, and diagnostics.

| tolerance allocation | compute s | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0-2 only | 19.45 | -0.00086 | -0.00039 | -0.02194 | -0.00250 | -0.03081 | -0.00180 | reject |
| 0-3 only | 20.06 | -0.00065 | -0.00036 | -0.01529 | -0.00392 | -0.02128 | -0.00120 | reject |
| double exact | 22.25 | +0.00154 | -0.00019 | -0.00663 | -0.00099 | -0.00856 | -0.00007 | reject |
| double close | 23.81 | +0.00129 | -0.00017 | -0.01062 | -0.00010 | -0.01613 | -0.00040 | reject |
| double 5 ms | 22.40 | -0.00156 | -0.00004 | +0.00048 | +0.00100 | +0.00230 | -0.00024 | provisional |

- Interpretation: shortening or emphasizing close/exact tolerances
  significantly worsens F1 area and 5 ms F1. Duplicating 5 ms supervision
  significantly improves voltage and intervention SERA; all F1 points also
  improve but are individually uncertain. It adds no trigger and costs no more
  than control.
- Decision: treat double-5 ms as a provisional promotion. Its effect is small,
  so iteration 17 repeats old versus new objectives under a different frozen
  schedule and initialization before updating the retained recipe.
- Artifacts: `artifacts/diagnostics/iteration-016-f1-tolerance-curriculum/`.

### Iteration 017 premise - seed replication

- The iteration-16 paired bootstrap quantifies held-out sample uncertainty but
  not training-seed/schedule uncertainty. The proposed gain is small enough
  that one-seed promotion would be overconfident.
- Decision: rerun only the old and double-5 ms objectives with seed 20260712,
  matched initialization and a new frozen morphology-balanced 120k schedule.

### Iteration 017 - 2026-07-12

- Regime: training-seed and frozen-schedule replication of iteration 16.
- Controls: seed 20260712, exactly balanced 120k schedule (24k per each of five
  morphologies), identical 69,087 parameters, matched initialization, batch 32,
  3,750 updates, optimizer, and diagnostics.
- Result: double-5 ms supervision improves F1 area by 0.00615 significantly.
  Exact F1 rises 0.00048, 5 ms F1 rises 0.00475, voltage SERA improves 0.00016,
  and dynamics improves 0.00045, all uncertain; intervention SERA regresses
  0.00024 uncertainly. Compute is 21.90 s versus 21.82 s control. Both models
  have the same 15 attention triggers on this seed.
- Interpretation: the original seed's significant SERA gain does not replicate
  at significance, but the candidate now significantly improves its direct F1
  area target without a significant core regression. The direction of all F1
  deltas is consistent across seeds.
- Decision: retain as provisional and run one final seed/schedule replication
  before changing the durable training recipe.
- Artifacts: `artifacts/diagnostics/iteration-017-double-tolerant-replication/`.

### Iteration 018 premise - final seed replication

- Two seeds show complementary significant gains (SERA on seed 1, F1 area on
  seed 2) and no significant regression, but a third matched seed reduces the
  chance of promoting a small unstable effect.
- Decision: repeat the same two-model comparison at seed 20260713.

### Iteration 018 - 2026-07-12

- Regime: third training-seed/schedule replication at seed 20260713.
- Controls: another exactly balanced 120k schedule, identical 69,087
  parameters, matched initialization, batch 32, 3,750 updates, optimizer, and
  diagnostics.
- Result: double-5 ms supervision improves voltage SERA by 0.00112
  significantly. F1 area rises 0.00111 and 5 ms F1 rises 0.00116 uncertainly;
  exact F1 regresses 0.00355 significantly. Dynamics and intervention are
  neutral. Compute is 22.24 s versus 22.16 s control; no trigger is added.
- Across-seed interpretation: mean deltas over three seeds are approximately
  -0.00095 voltage SERA, -0.00014 dynamics SERA, +0.00258 F1 area, -0.00069
  exact F1, +0.00274 5 ms F1, and -0.00002 intervention SERA. F1 area and 5 ms
  F1 improve on every seed; exact F1 is mixed and significantly worse once.
- Decision: promote double-5 ms supervision because the stated event target is
  F1 over the time curve, with no mean/core SERA cost. Keep the exact-F1
  tradeoff explicit. The reusable recipe is
  `default_axomamba_training_objective()`.
- Artifacts: `artifacts/diagnostics/iteration-018-double-tolerant-replication-2/`.

### Iteration 019 premise - exact SERA integration

- Training approximates direct root-SERA with 21 relevance thresholds, while
  AxoBench evaluates with 101. This is a concrete metric-alignment mismatch,
  but denser thresholds may increase loss compute.
- Decision: compare 11, 51, and exact 101 thresholds against the promoted
  21-threshold recipe, retaining double-5 ms F1 supervision.

### Iteration 019 - 2026-07-12

- Regime: lower-temperature direct-SERA integration resolution.
- Controls: identical 69,087 parameters, initialization, promoted double-5 ms
  objective, frozen 120k schedule, batch 32, 3,750 updates, optimizer, and
  diagnostics.

| SERA thresholds | compute s | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 22.32 | +0.00071 | -0.00006 | +0.00061 | +0.00001 | -0.00014 | +0.00003 | reject |
| 51 | 22.16 | -0.00014 | +0.00002 | +0.00167 | -0.00013 | -0.00001 | -0.00006 | provisional |
| 101 | 22.18 | -0.00024 | -0.00008 | +0.00106 | +0.00023 | -0.00029 | -0.00006 | weaker |

- Interpretation: 51 thresholds significantly improves voltage and
  intervention SERA with no significant core regression; F1 area rises
  uncertainly. Exact 101 thresholds improves voltage only and does not improve
  the metric pattern or compute. Dense masks add no measurable model-level
  compute cost in this implementation.
- Decision: carry 51 thresholds provisionally into the scheduled 300k
  iteration-20 comparison against 21 and 101.
- Artifacts: `artifacts/diagnostics/iteration-019-sera-resolution/`.

### Iteration 020 maintenance - 2026-07-12

- Pruned failed post-iteration-10 paths: rectified/thresholded voltage-to-spike
  coupling, spike-to-soma residuals, causal refractory inhibition, and fixed
  soma output scaling. Historical manifests and diagnostics remain.
- Long-run question: does 51-threshold SERA retain an advantage over 21 and
  exact 101 after 300k presentations?

### Iteration 020 - 2026-07-12

- Regime: scheduled 300k long-run SERA-resolution comparison after cleanup.
- Baseline: 51 thresholds. Candidates: 21 and exact evaluation-matched 101.
- Controls: promoted double-5 ms objective, identical 69,087 parameters,
  initialization, frozen balanced 300k schedule, batch 32, 9,375 updates,
  optimizer, and diagnostics. Compute is 55.38-57.41 s.

| SERA thresholds | voltage SERA | dynamics SERA | F1 area | exact F1 | 5 ms F1 | intervention SERA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | 17.55387 | 21.33644 | 0.14579 | 0.04629 | 0.20915 | 17.66828 |
| 21 | 17.55391 | 21.33628 | 0.14497 | 0.04666 | 0.20798 | 17.66841 |
| 101 | 17.55351 | 21.33637 | 0.14711 | 0.04813 | 0.20913 | 17.66813 |

- Interpretation: exact 101-threshold training significantly improves voltage
  SERA by 0.00036 mV and intervention SERA by 0.00015 mV versus 51; F1 area
  and exact F1 also rise uncertainly, dynamics/5 ms are neutral, and compute is
  matched. The 21-threshold approximation is weaker on response and F1 area.
- Decision: promote 101 thresholds. The reusable objective now mirrors
  evaluation resolution exactly.
- Artifacts: `artifacts/diagnostics/iteration-020-long-sera-resolution/`.

### Iteration 021 premise - state-reset burn-in

- Training resets Mamba state on each 500-step sampled window and ignores the
  first 150 steps, while evaluation ignores 500 steps of a full trace. Reset
  contamination may therefore remain in the supervised training suffix.
- Decision: compare burn-ins 0, 50, 250, and 350 against 150 on identical
  windows. Parameters, forward compute, presentations, and updates remain
  matched; only supervised suffix length changes.

### Iteration 021 - 2026-07-12

- Regime: protocol-alignment test of reset-state burn-in.
- Controls: identical 69,087 parameters, initialization, exact-101 SERA plus
  double-5 ms objective, frozen 120k schedule, batch 32, 3,750 updates,
  optimizer, full-window forward compute, and diagnostics.

| burn-in | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | -0.00733 | -0.00063 | +0.00211 | +0.00006 | +0.00569 | +0.00012 | positive |
| 50 | -0.00729 | -0.00032 | +0.00226 | -0.00230 | +0.00591 | +0.00053 | provisional |
| 250 | +0.00692 | +0.00031 | -0.00052 | -0.00092 | +0.00105 | +0.00210 | reject |
| 350 | +0.00777 | +0.00010 | +0.00369 | +0.00179 | +0.00594 | +0.00243 | reject |

- Interpretation: longer burn-in significantly worsens voltage and response,
  so reset-state contamination is not helped by discarding more labels.
  Burn-in 50 significantly improves voltage and 5 ms F1; F1 area also rises,
  with dynamics/response neutral. It adds a marginal default-threshold count
  watch (ratio 1.282 versus upper bound 1.25), while actual count error improves
  0.755 to 0.727. Burn-in 0 has a similar but weaker F1 pattern.
- Decision: provisionally select burn-in 50 and replicate at seed 20260712
  before deciding whether the marginal calibration trigger is meaningful.
- Artifacts: `artifacts/diagnostics/iteration-021-state-reset-burn-in/`.

### Iteration 022 premise - burn-in replication

- The burn-in 50 gain is strong enough to matter but changes a calibration
  watch by only 0.032 beyond its bound. A second training seed distinguishes a
  real context/supervision gain from one-seed calibration movement.
- Decision: rerun burn-in 150 versus 50 at seed 20260712 on a new balanced 120k
  schedule, retaining all promoted objective/model settings.

### Iteration 022 - 2026-07-12

- Regime: second-seed replication of burn-in 50 versus 150.
- Controls: seed 20260712, balanced 120k schedule, identical 69,087
  parameters, matched initialization, exact-101/double-5 ms objective, batch
  32, 3,750 updates, optimizer, forward compute, and diagnostics.
- Result: burn-in 50 improves voltage SERA by 0.01633 mV significantly. F1
  area rises 0.00033 and exact F1 rises 0.00173 uncertainly; dynamics is
  neutral, 5 ms F1 falls 0.00185 uncertainly, and intervention SERA regresses
  0.00077 uncertainly. Compute is 22.32 s versus 23.02 s.
- Trigger interpretation: both control and candidate trip default-threshold
  count on this seed (1.411 and 1.529), and candidate count error is slightly
  better (0.7837 versus 0.7843). On seed 1, the candidate's 1.282 ratio only
  marginally exceeded 1.25 while count error improved materially. We use the
  user's one allowed liberalized-trigger exception for this crossing without
  changing the diagnostic globally.
- Decision: promote burn-in 50. It gives a robust voltage-SERA gain across two
  seeds with no significant core regression. The reusable constant is
  `AXOMAMBA_DEFAULT_BURN_IN`.
- Artifacts: `artifacts/diagnostics/iteration-022-burn-in-replication/`.

### Iteration 023 premise - direct SERA weight after protocol changes

- Direct-SERA weight 0.005 was last balanced before exact-101 integration,
  double-5 ms supervision, and burn-in 50 changed the loss geometry and amount
  of supervised context.
- Decision: recheck nearby SERA weights 0.0025, 0.0075, and 0.01 against 0.005
  under the fully promoted recipe, with all candidates using burn-in 50.

### Iteration 023 - 2026-07-12

- Regime: lower-temperature direct-SERA weight retune under all promoted model,
  objective, and burn-in settings.
- Controls: identical 69,087 parameters, initialization, exact-101/double-5 ms
  objective shape, burn-in 50, frozen 120k schedule, batch 32, 3,750 updates,
  optimizer, and diagnostics.

| SERA weight | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0025 | +0.02872 | +0.00011 | +0.00263 | +0.00373 | +0.00309 | +0.00557 | reject |
| 0.0075 | -0.02359 | +0.00009 | -0.00193 | +0.00313 | -0.00333 | -0.00455 | provisional |
| 0.01 | -0.04278 | +0.00028 | -0.00791 | +0.00630 | -0.01246 | -0.00894 | reject F1 tradeoff |

- Interpretation: 0.0075 significantly improves voltage and intervention SERA
  and raises exact F1; F1 area/5 ms point estimates are slightly lower but
  uncertain. Weight 0.01 improves both SERA axes and exact F1 more, but
  significantly regresses F1 area and 5 ms F1. Weight 0.0025 regresses SERA.
- Decision: provisionally select 0.0075 and replicate against 0.005 on seed
  20260712 before promotion.
- Artifacts: `artifacts/diagnostics/iteration-023-sera-weight-retune/`.

### Iteration 024 premise - SERA-weight replication

- The 0.0075 gain is large on both SERA axes but allows small uncertain losses
  on the time-curve F1 metrics. A second seed must preserve the SERA advantage
  without making the F1 tradeoff significant.
- Decision: rerun 0.005 versus 0.0075 at seed 20260712 with all other promoted
  settings fixed.

### Iteration 024 - 2026-07-12

- Regime: second-seed direct-SERA weight replication.
- Controls: seed 20260712, identical 69,087 parameters, matched initialization,
  exact-101/double-5 ms objective, burn-in 50, balanced 120k schedule, batch 32,
  3,750 updates, optimizer, and diagnostics.
- Result: weight 0.0075 improves voltage SERA by 0.02540 mV, intervention SERA
  by 0.00343 mV, and 5 ms F1 by 0.00834, all significant. F1 area rises
  0.00289 and exact F1 rises 0.00127 uncertainly; dynamics is neutral. Compute
  is 22.20 s versus 22.89 s control.
- Interpretation: the SERA gains replicate almost exactly, and the prior
  uncertain F1 decreases reverse on seed 2, including a significant 5 ms gain.
  There is no balanced-metric tradeoff across seeds.
- Decision: promote direct-SERA weight 0.0075 in the reusable objective.
- Artifacts: `artifacts/diagnostics/iteration-024-sera-weight-replication/`.

### Iteration 025 premise - F1 coefficient after stronger SERA

- Iteration 5 showed F1 coefficients above 0.05 traded away SERA under the old
  recipe. The now-replicated 0.0075 SERA weight may support somewhat stronger
  time-curve supervision without that regression.
- Decision: compare F1 weights 0.04, 0.06, 0.075, and 0.1 against 0.05 with all
  other promoted settings fixed.

### Iteration 025 - 2026-07-12

- Regime: lower-temperature F1 coefficient retune after SERA weight 0.0075.
- Controls: identical 69,087 parameters, initialization, exact-101/double-5 ms
  shape, burn-in 50, frozen 120k schedule, batch 32, 3,750 updates, optimizer,
  and diagnostics.

| F1 weight | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.04 | +0.00021 | -0.00030 | -0.00871 | +0.00200 | -0.01105 | -0.00125 | reject |
| 0.06 | -0.00214 | -0.00005 | -0.00205 | -0.00439 | -0.00284 | -0.00032 | reject |
| 0.075 | -0.00182 | +0.00094 | -0.00117 | -0.00243 | +0.00135 | +0.00006 | reject |
| 0.1 | +0.00076 | +0.00146 | +0.00187 | +0.00238 | +0.00690 | +0.00204 | reject |

- Interpretation: reducing F1 weight significantly regresses the F1 curve.
  Weight 0.06 slightly improves SERA but significantly regresses exact F1.
  Weight 0.1 improves F1 point estimates but significantly regresses
  intervention SERA. No candidate improves the balanced core set.
- Decision: retain F1 coefficient 0.05.
- Artifacts: `artifacts/diagnostics/iteration-025-f1-weight-retune/`.

### Iteration 026 premise - final-recipe cosine rate

- The 6e-4 cosine rate was last swept before exact-101 SERA, double-5 ms F1,
  burn-in 50, and SERA weight 0.0075 finalized the gradient geometry.
- Decision: compare nearby peak rates 4.5e-4, 7.5e-4, and 9e-4 against 6e-4
  with the complete promoted recipe fixed.

### Iteration 026 - 2026-07-12

- Regime: final-recipe cosine peak-rate check.
- Controls: identical 69,087 parameters, initialization, promoted objective,
  burn-in 50, frozen 120k schedule, batch 32, 3,750 updates, AdamW cosine, and
  diagnostics.

| peak rate | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4.5e-4 | +0.00660 | -0.00026 | -0.00635 | +0.00153 | -0.01046 | +0.00261 | reject |
| 7.5e-4 | -0.00744 | -0.00076 | -0.00170 | -0.00455 | -0.00354 | -0.00267 | provisional |
| 9e-4 | -0.01157 | -0.00043 | -0.00868 | -0.00316 | -0.01442 | -0.00535 | reject F1 tradeoff |

- Interpretation: 7.5e-4 significantly improves voltage and intervention SERA
  while F1 area/5 ms/exact are uncertainly lower and dynamics neutral. At
  9e-4 the SERA gains increase but F1 area and 5 ms regress significantly.
  Lower 4.5e-4 regresses SERA and F1.
- Decision: provisionally select 7.5e-4 and replicate against 6e-4 at seed
  20260712 before changing the durable optimizer.
- Artifacts: `artifacts/diagnostics/iteration-026-final-recipe-learning-rate/`.

### Iteration 027 premise - learning-rate replication

- The 7.5e-4 candidate repeats the familiar SERA/F1 frontier but stays within
  F1 uncertainty on seed 1. Replication must preserve SERA gains without a
  significant F1 regression.
- Decision: rerun 6e-4 versus 7.5e-4 at seed 20260712 under the full promoted
  recipe.

### Iteration 027 - 2026-07-12

- Regime: second-seed cosine-rate replication.
- Controls: seed 20260712, identical 69,087 parameters, matched initialization,
  full promoted objective/burn-in, balanced 120k schedule, batch 32, 3,750
  updates, optimizer, and diagnostics.
- Result: 7.5e-4 significantly regresses voltage SERA by 0.00561 mV. F1 area
  rises 0.00233 and intervention SERA improves 0.00032 uncertainly; dynamics,
  exact F1, and 5 ms F1 are neutral. Compute is matched.
- Interpretation: the seed-1 voltage/intervention gain does not replicate and
  reverses significantly on voltage. The higher rate is not robust.
- Decision: retain 6e-4 cosine and close local optimizer tuning.
- Artifacts: `artifacts/diagnostics/iteration-027-learning-rate-replication/`.

### Iteration 028 premise - final-recipe coupling scale

- Voltage-to-spike coupling is the only successful architecture mechanism, but
  its scale 4.0 was selected before exact-101 SERA, double-5 ms supervision,
  burn-in 50, and SERA weight 0.0075 changed gradients.
- Decision: remap coupling scales 3.0, 3.5, 4.5, 5.0, and 6.0 against 4.0 under
  the finalized training recipe. Parameters and compute remain matched.

### Iteration 028 - 2026-07-12

- Regime: final-recipe remap of the successful voltage-to-spike mechanism.
- Controls: identical 69,087 parameters, initialization, promoted objective and
  burn-in, frozen 120k schedule, batch 32, 3,750 updates, optimizer, and
  diagnostics.

| coupling | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3.0 | -0.00082 | -0.00076 | -0.00246 | -0.00063 | -0.00123 | +0.00080 | reject |
| 3.5 | -0.00206 | -0.00025 | -0.00266 | -0.00181 | -0.00366 | +0.00008 | reject |
| 4.5 | +0.00041 | +0.00005 | -0.00426 | +0.00013 | -0.00923 | -0.00152 | reject |
| 5.0 | +0.00273 | +0.00018 | -0.00664 | -0.00153 | -0.00850 | -0.00115 | reject |
| 6.0 | +0.00450 | +0.00086 | -0.01261 | -0.00426 | -0.01824 | +0.00020 | reject |

- Interpretation: 4.0 remains the balanced optimum. Lower 3.5 improves voltage
  but lowers all F1 points; 4.5 significantly regresses 5 ms F1, and 5-6
  significantly regress the F1 curve. No candidate improves the full core set.
- Decision: retain coupling scale 4.0.
- Artifacts: `artifacts/diagnostics/iteration-028-final-recipe-coupling-scale/`.

### Iteration 029 premise - cumulative contribution decomposition

- Model and local recipe settings are now stable. Before the next long run, a
  factorial ablation can measure whether coupling and the promoted recipe each
  contribute under the same current diagnostics, and whether they interact.
- Decision: compare final coupled recipe against final recipe without coupling,
  old recipe with coupling, and old recipe without coupling. All models retain
  exactly 69,087 parameters and the same 120k presentations/updates; candidate
  loss compute is reported.

### Iteration 029 - 2026-07-12

- Regime: cumulative 2x2 factorial ablation.
- Controls: identical 69,087 parameters and initialization, frozen balanced
  120k schedule, batch 32, 3,750 updates, AdamW cosine, and paired diagnostics.
  The factors are fixed coupling scale 4 versus 0 and the promoted recipe
  versus the old burn-in/objective recipe.

| candidate vs final | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final, no coupling | +0.01097 | -0.00018 | -0.00664 | -0.00459 | -0.01088 | +0.00578 |
| old recipe, coupling | +0.03268 | +0.00035 | -0.00188 | -0.00207 | -0.00458 | +0.00432 |
| old recipe, no coupling | +0.04747 | +0.00003 | -0.01056 | -0.00576 | -0.01417 | +0.01309 |

- Pairwise old-recipe check: removing coupling significantly worsens voltage
  SERA by 0.01480 mV, F1 area by 0.00868, 5 ms F1 by 0.00960, and intervention
  SERA by 0.00877 mV; dynamics and exact F1 are uncertain.
- Interpretation: both promoted factors contribute independently. The training
  recipe supplies most of the voltage gain, while coupling supplies a robust
  F1/intervention gain under both recipes. Their interaction is small relative
  to the main effects. The final model is best on every primary metric.
- Decision: retain coupling and the complete promoted training recipe. Carry
  the two single-factor ablations, but not the redundant double ablation, into
  iteration 30's 300k validation.
- Artifacts: `artifacts/diagnostics/iteration-029-cumulative-ablation/`.

### Iteration 030 premise - long-horizon cumulative validation

- A 120k improvement can reflect faster convergence rather than a lasting
  model advantage. The required tenth-iteration long run tests whether the two
  independently useful factors survive 300k presentations.
- Decision: train the final model, final recipe without coupling, and old
  recipe with coupling for 300k presentations on one shared balanced schedule,
  then run the full paired diagnostic panel. Perform the scheduled dead-code
  and failed-modification cleanup after interpreting the results.

### Iteration 030 - 2026-07-12

- Regime: scheduled 300k long-horizon validation and cleanup.
- Controls: identical 69,087 parameters and initialization, balanced frozen
  300k schedule, batch 32, 9,375 updates, AdamW cosine, and paired diagnostics.

| ablation vs final | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no coupling | +0.00651 | +0.00076 | -0.00381 | -0.00390 | +0.00007 | -0.00096 |
| old recipe | +0.03321 | +0.00012 | +0.00070 | -0.00381 | +0.00087 | +0.00512 |

- Interpretation: the promoted recipe is not merely an early-convergence
  accelerator. It significantly improves voltage and intervention SERA over
  the old recipe after 300k while F1 area/5 ms are tied and exact F1 is
  uncertainly higher. Coupling's broad 120k effect narrows with training, but
  it retains a significant voltage-SERA gain and uncertainly higher F1 area
  and exact F1 without a significant primary-metric regression.
- Decision: retain the complete promoted recipe and coupling scale 4.0.
- Cleanup: no failed mechanism from iterations 21-29 remained executable.
  Corrected the reusable panel's stale pre-loop default so direct callers now
  receive burn-in 50 and the promoted SERA/F1 objective; collapsed a duplicate
  coupling condition in the shared-readout path. Historical artifacts remain.
- Artifacts: `artifacts/diagnostics/iteration-030-long-cumulative-validation/`.

### Iteration 031 premise - intervention sensitivity audit

- The long model still has intervention gain 0.0045, response skill 0.0007,
  voltage gain 0.204, and lag-1 residual correlation 0.416. This is a dominant
  failure mode rather than a local optimizer issue.
- Decision: before modifying the model, quantify on training traces whether a
  parameter-neutral causal aggregate-input residual explains held-out soma
  residuals and response dynamics. Only run the architecture panel if the
  measured relationship supports it.

### Iteration 031 pre-run audit - 2026-07-12

- Audit: 2,048 morphology-balanced training windows, seven causal time
  constants, and separate excitatory/inhibitory aggregate traces.
- Result: the current model already correlates 0.81 with the strongest signed
  aggregate-input trace, while the best candidate skip (tau 50 ms) explains
  only 0.35% of remaining soma variance and essentially none of residual dV.
- Decision: reject the direct input skip without a 120k training run. The
  measured intervention failure instead motivates supervised paired training.
- Revised experiment: split each morphology-condition's eight intervention
  files by parity. Train only on even files and diagnose only on odd files.
  Compare standard paired-data augmentation with response-SERA weights 0.001,
  0.0025, and 0.005. All candidates use alternating paired rows, identical
  parameters/initialization, 120k trace presentations, batch 32, and shared
  loading. The base-only 120k baseline is cached once on the odd-file protocol.
- Audit artifact: `artifacts/diagnostics/iteration-031-input-residual-audit.json`.

### Iteration 031 - 2026-07-12

- Regime: disjoint paired-intervention supervision and direct response-SERA.
- Controls: 69,087 parameters, common initialization, even-file training and
  odd-file diagnostics, balanced 120k trace presentations, batch 32, 3,750
  updates, AdamW cosine, shared pair-preserving loading, and cached base-only
  baseline diagnostics.

| candidate vs base-only | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired standard | -0.04065 | -0.00050 | -0.01146 | -0.00316 | -0.02244 | -0.00251 |
| response 0.001 | -0.03913 | -0.00045 | -0.01226 | -0.00352 | -0.02004 | -0.00227 |
| response 0.0025 | -0.03833 | -0.00007 | -0.01003 | -0.00086 | -0.01436 | -0.00217 |
| response 0.005 | -0.03915 | -0.00021 | -0.01486 | -0.00131 | -0.02340 | -0.00180 |

- Interpretation: paired data significantly improves ordinary voltage SERA,
  but its intervention-SERA gain is only 0.0025 mV against a 17.57 mV error
  and it significantly regresses F1 area and 5 ms F1. It raises response gain
  for dropout/silencing conditions to 0.03-0.06, but temporal jitter remains
  at 0.004. Tested response weights add no benefit over plain augmentation.
- Validity limitation: response-loss candidates use 29.1-29.3 s model compute
  versus 22.9 s for the zero-weight control because the control skips the
  response calculation. Presentations, updates, and parameters are matched,
  but the explicit-loss comparison must be repeated with matched compute.
- Decision: do not promote paired augmentation. Iteration 32 repeats the
  response weights with a zero-weight response-compute control and adds one
  high-weight feasibility point before closing the mechanism.
- Artifacts: `artifacts/diagnostics/iteration-031-paired-intervention-training/`.

### Iteration 032 premise - compute-matched response-SERA feasibility

- Direct response weights through 0.005 fail to move held-out response, but a
  larger coefficient can distinguish insufficient gradient scale from a
  structural inability to learn intervention deltas.
- Decision: compute response SERA for every candidate, including weight zero,
  and compare weights 0.001, 0.0025, 0.005, and 0.02 on the same paired frozen
  schedule. Cache the iteration-31 paired-standard output as baseline.

### Iteration 032 - 2026-07-12

- Regime: compute-matched direct response-SERA feasibility panel.
- Controls: every candidate computes the response-SERA graph, takes 28.7-29.5
  s model compute, and shares 69,087 parameters, initialization, paired data,
  120k presentations, batch 32, 3,750 updates, optimizer, and diagnostics.

| response weight vs compute control | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | +0.00152 | +0.00005 | -0.00080 | -0.00035 | +0.00240 | +0.00024 |
| 0.0025 | +0.00231 | +0.00043 | +0.00147 | +0.00232 | +0.00813 | +0.00035 |
| 0.005 | +0.00130 | +0.00022 | -0.00463 | +0.00164 | -0.00288 | +0.00065 |
| 0.02 | -0.00626 | +0.00013 | -0.00744 | +0.00051 | -0.00750 | +0.00124 |

- Interpretation: the zero-weight compute control reproduces the cached model
  to numerical precision, validating the control. No direct response weight
  improves held-out response; 0.005 and 0.02 significantly regress response,
  and 0.02 also regresses F1 area. A separate 60k-window audit confirms every
  paired window carries substantial teacher response, so random-window
  dilution does not explain the failure.
- Decision: reject paired intervention augmentation and direct response-SERA.
  Retain the code as experimental infrastructure until the iteration-40
  cleanup, but return the promoted model/data recipe to base-only training.
- Artifacts: `artifacts/diagnostics/iteration-032-response-sera-compute-matched/`
  and `artifacts/diagnostics/iteration-033-response-window-audit.json`.

### Iteration 033 premise - branch routing overlap

- The current morphology route maps 1,278 channels into 45 x 100 = 4,500
  overlapping synapse slots before branch summation. This produces broad,
  correlated branch aggregates and can erase channel-specific perturbations;
  only 29 slots per branch are required to cover every input once.
- Decision: map 29, 40, 60, and 80 slots per branch against the reproduced
  100-slot route. Branch count, projection size, all 69,087 trainable
  parameters, initialization, base-only 120k schedule, updates, and optimizer
  remain fixed. Candidates use no more compute than the control.

### Iteration 033 - 2026-07-12

- Regime: parameter-matched branch-routing overlap map.
- Controls: exact 69,087 parameters, common initialization, balanced frozen
  120k base schedule, batch 32, 3,750 updates, optimizer, and diagnostics.
  Candidate compute is 20.84-21.57 s versus 23.17 s for route 100.

| slots/branch | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | -0.03770 | -0.00007 | -0.00890 | -0.00502 | -0.01172 | +0.00109 | reject tradeoff |
| 40 | +0.00381 | +0.00208 | -0.01250 | -0.00613 | -0.02142 | +0.00402 | reject |
| 60 | -0.00206 | -0.00276 | +0.00011 | +0.00101 | -0.00571 | -0.00202 | provisional |
| 80 | +0.00011 | -0.00070 | -0.00587 | -0.00343 | -0.00650 | -0.00270 | reject |

- Interpretation: route 60 significantly improves dynamics and intervention
  SERA, with favorable uncertain voltage/F1-area/exact deltas and only an
  uncertain 5 ms decrease. It also removes 40% of routing slots and is 8.9%
  faster in measured model compute. More aggressive reduction produces a
  voltage/F1 frontier rather than a balanced gain.
- Decision: provisionally select 60 slots per branch and replicate route 60
  versus route 100 at seed 20260712 before changing the reusable default.
- Artifacts: `artifacts/diagnostics/iteration-033-routing-overlap/`.

### Iteration 034 premise - route-60 replication

- The route-60 gain is mechanistically plausible and favorable on five of six
  primary point estimates, but its significant SERA effects are small enough
  to require a second training seed.
- Decision: rerun only route 100 and route 60 on a seed-20260712 frozen 120k
  schedule with identical parameters, updates, optimizer, and diagnostics.

### Iteration 034 - 2026-07-12

- Regime: second-seed route-60 replication.
- Controls: seed 20260712, exact 69,087 parameters, balanced frozen 120k
  schedule, batch 32, 3,750 updates, optimizer, and paired diagnostics.
- Result: route 60 regresses voltage SERA by 0.02684 mV, intervention SERA by
  0.00261 mV, and 5 ms F1 by 0.02029 significantly. Dynamics is neutral; F1
  area and exact F1 are uncertainly lower. Route 60 remains 8.0% faster.
- Interpretation: the seed-1 SERA gain reverses on seed 2 and is accompanied by
  a large replicated-direction F1 loss. Efficiency does not compensate for
  the quality regression under the stated core metrics.
- Decision: reject route 60 and retain 100 slots per branch.
- Artifacts: `artifacts/diagnostics/iteration-034-route-60-replication/`.

### Iteration 035 premise - exact-budget branch granularity

- Reducing overlap at fixed 45 branches does not robustly help, but it leaves
  the 45-dimensional branch-sum bottleneck intact. Exact parameter-budget
  search found finer branch partitions that trade Mamba width/state capacity
  for input granularity without adding any trainable parameter.
- Decision: compare exact-69,087-parameter allocations (60 branches, width 58,
  state 30), (91, 65, 9), (122, 66, 4), and (143, 63, 10) against the current
  (45, 64, 16) model. Each candidate uses the minimum slots needed to cover
  all 1,278 inputs, the same 120k schedule and updates, and reports compute.

### Iteration 035 - 2026-07-12

- Regime: exact-parameter-budget spatial/temporal allocation panel.
- Controls: every model has exactly 69,087 trainable parameters, balanced
  frozen 120k data, batch 32, 3,750 updates, optimizer, and diagnostics.
  Candidates take 20.85-21.21 s model compute versus 23.16 s for control.

| allocation (branches/width/state) | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 60/58/30 | +0.00073 | -0.00223 | -0.01579 | -0.00046 | -0.02579 | -0.00059 | reject |
| 91/65/9 | +0.02287 | +0.00068 | +0.01923 | +0.00240 | +0.02935 | +0.00848 | reject tradeoff |
| 122/66/4 | -0.02639 | -0.00102 | +0.00392 | +0.00544 | -0.00153 | -0.00173 | provisional |
| 143/63/10 | -0.02436 | +0.00498 | +0.00162 | +0.00103 | -0.00226 | -0.00321 | reject dynamics |

- Interpretation: 122/66/4 is balanced: significantly better voltage SERA,
  favorable uncertain dynamics/response/F1-area/exact deltas, essentially
  flat 5 ms F1, exact parameter parity, and 9.9% less compute. The panel also
  shows meaningful frontiers rather than uniform gains at 91 and 143 branches.
- Decision: provisionally select 122/66/4 and replicate against 45/64/16 on
  seed 20260712 before changing reusable defaults.
- Artifacts: `artifacts/diagnostics/iteration-035-exact-budget-branch-granularity/`.

### Iteration 036 premise - 122-branch allocation replication

- The 122-branch candidate is the first exact-budget reallocation with no
  adverse primary point estimate except a negligible 5 ms delta. Its voltage
  gain is large enough to demand an independent schedule/initialization seed.
- Decision: rerun current 45/64/16 versus 122/66/4 at seed 20260712 with exact
  parameter, presentation, update, optimizer, and diagnostic matching.

### Iteration 036 - 2026-07-12

- Regime: second-seed exact-budget 122/66/4 replication.
- Controls: seed 20260712, exact 69,087 parameters, balanced frozen 120k
  schedule, batch 32, 3,750 updates, optimizer, and diagnostics.
- Result: 122/66/4 significantly improves voltage SERA by 0.01648 mV and
  dynamics SERA by 0.00385 mV/ms. F1 area and exact F1 rise uncertainly, 5 ms
  F1 falls 0.00413 uncertainly, and compute falls 9.4%. Intervention SERA,
  however, significantly regresses by 0.00724 mV.
- Interpretation: the ordinary voltage/dynamics and efficiency gains replicate,
  but the intervention axis reverses from a small seed-1 improvement to a
  significant seed-2 regression. This fails the balanced core acceptance bar.
- Decision: reject 122/66/4 and retain 45/64/16.
- Artifacts: `artifacts/diagnostics/iteration-036-branch-122-replication/`.

### Iteration 037 premise - complete exact-budget allocation set

- The exact 69,087-parameter search produced six non-baseline allocations.
  Four are now tested; the remaining 101/56/32 and 118/58/25 points allocate
  substantially more temporal state than the response-unstable 122/66/4 point.
- Decision: test the two remaining exact solutions against 45/64/16 on seed 1.
  This closes the finite exact-budget granularity set without arbitrary shape
  search; presentations, updates, parameters, optimizer, and diagnostics match.

### Iteration 037 - 2026-07-12

- Regime: completion of the finite exact-budget allocation set.
- Controls: exact 69,087 parameters, common seed-1 frozen 120k schedule, batch
  32, 3,750 updates, optimizer, and diagnostics. Alternatives take 20.83 s
  model compute versus 23.01 s for control.

| allocation | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 101/56/32 | +0.01000 | +0.00120 | +0.00727 | -0.00178 | +0.01323 | +0.00348 | reject SERA |
| 118/58/25 | -0.00720 | +0.00160 | -0.00359 | -0.00365 | -0.00807 | -0.00149 | provisional |

- Interpretation: 101/56/32 significantly regresses voltage and intervention
  SERA despite favorable F1 area/5 ms. The 118/58/25 point significantly
  improves voltage and has favorable intervention response, but dynamics and
  all F1 estimates are lower and uncertain. It has no significant regression
  and is 9.5% faster, so one replication is warranted.
- Decision: replicate 118/58/25 on seed 20260712. This is the final routing
  allocation under consideration; reject the family if the F1/SERA balance
  does not replicate.
- Artifacts: `artifacts/diagnostics/iteration-037-complete-exact-budget-set/`.

### Iteration 038 premise - 118-branch final replication

- 118/58/25 is the only remaining exact-budget point with favorable voltage
  and intervention estimates and no significant primary regression, but its
  F1 point estimates are consistently lower.
- Decision: rerun 45/64/16 versus 118/58/25 on seed 20260712 with exact
  parameter, presentation, update, optimizer, and diagnostic matching.

### Iteration 038 - 2026-07-12

- Regime: second-seed 118/58/25 replication.
- Controls: seed 20260712, exact 69,087 parameters, frozen balanced 120k
  schedule, batch 32, 3,750 updates, optimizer, and diagnostics.
- Result: 118/58/25 significantly improves intervention SERA by 0.00328 mV.
  Dynamics, F1 area, and 5 ms F1 improve uncertainly; exact F1 falls 0.00125
  uncertainly. Voltage SERA significantly regresses by 0.00689 mV. Compute is
  8.7% lower.
- Two-seed means: voltage -0.00015 mV, dynamics +0.00054 mV/ms, F1 area
  +0.00204, exact F1 -0.00245, 5 ms F1 +0.00027, and intervention SERA
  -0.00239 mV. Intervention and efficiency improve consistently, while
  opposing significant voltage effects cancel and exact F1 trends lower.
- Decision: run one predeclared third-seed tie-break. Promote to the iteration-40
  long panel only if mean response stays favorable, no F1-area/5 ms regression
  emerges, and the third seed does not significantly regress voltage.
- Artifacts: `artifacts/diagnostics/iteration-038-branch-118-replication/`.

### Iteration 039 premise - 118-branch third-seed tie-break

- The candidate has a consistent intervention/efficiency benefit but unstable
  voltage and uncertain exact-F1 cost. A third seed is more informative than
  another architecture variant before the scheduled long run.
- Decision: rerun 45/64/16 versus 118/58/25 at seed 20260713 under the same
  exact controls and apply the predeclared acceptance rule above.

### Iteration 039 - 2026-07-12

- Regime: predeclared third-seed tie-break.
- Controls: seed 20260713, exact 69,087 parameters, frozen balanced 120k
  schedule, batch 32, 3,750 updates, optimizer, and diagnostics.
- Result: 118/58/25 significantly improves voltage SERA by 0.03030 mV,
  intervention SERA by 0.00494 mV, and F1 area by 0.00843. Dynamics improves
  0.00050 uncertainly, exact F1 is unchanged, 5 ms F1 rises 0.01092
  uncertainly, and model compute falls 8.9%.
- Three-seed means: voltage -0.01020 mV, dynamics +0.00019 mV/ms, F1 area
  +0.00417, exact F1 -0.00163, 5 ms F1 +0.00382, and intervention SERA
  -0.00324 mV. The candidate improves four core axes and efficiency, ties
  dynamics, and has a small exact-F1 cost without a significant exact
  regression on any seed.
- Decision: the candidate passes the predeclared gate. Carry 118/58/25 versus
  45/64/16 into iteration 40's 300k validation; promote reusable defaults only
  if the long run preserves response and the balanced F1/SERA profile.
- Artifacts: `artifacts/diagnostics/iteration-039-branch-118-third-seed/`.

### Iteration 040 premise - long exact-budget architecture gate

- The exact-budget 118/58/25 allocation has a replicated 120k response,
  voltage, F1-area/5 ms, and efficiency benefit, but longer optimization can
  change its width/state tradeoff and exact-F1 behavior.
- Decision: train 45/64/16 and 118/58/25 for 300k presentations on the shared
  seed-1 schedule. Promote only if no core metric significantly regresses and
  the response/voltage or F1-area benefit persists. Then perform the scheduled
  cleanup, including removal of failed paired-response training code.

### Iteration 040 - 2026-07-12

- Regime: scheduled 300k exact-budget architecture gate and cleanup.
- Controls: exact 69,087 parameters, common initialization, balanced frozen
  300k schedule, batch 32, 9,375 updates, optimizer, and paired diagnostics.

| 118/58/25 vs 45/64/16 | delta |
| --- | ---: |
| voltage SERA | -0.00590 mV |
| dynamics SERA | +0.00163 mV/ms |
| F1 area | +0.00824 |
| exact F1 | -0.00919 |
| 5 ms F1 | +0.01561 significant |
| intervention SERA | +0.00042 mV |

- Interpretation: the long run preserves the F1-across-time and voltage point
  benefits, significantly improves 5 ms F1, has no significant core
  regression, and reduces model compute from 57.99 to 53.16 s (8.3%).
  Intervention/dynamics are tied. Exact-only F1 is lower but uncertain; the
  predeclared objective prioritizes F1 across time rather than exact-only F1.
- Decision: promote 118 branches, 11 slots per branch, model width 58, and
  state size 25 as the reusable AxoMamba default. Keep coupling 4.0 and the
  promoted training recipe unchanged.
- Cleanup: removed the failed paired-response objective and intervention-pair
  training/view execution paths from production code and tests. Historical
  manifests/diagnostics remain. Renamed the stale
  `configs/axomamba_45x64.json` path to `configs/axomamba.json` across code.
- Artifacts: `artifacts/diagnostics/iteration-040-long-branch-118-gate/`.

### Iteration 041 premise - promoted-allocation coupling recheck

- The successful fixed voltage-to-spike coupling scale 4.0 was optimized in
  the old 45/64/16 allocation. The promoted 118/58/25 model changes latent
  width/state and improves tolerant F1, so coupling may now sit at a different
  SERA/F1 balance point.
- Decision: remap nearby coupling scales around 4.0 under the new durable
  allocation, preserving exact parameters, data, updates, optimizer, and
  compute. Start with 2.0, 3.0, 4.0 control, 5.0, and 6.0 at 120k.

### Iteration 041 - 2026-07-12

- Regime: coupling remap under promoted 118/58/25 allocation.
- Controls: exact 69,087 parameters, common initialization, frozen 120k
  schedule, batch 32, 3,750 updates, optimizer, and diagnostics.

| coupling | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | -0.01022 | +0.00005 | -0.00362 | -0.00264 | -0.00242 | -0.00350 | provisional |
| 3 | -0.01997 | -0.00092 | -0.00686 | -0.00203 | -0.00726 | -0.00456 | reject tradeoff |
| 5 | +0.00738 | -0.00124 | -0.00621 | -0.00289 | -0.00630 | +0.00002 | reject |
| 6 | +0.00763 | -0.00196 | -0.00795 | +0.00104 | -0.01197 | +0.00257 | reject |

- Interpretation: scale 2 significantly improves voltage and intervention
  SERA with neutral dynamics and only uncertain modest F1 decreases. Scale 3
  extends SERA gains but moves farther down the F1 frontier. Higher scales
  significantly regress voltage/F1 and, at 6, intervention response.
- Decision: provisionally select coupling 2.0 and replicate versus 4.0 at seed
  20260712 before modifying `configs/axomamba.json`.
- Artifacts: `artifacts/diagnostics/iteration-041-promoted-coupling-remap/`.

### Iteration 042 premise - coupling-2 replication

- Scale 2 is the balanced point in the promoted-allocation map, but its F1
  estimates are all lower and seed sensitivity has repeatedly mattered.
- Decision: rerun coupling 4 versus 2 at seed 20260712 with exact parameter,
  presentation, update, optimizer, compute, and diagnostic matching.

### Iteration 042 - 2026-07-12

- Regime: second-seed coupling-2 replication under 118/58/25.
- Controls: seed 20260712, exact parameters, frozen balanced 120k schedule,
  batch 32, 3,750 updates, optimizer, and diagnostics.
- Result: coupling 2 significantly regresses intervention SERA by 0.00376 mV.
  Voltage and dynamics are tied; F1 area, exact, and 5 ms are all lower
  uncertainly (5 ms -0.00993).
- Interpretation: seed-1 voltage/response gains do not replicate and response
  reverses significantly. The lower coupling is not robust.
- Decision: reject coupling 2 and retain coupling 4.0 in AxoMamba defaults.
- Artifacts: `artifacts/diagnostics/iteration-042-coupling-2-replication/`.

### Iteration 043 premise - promoted-architecture objective retune

- Direct SERA 0.0075 and F1-curve 0.05 were tuned under 45/64/16. The promoted
  118/58/25 allocation changes state width and the observed F1/SERA frontier,
  while both metric-aligned loss graphs remain the explicit optimization target.
- Decision: compare SERA weights 0.005, 0.01, 0.015 and F1 weights 0.04, 0.06,
  0.075 against 0.0075/0.05, plus a joint 0.01/0.06 point. Every candidate
  computes identical losses and shares parameters, data, updates, and optimizer.

### Iteration 043 - 2026-07-12

- Regime: metric-aligned objective retune under promoted architecture.
- Controls: exact parameters, common initialization, frozen 120k schedule,
  batch 32, 3,750 updates, optimizer, identical loss graphs, and diagnostics.

| objective change | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SERA 0.005 | +0.02655 | +0.00029 | -0.00066 | +0.00122 | +0.00060 | +0.00369 | reject |
| SERA 0.01 | -0.02296 | -0.00030 | -0.00354 | -0.00282 | -0.00386 | -0.00352 | provisional |
| SERA 0.015 | -0.06383 | -0.00073 | -0.01138 | -0.00282 | -0.01171 | -0.00758 | reject tradeoff |
| F1 0.04 | -0.00048 | -0.00051 | -0.00587 | +0.00072 | -0.00328 | -0.00062 | reject |
| F1 0.06 | -0.00095 | +0.00043 | -0.00474 | -0.00047 | -0.00184 | -0.00002 | reject |
| F1 0.075 | -0.00056 | +0.00078 | -0.00387 | +0.00032 | -0.00256 | +0.00036 | reject |
| SERA 0.01 + F1 0.06 | -0.02327 | -0.00007 | -0.00402 | +0.00002 | -0.00558 | -0.00293 | reject |

- Interpretation: direct SERA 0.01 significantly improves voltage and
  intervention SERA with neutral dynamics and only uncertain modest F1 costs.
  SERA 0.015 crosses into significant F1-area/5 ms regression. Lower SERA
  regresses voltage/response; changing F1 weight does not improve the F1 curve,
  and the joint point does not repair the frontier.
- Decision: provisionally select direct SERA 0.01 and replicate versus 0.0075
  at seed 20260712 before changing the durable training helper.
- Artifacts: `artifacts/diagnostics/iteration-043-promoted-objective-retune/`.

### Iteration 044 premise - direct-SERA 0.01 replication

- SERA 0.01 is the strongest non-significantly-harmful point, but coupling and
  routing candidates have shown that seed-1 SERA gains can reverse.
- Decision: rerun 0.0075 versus 0.01 at seed 20260712 with exact parameter,
  presentation, update, optimizer, compute, and diagnostic matching.

### Iteration 044 - 2026-07-12

- Regime: predeclared second-seed direct-SERA replication.
- Controls: seed 20260712, exact 69,087 parameters, common initialization,
  frozen balanced 120k schedule, batch 32, 3,750 updates, identical loss
  graphs, optimizer, and diagnostics.

| 0.01 vs 0.0075 | delta |
| --- | ---: |
| voltage SERA | -0.02836 mV significant |
| dynamics SERA | -0.00053 mV/ms |
| F1 area | -0.00184 |
| exact F1 | -0.00234 |
| 5 ms F1 | -0.00205 |
| intervention SERA | -0.00341 mV significant |

- Interpretation: the seed-1 voltage and intervention gains replicate with no
  significant F1 or dynamics regression and no new attention trigger. Across
  the two seeds, mean deltas are -0.02566 mV voltage SERA, -0.00041 mV/ms
  dynamics SERA, -0.00269 F1 area, -0.00258 exact F1, -0.00295 5 ms F1, and
  -0.00346 mV intervention SERA. The uncertain small F1 cost is consistent,
  but the two primary SERA gains are significant on both seeds.
- Decision: promote direct-SERA weight 0.01 in the reusable AxoMamba training
  objective. The candidate passes the predeclared gate because both SERA gains
  replicate, no core F1 metric significantly regresses, and the failure-mode
  panel is unchanged.
- Artifacts: `artifacts/diagnostics/iteration-044-sera-0p01-replication/`.

### Iteration 045 premise - promoted-recipe cosine-rate recheck

- The 6e-4 peak rate was robust under the old 45/64/16 allocation and SERA
  weight 0.0075, but both latent allocation and metric-aligned gradient scale
  have changed. This can shift the stable optimization step size.
- Decision: rerun the previously bounded 4.5e-4, 6e-4, 7.5e-4, and 9e-4 panel
  under 118/58/25 and direct-SERA 0.01. Parameters, loss graphs, data, updates,
  optimizer family, and diagnostics remain exact matched controls.

### Iteration 045 - 2026-07-12

- Regime: bounded optimizer tuning after two durable recipe changes.
- Controls: exact 69,087 parameters, common initialization, direct-SERA 0.01,
  frozen 120k schedule, batch 32, 3,750 updates, AdamW cosine, identical loss
  graphs, and diagnostics.

| peak rate | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4.5e-4 | -0.01534 | -0.00040 | -0.01606 | +0.00075 | -0.02492 | -0.00266 | reject tradeoff |
| 7.5e-4 | -0.01014 | -0.00102 | +0.00182 | +0.00222 | +0.00325 | +0.00025 | secondary |
| 9e-4 | -0.01956 | -0.00160 | -0.00017 | +0.00317 | +0.00397 | -0.00203 | provisional |

- Interpretation: 9e-4 significantly improves voltage and intervention SERA,
  while dynamics, exact F1, and 5 ms F1 all improve uncertainly and F1 area is
  flat. It introduces no attention change. The 7.5e-4 point is milder and does
  not improve response; 4.5e-4 significantly regresses F1 area and 5 ms F1.
  Unlike the old-architecture rate sweep, 9e-4 no longer pays a tolerant-F1
  cost, consistent with the changed allocation and stronger SERA gradients.
- Decision: provisionally select 9e-4 and replicate against 6e-4 at seed
  20260712. Promote only if the SERA gains replicate without a significant F1
  regression or new trigger.
- Artifacts: `artifacts/diagnostics/iteration-045-promoted-learning-rate/`.

### Iteration 046 premise - 9e-4 cosine-rate replication

- The higher rate improves five of six point estimates under the promoted
  model/objective, but its old-architecture voltage gain reversed on seed 2.
- Decision: rerun 6e-4 versus 9e-4 at seed 20260712 under exact parameter,
  presentation, update, objective, optimizer-family, compute, and diagnostic
  controls.

### Iteration 046 - 2026-07-12

- Regime: second-seed optimizer replication.
- Controls: seed 20260712, exact 69,087 parameters, common initialization,
  promoted objective, frozen balanced 120k schedule, batch 32, 3,750 updates,
  AdamW cosine, and diagnostics.
- Result: 9e-4 improves voltage SERA significantly by 0.00490 mV and F1 area
  by 0.00249 uncertainly, but intervention SERA significantly regresses by
  0.00506 mV. Dynamics regresses 0.00030 mV/ms uncertainly; exact F1 rises
  0.00182 and 5 ms F1 falls 0.00064, both uncertain.
- Interpretation: the voltage gain replicates, unlike under the old model, but
  response reverses from a significant gain to a larger significant loss.
  Two-seed means improve voltage by 0.01223 mV, dynamics by 0.00065 mV/ms,
  F1 area by 0.00116, exact F1 by 0.00250, and 5 ms F1 by 0.00166, while
  response SERA regresses by 0.00152 mV. The response failure violates the
  predeclared promotion rule.
- Decision: reject 9e-4 and retain 6e-4. Test the milder 7.5e-4 secondary lead
  once on seed 2 because it improved voltage, dynamics, and all F1 points on
  seed 1 without the aggressive response movement.
- Artifacts: `artifacts/diagnostics/iteration-046-learning-rate-replication/`.

### Iteration 047 premise - 7.5e-4 balanced-rate replication

- Rate 7.5e-4 is the only remaining point from iteration 45 with favorable
  voltage, dynamics, and every F1 estimate; response was tied rather than
  improved. It may be a more robust convergence-speed point than 9e-4.
- Decision: rerun 6e-4 versus 7.5e-4 at seed 20260712. Promote only if the
  balanced SERA/F1 profile replicates with no significant response regression.

### Iteration 047 - 2026-07-12

- Regime: final bounded optimizer replication.
- Controls: seed 20260712, exact 69,087 parameters, common initialization,
  promoted objective, frozen balanced 120k schedule, batch 32, 3,750 updates,
  AdamW cosine, and diagnostics.
- Result: 7.5e-4 significantly improves voltage SERA by 0.00719 mV but
  significantly regresses intervention SERA by 0.00205 mV. Dynamics and 5 ms
  F1 are effectively tied; F1 area rises 0.00010 and exact F1 0.00133,
  uncertainly.
- Interpretation: like 9e-4, the milder rate preserves voltage convergence but
  fails response robustness on seed 2. Its seed-1 balanced profile does not
  replicate, and the predeclared response gate fails.
- Decision: reject 7.5e-4, retain 6e-4, and close local cosine-rate tuning.
- Artifacts: `artifacts/diagnostics/iteration-047-learning-rate-7p5-replication/`.

### Iteration 048 premise - causal latent lead compensation

- The promoted baseline has residual lag 0.414, high-dV/dt error 2.85x, and
  intervention gain about 0.005. Fixed output scaling failed previously,
  suggesting temporal compression rather than simple amplitude calibration.
- Hypothesis: a parameter-neutral causal readout transform
  `h_t + alpha * (h_t - h_{t-1})` can restore fast latent changes, reducing
  residual lag/high-dVdt error and improving dynamics, intervention response,
  or spike timing without changing the learned state-space trunk.
- Decision: compare alpha 0.1, 0.25, 0.5, and 1.0 against zero. Every candidate
  executes the same lead operation and has exact parameters, data, updates,
  objective, optimizer, and diagnostics.

### Iteration 048 - 2026-07-12

- Regime: mechanism-driven, parameter-neutral temporal intervention.
- Controls: exact 69,087 parameters and computation graph, common
  initialization, frozen 120k schedule, batch 32, 3,750 updates, promoted
  objective/optimizer, and diagnostics.

| lead alpha | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | -0.00449 | -0.00031 | +0.00652 | +0.00487 | +0.00769 | -0.00093 | provisional |
| 0.25 | -0.00993 | -0.00047 | -0.00790 | +0.00172 | -0.01790 | -0.00119 | reject tradeoff |
| 0.5 | -0.01321 | -0.00208 | -0.00875 | -0.00087 | -0.01519 | +0.00097 | reject |
| 1.0 | -0.01329 | -0.00463 | -0.00597 | +0.00180 | -0.00725 | +0.00270 | reject |

- Interpretation: alpha 0.1 significantly improves voltage SERA,
  intervention SERA, F1 area, exact F1, and 5 ms F1; dynamics also improves
  uncertainly. It adds no trigger. The premise diagnostics move in the
  hypothesized direction but only slightly: residual lag -0.00010,
  high-dVdt ratio -0.00023, near-spike ratio -0.00036, and intervention gain
  +0.00003. Thus the transform may improve trainability more than directly
  correcting the reported lag. Larger scales monotonically improve voltage
  and, at 1.0, dynamics, but cross into significant tolerant-F1 or response
  regressions.
- Decision: provisionally select alpha 0.1 and replicate at seed 20260712.
  Require no significant core regression and preservation of the broad
  SERA/F1/response benefit before carrying it into iteration 50's long gate.
- Artifacts: `artifacts/diagnostics/iteration-048-readout-lead/`.

### Iteration 049 premise - lead-0.1 replication

- Alpha 0.1 is the first post-architecture candidate to significantly improve
  five core metrics at once, but its direct diagnostic mechanism effect is
  small and seed sensitivity remains the dominant false-positive mode.
- Decision: rerun alpha 0 versus 0.1 at seed 20260712 under exact parameter,
  operation, presentation, update, objective, optimizer, and diagnostic
  controls. If it passes, iteration 50 will test it at 300k and decompose it
  from the promoted SERA objective before scheduled cleanup.

### Iteration 049 - 2026-07-12

- Regime: second-seed causal-lead replication.
- Controls: seed 20260712, exact 69,087 parameters and operation graph, common
  initialization, frozen balanced 120k schedule, batch 32, 3,750 updates,
  promoted objective/optimizer, and diagnostics.
- Result: alpha 0.1 significantly improves voltage SERA by 0.00152 mV.
  Dynamics improves 0.00039 mV/ms, F1 area 0.00286, exact F1 0.00317, and
  5 ms F1 0.00326, all uncertainly. Intervention SERA regresses only
  0.00027 mV uncertainly, and no trigger changes.
- Interpretation: the seed-1 five-significant-metric effect shrinks but does
  not reverse on any SERA/F1 axis. Two-seed means improve voltage SERA by
  0.00301 mV, dynamics SERA by 0.00035 mV/ms, F1 area by 0.00469, exact F1 by
  0.00402, 5 ms F1 by 0.00547, and intervention SERA by 0.00033 mV. This is a
  replicated balanced profile rather than a single-seed response tradeoff.
- Decision: the candidate passes the 120k gate. Carry alpha 0.1 into iteration
  50's 300k 2x2 factorial with direct-SERA weights 0.0075 and 0.01, then
  promote or remove the new path during scheduled cleanup.
- Artifacts: `artifacts/diagnostics/iteration-049-readout-lead-replication/`.

### Iteration 050 premise - long lead/objective factorial gate

- Direct-SERA 0.01 and readout lead 0.1 were each promoted from replicated
  120k evidence, but neither has been validated jointly at 300k. Longer
  optimization may erase the lead's convergence benefit or expose an
  objective interaction.
- Decision: train the exact 2x2 factorial of SERA weight 0.0075/0.01 and lead
  alpha 0/0.1 for 300k presentations. Use 0.01/0 as the current baseline.
  Promote lead 0.1 only if 0.01/0.1 has no significant core regression and
  preserves a balanced F1/SERA benefit. Retain SERA 0.01 only if its long-run
  comparison remains non-harmful. Perform iteration-50 dead-code cleanup and
  full verification after the decision.

### Iteration 050 - 2026-07-12

- Regime: scheduled 300k exact-control factorial gate and cleanup.
- Controls: exact 69,087 parameters and operation graph, common initialization,
  balanced frozen 300k schedule, batch 32, 9,375 updates, optimizer, and
  diagnostics. The four candidates form the exact factorial of SERA weight
  0.0075/0.01 and lead alpha 0/0.1.

| candidate vs SERA 0.01 / lead 0 | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SERA 0.01 / lead 0.1 | +0.00258 | -0.00084 | -0.00417 | -0.00399 | -0.00567 | -0.00073 | reject long |
| SERA 0.0075 / lead 0 | +0.02239 | +0.00026 | -0.00220 | -0.00425 | +0.00101 | +0.00358 | reject objective |
| SERA 0.0075 / lead 0.1 | +0.02508 | -0.00081 | +0.00079 | -0.00331 | +0.00166 | +0.00261 | reject objective |

- Interpretation: direct-SERA 0.01 is decisively durable. Reverting to 0.0075
  significantly regresses voltage and intervention SERA, with no compensating
  significant F1 benefit. Lead 0.1 at the promoted objective significantly
  improves intervention SERA and improves dynamics uncertainly, but
  significantly regresses voltage and lowers every F1 estimate. Under SERA
  0.0075, the lead similarly trades about +0.00268 mV voltage for improved
  dynamics, response, and F1 point estimates. Thus the 120k lead behaves as a
  convergence aid or frontier shift, not a durable balanced improvement.
- Decision: retain direct-SERA 0.01 and reject readout lead 0.1 at the long
  gate. Keep the durable 118/58/25, coupling-4, burn-in-50, SERA-0.01,
  F1-0.05 recipe.
- Cleanup: removed the failed readout-lead configuration, forward operation,
  and test. Historical iteration 48-50 manifests and diagnostics remain. No
  other iteration 41-49 candidate introduced production code, so no unrelated
  model capability was removed.
- Verification: AxoSim passes 81 tests with the AxoBench source bridge;
  AxoBench's focused iteration evaluator passes 23 tests. The promoted model
  remains exactly 69,087 parameters at 118/58/25, with direct-SERA 0.01 and
  F1-curve 0.05.
- Artifacts: `artifacts/diagnostics/iteration-050-long-lead-objective-factorial/`.

### Iteration 051 premise - Mamba decay-spectrum initialization

- The model still exhibits residual lag, high-dVdt error, and weak
  intervention gain. Mamba's `A_log` initializes its continuous-time decay
  spectrum; shifting it changes initial state timescales without changing
  architecture, parameters, forward compute, data, or optimizer steps.
- Hypothesis: faster initial decays may improve spike timing/high-dVdt
  tracking, while slower decays may improve intervention memory. A symmetric
  panel can identify whether either mechanism survives end-to-end training.
- Decision: compare initial decay-rate multipliers 0.5, 1/sqrt(2), sqrt(2),
  and 2.0 against 1.0 by adding shifts -log(2), -log(sqrt(2)),
  +log(sqrt(2)), and +log(2) to every Mamba `A_log`. A unit test verifies that
  no other initialized tensor or parameter count changes.
- Fidelity correction: the first attempted execution was invalid because the
  runner's common-state copy overwrote constructor-time `A_log` shifts, making
  every candidate numerically identical. It is not logged as a result. The
  rerun applies exact named-parameter offsets after common-state matching; a
  regression test now protects this ordering.

### Iteration 051 - 2026-07-12

- Regime: Mamba-specific state-timescale initialization after a corrected
  runner-level fidelity check.
- Controls: exact 69,087 parameters, verified post-match `A_log` offsets,
  common non-`A_log` initialization, frozen 120k schedule, batch 32, 3,750
  updates, promoted objective/optimizer, and diagnostics.

| initial decay rate | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5x | -0.01939 | +0.00004 | -0.00255 | +0.00249 | -0.00304 | -0.00620 | provisional |
| 0.707x | -0.00798 | +0.00029 | -0.00246 | +0.00036 | -0.00324 | -0.00227 | secondary |
| 1.414x | +0.00040 | -0.00004 | -0.00170 | +0.00569 | -0.00310 | +0.00300 | reject tradeoff |
| 2.0x | +0.00021 | -0.00099 | -0.01304 | +0.00567 | -0.01888 | +0.00443 | reject |

- Interpretation: slower initialization coherently improves voltage and
  intervention SERA, while faster initialization moves toward exact F1 at the
  cost of tolerant F1 and response. The 0.5x candidate significantly improves
  voltage and response with only uncertain small F1-area/5 ms decreases and
  an exact-F1 increase. Its final `A_log` means remain about 0.686 below the
  control, so training does not erase the intervention. Residual lag falls
  0.00020 and intervention gain rises 0.00018; high-dVdt and near-spike ratios
  worsen slightly, consistent with the measured frontier.
- Decision: provisionally select 0.5x and replicate at seed 20260712. Promote
  only if the SERA gains replicate and no F1 metric significantly regresses.
- Artifacts: `artifacts/diagnostics/iteration-051-mamba-decay-initialization/`.

### Iteration 052 premise - half-rate Mamba decay replication

- The slower spectrum has a coherent mechanism and large SERA/response gains,
  but its tolerant-F1 point estimates are lower. Seed-2 replication must
  distinguish a robust balanced gain from another SERA/F1 frontier shift.
- Decision: rerun 1x versus 0.5x initial decay at seed 20260712 with exact
  post-match offsets, parameters, presentations, updates, optimizer, compute,
  and diagnostics.

### Iteration 052 - 2026-07-12

- Regime: second-seed Mamba-timescale replication.
- Controls: seed 20260712, exact post-match offsets, 69,087 parameters,
  common remaining initialization, frozen balanced 120k schedule, batch 32,
  3,750 updates, promoted objective/optimizer, and diagnostics.
- Result: half-rate decay significantly improves voltage SERA by 0.00389 mV,
  but significantly regresses intervention SERA by 0.00151 mV and 5 ms F1 by
  0.00844. Dynamics regresses 0.00100 mV/ms, F1 area 0.00526, and exact F1
  0.00044, all uncertainly.
- Interpretation: the seed-1 response gain reverses and the tolerant-F1 risk
  becomes significant. Uniformly slower state dynamics are not robust, even
  though their voltage benefit replicates.
- Decision: reject uniform half-rate decay and retain the standard Mamba
  initialization. Keep the generic, tested post-match offset infrastructure.
- Artifacts: `artifacts/diagnostics/iteration-052-mamba-half-decay-replication/`.

### Iteration 053 premise - layer-separated Mamba timescales

- Iteration 51 showed a coherent frontier: slower Mamba decays improve
  voltage/response, while faster decays improve exact timing/dynamics but hurt
  tolerant F1 and response. Applying one rate to both layers may erase the
  complementary benefit.
- Hypothesis: a slow first layer can integrate membrane history while a fast
  second layer restores event timing; the reverse ordering and one-layer-only
  controls distinguish hierarchy from total decay allocation.
- Decision: compare slow-first only, slow-second only, slow-first/fast-second,
  and fast-first/slow-second against standard initialization. Use exact
  +/-log(2) post-match offsets, identical parameters/compute/data/updates, and
  the promoted objective.

### Iteration 053 - 2026-07-12

- Regime: mechanistic layer-order decomposition of Mamba timescales.
- Controls: exact post-match offsets, 69,087 parameters, common remaining
  initialization, frozen 120k schedule, batch 32, 3,750 updates, promoted
  objective/optimizer, and diagnostics.

| timescale layout | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| slow first | -0.00370 | -0.00038 | -0.00775 | -0.00007 | -0.01263 | -0.00291 | reject tradeoff |
| slow second | -0.02747 | -0.00063 | -0.00519 | +0.00148 | -0.01387 | -0.00329 | reject tradeoff |
| slow first / fast second | -0.00612 | -0.00109 | -0.01538 | +0.00151 | -0.01986 | -0.00057 | reject |
| fast first / slow second | -0.01394 | +0.00109 | -0.00908 | +0.00541 | -0.01562 | +0.00023 | reject |

- Interpretation: slow-first and slow-second each significantly improve
  voltage and response, but both significantly regress 5 ms F1. Both mixed
  hierarchies significantly regress F1 area and 5 ms F1; the reverse ordering
  improves exact F1 but regresses dynamics. Layer order changes the frontier
  but does not combine its benefits.
- Decision: reject all layer-separated decay initializations, retain standard
  Mamba initialization, and close the timescale-initialization route.
- Artifacts: `artifacts/diagnostics/iteration-053-layer-separated-decay/`.

### Iteration 054 premise - soft-OR tolerant-event objective

- The current tolerant-F1 surrogate uses max pooling over predicted event
  probabilities, routing recall gradient only through the local maximum. This
  is a poor approximation when event mass is distributed around the correct
  time, and the model retains a large timing-tolerance gap.
- Hypothesis: soft-OR window probability `1 - product(1 - p_t)` gives dense
  nearby-event recall gradients while the existing precision denominator
  still penalizes duplicate event mass, improving actual F1 across time.
- Decision: blend soft-OR into the legacy max-pool F1 surrogate at weights
  0.25, 0.5, 0.75, and 1.0 against zero. Every candidate computes both losses,
  including the zero control, so loss graph/compute, parameters, data, updates,
  and optimizer are matched.

### Iteration 054 - 2026-07-12

- Regime: metric-aligned surrogate replacement with matched loss graphs.
- Controls: exact 69,087 parameters, common initialization, both F1 losses
  computed for every candidate, frozen 120k schedule, batch 32, 3,750 updates,
  promoted remaining objective/optimizer, and diagnostics.

| soft-OR mix | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | -0.00135 | +0.00007 | -0.00251 | +0.00536 | -0.00436 | -0.00005 | reject mechanism |
| 0.5 | -0.00367 | -0.00018 | -0.01111 | +0.00821 | -0.01705 | -0.00041 | reject |
| 0.75 | -0.00572 | -0.00065 | -0.03160 | +0.00951 | -0.04883 | -0.00072 | reject |
| 1.0 | -0.00627 | -0.00131 | -0.05186 | +0.01347 | -0.08128 | -0.00057 | reject |

- Interpretation: soft-OR monotonically improves exact F1, voltage, and
  dynamics, but monotonically collapses the actual one-to-one tolerance curve.
  At full weight, refractory violations rise from 0.173 to 0.684,
  default-threshold count falls from 1.49x to 0.57x, and count error worsens by
  0.120. The surrogate rewards clustered event mass that exact matching can
  partially use but tolerant one-to-one matching penalizes. Even mix 0.25
  lowers F1 area/5 ms and raises refractory violations. Computing both losses
  also raises control model compute from about 22.7 to 29.9 s.
- Decision: reject soft-OR as a standalone F1 surrogate. Use its diagnosed
  cluster failure for one bounded exclusion-loss test before pruning the path.
- Artifacts: `artifacts/diagnostics/iteration-054-soft-or-f1/`.

### Iteration 055 premise - soft-OR with refractory exclusion

- Soft-OR's measured failure is clustered probability mass: refractory
  violations and count error rise as the blend increases. A differentiable
  close-pair penalty directly targets expected pairs within five steps while
  preserving soft-OR's dense timing gradient.
- Scale audit: on eight frozen batches from the promoted checkpoint, the
  exclusion loss averages 2.72 per true event. Weights 0.00125, 0.0025, and
  0.005 contribute about 0.0034, 0.0068, and 0.0136 to total loss, below the
  existing F1 contribution near 0.043.
- Decision: compare exclusion-only 0.0025 and soft-OR-0.25 plus exclusion
  0/0.00125/0.0025/0.005 against the legacy control. Every candidate computes
  soft-OR and exclusion graphs, including zero-weight controls, with exact
  parameters/data/updates/optimizer matching.

### Iteration 055 - 2026-07-12

- Regime: mechanism repair of the soft-OR cluster failure with scale-audited,
  compute-matched exclusion losses.
- Controls: exact 69,087 parameters, common initialization, soft-OR and
  exclusion graphs computed for all six candidates, frozen 120k schedule,
  batch 32, 3,750 updates, and promoted remaining objective/optimizer.

| objective change | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exclusion 0.0025 only | +0.02147 | -0.00046 | -0.00980 | -0.00053 | -0.01291 | +0.00799 | reject |
| soft-OR 0.25 | -0.00135 | +0.00007 | -0.00251 | +0.00536 | -0.00436 | -0.00005 | reject |
| soft-OR 0.25 + exclusion 0.00125 | +0.02638 | -0.00193 | +0.01608 | +0.01032 | +0.02457 | +0.00655 | reject tradeoff |
| soft-OR 0.25 + exclusion 0.0025 | +0.02187 | -0.00024 | +0.00136 | -0.00226 | +0.00285 | +0.00844 | reject |
| soft-OR 0.25 + exclusion 0.005 | -0.00434 | +0.00134 | +0.00094 | -0.00228 | +0.00738 | +0.00139 | reject failure mode |

- Interpretation: exclusion at 0.00125 genuinely repairs the event metric,
  significantly improving F1 area, exact F1, and 5 ms F1, but it
  significantly regresses voltage and intervention SERA. Exclusion alone and
  0.0025 similarly regress SERA/response. Weight 0.005 has no significant core
  regression and improves voltage/5 ms point estimates, but collapses
  default-threshold event count from 1.49x to 0.22x while only uncertainly
  improving F1. It therefore fails the fast diagnostic guard despite reducing
  refractory violations from 0.173 to 0.098 and count error by 0.031.
- Decision: reject all soft-OR/exclusion candidates and close the path. Keep
  the production objective on the legacy max-pool surrogate; prune the failed
  optional objective code at iteration 60 unless needed by a later accepted
  result.
- Artifacts: `artifacts/diagnostics/iteration-055-soft-or-exclusion/`.

### Iteration 056 premise - shared-weight recurrent Mamba refinement

- The two-layer Mamba trunk is applied once. Reapplying the same blocks can
  iteratively refine latent state without adding parameters, potentially
  improving long-horizon voltage/response and event timing together.
- Decision: execute exactly two passes for every candidate and blend the
  second-pass state at 0, 0.25, 0.5, 0.75, and 1.0. The zero control executes
  and backpropagates through the second pass with zero contribution, matching
  parameters and model compute. Because inference compute nearly doubles,
  require a material multi-axis core gain for promotion.

### Iteration 056 - 2026-07-12

- Regime: parameter-neutral recurrent-depth architecture test with matched
  two-pass forward/backward compute.
- Controls: exact 69,087 parameters, common initialization, two Mamba passes
  for every candidate, frozen 120k schedule, batch 32, 3,750 updates, promoted
  objective/optimizer, and diagnostics. The zero-mix control exactly matches
  the ordinary one-pass metrics.

| second-pass mix | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | -0.00519 | -0.00121 | -0.00221 | +0.00395 | +0.00162 | -0.00395 | provisional balanced |
| 0.5 | -0.02077 | -0.00692 | -0.00560 | -0.00103 | -0.00024 | -0.00999 | provisional SERA |
| 0.75 | -0.02051 | -0.00764 | -0.00326 | +0.00396 | -0.00341 | -0.00954 | secondary |
| 1.0 | -0.02133 | -0.00745 | -0.00373 | +0.00381 | -0.00631 | -0.00842 | reject frontier |

- Interpretation: recurrent refinement produces a coherent, saturating SERA
  gain. Mix 0.25 significantly improves voltage and response while preserving
  a balanced F1 profile; five of six point estimates improve. Mix 0.5
  significantly improves all three SERA axes, with only uncertain modest F1
  decreases. Larger mixes add no SERA gain and move farther down tolerant F1.
  The matched zero control costs 28.24 s versus about 22.73 s for the recent
  one-pass control, a 24% model-compute increase rather than a full doubling.
- Decision: replicate mixes 0.25 and 0.5 against zero at seed 20260712. A
  promotion must preserve significant SERA gains without a significant F1
  regression and be large enough to justify the measured compute cost.
- Artifacts: `artifacts/diagnostics/iteration-056-recurrent-second-pass/`.

### Iteration 057 premise - recurrent refinement replication

- Mix 0.25 is the balanced point; mix 0.5 is the strongest all-SERA point.
  Seed sensitivity has repeatedly converted response gains into regressions,
  and the added compute raises the required evidence bar.
- Decision: rerun zero, 0.25, and 0.5 at seed 20260712 with exact two-pass
  compute, parameters, presentations, updates, objective, optimizer, and
  diagnostics.

### Iteration 057 - 2026-07-12

- Regime: second-seed recurrent-depth replication at matched two-pass compute.
- Controls: seed 20260712, exact 69,087 parameters, two passes for all models,
  frozen balanced 120k schedule, batch 32, 3,750 updates, promoted
  objective/optimizer, and diagnostics.

| mix | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | -0.00922 | -0.00047 | -0.00003 | -0.00232 | -0.00333 | -0.00002 | reject compute/value |
| 0.5 | -0.01178 | -0.00016 | -0.00400 | +0.00336 | -0.00686 | -0.00182 | provisional |

- Interpretation: mix 0.25 replicates voltage but not response, leaving too
  little benefit for the 24% compute premium. Mix 0.5 again significantly
  improves voltage and response with no significant core regression. Its
  two-seed means are -0.01628 mV voltage, -0.00354 mV/ms dynamics,
  -0.00480 F1 area, +0.00116 exact F1, -0.00355 5 ms F1, and -0.00590 mV
  intervention response. The SERA benefit is robust, while tolerant F1 is the
  remaining tradeoff.
- Decision: retain mix 0.5 as the long-run lead, reject 0.25 for insufficient
  value, and map 0.35/0.4/0.45 to find a better SERA/F1 balance before the
  iteration-60 long gate.
- Artifacts: `artifacts/diagnostics/iteration-057-recurrent-second-pass-replication/`.

### Iteration 058 premise - recurrent-mix frontier refinement

- Recurrent mix 0.25 preserves F1 but lacks response value; 0.5 robustly
  improves SERA/response but lowers tolerant-F1 point estimates. Since all
  nonzero mixes incur identical two-pass compute, the useful question is the
  best behavioral blend, not a compute interpolation.
- Decision: compare 0.35, 0.4, and 0.45 against zero on seed 1. Carry the
  smallest point that preserves material SERA/response gains without a
  significant F1 regression into seed-2 replication; keep 0.5 as fallback.

### Iteration 058 - 2026-07-12

- Regime: low-temperature refinement of a replicated recurrent architecture.
- Controls: exact two-pass compute, 69,087 parameters, common initialization,
  frozen 120k schedule, batch 32, 3,750 updates, promoted objective/optimizer,
  and diagnostics.

| mix | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.35 | -0.00642 | -0.00283 | -0.00811 | +0.00200 | -0.00471 | -0.00455 | reject frontier |
| 0.4 | -0.00877 | -0.00381 | -0.00771 | +0.00253 | -0.00313 | -0.00544 | reject frontier |
| 0.45 | -0.01490 | -0.00537 | -0.00610 | +0.00271 | -0.00155 | -0.00682 | secondary |

- Interpretation: every intermediate mix significantly improves all three
  SERA axes, and larger mix monotonically strengthens the gain. Mix 0.45
  preserves exact/5 ms F1 better than 0.35-0.4, but its F1-area decrease is
  not better than mix 0.5's and it gives up material SERA/response. The
  refinement does not identify a superior frontier point.
- Decision: retain mix 0.5 as the recurrent lead. Run one third seed before
  iteration 60's 300k gate because tolerant-F1 estimates remain consistently
  lower and the architecture has a compute cost.
- Artifacts: `artifacts/diagnostics/iteration-058-recurrent-mix-frontier/`.

### Iteration 059 premise - recurrent-0.5 third seed

- Mix 0.5 robustly improves voltage/response across two seeds and all three
  SERA axes on seed 1, but F1 area/5 ms are modestly lower. A third seed gives
  a more stable benefit/cost estimate before spending the scheduled long run.
- Decision: rerun zero versus 0.5 at seed 20260713 with exact two-pass compute,
  parameters, presentations, updates, objective, optimizer, and diagnostics.
  Carry to iteration 60 only if no significant core regression appears.

### Iteration 059 - 2026-07-12

- Regime: predeclared third-seed recurrent-depth tie-break.
- Controls: seed 20260713, exact two-pass compute, 69,087 parameters, frozen
  balanced 120k schedule, batch 32, 3,750 updates, promoted
  objective/optimizer, and diagnostics.
- Result: mix 0.5 significantly regresses voltage SERA by 0.05131 mV and
  intervention SERA by 0.00821 mV. Dynamics regresses 0.00129 mV/ms
  uncertainly; F1 area, exact F1, and 5 ms F1 improve by 0.00162, 0.00161,
  and 0.00269 uncertainly.
- Interpretation: the strong seed-1/2 SERA benefit reverses on seed 3, while
  the architecture still incurs 24% more compute. This violates the
  predeclared long-gate rule and makes the three-seed average insufficiently
  robust for the cost.
- Decision: reject recurrent second-pass refinement and do not carry it into
  the long run. Queue the optional config/forward/test code for iteration-60
  cleanup.
- Artifacts: `artifacts/diagnostics/iteration-059-recurrent-half-third-seed/`.

### Iteration 060 premise - long promoted-allocation seed-2 revalidation

- The promoted 118/58/25 allocation beat 45/64/16 at 300k on seed 1 under
  SERA 0.0075, and current SERA 0.01 was validated at 300k only within
  118/58/25. Their interaction has not been checked on a second long seed.
- Decision: compare current 118 branches/11 slots/width 58/state 25 against
  historical 45 branches/100 slots/width 64/state 16 at seed 20260712 for
  300k presentations under the promoted objective. Both have exactly 69,087
  parameters and share data, initialization policy, updates, optimizer, and
  diagnostics. Then prune failed iteration-51-59 production paths and run
  full AxoSim/AxoBench verification.

### Iteration 060 - 2026-07-12

- Regime: scheduled 300k exact-budget architecture revalidation and cleanup.
- Controls: seed 20260712, exact 69,087 parameters, common initialization
  policy, frozen balanced 300k schedule, batch 32, 9,375 updates, promoted
  SERA-0.01/F1-0.05 objective, optimizer, and diagnostics.

| 45/64/16 vs 118/58/25 | delta |
| --- | ---: |
| voltage SERA | +0.03123 mV significant regression |
| dynamics SERA | +0.00294 mV/ms |
| F1 area | -0.00719 |
| exact F1 | -0.00210 |
| 5 ms F1 | -0.00901 |
| intervention SERA | +0.00548 mV significant regression |

- Interpretation: the promoted 118/58/25 allocation wins all six core point
  estimates on long seed 2. Its voltage and response gains are significant,
  while compute is matched at 55.40 versus 55.67 s. Together with iteration
  40 and the three 120k seeds, this strongly supports the allocation under
  both SERA 0.0075 and the durable SERA 0.01 objective.
- Decision: retain 118 branches, 11 slots, width 58, and state 25 as the
  reusable AxoMamba default.
- Cleanup: removed failed recurrent-second-pass and soft-OR/exclusion config,
  forward, objective, function, and test paths. Historical manifests and
  diagnostics remain. Retained generic post-common-state parameter offsets,
  because they fixed runner fidelity and add no model/runtime behavior.
- Verification: AxoSim passes 82 tests with the AxoBench source bridge and the
  AxoBench iteration evaluator passes 23 focused tests.
- Artifacts: `artifacts/diagnostics/iteration-060-long-allocation-seed2/`.

### Iteration 061 premise - partial final normalization

- The model's least-squares voltage gain remains near 0.2 and intervention
  gain near zero. Final LayerNorm removes absolute latent mean/scale before
  both readout heads, which may compress state amplitude even though block
  pre-normalization already stabilizes the Mamba trunk.
- Hypothesis: blending raw hidden state into the final normalized features can
  preserve absolute amplitude and intervention sensitivity while retaining
  enough normalization for optimization and spike timing.
- Decision: compare normalized fractions 0.75, 0.5, 0.25, and 0 against the
  fully normalized 1.0 control. Every candidate computes both raw and
  normalized paths, retains identical LayerNorm parameters and model compute,
  and shares data, updates, objective, optimizer, and diagnostics.

### Iteration 061 - 2026-07-12

- Regime: parameter/compute-matched final-feature normalization ablation.
- Controls: exact 69,087 parameters, common initialization, raw and normalized
  paths for all candidates, frozen 120k schedule, batch 32, 3,750 updates,
  promoted objective/optimizer, and diagnostics.

| normalized fraction | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.75 | -0.01125 | -0.00096 | -0.00654 | +0.00286 | -0.00881 | -0.00261 | reject premise |
| 0.5 | -0.00694 | -0.00133 | -0.00436 | +0.00230 | -0.00667 | -0.00289 | reject premise |
| 0.25 | -0.00083 | -0.00376 | -0.00268 | -0.00379 | -0.00501 | -0.00011 | reject |
| 0 | +0.00375 | -0.00425 | -0.01449 | -0.00215 | -0.01826 | +0.00356 | reject |

- Interpretation: partial normalization again moves along the SERA/F1
  frontier. Fractions 0.5-0.75 significantly improve voltage/response but
  lower F1 area/5 ms. Removing normalization significantly regresses response
  and tolerant F1. The proposed amplitude mechanism is not supported: at 0.5,
  voltage gain falls from 0.21261 to 0.21112, while intervention gain rises
  only 0.00013. The SERA effect is not a restoration of absolute gain.
- Decision: reject global final-normalization blending without replication.
  Test a soma-only blend while keeping spike features fully normalized to
  determine whether the F1 tradeoff comes from perturbing the spike head.
- Artifacts: `artifacts/diagnostics/iteration-061-final-norm-mix/`.

### Iteration 062 premise - soma-only final normalization

- Global raw-state blending improves SERA but also perturbs the spike head,
  obscuring whether F1 loss is necessary. AxoMamba already has separate heads
  because of the multiplicative soma gate.
- Hypothesis: blend raw state only into the soma readout/gate while leaving the
  spike head fully normalized. This isolates voltage amplitude/state capacity
  and may preserve F1 while retaining the SERA/response benefit.
- Decision: compare soma normalized fractions 0.75, 0.5, 0.25, and 0 against
  1.0. Every candidate computes both paths with identical parameters, data,
  updates, objective, optimizer, and diagnostics; spike features are unchanged.

### Iteration 062 - 2026-07-12

- Regime: parameter/compute-matched soma-readout normalization ablation.
- Controls: exact 69,087 parameters, common initialization, both raw and
  normalized paths in every candidate, frozen 120k schedule, batch 32, 3,750
  updates, promoted objective/optimizer, and identical diagnostics.

| soma normalized fraction | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.75 | -0.02148 significant | -0.00217 | -0.00543 | +0.00184 | -0.00732 | -0.00522 significant | provisional |
| 0.5 | -0.01772 significant | -0.00165 | -0.01445 significant | +0.00593 | -0.02529 significant | -0.00315 significant | reject tradeoff |
| 0.25 | -0.00391 significant | -0.00136 | -0.01307 significant | +0.00487 | -0.01891 significant | +0.00247 significant | reject |
| 0 | -0.00119 | -0.00542 significant | -0.00894 | +0.00559 | -0.01216 | -0.00195 | reject |

- Interpretation: isolating the blend to the soma path improves the global
  blend frontier. Fraction 0.75 roughly doubles the global 0.75 voltage gain,
  doubles its intervention gain, and has smaller uncertain tolerant-F1 losses.
  However, it does not repair the proposed amplitude mechanism: voltage gain
  remains 0.211 versus 0.213. It also worsens already-triggered count ratio
  from 1.49 to 1.73 and refractory violations from 0.173 to 0.188. Fractions
  at or below 0.5 incur significant tolerant-F1 regressions.
- Decision: provisionally retain soma fraction 0.75 as a downstream result,
  not as validation of the amplitude hypothesis. Replicate it on seed 2 while
  interpolating 0.80-0.90 to seek a more balanced spike-health/F1 point.
- Compute: all candidates have 69,087 parameters and about 21.2-22.8 seconds
  candidate compute; the shared five-model panel took 196.93 seconds.
- Artifacts: `artifacts/diagnostics/iteration-062-soma-final-norm-mix/`.

### Iteration 063 premise - soma-normalization replication and interpolation

- Iteration 62 found a significant SERA/response benefit at soma fraction
  0.75, but tolerant F1, count calibration, and refractory behavior moved in
  the wrong direction. A second seed is required before promotion.
- Hypothesis: fractions 0.80-0.90 retain a replicated share of the soma-only
  SERA gain while reducing tolerant-F1 and spike-health costs.
- Decision: at seed 20260712 compare soma fractions 0.75, 0.80, 0.85, and
  0.90 against the fully normalized control. Require replicated voltage and
  intervention gains, no significant F1 regression, and no materially worse
  count/refractory behavior. All candidates remain exact parameter/compute
  matches on the same frozen schedule.

### Iteration 063 - 2026-07-12

- Regime: second-seed replication and interpolation on the frozen 120k
  schedule; exact 69,087 parameters, batch 32, 3,750 updates, and matched
  objective, optimizer, initialization policy, and diagnostics.

| soma normalized fraction | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.90 | +0.01116 significant | -0.00032 | -0.00171 | -0.00129 | -0.00453 | +0.00104 significant | reject replication |
| 0.85 | +0.01525 significant | -0.00103 | -0.00146 | +0.00230 | -0.00513 | +0.00157 significant | reject replication |
| 0.80 | +0.01703 significant | -0.00180 | -0.00057 | +0.00216 | -0.00461 | +0.00177 significant | reject replication |
| 0.75 | +0.02024 significant | -0.00218 | -0.00068 | +0.00381 | -0.00546 | +0.00238 significant | reject replication |

- Interpretation: the seed-1 SERA benefit reverses monotonically on seed 2.
  Every fraction significantly regresses both voltage and intervention SERA;
  reducing the normalized fraction increases both regressions. Dynamics and
  F1 changes are uncertain and cannot rescue the failed primary result.
- Fidelity note: soma-only blending still changes spike logits indirectly
  because the promoted spike head adds four times the reported soma output.
  Thus iteration 62 did not isolate voltage readout from spike coupling even
  though the spike feature tensor itself remained normalized.
- Decision: reject all soma-normalization blends. Test the sharper causal
  hypothesis by driving spike coupling from a separately computed fully
  normalized soma while allowing the reported soma output to use the blend.
- Compute: five-model shared panel 196.81 seconds; each candidate about
  21.3-23.0 seconds model compute with exact parameter matching.
- Artifacts: `artifacts/diagnostics/iteration-063-soma-final-norm-replication/`.

### Iteration 064 premise - isolate soma readout from spike coupling

- The prior soma-only intervention leaked into spike timing through the
  promoted `spike + 4 * soma` coupling. This can explain tolerant-F1 movement
  without requiring a representation tradeoff.
- Hypothesis: use the fully normalized soma path solely as the spike-coupling
  reference while a mixed soma path predicts voltage. If coupling leakage
  caused the tradeoff, isolated candidates should match the normalized
  control's spike output at initialization and retain more F1 after training.
- Decision: implement nullable `output`/`normalized` coupling sources. Both
  modes always compute both soma paths, so the control, leaky comparison, and
  isolated candidates have identical parameters and model compute.

### Iteration 064 - 2026-07-12

- Regime: scientific coupling-leakage test with exact 69,087 parameters and
  matched double-soma-path compute, frozen 120k schedule, batch 32, 3,750
  updates, common initialization, objective, optimizer, and diagnostics.

| candidate | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.75, output coupling | -0.02148 significant | -0.00217 | -0.00543 | +0.00184 | -0.00732 | -0.00522 significant | reference |
| 0.75, normalized coupling | +0.01078 significant | -0.00080 | -0.00604 | +0.00305 | -0.00905 | +0.00013 | reject premise |
| 0.5, normalized coupling | +0.00100 | -0.00102 | -0.00752 | +0.00314 | -0.00976 significant | +0.00234 significant | reject |
| 0, normalized coupling | -0.01088 significant | -0.00518 significant | -0.00727 | +0.00128 | -0.00870 | -0.00282 significant | reject tradeoff |

- Interpretation: coupling leakage does not cause the soma-blend F1 tradeoff.
  At the same 0.75 fraction, ordinary output coupling dominates isolated
  normalized coupling on every core point estimate except exact F1. Removing
  normalization entirely under isolation improves all SERA axes but still
  lowers tolerant F1, so the tradeoff persists without leakage.
- Decision: reject isolated coupling and retain the promoted normalized soma
  path/coupling. Move to an AP-specific causal voltage correction aimed at
  the persistent near-spike error rather than continuing normalization sweeps.
- Compute: shared five-model panel 198.86 seconds; per-candidate model compute
  21.4-23.0 seconds with exact parameter/compute matching.
- Artifacts: `artifacts/diagnostics/iteration-064-isolated-soma-coupling/`.

### Iteration 065 premise - causal spike-conditioned voltage waveform

- Near-spike voltage remains the largest state-conditioned failure (roughly
  4.47 times global RMSE). A shared latent readout must simultaneously model
  subthreshold voltage and sparse AP waveforms, despite their very different
  scales and frequencies.
- Hypothesis: a tiny causal convolution from predicted spike probability to
  soma voltage can specialize AP shape, improve SERA/dynamics near events,
  and leave spike output unchanged. Detaching its spike input tests whether
  voltage gradients should remain out of the spike-timing path.
- Decision: add a zero-initialized causal waveform head that is present and
  computed in every candidate. Compare disabled, detached, joint-gradient,
  and stronger detached optimization scales with identical added parameters
  and compute. Require improved near-spike/high-dVdt diagnostics plus core
  SERA without F1 or physiological regression.

### Iteration 065 - 2026-07-12

- Regime: scientific AP-head test; 69,103 exact matched parameters, identical
  16 ms causal convolution compute, frozen 120k schedule, batch 32, 3,750
  updates, common initialization, objective, optimizer, and diagnostics.

| waveform mode | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| detached scale 1 | -0.00664 significant | -0.00064 | -0.00175 | +0.00023 | -0.00425 | -0.00135 significant | secondary |
| detached scale 4 | -0.01259 significant | -0.00152 | -0.00144 | +0.00632 significant | -0.00608 | -0.00345 significant | provisional |
| joint scale 1 | -0.00368 significant | -0.00077 | -0.00078 | +0.00177 | -0.00339 | -0.00215 significant | secondary |
| joint scale 4 | -0.00345 | -0.00500 significant | -0.00717 | +0.00242 | -0.00884 | -0.00921 significant | reject tradeoff |

- Interpretation: detached scale 4 is the best downstream point, improving
  voltage/response SERA and exact F1 significantly without a significant
  tolerant-F1 regression. Joint scale 4 sends voltage gradients into spike
  timing and moves to a worse F1 frontier. The AP-specific premise is weak:
  detached scale 4 changes near-spike ratio only 4.47260 to 4.47236 and
  high-dVdt ratio 2.85359 to 2.85347. Its gain is broader calibration, not a
  material AP-waveform repair; refractory violations also rise 0.173 to 0.209.
- Decision: reject joint gradients. Replicate detached scale 4 on seed 2 and
  extend the monotonic detached scale sweep to 8/16 before any promotion.
- Compute: shared panel 199.14 seconds; all candidates have 69,103 parameters
  and about 21.5-23.4 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-065-spike-waveform-head/`.

### Iteration 066 premise - detached waveform replication and scale

- Detached scale 4 improved three core metrics significantly on seed 1, but
  did not materially repair the proposed AP-specific diagnostic and worsened
  refractory violations. Replication is required.
- Hypothesis: the detached scale trend has a robust optimum that preserves
  tolerant F1 and spike health while retaining voltage/response gains.
- Decision: on seed 20260712 compare disabled control with detached scales 2,
  4, 8, and 16. All use the same 16-weight causal head and compute. Require
  seed-2 SERA gains, no significant F1 regression, and no material additional
  refractory/count degradation.

### Iteration 066 - 2026-07-12

- Regime: second-seed detached-waveform replication/scale panel; exact 69,103
  parameters, identical causal-head compute, frozen 120k schedule, batch 32,
  3,750 updates, common objective, optimizer, and diagnostics.

| detached scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | -0.00910 significant | -0.00259 significant | +0.00135 | +0.00193 | +0.00052 | -0.00187 significant | provisional lead |
| 4 | -0.00221 | -0.00392 significant | -0.00134 | +0.00407 | -0.00565 | -0.00241 significant | secondary |
| 8 | +0.00005 | -0.00349 significant | -0.00322 | +0.00168 | -0.01034 significant | -0.00368 significant | reject tradeoff |
| 16 | -0.00296 | -0.00210 | -0.00457 | +0.00339 | -0.01355 significant | -0.00458 significant | reject tradeoff |

- Interpretation: scale 2 improves all six core point estimates and all three
  SERA axes significantly, replicating the useful detached-head regime on a
  second seed. Larger scales increasingly trade 5 ms F1 for response SERA.
  Scale 2 modestly improves near-spike ratio 4.47311 to 4.47169, high-dVdt
  ratio 2.85329 to 2.85240, voltage bias, and morphology disparity. Count
  ratio rises 1.313 to 1.333 and refractory violations 0.2141 to 0.2158, with
  no new trigger.
- Decision: carry detached scale 2 to a third seed before architecture/kernel
  refinement or promotion. Reject scales 8/16 and retain scale 4 only as
  evidence that the viable interval spans both first seeds.
- Compute: shared panel 195.91 seconds; all candidates have 69,103 parameters
  and about 21.1-23.0 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-066-spike-waveform-replication/`.

### Iteration 067 premise - detached waveform third-seed gate

- Scale 2 improves every core point estimate on seed 2; neighboring scales 1
  and 4 improve voltage/response on seed 1. A third seed tests robustness
  before spending iterations on waveform support or carrying it to 300k.
- Decision: compare disabled versus detached scale 2 at seed 20260713 with the
  same 16-weight head, frozen 120k schedule, parameters, compute, updates,
  objective, optimizer, and diagnostics. Require SERA gains without a
  significant F1 or failure-mode regression.

### Iteration 067 - 2026-07-12

- Regime: third-seed robustness gate; exact 69,103 parameters, matched causal
  head compute, frozen 120k schedule, batch 32, 3,750 updates, common
  objective, optimizer, initialization, and diagnostics.
- Result: detached scale 2 significantly regresses voltage SERA by 0.01546 mV.
  Dynamics regresses 0.00142 mV/ms, response 0.00085 mV, F1 area 0.00174,
  and 5 ms F1 0.00213 uncertainly; exact F1 improves 0.00280 uncertainly.
- Interpretation: the two-seed SERA improvement does not survive seed 3, and
  the AP-specific premise was already weak. The 16 added weights do not yield
  a robust benefit over the promoted model.
- Decision: reject the spike-conditioned waveform head and queue its config,
  module, forward, and tests for iteration-70 cleanup. Do not spend iteration
  68 on kernel support refinement.
- Compute: two-model shared panel 134.01 seconds; candidate compute 21.00
  versus 22.38 seconds with exact parameter matching.
- Artifacts: `artifacts/diagnostics/iteration-067-spike-waveform-third-seed/`.

### Iteration 068 premise - explicit morphology conditioning

- The frozen schedule is balanced across five morphologies, and diagnostics
  score morphology disparity/interventions, but the model receives only the
  event tensor. Identical event patterns can map to different voltage/spike
  responses across connectomes, making the current target conditionally
  ambiguous.
- Hypothesis: a learned morphology embedding added before the temporal trunk
  improves sample efficiency, voltage/intervention SERA, and morphology
  disparity without harming timing. This tests missing task information rather
  than adding generic capacity.
- Decision: extend the model, shared trainer, checkpoint predictor, and
  AxoBench cached evaluator to pass explicit morphology IDs. Compare an
  inactive scale-0 embedding control against bounded active scales under
  identical embedding parameters and compute on the same five-morphology
  120k schedule. Verify context propagation through standard and intervention
  diagnostics before interpreting metrics.

### Iteration 068 - 2026-07-12

- Regime: high-temperature missing-information test; exact 69,377 parameters,
  identical five-row embedding lookup/add compute, frozen balanced 120k
  schedule, batch 32, 3,750 updates, common initialization, objective,
  optimizer, and context-aware standard/intervention diagnostics.

| embedding scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | -0.14807 significant | -0.00564 significant | +0.03464 significant | +0.01826 significant | +0.04330 significant | -0.02561 significant | provisional balanced lead |
| 0.25 | -0.17244 significant | -0.00741 significant | +0.02484 significant | +0.01161 significant | +0.03346 significant | -0.02915 significant | provisional SERA lead |
| 0.5 | +0.00922 | -0.00240 | -0.01201 significant | +0.00384 | -0.01758 | +0.01476 significant | reject |
| 1.0 | -0.00369 | -0.00436 significant | -0.01038 | +0.00874 | -0.02020 significant | +0.00785 | reject |

- Interpretation: explicit morphology identity produces the largest and
  broadest improvement in the loop. Scales 0.1 and 0.25 significantly improve
  every core metric, validating the missing-information premise. Scale 0.1 is
  more balanced: voltage gain rises 0.213 to 0.236, bias falls 0.059 to 0.013,
  count ratio improves 1.495 to 1.429, intervention direction rises 0.642 to
  0.657, and morphology disparity falls 1.396 to 1.372. Refractory violations
  worsen 0.173 to 0.254 and near-spike/high-dVdt ratios rise; scale 0.25 also
  worsens count ratio to 1.937. No new attention trigger appears.
- Decision: provisionally select scale 0.1, retain 0.25 as SERA-focused, and
  replicate both on seed 2 while interpolating 0.05/0.15. Promotion requires
  replicated all-core gains and scrutiny of refractory/count behavior.
- Compute: shared five-model panel 197.81 seconds; all candidates have 69,377
  parameters and about 21.3-23.1 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-068-morphology-conditioning/`.

### Iteration 069 premise - morphology-conditioning replication/refinement

- Scales 0.1 and 0.25 improve all six core metrics significantly on seed 1,
  but larger scales fail and spike-health diagnostics expose a bounded useful
  regime. A second seed is required before the scheduled long run.
- Hypothesis: conditioning remains strongly beneficial around 0.1, with a
  lower/intermediate scale improving the refractory/count tradeoff.
- Decision: on seed 20260712 compare inactive control with scales 0.05, 0.1,
  0.15, and 0.25 under identical 69,377 parameters, context, compute, frozen
  120k schedule, updates, objective, optimizer, and diagnostics. Select the
  replicated balanced point for iteration 70's 300k gate.

### Iteration 069 - 2026-07-12

- Regime: seed-2 replication/refinement; exact 69,377 parameters, identical
  embedding/context compute, frozen balanced 120k schedule, batch 32, 3,750
  updates, common initialization, objective, optimizer, and diagnostics.

| embedding scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | -0.07552 significant | -0.00140 | +0.02820 significant | +0.01214 significant | +0.03261 significant | -0.02099 significant | long health lead |
| 0.1 | -0.10569 significant | -0.00727 significant | +0.02679 significant | +0.01323 significant | +0.03861 significant | -0.01736 significant | long replicated lead |
| 0.15 | -0.12067 significant | -0.00655 significant | +0.02880 significant | +0.01203 significant | +0.04545 significant | -0.02772 significant | long metric lead |
| 0.25 | -0.12125 significant | -0.00573 significant | +0.01937 significant | +0.01382 significant | +0.01991 significant | -0.02505 significant | secondary |

- Interpretation: morphology conditioning robustly replicates. Every tested
  scale significantly improves voltage, response, and all F1 metrics; scales
  0.1-0.25 also improve dynamics significantly. Scale 0.1 now has two-seed
  support. Scale 0.15 is strongest on seed-2 response and tolerant F1, but
  refractory violations rise 0.214 to 0.289. Scale 0.05 improves count ratio
  1.313 to 1.263 and raises refractory only to 0.235 while retaining broad
  significant gains. Morphology disparity improves at every reported scale.
- Decision: run scales 0.05, 0.1, and 0.15 against inactive control at 300k
  on seed 1. Select the best long-run core/health point, then promote and prune
  failed iteration-61-67 paths only if the long gate passes.
- Compute: shared panel 196.49 seconds; all candidates have 69,377 parameters
  and about 21.3-23.0 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-069-morphology-replication/`.

### Iteration 070 premise - long morphology gate and scheduled cleanup

- Morphology conditioning improves all core metrics significantly across two
  120k seeds around scale 0.1. The remaining question is whether that benefit
  persists at the standard 300k long checkpoint and which scale best balances
  refractory/count behavior.
- Decision: at seed 20260711 compare inactive scale 0 with scales 0.05, 0.1,
  and 0.15 for 300k presentations (9,375 updates), exact 69,377 parameters,
  identical context/compute, objective, optimizer, and diagnostics. A passing
  candidate is promoted into reusable AxoMamba. Then remove rejected
  normalization, isolated-coupling, and waveform code/tests; retain the tested
  context path and run full AxoSim/AxoBench verification.

### Iteration 070 - 2026-07-12

- Regime: scheduled 300k promotion gate and cleanup; exact 69,377 parameters,
  identical context/embedding compute, balanced 60k presentations per each of
  five morphologies, batch 32, 9,375 updates, common initialization,
  objective, optimizer, and diagnostics.

| embedding scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | -0.12177 significant | -0.01088 significant | +0.03879 significant | +0.01835 significant | +0.05219 significant | -0.04314 significant | pass |
| 0.1 | -0.11634 significant | -0.01234 significant | +0.03920 significant | +0.01504 significant | +0.05066 significant | -0.04056 significant | pass |
| 0.15 | -0.12598 significant | -0.01065 significant | +0.04927 significant | +0.01713 significant | +0.06579 significant | -0.03443 significant | promote |

- Interpretation: the missing-information result strengthens with longer
  training. Every scale significantly improves all six core metrics. Scale
  0.15 leads voltage SERA, F1 area, and 5 ms F1; scale 0.05 leads response and
  exact F1. Scale 0.15 has the best candidate morphology disparity (1.375),
  lowest candidate refractory rate (0.194), improves count ratio versus the
  control (1.827 versus 1.954), and retains a large significant response gain.
  It is the strongest overall long-run core/health point.
- Decision: promote morphology scale 0.15 into `configs/axomamba.json`,
  `default_axomamba_config`, and stable `checkpoints/axomamba.pt`. The reusable
  AxoMamba now has 69,377 parameters and requires one of the five configured
  morphology IDs per sample.
- Cleanup: removed rejected global/soma normalization blends, coupling-source
  isolation, and spike-waveform config/module/forward/test paths. Historical
  manifests and diagnostics remain. Retained morphology context propagation
  and generic post-match initialization offsets.
- Verification: AxoSim passes 85 tests; AxoBench iteration evaluator passes
  24 tests; no rejected-symbol references remain; the stable checkpoint is
  byte-identical to the promoted 300k scale-0.15 run.
- Compute: shared four-model panel 440.36 seconds; candidate model compute
  53.75-58.09 seconds with exact matching.
- Artifacts: `artifacts/diagnostics/iteration-070-long-morphology-gate/` and
  `checkpoints/axomamba.pt`.

### Iteration 071 premise - morphology-conditioned final FiLM

- Additive pre-trunk morphology context is now robustly useful. It changes
  temporal dynamics and the readouts through one shared vector, but final
  LayerNorm can attenuate morphology-specific scale/offset information before
  the spike and soma heads.
- Hypothesis: a morphology-specific final FiLM modulation on top of the
  promoted pre-trunk embedding improves per-morph readout calibration and
  intervention behavior while retaining the learned temporal benefit.
- Decision: add a zero-initialized morphology FiLM table to every candidate,
  computed in all modes for exact parameter/compute matching. Compare inactive
  control and bounded FiLM scales on the promoted additive scale-0.15 model.
  Require core gains plus improved count/morphology/intervention diagnostics;
  reject a mere refractory/F1 tradeoff.

### Iteration 071 - 2026-07-12

- Regime: matched final-FiLM test on promoted morphology-conditioned
  AxoMamba; exact 69,957 parameters, identical FiLM lookup/modulation compute,
  frozen 120k schedule, batch 32, 3,750 updates, common initialization,
  objective, optimizer, and diagnostics.

| FiLM scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | -0.00089 significant | -0.00006 | -0.00268 significant | -0.00077 | -0.00467 significant | -0.00064 significant | reject tradeoff |
| 0.1 | -0.00085 significant | -0.00003 | -0.00195 | -0.00024 | -0.00433 significant | -0.00118 significant | reject tradeoff |
| 0.25 | +0.00142 significant | -0.00029 | -0.00060 | -0.00034 | -0.00144 | -0.00125 significant | reject |
| 0.5 | +0.00162 significant | -0.00045 | -0.00243 | +0.00162 | -0.00294 | -0.00091 significant | reject |

- Interpretation: final morphology-specific scale/offset modulation adds no
  useful frontier beyond the pre-trunk embedding. Low scales produce tiny
  voltage/response gains but significant tolerant-F1 regressions; larger
  scales significantly regress voltage. The effect is orders of magnitude
  smaller than the promoted embedding result.
- Decision: reject final FiLM without replication and queue its optional path
  for iteration-80 cleanup. Test morphology where it mechanistically changes
  the connectome computation: branch-current gains before saturation/Mamba.
- Compute: shared panel 201.23 seconds; all candidates have 69,957 parameters
  and about 22.0-23.9 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-071-morphology-film/`.

### Iteration 072 premise - morphology-specific branch gains

- Different morphologies change dendritic routing and electrotonic influence,
  but the current model uses one fixed branch reduction/projection for all
  specimens and supplies morphology only as a global hidden-state vector.
- Hypothesis: zero-initialized morphology-specific gains on the 118 routed
  branch currents, applied before tanh saturation and Mamba, improve voltage,
  intervention response, and morphology disparity beyond global context.
- Decision: add the same five-by-118 gain table and lookup compute to every
  candidate, compare inactive and bounded scales on promoted AxoMamba, and
  require broad core gains plus morphology/intervention improvement without a
  significant F1 or spike-health regression.

### Iteration 072 - 2026-07-12

- Regime: matched morphology-by-branch gain test on promoted AxoMamba; exact
  69,967 parameters, identical lookup/tanh/multiply compute, frozen 120k
  schedule, batch 32, 3,750 updates, common initialization, objective,
  optimizer, and diagnostics.

| branch-gain scale | delta voltage SERA | delta dynamics SERA | delta F1 area | delta exact F1 | delta 5 ms F1 | delta intervention SERA | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | -0.00124 significant | -0.00024 significant | +0.00044 | +0.00127 significant | -0.00097 | -0.00031 significant | secondary |
| 0.25 | -0.00304 significant | -0.00044 significant | +0.00242 | +0.00171 | +0.00242 | -0.00078 significant | secondary |
| 0.5 | -0.00554 significant | -0.00080 significant | +0.00269 | +0.00244 | +0.00145 | -0.00143 significant | provisional |
| 1.0 | -0.00975 significant | -0.00142 significant | +0.00574 significant | +0.00505 significant | +0.00526 | -0.00240 significant | provisional lead |

- Interpretation: the effect is monotonic and scale 1 improves five core
  metrics significantly while 5 ms F1 also rises. Intervention direction
  improves 0.649 to 0.655, intervention gain 0.00518 to 0.00563, voltage gain
  0.2344 to 0.2367, and refractory violations fall 0.243 to 0.236. Morphology
  disparity is flat/slightly worse (1.37186 to 1.37220) and count ratio rises
  1.670 to 1.699, so the connectome premise is only partially supported.
- Decision: provisionally select scale 1 and replicate on seed 2. Extend the
  still-monotonic scale curve to 2/4 while retaining a 0.75 interpolation;
  require replicated broad gains and no new spike-health regression.
- Compute: shared panel 199.56 seconds; all candidates have 69,967 parameters
  and about 21.7-23.5 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-072-morphology-branch-gain/`.

### Iteration 073 premise - branch-gain replication and scale extension

- Morphology-specific branch gains improve nearly every core metric
  monotonically through scale 1 on seed 1, with favorable intervention and
  refractory movement but no morphology-disparity improvement.
- Decision: on seed 20260712 compare inactive control with scales 0.75, 1, 2,
  and 4 under identical 69,967 parameters, compute, context, frozen 120k
  schedule, objective, optimizer, and diagnostics. Select only a replicated
  balanced point; reject high scales if F1/count/physiology deteriorate.

### Iteration 073 - 2026-07-12

- Regime: seed-2 branch-gain replication/extension; exact 69,967 parameters,
  identical compute/context, frozen 120k schedule, batch 32, 3,750 updates,
  common objective, optimizer, initialization, and diagnostics.

| scale | delta voltage | delta dynamics | delta F1 area | delta exact F1 | delta 5 ms F1 | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.75 | -0.00408 significant | -0.00082 significant | -0.00287 | +0.00123 | -0.00837 significant | -0.00010 | reject tradeoff |
| 1 | -0.00491 significant | -0.00090 | -0.00041 | +0.00220 | -0.00671 | -0.00013 | third seed |
| 2 | -0.00936 significant | -0.00130 | +0.00240 | +0.00342 | -0.00411 | -0.00074 | third-seed lead |
| 4 | -0.01039 significant | -0.00228 | +0.00863 significant | +0.00643 significant | -0.00000 | +0.00059 | F1-focused third seed |

- Interpretation: voltage gains replicate and larger scales recover F1 area,
  but the all-core seed-1 pattern weakens. Scale 0.75 significantly regresses
  5 ms F1. Scale 2 is balanced; scale 4 improves F1 area/exact significantly
  but slightly regresses response. No point is ready for promotion.
- Decision: run scales 1/2/4 on seed 3 against inactive control. Promote only
  a point with robust voltage gain and no significant response/F1 regression.
- Artifacts: `artifacts/diagnostics/iteration-073-morphology-branch-gain-replication/`.

### Iteration 074 premise - branch-gain third-seed gate

- Branch gains improve voltage across two seeds, but timing/response tradeoffs
  differ by scale. A third seed separates robust value from seed-specific
  frontier movement.
- Decision: compare inactive control with scales 1, 2, and 4 at seed 20260713,
  exact parameters/compute/context and frozen 120k budget. Reject the branch
  gain path if no scale retains voltage gains without significant core harm.

### Iteration 074 - 2026-07-12

- Regime: third-seed branch-gain gate; exact 69,967 parameters and matched
  compute/context on the frozen 120k schedule.

| scale | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -0.00585 significant | -0.00198 significant | -0.00144 | -0.00047 | -0.00032 | +0.00184 significant | reject |
| 2 | -0.00889 significant | -0.00365 significant | +0.00384 | +0.00384 | +0.00379 | +0.00431 significant | reject |
| 4 | +0.00552 | -0.00437 significant | +0.01199 significant | +0.00615 | +0.01574 significant | +0.00163 | reject frontier |

- Interpretation: lower scales retain voltage/dynamics gains but significantly
  regress intervention response; scale 4 flips to a dynamics/F1 tradeoff and
  loses voltage. No scale satisfies the gate across three seeds.
- Decision: reject morphology-specific branch gains and queue the optional
  path for iteration-80 cleanup. Retain only promoted global morphology
  embedding. Test parameter-neutral reinjection of that embedding through
  Mamba depth next.
- Artifacts: `artifacts/diagnostics/iteration-074-morphology-branch-gain-third-seed/`.

### Iteration 075 premise - morphology embedding reinjection through depth

- The promoted morphology embedding is added once before two Mamba blocks;
  residual dynamics and normalization may attenuate its influence with depth.
- Hypothesis: reinjecting the same learned embedding before each Mamba block
  preserves morphology context in temporal dynamics without adding capacity.
- Decision: compute reinjection for all candidates and compare zero with
  bounded scales under exact 69,377 parameters/compute, promoted context, and
  the frozen 120k setup. Require gains beyond simple context rescaling.

### Iteration 075 - 2026-07-12

- Result: reject reinjection. Scale 0.025 improves voltage by 0.01289 mV but
  significantly regresses response by 0.00468 mV. Scale 0.05 regresses
  dynamics 0.00243 and response 0.00044 uncertainly. Scale 0.1 significantly
  regresses response and 5 ms F1; scale 0.15 significantly regresses voltage,
  response, F1 area, and 5 ms F1.
- Interpretation: repeated context injection mostly over-amplifies the already
  sufficient promoted embedding and adds no balanced value.
- Decision: reject and queue reinjection for iteration-80 cleanup. Retune the
  direct-SERA/F1 objective on the stronger conditioned architecture, since its
  current weights were selected before morphology context existed.
- Artifacts: `artifacts/diagnostics/iteration-075-morphology-reinjection/`.

### Iteration 076 premise - conditioned-model objective retune

- Morphology conditioning materially changes model capacity and gradient
  allocation; the durable SERA 0.01/F1 0.05 objective was tuned on the older
  unconditioned model.
- Decision: compare stronger SERA/F1 weights and a joint point against the
  promoted objective with identical model, data, parameters, updates, and
  loss-computation paths. Optimize the declared core metrics directly.

### Iteration 076 - 2026-07-12

- Regime: metric-aligned objective panel on promoted morphology-conditioned
  AxoMamba; exact 69,377 parameters, frozen 120k schedule, batch 32, 3,750
  updates, common seed/initialization/data/optimizer, and identical SERA/F1
  loss paths.

| objective change | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F1 0.06 | +0.00242 significant | +0.00001 | +0.00209 | +0.00134 | +0.00090 | +0.00061 significant | reject |
| F1 0.075 | +0.01175 significant | +0.00035 | +0.00663 significant | +0.00203 | +0.00697 | +0.00503 significant | reject tradeoff |
| SERA 0.015 | -0.02866 significant | -0.00001 | -0.00106 | +0.00270 | -0.00191 | -0.00329 significant | replicate bracket |
| SERA 0.015 + F1 0.06 | -0.02437 significant | -0.00022 | +0.00077 | +0.00353 | +0.00028 | +0.00003 | secondary |
| SERA 0.02 | -0.03901 significant | +0.00080 | +0.00121 | +0.00339 | +0.00030 | -0.00726 significant | replication lead |

- Interpretation: increasing direct SERA weight produces the intended voltage
  and intervention-response gains without degrading any F1 point estimate at
  0.02. Increasing F1 weight raises F1 but significantly harms both SERA
  metrics, so the old F1 weight remains preferable. SERA 0.02 also improves
  voltage gain (0.2344 to 0.2456), intervention gain (0.00518 to 0.00564),
  count error, near-spike error, and morphology disparity slightly. Its main
  health cost is refractory violations increasing from 0.243 to 0.276.
- Decision: replicate direct-SERA weighting on seed 20260712 with a local
  0.0125/0.015/0.02/0.025/0.03 bracket. Require replicated voltage and
  response gains, no significant F1 regression, and inspect refractory/count
  movement before promotion.
- Compute: shared panel completed with 21.4-23.1 seconds model compute per
  candidate.
- Artifacts: `artifacts/diagnostics/iteration-076-conditioned-objective-retune/`.

### Iteration 077 premise - direct-SERA objective replication and bracket

- Direct SERA weight 0.02 significantly improves voltage and intervention
  response on the conditioned model while preserving all F1 point estimates.
- Decision: use a fresh seed and compare the promoted 0.01 control with
  0.0125-0.03 under identical 69,377 parameters, loss computation, frozen
  schedule, updates, optimizer, and diagnostics. Select a robust balanced
  weight rather than the largest single-seed SERA gain.

### Iteration 077 - 2026-07-12

- Regime: seed-2 direct-SERA replication and local bracket; exact 69,377
  parameters, frozen balanced 120k schedule, batch 32, 3,750 updates, and
  common initialization/data/optimizer/diagnostics.

| SERA weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0125 | -0.01109 significant | +0.00022 | -0.00804 significant | +0.00023 | -0.01182 significant | +0.00271 significant | reject |
| 0.015 | -0.02490 significant | +0.00027 | -0.00850 significant | -0.00157 | -0.01520 significant | -0.00006 | reject |
| 0.02 | -0.04144 significant | -0.00036 | -0.00908 significant | -0.00106 | -0.01751 significant | -0.00178 significant | compensation lead |
| 0.025 | -0.04857 significant | +0.00081 | -0.01197 significant | -0.00340 | -0.02032 significant | +0.00257 significant | reject |
| 0.03 | -0.05946 significant | +0.00186 | -0.01138 significant | -0.00138 | -0.02042 significant | +0.00288 | reject |

- Interpretation: direct SERA reliably buys voltage accuracy, but the
  previously neutral F1 response does not replicate: every stronger weight
  significantly regresses F1 area and 5 ms F1 on seed 2. Weight 0.02 is the
  only point that also significantly improves response. Its voltage gain moves
  0.2360 to 0.2451 and intervention gain 0.00578 to 0.00606; refractory rate
  is essentially flat (0.2889 to 0.2896), while count ratio worsens 1.449 to
  1.553. This is a specific objective-allocation tradeoff, not broad collapse.
- Decision: do not promote a SERA-only change. Use the measured SERA margin
  and F1-weight response to test a joint compensation frontier in iteration
  78; require no significant F1 regression and gains in both voltage and
  response before any third-seed gate.
- Compute: 216.75 seconds shared panel wall time; each candidate used 21.2-
  23.0 seconds model compute and exactly 3,750 optimizer steps.
- Artifacts: `artifacts/diagnostics/iteration-077-direct-sera-replication/`.

### Iteration 078 premise - joint SERA/F1 compensation frontier

- Stronger SERA weighting has a reproducible voltage margin but shifts the
  model away from tolerant spike timing on seed 2. Separate F1 weighting moves
  F1 in the opposite direction at a smaller SERA cost.
- Decision: test a targeted diagonal frontier rather than another scalar
  sweep: SERA 0.015-0.025 paired with F1 0.075-0.125. Keep an unchanged
  control and identical 69,377 parameters, compute paths, seed-2 schedule,
  updates, optimizer, and diagnostics.

### Iteration 078 - 2026-07-12

- Regime: same-seed joint-objective compensation panel; exact 69,377
  parameters, frozen balanced 120k schedule, batch 32, 3,750 updates, common
  initialization/data/optimizer, and identical objective compute paths.

| SERA / F1 | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.015 / 0.075 | -0.01244 significant | +0.00021 | -0.00471 | -0.00147 | -0.00925 significant | -0.00041 | reject |
| 0.020 / 0.075 | -0.02662 significant | +0.00090 | -0.00607 | -0.00414 | -0.01400 significant | -0.00122 | reject |
| 0.020 / 0.100 | -0.01493 significant | +0.00272 significant | +0.00139 | +0.00124 | -0.00498 | +0.00120 | reject dynamics |
| 0.025 / 0.100 | -0.02582 significant | +0.00238 | +0.00455 | +0.00265 | +0.00117 | +0.00275 | fresh-seed lead |
| 0.025 / 0.125 | -0.01565 significant | +0.00363 significant | +0.00524 | +0.00287 | +0.00038 | +0.00669 significant | reject |

- Interpretation: F1 weight 0.10 is sufficient to recover the tolerant-F1
  loss from stronger SERA. The 0.025/0.10 point improves voltage significantly
  and all F1 point estimates, with response/dynamics uncertainly worse. It
  improves normalized count error 0.533 to 0.522 and refractory violations
  0.289 to 0.239, but default-threshold count ratio worsens 1.449 to 2.303 and
  intervention direction falls 0.654 to 0.645. Higher F1 weight creates
  significant dynamics/response failures.
- Decision: carry 0.025/0.10 to a fresh-seed gate, but explicitly test stronger
  spike-BCE pressure to correct its overprediction. Include a lower joint
  point with the same calibration pressure. Advance only a no-significant-
  regression result to iteration 80.
- Compute: 216.46 seconds shared panel wall time; 21.1-22.9 seconds model
  compute per candidate.
- Artifacts: `artifacts/diagnostics/iteration-078-joint-objective-compensation/`.

### Iteration 079 premise - fresh-seed joint objective and count calibration

- Joint SERA/F1 weighting recovers the seed-2 F1 tradeoff, but the best core
  point overpredicts spikes at the default threshold. Spike BCE is the direct
  differentiable pressure against that calibration failure.
- Decision: on seed 20260713 compare the unchanged promoted objective, the
  uncalibrated 0.025/0.10 lead, spike-BCE weights 0.35/0.50 on that lead, and
  lower 0.0225/0.09-0.10 joint points with BCE 0.35. Hold parameters, compute
  paths, data balance, presentations, updates, and diagnostics fixed.

### Iteration 079 - 2026-07-12

- Regime: seed-3 joint-objective robustness and calibration gate; exact 69,377
  parameters, frozen balanced 120k schedule, batch 32, 3,750 updates, and
  matched initialization/data/optimizer/objective compute.

| SERA / F1 / BCE | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.025 / 0.10 / 0.25 | -0.03838 significant | +0.00114 | +0.00519 | -0.00443 | +0.00525 | -0.00155 | long-run lead |
| 0.025 / 0.10 / 0.35 | -0.03643 significant | +0.00547 significant | +0.01433 significant | +0.00681 | +0.01998 significant | +0.00504 significant | reject tradeoff |
| 0.025 / 0.10 / 0.50 | -0.01453 significant | +0.00110 | +0.00589 | -0.00157 | +0.00562 | +0.00472 significant | reject |
| 0.0225 / 0.09 / 0.35 | -0.01759 significant | +0.00517 significant | +0.00879 | +0.00301 | +0.00663 | +0.03818 significant | reject |
| 0.0225 / 0.10 / 0.35 | -0.01062 significant | +0.00593 significant | +0.00706 | +0.00562 | +0.01034 | +0.03656 significant | reject |

- Interpretation: stronger BCE does not repair calibration cleanly; it shifts
  the frontier toward F1 while significantly harming response and often
  dynamics. The original 0.025/0.10/0.25 joint point is the sole candidate
  without a significant core regression and now has two-seed support for a
  voltage gain with tolerant-F1 preservation. On seed 3 it improves refractory
  violations 0.297 to 0.280, normalized count error 0.595 to 0.512,
  intervention direction/gain, voltage gain, and morphology disparity. Its
  fixed-threshold count ratio worsens 1.431 to 2.909 and remains the explicit
  long-run failure gate.
- Decision: advance only SERA 0.025/F1 0.10/BCE 0.25 to the scheduled 300k
  iteration-80 comparison against the promoted objective on a fresh seed.
  Promote only if core gains survive and count/physiology do not collapse.
- Compute: 218.62 seconds shared panel wall time; 21.4-23.2 seconds model
  compute per candidate.
- Artifacts: `artifacts/diagnostics/iteration-079-joint-objective-calibration/`.

### Iteration 080 premise - 300k joint-objective promotion gate and cleanup

- The SERA 0.025/F1 0.10 objective improves voltage significantly on two 120k
  seeds while preserving tolerant F1; its unresolved risk is fixed-threshold
  spike overprediction.
- Decision: compare unchanged objective and joint lead for 300k presentations
  on fresh seed 20260714 with identical model, initialization, balanced data,
  9,375 optimizer steps, and diagnostics. After evaluation, remove rejected
  final-FiLM, morphology branch-gain, and morphology-reinjection code paths,
  then run full AxoSim and AxoBench evaluator verification.

### Iteration 080 - 2026-07-12

- Regime: scheduled fresh-seed 300k promotion gate; exact 69,377 parameters,
  balanced 60k presentations per morphology, batch 32, 9,375 updates, common
  initialization/data/optimizer/compute, followed by scheduled cleanup.

| objective | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SERA 0.025 / F1 0.10 | -0.05562 significant | -0.00297 | -0.00183 | -0.00008 | -0.00398 | +0.00474 significant | reject long |

- Interpretation: the joint objective's voltage benefit strengthens with
  longer training, but the intervention-response gain does not survive: it
  significantly reverses at 300k, while every F1 point estimate also slips.
  Fixed-threshold count ratio worsens 1.685 to 3.062 and normalized count
  error worsens 0.522 to 0.535. Refractory violations, voltage gain,
  intervention gain, and morphology disparity improve, but these secondary
  gains do not override the core response failure.
- Decision: reject the objective-retune branch and retain the promoted SERA
  0.01/F1 0.05/BCE 0.25 recipe and stable iteration-70 checkpoint.
- Cleanup: removed rejected final morphology FiLM, morphology-specific branch
  gains, and morphology reinjection from active config, model, forward path,
  defaults, and tests. Historical manifests and diagnostics remain as evidence.
- Compatibility: the stable checkpoint exposed six older retired config keys.
  Added an explicit retired-key compatibility filter at checkpoint loading and
  a regression test; unknown non-retired keys remain strict. The real stable
  checkpoint loads as AxoMamba with 69,377 parameters, morphology scale 0.15,
  and five morphology IDs.
- Verification: AxoSim 86/86 and AxoBench iteration evaluator 24/24 pass after
  cleanup and compatibility repair.
- Compute: 334.84 seconds shared panel wall time; control/candidate model
  compute 57.91/54.48 seconds for 9,375 updates each.
- Artifacts: `artifacts/diagnostics/iteration-080-joint-objective-long-gate/`.

### Iteration 081 premise - teacher-grounded intervention training

- Intervention response remains the weakest declared core behavior. The v1
  dataset contains 12,800 paired teacher examples across five interventions
  and five morphologies, with baseline and intervention inputs/targets for
  direct supervised use; current training consumes none of them.
- Decision: extend the frozen shared-data runner to mix paired intervention
  windows without changing total presentations or optimizer steps. First test
  intervention-data fractions under shared loading and matched model compute;
  then add an explicit paired response loss only if data exposure alone is not
  sufficient. Keep held-out diagnostic intervention shards separate.

### Iteration 081 - 2026-07-12

- Regime: leakage-safe teacher-grounded intervention-data fraction panel;
  exact 69,377 parameters, 120k total forward presentations, batch 32, 3,750
  updates, nested frozen replacement masks, common initialization/objective,
  shared ordinary/alternate loading, and 25 diagnostic shards held out whole.

| intervention fraction | actual presentations | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 5,987 | +0.00336 significant | +0.00095 | -0.00200 | +0.00417 | +0.00495 | +0.00231 significant | reject |
| 0.10 | 11,968 | +0.01739 significant | +0.00223 | -0.00717 | -0.00244 | -0.00180 | +0.00098 | reject |
| 0.25 | 29,966 | -0.05136 significant | +0.00290 | -0.01470 significant | -0.00744 | -0.01338 | -0.01618 significant | paired-loss basis |
| 0.50 | 59,956 | +0.05071 significant | +0.00078 | -0.00593 | +0.00337 | -0.00636 | -0.00021 | reject |
| 1.00 | 120,000 | +0.03277 significant | +0.00762 significant | -0.01994 significant | +0.00435 | -0.02948 significant | +0.01090 significant | reject |

- Interpretation: intervention examples contain useful signal, but simple
  replacement has a narrow non-monotonic optimum. At 25%, voltage and held-out
  response SERA improve significantly, voltage/intervention gain increase,
  refractory violations fall 0.253 to 0.223, and morphology disparity falls
  slightly. F1 area significantly regresses, count error rises 0.561 to 0.572,
  and default-threshold count ratio rises 1.258 to 1.506. Lower fractions do
  not improve response and full replacement broadly fails.
- Decision: do not promote data replacement alone. Use matched pairs directly
  in iteration 82: reserve eight of each 32 forward slots for four paired
  baseline/intervention examples, retain 24 ordinary examples, and optimize a
  differentiable response-SERA term. This keeps the 120k forward budget and
  optimizer steps exact while reducing intervention exposure from 25% to 12.5%.
- Engineering: added a frozen morphology-aligned intervention schedule,
  whole-shard evaluator holdout, shared ordinary/alternate loading, nested
  candidate masks, exact mix accounting, and tests. End-to-end smoke passed;
  full AxoSim suite passed 88/88 before the run.
- Compute: 270.63 seconds shared panel wall time versus about 21.6-24.1 seconds
  model compute per candidate; compressed pair loading is material but the
  schedule is reusable.
- Artifacts: `artifacts/diagnostics/iteration-081-intervention-data-mix/`.

### Iteration 082 premise - direct paired response-SERA training

- Simple intervention exposure improves held-out response only at a point that
  loses spike fidelity. The paired corpus supplies the missing supervision:
  the teacher response delta for the same baseline/intervention trace.
- Decision: keep each update at 32 forward sequences by using 24 ordinary
  windows plus four baseline/intervention pairs. Compare paired-data-only
  control and response-SERA weights against the ordinary control with common
  initialization, parameters, forward budget, optimizer steps, and held-out
  diagnostics. Require response gain without significant F1 harm.

### Iteration 082 - 2026-07-12

- Regime: paired response-SERA objective panel; exact 69,377 parameters, 120k
  forward sequences, batch 32, 3,750 updates, 15,000 pairs per paired model,
  common initialization/schedule/optimizer, and whole-shard held-out diagnostics.

| paired setup | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| data only | -0.01325 significant | +0.00002 | -0.00383 | +0.00165 | -0.00157 | -0.00145 | fresh-seed fraction sweep |
| response 0.0025 | -0.01252 significant | -0.00059 | -0.00362 | +0.00519 | +0.00089 | -0.00028 | reject objective |
| response 0.005 | -0.00655 significant | -0.00086 | -0.00355 | +0.00748 | -0.00315 | +0.00135 | reject |
| response 0.01 | +0.02314 significant | -0.00052 | -0.00463 | +0.00325 | -0.00741 | +0.00215 significant | reject |
| response 0.02 | +0.02651 significant | +0.00049 | -0.00430 | +0.00759 | -0.00426 | +0.00388 significant | reject |

- Interpretation: paired data alone is the best point: it significantly
  improves voltage with a small response gain and no significant F1 loss.
  Positive response-SERA weights monotonically weaken response transfer even
  though the training response component decreases (5.820 at weight zero to
  5.790 at weight 0.02). At 0.01/0.02, held-out response and voltage regress
  significantly. The differentiable proxy overfits training-pair deltas and
  is rejected. Paired data only slightly improves threshold count calibration
  and morphology disparity but worsens intervention direction/gain, count
  error, voltage gain, and refractory rate.
- Decision: retain paired data as a provisional data-distribution result, not
  the response loss. On a fresh seed, sweep paired sequence allocation from
  12.5% to 75% to find whether a balanced response/F1 point replicates.
- Engineering: pair packing keeps 120k total forward sequences exact; response
  losses mirror AxoBench relevance/SERA structure. Added tests and a real-data
  smoke. Weight-zero pair panels now skip unused response-SERA computation.
- Compute: 353.10 seconds shared panel wall time; ordinary model compute 24.48
  seconds and paired models about 28.5 seconds due matched response-loss work.
- Artifacts: `artifacts/diagnostics/iteration-082-paired-response-sera/`.

### Iteration 083 premise - paired-data allocation replication

- Paired data without an explicit response loss improves voltage and slightly
  improves response at 25% paired sequence allocation, with no significant
  core regression. The optimum allocation is unknown and the result is one seed.
- Decision: on seed 20260716 compare ordinary training with 12.5%, 25%, 37.5%,
  50%, and 75% paired sequence slots, no response loss, exact 120k forward
  sequences/3,750 updates/69,377 parameters, shared held-out-safe loading, and
  full diagnostics. Require replicated response or broader core value.

### Iteration 083 - 2026-07-12

- Regime: fresh-seed paired-data allocation sweep; exact 69,377 parameters,
  120k forward sequences, batch 32, 3,750 updates, no response proxy, common
  initialization/optimizer, shared morphology-aligned pairs, and whole-shard
  held-out diagnostics.

| paired sequence fraction | response pairs | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.125 | 7,500 | -0.00818 significant | +0.00101 | +0.00283 | -0.00454 | +0.01029 significant | -0.00287 significant | seed-3 lead |
| 0.250 | 15,000 | -0.00484 significant | +0.00135 | +0.00252 | +0.00008 | +0.01021 significant | -0.00353 significant | replicated lead |
| 0.375 | 22,500 | -0.00086 | +0.00188 | +0.00166 | -0.00412 | +0.00855 | -0.00169 significant | secondary |
| 0.500 | 30,000 | -0.00795 significant | +0.00035 | -0.00228 | -0.00039 | +0.00333 | -0.00468 significant | response lead, F1 mixed |
| 0.750 | 45,000 | +0.00266 | +0.00045 | -0.00159 | -0.00120 | +0.00402 | -0.00379 significant | reject |

- Interpretation: paired teacher data robustly improves held-out intervention
  behavior without the failed response proxy. Fractions 0.125 and 0.25
  significantly improve voltage, response, and 5 ms F1 simultaneously; F1
  area also rises and dynamics/exact F1 remain uncertain. Fraction 0.25 now
  has two-seed support for voltage/response gains with no significant core
  regression. Fraction 0.125 improves count error 0.518 to 0.498 and
  refractory violations 0.271 to 0.238; fraction 0.25 improves count ratio
  1.357 to 1.320 and refractory violations to 0.245. Both slightly weaken
  intervention direction/gain, so a third seed remains necessary.
- Decision: run a seed-3 low-allocation bracket 0.0625-0.3125. Promote the
  smallest fraction that retains response/voltage/tolerant-F1 gains without a
  significant core or health regression.
- Compute: 323.55 seconds shared panel wall time; response-loss skipping cuts
  paired model compute to about 22.5 seconds, leaving compressed pair loading
  as the dominant overhead.
- Artifacts: `artifacts/diagnostics/iteration-083-paired-data-allocation/`.

### Iteration 084 premise - paired-data third-seed low-allocation gate

- Paired fractions 0.125/0.25 improve voltage, response, and 5 ms F1 on seed 2;
  0.25 also had a balanced seed-1 result. Lower allocation is cheaper and may
  preserve standard-distribution exact F1 better.
- Decision: on seed 20260717 compare ordinary training with fractions 0.0625,
  0.125, 0.1875, 0.25, and 0.3125 under exact forward/update/parameter/data-
  loading controls and held-out diagnostics. Use this as the promotion gate.

### Iteration 084 - 2026-07-12

- Regime: third-seed paired-data promotion gate; exact 69,377 parameters,
  120k forward sequences, batch 32, 3,750 updates, common initialization,
  held-out-safe shared pair loading, no response proxy, and full diagnostics.

| paired fraction | response pairs | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0625 | 3,750 | +0.00331 | -0.00028 | -0.00123 | +0.00118 | -0.00501 | -0.00271 significant | unbalanced lead |
| 0.125 | 7,500 | +0.05651 significant | -0.00056 | +0.00068 | +0.00050 | -0.00656 | +0.01195 significant | reject |
| 0.1875 | 11,250 | +0.05384 significant | +0.00023 | +0.00229 | +0.00056 | -0.01044 | +0.01208 significant | reject |
| 0.25 | 15,000 | +0.05007 significant | -0.00170 | -0.00558 | -0.00013 | -0.02424 significant | +0.01280 significant | reject |
| 0.3125 | 18,750 | +0.05174 significant | -0.00254 | -0.01543 significant | -0.00136 | -0.03103 significant | +0.01290 significant | reject |

- Interpretation: the seed-2 paired-data frontier does not replicate. Every
  fraction at or above 0.125 significantly regresses voltage and held-out
  response; higher fractions also significantly lose tolerant F1. Fraction
  0.0625 significantly improves response but moves voltage, F1 area, and 5 ms
  F1 adversely and uncertainly. Across three seeds, paired fraction 0.25 has
  two favorable results and one decisive multi-core failure, so promotion
  would be seed selection rather than robust improvement.
- Decision: reject paired data as a promoted recipe and retain ordinary
  training. Keep the leakage-safe pair machinery temporarily for future
  literature-grounded experiments, but queue the failed direct response-SERA
  objective and unused replacement modes for iteration-90 cleanup unless new
  evidence requires them. Run the requested broad literature scan before
  selecting iteration 85.
- Compute: 323.02 seconds shared panel wall time; paired model compute about
  22.1 seconds, confirming compressed pair loading dominates latency.
- Artifacts: `artifacts/diagnostics/iteration-084-paired-data-third-seed/`.

### Literature refresh after iteration 084

- Decision: conduct a broad, source-verified scan across neural dynamical
  systems, operator/ODE simulation, biological and spiking networks,
  intervention-aware surrogate modeling, and time-series/state-space models.
  Store the review in `literature/iteration-084-large-scan.md`, rank concrete
  AxoMamba experiments, and schedule the next refresh at iteration 100.
- Completed: `literature/iteration-084-large-scan.md` records the protocol,
  source map, transferability analysis, ranked experiment queue, and limits.
  The strongest immediate experiment is a causal multi-timescale filtered
  spike-train loss, followed by an exact-budget parallel fast/slow Mamba path.
  Neural-ODE/operator rollout methods and the six-state BSSM are deferred until
  a matching failure premise exists. The next scan is iteration 100, then every
  tenth iteration.

### Iteration 085 premise - multi-timescale filtered spike-train supervision

- The promoted F1 surrogate compares pooled event probability inside tolerance
  windows, while the baseline still has exact F1 0.054, 5 ms F1 0.261, count
  error 0.470, and a 0.207 tolerance gap. Van Rossum/SuperSpike-style filtered
  event errors provide dense, scale-explicit timing and count supervision.
- Decision: add a normalized causal filtered-probability loss and compare 1,
  2, and 5 ms filters plus a 1/2/5 ms blend against the unchanged promoted
  objective. Select a single conservative coefficient from measured initial
  gradient scale; require F1-area and exact-or-5-ms improvement without
  significant voltage or response-SERA regression.

### Iteration 085 - 2026-07-12

- Regime: literature-grounded scientific objective test; exact 69,377
  parameters, 120k presentations, batch 32, 3,750 updates, common
  initialization, frozen balanced schedule, and full held-out diagnostics.
- Gradient-scale audit: after symmetric signal-energy normalization, raw
  filtered-loss gradient norms were 21-25 at initialization and 58-60 on the
  promoted checkpoint. Weight `2e-4` matches the existing weighted F1 gradient
  at initialization and is about one fifth of it on the promoted checkpoint.
  The coefficient was frozen across candidates.

| filter | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 ms | +0.00202 significant | +0.00051 significant | -0.00053 | -0.00195 | +0.00099 | -0.00046 significant | reject |
| 2 ms | +0.00201 significant | +0.00045 significant | -0.00032 | -0.00202 | +0.00131 | -0.00041 significant | reject |
| 5 ms | +0.00178 significant | +0.00042 significant | -0.00061 | -0.00153 | +0.00098 | -0.00037 significant | reject |
| 1/2/5 ms | +0.00196 significant | +0.00044 significant | -0.00084 | -0.00200 | +0.00023 | -0.00043 significant | reject |

- Interpretation: all timescales induce the same small trade, so there is no
  scale-specific mechanism to replicate. Tolerant F1 rises by at most 0.00131,
  but exact F1 and F1 area fall, voltage and dynamics SERA regress
  significantly, and spike-count error is slightly worse for every candidate.
  The statistically significant response gain is below 0.0005 mV and is not a
  material counterweight. This filtered loss does not repair the pooled-F1
  objective's failure in this model.
- Decision: reject filtered spike-train supervision and do not tune its
  coefficient. Retain its small executable path only until the iteration-90
  cleanup, then remove it unless a later retained experiment depends on it.
  Advance to an exact-budget parallel fast/slow Mamba architecture.
- Compute: 203.96 seconds panel wall time; candidate model compute 22.7-24.1
  seconds versus 23.3 seconds for control. Parameters, presentations, and
  optimizer steps are exactly matched; the filtering operation adds only its
  inherent objective compute.
- Artifacts: `artifacts/diagnostics/iteration-085-filtered-spike-train-loss/`.

### Iteration 086 premise - exact-budget parallel fast/slow Mamba

- Multi-scale Mamba, parallel multi-compartment spiking models, and dual-memory
  networks independently motivate distinct temporal state paths. Earlier
  AxoMamba tests changed convolutional preprocessing or decay initialization
  inside one sequential path; they did not compare parallel state evolution.
- Decision: reinterpret the promoted model's exact same two initialized Mamba
  blocks as parallel paths. Block 0 processes every 1 ms. Block 1 processes a
  causal trailing-average stream sampled and held every 2, 4, 8, or 16 ms; its
  residual is fused into the fast path. Parameters remain exactly 69,377, with
  identical seed, frozen I85 schedule, objective, presentations, and updates.
  Reuse I85's cached control evaluation and checkpoint rather than retraining or
  reevaluating an identical baseline.

### Iteration 086 - 2026-07-12

- Regime: literature-grounded structural test; exact 69,377 parameters, 120k
  presentations, batch 32, 3,750 updates, common initialization, frozen I85
  schedule, and cached-control held-out diagnostics.

| slow stride | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | compute seconds | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 ms | -0.01251 significant | -0.00184 | -0.02909 significant | +0.00117 | -0.04502 significant | +0.00221 | 23.33 | reject |
| 4 ms | -0.01147 | +0.00398 | -0.05971 significant | -0.01359 significant | -0.08432 significant | +0.00982 significant | 21.71 | reject |
| 8 ms | +0.09606 significant | +0.00079 | -0.02492 significant | -0.01263 significant | -0.02974 significant | +0.02322 significant | 21.53 | reject |
| 16 ms | +0.03716 significant | -0.00245 | -0.01267 significant | -0.00873 significant | -0.01813 significant | +0.01658 significant | 21.50 | reject |

- Interpretation: replacing sequential depth with parallel temporal paths
  destroys tolerant event fidelity. The 2 ms path buys a real voltage-SERA
  improvement but loses 0.029 F1 area and 0.045 at 5 ms; larger strides do not
  recover a balanced point and usually regress intervention response. Count
  error and refractory violations also worsen sharply. Reduced-rate state is
  computationally valid and somewhat faster, but the second sequential Mamba
  block is essential to event processing in the current width/budget.
- Decision: reject the complete parallel replacement family without stride
  tuning. Queue `parallel_slow_stride` and its implementation/tests for I90
  removal. Run the literature scan's frequency-band premise audit next; do not
  add a frequency objective unless held-out spectra identify a concrete defect.
- Compute: 177.30 seconds panel wall time; each candidate has the exact baseline
  parameter/update/presentation count. Per-candidate compute is 21.5-23.3
  seconds versus the cached sequential control's 23.3 seconds.
- Artifacts: `artifacts/diagnostics/iteration-086-parallel-fast-slow-mamba/`.

### Iteration 087 premise - frequency-banded voltage error

- Held-out spectral audit over 90 scored traces shows a monotonic amplitude
  collapse despite correct slow variation: prediction/teacher amplitude gain is
  0.998 below 5 Hz, 0.672 at 5-20 Hz, 0.481 at 20-80 Hz, and 0.312 at 80-250
  Hz. Intervention-response gains are 0.762, 0.488, 0.323, and 0.216 over the
  same bands. The two middle/high bands carry 31.7% of ordinary teacher power
  and 51.3% of intervention-response power.
- Decision: the premise passes. Add a normalized frequency-banded voltage-error
  objective and compare 20-80 Hz, 80-250 Hz, their equal normalized blend, and
  broad 20-250 Hz. Select one common coefficient from a gradient-scale audit;
  keep architecture, seed, frozen I85 schedule, parameter count, presentations,
  and updates fixed, and reuse the I85 baseline cache.
- Artifact: `artifacts/diagnostics/iteration-087-spectral-premise-audit.json`.

### Iteration 087 - 2026-07-12

- Regime: premise-gated objective test; exact 69,377 parameters, 120k
  presentations, batch 32, 3,750 updates, common initialization, frozen I85
  schedule, and cached-control held-out diagnostics. Frequency-loss weight
  `0.005` was fixed across all diagnosed bands after gradient-scale audit.

| frequency band | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20-80 Hz | +0.00200 significant | +0.00059 | +0.00194 | -0.00201 | +0.00145 | -0.00172 significant | reject |
| 80-250 Hz | +0.01243 significant | -0.00826 significant | -0.00091 | -0.00721 significant | +0.00281 | -0.00777 significant | reject |
| equal 20-80/80-250 Hz | +0.01002 significant | -0.00571 significant | +0.00086 | -0.00684 significant | +0.01019 | -0.00240 significant | reject |
| broad 20-250 Hz | -0.01668 significant | -0.00282 significant | +0.01054 significant | -0.01044 significant | +0.02225 significant | -0.00683 significant | replicate/lower-weight gate |

- Interpretation: broad-band supervision is the first candidate in this phase
  to significantly improve voltage SERA, dynamics SERA, response SERA, F1 area,
  and 5 ms F1 together. Its exact-F1 loss is equally clear and blocks promotion.
  A post-run spectral audit finds only small relative-error reductions
  (20-80 Hz 0.90905 to 0.90709; 80-250 Hz 0.98501 to 0.98163) and no amplitude
  restoration; gain actually falls from 0.481 to 0.472 and 0.312 to 0.301.
  Thus the downstream lead is real on this seed, while the amplitude-repair
  explanation is not supported.
- Decision: carry only the broad 20-250 Hz objective to a fresh seed, comparing
  weight 0.005 with a predeclared half-weight 0.0025 intended to preserve exact
  F1. Do not advance the disjoint-band variants. Promotion still requires no
  significant exact-F1 regression and replicated SERA/F1-area benefit.
- Compute: 181.61 seconds panel wall time; candidate model compute 22.5-24.2
  seconds. Parameters, presentations, and optimizer steps exactly match the
  cached control.
- Artifacts: `artifacts/diagnostics/iteration-087-frequency-voltage-loss/` and
  `artifacts/diagnostics/iteration-087-broad-candidate-spectral-audit.json`.

### Iteration 088 premise - broad-frequency replication and exact-F1 gate

- The 20-250 Hz weight-0.005 candidate has a five-core significant lead on
  seed 20260718 but significantly loses exact F1, and it reduces band-relative
  error without restoring amplitude. This may be a useful objective trade or a
  seed-specific threshold shift.
- Decision: on fresh seed/schedule 20260719 compare the unchanged objective
  with broad-frequency weights 0.0025 and 0.005. The half-weight is the sole
  corrective candidate and must preserve exact F1 while retaining SERA/F1-area
  benefit; the original weight must replicate before either advances.

### Iteration 088 - 2026-07-12

- Regime: fresh-seed replication and bounded coefficient correction; exact
  69,377 parameters, 120k presentations, batch 32, 3,750 updates, common
  initialization, fresh balanced schedule, and full held-out diagnostics.

| weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0025 | -0.00947 significant | -0.00145 | -0.00312 | -0.00037 | -0.00596 | -0.00686 significant | reject |
| 0.005 | -0.01554 significant | -0.00184 | -0.00065 | -0.00089 | -0.00165 | -0.00524 significant | third-seed gate |

- Interpretation: weight 0.005 replicates its voltage- and response-SERA gains
  and no longer significantly loses exact F1, but its seed-1 F1-area and 5 ms
  improvements disappear. Half-weight does not preserve tolerant F1 better and
  yields a smaller voltage gain. The result is a robust two-axis SERA lead with
  unresolved F1 neutrality, not a replicated five-core win.
- Decision: reject weight 0.0025. Run one final fresh-seed control versus weight
  0.005. Carry it to the 300k iteration-90 gate only if the third seed preserves
  all F1 metrics and retains a material SERA advantage.
- Compute: 157.55 seconds panel wall time; 22.6-22.8 seconds candidate model
  compute, with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-088-frequency-broad-replication/`.

### Iteration 089 premise - broad-frequency third-seed gate

- Weight 0.005 has two significant voltage/response-SERA wins but one strong
  and one absent F1-area result. A third independent seed is necessary before
  spending the scheduled 300k budget.
- Decision: compare only unchanged control and broad 20-250 Hz weight 0.005 on
  seed/schedule 20260720. Require no significant F1 regression and a material
  voltage or response gain; otherwise close the path at iteration 90 cleanup.

### Iteration 089 - 2026-07-12

- Regime: third independent 120k seed; exact 69,377 parameters, batch 32,
  3,750 updates, common initialization, fresh balanced schedule, and full
  held-out diagnostics.
- Results: weight 0.005 changes voltage SERA by +0.00001, dynamics by -0.00175,
  response SERA by -0.00123, F1 area by +0.00799 significant, exact F1 by
  +0.00272, and 5 ms F1 by +0.00647. No core metric regresses significantly.
- Interpretation: the candidate's third seed is F1-positive and SERA-neutral,
  complementing two prior significant voltage/response wins. Across the three
  short seeds, mean deltas are approximately -0.0107 voltage SERA, -0.00443
  response SERA, +0.00596 F1 area, -0.00287 exact F1, and +0.00903 at 5 ms.
  Seed heterogeneity remains, but the aggregate is balanced enough for the
  scheduled long test.
- Decision: advance unchanged broad 20-250 Hz weight 0.005 against control at
  300k on the same seed. Promote only if the long run has no significant core
  regression and retains a useful SERA or F1-area gain.
- Compute: 135.08 seconds panel wall time; 22.5/22.7 seconds model compute with
  exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-089-frequency-broad-third-seed/`.

### Iteration 090 premise - 300k broad-frequency promotion gate and cleanup

- Decision: train control and broad 20-250 Hz weight 0.005 for exactly 300k
  presentations (9,375 optimizer steps) on seed 20260720. In parallel, audit
  rejected executable paths; after diagnostics, delete failed intervention
  mixing/paired-response, filtered-spike-loss, and parallel-fast/slow code plus
  tests/config knobs, while retaining historical manifests and result reports.
  Run full AxoSim/AxoBench tests and stable-checkpoint loading after cleanup.

### Iteration 090 - 2026-07-12

- Regime: scheduled long promotion gate; exact 69,377 parameters, 300k
  presentations, batch 32, 9,375 updates, common initialization, frozen
  balanced schedule, and full held-out diagnostics.
- Results: broad-frequency weight 0.005 changes voltage SERA by -0.00372
  significant, dynamics by -0.00076, response SERA by -0.00112, F1 area by
  -0.00005, exact F1 by +0.00251, and 5 ms F1 by -0.00239. Only voltage reaches
  significance; no metric significantly regresses.
- Interpretation: the long candidate is safe but not materially better. Its
  short-run F1-area/5 ms and response gains vanish, and its voltage benefit
  shrinks substantially. The post-I87 audit also showed that the objective did
  not restore the diagnosed frequency amplitudes. A tiny paired-significant
  voltage delta alone is insufficient to add a permanent objective and FFT path.
- Decision: reject broad-frequency promotion, retain the existing objective and
  stable AxoMamba checkpoint, and remove the entire failed I81-I90 executable
  branch during scheduled cleanup. Historical manifests and diagnostics remain.
- Compute: 337.69 seconds panel wall time; 57.2/57.8 seconds model compute,
  exactly 300k presentations and 9,375 updates each. The frozen 300k schedule is
  retained for reuse.
- Artifacts: `artifacts/diagnostics/iteration-090-long-frequency-gate/`.

### Iteration 090 cleanup - 2026-07-12

- Removed the complete rejected I81-I90 executable branch: frozen
  intervention mixing and paired-response accounting/objectives, causal
  filtered spike-train losses, parallel fast/slow Mamba, and frequency-band
  voltage objectives. Removed their runtime knobs, validation, runner paths,
  and tests together; no retired symbol remains under `src/`, `scripts/`,
  `tests/`, or `configs/`.
- Retained all manifests, diagnostic outputs, and this written record. Those
  are historical evidence, not live model branches. Disk usage is dominated
  by retained experiment artifacts (`docs` 4.0 GB and `runs` 1.1 GB), not a
  12k-line executable expansion. The reusable autoresearch implementation is
  now 656 lines in `src/axosim/autoresearch.py` plus 424 runner lines and 337
  focused test lines.
- Verification: AxoSim 86/86 tests pass, AxoBench's iteration evaluator 24/24
  passes, and the stable checkpoint loads as `AxoMamba` with 69,377 parameters
  and the five promoted morphology IDs. The promoted architecture, objective,
  config, and checkpoint are unchanged.
- Decision: begin iteration 91 with a diagnostic-only audit of the promoted
  latent representation and signed synaptic-current structure. Implement a
  new side path or auxiliary target only if the audit exposes a concrete,
  decodable residual defect.

### Iteration 091 premise - E/I information and selective timescales

- A 30-trace, morphology-balanced held-out audit finds that only 3.22% of
  active branch-times contain simultaneous excitation and inhibition, erasing
  6.06% of total routed event mass. Ridge probes using signed branch current,
  separate E/I currents, or collision current all worsen held-out residual
  RMSE; the final latent improves it by only 0.35%. Its 58 dimensions have
  participation-ratio rank 2.05 and rank 12 at 95% variance, but the unused
  directions do not contain an obvious linear voltage correction.
- The persistent temporal defect remains real: held-out residual lag-1 is
  0.419. A second audit finds that layer 1's learned Mamba timestep is almost
  invariant to sparse input (event-load correlation 0.003, burst/quiet ratio
  1.010), whereas layer 2 responds modestly (correlation 0.062, ratio 1.316).
- Decision: reject E/I splitting. Test a learned unsigned event-load embedding
  before layer 1 at scales 0.025-0.2. Include the embedding and forward path in
  the scale-zero control so all candidates have exactly the same parameters
  and compute. Require a core improvement without response/F1 regression.
- Artifacts: `artifacts/diagnostics/iteration-091-ei-premise-audit/`.

### Iteration 091 - 2026-07-12

- Regime: premise-gated input-conditioned-timescale proxy; exact 69,435
  parameters, 120k presentations, batch 32, 3,750 updates, common
  initialization, fresh balanced frozen schedule, and full held-out
  diagnostics. Every candidate carries and computes the same 58-parameter
  load embedding; only the fixed scale differs.

| load scale | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.025 | -0.04992 significant | -0.00035 | -0.02519 significant | -0.00704 | -0.03151 significant | +0.00526 significant | reject tradeoff |
| 0.05 | +0.02977 significant | +0.00293 | -0.00301 | +0.00278 | -0.01526 | +0.00512 significant | reject |
| 0.1 | -0.01277 significant | +0.00077 | -0.01719 significant | -0.00195 | -0.02649 significant | +0.00171 | reject tradeoff |
| 0.2 | +0.03751 significant | -0.00061 | -0.00955 | +0.00110 | -0.02013 significant | +0.01631 significant | reject |

- Interpretation: unsigned load is powerful enough to move voltage SERA, but
  it shifts rather than improves the core frontier. The best voltage point
  loses response and tolerant spike timing significantly, and no scale
  approaches a balanced result. This does not justify true custom-delta
  machinery: the cheap proxy demonstrates that event-load conditioning is not
  missing a uniformly useful signal under the current objective.
- Decision: reject the complete load-conditioning family without coefficient
  tuning. Queue its config, parameter, forward path, and tests for iteration
  100 cleanup. Retain the promoted model/checkpoint unchanged.
- Compute: 197.87 seconds shared-panel wall time; candidate model compute
  21.65-23.14 seconds with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-091-input-load-conditioning/`.

### Iteration 092 premise - direct near-spike voltage supervision

- Held-out voltage RMSE within 5 ms of teacher spikes is 4.54 times global
  RMSE, the largest localized voltage failure. Previous waveform and frequency
  heads did not train on this diagnostic directly.
- A one-batch gradient audit measures radius-5 near-spike RMSE gradient norm
  26.43 versus 3.20 for the quadrature of existing weighted terms. Weights
  0.003/0.006/0.012/0.024 therefore span roughly 2.5%-20% of existing gradient
  scale without arbitrary orders-of-magnitude search.
- Decision: compare that bracket using a common 5 ms target-spike mask. The
  weight-zero control computes and reports the same masked loss, preserving
  parameter, model-compute, data, and update matching.
- Artifact: `artifacts/diagnostics/iteration-092-near-spike-voltage-loss/gradient-scale-audit.json`.

### Iteration 092 - 2026-07-12

- Regime: direct failure-diagnostic objective; exact 69,377 parameters, 120k
  presentations, batch 32, 3,750 updates, common initialization/frozen
  schedule, matched mask computation, and full held-out diagnostics.

| weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.003 | -0.02269 significant | -0.00405 significant | -0.01605 significant | -0.00026 | -0.02697 significant | -0.00863 significant | compensation basis |
| 0.006 | -0.03574 significant | -0.00487 significant | -0.01971 significant | -0.00063 | -0.03641 significant | -0.01119 significant | reject tradeoff |
| 0.012 | -0.07497 significant | -0.00258 | -0.02645 significant | -0.00008 | -0.04457 significant | -0.01405 significant | reject tradeoff |
| 0.024 | -0.09126 significant | -0.00389 | -0.03529 significant | -0.00968 | -0.04415 significant | -0.01513 significant | reject tradeoff |

- Interpretation: every coefficient improves voltage and response SERA
  significantly, and the two smallest also improve dynamics significantly.
  The objective therefore exposes real optimization headroom, but monotonically
  damages tolerant event fidelity. At weight 0.003 near-spike error improves
  only 4.5357 to 4.5293, while spike-count error rises to 0.648 and refractory
  violations to 0.293; the downstream SERA gain is broader than the proposed
  localized repair.
- Decision: reject weights 0.006-0.024. Give weight 0.003 one fresh-seed
  replication plus a bounded F1-weight compensation bracket at
  0.075/0.10/0.125. Close the path if no point recovers tolerant F1 while
  preserving its three-axis SERA gain.
- Compute: 199.57 seconds panel wall; candidate model compute 21.94-23.65
  seconds with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-092-near-spike-voltage-loss/`.

### Iteration 093 premise - fresh-seed F1 compensation gate

- Near-spike weight 0.003 improves all three SERA axes significantly on seed 1
  but significantly lowers F1 area and 5 ms F1. This could be a seed-specific
  threshold shift or a structural objective conflict.
- Decision: on fresh seed 20260722, replicate weight 0.003 unchanged and
  compare F1 weights 0.075/0.10/0.125. Keep a common radius-5 loss computation
  in the baseline. Advance only a point with no significant core regression.

### Iteration 093 - 2026-07-12

- Regime: fresh-seed replication and bounded F1 compensation; exact 69,377
  parameters, 120k presentations, batch 32, 3,750 updates, common fresh
  initialization/schedule, matched objective computation, and diagnostics.

| near / F1 weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.003 / 0.05 | -0.00898 significant | +0.00056 | +0.01272 significant | +0.00454 | +0.01769 significant | -0.00965 significant | third-seed lead |
| 0.003 / 0.075 | -0.01074 significant | +0.00180 | +0.00662 significant | +0.00389 | +0.00760 significant | -0.00632 significant | secondary |
| 0.003 / 0.10 | -0.01133 significant | +0.00403 significant | +0.00824 significant | +0.00659 significant | +0.00842 significant | -0.00342 significant | reject dynamics |
| 0.003 / 0.125 | -0.01130 significant | +0.00546 significant | +0.00490 | +0.00694 significant | +0.00283 | -0.00076 | reject dynamics |

- Interpretation: the unchanged objective is the strongest balanced point on
  seed 2, significantly improving voltage, response, F1 area, and 5 ms F1;
  exact F1 also rises and only dynamics moves adversely, uncertainly. It also
  reduces count error 0.563 to 0.536 and refractory violations 0.253 to 0.233.
  Thus iteration 92's F1 collapse is not structural, but the sign reversal
  demonstrates substantial seed heterogeneity. Stronger F1 pressure is
  unnecessary and at 0.10+ causes significant dynamics regression.
- Decision: reject F1 compensation. Third-seed test near-spike weights
  0.001/0.002/0.003 under the unchanged F1 objective. The smaller points are
  predeclared safeguards against seed-1 F1 sensitivity; close the path if no
  coefficient is balanced.
- Compute: 200.69 seconds panel wall; 21.93-23.73 seconds candidate model
  compute with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-093-near-spike-f1-compensation/`.

### Iteration 094 premise - third-seed low-weight robustness bracket

- Weight 0.003 has opposite tolerant-F1 signs across two seeds despite
  consistent SERA potential. Smaller weights may preserve its voltage signal
  while reducing gradient interference with spike calibration.
- Decision: on fresh seed 20260723 compare weights 0.001/0.002/0.003 with the
  matched weight-zero control. Require no significant core regression and
  favor a point with response plus F1 movement in the same direction.

### Iteration 094 - 2026-07-12

- Regime: third independent seed and bounded lower-weight bracket; exact
  69,377 parameters, 120k presentations, batch 32, 3,750 updates, common
  initialization/schedule, matched mask compute, and full diagnostics.

| weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.001 | -0.00229 significant | -0.00264 significant | +0.00466 | +0.00572 | +0.00762 | +0.00074 | secondary |
| 0.002 | -0.00610 significant | -0.00357 significant | +0.00363 | +0.00314 | +0.00637 | -0.00121 significant | replication lead |
| 0.003 | -0.00800 significant | -0.00252 | +0.00096 | +0.00587 | -0.00028 | +0.00329 significant | reject response |

- Interpretation: the useful interval is below 0.003. Weight 0.002 improves
  all six core point estimates and all three SERA axes significantly, while
  0.001 is F1-positive but response-neutral/adverse. Weight 0.003 confirms the
  seed-sensitive response failure and loses its tolerant-F1 gain. The result
  is a new lower-weight lead, not evidence that 0.003 is robust.
- Decision: reject 0.003. Fresh-seed bracket 0.0015/0.002/0.0025 around the
  balanced point; carry only a replicated no-regression coefficient toward
  iteration 100.
- Compute: 178.23 seconds panel wall; 21.98-23.68 seconds candidate compute,
  with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-094-near-spike-third-seed/`.

### Iteration 095 premise - final bounded near-spike replication

- Weight 0.002 is balanced on its first direct test, but a single seed is not
  enough and the user explicitly prefers structural progress over scalar
  polishing. This is the final coefficient iteration for the mechanism.
- Decision: fresh-seed compare 0.0015/0.002/0.0025. Retain a coefficient only
  as a fixed long-gate candidate if it produces a practically material effect;
  do not continue local tuning afterward.

### Iteration 095 - 2026-07-12

- Regime: final bounded coefficient replication; exact 69,377 parameters,
  120k presentations, batch 32, 3,750 updates, fresh common initialization and
  schedule, matched mask compute, and full diagnostics.

| weight | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0015 | -0.01552 significant | -0.00128 | +0.00072 | -0.00130 | -0.00341 | -0.00474 significant | close tuning |
| 0.002 | -0.02357 significant | -0.00124 | +0.00099 | +0.00302 | -0.00243 | -0.00655 significant | fixed long candidate |
| 0.0025 | -0.02654 significant | +0.00110 | +0.00327 | +0.00497 significant | +0.00028 | -0.00721 significant | no further tuning |

- Interpretation: all coefficients improve voltage and response significantly,
  but only voltage is clearly material under the tightened practical-effect
  rule; F1 changes are small. Weight 0.002 now has two-seed mean deltas of
  about -0.01484 voltage, -0.00240 dynamics, -0.00388 response, +0.00231 F1
  area, +0.00308 exact F1, and +0.00197 at 5 ms. This is balanced enough to
  preserve as a fixed long-run candidate, not strong enough to justify more
  scalar search.
- Decision: end the objective-tuning batch. Keep weight 0.002 unchanged for a
  possible iteration-100 gate and pivot immediately to structural,
  literature-derived mechanisms. Low `1e-3` voltage deltas will no longer be
  called meaningful solely because bootstrap significance fires.
- Compute: 177.34 seconds panel wall; 21.53-23.17 seconds candidate compute,
  with exact parameter/presentation/update matching.
- Artifacts: `artifacts/diagnostics/iteration-095-near-spike-weight-replication/`.

### Iteration 096 premise - multi-timescale liquid soma state

- CfC/LTC-style closed-form dynamics and dual-memory models suggest an
  input-conditioned mixture of temporal states rather than one fixed smoothed
  output. Simple output smoothing was already rejected, so it is included as
  the structural control rather than treated as the proposed mechanism.
- Implementation: every candidate computes the raw soma path plus causal 5 ms
  and 25 ms exponential states and carries the same zero-weight-initialized
  three-way gate. Inactive, fixed-mixture, and feature-adaptive candidates
  therefore have exact parameter/compute matching; fixed and adaptive outputs
  are identical at initialization. Tests verify inactive equivalence and
  causality.
- Decision: train the three structures on a fresh 120k seed. The adaptive path
  must materially beat both inactive AxoMamba and fixed smoothing without a
  core regression; otherwise reject the liquid-readout family.

### Iteration 096 - 2026-07-12

- Regime: literature-derived structural readout test; exact 69,554 parameters,
  120k presentations, batch 32, 3,750 updates, common fresh initialization and
  schedule, identical fast/5 ms/25 ms path compute, and full diagnostics.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed mixture | +0.00017 | +0.00165 | +0.00808 significant | +0.00275 | +0.00614 | +0.00915 significant | reject smoothing |
| adaptive mixture | -0.02117 significant | -0.00002 | +0.00307 | -0.00481 | +0.00000 | +0.01154 significant | reject response |

- Interpretation: feature-adaptive mixing produces a practically material
  voltage gain but worsens intervention response by half that magnitude in the
  wrong direction. A 20-trace gate audit confirms real structural use: raw
  occupancy collapses to 0.27%, the 5 ms path receives 81.5%, and the 25 ms
  path 18.2%; near spikes shift further to 88.0% on the 5 ms path. The model
  learned strong smoothing and that mechanism is harmful for perturbation
  response. Fixed smoothing independently fails response and does not improve
  voltage.
- Decision: reject the full liquid-readout family without tuning mix priors,
  raw floors, or taus. Queue it for iteration-100 cleanup. Move the structural
  intervention upstream into state construction using explicit six-state
  fast/slow E/I current ports suggested by the BSSM/hybrid literature and the
  older RNN-vs-Mamba state diagnostics.
- Compute: 159.19 seconds panel wall; 22.44-24.47 seconds candidate model
  compute with exact parameters, presentations, updates, and paths.
- Artifacts: `artifacts/diagnostics/iteration-096-liquid-soma-state/`, including
  `gate-occupancy-audit.json`.

### Iteration 097 premise - six-state biophysical current ports

- Prior RNN activation diagnostics found explicit memory channels correlated
  strongly with slow E/I traces while AxoMamba's final state remained
  low-rank. The existing standalone BSSM is an unfused, hard-coded full
  replacement and is not ready for a fair 120k panel, but the literature scan
  ranked a bounded hybrid current-state path as the defensible next step.
- Implementation: derive six causal, input-only state proxies: fast/slow
  excitation (3/50 ms), fast/slow inhibition (3/50 ms), a 20 ms signed
  membrane proxy, and a 100 ms load/adaptation proxy. A zero-initialized
  learned 6-to-58 port injects these states before Mamba, after Mamba, or at
  both locations. Every candidate computes the same states/projection and has
  identical 69,783 parameters; tests verify E/I separation, timescale behavior,
  initial equivalence, and causality.
- Decision: compare inactive/pre/post/pre+post placement on the frozen seed-96
  schedule. Require a practically material core gain without significant
  regression, then audit projection/state use before replication.

### Iteration 097 - 2026-07-12

- Regime: structural state-construction/interface test; exact 69,783
  parameters, 120k presentations, batch 32, 3,750 updates, common
  initialization and cached frozen schedule, identical six-state/projection
  compute, and full held-out diagnostics.

| placement | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pre | +0.02226 significant | -0.00451 | -0.00914 | -0.00427 | -0.01594 significant | -0.01488 significant | reject tradeoff |
| post | -0.00456 | -0.00064 | -0.00029 | -0.00024 | -0.00923 | +0.00479 significant | reject response |
| pre+post | +0.00019 | -0.00523 | +0.00206 | +0.00243 | -0.00545 | -0.01440 significant | fresh-seed lead |

- Interpretation: current states help intervention response only when they
  shape Mamba state upstream; the direct post port alone harms response.
  Pre-only pays material voltage and 5 ms F1 costs, while reusing the same port
  after Mamba cancels the voltage tradeoff and preserves a material response
  gain. A held-out state audit confirms nontrivial, interpretable use: port RMS
  is 82.8% of branch-projected hidden RMS, the 20 ms membrane state has the
  largest projection-column norm (0.555), and correlates 0.514 with target
  voltage. Fast/slow E states also correlate 0.333/0.361; I states have the
  expected negative sign.
- Decision: reject pre and post alone. Replicate unchanged pre+post on a fresh
  seed; do not tune timescales or injection magnitude. A second balanced,
  practically material response result is required for the long gate.
- Compute: 183.31 seconds panel wall; 23.12-25.10 seconds candidate compute,
  with exact parameter/presentation/update/path matching.
- Artifacts: `artifacts/diagnostics/iteration-097-hybrid-current-ports/`,
  including `state-use-audit.json`.

### Iteration 098 premise - hybrid-current structural replication

- Pre+post six-state ports improve response materially without a significant
  core regression on seed 1 and show mechanistically coherent state use.
- Decision: fresh seed 20260726 compares only inactive and unchanged pre+post
  ports. Replication requires no significant core regression and at least
  0.01 mV response or voltage benefit, or an equivalently material F1 gain.

### Iteration 098 - 2026-07-12

- Regime: unchanged structural replication on a fresh seed; exact 69,783
  parameters, 120k presentations, batch 32, 3,750 updates, common fresh
  initialization/schedule, matched six-state compute, and diagnostics.
- Results: pre+post changes voltage SERA by +0.02464 mV significant, dynamics
  by -0.00018, response by -0.00029, F1 area by -0.00082, exact F1 by +0.00948
  significant, and 5 ms F1 by -0.00665.
- Interpretation: the material seed-1 response/dynamics gain disappears and a
  material voltage regression replaces it. Exact timing improves, but the
  declared primary F1-over-time and response axes do not. Mechanistic state use
  on seed 1 was real, yet it did not create a robust inductive-bias advantage.
- Decision: reject hybrid-current ports without timescale or placement tuning;
  queue the complete path for iteration-100 cleanup.
- Compute: 137.41 seconds panel wall; 22.85/24.44 seconds model compute with
  exact parameter/presentation/update/path matching.
- Artifacts: `artifacts/diagnostics/iteration-098-hybrid-current-replication/`.

### Iteration 099 premise - fused polarity-structured Mamba state groups

- The spec's standalone six-state BSSM feasibility gate fails: at batch 32 x
  500, median forward/backward step time is 0.42949 seconds versus 0.002468 for
  AxoMamba, a 174.0x ratio against the predeclared <2x requirement. It also has
  only 6,161 versus 69,377 parameters, so a 120k quality comparison would be
  both prohibitively slow and capacity-confounded.
- Structural alternative: preserve fused Mamba execution but partition its 58
  channels into two independent 29-channel state groups. State dimension 55
  plus four learned per-channel E/I port vectors yields exactly 69,377 total
  parameters, matching promoted AxoMamba. Compare ordinary ungrouped Mamba,
  grouped inactive, grouped signed-load, and grouped separate-E/I ports.
  Signed and separate ports start identically to grouped inactive through
  zero-initialized residual/modulation scales; tests verify budget and causality.
- Decision: run all four on the cached seed-98 schedule. A separate-E/I result
  must beat both ordinary Mamba and grouped controls materially; higher latent
  rank or biological semantics alone are not acceptance criteria.
- Artifact: `artifacts/diagnostics/iteration-099-bssm-feasibility/step-benchmark.json`.

### Iteration 099 - 2026-07-12

- Regime: exact-budget fused structural comparison; every model has exactly
  69,377 parameters, 120k presentations, batch 32, 3,750 updates, the same
  frozen schedule/seed/objective, and full diagnostics. Grouped candidates use
  2 x 29-channel Mamba groups with state dimension 55.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grouped inactive | +0.01001 significant | +0.00134 | -0.00966 | -0.00718 | -0.01163 significant | +0.00046 | reject grouping |
| grouped signed port | +0.00076 | +0.00067 | -0.01800 significant | -0.00565 | -0.02484 significant | -0.00151 | reject F1 |
| grouped separate E/I | +0.03421 significant | +0.00019 | -0.00738 | +0.00032 | -0.01133 | +0.00772 significant | reject |

- Interpretation: the generic factorization already harms voltage and event
  timing, and explicit polarity does not recover it. Separate E/I semantics
  are actively worse than the grouped control on voltage/response; signed
  semantics preserve SERA but strongly damage the F1 curve. The result agrees
  with earlier generic grouped-Mamba evidence: higher or partitioned state
  diversity is not enough when it disrupts cross-channel event computation.
- Decision: reject the full grouped/polarity family without state-size or port
  tuning. Queue its config, parameters, forward path, and tests for iteration
  100 cleanup. The standalone BSSM remains gated out by speed.
- Compute: 193.77 seconds panel wall; grouped candidates require 27.43-27.69
  seconds versus 22.74 for promoted Mamba despite exact parameter equality.
- Artifacts: `artifacts/diagnostics/iteration-099-polarity-grouped-mamba/` and
  `artifacts/diagnostics/iteration-099-bssm-feasibility/`.

### Iteration 100 premise - long near-spike gate, cleanup, and literature refresh

- No structural candidate from iterations 96-99 replicated. Near-spike weight
  0.002 is the only surviving material short-run lead: two tested seeds average
  about -0.01484 mV voltage, -0.00388 mV response, and favorable mean F1, but
  the effects are heterogeneous and require a 300k decision.
- Decision: train matched control and unchanged near-spike weight 0.002 for
  exactly 300k presentations/9,375 updates on fresh seed 20260727. Promote only
  for a practically material long-run gain with no significant core regression.
  Then remove failed I91-I99 executable branches, verify both repos/checkpoint,
  and complete the scheduled large adjacent-literature scan for the next
  structural phase.

### Iteration 100 - 2026-07-12

- Regime: fresh-seed 300k promotion gate; exact 69,377 parameters, batch 32,
  9,375 optimizer updates, common initialization and frozen balanced schedule,
  identical model compute, and full paired diagnostics.

| objective | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| near-spike 0.002, radius 5 | +0.00500 | -0.00349 significant | +0.01018 significant | -0.00041 | +0.01904 significant | -0.01663 significant | promote |

- Interpretation: the near-spike objective does not produce a practically
  meaningful voltage improvement at 300k, and no such claim is made. It does
  deliver a material intervention-response gain and coherent improvements in
  dynamics and tolerant spike timing without a significant core regression.
  That is sufficient under the declared multi-axis promotion rule. The result
  also closes scalar tuning: weight 0.002 is a retained training component,
  not a reason to spend further iterations polishing millivolt thousandths.
- Promotion: copied the candidate checkpoint to `checkpoints/axomamba.pt` and
  made near-spike weight 0.002/radius 5 part of the reusable default objective.
  Stable loading reports 69,377 parameters, all five morphology IDs, 300,000
  presentations, 9,375 updates, and seed 20260727.
- Cleanup: deleted rejected input-load, liquid-soma, hybrid-current, and
  grouped/polarity executable branches and their tests/config knobs. Historical
  manifests and diagnostics remain. Retired checkpoint keys are ignored
  explicitly for compatibility.
- Verification: AxoSim passes 88/88; AxoBench iteration evaluation passes
  24/24; the promoted checkpoint loads successfully after cleanup.
- Compute: 335.39 seconds panel wall; 58.58/55.15 seconds candidate model
  compute at 300k presentations. The difference is objective overhead, not
  unequal exposure or optimizer steps.
- Literature: completed `literature/iteration-100-large-scan.md`. The next
  phase prioritizes structural state placement and recurrence design over
  parameter tuning. Iteration 101 tests shared branch-local temporal state
  before aggregation, with exact-budget and compute-aware controls.
- Artifacts: `artifacts/diagnostics/iteration-100-long-near-spike-gate/` and
  `manifests/iteration-100-long-near-spike-gate.json`.

### Iteration 101 premise - temporal state before branch aggregation

- Premise: promoted AxoMamba mixes 118 nonlinear branch currents into 58 hidden
  channels before any recurrent state exists. If compartment histories matter,
  this ordering discards branch-local temporal information too early.
- Implementation: apply one scalar-width Mamba residual block with shared
  weights independently to all branch sequences before `branch_projection`.
  The inactive and active candidates both instantiate and execute the same
  path and have exactly 69,552 parameters; only the fixed structural residual
  connection differs (0 versus 1). Tests verify inactive equivalence,
  causality, branch-permutation equivariance, and local parameter gradients.
- Feasibility: at batch 32 x 500, median forward/backward step latency is
  0.01472/0.01479 seconds for inactive/active versus 0.00276 for promoted
  AxoMamba; peak allocated memory is 1.94 GB versus 0.27 GB. The 5.3x cost is
  acceptable for one bounded panel but raises the usefulness threshold.
- Decision: run both candidates for 120k presentations on fresh seed 20260728
  with the promoted objective and full diagnostics. Require no significant core
  regression and at least 0.01 mV voltage/response SERA, 0.01 F1, or an
  equivalently compelling robustness gain; low-1e-3 movement cannot advance.

### Iteration 101 - 2026-07-12

- Regime: literature-derived structural state-placement test; exact 69,552
  parameters, 120k presentations, batch 32, 3,750 updates, common fresh
  initialization/schedule, identical local/global recurrence compute, and full
  diagnostics.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scalar shared branch-local state | +0.00362 significant | -0.00014 | 0.00000 | -0.00113 | 0.00000 | +0.00113 significant | reject |

- Interpretation: branch-local recurrence is a costly practical no-op. It
  significantly regresses voltage and response, but both changes are below the
  materiality floor; F1 area and 5 ms F1 are numerically identical. The active
  model's training loss also differs from control by only 0.000006.
- State-use audit: the local path is not dead. Its learned residual scale moves
  from +0.1 to -0.107, and branch trajectories remain distinct (mean absolute
  inter-branch correlation 0.0069), but its correction is only 1.06% of branch
  input RMS. The scalar shared output behaves as a weak filter and does not
  support the full multicompartment premise.
- Decision: reject scalar branch-local state and do not tune its coupling or
  state dimension. One multidimensional local-state correction remains
  scientifically distinguishable, but the now-available official Mamba-3
  recurrence receives a feasibility audit first because it is stronger
  accepted evidence and avoids another hand-designed filter.
- Compute: 225.11 seconds panel wall; 69.13/68.41 seconds model compute, 5.3x
  promoted step latency, and 1.94 GB measured peak allocation.
- Artifacts: `artifacts/diagnostics/iteration-101-branch-local-mamba/`, including
  `local-state-use-audit.json`.

### Iteration 102 premise - efficient branch-specific memory bank

- Mamba-3 feasibility: the installed official 2.3.2.post1 implementation exposes
  Mamba-3, but SISO fails Triton compilation under both installed 3.3.1 and an
  isolated 3.4.0 runtime; recommended-width bf16 MIMO fails TileLang compilation
  on the RTX 5090. This is a kernel gate, not a quality result, and no local
  reimplementation will substitute for the official recurrence.
- Structural correction: replace I101's thousands of tiny Mamba launches with
  vectorized 5 ms and 50 ms causal branch memories. A learned 118 x 2 mixing
  table assigns fast/slow memory independently per branch before the unchanged
  branch projection/global Mamba. The table starts at zero, preserving the
  promoted function exactly.
- Controls: inactive and active candidates both compute both filters and have
  exactly 69,613 parameters. Median batch-32 x 500 step latency is
  0.00333/0.00335 seconds with 287 MB peak memory, about 1.21x promoted latency.
  Tests verify initial equivalence, causality, branch-specific weights, and
  gradients.
- Decision: run a fresh-seed 120k panel. Require a material core gain with no
  significant regression; do not tune taus or memory scale if the premise
  fails.

### Iteration 102 - 2026-07-12

- Regime: structural local-memory test; exact 69,613 parameters, 120k
  presentations, batch 32, 3,750 updates, common initialization/schedule,
  identical filter compute, and full diagnostics.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| branch-specific 5/50 ms memory | -0.00853 significant | -0.00038 | +0.00428 | +0.00259 | +0.00254 | -0.00756 significant | unchanged replication |

- Interpretation: all six core point estimates improve, but no individual
  effect clears the ordinary materiality threshold. This is promising jointly,
  not a positive result. Unlike I101, the mechanism is efficient and strongly
  used: learned fast/slow weight RMS is 0.112/0.066, 78.8% of branches are
  fast-dominant, and the two branch-weight columns have cosine 0.593.
- Decision: run exactly one unchanged fresh-seed replication. A reversal closes
  the family; replication requires a material multi-axis mean or one effect
  clearing 0.01 without a significant core regression. Do not tune taus,
  kernel, scale, or objective.
- Compute: 140.94 seconds panel wall; 25.57/24.84 seconds model compute and
  about 1.21x promoted step latency.
- Artifacts: `artifacts/diagnostics/iteration-102-branch-memory-bank/`, including
  `branch-memory-use-audit.json`.

### Iteration 103 premise - unchanged branch-memory replication

- Premise: the efficient 5/50 ms branch-memory bank moves all six core metrics
  favorably on seed 1 and learns heterogeneous branch-specific mixtures, but
  its effects are individually sub-material.
- Decision: fresh seed 20260730 repeats inactive versus active with no changes.
  Advance only if the joint profile survives and becomes practically material;
  otherwise reject without scalar refinement.

### Iteration 103 - 2026-07-12

- Regime: unchanged structural replication; exact 69,613 parameters, fresh seed
  20260730, 120k presentations, 3,750 updates, matched compute, and diagnostics.
- Results: voltage changes by -0.00502 mV significant, dynamics by +0.00157,
  response by +0.00396 mV significant, F1 area by -0.00170, exact F1 by
  +0.00136, and 5 ms F1 by -0.00041.
- Interpretation: the uniformly favorable seed-1 profile reverses on response,
  dynamics, F1 area, and 5 ms F1; every effect remains practically negligible.
  The learned branch-memory mechanism is real but does not yield a robust
  surrogate advantage.
- Decision: reject branch memory without tau/kernel/scale tuning and queue all
  I101-I103 local-state code for I110 cleanup. Iteration 104 performs the
  predeclared high-order observability audit before changing architecture.
- Compute: 140.48 seconds panel wall; 25.30/24.62 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-103-branch-memory-replication/`.

### Iteration 104 premise - conditional delay observability

- Premise: a first-order hidden state may omit multi-lag input information that
  remains predictive of voltage/dynamics residuals. Explicit delay coordinates
  are justified only if that information survives conditioning on current
  AxoMamba hidden state and current branch currents.
- Audit: fit trace-disjoint ridge probes on all five morphologies using hidden,
  hidden+current branch currents, and hidden+current plus branch delays at
  1/2/5/10/25/50/100 ms. Train on 80 traces and test on 40, sampling 512 scored
  steps per trace. The premise requires at least 1% held-out residual-RMSE
  reduction and nonadverse voltage movement in at least four morphologies.
- Decision: build no high-order architecture unless this audit passes.

### Iteration 104 - 2026-07-12

- Regime: scientific premise audit on the promoted 300k checkpoint, with
  trace-disjoint train/test probes balanced across all five morphologies.
- Results: delayed branch coordinates worsen voltage residual RMSE by 1.00%
  and dynamics residual RMSE by 1.15% relative to hidden+current branches.
  Voltage RMSE worsens in all five morphologies. Hidden-state residual R2 is
  already low (0.0229 voltage, 0.0013 dynamics), and delays reduce it further.
- Decision: the high-order delay-coordinate premise fails its predeclared 1%
  reduction/four-of-five gate. Build no delay-state architecture and do not
  tune lags or ridge strength. Move to an official recurrence replacement.
- Artifact: `artifacts/diagnostics/iteration-104-delay-observability-audit.json`.

### Iteration 105 premise - exact-budget official Mamba-2 trunk

- Premise: Mamba-2's structured state-space dual recurrence changes state
  interaction more substantially than the failed filters/ports while remaining
  available through the official `mamba-ssm` package. It has not been tested in
  the loop.
- Feasibility: the closest conservative geometry is model width 64, state 32,
  head 32, one group, two layers. It has 69,001 active parameters and a measured
  0.00554 second batch-32 x 500 step. Adding 376 explicitly unused budget
  parameters yields exactly 69,377, matching promoted Mamba-1 without giving
  Mamba-2 an active-capacity advantage.
- Decision: compare unchanged promoted Mamba-1 against exact-budget Mamba-2 on
  fresh seed 20260732, 120k presentations, and full diagnostics. Require a
  material core gain with no significant regression; 2x latency raises the
  usefulness bar.

### Iteration 105 - 2026-07-12

- Regime: exact-budget official recurrence replacement; both models have
  69,377 parameters, 120k presentations, batch 32, 3,750 updates, common data
  schedule/objective, and full diagnostics.
- Results: Mamba-2 changes voltage by +0.05950 mV significant, dynamics by
  -0.00020, response by +0.00117, F1 area by -0.02685 significant, exact F1 by
  -0.01151 significant, and 5 ms F1 by -0.03047 significant.
- Interpretation: structured state-space duality is substantially worse for
  this small biological-surrogate geometry. It harms both voltage and all event
  metrics, offers no intervention benefit, and takes 33.17 versus 23.52 seconds
  model compute after warmup.
- Decision: reject Mamba-2 without geometry or objective tuning; queue inert
  budget-padding support for I110 cleanup if no later exact-budget experiment
  needs it.
- Artifacts: `artifacts/diagnostics/iteration-105-mamba2-trunk/`.

### Iteration 106 premise - exact-budget Mamba-GRU hybrid

- Premise: an older same-data RNN run achieved the best voltage RMSE in its
  comparison, while Mamba contributes efficient selective memory. Replace the
  second Mamba block with a standard GRU residual tail and allocate the freed
  parameters to state 54 in the remaining Mamba block.
- Controls: both architectures have exactly 69,377 parameters with unchanged
  width/routing/heads/morphology conditioning. A batch-32 x 500 benchmark gives
  0.00320 seconds and 293 MB for the hybrid versus 0.00374 seconds and 267 MB
  for control, a 14% latency improvement. Tests verify causality and GRU
  gradients.
- Decision: run one fresh-seed 120k panel under the promoted objective. Require
  no significant core regression and a material quality gain, or near-parity
  with a robust efficiency win.

### Iteration 106 - 2026-07-12

- Regime: exact-budget hybrid recurrence test; both models have 69,377
  parameters, 120k presentations, batch 32, 3,750 updates, common fresh
  schedule/objective, and full diagnostics.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| state-54 Mamba + GRU tail | -0.03944 significant | -0.00247 | +0.00827 | +0.01080 | +0.00816 | -0.02058 significant | unchanged replication |

- Interpretation: every core point estimate improves, with practically
  material voltage/response gains and exact F1 above 0.01. The GRU is not an
  inert parameter substitution: its learned update is 29.3% of incoming hidden
  RMS and residual scales span 0.078-0.334. Model compute is 22.30 versus 23.19
  seconds, a 3.8% measured efficiency gain.
- Decision: replicate the unchanged exact-budget architecture on fresh seed
  20260734. No state-size, ordering, or objective tuning before replication.
- Compute: 136.09 seconds panel wall with identical presentation/update count.
- Artifacts: `artifacts/diagnostics/iteration-106-mamba-gru-hybrid/`, including
  `gru-tail-use-audit.json`.

### Iteration 107 premise - unchanged Mamba-GRU replication

- Premise: the hybrid materially improves voltage and intervention response,
  raises all event metrics, and is slightly faster on seed 1.
- Decision: fresh seed 20260734 repeats the exact state-54 Mamba + GRU tail.
  Require no significant core regression and a material voltage/response/F1
  gain or coherent two-seed profile before any long-gate planning.

### Iteration 107 - 2026-07-12

- Regime: unchanged exact-budget replication on fresh seed 20260734; 120k
  presentations, 3,750 updates, matched compute, and diagnostics.
- Results: voltage changes by +0.01213 mV significant, dynamics by -0.00339,
  response by -0.00525 mV significant, F1 area by -0.02975 significant, exact
  F1 by -0.00352, and 5 ms F1 by -0.03618 significant.
- Interpretation: the seed-1 all-axis gain does not replicate. A small
  response/dynamics benefit survives, but event fidelity collapses and voltage
  materially regresses. The speed advantage is real (22.34 versus 23.24 seconds
  model compute), but cannot compensate for the core failures.
- Decision: reject Mamba-then-GRU as a promotion candidate. Test one structural
  order correction at I108 because the failure specifically appears after the
  GRU tail: place GRU first and retain Mamba as the final event-selective block.
  Include the failed tail on the same fresh schedule for a direct order control.
- Artifacts: `artifacts/diagnostics/iteration-107-mamba-gru-replication/`.

### Iteration 108 premise - hybrid recurrence ordering

- Premise: a final GRU can smooth Mamba's event-selective representation; a
  GRU-first/Mamba-last hybrid may retain nonlinear recurrent state while letting
  Mamba perform the final selective event computation.
- Controls: promoted two-Mamba, Mamba-then-GRU, and GRU-then-Mamba each have
  exactly 69,377 parameters and the same width/routing/heads/objective. Both
  hybrids use one state-54 Mamba and one GRU; only order differs.
- Decision: run all three on fresh seed 20260735. Close the full hybrid family
  unless GRU-first yields a material balanced gain without significant core
  regression; no state-size or blend tuning follows a failure.

### Iteration 108 - 2026-07-12

- Regime: exact-budget three-way recurrence-order comparison; all candidates
  have 69,377 parameters, 120k presentations, 3,750 updates, common fresh
  schedule/objective, and full diagnostics.

| order | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mamba then GRU | -0.05584 significant | +0.00044 | -0.03800 significant | -0.00420 | -0.05448 significant | -0.01952 significant | reject F1 |
| GRU then Mamba | +0.02970 significant | -0.00371 | -0.01342 | +0.00285 | -0.01250 | +0.01360 significant | reject |

- Interpretation: ordering changes the tradeoff but cannot balance it.
  Mamba-last does not preserve response or voltage, while GRU-last again buys
  SERA by collapsing tolerant F1. Both hybrids remain slightly faster, but the
  core regressions are material.
- Decision: reject the complete hybrid family without state-size, blend, gate,
  or objective tuning; queue all GRU-tail code for I110 cleanup.
- Compute: 157.81 seconds panel wall; 22.20/22.38 seconds hybrid compute versus
  23.12 seconds control.
- Artifacts: `artifacts/diagnostics/iteration-108-hybrid-ordering/`.

### Iteration 109 premise - cooperative bilinear branch integration

- Premise: multicompartment work emphasizes nonlinear cooperation between
  dendritic sources, while AxoMamba currently applies independent branch tanh
  followed by linear aggregation. A low-rank multiplicative branch interaction
  adds cooperative integration before temporal state.
- Controls: rank-4 interaction costs 1,176 active parameters. Reducing both
  Mamba states from 25 to 23 frees 1,392; 216 explicitly inert parameters make
  promoted state-25, state-23 inactive, and state-23 active models exactly
  69,377 parameters. Active/inactive interaction compute is identical; measured
  overhead is 4.3%. The output projection starts at zero.
- Decision: run all three on fresh seed 20260736. The active model must beat
  both promoted and inactive controls materially with no significant core
  regression; otherwise close the mechanism before I110.

### Iteration 109 - 2026-07-12

- Regime: exact-budget three-way structural test; all candidates have 69,377
  parameters, 120k presentations, 3,750 updates, common fresh schedule/objective,
  and full diagnostics.

| structure | delta voltage | delta dynamics | delta F1 area | delta exact | delta 5 ms | delta response | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| state-23 inactive | -0.01248 significant | -0.00052 | +0.01479 significant | -0.00240 | +0.01753 significant | +0.01579 significant | reject response |
| state-23 bilinear | -0.01643 significant | -0.00121 | +0.02501 significant | +0.00413 | +0.02689 significant | +0.01499 significant | long closure only |

- Interpretation: state reallocation creates a material response tradeoff, but
  the cooperative path has genuine incremental value over inactive: about
  -0.00395 mV voltage, +0.01022 F1 area, +0.00652 exact F1, +0.00936 at 5 ms,
  and slightly better response. It does not pass the short-run promotion gate.
- Decision: use I110's mandatory 300k signal to close the active architecture
  against promoted state-25. Promotion is allowed only if the response
  regression disappears while the material F1/voltage gain persists. No rank,
  state-size, scale, or objective tuning.
- Compute: 159.31 seconds panel wall; active/inactive model compute 22.65/22.82
  versus 23.65 seconds control.
- Artifacts: `artifacts/diagnostics/iteration-109-bilinear-branch-integration/`.

### Iteration 110 premise - long bilinear closure, cleanup, and literature refresh

- Decision: train promoted state-25 and unchanged rank-4 state-23 bilinear for
  300k presentations on fresh seed 20260737. Promote only if no significant
  core regression remains and at least one material gain persists.
- Maintenance: after diagnostics, remove all failed I101-I109 executable paths,
  tests, config knobs, the completed delay-audit helper, and isolated Mamba-3
  feasibility environment. Retain manifests, diagnostics, literature, and
  retired-key checkpoint compatibility; run full AxoSim/AxoBench verification.
- Literature: complete the scheduled I110 adjacent-work refresh and use it to
  select a new structural phase rather than returning to coefficient tuning.

### Iteration 110 - 2026-07-12

- Regime: mandatory 300k structural closure on fresh seed 20260737. Promoted
  state-25 and unchanged rank-4 state-23 bilinear models each receive 300k
  presentations, 9,375 optimizer updates, identical objective/data order, and
  exactly 69,377 training parameters.
- Results: bilinear changes voltage by -0.04746 mV significant, dynamics by
  +0.00035 uncertain, response by -0.00591 mV significant, F1 area by +0.00930,
  exact F1 by +0.00496, and 5 ms F1 by +0.00919. No core metric regresses
  significantly.
- Interpretation: cooperative branch integration produces a material voltage
  gain and survives the long fresh-seed gate. The smaller favorable response
  and event effects are supporting evidence, not inflated into separate claims.
- Decision: promote state-23/rank-4 bilinear AxoMamba. Strip the 216 inert
  comparison parameters after verifying bitwise GPU output equality; deployed
  functional count is 69,161 and metadata retains the exact-budget training
  count of 69,377.
- Maintenance: removed failed I101-I109 branch-local, branch-memory, GRU-tail,
  and padding paths, associated tests, the completed delay audit, and the
  isolated Mamba-3 environment. Historical diagnostics and manifests remain.
- Verification: AxoSim 89/89, AxoBench evaluator 24/24, clean diff check, and a
  successful load of the promoted five-morphology 300k checkpoint.
- Literature and next decision: `literature/iteration-110-scan.md` ranks a
  temporal low-rank Volterra interaction first. I111 will compare it against an
  exact-parameter, matched-compute instantaneous control, with no timescale or
  coupling sweep.
- Artifacts: `artifacts/diagnostics/iteration-110-long-bilinear-gate/`,
  `manifests/iteration-110-long-bilinear-gate.json`, and
  `literature/iteration-110-scan.md`.

### Iteration 111 premise - temporal low-rank branch interaction

- Premise: the promoted instantaneous rank-4 interaction captures simultaneous
  branch cooperation, but not interactions between present activity and causal
  branch context. A low-rank temporal Volterra factor can represent that class
  without an additive lag stack.
- Structure: retain `silu(U b_t)` and replace the second factor with four
  rank-aligned learned causal exponential states initialized at 5/20/50/100 ms
  before multiplication. The interaction remains upstream of both Mamba blocks.
- Controls: active and instantaneous compute-control models each have 69,165
  parameters, execute the same causal filters in forward/backward, allocate 278
  MB, and benchmark at 3.220/3.234 ms per batch-32 training step. The original
  69,161-parameter promoted model is a secondary implementation reference only.
- Decision: run all three for 120k presentations on frozen fresh seed 20260738.
  Require a material core gain over the exact control with no significant core
  regression; no timescale, kernel, or scale sweep follows a clean failure.

### Iteration 111 - 2026-07-12

- Regime: structural temporal-interaction test; active and compute-control each
  have 69,165 parameters, 120k presentations, batch 32, 3,750 updates, common
  fresh schedule/objective, and full diagnostics. The original 69,161-parameter
  model is output-identical to the compute control to about 1e-8.
- Results: temporal interaction changes voltage by +0.00140 mV uncertain,
  dynamics by -0.00100 uncertain, response by +0.00216 mV significant, F1 area
  by +0.01594 significant, exact F1 by +0.00667 uncertain, and 5 ms F1 by
  +0.02342 significant.
- Interpretation: the temporal factor materially improves tolerant event
  fidelity while voltage/dynamics are practically unchanged. The statistically
  detectable response loss is only 0.00216 mV, below the predeclared 0.01 mV
  practical threshold; treating that alone as a veto would repeat the
  significance-without-materiality error the strategy forbids. Count error
  improves from 0.608 to 0.510, default-threshold count remains inside its
  envelope, and refractory violations rise slightly from 0.263 to 0.268.
- Mechanism audit: learned temporal constants move from 5/20/50/100 ms to
  4.7/11.6/30.2/67.2 ms and the interaction output weight RMS grows to 0.0968,
  showing nontrivial use rather than an inactive near-control.
- Decision: unchanged fresh-seed replication at I112. Do not tune timescales,
  kernel, rank, or objective. Require replicated material F1 improvement with
  no practically material SERA regression before planning a long gate.
- Compute: panel wall 159.13 seconds; temporal model compute 22.87 seconds,
  exact control 24.69 seconds. Independent repeated-step benchmark was
  3.220/3.234 ms with equal 278 MB allocation.
- Artifacts: `artifacts/diagnostics/iteration-111-temporal-bilinear-volterra/`
  and `manifests/iteration-111-temporal-bilinear-volterra.json`.

### Iteration 112 premise - unchanged temporal-interaction replication

- Decision: repeat exact instantaneous control and temporal-factor model on
  fresh seed 20260739, each with 69,165 parameters, 120k presentations, 3,750
  updates, and the same objective. Close the temporal family on a material F1
  reversal or material SERA loss; otherwise use the two-seed profile to decide
  whether a later long gate is warranted.

### Iteration 112 - 2026-07-12

- Regime: unchanged exact-control replication on fresh seed 20260739; both
  models have 69,165 parameters, 120k presentations, batch 32, 3,750 updates,
  common schedule/objective, and full diagnostics.
- Results: temporal interaction changes voltage by +0.00328 mV uncertain,
  dynamics by +0.00285 significant, response by -0.00312 mV significant, F1
  area by -0.01182 significant, exact F1 by -0.00009, and 5 ms F1 by -0.02556
  significant.
- Interpretation: the material tolerant-F1 improvement reverses on replication.
  SERA changes remain practically small, but the F1-area and 5 ms failures are
  both material and significant. Lower mean training loss on both seeds did not
  translate into stable held-out event fidelity.
- Decision: reject and close temporal bilinear filtering without timescale,
  kernel, rank, or objective tuning. Retain its code only until scheduled I120
  cleanup; move to the literature-ranked interaction-as-modulation test.
- Compute: panel wall 138.36 seconds; active/control model compute 23.35/24.95
  seconds under identical presentations and updates.
- Artifacts: `artifacts/diagnostics/iteration-112-temporal-bilinear-replication/`
  and `manifests/iteration-112-temporal-bilinear-replication.json`.

### Iteration 113 premise - branch interaction as hidden modulation

- Premise: cooperative dendritic features may select or gate the global
  soma-like computation rather than contribute an additive current. Replace
  the promoted update `h + p` with `h + h * tanh(p)` while preserving the same
  rank-4 branch factors and placement before Mamba.
- Controls: additive and modulation models each have exactly 69,161 parameters,
  execute both projected and modulated paths in forward/backward, allocate the
  same 287 MB, and alternate-order benchmarking gives 3.07/2.88 ms mean steps.
  Both are output-identical at zero-initialized interaction projection.
- Decision: run both on fresh seed 20260740 for 120k presentations and full
  diagnostics. Require a material core gain with no material core regression;
  do not sweep modulation scales or nonlinearities after a clean failure.

### Iteration 113 - 2026-07-12

- Regime: exact-parameter structural placement test; additive and modulation
  models each have 69,161 parameters, 120k presentations, batch 32, 3,750
  updates, common fresh schedule/objective, and full diagnostics.
- Results: modulation changes voltage by -0.00143 mV, dynamics by -0.00060,
  response by -0.00191 mV significant, F1 area by -0.00040, exact F1 by
  +0.00016, and 5 ms F1 by +0.00136. Every movement is sub-material.
- Interpretation: interaction-as-gate is a clean practical null. Its slightly
  lower training loss and statistically detectable 0.0019 mV response movement
  do not meet the research threshold.
- Decision: reject without replication, scale tuning, or activation variants.
  Retain the mode only until scheduled I120 cleanup.
- Compute: panel wall 137.62 seconds; modulation/additive model compute
  22.67/24.11 seconds. Alternating repeated-step benchmarking finds no stable
  active-path slowdown.
- Artifacts: `artifacts/diagnostics/iteration-113-bilinear-hidden-modulation/`
  and `manifests/iteration-113-bilinear-hidden-modulation.json`.

### Infrastructure - exact fast frozen schedules

- Finding: each fresh seed spent about 100 seconds decompressing randomly
  selected shards only to read the input time dimension before writing a 408 KB
  index schedule.
- Change: read the NPY header inside each NPZ (or mmap the split NPY input) to
  obtain shape without materializing payload arrays. Schedule sampling logic and
  RNG call order are unchanged.
- Verification: rebuilding the real 120k seed-20260740 schedule takes 0.102
  seconds and exactly matches all old shard indices, sample indices, starts,
  and metadata. New tests prevent full-shard loads during schedule creation;
  the full suite passes 95/95.

### Iteration 114 premise - alternate branch forcing and temporal evolution

- Premise: state-space neural-operator work separates external forcing from
  homogeneous temporal evolution. AxoMamba currently presents routed branch
  forcing once, before two Mamba blocks. Reinjecting the full learned
  branch/bilinear/morphology representation between blocks lets the second
  transition correct dynamics using spatial forcing that the first block may
  have compressed.
- Controls: active and inactive models each have exactly 69,219 parameters and
  execute the same forcing multiplication/addition. The per-channel reinjection
  gain starts at zero, so both are initially output-identical. Both allocate 288
  MB; median repeated training steps are approximately 3.12/3.07 ms.
- Negative-evidence boundary: this is not the failed morphology-only
  reinjection or a hand-built aggregate-current port; it reinjects the complete
  learned branch operator once between temporal transitions.
- Decision: run both for 120k presentations on fresh seed 20260741. Require a
  material core gain without material regression; no reinjection scale or
  placement sweep follows a clean failure.

### Iteration 114 - 2026-07-12

- Regime: exact-parameter branch-forcing alternation test; active and inactive
  models each have 69,219 parameters, 120k presentations, batch 32, 3,750
  updates, common fresh schedule/objective, and full diagnostics.
- Results: forcing reinjection changes voltage by -0.00472 mV significant,
  dynamics by +0.00009, response by +0.00302 mV significant, F1 area by
  +0.00449 significant, exact F1 by -0.00039, and 5 ms F1 by +0.00555.
- Interpretation: all effects are below practical materiality and form a small
  voltage/F1 versus response tradeoff. The mechanism is not dormant: all 58
  gains move, with RMS 0.0192 and maximum magnitude 0.0546.
- Decision: reject without replication, scale tuning, or alternate placement.
  Retain the path only until scheduled I120 cleanup.
- Compute: panel wall 137.13 seconds; active/control model compute 22.85/24.34
  seconds. The fast header schedule removed about 100 seconds of setup latency.
- Artifacts: `artifacts/diagnostics/iteration-114-branch-forcing-alternation/`
  and `manifests/iteration-114-branch-forcing-alternation.json`.

### Iteration 115 premise - interaction-conditioned state updates

- Premise: the promoted branch interaction may be useful not only as an
  additive input feature but as a selector of temporal state evolution. Use
  its per-time, per-channel projection to multiply each Mamba residual update
  by `1 + tanh(p)`, while retaining the promoted additive interaction.
- Controls: active and scale-zero control have exactly 69,161 parameters,
  execute the same projection/gate operations, start output-identically because
  the interaction output is zero initialized, and allocate the same 302 MB.
  Median repeated steps are 3.00/2.93 ms.
- Decision: run both on fresh seed 20260742 for 120k presentations and full
  diagnostics. Require a material core gain without material regression; no
  gate-scale or block-placement tuning follows a clean failure.

### Iteration 115 - 2026-07-12

- Regime: exact-parameter interaction-conditioned state-update test; active and
  scale-zero models each have 69,161 parameters, 120k presentations, batch 32,
  3,750 updates, common fresh schedule/objective, and full diagnostics.
- Results: state gating changes voltage by +0.00022 mV, dynamics by +0.00011,
  response by -0.00011 mV, F1 area by -0.00233, exact F1 by -0.00106, and
  5 ms F1 by -0.00334 significant.
- Interpretation: SERA is exactly tied for practical purposes and every event
  metric moves adversely. This is a clean null-to-small-regression, not a
  candidate for replication.
- Decision: reject without gate-scale, per-block, or placement tuning. Retain
  the path only until I120 cleanup.
- Compute: panel wall 137.39 seconds; active/control model compute 22.81/24.33
  seconds under identical presentations and updates.
- Artifacts: `artifacts/diagnostics/iteration-115-interaction-state-gating/`
  and `manifests/iteration-115-interaction-state-gating.json`.

### Iteration 116 premise - low-rank dynamic branch routing

- Premise: the promoted rank-4 term emits only 58 aggregate interaction
  features. Dendritic branch gating instead lets global branch context change
  the gain of each of 118 branches before projection, preserving branch
  identity for the existing projection and bilinear interaction.
- Structure: rank-4 `U`/`V` routing computes
  `b * (1 + tanh(V silu(U b)))` before branch projection. The output map starts
  at zero and the promoted bilinear path remains unchanged.
- Controls: active and scale-zero control each have exactly 70,105 parameters,
  execute the same route, allocate the same 311 MB, and are initially
  output-identical. Median repeated steps are 3.18/3.17 ms.
- Decision: run both on fresh seed 20260743 for 120k presentations and full
  diagnostics. Require a material core gain without material regression; do
  not sweep rank or gate scale after a clean failure.

### Iteration 116 revised premise - 300k teacher distillation

- Direction change: explicit latent regularization/distillation had not been
  tested. It takes priority over dynamic branch routing. More broadly, the
  research target is now a substantial fraction of the roughly 20 mV base
  error, so sub-0.01 mV changes remain mechanistic evidence at most, not the
  desired endpoint.
- Teacher: use the promoted iteration-110 300k AxoMamba. No saved larger model
  has stronger validated diagnostics, so older checkpoints are not treated as
  teachers merely because they are larger.
- Structure: every 120k student carries the same train-only 58x58 affine
  adapter, initialized to identity, which aligns the student's final normalized
  latent to the frozen teacher latent. Output distillation matches teacher spike
  probabilities and voltage. The adapter is inference-inert and can be stripped
  if the method succeeds.
- Scale audit: over eight real batches at fresh initialization, task loss is
  0.8353, latent mismatch 2.1398, and output mismatch 0.4048. Weights 0.10 and
  0.50 each contribute about 25% of initial task loss; the combined arm is an
  aggressive 50% teacher constraint rather than a tiny regularizer.
- Controls: compute control, output-only, latent-only, and combined students
  each have exactly 72,583 training parameters, run one shared frozen-teacher
  forward per batch, and compute both distillation losses. Only weights differ.
- Decision: run all four at 120k on seed 20260743 with full diagnostics. Judge
  sample efficiency against the compute control and also report the gap to the
  300k teacher. No coefficient sweep follows a clean failure.

### Iteration 116 - 2026-07-12

- Regime: train-only representation/output distillation from the promoted I110
  300k teacher into four 120k students on seed 20260743. Every student has
  72,583 training parameters, 3,750 updates, the same identity-initialized
  58x58 latent adapter, shared teacher forward, and computes both losses; only
  the selected loss weights differ.
- Results versus the zero-weight compute control: output-only changes voltage
  by -0.01261 mV and response by -0.00339 mV, but loses 0.01402 F1 area and
  0.02004 at 5 ms. Latent-only regresses voltage by +0.04496 mV, response by
  +0.01046 mV, F1 area by -0.01749, and 5 ms F1 by -0.02125. Combined
  latent+output changes voltage by +0.01288 mV and loses 0.02429 F1 area and
  0.03183 at 5 ms. Dynamics changes are all below 0.0021 mV/ms.
- Scale interpretation: the best voltage movement is only 0.073% of the
  17.3835 mV control error and comes with material event-fidelity losses. The
  control itself remains 0.0421 mV behind the 300k teacher's 17.3414 mV,
  confirming that more training helps modestly but teacher matching does not
  transfer that advantage at 120k.
- Mechanism audit: the final latent mismatch falls from about 2.14 initially to
  0.0197-0.0210 in the latent-weighted arms, so the failure is not an inactive
  adapter or inability to fit the teacher representation. Forcing a young
  student's normalized latent toward the mature teacher harms the task-relevant
  optimization path.
- Decision: reject latent, output, and combined distillation without a weight,
  layer, or adapter sweep. Retain the general possibility of distillation only
  for a future demonstrably stronger teacher; move immediately to a
  high-capacity LaMO-derived latent spatial operator.
- Compute: four-student panel wall 189.67 seconds; candidate model compute
  23.78-24.76 seconds each plus 4.16 seconds total shared teacher compute.
- Artifacts: `artifacts/diagnostics/iteration-116-latent-output-distillation/`
  and `manifests/iteration-116-latent-output-distillation.json`.

### Iteration 117 premise - latent branch Mamba operator

- Premise: the promoted model collapses 118 branches to one 58-dimensional
  stream before temporal evolution. LaMO's learned kernel slicing suggests
  preserving several latent spatial states, evolving each state with a shared
  Mamba operator, and decoding them only after temporal processing. This gives
  distinct branch mixtures distinct causal memory while avoiding the failed
  118-way scalar local-Mamba cost.
- Structural panel: test a residual latent operator and a replacement latent
  operator, plus the promoted model and a genuine widened/deeper global-Mamba
  capacity control. Use learned branch-to-token kernel weights, multiple latent
  tokens, shared causal temporal Mamba, and learned token decoding.
- Validity gate: match the structural candidates and capacity control as
  closely as discrete width/state choices allow, report exact trainable
  parameters, measured step latency, peak allocation, and presentations. Do
  not call an unmatched axis causal. Require a gain measured in meaningful
  hundredths of a mV/F1 at minimum; the research target is a visible fraction
  of the approximately 20 mV base error, not another 1e-3 movement.
- Concrete controls: the six-token, width-32 residual operator and a global
  three-layer width-59/state-18 Mamba each have exactly 96,706 trainable
  parameters. Their measured batch-32 x 500 training steps are 6.05 and 3.37
  ms, so parameter equality alone leaves a compute advantage. A second global
  six-layer width-64/state-23 control costs 6.12 ms, within 1.2% of the latent
  operator, and has 222,881 parameters, making it a deliberately stronger
  compute-matched control. The promoted 69,161-parameter/2.69 ms model remains
  the deployment baseline. Replacement has 89,746 parameters and 5.96 ms; it
  is interpreted as a forced-factorization probe, not an exact-budget contrast.
- Resource audit: residual/replacement peak allocations are 535.5/535.4 MB;
  exact-parameter global is 307.1 MB and promoted is 273.4 MB. The compute-
  matched global is approximately 458 MB. All candidates use exactly 120k
  presentations, batch 32, 3,750 updates, one frozen seed-20260744 schedule,
  and the promoted training objective.

### Iteration 117 - 2026-07-12

- Regime: literature-derived high-capacity spatial-state factorization plus
  orthogonal parameter and compute controls. All five models use seed 20260744,
  120k presentations, batch 32, 3,750 updates, a shared frozen schedule, and
  full diagnostics.
- Exact-parameter global result: the 96,706-parameter three-layer global Mamba
  improves voltage by 0.01650 mV, response by 0.00888 mV, F1 area by 0.02243,
  and 5 ms F1 by 0.03851; all four paired intervals exclude zero. Its measured
  model compute is 25.88 seconds versus 24.98 for the promoted model, only 3.6%
  more in the real panel.
- Compute-matched global result: the 222,881-parameter six-layer global Mamba
  improves voltage by 0.03469 mV, response by 0.01226 mV, F1 area by 0.04528,
  and 5 ms F1 by 0.06133, all significant. Its model compute is 35.23 seconds,
  essentially identical to the residual operator's 35.31 seconds.
- Latent operator result: the 96,706-parameter residual operator changes
  voltage by +0.00804 mV significantly, response by +0.00118 mV, dynamics by
  +0.00323 mV/ms, F1 area by +0.02315, and 5 ms F1 by +0.03512. It is dominated
  by its exact-parameter global control: similar F1, but 0.02454 mV worse
  voltage and 0.01006 mV worse response. The 89,746-parameter replacement
  operator regresses voltage by 0.04676 mV despite similar F1 gains.
- Mechanism audit: the residual path is strongly used, with latent-update RMS
  0.1694 versus base-projection RMS 0.2381 (71.1%). Its assignment temperature
  falls from 0.5 to 0.415, yet assignments collapse toward uniform: normalized
  entropy is 0.9958, mean maximum probability is 0.196 versus 0.167 uniform,
  and token masses are all 19.46-19.93. Replacement entropy is 0.9979. The
  failure is redundant spatial states rather than an inactive residual.
- Interpretation: learned latent slicing does not beat ordinary global depth
  under exact capacity and costs 36% more measured model compute than that
  control. Global depth/state reallocation is the balanced lead, and larger
  depth/width is the strongest compute-matched lead. Their voltage gains are
  still only 0.095% and 0.199% of base error, so neither meets the aggressive
  fraction-of-error endpoint yet.
- Decision: reject replacement and unconstrained learned slicing. Replicate the
  two global leads unchanged on a fresh seed. In the same panel, test one
  exact-parameter/compute fixed-sparse slice operator against an unchanged
  learned-slice replication; this directly tests whether preventing the
  observed assignment collapse rescues the literature premise.
- Compute: five-model panel wall 243.18 seconds; individual model compute
  24.98, 25.88, 35.23, 35.31, and 34.91 seconds for promoted, exact-global,
  large-global, residual-latent, and replacement-latent respectively.
- Artifacts: `artifacts/diagnostics/iteration-117-latent-branch-operator/` and
  `manifests/iteration-117-latent-branch-operator.json`.

### Iteration 118 premise - global-depth replication and sparse latent slices

- Replication: rerun promoted, exact-parameter three-layer global, and
  compute-matched six-layer global configurations unchanged on seed 20260745.
  A core reversal rejects the corresponding lead; a replicated balanced gain
  advances it without width/state tuning.
- Sparse-slice premise: the learned operator failed while collapsing six token
  assignments to almost uniform mixtures. Use balanced fixed sparse branch
  partitions while retaining and executing the same assignment logits,
  temperature, kernel, shared latent Mamba, decoder, parameters, and compute.
  Compare against unchanged learned slicing on the same fresh seed.
- Acceptance: sparse slicing must materially beat learned slicing and the
  exact-parameter global model on core metrics without a new failure trigger.
  Merely preserving token identity is a mechanism check, not sufficient for
  promotion.

### Iteration 118 - 2026-07-12

- Regime: unchanged fresh-seed replication of both global-depth leads plus an
  exact-parameter/compute learned-versus-fixed latent assignment test. All
  models use seed 20260745, 120k presentations, batch 32, 3,750 updates, one
  shared frozen schedule, and full diagnostics.
- Exact-parameter global replication: voltage reverses from -0.01650 to
  +0.04222 mV significant and response from -0.00888 to +0.00899 mV
  significant. Dynamics improves by 0.00584 mV/ms, but F1-area/5 ms gains
  shrink to +0.00522/+0.00083 uncertain. Reject the depth/state reallocation.
- Compute-matched large-global replication: voltage reverses from -0.03469 to
  +0.06459 mV significant. Response changes by -0.00202 mV, F1 area by
  +0.00691, and 5 ms F1 by +0.00934, all uncertain; exact F1 improves 0.01229.
  The material voltage failure rejects scale-up despite event improvements.
- Learned-slice replication: voltage again regresses significantly (+0.00706
  mV), response changes -0.00138 mV, F1 area -0.00209, and 5 ms F1 +0.00141.
  This confirms that the I117 latent operator is not a stable lead.
- Fixed-sparse result: versus promoted, voltage changes -0.00449 mV uncertain
  and response improves -0.01181 mV significantly, but F1 area regresses
  -0.01431 and 5 ms F1 -0.02144 significantly. Relative to learned slicing,
  fixed partitions improve voltage/response by 0.01154/0.01043 mV while losing
  0.01222 F1 area and 0.02285 at 5 ms. Preventing assignment collapse exposes
  a SERA-versus-event frontier rather than a balanced operator advantage.
- Decision: reject exact-depth reallocation, large global scaling, learned
  slices, and fixed sparse slices without further state/width/token tuning.
  Keep the promoted I110 model. I119 moves upstream to the missing learned
  synaptic efficacy before branch summation, a structural component present in
  AxoSim's biological ELM but absent from AxoMamba.
- Compute: panel wall 244.66 seconds; exact-global 26.06 seconds, large-global
  35.43 seconds, and learned/fixed latent 35.39/35.44 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-118-global-depth-replication-sparse-slices/`
  and `manifests/iteration-118-global-depth-replication-sparse-slices.json`.

### Iteration 119 premise - learned synaptic efficacy before branch integration

- Missing-information mechanism: AxoMamba routes the 1,278 signed site channels
  into 118 x 11 slots and performs an unweighted sum before any learned layer.
  Channels merged inside one branch therefore become observationally
  indistinguishable to every downstream Mamba block. AxoSim's biological ELM
  instead learns one efficacy per routed synapse before branch integration,
  matching the structured dendritic architecture literature.
- Intervention: learn positive per-slot gains initialized exactly at one, then
  sum within each branch. An exact control owns and executes the same log-gain,
  exponential, multiplication, and sum path with scale zero. Active and control
  have the same 70,459 parameters and compute; the ordinary promoted model is
  included to verify that the dormant path is behaviorally inert.
- Acceptance: require a material balanced SERA/F1 gain over the exact compute
  control, with learned gains measurably heterogeneous and no failure trigger.
  Do not tune gain bounds, regularization, or morphology conditioning after a
  clean failure.
- Fidelity audit: active and zero-scale control each have exactly 70,459
  parameters, are output-identical at initialization, and take 3.08/3.11 ms per
  batch-32 x 500 training step versus 3.11 ms for promoted. Both allocate 353.3
  MB. The control's gain gradients are exactly zero while the active gradients
  are finite and nonzero; 106/106 tests pass.

### Iteration 119 - 2026-07-12

- Regime: exact-parameter/compute missing-information test on seed 20260746,
  120k presentations, batch 32, 3,750 updates, a shared frozen schedule, and
  full diagnostics. The zero-scale synapse path is the diagnostic baseline;
  the ordinary promoted model verifies its inertness.
- Results: learned synaptic efficacy improves voltage by 0.03281 mV, dynamics
  by 0.00115 mV/ms, response by 0.00607 mV, F1 area by 0.01715, exact F1 by
  0.00348, and 5 ms F1 by 0.01849. Paired intervals classify voltage,
  response, F1 area, and 5 ms F1 as improved; no core point estimate regresses.
- Control audit: promoted and the 70,459-parameter zero-scale control differ by
  only 3.5e-8 mV voltage, 2.8e-8 mV response, 1.7e-9 mV/ms dynamics, and zero
  on all F1 metrics after full training. This validates exact behavioral
  inertness rather than merely matching at initialization.
- Mechanism audit: all 1,278 valid synaptic gains move heterogeneously. Gains
  have mean 0.983, standard deviation 0.111, 10th/90th percentiles 0.863/1.111,
  and range 0.731-1.581; median within-branch standard deviation is 0.0866.
  Excitatory and inhibitory means are 1.000 and 0.965. The model uses site
  identity rather than learning one global rescale.
- Interpretation: this is a balanced and attributable structural lead, though
  its voltage gain is still only 0.189% of base error. It supports the premise
  that equal summation discarded useful synapse-specific efficacy, but does not
  yet establish the requested fraction-of-error reduction.
- Decision: run the unchanged active and exact compute control at the scheduled
  I120 300k/9,375-update gate on fresh seed 20260747. Promote only if the
  balanced profile survives. Perform mandatory cleanup and literature scan in
  parallel with the long run.
- Compute: panel wall 160.82 seconds; active 23.44 seconds versus exact control
  24.86 seconds and promoted 22.77 seconds model compute.
- Artifacts: `artifacts/diagnostics/iteration-119-synaptic-efficacy/` and
  `manifests/iteration-119-synaptic-efficacy.json`.

### Iteration 120 premise - long synaptic-efficacy gate and cleanup

- Long gate: exact active/control comparison at 70,459 parameters, seed
  20260747, 300k presentations, batch 32, 9,375 updates, common schedule,
  initialization, objective, and full diagnostics. No coefficient or optimizer
  change is allowed.
- Scheduled maintenance: remove executable paths and tests for rejected
  I111-I118 temporal interaction, modulation, reinjection, state-gating,
  distillation, and latent-token modifications while retaining historical
  manifests/diagnostics. Run the full AxoSim/AxoBench suite and load the stable
  checkpoint after pruning.
- Scheduled research refresh: perform the iteration-120 literature scan while
  the long GPU run executes, emphasizing architectures that preserve full
  synaptic/site information and can plausibly produce order-of-magnitude larger
  voltage gains than residual attachments.

### Iteration 120 maintenance - 2026-07-12

- Cleanup completed while the GPU gate runs. Removed executable configuration,
  constructor/forward paths, runner support, helper functions, and tests for
  rejected temporal bilinear filtering, bilinear modulation, branch forcing
  reinjection, interaction-conditioned state updates, dormant dynamic routing,
  grouped Mamba, latent/output distillation, and learned/fixed latent branch
  operators. Historical manifests, checkpoints, and diagnostics are retained.
- Verification: post-prune AxoSim plus AxoBench-integrated suite passes 93/93;
  `git diff --check` passes. The stable I110 checkpoint loads as AxoMamba with
  69,161 parameters and 300k training presentations. Removed serialized default
  fields are handled explicitly as retired checkpoint keys.
- Literature: completed `literature/iteration-120-synapse-structure-scan.md`.
  It ranks low-rank synapse-specific temporal bases first, morphology-conditioned
  synaptic dynamics second, and multiple sparse nonlinear branch subunits third.
  It explicitly carries forward the global-scale and latent-token rejections.

### Iteration 120 - 2026-07-12

- Regime: scheduled fresh-seed 300k promotion gate; active and zero-scale
  control each have exactly 70,459 parameters, batch 32, 9,375 updates, the
  same frozen seed-20260747 schedule, objective, initialization, and synapse
  gain compute path.
- Results: learned synaptic efficacy improves voltage by 0.07889 mV
  (95% paired interval -0.08656 to -0.07047), dynamics by 0.00343 mV/ms,
  response by 0.01244 mV, and 5 ms F1 by 0.01586; all four are significant.
  F1 area rises 0.00745 uncertainly. Exact F1 falls 0.00123 uncertainly, well
  below the practical threshold, with no significant core regression.
- Absolute active metrics: voltage SERA 17.30509 mV, dynamics SERA 21.32527
  mV/ms, intervention-response SERA 17.60488 mV, F1 area 0.21141, exact F1
  0.07104, and 5 ms F1 0.29494. The paired control voltage is 17.38398 mV.
- Mechanism audit: gain heterogeneity strengthens with training. The valid
  gains have mean 0.986, standard deviation 0.235, 10th/90th percentiles
  0.714/1.261, and range 0.514-2.141; excitatory/inhibitory means are
  1.010/0.962. This is a learned site-specific representation, not global
  rescaling.
- Interpretation: the short-run balanced profile survives and becomes stronger
  at 300k. The 0.07889 mV voltage gain is the largest replicated balanced
  structural improvement in this phase, but it is still only 0.454% of the
  control error, short of the desired visible fraction of roughly 20 mV.
- Decision: promote. `configs/axomamba.json`, `default_axomamba_config()`, and
  `checkpoints/axomamba.pt` now use `synapse_gain_scale=1.0`; the deployed
  checkpoint is the exact I120 active artifact with 70,459 parameters and 300k
  presentations. I121 follows the literature-ranked heterogeneous synaptic
  temporal-basis hypothesis rather than tuning the scalar gains.
- Compute: panel wall 341.33 seconds; active/control model compute 58.06/61.37
  seconds. Cleanup and literature scan completed during the run.
- Artifacts: `artifacts/diagnostics/iteration-120-long-synaptic-efficacy/`,
  `manifests/iteration-120-long-synaptic-efficacy.json`, and
  `literature/iteration-120-synapse-structure-scan.md`.

### Iteration 121 premise - heterogeneous synaptic temporal bases

- Premise: I120 establishes that synapse identity matters before branch
  aggregation. HetSyn predicts that synapse-specific temporal response matters
  as well. Give each routed synapse a learned mixture over a compact causal
  timescale basis, aggregate mixture-specific branch drives, and filter only
  after this identity-aware mixing.
- Efficient form: use four branch streams corresponding to event-local through
  membrane-scale causal decays; each synapse has positive efficacy and mixture
  weights. Linearity permits aggregation before filtering, avoiding 1,278 full
  state traces. Feed the 118 x 4 features through a learned projection into the
  promoted Mamba trunk.
- Controls: a tied-mixture control executes identical filters/mixing/projection
  and has exact parameters/compute; a global-capacity control brackets any
  unavoidable projection-size increase. Require a balanced material gain and
  measured mixture heterogeneity. Do not tune timescales after a clean failure.

### Iteration 121 - 2026-07-12

- Regime: 120k presentations, batch 32, 3,750 updates, seed 20260748, common
  schedule/objective/initialization. Tied and heterogeneous operators each
  have exactly 96,183 parameters and execute the same four causal filters. A
  96,187-parameter four-layer global-capacity control and the 70,459-parameter
  promoted instantaneous model bracket parameter and representation effects.
- Exact-control result: heterogeneous mixtures improve voltage by 0.02340 mV,
  dynamics by 0.00707 mV/ms, response by 0.00731 mV, exact F1 by 0.00944, F1
  area by 0.00457, and 5 ms F1 by 0.00458 versus tied mixtures. Voltage,
  dynamics, response, and exact F1 are significant; no core point estimate is
  adverse. The synapse-specific premise therefore has signal.
- Promoted-model result: a direct paired comparison shows heterogeneous bases
  significantly regress voltage by 0.03638 mV and 5 ms F1 by 0.02261 versus
  I120, despite significantly improving dynamics by 0.01852 mV/ms, response by
  0.00651 mV, and exact F1 by 0.01308. F1 area falls 0.00721 uncertainly. The
  global-capacity control also fails to dominate the promoted model.
- Mechanism audit: valid-synapse mixtures remain close to uniform, with mean
  normalized entropy 0.99783, logit standard deviation 0.07874, and mean basis
  probabilities [0.2477, 0.2536, 0.2494, 0.2494]. This is weak but nonzero
  specialization, not the decisive partition expected for a large endpoint.
- Compute: panel wall 205.11 seconds; tied/heterogeneous model compute
  33.44/32.59 seconds versus 22.97 for promoted and 28.02 for global capacity.
  The temporal path is about 42% slower in the shared training panel.
- Decision: reject temporal bases as a replacement representation and do not
  tune taus. I122 preserves the promoted instantaneous projection exactly and
  adds the same temporal operator as a zero-initialized residual, testing the
  observed complementary dynamics/response signal without surrendering the
  voltage/F1 path.
- Artifacts: `artifacts/diagnostics/iteration-121-heterogeneous-synaptic-timescales/`
  and `manifests/iteration-121-heterogeneous-synaptic-timescales.json`.

### Iteration 122 premise - residual causal multi-resolution forcing

- I121's instantaneous and temporal representations have complementary metric
  profiles. Preserve the promoted 118-to-58 instantaneous projection and add
  a separately projected four-timescale synaptic basis as a zero-initialized
  residual before Mamba. The candidate begins exactly as I120 and can learn to
  use temporal forcing only where useful.
- Compare heterogeneous and globally tied residual operators with exact
  parameters/compute, plus a promoted control. Acceptance requires recovering
  the I121 dynamics/response improvements without significant voltage or
  tolerant-F1 loss; otherwise reject the temporal-basis family.

### Iteration 122 - 2026-07-12

- Regime: seed 20260749, 120k presentations, batch 32, 3,750 updates. Tied and
  heterogeneous residuals have exactly 103,085 parameters and identical
  measured 5.35/5.34 ms synthetic step latency. Both begin output-identical to
  promoted I120 because the additional temporal projection is zero-initialized.
- Exact-control result: heterogeneous residuals improve all six point estimates
  over tied residuals: voltage -0.01507 mV, dynamics -0.00339 mV/ms, response
  -0.00239 mV, F1 area +0.00751, exact F1 +0.00235, and 5 ms F1 +0.00598.
  Voltage, dynamics, and response are significant.
- Promoted-model result: direct paired comparison rejects the candidate.
  Dynamics improves 0.01124 mV/ms and response 0.01206 mV significantly, but
  voltage regresses 0.03037 mV, F1 area falls 0.01619, and 5 ms F1 falls
  0.02871, all significant. Exact F1 rises 0.00458 uncertainly.
- Activity audit: the heterogeneous temporal projection reaches weight RMS
  0.04223 versus 0.05472 for the base branch projection, so failure is not an
  inert zero-initialized path. Mixture entropy remains high at 0.99606.
- Decision: reject and close fixed temporal-basis replacement/residual
  architectures. Do not tune residual scale or taus. I123 advances to the
  literature-ranked multiple sparse nonlinear subunits per branch, preserving
  within-branch spatial event patterns rather than adding another time filter.
- Compute: panel wall 205.60 seconds; heterogeneous/tied residual compute
  32.82/33.59 seconds, promoted 23.00 seconds, global control 28.12 seconds.
- Artifacts: `artifacts/diagnostics/iteration-122-residual-multiresolution-forcing/`
  and `manifests/iteration-122-residual-multiresolution-forcing.json`.

### Iteration 123 premise - sparse nonlinear dendritic subunits

- Current branch aggregation compresses 11 neighboring routed synapses to one
  scalar before the branch tanh. Three contiguous subunits per branch retain
  local co-activation patterns and apply the retained nonlinearity before a
  zero-initialized residual projection into Mamba.
- The exact control owns and executes identical per-subunit gains, nonlinearities,
  and projection but gives every subunit the full dense 11-slot receptive field.
  The active model uses three variance-normalized contiguous receptive fields.
  This isolates restricted dendritic structure from parameter count/compute.

### Iteration 123 - 2026-07-12

- Regime: seed 20260750, 120k presentations, batch 32, 3,750 updates. Dense and
  contiguous subunit models each have exactly 94,943 parameters and 4.05/4.04
  ms synthetic step latency. A 94,939-parameter global-depth control and the
  promoted 70,459-parameter model complete the panel.
- Exact-control result: contiguous sparse subunits improve voltage 0.00861 mV,
  F1 area 0.01754, exact F1 0.01316, and 5 ms F1 0.01838 over dense subunits,
  all significant. Dynamics changes +0.00081 mV/ms and response +0.00014 mV,
  both uncertain. Restricted receptive fields specifically improve voltage and
  event fidelity while dense subunit capacity supplies the SERA foundation.
- Promoted-model result: direct paired comparison is favorable and significant
  on every core metric: voltage -0.08208 mV, dynamics -0.01007 mV/ms, response
  -0.02646 mV, F1 area +0.02392, exact F1 +0.01210, and 5 ms F1 +0.02124.
  Absolute voltage is 17.30741 mV at only 120k presentations.
- Global-capacity result: sparse subunits improve voltage 0.03351 mV
  significantly and are uncertainly favorable on dynamics/F1 area/5 ms, while
  global capacity improves response 0.00872 mV significantly and exact F1
  0.00235 uncertainly. Neither dominates, but generic capacity does not explain
  the sparse model's voltage advantage.
- Activity audit: the sparse residual projection reaches RMS 0.02159 versus
  0.05443 for the base projection, and valid subunit gains have std 0.04677 and
  range 0.788-1.396. The zero-initialized path is active and heterogeneous.
- Decision: advance unchanged to an I124 fresh-seed replication with the same
  exact dense, promoted, and global controls. Do not vary subunit count or mask
  geometry before replication.
- Compute: panel wall 197.39 seconds; sparse/dense/global/promoted model compute
  28.05/28.98/28.27/23.12 seconds.
- Artifacts: `artifacts/diagnostics/iteration-123-sparse-dendritic-subunits/`
  and `manifests/iteration-123-sparse-dendritic-subunits.json`.

### Iteration 124 premise - unchanged sparse-subunit replication

- Repeat I123 on seed 20260751 with the exact same architecture, objective,
  presentation budget, and controls. Promote to the 300k gate only if sparse
  subunits remain balanced versus promoted I120 and retain an attributable
  voltage/F1 advantage over exact dense subunits.

### Iteration 124 - 2026-07-12

- Regime: unchanged seed-20260751 replication, 120k presentations, exact same
  four-candidate controls and compute as I123.
- Promoted-model result: sparse subunits again improve every point estimate.
  Direct paired deltas are voltage -0.04356 mV and response -0.00470 mV,
  significant; dynamics -0.00384 mV/ms, F1 area +0.00935, exact F1 +0.00923,
  and 5 ms F1 +0.00467 are uncertain. The balanced model-level lead replicates.
- Exact-control result: sparse versus dense improves voltage 0.01167 mV and
  response 0.00487 mV significantly, dynamics changes +0.00146 uncertainly,
  exact F1 is tied, and the I123 F1-area/5 ms gains reverse by
  -0.00988/-0.01383 uncertainly. Restricted structure has a repeatable small
  voltage effect, but not a repeatable event-fidelity effect.
- Global-capacity result: sparse and the 94,939-parameter global model are tied
  on voltage, dynamics, and every F1 metric; sparse response is 0.00405 mV
  worse significantly. Thus the broad model gain is not uniquely attributable
  to dendritic sparsity and remains below one percent of base voltage error.
- Decision: do not promote or long-gate the subunit model. Retain the two-seed
  result as a capacity/structure lead, but move to morphology-conditioned
  per-synapse efficacy, the next dataset-specific literature-ranked mechanism.
- Compute: panel wall 196.11 seconds; sparse/dense/global/promoted model compute
  28.08/29.02/28.37/23.19 seconds.
- Artifacts: `artifacts/diagnostics/iteration-124-sparse-subunit-replication/`
  and `manifests/iteration-124-sparse-subunit-replication.json`.

### Iteration 125 premise - morphology-conditioned synaptic efficacy

- The dataset's normalized-by-type-arclength generation maps a common source
  channel to different physical segments in each morphology; for one recorded
  cell only 264 of 639 source sites per sign are unique after routing. A single
  shared efficacy therefore cannot represent morphology-specific site impact.
- Add a five-morphology log-gain table before branch aggregation, initialized
  at zero on top of the promoted shared synaptic gains. Active and scale-zero
  controls have exact parameters/compute and begin identically. Require improved
  average and morphology-disparity/intervention behavior without F1 collapse.

### Iteration 125 - 2026-07-12

- Regime: seed 20260752, 120k presentations, batch 32, 3,750 updates. Active
  and scale-zero morphology-synapse paths each have exactly 76,949 parameters,
  identical initialization, and 3.07/3.06 ms synthetic step latency. The
  ordinary promoted model verifies the zero-scale path to 5.7e-8 mV; a
  76,903-parameter global control brackets generic capacity.
- Exact-control result: active conditioning significantly improves all six
  core metrics: voltage -0.08575 mV, dynamics -0.00442 mV/ms, response
  -0.02308 mV, F1 area +0.02039, exact F1 +0.01210, and 5 ms F1 +0.02316.
  Absolute voltage is 17.28905 mV, the best 120k result in this phase.
- Global-capacity result: morphology-conditioned synapses improve voltage
  0.04728 mV, F1 area 0.02930, and 5 ms F1 0.03242 significantly over global
  capacity; dynamics is tied, exact F1 rises 0.01092 uncertainly, and response
  regresses 0.00445 mV significantly. This is a favorable but non-dominating
  structural tradeoff, not merely a parameter-count effect.
- Mechanism audit: per-morphology valid gain standard deviations range
  0.0679-0.1234; mean between-morphology per-slot std is 0.08925 and maximum
  0.27688. Worst-morphology RMSE disparity changes adversely from 1.378 to
  1.388, so the gain is average fidelity rather than disparity reduction.
- Decision: unchanged fresh-seed replication at I126. No gain-scale tuning.
  Advance only if the balanced exact-control profile and global-capacity
  voltage/F1 advantage survive.
- Compute: panel wall 185.77 seconds; active/control/global/promoted model
  compute 24.55/25.68/23.51/23.43 seconds.
- Artifacts: `artifacts/diagnostics/iteration-125-morphology-synaptic-efficacy/`
  and `manifests/iteration-125-morphology-synaptic-efficacy.json`.

### Iteration 126 premise - unchanged morphology-synapse replication

- Repeat I125 on seed 20260753 with identical architecture, controls, objective,
  and budget. Require no significant core regression versus exact scale-zero
  control and a material advantage that generic capacity does not dominate.

### Iteration 126 - 2026-07-12

- Regime: unchanged seed-20260753 replication, 120k presentations, exact same
  active/scale-zero/promoted/global panel as I125.
- Exact-control result: all six improvements replicate significantly: voltage
  -0.05972 mV, dynamics -0.00264 mV/ms, response -0.01255 mV, F1 area
  +0.02519, exact F1 +0.00958, and 5 ms F1 +0.03028. Absolute active voltage
  is 17.29454 mV and 5 ms F1 is 0.28089.
- Capacity control: the 76,903-parameter global model significantly regresses
  voltage, response, F1 area, and exact F1 versus the scale-zero control on
  this seed, while morphology-conditioned synapses improve all of them. The
  I125 response tradeoff against global capacity does not repeat.
- Decision: morphology-conditioned efficacy is the current provisional
  structural winner. I127 tests exact factorial composition with contiguous
  branch subunits; do not promote or alter the deployed checkpoint before the
  scheduled long gate.
- Compute: panel wall 184.75 seconds; active/control/global/promoted model
  compute 24.27/25.37/23.20/23.09 seconds.
- Artifacts: `artifacts/diagnostics/iteration-126-morphology-synapse-replication/`
  and `manifests/iteration-126-morphology-synapse-replication.json`.

### Post-I126 raw-voltage calibration audit

- A leakage-safe affine fit on the 30 calibration traces and evaluation on the
  90 scored traces used raw `soma_mv`, not the clipped normalized target.
  Global affine calibration improves voltage SERA from 17.35426 to 17.31883
  mV; five morphology-specific affine fits reach 17.28470 mV. Dynamics changes
  by less than 0.00037 mV/ms.
- The diagnostic voltage gain remains only 0.252 after morphology calibration,
  so scale/offset mismatch explains little of the roughly 17 mV error. Queue a
  direct morphology-specific soma calibration path after the structural
  composition test, but do not mistake it for a fraction-of-error solution.

### Iteration 127 premise - exact morphology x subunit composition

- Test a 2 x 2 factorial in which every candidate owns and executes the same
  morphology-synapse table and three contiguous subunit path. Fixed scales
  independently disable morphology gains and the subunit residual, yielding
  both-off, morphology-only, subunit-only, and combined candidates with exact
  parameters/compute.
- The combination advances only if it materially improves the replicated
  morphology-only model without a core regression. Include near-matched global
  capacity to prevent attributing an additive gain to total size.

### Iteration 127 - 2026-07-12

- Regime: exact 2 x 2 factorial at 101,433 parameters and 4.48-4.49 ms per
  synthetic step in every cell, seed 20260754, 120k presentations. The global
  control also has exactly 101,433 parameters.
- Main effects versus both-off: morphology-only improves all six metrics
  significantly (voltage -0.07665 mV); sparse-only improves voltage, dynamics,
  and response significantly but loses 5 ms F1 by 0.01146 uncertainly.
- Combined result versus both-off: voltage -0.11181 mV, dynamics -0.00746
  mV/ms, response -0.02104 mV, F1 area +0.01859, exact F1 +0.00566, and 5 ms
  F1 +0.01332. Voltage/dynamics/response/F1 area are significant; no core
  metric significantly regresses.
- Direct interaction: combined versus morphology-only improves voltage
  0.03515 mV and response 0.00820 mV significantly. Dynamics improves 0.00314
  uncertainly, F1 area rises 0.00272 uncertainly, exact F1 falls 0.00363
  uncertainly, and 5 ms F1 falls 0.00337 uncertainly. The subunit contribution
  is additive on SERA without a demonstrated event cost.
- Capacity control: exact global capacity regresses voltage 0.01288 mV versus
  both-off and improves response 0.01024 mV; it is dominated by the combined
  model on every core point estimate except exact F1, where combined is also
  higher by 0.00700.
- Decision: I128 replicates morphology-only versus combined versus exact global
  unchanged. The combination is the provisional I130 long-gate candidate.
- Compute: five-candidate panel wall 234.91 seconds; factorial cell model
  compute 29.56-30.41 seconds and global 27.76 seconds.
- Artifacts: `artifacts/diagnostics/iteration-127-morphology-subunit-factorial/`
  and `manifests/iteration-127-morphology-subunit-factorial.json`.

### Iteration 128 premise - combined structural replication

- On seed 20260755, repeat the exact 101,433-parameter morphology-only,
  combined, and global-capacity candidates. Require the combined model to
  retain significant SERA improvement over morphology-only without a
  significant F1 regression before it can enter the long gate.

### Iteration 128 - 2026-07-12

- Regime: seed 20260755, unchanged 120k replication with morphology-only,
  combined, and global candidates all exactly 101,433 parameters.
- Combined versus morphology-only: all six point estimates improve. Voltage
  improves 0.07247 mV, dynamics 0.00806 mV/ms, response 0.01769 mV, and 5 ms
  F1 0.01670 significantly; F1 area rises 0.01107 and exact F1 0.00708
  uncertainly. Absolute combined metrics are voltage 17.25022 mV, dynamics
  21.32821 mV/ms, response 17.60323 mV, F1 area 0.19600, exact F1 0.07954,
  and 5 ms F1 0.26550.
- Global control: versus morphology-only, exact global capacity significantly
  regresses voltage 0.05010 mV, F1 area 0.01658, exact F1 0.01173, and 5 ms
  F1 0.01839. The combined architecture dominates global on every point
  estimate, so the replicated interaction is not a size effect.
- Decision: lock morphology-conditioned efficacy plus three contiguous
  subunits as the I130 structural candidate. I129 tests only the queued
  ten-parameter post-coupling morphology-specific soma affine under an exact
  inactive path; no structural or objective changes.
- Compute: three-candidate panel wall 176.99 seconds; combined 29.46 seconds,
  morphology-only 30.26 seconds, global 27.61 seconds.
- Artifacts: `artifacts/diagnostics/iteration-128-combined-structural-replication/`
  and `manifests/iteration-128-combined-structural-replication.json`.

### Iteration 129 premise - learned morphology soma calibration

- The leakage-safe post-hoc audit found a 0.06956 mV voltage-SERA gain from
  morphology-specific affine calibration. Add one log-scale and bias per
  morphology after soma-to-spike coupling, so voltage can calibrate without
  directly moving spike logits.
- Active and scale-zero candidates own and execute the same ten parameters and
  begin exactly as the locked combined architecture. Require voltage gain with
  no response/dynamics/F1 regression; this is a diagnostic-derived final
  refinement, not a coefficient sweep.

### Iteration 129 - 2026-07-12

- Regime: seed 20260756, 120k presentations, exact 101,443-parameter active and
  scale-zero combined models, with post-coupling application verified to leave
  spike logits unchanged.
- Result: learned affine calibration improves voltage 0.01625 mV significantly
  and dynamics 0.00041 mV/ms uncertainly. As intended, F1 metrics remain tied
  (all deltas <= 0.00489). However, intervention response regresses 0.02641 mV
  significantly, far exceeding the voltage gain.
- Decision: reject. The post-hoc calibration signal does not survive end-to-end
  training as a balanced model improvement. I130 excludes the affine and
  long-gates both-off, morphology-only, combined, and exact global capacity.
- Compute: panel wall 150.62 seconds; active/control model compute 30.06/30.91
  seconds.
- Artifacts: `artifacts/diagnostics/iteration-129-morphology-soma-affine/` and
  `manifests/iteration-129-morphology-soma-affine.json`.

### Iteration 130 premise - long structural promotion gate

- Run 300k presentations, batch 32, 9,375 updates on seed 20260757. All four
  candidates have exactly 101,433 parameters and share compute: both mechanisms
  off, morphology synapses only, morphology synapses plus sparse subunits, and
  global capacity.
- Promote only if morphology conditioning remains balanced over both-off and
  subunits retain a balanced additive gain over morphology-only at long budget.
  Complete mandatory failed-path cleanup, full verification, and a literature
  scan during the long run.

### Iteration 130 maintenance - 2026-07-12

- Cleanup removed executable config, parameters, forward/helper paths, and
  tests for rejected I121-I122 synaptic temporal bases and I129 morphology soma
  affine. Historical manifests/diagnostics remain; retired config keys preserve
  compatibility. Accepted shared/morphology synaptic gains and branch subunits
  remain.
- Checkpoint verification exposed and fixed one deterministic-buffer migration:
  checkpoints created before branch subunits lacked `branch_subunit_mask`.
  The loader now reconstructs that derived buffer only when absent while
  retaining strict learned-parameter loading. The deployed 70,459-parameter
  300k checkpoint loads successfully.
- Verification: cleaned AxoSim+AxoBench-integrated suite passes 97/97,
  AxoBench evaluator passes 24/24, and `git diff --check` passes.
- Literature refresh completed at
  `literature/iteration-130-geometry-event-operator-scan.md`. It ranks a
  verified normalized-coordinate synaptic operator first, an SWC tree graph
  encoder second, geometry-organized subunits third, and an explicit
  event/change decoder fourth.

### Iteration 130 - 2026-07-12

- Regime: scheduled 300k gate, seed 20260757, batch 32, 9,375 updates. Both-off,
  morphology-only, combined, and global candidates each have exactly 101,433
  parameters and share schedule, initialization, objective, and compute paths.
- Morphology-only versus both-off: all six metrics improve significantly:
  voltage -0.15277 mV, dynamics -0.02285 mV/ms, response -0.04586 mV, F1 area
  +0.06470, exact F1 +0.04552, and 5 ms F1 +0.06196. Absolute metrics are
  voltage 17.15971 mV, dynamics 21.30351 mV/ms, response 17.57034 mV, F1 area
  0.28866, exact F1 0.11421, and 5 ms F1 0.36600.
- Combined frontier: versus morphology-only, subunits improve voltage 0.04103
  mV, F1 area 0.01630, and 5 ms F1 0.02880 significantly; exact F1 rises
  0.00943 uncertainly. Dynamics regresses 0.00461 mV/ms uncertainly and response
  regresses 0.01388 mV significantly. This is a useful voltage/event frontier
  point but fails the balanced promotion criterion.
- Global control: morphology-only beats the exact global-capacity model on all
  six point estimates. At long budget, generic capacity no longer explains the
  morphology effect.
- Mechanism audit: morphology-specific gain std is 0.194-0.313 by cell; mean
  between-morphology slot std is 0.236 and maximum 0.664. Voltage gain improves
  from 0.257 to 0.288 and default-threshold count ratio from 1.684 to 1.650.
  Refractory violations worsen from 0.243 to 0.303 and morphology disparity from
  1.387 to 1.405; both were already active failure triggers, and neither is a
  promotion claim.
- Decision: promote morphology-conditioned synaptic efficacy only. The deployed
  `checkpoints/axomamba.pt`, `configs/axomamba.json`, and
  `default_axomamba_config()` use `morphology_synapse_gain_scale=1.0`. The
  checkpoint is output-identical to the long-gate candidate after stripping
  24,484 dormant factorial-control parameters, leaving 76,949 functional
  parameters and 300k training presentations.
- Improvement over deployed I120: voltage -0.14539 mV, dynamics -0.02176
  mV/ms, response -0.03454 mV, F1 area +0.07725, exact F1 +0.04317, and 5 ms
  F1 +0.07106. The voltage gain is about 0.84% of I120 error: materially larger
  and balanced, but still not the requested large fraction of roughly 20 mV.
- Verification: stripped/full CUDA outputs match exactly; config/default/
  checkpoint match; AxoSim passes 97/97, AxoBench 24/24, and `git diff --check`
  passes. Deployed SHA-256 is
  `4744945e49aa598878c3d73982f959ee157a9c722c23534a13bb4fe60281f7ca`.
- Compute: four-candidate panel wall 512.14 seconds; morphology-only/combined/
  both-off/global model compute 74.55/74.19/76.49/69.75 seconds.
- Artifacts: `artifacts/diagnostics/iteration-130-long-structural-gate/`,
  `manifests/iteration-130-long-structural-gate.json`, and
  `literature/iteration-130-geometry-event-operator-scan.md`.

### Iteration 131 premise - verified normalized-coordinate routing

- Reconstruct the five exact normalized-by-type-arclength source-to-segment
  maps from the retained SWCs and AxoBench generation code. Validate segment,
  unique-source, unused-source, and duplication summaries against every shard
  morphology before using the maps as model inputs.
- If reconstruction passes, test a shared coordinate-conditioned synaptic
  efficacy operator against exact shuffled-coordinate and constant-coordinate
  controls on top of promoted I130. If it fails, do not claim graph/topology
  use and move to the event/change decoder ranked by the I130 scan.

### Iteration 131 route audit and controlled panel - 2026-07-12

- Route reconstruction passed all 1,285 shard metadata checks across five
  morphologies. The exact source-to-segment summaries range from 264 to 370
  unique used sites per morphology and retain every recorded duplicated and
  unused source count.
- Artifact: `artifacts/routes/v1-normalized-route-features.npz` contains real,
  deterministically within-type shuffled, and constant-coordinate tensors of
  shape `(5, 1278, 9)`. Operative continuous features are normalized to
  `[0,1]`; sign remains `-1/+1`.
- Intervention: a shared `9 -> 16 -> 1` MLP predicts a positive multiplicative
  synaptic efficacy residual from the fixed coordinate tensor after the
  promoted categorical morphology gain. Its output layer is initialized to
  zero, making every candidate exactly equal at step zero.
- Controls: real, shuffled, constant, and feature-scale-zero candidates each
  have 77,125 parameters and execute the same feature path. They share seed
  20260758, frozen 120k schedule, 3,750 optimizer updates, objective, and
  initialization. The shuffled control preserves feature marginals within
  morphology, input sign, and apical/basal type.
- Verification before launch: artifact morphology ordering and tensor shapes
  are checked by the runner; the artifact SHA-256 is recorded with each run;
  targeted AxoSim tests pass 33/33; all four CUDA outputs match at
  initialization.
- Manifest: `manifests/iteration-131-normalized-coordinate-operator.json`.

### Iteration 131 - 2026-07-12

- Regime: 120k presentations, seed 20260758, batch 32, 3,750 updates. Real,
  shuffled, constant, and feature-scale-zero candidates each have exactly
  77,125 parameters, use the same initialization and schedule, and execute the
  same coordinate MLP.
- Real versus shuffled coordinates improves all six metrics significantly:
  voltage -0.20633 mV, dynamics -0.00960 mV/ms, response -0.04473 mV, F1 area
  +0.08387, exact F1 +0.05322, and 5 ms F1 +0.08092. Absolute real-coordinate
  metrics are voltage 17.11676 mV, dynamics 21.32265 mV/ms, response 17.58570
  mV, F1 area 0.28279, exact F1 0.12525, and 5 ms F1 0.35496.
- Real versus exact scale-zero also improves all six metrics significantly:
  voltage -0.20765 mV, dynamics -0.01295 mV/ms, response -0.04467 mV, F1 area
  +0.09214, exact F1 +0.05073, and 5 ms F1 +0.09714. Constant coordinates
  are effectively tied to shuffled coordinates, isolating true route structure.
- Failure audit: voltage gain rises from 0.254 to 0.300 and spike-count error
  falls from 0.539 to 0.388 versus scale-zero. Existing near-spike,
  default-threshold-count, refractory, and morphology-disparity triggers worsen;
  these remain explicit risks for replication rather than being hidden by the
  core-metric success.
- Interpretation: the shared operator learns route correspondence, not extra
  parameters or feature marginals. This is a material, balanced structural
  effect, but 0.208 mV remains only about 1.2% of the 17.3 mV error scale.
  Replicate before promotion and use the verified tree as a higher-temperature
  extension.
- Compute: four-model shared panel wall 194.72 seconds; per-model compute
  26.27-27.30 seconds. Artifact SHA-256 is
  `7896d1d56c727adc1b969ca296d76b258f30be537cbdcfca76b0bcbda950c78b`.
- Artifacts: `artifacts/diagnostics/iteration-131-normalized-coordinate-operator/`,
  `artifacts/routes/v1-normalized-route-features.npz`, and
  `manifests/iteration-131-normalized-coordinate-operator.json`.

### Iteration 132 premise - SWC-tree event propagation

- Replicate I131 real-coordinate versus feature-scale-zero on seed 20260759.
  In parallel, propagate sparse input events over the exact source graph
  reconstructed from NEURON section parent edges, retaining the coordinate
  gain as a separate mechanism.
- The graph residual learns two shared terms: neighbor event and local-event
  times neighbor-event interaction. It begins exactly at zero. Real-tree,
  degree-preserving rewired-tree, coordinate-only, and full scale-zero models
  each have 77,127 parameters and execute the same degree-four gather path.
- Route audit finds 255-359 source edges per sign, maximum degree four. The
  rewired controls preserve every node degree and differ in 5,990 neighbor
  indices across the five morphologies. Full AxoSim tests pass 102/102 and the
  AxoBench evaluator passes 24/24 before launch.
- Manifest: `manifests/iteration-132-tree-graph-event-operator.json`.

### Iteration 132 - 2026-07-12

- Regime: fresh seed 20260759, 120k presentations, batch 32, 3,750 updates.
  Coordinate-only, real-tree, rewired-tree, and all-off candidates each have
  77,127 parameters and execute the same graph gather and feature paths.
- Coordinate replication versus scale-zero improves all six metrics
  significantly: voltage -0.21223 mV, dynamics -0.01339 mV/ms, response
  -0.05086 mV, F1 area +0.08737, exact F1 +0.05599, and 5 ms F1 +0.09170.
  Absolute coordinate metrics are voltage 17.08338 mV, dynamics 21.31589
  mV/ms, response 17.57430 mV, F1 area 0.25553, exact F1 0.11968, and 5 ms
  F1 0.32129.
- Real tree versus coordinate-only improves voltage 0.00448 mV and exact F1
  0.00579 significantly, with F1 area +0.00395 and 5 ms +0.00385 uncertain.
  Response regresses 0.00048 mV uncertainly. This is below the practical
  threshold despite the learned neighbor/event weights reaching 0.267/0.066.
- Rewired tree versus coordinate-only significantly regresses voltage by
  0.01662 mV and response by 0.00695 mV; all F1 point estimates also decline.
  Real and rewired models learn similar message magnitudes, so the contrast is
  evidence for true topology, not an inactive rewired arm. The current
  additive event operator is nevertheless rejected as too small.
- Compute: four-model shared panel wall 238.98 seconds; per-model compute
  37.55-38.51 seconds. Artifact SHA-256 is
  `a0271d6a337f737d4d416976895cd872493d8eea95cfb9eba5e80f4191b03039`.
- Artifacts: `artifacts/diagnostics/iteration-132-tree-graph-event-operator/`
  and `manifests/iteration-132-tree-graph-event-operator.json`.

### Iteration 133 premise - exact route-validity masking

- I131's learned coordinate delta correlates 0.960 with whether a source site
  is actually connected in the target morphology. The dataset generator maps
  only 528-740 of 1,278 signed source channels into each target cell; remaining
  events are known disconnected nuisance inputs, not latent biological causes.
- Apply the verified binary route mask before branch routing. Test real mask
  plus coordinates, degree/count-matched shuffled mask plus coordinates,
  real-mask-only, coordinate-only, and all-off in one exact 77,125-parameter
  panel. Every candidate executes both buffer paths; no mask coefficient sweep.
- The real and shuffled masks retain identical per-morphology counts and differ
  at 2,962 channel assignments. The fixed artifact is
  `artifacts/routes/v1-route-validity-masks.npz` with SHA-256
  `d0ac692bc9e84fc7a6c5888147ea7836c6af8b5a7a5ce921523b3c615f706514`.
- Manifest: `manifests/iteration-133-route-validity-mask.json`.

### Iteration 133 - 2026-07-12

- Regime: seed 20260760, 120k presentations, 3,750 updates. Coordinate-only,
  real-mask-plus-coordinate, shuffled-mask-plus-coordinate, real-mask-only,
  and all-off candidates each have 77,125 parameters and execute both paths.
- Coordinate-only versus all-off replicates all six core gains significantly
  for a third seed: voltage -0.19502 mV, dynamics -0.01449 mV/ms, response
  -0.04265 mV, F1 area +0.08909, exact F1 +0.05265, and 5 ms F1 +0.10316.
- Hard real mask plus coordinates regresses voltage 0.07996 mV, response
  0.04847 mV, and exact F1 0.01804 significantly versus coordinate-only;
  F1 area -0.01213 and 5 ms -0.01771 are uncertainly adverse. Real-mask-only
  is also dominated by coordinate-only.
- Shuffled hard masking is catastrophic: voltage +0.47293 mV, response
  +0.10694 mV, F1 area -0.14961, exact F1 -0.08746, and 5 ms F1 -0.16960,
  all significant. Correct route validity is informative, but disconnected
  events retain correlated rate information and should be softly weighted,
  not discarded.
- Decision: reject hard route masking without a floor sweep. The full
  coordinate operator already learns the needed soft distinction and remains
  the experimental baseline.
- Compute: five-model panel wall 223.95 seconds; per-model compute 26.89-27.95
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-133-route-validity-mask/`.

### Iteration 134 premise - explicit event/change decoder

- A geometry-organized subunit premise was reconstructed and canceled before
  training: normalized-arclength masks differ from current contiguous subunits
  in only 110 of 19,470 assignments because the interlocking branch route is
  already almost perfectly arclength ordered. The temporary path/artifact were
  deleted rather than spending an iteration on an effectively identical model.
- Following CP-DSA/Eventax and the persistent high-dV/dt/near-spike failures,
  add a zero-output residual decoder `118 -> 58 -> 32 -> 2` from branch-current
  changes. Compare adjacent change, state-times-change interaction, detached
  change, and a state-only bypass capacity control.
- All five candidates have 86,095 parameters, identical initial outputs,
  common seed 20260761/schedule/objective, and execute the same source
  projection and decoder path. The scale-zero coordinate model is the baseline.
- Manifest: `manifests/iteration-134-event-change-decoder.json`.

### Iteration 134 - 2026-07-12

- Regime: seed 20260761, 120k presentations, 3,750 updates. Coordinate
  compute-control, state bypass, change, change-interaction, and detached-change
  candidates each have 86,095 parameters and identical initial outputs.
- Ordinary change decoder versus control significantly improves all six core
  metrics: voltage -0.10956 mV, dynamics -0.01293 mV/ms, response -0.06836 mV,
  F1 area +0.02492, exact F1 +0.01202, and 5 ms F1 +0.03070. Absolute metrics
  are voltage 17.04734 mV, dynamics 21.29653 mV/ms, response 17.54040 mV,
  F1 area 0.31118, exact F1 0.14055, and 5 ms F1 0.38994.
- State-only bypass improves voltage 0.11370 mV and all three F1 metrics
  significantly, but response changes only +0.00034 mV uncertainly. Direct
  change versus state improves dynamics 0.01124 mV/ms and response 0.06870 mV
  significantly, while exact F1 falls 0.00856 significantly; voltage/F1-area/
  5 ms differences are uncertain. Thus direct decoder capacity explains much
  of voltage/F1, while temporal changes uniquely explain intervention response.
- Detached change also improves all six metrics versus control but is slightly
  dominated by ordinary change on every point estimate. State-times-change
  loses dynamics/response and is rejected.
- Decision: replicate ordinary change unchanged and compose it with the next
  high-temperature route mechanism. Do not tune decoder width or scale.
- Compute: five-model panel wall 227.70 seconds; per-model compute 27.42-28.86
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-134-event-change-decoder/`.

### Iteration 135 premise - exact target-segment routing

- The simulator routes each target segment from an explicit source index, but
  AxoMamba still sums a generic interlocking source layout. Reconstruct each
  morphology's 265-377 target segments into 118 geometry-ordered branches,
  duplicating source events exactly when one source drives multiple target
  segments and pairing excitatory/inhibitory events at each target site.
- Real physical routes contain 530-754 active signed segment slots; variance-
  normalize each branch and compare against a source-label-shuffled physical
  route with identical target occupancy/multiplicity. A generic route encoded
  through the same morphology-buffer gather is output-identical to the current
  path (`max_abs_diff=0.0`).
- Five exact 86,095-parameter arms jointly replicate change, test physical
  routing with/without change, and isolate real correspondence from shuffled
  physical topology on seed 20260762. All use the same coordinate decoder,
  model compute path, schedule, and objective.
- Artifact: `artifacts/routes/v1-morphology-physical-routing.npz`, SHA-256
  `157a60dd4629be68a1a30f7549598d3e500e784abeb51efdb0c09fdbbcd7a12a`.
  Manifest: `manifests/iteration-135-physical-segment-routing.json`.

### Iteration 135 - 2026-07-12

- Regime: seed 20260762, 120k presentations, 3,750 updates. Generic/physical
  routing crossed with change off/on plus source-shuffled physical change;
  all candidates have 86,095 parameters and matched buffer-gather compute.
- Generic change fails to replicate I134's SERA gains: voltage +0.02036 mV,
  dynamics +0.01405 mV/ms, and response +0.01557 mV regress significantly,
  although F1 area +0.01298 and 5 ms +0.02078 improve significantly.
- Physical route alone likewise regresses voltage 0.05558 mV and response
  0.03025 mV significantly. The combined physical-route change model instead
  improves voltage 0.01776 mV, response 0.02164 mV, F1 area 0.03280, exact F1
  0.02065, and 5 ms F1 0.03564 significantly versus generic control; dynamics
  improves 0.00431 mV/ms uncertainly.
- Direct interaction: physical change versus physical coordinate improves
  voltage 0.07333 mV, response 0.05189 mV, F1 area 0.02440, exact F1 0.02521,
  and 5 ms F1 0.02331 significantly; dynamics -0.00890 mV/ms is uncertain.
- Source-shuffled physical change catastrophically regresses every metric,
  including voltage +0.49755 mV and F1 area -0.13337. Exact source-to-target
  correspondence is necessary; occupancy and compute do not explain the real
  route result.
- Decision: neither main effect is stable enough to retain alone. Replicate
  the exact factorial unchanged on a fresh seed; promote only a reproducible
  composition, not the favorable interaction from one seed.
- Compute: five-model panel wall 227.62 seconds; per-model compute 27.85-29.10
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-135-physical-segment-routing/`.

### Iteration 136 premise - physical/change interaction replication

- Repeat the exact five-arm I135 panel on seed 20260763 with no architecture,
  route normalization, objective, or optimizer change. Require physical change
  to beat physical-coordinate and generic-change controls on balanced SERA/F1,
  while shuffled routing remains adverse.
- Manifest: `manifests/iteration-136-physical-change-replication.json`.

### Iteration 136 - 2026-07-12

- Regime: unchanged I135 factorial on seed 20260763, 120k presentations,
  3,750 updates, 86,095 parameters per candidate, common schedule/compute.
- Physical change versus generic control regresses voltage 0.05746 mV, F1 area
  0.03011, exact F1 0.01658, and 5 ms F1 0.04331 significantly. Dynamics
  improves 0.01050 mV/ms significantly and response 0.00640 mV uncertainly;
  this is the opposite balanced decision from I135.
- Generic change also fails: voltage +0.05442 mV, F1 area -0.01804, and 5 ms
  -0.02076 regress significantly. Physical route without change improves
  response 0.01232 mV and 5 ms F1 0.01554 significantly, but voltage moves
  +0.00750 mV uncertainly and other axes are mixed.
- Shuffled physical routing again catastrophically regresses all metrics,
  confirming exact correspondence for a second seed. The useful real-route
  downstream effect and its interaction with change are nevertheless unstable.
- Decision: reject event-change decoder, physical routing, and their
  composition without width/scale/normalization tuning. Queue executable paths
  for I140 cleanup; retain artifacts as negative evidence. Coordinate efficacy
  remains the only robust promoted-to-long-gate candidate from I131-I136.
- Compute: five-model panel wall 227.43 seconds; per-model compute 27.67-28.85
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-136-physical-change-replication/`.

### Iteration 137 premise - causal voltage-state residual

- The remaining failure is not merely branch correspondence: voltage gain
  remains about 0.30 and temporal residual lag stays active after coordinate
  conditioning. Following dynamic-system/Neural-ODE surrogate practice, test a
  decoder that emits a zero-initialized voltage forcing and evolves it through
  a causal leaky state before adding it to the direct soma readout.
- Compare leaky-state, cumulative-state, and non-integrated residual outputs
  with identical forcing MLP, learned decay, and all state computations. This
  distinguishes state evolution from generic output capacity without changing
  Mamba or the validated coordinate baseline.

### Iteration 137 - 2026-07-12

- Regime: seed 20260764, 120k presentations, 3,750 updates. Scale-zero,
  direct, leaky, and cumulative state candidates each have 79,162 parameters,
  identical initial outputs, and compute all state forms.
- Leaky state improves voltage 0.03145 mV and exact F1 0.01956 significantly,
  with dynamics -0.00201 mV/ms and response -0.00228 mV uncertainly favorable.
  It significantly regresses 5 ms F1 by 0.02144; F1 area falls 0.00720
  uncertainly. The balanced criterion fails.
- Direct residual improves voltage 0.01598 mV and response 0.00396 mV
  significantly, but significantly regresses F1 area 0.01887 and 5 ms F1
  0.02577. Cumulative state is decisively harmful: voltage +0.24560 mV and
  response +0.30463 mV.
- Interpretation: leaky evolution has some unique value over a direct residual,
  especially exact timing, but the state formulation trades tolerant timing
  rather than fixing it. Reject without decay/kernel/scale tuning.
- Compute: four-model panel wall 199.88 seconds; per-model compute 27.35-29.06
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-137-voltage-state-decoder/`.

### Iteration 138 premise - morphology-conditioned Mamba adapters

- Coordinate conditioning robustly identifies morphology-specific input
  efficacy, while temporal/output mechanisms remain seed-unstable. Add rank-4
  low-rank residual adapters after each Mamba block, indexed by morphology, so
  each connectome can alter temporal dynamics without duplicating the trunk.
- Compare morphology-specific adapters against an exact shared adapter formed
  from the mean of the same parameter tables and against scale zero. All paths
  own and execute identical adapter tensors; only conditioning differs. This
  also tests the queued LoRA-inside-Mamba idea under the now-promising
  coordinate model rather than freezing a weak teacher.

### Iteration 138 - 2026-07-12

- Regime: seed 20260765, 120k presentations, 3,750 updates. Scale-zero,
  morphology-specific, and shared rank-4 adapters each have 81,765 parameters,
  identical initial outputs, and the same adapter tables/compute.
- Morphology-specific adapters significantly improve F1 area 0.01136, exact
  F1 0.00993, and 5 ms F1 0.01169, but significantly regress voltage by
  0.01163 mV. Dynamics improves 0.00448 mV/ms uncertainly and response regresses
  0.00077 mV uncertainly.
- Shared adapters regress voltage 0.02327 mV and response 0.00950 mV
  significantly while improving F1 area/5 ms. Specific conditioning is better
  than generic adapter capacity, but neither yields a balanced frontier.
- Decision: reject without rank/scale tuning. Do not use adapter-only training;
  the adapters do not improve the fully trained model's balanced objective.
- Compute: three-model panel wall 171.87 seconds; per-model compute 27.12-28.87
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-138-morphology-mamba-adapters/`.

### Iteration 139 premise - coordinate mechanism ablation

- Full coordinate conditioning has now improved all six core metrics
  significantly versus exact controls on seeds 20260758, 20260759, and
  20260760. Its learned gain correlated most strongly with route occupancy,
  multiplicity, and target length, but hard validity masking failed.
- Keep the exact `9 -> 16 -> 1` path and compare full features against
  occupancy/multiplicity only, continuous source/target geometry only,
  validity only, and within-type shuffled full features. This identifies which
  feature family deserves I140's 300k gate without changing parameters or
  compute.
- Artifact: `artifacts/routes/v1-coordinate-feature-ablation.npz`, SHA-256
  `bb6ac5cb50c97df6dd10062824e25dd40ae981fa6c21b20465e4a5a6c6924b88`.
  Manifest: `manifests/iteration-139-coordinate-feature-ablation.json`.

### Iteration 139 - 2026-07-12

- Regime: seed 20260766, 120k presentations, 3,750 updates. All five arms use
  the same `9 -> 16 -> 1` operator, 77,125 parameters, initialization,
  schedule, and compute; only the frozen feature tensor differs.
- Occupancy/multiplicity reproduces the full feature model: voltage differs by
  +0.00179 mV, dynamics by +0.00109 mV/ms, F1 area by -0.00028, exact F1 by
  +0.00169, and 5 ms F1 by +0.00331, all uncertain. Response improves 0.00481
  mV significantly. Its absolute metrics are voltage 17.10682 mV, response
  17.57973 mV, F1 area 0.28111, exact F1 0.12500, and 5 ms F1 0.35056.
- Continuous geometry is almost identical to shuffled full features. Versus
  full, geometry regresses voltage 0.18671 mV, response 0.01797 mV, F1 area
  0.08327, exact F1 0.04760, and 5 ms F1 0.08640 significantly. Shuffling
  produces the same pattern: +0.18412 mV voltage and -0.08337 F1 area.
- Validity alone retains much of the effect but significantly loses 0.01145 mV
  voltage and 0.00715 F1 area versus full. The soft learned interaction of
  occupancy, multiplicity, and target length is therefore more informative
  than a binary validity flag; metric cable coordinates add no detectable
  value in this operator.
- Decision: carry both compact occupancy and full features into I140's 300k
  gate, with a newly matched shuffled-occupancy tensor and a scale-zero arm.
  Prefer compact occupancy if it remains equivalent because it gives a sharper
  biological claim and avoids irrelevant geometry inputs.
- Compute: five-model panel wall 220.69 seconds; per-model compute 26.37-27.58
  seconds. Artifacts are under
  `artifacts/diagnostics/iteration-139-coordinate-feature-ablation/`.

### Iteration 140 premise - long route-occupancy gate

- Test whether the robust 120k route effect survives 300k presentations and
  whether real route identity, rather than operator capacity, causes it.
  Compare real occupancy, full features, within-type shuffled occupancy, and
  real occupancy at scale zero. Every arm owns the same MLP and 77,125
  parameters and uses seed 20260767 with the frozen balanced schedule.
- Require a balanced real-occupancy improvement over both scale-zero and
  shuffled controls, no new disqualifying diagnostics, and no material loss
  versus full features before promotion.
- Artifact: `artifacts/routes/v1-i140-feature-gate.npz`, SHA-256
  `621efb764f427b998d0b775a4e5cc961a2a9d4144eae5369c058e11bf9b3beda`.
  Manifest: `manifests/iteration-140-route-occupancy-long-gate.json`.

### Iteration 140 maintenance - 2026-07-12

- Removed executable config, buffers, modules, forward branches, and tests for
  rejected I132-I138 hard validity, tree event propagation, morphology-specific
  physical routing, event/change decoder, voltage-state decoder, and
  morphology-conditioned Mamba adapter experiments. The route reconstruction
  utility now emits only the accepted feature/operator artifacts.
- Historical manifests, checkpoints, and diagnostics remain. Removed config
  names are explicitly retired in checkpoint compatibility rather than kept as
  dormant dataclass fields.
- Verification: AxoSim 100/100 passed; AxoBench 131 passed and 1 skipped; stable
  AxoMamba checkpoint loads as 76,949 parameters; Python compilation and
  `git diff --check` pass.
- Literature scan: `literature/iteration-140-dual-path-dynamics-scan.md` screens
  2024-2026 dual-path neural operators, local/global dynamics, state-space
  operators, graph-integrator simulators, predictor-correctors, history-state
  operators, and structured dendritic sampling. It ranks a generic-global plus
  physical-local residual route first, with exact capacity and shuffled-route
  controls and a two-seed stopping rule.

### Iteration 140 - 2026-07-12

- Regime: scheduled 300k gate, seed 20260767, 9,375 updates, four candidates,
  77,125 parameters each, common initialization/schedule/objective and the same
  feature-MLP compute. Panel wall time was 487.52 seconds; candidate model
  compute was 66.60-69.48 seconds.
- Real route occupancy reaches voltage SERA 16.91837 mV, dynamics SERA
  21.25254 mV/ms, response SERA 17.48370 mV, F1 area 0.35302, exact F1
  0.18678, and 5 ms F1 0.42604.
- Versus exact scale zero, real occupancy significantly improves all six core
  metrics: voltage -0.19773 mV, dynamics -0.04494 mV/ms, response -0.06372 mV,
  F1 area +0.07157, exact F1 +0.06333, and 5 ms F1 +0.06571.
- Versus within-type shuffled occupancy, real occupancy also significantly
  improves all six: voltage -0.16573 mV, dynamics -0.04419 mV/ms, response
  -0.06709 mV, F1 area +0.07496, exact F1 +0.06474, and 5 ms F1 +0.07569.
  The benefit is route identity rather than feature-operator capacity.
- Full nine-feature routing is statistically equivalent on five metrics but
  significantly regresses voltage 0.00839 mV versus compact occupancy.
  Continuous coordinates therefore add no useful signal beyond the discrete
  occupancy/multiplicity/target-length mechanism.
- Decision: promote compact occupancy. The deployed config/default now owns the
  `9 -> 16 -> 1` operator, and the exact frozen occupancy tensor is embedded in
  the checkpoint. Candidate-to-deployment output identity is exact
  (`max_abs_diff=0.0`); deployed checkpoint has 77,125 parameters and SHA-256
  `af315706c3b50e1a133cfa6835fe2221cfc934e199023c7d98abaf1561908aee`.
- Artifacts: `artifacts/diagnostics/iteration-140-route-occupancy-long-gate/`
  and `manifests/iteration-140-route-occupancy-long-gate.json`.

### Iteration 141 premise - dual-route residual fusion

- I135 replaced generic routing with physical target-segment routing and was
  seed-unstable. The I140 literature scan instead predicts complementary local
  and global paths: preserve the promoted generic route exactly and add a
  separately projected physical-route residual before the Mamba trunk.
- A shared `118 -> 32 -> 58` route projection adds 5,664 parameters and has a
  zero-initialized output. Compare scale zero, a duplicate generic route
  (capacity control), real physical routing, and within-type shuffled physical
  routing. Every arm has 82,789 parameters, executes the same gather/projection,
  starts output-identical, and retains promoted route occupancy on the global
  path.
- Pass only if real physical routing beats both duplicate-generic capacity and
  shuffled-route controls materially, ideally by at least 0.10 mV voltage or a
  comparably large balanced multi-axis gain. Do not tune width or scale first.
- Manifest: `manifests/iteration-141-dual-route-residual.json`.

### Iteration 141 - 2026-07-12

- Regime: seed 20260768, 120k presentations, 3,750 updates. Scale-zero,
  duplicate-generic, real-physical, and shuffled-physical residuals each have
  82,789 parameters, output-identical initialization, and matched route gather
  and projection compute. Panel wall time was 199.40 seconds; candidate model
  compute was 27.75-28.58 seconds.
- Real physical routing versus scale zero significantly regresses voltage
  0.03851 mV while improving F1 area 0.01083 and 5 ms F1 0.02001. Dynamics
  improves 0.00062 mV/ms and response regresses 0.00172 mV uncertainly; exact
  F1 rises only 0.00113 uncertainly.
- Duplicate generic routing improves F1 area 0.02328, exact F1 0.01657, and
  5 ms F1 0.02246 significantly, but regresses voltage 0.00716 mV and dynamics
  0.00476 mV/ms significantly. Real physical routing is worse than this
  capacity control by 0.03135 mV voltage, 0.01245 F1 area, and 0.01545 exact
  F1 on point estimates.
- Shuffled physical routing is nearly inert except for a 0.00339 mV response
  improvement and 0.00556 mV voltage regression. Real route structure changes
  behavior, but not toward the balanced objective.
- Decision: reject the dual-route family without width/scale tuning. The
  predeclared physical-over-generic and practical-effect criteria fail. Keep
  the I140 deployment unchanged and move to forcing/state decomposition.
- Artifacts: `artifacts/diagnostics/iteration-141-dual-route-residual/` and
  `manifests/iteration-141-dual-route-residual.json`.

### Iteration 142 premise - forcing/state decomposition before Mamba

- I140's residual lag, low voltage gain, high-dV/dt error, and near-spike error
  remain active. Graph-integrator and stochastic-operator work separates
  instantaneous forcing from deterministic evolution. I134 injected event
  change only at the output and was unstable; test whether explicit branch
  change is useful when Mamba can evolve it jointly with state.
- Add a zero-initialized `118 -> 32 -> 58` forcing residual before the first
  Mamba block. Compare scale zero, duplicate branch state, causal signed branch
  change, and change plus state-change interaction. Every arm computes all
  source forms, has 82,789 parameters, and begins output-identical.
- The change path must beat the state-capacity control on dynamics plus at least
  one of voltage, response, or tolerant F1. Require a practical balanced effect;
  do not tune width/scale after a failed source comparison.
- Manifest: `manifests/iteration-142-branch-forcing-decomposition.json`.

### Iteration 142 - 2026-07-12

- Regime: seed 20260769, 120k presentations, 3,750 updates. Scale-zero, state,
  signed-change, and change-interaction paths each have 82,789 parameters,
  output-identical initialization, and matched source/projection compute. Panel
  wall time was 198.23 seconds; model compute was 26.91-28.16 seconds.
- Change interaction versus scale zero significantly improves voltage 0.06131
  mV and response 0.02717 mV. Dynamics improves 0.00149 mV/ms, F1 area 0.00319,
  and 5 ms F1 0.00696 uncertainly; exact F1 falls 0.00867 uncertainly. Absolute
  metrics are voltage 17.00215 mV, response 17.54158 mV, dynamics 21.29708
  mV/ms, F1 area 0.27465, exact F1 0.11576, and 5 ms F1 0.35102.
- Plain state residual significantly regresses voltage 0.01705 mV, response
  0.01957 mV, and exact F1 0.01283. Plain signed change is worse: voltage
  +0.02606 mV, response +0.01610 mV, and exact F1 -0.01633, despite a 0.01813
  5 ms F1 gain. The useful mechanism is nonlinear state-change interaction,
  not residual capacity or derivatives alone.
- Direct point differences versus state capacity favor interaction by 0.07836
  mV voltage, 0.00646 mV/ms dynamics, 0.04674 mV response, 0.00225 F1 area,
  and 0.00416 exact F1; 5 ms F1 is 0.00554 lower. This clears the source-
  specificity premise but remains below the desired 0.10 mV practical target.
- Decision: replicate the exact four-arm panel on seed 20260770 before
  extending the interaction. No width or scale tuning.
- Artifacts: `artifacts/diagnostics/iteration-142-branch-forcing-decomposition/`
  and `manifests/iteration-142-branch-forcing-decomposition.json`.

### Iteration 143 premise - branch-forcing replication

- Repeat I142 unchanged on seed 20260770. Require change interaction to retain
  balanced voltage/response gains over both scale-zero and state-capacity
  controls; plain change should remain inferior. A reversal closes the family,
  while replication permits a structural interaction extension.
- Manifest: `manifests/iteration-143-branch-forcing-replication.json`.

### Iteration 143 - 2026-07-12

- Regime: unchanged I142 panel on seed 20260770, 120k presentations, 3,750
  updates, 82,789 parameters per candidate, and output-identical initialization.
  Panel wall time was 198.65 seconds; model compute was 26.99-28.38 seconds.
- Change interaction does not replicate voltage: it regresses voltage 0.01411
  mV significantly, while improving dynamics 0.00638 mV/ms uncertainly,
  response 0.00827 mV significantly, and F1 area/exact/5 ms by 0.01455,
  0.01207, and 0.01703 significantly. It is an event-favorable tradeoff.
- Plain signed change reverses its I142 result and significantly improves all
  SERA metrics: voltage -0.05212 mV, dynamics -0.00859 mV/ms, and response
  -0.01609 mV. Exact F1 rises 0.01126 uncertainly; F1 area and 5 ms fall only
  0.00338 and 0.00602 uncertainly. State capacity again regresses response.
- Interpretation: explicit change forcing has value on both seeds, but the
  winning hand-crafted basis alternates. I142 interaction repaired SERA while
  plain change failed; I143 plain change repairs SERA while interaction moves
  event metrics. Treat this as a basis-selection failure rather than a robust
  interaction replication.
- Decision: do not promote either source. I144 gives one projection both signed
  change and state-change interaction, with duplicated single-source controls
  so any gain must exceed doubled capacity for either basis alone.
- Artifacts: `artifacts/diagnostics/iteration-143-branch-forcing-replication/`
  and `manifests/iteration-143-branch-forcing-replication.json`.

### Iteration 144 premise - learned joint forcing basis

- I142 and I143 alternate between interaction and plain change as the SERA
  winner, while the other basis carries event value. Let one projection
  learn both `[delta_b, b * delta_b]` instead of selecting a basis by hand.
- Compare joint features against duplicated-change, duplicated-interaction,
  and joint scale-zero controls. Every arm has a `236 -> 32 -> 58` path,
  86,565 parameters, identical active input width/compute, and output-identical
  initialization. Joint forcing must beat both duplicate controls on a
  balanced frontier and materially exceed the seed-specific single-basis gain.
- Manifest: `manifests/iteration-144-joint-forcing-basis.json`.

### Iteration 144 - 2026-07-12

- Regime: seed 20260771, 120k presentations, 3,750 updates. Joint, duplicate-
  change, duplicate-interaction, and scale-zero arms each expose 236 forcing
  channels to the same `236 -> 32 -> 58` path, have 86,565 parameters, and
  begin output-identical. Panel wall time was 199.71 seconds; model compute was
  27.43-28.79 seconds.
- Joint forcing significantly improves dynamics 0.01130 mV/ms and F1 area,
  exact F1, and 5 ms F1 by 0.02694, 0.02677, and 0.04001, but significantly
  regresses voltage 0.01267 mV and response 0.00618 mV. This is a clear
  SERA/event frontier, not the intended balanced synthesis.
- Duplicate interaction has the same failure direction: voltage +0.01867 mV
  and response +0.02068 mV, despite F1 gains. Duplicate change is balanced on
  this seed, significantly improving voltage 0.02591 mV, dynamics 0.01687
  mV/ms, and response 0.00732 mV; all F1 point estimates also improve.
- Interpretation: plain change has now helped on I143-I144 but failed on I142,
  while interaction helped I142 and failed voltage on I143-I144. Joint access
  does not stabilize the tradeoff. Duplicate change's 0.026 mV gain is too
  small to justify more source/width optimization under the aggressive target.
- Decision: reject the joint basis and close forcing-basis refinement. Keep the
  I140 deployment. Move to a predictor-corrector transition between existing
  Mamba blocks, which changes state evolution rather than input basis width.
- Artifacts: `artifacts/diagnostics/iteration-144-joint-forcing-basis/` and
  `manifests/iteration-144-joint-forcing-basis.json`.

### Iteration 145 premise - error-driven predictor-corrector transition

- CoEvol-NO motivates a persistent predictor-corrector state rather than an
  additive output residual or repeated trunk. Between AxoMamba blocks 1 and
  2, project the first-block hidden state and current branch forcing into a
  shared 16-dimensional coordinate and correct hidden state from their error.
- Compare error-driven correction against state-only, forcing-only, and
  scale-zero controls. Every arm computes both predictors and the same
  `16 -> 16 -> 58` correction, has 81,289 parameters, and starts output-
  identical. Require error feedback to beat both single-source controls
  materially on voltage/response and preserve F1; otherwise reject without
  predictor dimension or correction-scale tuning.
- Manifest: `manifests/iteration-145-predictor-corrector-transition.json`.

### Iteration 145 - 2026-07-12

- Regime: seed 20260772, 120k presentations, 3,750 updates. Error, state,
  forcing, and scale-zero transitions each compute both 16-dimensional
  predictors and the same corrector, have 81,289 parameters, and start output-
  identical. Panel wall time was 201.19 seconds; model compute was 27.63-29.39
  seconds.
- Error-driven correction significantly improves F1 area 0.02033, exact F1
  0.02384, and 5 ms F1 0.02068; dynamics improves 0.00740 mV/ms uncertainly.
  It regresses response 0.00894 mV significantly and voltage 0.00495 mV
  uncertainly, failing the balanced and source-specific gate.
- State-only feedback similarly improves dynamics 0.01374 mV/ms and exact F1
  0.01606 significantly but regresses response 0.00993 mV. Forcing-only is the
  only balanced arm, with favorable point estimates on all six axes, but its
  voltage gain is only 0.00576 mV and all other changes are about 0.003 or less.
- Decision: reject predictor-corrector transition without dimension/scale
  tuning. It mostly shifts the SERA/event frontier and does not approach the
  aggressive voltage target. I146 exploits route occupancy more structurally
  by preserving used and unused source streams rather than hard masking either.
- Artifacts: `artifacts/diagnostics/iteration-145-predictor-corrector-transition/`
  and `manifests/iteration-145-predictor-corrector-transition.json`.

### Iteration 146 premise - soft used/unused route partition

- I139-I140 prove route occupancy is the largest recent balanced mechanism,
  while I133 proves deleting unused sources is harmful. Preserve the promoted
  total-current path and add a residual that separately projects currents from
  morphology-used and morphology-unused sources before Mamba.
- Compare real partition, within-type shuffled partition, duplicate total-
  current capacity, and scale zero. Every arm computes used/unused currents
  and the same `236 -> 32 -> 58` projection, has 86,565 parameters, retains
  every event in the main path, and starts output-identical. Require real
  partition to beat both duplicate-total and shuffled controls materially,
  targeting at least 0.10 mV voltage or an equally large balanced gain.
- Validity artifact SHA-256:
  `d0ac692bc9e84fc7a6c5888147ea7836c6af8b5a7a5ce921523b3c615f706514`.
- Manifest: `manifests/iteration-146-route-partition-residual.json`.

### Iteration 146 - 2026-07-12

- Regime: seed 20260773, 120k presentations, 3,750 updates. Real, shuffled,
  duplicate-total, and scale-zero arms each compute used/unused currents and
  the same `236 -> 32 -> 58` residual, have 86,565 parameters, preserve the
  promoted main path, and start output-identical. Panel wall time was 208.25
  seconds; model compute was 29.94-30.87 seconds.
- Real partition improves every core point estimate versus scale zero: voltage
  -0.01737 mV, dynamics -0.01025 mV/ms, response -0.02213 mV, F1 area
  +0.01027, exact F1 +0.01129, and 5 ms F1 +0.01212. Voltage, dynamics,
  response, and exact F1 are significant; tolerant F1 gains are uncertain.
- Shuffled partition loses F1 area 0.01524 and 5 ms F1 0.01867 significantly
  versus zero. Real partition beats shuffled by 0.01343 mV voltage, 0.00799
  mV/ms dynamics, 0.01765 mV response, 0.02551 F1 area, 0.00842 exact F1,
  and 0.03079 at 5 ms on point estimates. Biological source identity matters.
- Duplicate total improves voltage 0.01528 mV but loses F1 area 0.03664, exact
  F1 0.01258, and 5 ms F1 0.03069 significantly. Real partition dominates it
  on response and every event metric, but is 0.00210 mV worse on voltage.
- Decision: the mechanism is valid and balanced but its 0.017 mV voltage gain
  misses the predeclared aggressive bar. Do not replicate the binary split.
  I147 projects route occupancy, multiplicity, and target length as separate
  branch-current moments, testing the complete feature family isolated by I139.
- Artifacts: `artifacts/diagnostics/iteration-146-route-partition-residual/` and
  `manifests/iteration-146-route-partition-residual.json`.

### Iteration 147 premise - route-conditioned branch moments

- Binary route partition is real but small. I139 identified target occupancy,
  multiplicity, and target length as the full useful feature family. Compute
  branch-current moments weighted by each feature, concatenate them with total
  current, and project the resulting 472-channel field before Mamba.
- Compare real moments against shuffled moments, four duplicated total-current
  fields, and scale zero. Every arm computes the same `472 -> 32 -> 58` path,
  has 94,117 parameters, preserves the promoted route-gain/main paths, and
  starts output-identical. Require real moments to beat both controls with at
  least 0.10 mV voltage or an equally substantial balanced gain.
- Artifact: `artifacts/routes/v1-i147-route-moments.npz`, SHA-256
  `a22a2464432c99cdc4a03c8e20151300b225c1a8756a02b080ccdeaaad0d744d`.
- Manifest: `manifests/iteration-147-route-moment-residual.json`.

### Iteration 147 - 2026-07-12

- Regime: seed 20260774, 120k presentations, 3,750 updates. Real, shuffled,
  duplicate-total, and scale-zero candidates each compute a `472 -> 32 -> 58`
  route-moment residual, have 94,117 parameters, and start output-identical.
  Panel wall time was 212.93 seconds; model compute was 29.70-32.57 seconds.
- Real moments improve dynamics 0.00972 mV/ms significantly, but regress
  voltage 0.00870 mV and response 0.01177 mV significantly. F1 area and 5 ms
  rise 0.00355 and 0.00587 uncertainly; exact F1 falls 0.00370 uncertainly.
- Shuffled moments have almost the same SERA behavior: dynamics -0.00562
  mV/ms, voltage +0.01015 mV, and response +0.01169 mV versus zero. Direct real
  versus shuffled differences are practically negligible on voltage/response
  and mixed on events. Duplicate total also regresses response 0.01248 mV.
- Interpretation: static per-synapse gains exploit source identity, but
  aggregating feature-weighted currents into branch moments loses that useful
  correspondence. More route features do not rescue the residual operator.
- Decision: reject and close route partition/moment residuals. Retain only the
  I140 static occupancy operator. I148 targets the persistent morphology error
  disparity with hidden-dependent readout experts rather than another affine
  or branch-gain calibration.
- Artifacts: `artifacts/diagnostics/iteration-147-route-moment-residual/` and
  `manifests/iteration-147-route-moment-residual.json`.

### Iteration 148 premise - morphology-specific nonlinear readout experts

- Worst-morphology voltage RMSE remains about 1.44-1.46x the median. Earlier
  morphology affine calibration and branch gains failed, but no experiment
  has allowed the full temporal hidden representation to map differently to
  spike and soma outputs for each morphology.
- Add one zero-initialized residual linear readout per morphology before the
  existing voltage-to-spike coupling. Compare specific experts against the
  mean shared table, cyclically wrong morphology assignment, and scale zero.
  Every arm owns the same 590 expert parameters, has 77,715 total parameters,
  executes the same hidden-dependent correction, and starts output-identical.
- Require specific experts to beat shared and shuffled controls materially
  on voltage/response while reducing morphology disparity without F1 loss.
- Manifest: `manifests/iteration-148-morphology-readout-experts.json`.

### Iteration 148 - 2026-07-12

- Regime: seed 20260775, 120k presentations, 3,750 updates. Specific, shared,
  cyclic-shuffled, and scale-zero arms own the same 590 residual-readout
  parameters, have 77,715 total parameters, and start output-identical. Panel
  wall time was 198.00 seconds; model compute was 27.01-28.43 seconds.
- Specific experts significantly regress voltage 0.10337 mV, response 0.01854
  mV, F1 area 0.02094, exact F1 0.01606, and 5 ms F1 0.01980; dynamics also
  regresses 0.00227 mV/ms uncertainly. This is a broad failure.
- Shared experts improve dynamics 0.00671 mV/ms but significantly regress
  voltage 0.04885 mV, response 0.01123 mV, F1 area 0.01118, and 5 ms F1
  0.01795. Generic residual-readout capacity is harmful as well.
- Validity note: because each candidate is trained end to end, cyclically
  relabeling the expert table is permutation-symmetric. Specific and shuffled
  arms therefore train to numerically identical models; the shuffled arm is
  not a valid assignment-identity control. The scale-zero and shared controls
  remain valid and reject the architecture independently, so no rerun is needed.
- Decision: reject hidden-dependent morphology readouts. I149 tests a more
  aggressive task-specific temporal-head architecture: two extra Mamba
  residual blocks specialize after the shared trunk, compared with averaged
  shared corrections and exact scale zero.
- Artifacts: `artifacts/diagnostics/iteration-148-morphology-readout-experts/`
  and `manifests/iteration-148-morphology-readout-experts.json`.

### Iteration 149 premise - task-specific temporal Mamba heads

- Repeated experiments expose a voltage-versus-event frontier, suggesting
  negative transfer from forcing spike and soma outputs through one final
  temporal representation. Add two zero-initialized Mamba residual blocks
  after the shared trunk, one available to each task-specific readout.
- Compare specialized spike/soma corrections against an averaged shared
  correction, soma-only correction, and scale zero. Every arm computes both
  extra Mamba blocks, has 137,329 parameters, and starts output-identical;
  scale zero exactly preserves I140 behavior under matched added compute.
- Require specialization to materially beat shared capacity on both SERA and
  F1, targeting at least 0.10 mV voltage or a comparably large balanced gain.
- Manifest: `manifests/iteration-149-task-specific-mamba-heads.json`.

### Iteration 149 - 2026-07-12

- Regime: seed 20260776, 120k presentations, 3,750 updates. Scale-zero,
  shared, specialized, and soma-only arms compute both added Mamba blocks,
  have 137,329 parameters, and start output-identical. Panel wall time was
  221.50 seconds; model compute was 33.44-33.52 seconds per arm.
- Specialized heads versus scale zero improve response SERA by 0.02294 mV,
  F1 area by 0.02911, exact F1 by 0.01263, and 5 ms F1 by 0.03110, but
  significantly regress voltage SERA by 0.01800 mV. The specialization premise
  therefore fails its balanced criterion.
- Shared temporal correction improves voltage by 0.05033 mV and F1
  area/exact/5 ms by 0.04295/0.02279/0.04312, all significant. Dynamics
  improves 0.00788 mV/ms uncertainly; response regresses 0.00337 mV
  uncertainly. Soma-only is balanced but improves voltage by only 0.01635 mV
  and leaves event metrics tied.
- Interpretation: the evidence supports additional generic temporal depth,
  not negative-transfer relief from task separation. Shared beats specialized
  on voltage by 0.06833 mV and all F1 summaries, while specialized beats shared
  on response by 0.02631 mV; this is a genuine output frontier rather than a
  universal win.
- Decision: reject specialization and soma-only. Advance shared depth to I150's
  fresh-seed 300k gate against exact scale zero, retaining specialized as the
  strongest response-oriented control. Require the shared arm to preserve a
  material balanced lead before promotion.
- Artifacts: `artifacts/diagnostics/iteration-149-task-specific-mamba-heads/`
  and `manifests/iteration-149-task-specific-mamba-heads.json`.

### Iteration 150 premise - long shared-temporal-depth gate

- Run scale-zero, shared, and specialized task-head modes for 300k
  presentations, 9,375 optimizer steps, and a fresh balanced schedule at seed
  20260777. Every arm has 137,329 parameters and executes both added blocks.
- Primary question: does I149's 0.050 mV voltage plus broad F1 lead grow or
  persist at the deployment budget without intervention-response regression?
  The specialized arm tests whether longer training resolves its voltage versus
  response frontier; it is not a substitute for the exact scale-zero control.
- Scheduled maintenance: while the GPU gate runs, prune all rejected I141-I149
  executable paths except task-head Mamba until the gate resolves, run full
  AxoSim/AxoBench verification, load the stable I140 checkpoint, and complete
  the iteration-150 structural literature scan.
- Manifest: `manifests/iteration-150-shared-temporal-depth-long-gate.json`.

### Iteration 150 maintenance - 2026-07-12

- Cleanup: removed executable config fields, validation, buffers, modules,
  forward paths, helpers, and tests for rejected I141-I148 morphology-route
  residuals, forcing/corrector paths, route partitions/moments, and morphology
  readout experts. Historical manifests, diagnostics, and negative results
  remain. Retired checkpoint keys preserve historical config loading.
- Verification: AxoSim passes 101/101, AxoBench passes 131 with 1 skip,
  `git diff --check` passes, and the deployed I140 checkpoint loads in the
  Mamba environment with 77,125 parameters and 300k presentation metadata.
- Literature: completed
  `literature/iteration-150-multiscale-hybrid-dynamics-scan.md`. It ranks a
  parameter-controlled fast/slow latent Mamba first if I150 confirms useful
  temporal depth, then trajectory-regime-gated dynamics and an explicit leaky
  membrane backbone. This queue excludes another static route residual or
  coefficient sweep.

### Iteration 150 - 2026-07-12

- Regime: fresh seed 20260777, 300k presentations, 9,375 updates. Scale-zero,
  shared, and specialized arms each have 137,329 parameters and execute both
  task-head Mamba blocks. Panel wall time was 472.66 seconds; model compute was
  83.68-86.23 seconds per arm.
- Shared depth versus exact scale zero significantly regresses voltage SERA by
  0.08268 mV and intervention-response SERA by 0.00903 mV. It significantly
  improves dynamics SERA by 0.02302 mV/ms and F1 area/exact/5 ms by
  0.02044/0.01380/0.02405. This is a disqualifying deployment tradeoff.
- Specialized depth also significantly regresses voltage by 0.04054 mV while
  improving dynamics by 0.00986 mV/ms, F1 area by 0.04304, and 5 ms F1 by
  0.05521. Response changes by only +0.00133 mV uncertainly. Longer training
  does not resolve the voltage/event frontier observed at I149.
- Decision: reject both task-head modes and keep I140 deployed. Finish cleanup
  by deleting task-head config, modules, forward logic, helper, and test, with
  retired checkpoint keys. Post-gate AxoSim passes 100/100 and stable I140 still
  loads at 77,125 parameters.
- Next: because generic temporal depth fails at 300k, do not run the scan's
  conditional fast/slow Mamba candidate. I151 tests the structurally different
  fallback: an explicit causal leaky membrane state driven by learned branch
  current, with a learned residual and exact generic-capacity control.
- Artifacts: `artifacts/diagnostics/iteration-150-shared-temporal-depth-long-gate/`
  and `manifests/iteration-150-shared-temporal-depth-long-gate.json`.

### Iteration 151 premise - explicit leaky membrane state

- Hypothesis: one shared Mamba state is forced to represent both the slowly
  integrated membrane trajectory and fast nonlinear events. A causal leaky
  state bank driven by learned branch current can provide the stable soma head
  with an explicit integration backbone, leaving Mamba to model the residual.
- Structure: project 118 routed branch currents into 16 bounded drive channels;
  integrate each with a learned decay and a 129-step strictly causal grouped
  convolution; decode a zero-initialized scalar soma correction before the
  existing voltage-to-spike coupling. Initial decay time constants span 2-64
  steps once, without a sweep.
- Controls: exact scale zero, active leaky state, and active instantaneous drive
  have 79,045 parameters, execute both drive and leaky convolution, share seed,
  schedule, optimizer, objective, and 120k presentations, and start
  output-identical. Instantaneous drive isolates temporal integration from the
  same added nonlinear capacity.
- Acceptance: require at least 0.05 mV voltage improvement or a comparably large
  balanced voltage/response/F1 gain over scale zero, and require leaky state to
  beat instantaneous capacity. Audit learned decays and high-dV/dt/near-spike
  diagnostics before attributing mechanism.
- Manifest: `manifests/iteration-151-leaky-membrane-state.json`.

### Iteration 151 - 2026-07-12

- Regime: seed 20260778, 120k presentations, 3,750 updates. Scale-zero, leaky,
  and instantaneous arms have 79,045 parameters, execute the same projection
  and causal convolution, and start output-identical. Panel wall time was
  171.72 seconds; model compute was 27.21-28.40 seconds.
- Leaky state significantly improves voltage by 0.02589 mV, dynamics by
  0.01491 mV/ms, response by 0.00850 mV, and exact F1 by 0.01813. F1 area
  improves 0.00691 uncertainly and 5 ms F1 is tied. The instantaneous decoder
  is essentially flat on voltage (-0.00020 mV) and significantly regresses
  response by 0.00111 mV, isolating temporal integration as the useful part.
- Mechanism audit: learned leaky time constants move from the initialized
  2.00-63.94 range to an active 2.94-54.00 range; decoder norm reaches 0.341 and
  drive norm 3.879. Voltage gain rises 0.00457, residual lag falls 0.00506, and
  intervention gain rises 0.00213. High-dV/dt and near-spike error ratios
  worsen slightly, so the gain is smoother trajectory integration rather than
  fast-event repair.
- Decision: mechanism-positive but below the 0.05 mV carry bar. Do not tune
  dimension, kernel, scale, or initial taus. I152 makes one structural
  expansion grounded in dendritic temporal heterogeneity: integrate each branch
  before mixing, against mixed-state and shared-branch-decay controls.
- Artifacts: `artifacts/diagnostics/iteration-151-leaky-membrane-state/` and
  `manifests/iteration-151-leaky-membrane-state.json`.

### Iteration 152 premise - branchwise membrane integration

- Preserve each of 118 routed branch currents through its own causal leaky
  state before the learned 118-to-16 drive projection. This changes the order
  from mix-then-integrate to integrate-then-mix and gives dendritic sources
  distinct temporal responses.
- Compare heterogeneous branch decays with a single shared effective branch
  decay, I151's mixed latent-state mechanism, and exact scale zero. Every arm
  owns all mixed and branchwise decays, executes both convolution banks, has
  identical parameters/compute, and starts output-identical.
- Require branchwise heterogeneous state to beat mixed and shared controls by
  at least 0.05 mV voltage or a comparably strong balanced core gain. If it
  does not, close the explicit membrane-state family rather than tune taus.
- Manifest: `manifests/iteration-152-branchwise-membrane-state.json`.

### Iteration 152 - 2026-07-12

- Regime: seed 20260779, 120k presentations, 3,750 updates. Scale-zero, mixed,
  branchwise shared-decay, and branchwise heterogeneous-decay arms each have
  79,163 parameters, execute both convolution banks, and start output-identical.
- Heterogeneous branch timing significantly regresses voltage by 0.02172 mV
  and improves dynamics only uncertainly by -0.00118 mV/ms. It significantly
  improves response by 0.00489 mV and F1 area/exact/5 ms by
  0.01746/0.01713/0.01951, reproducing the familiar voltage/event frontier.
- Shared branch decay regresses voltage by 0.02698 mV and dynamics by 0.00426
  mV/ms while improving response by 0.01439 mV. Mixed latent integration also
  regresses voltage by 0.01312 mV despite response and F1 gains. Thus I151's
  small balanced lead does not replicate in this matched expanded panel.
- Decision: reject branchwise, shared, and mixed states; close the explicit
  membrane-state family without dimension, kernel, tau, or scale tuning. Keep
  code only until I160 cleanup. I153 tests learned trajectory regimes between
  temporal layers, the next distinct architecture ranked by the I150 scan.
- Artifacts: `artifacts/diagnostics/iteration-152-branchwise-membrane-state/`
  and `manifests/iteration-152-branchwise-membrane-state.json`.

### Iteration 153 premise - hybrid regime transition

- Neural hybrid-system work suggests that one continuous flow is a poor model
  for trajectories with event-triggered transitions. Add two zero-initialized
  residual transition experts between AxoMamba's first and second Mamba layers.
  A causal gate sees normalized hidden state plus current branch change and
  selects between subthreshold and event-transition updates.
- Compare learned gating with a fixed equal expert mixture and exact scale zero.
  Every arm owns and computes both experts and the learned gate, starts
  output-identical, and has matched parameters, updates, presentations, and
  schedule. Audit gate occupancy and association with high branch change.
- Require a material balanced SERA/F1 gain over both controls. A voltage/event
  tradeoff or constant gate closes regime experts without gate tuning.
- Manifest: `manifests/iteration-153-hybrid-regime-transition.json`.

### Iteration 153 - 2026-07-12

- Regime: seed 20260780, 120k presentations, 3,750 updates. Scale-zero, fixed
  expert mixture, and learned gate arms each have 83,834 parameters, compute
  both experts and the gate, and start output-identical.
- Learned gating improves response by 0.01539 mV and exact F1 by 0.01282
  significantly, but regresses voltage by 0.00311 mV and dynamics by 0.00483
  mV/ms uncertainly; F1 area/5 ms gains are uncertain. The fixed mixture has a
  better voltage point estimate (-0.00684 mV) with similar response/exact-F1
  gains, so gating does not beat generic transition capacity.
- Gate audit over 128,000 held-out time points shows mean 0.601, standard
  deviation 0.216, range 0.032-1.000, branch-change correlation 0.081, and
  occupancy 0.570/0.631 in low/high-change deciles. The gate is active and
  weakly regime-associated; its failure is not constant routing.
- Decision: reject learned and shared regime experts without hidden-size,
  scale, placement, or gate tuning. Keep code until I160 cleanup. I154 changes
  training structure rather than adding another residual: compositional
  voltage-flow and event-timing pretraining inside the fixed presentation
  budget, motivated by Hocker et al., Nature Machine Intelligence 2025.
- Artifacts: `artifacts/diagnostics/iteration-153-hybrid-regime-transition/`
  and `manifests/iteration-153-hybrid-regime-transition.json`.

### Iteration 154 premise - compositional objective curriculum

- Complex recurrent tasks can train more efficiently when relevant
  subcomputations are learned before the full task. AxoMamba's persistent
  voltage/event frontier supplies two explicit subtasks: continuous voltage
  flow and sparse event timing.
- Compare unchanged joint training with voltage-first then joint, event-first
  then joint, and voltage-then-event-then-joint curricula. Every arm uses the
  same deployed I140 architecture, frozen schedule, 120k presentations, 3,750
  optimizer steps, learning-rate schedule, and computes all loss components;
  only the active weights change by declared phase.
- This is a sample-efficiency premise, not a coefficient sweep. Require a
  material balanced SERA/F1 gain over uninterrupted joint training; otherwise
  reject curricula without phase-length or weight tuning.
- Manifest: `manifests/iteration-154-compositional-objective-curriculum.json`.

### Iteration 154 - 2026-07-12

- Regime: seed 20260781, deployed 77,125-parameter I140 architecture, 120k
  presentations, and 3,750 steps. Joint-throughout, voltage-first, event-first,
  and voltage-event-joint arms share schedule, optimizer, LR schedule, and all
  model compute. Exact phase objectives are serialized in checkpoints.
- Event-first regresses voltage by 0.15496 mV, dynamics by 0.01543 mV/ms,
  response by 0.01932 mV, F1 area by 0.03706, exact F1 by 0.02375, and 5 ms F1
  by 0.03949, all significant. The three-stage curriculum is worse still.
- Voltage-first improves F1 area/exact/5 ms by 0.02436/0.01079/0.03046, but
  significantly regresses voltage/dynamics/response by
  0.09259/0.01305/0.01243. It amplifies rather than resolves task interference.
- Decision: reject curricula without phase-length, ordering, or weight tuning.
  Retain schedule machinery only until I160 cleanup. I155 returns to a
  high-temperature architecture: a causal multiscale local TCN operator in
  parallel with global Mamba, motivated by MambaNO and the original NeuronIO
  TCN evidence.
- Artifacts: `artifacts/diagnostics/iteration-154-compositional-objective-curriculum/`
  and `manifests/iteration-154-compositional-objective-curriculum.json`.

### Iteration 155 premise - parallel local/global temporal operator

- Mamba supplies global selective state evolution, but persistent high-dV/dt
  and near-spike errors may require an explicit local multiscale operator. Run
  a three-block causal depthwise-separable TCN with dilations 1/4/16 from the
  encoded branch stream in parallel with the unchanged Mamba trunk, then fuse
  through a zero-initialized projection before the stable heads.
- Compare dilated TCN with three undilated blocks and exact scale zero. All arms
  own identical convolution/pointwise/projection parameters, execute three
  blocks, share initialization and training, and start output-identical.
- Require at least 0.05 mV voltage gain or a comparably strong balanced core
  gain, with dilated structure materially beating undilated capacity. Audit
  high-dV/dt and near-spike ratios before attributing a local-operator effect.
- Manifest: `manifests/iteration-155-local-global-tcn-operator.json`.

### Iteration 155 - 2026-07-12

- Regime: seed 20260782, 120k presentations, 3,750 updates. Scale-zero,
  dilated 1/4/16, and undilated 1/1/1 paths each have 91,451 parameters,
  execute three depthwise-separable TCN blocks, and start output-identical.
  Panel wall time was 174.85 seconds; model compute was about 28.27 seconds.
- Dilated local/global TCN significantly improves voltage by 0.09780 mV,
  response by 0.03385 mV, and F1 area/exact/5 ms by
  0.02560/0.01712/0.03032. Dynamics regresses 0.00411 mV/ms uncertainly. This
  is the first post-I140 short candidate to approach the declared 0.10 mV bar
  while improving both intervention and event metrics.
- Undilated TCN also significantly improves voltage by 0.08918 mV, response by
  0.02409 mV, and all F1 summaries. Dilated versus undilated point differences
  are -0.00863 mV voltage, -0.00975 mV response, and +0.00125 to +0.00497 F1;
  most benefit is a full local operator, with a smaller multiscale increment.
- Mechanism audit: dilated TCN raises voltage gain 0.29778 to 0.31851 and lowers
  residual lag 0.30308 to 0.29283. High-dV/dt and near-spike error ratios rise
  0.01366/0.01818 and morphology disparity rises 0.00989, so it improves broad
  trajectory fit rather than the AP core.
- Decision: replicate scale-zero, dilated, and undilated unchanged on I156 seed
  20260783. Require the dilated arm to retain a material balanced lead and at
  least match undilated capacity before planning the I160 long gate.
- Artifacts: `artifacts/diagnostics/iteration-155-local-global-tcn-operator/`
  and `manifests/iteration-155-local-global-tcn-operator.json`.

### Iteration 156 premise - local/global TCN replication

- Repeat I155's exact architecture, controls, objective, and 120k budget on a
  fresh frozen schedule. No dilation, scale, width, kernel, or optimizer change.
- Replication must preserve a balanced material lead over scale zero. The
  dilated arm must also match or beat undilated local capacity on the primary
  SERA axes to support a multiscale claim.
- Manifest: `manifests/iteration-156-local-global-tcn-replication.json`.

### Iteration 156 - 2026-07-12

- Regime: exact I155 replication on seed 20260783, 120k presentations, 3,750
  updates, and 91,451 matched parameters for scale-zero, dilated, and
  undilated paths.
- Dilated TCN significantly improves voltage by 0.03850 mV, dynamics by
  0.00397 mV/ms, response by 0.05729 mV, F1 area by 0.03594, exact F1 by
  0.02646, and 5 ms F1 by 0.02660. The point estimates are favorable on all
  six core metrics; the evaluator labels dynamics uncertain.
- Undilated TCN is also balanced but weaker on voltage/response/dynamics:
  -0.02587/+0.00005/-0.04988 versus dilated -0.03850/-0.00397/-0.05729.
  Dilated also leads F1 area and 5 ms F1, while exact F1 differs by -0.00054.
- Two-seed dilated means versus scale zero are voltage -0.06815 mV, dynamics
  +0.00007 mV/ms, response -0.04557 mV, F1 area +0.03077, exact +0.02179,
  and 5 ms +0.02846. This is replicated balanced evidence, although still
  short of a large fraction of 20 mV.
- Decision: lock local TCN for an I160 long gate. I157 tests only structural
  fusion placement under exact parameter and path-count matching; no dilation,
  width, scale, kernel, objective, or optimizer tuning.
- Artifacts: `artifacts/diagnostics/iteration-156-local-global-tcn-replication/`
  and `manifests/iteration-156-local-global-tcn-replication.json`.

### Iteration 157 premise - local/global fusion placement

- Current parallel fusion computes local features from the encoded branch
  stream and adds them after the global Mamba trunk. Interlayer fusion instead
  lets the first Mamba state contextualize the local operator before layer two;
  post fusion applies local refinement directly to the completed global state.
- Compare parallel, interlayer, and post positions plus exact scale zero. Every
  arm uses the same three dilated blocks, zero-initialized projection, 91,451
  parameters, and one local-path execution per forward pass.
- Require a new position to materially beat replicated parallel fusion without
  a core regression. Otherwise retain parallel unchanged for I160.
- Manifest: `manifests/iteration-157-local-tcn-fusion-placement.json`.

### Iteration 157 - 2026-07-12

- Regime: seed 20260784, 120k presentations, 3,750 updates. Scale-zero,
  parallel, interlayer, and post positions each have 91,451 parameters, execute
  one identical dilated TCN, and start output-identical.
- Post fusion significantly improves voltage by 0.15320 mV, dynamics by
  0.03584 mV/ms, response by 0.07724 mV, F1 area by 0.04241, exact F1 by
  0.03705, and 5 ms F1 by 0.03835. This is the largest balanced short-run
  structural effect after I140 and exceeds the 0.10 mV practical bar.
- Interlayer fusion also improves all six metrics significantly: voltage
  -0.12806, dynamics -0.04015, response -0.04952, F1 area +0.04722, exact
  +0.02595, and 5 ms +0.04793. Parallel fusion independently replicates with
  voltage -0.07586 and all six significant improvements.
- Post versus interlayer favors voltage/response/exact by
  0.02514/0.02772/0.01110, while interlayer favors dynamics/F1 area/5 ms by
  0.00431/0.00481/0.00958. Post is provisional under the SERA-first policy,
  not yet a universal placement winner.
- Decision: repeat all four positions unchanged at I158 seed 20260785. Require
  post or interlayer to preserve a material balanced lead over parallel before
  selecting the I160 long candidate.
- Artifacts: `artifacts/diagnostics/iteration-157-local-tcn-fusion-placement/`
  and `manifests/iteration-157-local-tcn-fusion-placement.json`.

### Iteration 158 premise - local TCN placement replication

- Repeat scale-zero, parallel, interlayer, and post fusion unchanged on a fresh
  schedule. All architecture, initialization procedure, objective, compute,
  and 120k exposure controls remain fixed.
- Select a placement only if its balanced gain replicates and it materially
  improves the primary SERA axes over the twice-replicated parallel path.
- Manifest: `manifests/iteration-158-local-tcn-placement-replication.json`.

### Iteration 158 - 2026-07-12

- Regime: exact I157 placement replication on seed 20260785, 120k
  presentations, 3,750 updates, and four 91,451-parameter matched arms.
- Post fusion significantly improves voltage by 0.07419 mV, dynamics by
  0.01979 mV/ms, response by 0.04608 mV, F1 area by 0.01376, and exact F1 by
  0.01471. The 5 ms F1 gain is +0.00517 uncertain. All six point estimates are
  favorable, and the three SERA axes replicate strongly.
- Parallel fusion flips to a significant 0.02316 mV voltage regression with
  uncertain response/dynamics/F1 changes. Interlayer improves voltage by
  0.02142 mV but significantly regresses response by 0.01274 mV and 5 ms F1 by
  0.02098. Neither placement is robust enough for long gating.
- Two-seed post means versus scale zero are voltage -0.11369 mV, dynamics
  -0.02782 mV/ms, response -0.06166 mV, F1 area +0.02809, exact +0.02588, and
  5 ms +0.02176. Post is the first candidate in this phase with an average
  voltage gain above 0.10 mV and balanced gains across all primary axes.
- Decision: lock post placement. I159 compares dilated and undilated TCNs at
  post position under exact capacity matching, deciding the compact I160 long
  panel without tuning dilation values or kernels.
- Artifacts: `artifacts/diagnostics/iteration-158-local-tcn-placement-replication/`
  and `manifests/iteration-158-local-tcn-placement-replication.json`.

### Iteration 159 premise - post-TCN dilation attribution

- At parallel placement, dilation beat undilated capacity on voltage and
  response across I155-I156. Test whether that structural benefit transfers to
  the now-replicated post placement.
- Compare post-dilated, post-undilated, and exact scale zero with identical
  91,451 parameters, three local blocks, initialization, schedule, objective,
  and 120k presentations.
- Dilated post must retain a material balanced lead over zero and match or beat
  undilated post on primary SERA axes. Otherwise long-gate the simpler
  undilated post path or both controls as warranted.
- Manifest: `manifests/iteration-159-post-tcn-dilation-attribution.json`.

### Iteration 159 - 2026-07-12

- Regime: seed 20260786, 120k presentations, 3,750 updates. Scale-zero,
  post-dilated, and post-undilated arms each have 91,451 parameters, execute
  three TCN blocks, and start output-identical.
- Post-dilated significantly improves voltage by 0.12610 mV, dynamics by
  0.03857 mV/ms, response by 0.10343 mV, F1 area by 0.04256, exact F1 by
  0.03776, and 5 ms F1 by 0.02821.
- Post-undilated also improves all six significantly: voltage -0.11467,
  dynamics -0.04188, response -0.09987, F1 area +0.04944, exact +0.03295,
  and 5 ms +0.03899. Dilated leads voltage/response/exact by
  0.01142/0.00355/0.00482; undilated leads dynamics/F1 area/5 ms by
  0.00331/0.00687/0.01077.
- Interpretation: post fusion is robust; dilation changes the SERA/event point
  on a balanced frontier rather than dominating generic local capacity. Both
  deserve the long gate because they are exact-capacity alternatives and the
  short evidence does not justify deleting either.
- Decision: I160 long-gates post-dilated, post-undilated, and exact scale zero
  at 300k on seed 20260787. Promote only a balanced long winner. During the GPU
  run, prune rejected I151-I154 membrane, regime, and curriculum execution
  paths and run the scheduled literature scan.
- Artifacts: `artifacts/diagnostics/iteration-159-post-tcn-dilation-attribution/`
  and `manifests/iteration-159-post-tcn-dilation-attribution.json`.

### Iteration 160 premise - post-TCN long gate

- Run post-dilated, post-undilated, and scale-zero for 300k presentations,
  9,375 optimizer steps, and full diagnostics on a fresh balanced schedule.
- All arms retain exact 91,451 parameter, path-count, initialization, objective,
  optimizer, and data-order matching. The two active arms differ only in TCN
  dilation pattern 1/4/16 versus 1/1/1.
- Promote the strongest balanced long candidate only if core gains survive
  without a diagnostic failure. Then strip inactive comparison paths from the
  reusable configuration while preserving checkpoint compatibility.
- Manifest: `manifests/iteration-160-post-tcn-long-gate.json`.

### Iteration 160 maintenance - 2026-07-12

- Cleanup: removed executable config fields, validation, parameters, modules,
  forward paths, helpers, runner parsing, and tests for rejected I151-I154
  membrane-state, branchwise-decay, regime-expert, and objective-curriculum
  mechanisms. Historical manifests, checkpoints, diagnostics, and written
  results remain; retired config keys preserve historical checkpoint parsing.
- Verification: post-cleanup AxoSim passes 101/101, AxoBench passes 131 with
  1 skip, stable I140 loads at 77,125 parameters and 300k presentations, and
  `git diff --check` passes.
- Literature: completed `literature/iteration-160-post-refinement-scan.md`.
  If the long gate succeeds, it ranks a fixed-budget residual-refiner training
  phase first, followed by a causal multigrid post refiner. If the gate fails,
  neither should run because both depend on a real post-TCN mechanism.

### Iteration 160 - 2026-07-12

- Regime: 300k-presentation structural promotion gate on seed 20260787, with
  9,375 updates and exact 91,451-parameter, initialization, data-order,
  objective, optimizer, and executed-path matching.
- Post-dilated versus scale zero improves voltage SERA by 0.05636 mV and
  response SERA by 0.03860 mV significantly. Dynamics improves 0.00911 mV/ms
  uncertainly. Exact F1 gains 0.01517 significantly; F1 area and 5 ms F1
  decline 0.00968 and 0.01140 uncertainly.
- Post-undilated improves voltage/dynamics/response SERA significantly by
  0.02554/0.02052/0.02959 mV. Its F1 area/exact/5 ms changes are
  +0.00193/+0.00088/+0.01095 and uncertain. It is the more balanced
  dynamics/tolerant-F1 point, while dilation leads voltage, response, and exact
  F1 under the declared priorities.
- Absolute deployed metrics are voltage 16.88053 mV, dynamics 21.25655 mV/ms,
  response 17.43707 mV, F1 area 0.36867, exact F1 0.20155, and 5 ms F1
  0.44320. No new diagnostic attention trigger appears. Voltage null skill
  rises 0.32670 to 0.33336 and residual lag falls 0.27296 to 0.26230, while
  spike-count error rises 0.27963 to 0.41125 and refractory violations rise
  0.31483 to 0.38651.
- Decision: promote post-dilated under the SERA-first policy. The exact source
  checkpoint was copied to `checkpoints/axomamba.pt`; config/default/checkpoint
  load agree at 91,451 parameters and 300k presentations. SHA-256 is
  `b8b06ba1d93f7bfae9c30c297e6c5fc0e7e7d2e98dffcfa72c232a1e163bc78c`.
  I161 tests the scan's top follow-up: fixed-budget training of this promoted
  model as a pretrained global operator plus explicit local residual refiner.
- Artifacts: `artifacts/diagnostics/iteration-160-post-tcn-long-gate/`,
  `manifests/iteration-160-post-tcn-long-gate.json`, and
  `literature/iteration-160-post-refinement-scan.md`.

### Iteration 161 premise - pretrained residual-refiner optimization

- I160 establishes the post-TCN as a real local correction to a global Mamba
  operator. Test the literature scan's first recommendation directly: start
  every arm from the exact promoted 300k checkpoint, then spend one matched
  120k stream on either all parameters, only the 14,326 local-refiner
  parameters, or only the 77,125 non-refiner parameters.
- Revisit I116 latent/output distillation because its stated retry condition is
  now met: the teacher is structurally stronger. Unlike I116's random-start
  students, use the initial promoted model itself as a frozen functional and
  normalized-latent anchor during fine-tuning. This tests regularization around
  a good basin rather than forcing a young representation to imitate a mature
  one.
- Joint, latent-anchor, output-anchor, local-only, and trunk-only arms have the
  same 91,451 total parameters, initialization, forward paths, presentations,
  updates, data order, objective, and one shared frozen-teacher forward. The
  two selective arms necessarily differ in active-gradient count; this is the
  intervention, not an unreported compute match.
- Manifest: `manifests/iteration-161-pretrained-refiner-distillation.json`.

### Iteration 161 - 2026-07-12

- Regime: pretrained-model continuation and train-only distillation test. All
  five arms start from exact I160 SHA
  `b8b06ba1d93f7bfae9c30c297e6c5fc0e7e7d2e98dffcfa72c232a1e163bc78c`,
  use 120k presentations and 3,750 updates, and share one frozen-teacher
  forward and both computed distillation losses per batch.
- Against joint continuation, latent anchoring at 0.1 significantly regresses
  voltage/dynamics/response SERA by 0.00758/0.00445/0.00551, with only small
  uncertain F1 gains. Output anchoring at 0.5 significantly regresses those
  SERA axes by 0.01311/0.00500/0.00965 and has uncertain negative F1 changes.
  Distillation therefore fails under both random-start I116 and mature-start
  I161 conditions; latent mismatch itself falls to 0.00348, so it is active.
- Local-refiner-only training (14,326 active of 91,451 total parameters)
  significantly regresses all six metrics versus joint, including voltage by
  0.04066 mV and response by 0.02347 mV. Trunk-only training (77,125 active)
  significantly regresses voltage/dynamics/response by
  0.01279/0.00783/0.00728 and F1 area by 0.00571. The local/global operator
  needs co-adaptation; a separable residual-refiner phase is rejected.
- A direct paired comparison to the unfine-tuned I160 source confirms joint
  continuation significantly improves voltage/dynamics/response by
  0.04143/0.01590/0.02749, F1 area by 0.01763, and 5 ms F1 by 0.01734. Exact
  F1 improves 0.00807 uncertainly. Absolute metrics are voltage 16.83911,
  dynamics 21.24065, response 17.40957, F1 area 0.38630, exact 0.20962, and
  5 ms 0.46053.
- No new diagnostic trigger appears. Residual lag improves 0.26230 to 0.25653
  and voltage gain 0.33316 to 0.33676; spike-count error, refractory violation,
  and morphology disparity worsen slightly. Promote joint continuation as the
  cumulative-420k stable checkpoint, preserving the I160 source run and hash.
  New SHA-256 is
  `6fc514e4c217e021e00816ed602d07d66361cc0482fcf0c63179b7d916fd9884`.
- Artifacts: `artifacts/diagnostics/iteration-161-pretrained-refiner-distillation/`
  including `source-comparison/`, and
  `manifests/iteration-161-pretrained-refiner-distillation.json`.

### Iteration 162 premise - causal multigrid post refiner

- Translate the I160 scan's second-ranked multiscale-operator hypothesis into
  a strict causal test. Add a second three-block post-TCN path that sees a
  left-looking four-step average sampled at 4x lower rate, then repeat its
  completed coarse states causally back to the full grid.
- Compare the coarse path with a two-full-rate-path capacity control and an
  executed coarse path at scale zero. All arms start from stable I161, add the
  same 14,326 parameters for 105,777 total, execute both three-block paths,
  and are output-identical before training. They share initialization,
  presentations, updates, data, objective, and optimizer; only the second
  path's resolution or scale differs.
- Multigrid must materially beat both scale zero and dual-full capacity on the
  core SERA axes without event or localized-dynamics failure. Otherwise reject
  explicit coarse/fine separation rather than tuning pooling factor or scale.
- Manifest: `manifests/iteration-162-causal-multigrid-refiner.json`.

### Iteration 162 - 2026-07-12

- Regime: literature-backed causal coarse/fine post-refiner test from stable
  I161, using 120k presentations, 3,750 updates, and three exactly matched
  105,777-parameter arms.
- Causal multigrid versus scale zero significantly regresses voltage and
  response by 0.00344 and 0.00093 mV. It improves F1 area by 0.00313
  significantly, but dynamics -0.00062, exact +0.00015, and 5 ms +0.00392 are
  uncertain. The effect is too small and trades away the primary SERA axes.
- Dual-full-rate capacity changes voltage by +0.00002 mV, dynamics by
  -0.00053, response by -0.00117, F1 area by +0.00193, exact by +0.00070, and
  5 ms by +0.00200. Only response is significant. Multigrid therefore does not
  beat its exact-capacity control; the explicit scale-separation premise is
  rejected without pooling-factor or scale tuning.
- The scale-zero arm is also a clean 120k continuation of I161. Direct paired
  diagnostics show significant gains over the source in voltage -0.02648,
  dynamics -0.01608, F1 area +0.01337, and 5 ms F1 +0.01801. Response
  -0.00173 and exact F1 -0.00254 are uncertain. Its absolute metrics are
  voltage 16.81263, dynamics 21.22458, response 17.40785, F1 area 0.39968,
  exact 0.20708, and 5 ms 0.47854.
- Decision: promote the continuation effect, not the failed architecture. The
  zero-scaled coarse modules were stripped from 105,777 to 91,451 parameters;
  a CUDA identity test found maximum absolute output difference 0.0. Source
  artifact SHA is
  `4080da457a6dec549348aca7935501d44691e9ac03d73ae17baba4435d561465`;
  stable cumulative-540k SHA is
  `b2b8a90792beaaafcd064caf34d3631e111116d84521d6e731e330e7acc72584`.
- Artifacts: `artifacts/diagnostics/iteration-162-causal-multigrid-refiner/`
  including `source-comparison/`, and
  `manifests/iteration-162-causal-multigrid-refiner.json`.

### Iteration 163 premise - local residual-target auxiliary

- Test the I160 scan's third-ranked correction mechanism. For each arm, run
  the model normally and with the local post-TCN scale overridden to zero.
  Define the local prediction as full minus trunk-only voltage and train it
  toward a detached target-minus-trunk residual while retaining the full joint
  objective.
- Use one declared auxiliary weight 0.25, an equal-compute zero-weight control,
  and a shuffled-residual control that preserves target scale but breaks
  sample identity. Every arm starts from stable cumulative-540k I162, has
  91,451 parameters, executes both forwards and the residual loss, and matches
  initialization, presentations, updates, data order, optimizer, and task
  objective.
- Advance only if aligned residual supervision materially beats both zero and
  shuffled controls on SERA without event loss. Otherwise reject explicit
  residual assignment rather than tuning its coefficient.
- Manifest: `manifests/iteration-163-local-residual-auxiliary.json`.

### Iteration 163 - 2026-07-12

- Regime: literature-backed local residual-target auxiliary from stable I162.
  Zero, aligned, and shuffled arms have 91,451 parameters, 120k presentations,
  3,750 updates, and each executes both full and local-disabled forwards plus
  the same residual computation.
- Aligned residual supervision catastrophically regresses all six metrics:
  voltage +0.37378, dynamics +0.10568, response +0.24235 mV, F1 area -0.17817,
  exact -0.11966, and 5 ms -0.18976. Its final residual loss is 0.02141 and
  equals its inflated soma MSE, so the failure is active rather than inert.
- Shuffled supervision causes far smaller changes: voltage +0.00885, dynamics
  +0.00486, and response +0.00225 regress significantly, while F1 changes are
  uncertain. The contrast identifies forcing the local path to own the true
  sample-specific trunk residual as the damaging assumption. Reject the
  decomposition without reducing its weight.
- The zero-weight control continues I162 for another 120k. Direct paired
  comparison gives significant voltage -0.01044 and dynamics -0.00429 gains;
  response -0.00288 and F1 area/exact/5 ms
  +0.00523/+0.00339/+0.00662 are favorable but uncertain. Absolute metrics are
  voltage 16.80218, dynamics 21.22029, response 17.40496, F1 area 0.40491,
  exact 0.21047, and 5 ms 0.48516.
- Decision: retain the cumulative-660k control as best, but classify the
  voltage increment as borderline practical evidence and stop using plain
  continuation as the next idea. Source run SHA is
  `c5742c917355dcf37be5cfbfc655a477bc4036538fef29b9e5814d9c4152cae2`;
  stable SHA is
  `eb55f57ce90a58b0ffaa57285d9ca32d07a9be5e2caa28e7bb00667702e44238`.
- Artifacts: `artifacts/diagnostics/iteration-163-local-residual-auxiliary/`
  including `source-comparison/`, and
  `manifests/iteration-163-local-residual-auxiliary.json`.

### Iteration 164 premise - equal-capacity local ConvSSM refiner

- Under the tightened 0.1 mV bar, test the I160 scan's next operator-class
  change rather than another scalar or continuation tweak. Replace each of the
  three depthwise TCN blocks with an equal-parameter diagonal ConvSSM block:
  learned channelwise exponential state, input/state affine terms, pointwise
  mixing, and a learned residual scale.
- Compare local ConvSSM, the proven dilated TCN, and scale zero from common
  fresh initialization. Every arm has exactly 91,451 parameters, executes
  three local blocks and the projection, starts output-identical, and matches
  data, presentations, updates, optimizer, and objective. This intentionally
  measures architecture sample efficiency at 120k rather than another 660k
  continuation effect.
- Require at least 0.1 mV voltage improvement without core regression to
  advance. A smaller statistically significant difference closes this exact
  ConvSSM translation without decay/kernel tuning.
- Manifest: `manifests/iteration-164-local-convssm-refiner.json`.

### Iteration 164 - 2026-07-12

- Regime: equal-capacity operator-class replacement at fresh 120k sample
  efficiency. Scale-zero, dilated TCN, and diagonal ConvSSM each have 91,451
  parameters, common shared initialization, 3,750 updates, and full diagnostics.
- ConvSSM significantly regresses voltage by 0.05799 mV versus scale zero,
  while response improves 0.01074 and exact F1 improves 0.01416. Dynamics
  -0.00232, F1 area +0.00812, and 5 ms +0.01179 are uncertain. It also adds a
  voltage-bias trigger and is slower than the TCN.
- Dilated TCN changes voltage by +0.00160 uncertainly but significantly
  improves dynamics/response and all F1 summaries. It beats ConvSSM on five of
  six point estimates; ConvSSM leads only exact F1 by 0.00312.
- Decision: reject ConvSSM without decay, kernel, or coefficient tuning. The
  0.058 mV adverse effect is below the new advancement magnitude and points in
  the wrong primary direction. Keep stable cumulative-660k TCN unchanged.
- Artifacts: `artifacts/diagnostics/iteration-164-local-convssm-refiner/` and
  `manifests/iteration-164-local-convssm-refiner.json`.

### Iteration 165 - 2026-07-12

- Regime: high-leverage diagnostic premise test, not a training sweep. Fit one
  shared affine voltage transform on 120 validation traces balanced 24 per
  morphology (660,000 scored timesteps), freeze it, and compare against the
  exact stable model on held-out ordinary and intervention traces. Spike logits
  are untouched.
- The validation optimum is scale 0.95235 and bias 0.01536 in normalized
  voltage units. Thus the persistent roughly 0.34 diagnostic gain is not a
  simple output-amplitude calibration coefficient.
- Held-out voltage/dynamics/response SERA regress significantly by
  0.02971/0.00442/0.01532 mV; F1 is exactly unchanged by construction.
- Decision: reject affine calibration and do not add output-scale code. Move to
  exact-parameter macro capacity allocation capable of the new 0.1-1 mV bar.
- Artifacts: `artifacts/diagnostics/iteration-165-voltage-affine-calibration/`,
  including `calibration.json` and the paired comparison.

### Iteration 166 premise - exact-budget macro capacity allocation

- Test a larger 207,449-parameter regime where architecture differences can
  plausibly move voltage by 0.1-1 mV. Compare a wide/shallow global operator
  (width 100, two Mamba layers, state 12) against a narrow/deep operator
  (width 60, four Mamba layers, state 56).
- Both arms have exactly 207,449 active parameters, the same physical-route
  occupancy features, branch encoder/nonlinearities, post-dilated TCN, heads,
  seed, balanced 120k schedule, 3,750 updates, objective, and optimizer. Width,
  depth, and recurrent-state allocation necessarily change FLOPs; report the
  measured difference as part of the intervention.
- Require a balanced voltage delta of at least 0.1 mV to select an allocation.
  Smaller differences close this two-point large-budget allocation test rather
  than triggering local geometry tuning.
- Manifest: `manifests/iteration-166-large-capacity-allocation.json`.

### Iteration 166 - 2026-07-13

- Regime: exact 207,449-parameter macro allocation at fresh 120k sample
  efficiency, with the verified route-occupancy artifact active in both arms.
- Narrow/deep versus wide/shallow changes voltage by +0.01291 mV uncertainly,
  dynamics by -0.00519 uncertainly, and significantly regresses response by
  +0.03498 mV. It significantly improves F1 area/exact/5 ms by
  0.04362/0.01595/0.05083.
- Deep compute is 38.79 seconds versus 29.99 seconds for wide, a 29% penalty.
  The result misses the 0.1 mV bar and exposes a SERA/event frontier rather
  than a better allocation.
- Decision: do not tune width/state/depth locally. I167 isolates the promising
  macro signal: compare the 207k wide model to the standard 91k architecture
  on the exact same seed, schedule, route features, objective, and diagnostics.
  Parameter/compute growth is the declared capacity-scaling intervention.
- Artifacts: `artifacts/diagnostics/iteration-166-large-capacity-allocation/`
  and `manifests/iteration-166-large-capacity-allocation.json`.

### Iteration 167 premise - controlled capacity scaling

- Train the standard 91,451-parameter architecture on I166's exact frozen
  120k schedule and compare it to the cached 207,449-parameter wide/shallow
  baseline. Initialization seed, route occupancy, data order, updates,
  objective, optimizer, and diagnostics match; parameters and model compute
  differ intentionally because capacity scaling is the tested variable.
- Require at least 0.1 mV voltage gain for the large model without a core
  regression. If it passes, replicate capacity scaling on a fresh seed before
  the I170 long gate; otherwise close large-budget search.
- Manifest: `manifests/iteration-167-controlled-capacity-scaling.json`.

### Iteration 167 - 2026-07-13

- Regime: controlled capacity scaling on I166's exact seed, frozen 120k
  schedule, route occupancy, updates, objective, optimizer, and cached
  wide-model predictions. Parameter count is the declared intervention:
  standard 91,451 versus wide 207,449.
- The standard model improves voltage by 0.07280 mV, dynamics by 0.00104,
  response by 0.00908, F1 area by 0.00546, exact F1 by 0.01803, and 5 ms F1 by
  0.00579. Voltage, response, and exact F1 are significant; no metric favors
  the large model. Standard compute is 28.59 seconds versus 29.99 seconds.
- Decision: larger capacity is harmful at 120k and does not clear the 0.1 mV
  bar. I168 tests the aggressive reverse implication with 51k and 35k models
  on the same schedule, using the standard 91k checkpoint as a frozen cached
  baseline. This is a coarse capacity curve, not local width tuning.
- Artifacts: `artifacts/diagnostics/iteration-167-controlled-capacity-scaling/`
  and `manifests/iteration-167-controlled-capacity-scaling.json`.

### Iteration 168 premise - downward capacity scaling

- Test whether the 120k setting is substantially overparameterized. Compare
  width-40/state-16 (50,709 parameters) and width-32/state-8 (34,973) models
  against I167's 91,451-parameter standard model on the exact same seed,
  frozen schedule, route features, updates, objective, and optimizer.
- Parameters and compute differ intentionally as the scaling variable; both
  compact arms run in one shared-loader panel and use one frozen standard-model
  diagnostics cache. Require at least 0.1 mV voltage gain without core
  regression to advance a smaller budget.
- Manifest: `manifests/iteration-168-downward-capacity-scaling.json`.

### Iteration 168 - 2026-07-13

- Regime: coarse downward capacity scaling on I167's exact seed, frozen 120k
  schedule, route features, updates, objective, optimizer, and cached standard
  diagnostics. Capacity is the declared intervention.
- The 50,709-parameter model significantly regresses all six metrics versus
  91,451: voltage +0.12052, dynamics +0.01371, response +0.01938 mV, F1 area
  -0.03683, exact -0.01527, and 5 ms -0.04993.
- The 34,973-parameter model significantly regresses dynamics +0.01826,
  response +0.13976, and F1 area/exact/5 ms
  -0.05894/-0.04826/-0.06197; voltage +0.02430 also regresses significantly.
- Decision: close capacity scaling in both directions. The standard 91k budget
  is the robust 120k sweet spot. I169 tests long state continuity at matched
  parameters, timesteps, updates, and scored positions.
- Artifacts: `artifacts/diagnostics/iteration-168-downward-capacity-scaling/`
  and `manifests/iteration-168-downward-capacity-scaling.json`.

### Iteration 169 premise - matched-compute long state continuity

- Previous burn-in 0-350 ms tests already selected 50 ms, so do not repeat
  loss masking. Instead group four contiguous 500 ms presentation units from
  one trace. The reset control runs four independent 500 ms forwards; the
  continuous arm runs one 2,000 ms forward. Both score the same final 450 ms
  within every chunk.
- Both arms have 91,451 parameters, process exactly 60 million input timesteps,
  receive 120k 500-ms presentation units, take 3,750 optimizer steps, and share
  long samples, route features, objective, optimizer, and diagnostics. Only
  state continuity across artificial 500 ms boundaries differs; measured
  compute is reported.
- Require at least 0.1 mV voltage improvement without core regression to carry
  continuous-context training into I170's long gate.
- Manifest: `manifests/iteration-169-long-state-continuity.json`.

### Iteration 169 - 2026-07-13

- Regime: state-continuity test with matched 91,451 parameters, 60 million
  timesteps, 120k 500-ms presentation units, 3,750 updates, long samples,
  route features, objective, and scored positions. Reset runs four independent
  500 ms forwards; continuous runs one 2,000 ms forward.
- Continuous state significantly improves dynamics by 0.00496 mV/ms, but
  significantly regresses voltage by 0.00929 mV. Response improves only
  0.00011 uncertainly; F1 area/exact/5 ms improve
  0.00582/0.00206/0.00514 uncertainly.
- Compute is effectively matched in practice too: 29.43 seconds continuous
  versus 29.20 reset. The small tradeoff falsifies missing cross-window state
  as a 0.1-1 mV lever.
- Decision: reject continuous-context training. I170 uses the scheduled 300k
  gate for a causal frequency-specialized readout motivated by the measured
  20-250 Hz amplitude collapse, with raw-feature and scale-zero controls.
- Artifacts: `artifacts/diagnostics/iteration-169-long-state-continuity/` and
  `manifests/iteration-169-long-state-continuity.json`.

### Iteration 170 premise - causal frequency-specialized readout long gate

- The held-out I87 audit measured prediction/teacher amplitude gain collapsing
  from 0.998 below 5 Hz to 0.672/0.481/0.312 across 5-20/20-80/80-250 Hz.
  Frequency-band losses did not repair it. Test an architectural decomposition:
  a zero-initialized head receives normalized latent high-pass features
  (latent minus causal 8 ms exponential low-pass) and corrects both outputs.
- Compare high-pass, an equal-parameter raw-latent duplicate head, and an
  executed scale-zero high-pass path. All arms have 91,569 parameters, common
  initialization, active route occupancy, 300k presentations, 9,375 updates,
  objective, optimizer, and diagnostics.
- Require at least 0.1 mV voltage gain without core regression. Raw versus
  high-pass isolates frequency specialization from extra readout capacity.
  During the run, perform I170 cleanup and the scheduled literature scan.
- Manifest: `manifests/iteration-170-frequency-readout-long-gate.json`.

### Iteration 170 - 2026-07-13

- Regime: scheduled 300k structural gate. All arms have 91,569 parameters,
  common initialization, active route occupancy, 300k presentations, 9,375
  updates, objective, optimizer, and diagnostics. Raw versus high-pass isolates
  feature specialization; the executed scale-zero path controls capacity.
- The high-pass head materially fails: voltage +0.11651 mV, response +0.09340
  mV, dynamics +0.00587, exact F1 -0.02079, F1 area -0.00444, and 5 ms F1
  -0.00676. Voltage, response, and exact F1 regress significantly.
- The equal-parameter raw duplicate head improves all six point estimates:
  voltage -0.04626 mV, dynamics -0.00343, response -0.02240, F1 area +0.00382,
  exact +0.00123, and 5 ms +0.00507. Voltage and response are significant, but
  the voltage effect is below the 0.1 mV practical gate.
- Decision: reject both heads. The raw result is mechanistic evidence that an
  extra readout can help, not enough value to advance a 118-parameter feature.
  Stable remains cumulative-660k at SHA
  `eb55f57ce90a58b0ffaa57285d9ca32d07a9be5e2caa28e7bb00667702e44238`.
- Maintenance: removed live I161-I170 support for rejected distillation,
  residual-target, multigrid, ConvSSM, chunk-reset, selective-training,
  local-override, and frequency-head variants. Retained historical manifests,
  diagnostics, and reports. The scheduled scan ranks observed-state
  initialization, fixed-point refinement, multi-horizon state prediction, SWC
  tree neighborhoods, and supervised event-error coordinates. Verification:
  AxoSim 101 passed; AxoBench 131 passed, 1 skipped; `git diff --check` clean;
  stable checkpoint loads as 91,451 parameters with cumulative 660k metadata
  and unchanged SHA.
- Artifacts: `artifacts/diagnostics/iteration-170-frequency-readout-long-gate/`,
  `manifests/iteration-170-frequency-readout-long-gate.json`, and
  `literature/iteration-170-aggressive-dynamics-scan.md`.

### Iteration 171 premise - shared-weight fixed-point local refinement

- Test the literature-motivated premise that the learned post-Mamba local
  operator benefits from recurrent correction, rather than merely another
  execution of the same feed-forward path. Reapply the same TCN correction to
  its updated hidden state for two or three iterations.
- Each feedback arm has an equal-parameter, equal-call no-feedback control.
  Controls execute the same shared TCN two or three times but repeatedly read
  the original hidden state; zero-valued graph connections retain all calls in
  backward execution. All arms share 91,451 parameters, initialization, frozen
  120k schedule, updates, route occupancy, objective, and diagnostics.
- Compare feedback-2 only with no-feedback-2 and feedback-3 only with
  no-feedback-3. Require at least 0.1 mV voltage gain without core regression;
  a smaller delta can establish mechanism but cannot advance the feature.
- Manifest: `manifests/iteration-171-fixed-point-local-refinement.json`.

### Iteration 171 - 2026-07-13

- Regime: scientific mechanism test. Feedback-2/no-feedback-2 and
  feedback-3/no-feedback-3 have 91,451 shared parameters and exactly matched
  operator calls, initialization, frozen 120k schedule, 3,750 updates, route
  occupancy, objective, and diagnostics.
- Feedback-2 versus its control changes voltage -0.00246 mV, dynamics +0.00286,
  response +0.02050 mV, F1 area +0.00019, exact -0.00605, and 5 ms +0.00188.
  Response significantly regresses; all other assessments are uncertain.
- Feedback-3 changes voltage -0.01075 mV, dynamics +0.00415, response +0.00807
  mV, F1 area +0.01645, exact -0.00158, and 5 ms +0.01565. Voltage and tolerant
  F1 improve significantly, but response significantly regresses and the
  voltage effect is one tenth of the practical threshold.
- The no-feedback-3 versus no-feedback-2 voltage delta is 0.00000014 mV, with
  identical F1 summaries. This validates no-feedback-2 as the result-equivalent
  control for the three-pass arm despite one fewer discarded execution.
- Decision: reject fixed-point refinement and do not tune iteration count.
  Frequency-resolved diagnostics do not support the proposed mechanism:
  feedback-3 slightly worsens high-dV/dt and near-spike error ratios while only
  slightly reducing residual lag.
- Post-result premise audit: observed initial soma voltage is unusable on this
  benchmark. Across the same 120 balanced held-out traces it is exactly
  -76.0 mV with zero variance. A calibration/test oracle time-varying linear
  correction from initial voltage changes post-500 ms SERA by exactly 0.0 mV.
  Skip the observed-state model and advance directly to multi-horizon state
  supervision.
- Artifacts: `artifacts/diagnostics/iteration-171-fixed-point-local-refinement/`
  and `manifests/iteration-171-fixed-point-local-refinement.json`.

### Iteration 172 premise - multi-horizon physical-state supervision

- Test whether AxoMamba's trunk improves when it must encode future physical
  state at explicit short and multiscale horizons, as suggested by hybrid world
  models, rather than only the contemporaneous voltage mapping.
- Add three training-only voltage heads. The control predicts current voltage
  from all three; candidates predict 1/5/20 ms or 5/25/100 ms future voltage.
  All use differentiable SERA at weight 0.01 and the same 100-step crop, so
  target tensor sizes, head parameters, forward paths, and loss compute match.
- All arms have 91,628 parameters, common initialization, frozen 120k schedule,
  3,750 updates, route occupancy, primary objective, and diagnostics. Auxiliary
  heads are not used during inference. Require at least 0.1 mV voltage gain
  without response or event regression; inspect high-dV/dt and near-spike
  diagnostics to test the proposed sharp-event mechanism.
- Manifest: `manifests/iteration-172-multi-horizon-state-supervision.json`.

### Iteration 172 - 2026-07-13

- Regime: scientific structural test. Current, short-horizon, and multiscale
  arms each have 91,628 parameters, common initialization, frozen 120k
  schedule, 3,750 updates, three auxiliary heads, 100-step crop, loss weight,
  primary objective, route occupancy, and diagnostics.
- Short 1/5/20 ms horizons change voltage +0.00393 mV, dynamics +0.00225,
  response -0.00805 mV, F1 area +0.01294, exact +0.01363, and 5 ms +0.00910.
  Response and F1 area/exact improve significantly, but voltage slightly
  regresses and the effect scale is small.
- Multiscale 5/25/100 ms horizons regress voltage +0.02088 mV and response
  +0.02819 mV significantly. F1 changes are mixed. It slightly improves
  voltage gain/high-dV/dt/near-spike ratios but worsens residual lag, so the
  proposed future-state representation does not improve the scored task.
- Decision: reject both horizon sets and do not tune horizons or weight.
  A subsequent target-protocol audit found a much larger issue: the training
  channel is clipped at -55 mV, while diagnostics score raw voltage. Across 40
  held-out traces, reconstructed training voltage differs from raw soma by
  5.4264 mV RMSE and up to 98.6602 mV. I173 tests raw-target training directly.
- Artifacts: `artifacts/diagnostics/iteration-172-multi-horizon-state-supervision/`
  and `manifests/iteration-172-multi-horizon-state-supervision.json`.

### Iteration 173 premise - optimize the raw scored voltage target

- The historical training channel clips voltage above -55 mV, but current
  voltage, dynamics, and response SERA score raw soma voltage. The held-out
  target audit measures 5.4264 mV RMSE and 98.6602 mV maximum discrepancy
  between these views. Test whether this target mismatch, rather than model
  structure, drives the persistent 17 mV raw-voltage SERA.
- Compare clipped, 50/50 clipped/raw, and full raw targets. The selected view
  feeds soma MSE, direct SERA, near-spike voltage loss, and all shared trunk
  gradients; spike labels and F1 loss are unchanged. Raw soma is decompressed
  once into the frozen shared batch and transferred once for all models.
- All arms have 91,451 parameters, common initialization, frozen 120k schedule,
  3,750 updates, route occupancy, objective weights, raw-load path, target
  interpolation operations, and diagnostics. Only the predeclared target mix
  differs. Require at least 0.1 mV raw voltage gain without a core regression;
  prefer full raw if it dominates, otherwise treat half raw as a real
  clipped/raw task tradeoff rather than tuning the coefficient.
- Manifest: `manifests/iteration-173-raw-voltage-objective.json`.

### Iteration 173 - 2026-07-13

- Regime: scientific target-protocol correction. Clipped, half-raw, and full-
  raw arms each have 91,451 parameters, common initialization, frozen 120k
  schedule, 3,750 updates, route occupancy, objective weights, raw loading and
  interpolation graph, and diagnostics.
- Full raw improves all six core metrics versus clipped: voltage -0.54586 mV,
  dynamics -0.07290 mV/ms, response -0.20185 mV, F1 area +0.01652, exact
  +0.01713, and 5 ms +0.01180. All except 5 ms are significant; no core point
  estimate regresses. Absolute voltage is 16.51635 mV.
- Half raw also improves all six: voltage -0.35816 mV, dynamics -0.03893,
  response -0.12881, F1 area +0.01873, exact +0.00985, and 5 ms +0.02094.
  Full raw leads half raw on voltage, dynamics, response, and exact F1; half raw
  leads F1 area and 5 ms.
- Failure modes move in the intended direction under full raw: voltage gain
  0.3100 to 0.3645, residual lag 0.2958 to 0.2763, high-dV/dt ratio 3.0025 to
  2.9621, near-spike ratio 4.7144 to 4.6386, and spike-count error 0.4391 to
  0.3992. Intervention gain remains very poor but rises from 0.0151 to 0.0225.
- Decision: the raw scored target is a substantive, mechanism-consistent lead
  in the requested 0.1-1 mV range. I174 repeats the exact three-arm panel on a
  fresh seed before changing the reusable recipe. No mix/weight tuning.
- Artifacts: `artifacts/diagnostics/iteration-173-raw-voltage-objective/` and
  `manifests/iteration-173-raw-voltage-objective.json`.

### Iteration 174 premise - raw-target fresh-seed replication

- Repeat I173's clipped, half-raw, and full-raw panel unchanged on seed
  20260799 and its independently frozen 120k schedule. Parameters, objective
  weights, target interpolation, route occupancy, batch, updates, and
  diagnostics remain identical.
- Require both raw arms to retain at least 0.1 mV voltage gain without a core
  regression. Promote full raw toward I180's long gate only if it again leads
  SERA without losing the half-raw arm's event metrics; otherwise retain the
  replicated balanced arm and treat the target blend as a task frontier.
- Manifest: `manifests/iteration-174-raw-voltage-replication.json`.

### Iteration 174 - 2026-07-13

- Regime: unchanged fresh-seed replication with 91,451 parameters, frozen
  120k schedule, 3,750 updates, objective, raw loading/interpolation, route
  occupancy, and diagnostics matched across all arms.
- Full raw again gives large SERA gains: voltage -0.57356 mV, dynamics
  -0.09746 mV/ms, and response -0.21712 mV. Exact F1 improves +0.00931
  uncertainly, but F1 area -0.01211 and 5 ms -0.02067 significantly regress.
- Half raw improves voltage -0.31646 mV, dynamics -0.03163, and response
  -0.12354 significantly. F1 area/exact/5 ms change
  -0.00860/+0.00176/-0.01167 uncertainly.
- Across I173-I174, mean half-raw deltas favor all six metrics: voltage
  -0.33731 mV, dynamics -0.03528, response -0.12618, F1 area +0.00506, exact
  +0.00580, and 5 ms +0.00463. Full raw averages voltage -0.55971, dynamics
  -0.08518, response -0.20948, F1 area +0.00221, exact +0.01322, and 5 ms
  -0.00443.
- Decision: the target correction replicates strongly, but full raw is a
  SERA/tolerant-F1 frontier on seed 2. I175 starts all arms from the exact
  cumulative-660k stable checkpoint and gives clipped, half-raw, and full-raw
  objectives the same 120k adaptation budget. This tests whether mature event
  features can retain F1 while raw training repairs voltage.
- Artifacts: `artifacts/diagnostics/iteration-174-raw-voltage-replication/` and
  `manifests/iteration-174-raw-voltage-replication.json`.

### Iteration 175 premise - pretrained raw-target adaptation

- Start clipped, half-raw, and full-raw arms from the exact promoted
  cumulative-660k checkpoint, then give each the same 120k presentations and
  3,750 optimizer steps. This isolates adaptation target while matching
  initialization, incremental compute, data, objective weights, and model.
- Test whether the mature clipped-target representation preserves event F1 as
  raw voltage gradients repair the scored target mismatch. Require at least
  0.1 mV voltage gain and no significant F1 regression versus clipped
  continuation. Compare absolute metrics with the scratch I173-I174 runs, but
  do not treat their different total training budgets as a controlled delta.
- Manifest: `manifests/iteration-175-pretrained-raw-adaptation.json`.

### Iteration 175 - 2026-07-13

- Regime: matched adaptation from the exact cumulative-660k stable checkpoint.
  All arms receive the same 120k presentations, 3,750 steps, data, optimizer,
  objective weights, target path, and 91,451-parameter model.
- Full raw improves voltage -0.79302 mV, dynamics -0.18669 mV/ms, and response
  -0.39861 mV significantly versus clipped continuation. F1 area/exact/5 ms
  change -0.00250/-0.00478/-0.00682 uncertainly. Absolute metrics are voltage
  15.99141, dynamics 21.02444, response 16.99287, F1 area 0.41187, exact
  0.20969, and 5 ms 0.49183.
- Half raw improves voltage -0.55630 mV, dynamics -0.13072, and response
  -0.28518 significantly. Its F1 area/exact/5 ms changes
  -0.00122/+0.00263/-0.00560 are uncertain.
- Interpretation: mature clipped pretraining and raw adaptation compose. Full
  raw now clears the requested effect range while preserving event metrics
  within uncertainty and beats every recorded absolute voltage result.
- Decision: provisional full-raw lead for I180's long gate. Do not overwrite
  the stable checkpoint before long validation. I176 uses this checkpoint as
  common initialization to test an AP-specific causal waveform head under the
  now-correct raw objective; reject without another >=0.1 mV gain.
- Artifacts: `artifacts/diagnostics/iteration-175-pretrained-raw-adaptation/`
  and `manifests/iteration-175-pretrained-raw-adaptation.json`.

### Iteration 176 premise - AP-specific waveform under raw supervision

- Revisit the old causal spike-waveform premise only because I173-I175 changed
  its decisive assumption: voltage losses now see the raw AP waveform rather
  than a -55 mV-clipped target. Start all arms from I175's 15.99141 mV full-raw
  checkpoint and continue full-raw training for the same 120k budget.
- Every arm adds and executes the same zero-initialized 16-tap causal kernel
  from predicted spike probability to soma voltage. Compare scale zero,
  detached spike input, and joint-gradient spike input at the old replicated
  scale 2. Parameters (91,467), initialization, schedule, updates, raw
  objective, and diagnostics match exactly.
- Require another >=0.1 mV voltage gain without F1 regression and improved
  high-dV/dt/near-spike ratios. A smaller result is rejected immediately; do
  not reopen kernel or scale sweeps.
- Manifest: `manifests/iteration-176-raw-spike-waveform.json`.

### Iteration 176 - 2026-07-13

- Regime: structural AP-head test from I175 full raw, with matched 91,467
  parameters, zero initialization, causal-kernel execution, frozen 120k
  schedule, 3,750 updates, raw objective, and diagnostics.
- Detached waveform changes voltage -0.00008 mV, dynamics +0.00104, response
  +0.00072 mV, F1 area +0.00040, exact -0.00148, and 5 ms +0.00011. Joint
  changes voltage +0.00056 mV and all other metrics by at most 0.00201.
  Every assessment is uncertain.
- The scale-zero continuation reaches absolute voltage 15.98057 mV, only
  0.01084 mV below I175 after another 120k. This independently shows that
  plain raw continuation has entered diminishing returns.
- Decision: reject waveform support and queue it for I180 cleanup. I177 tests
  a higher-capacity soma-only temporal corrector with base-soma versus
  corrected-soma spike coupling, targeting the remaining raw voltage/event
  gradient conflict structurally.
- Artifacts: `artifacts/diagnostics/iteration-176-raw-spike-waveform/` and
  `manifests/iteration-176-raw-spike-waveform.json`.

### Iteration 177 premise - soma-only temporal corrector under raw supervision

- Start from I175 full raw and add a dedicated post-representation dilated TCN
  that emits only a soma correction. This is the raw-target revisit of I149's
  task-specific temporal depth, now with an explicit control for the known
  soma-to-spike coupling path.
- Compare scale zero, ordinary corrected-soma spike coupling, and base-soma
  spike coupling. The latter keeps reported voltage correction but drives the
  spike logit from the uncorrected soma, preventing direct F1 movement through
  coupling. All arms own and execute the same three TCN blocks and output head,
  have 102,471 parameters, common initialization, frozen 120k schedule, 3,750
  updates, full-raw objective, and diagnostics.
- Require >=0.1 mV voltage gain without F1 regression. Base coupling must
  preserve event metrics while retaining most voltage gain to validate the
  gradient-conflict premise; otherwise reject the whole added-capacity path.
- Manifest: `manifests/iteration-177-raw-soma-temporal-corrector.json`.

### Iteration 177 - 2026-07-13

- Regime: output-specific temporal-capacity test from I175 full raw. All arms
  have 102,471 parameters and matched TCN execution, initialization, frozen
  120k schedule, 3,750 updates, raw objective, and diagnostics.
- Corrected-soma coupling regresses voltage +0.00466 mV and dynamics +0.00232;
  response improves -0.00080 mV, and F1 area/exact/5 ms change
  -0.00164/+0.00148/-0.00176. Base-soma coupling regresses voltage +0.00500
  and dynamics +0.00338; every other delta is below 0.0012. All non-voltage
  assessments are uncertain.
- Interpretation: base coupling does isolate the correction from direct spike
  coupling as designed, but neither representation nor F1 benefits. Extra
  soma temporal capacity remains inactive at the task scale.
- Decision: reject both TCN variants and queue support for I180 cleanup. I178
  gives the path one discriminating use: explicit clipped-base/raw-final
  supervision to force AP residual decomposition. Without >=0.1 mV, close
  output-specific correction entirely.
- Artifacts: `artifacts/diagnostics/iteration-177-raw-soma-temporal-corrector/`
  and `manifests/iteration-177-raw-soma-temporal-corrector.json`.

### Iteration 178 premise - explicit clipped-base/raw-final decomposition

- I177 shows that a soma TCN does not spontaneously specialize. Force the
  hybrid decomposition suggested by raw-target diagnostics: supervise the
  uncorrected base soma on clipped, half-raw, or raw voltage while the TCN-
  corrected final soma is always supervised on full raw voltage. Drive spike
  coupling from base soma to prevent the correction from directly moving F1.
- All arms use the active 102,471-parameter soma TCN, full-raw primary losses,
  matched base MSE/SERA auxiliary weights 0.75/0.01, common I175
  initialization, frozen 120k schedule, 3,750 updates, and diagnostics. Only
  the auxiliary base target differs.
- Require >=0.1 mV voltage gain over raw-base control without F1 regression.
  If clipped-base fails, reject explicit AP residual decomposition rather than
  tuning auxiliary weights.
- Manifest: `manifests/iteration-178-clipped-base-raw-corrector.json`.

### Iteration 178 - 2026-07-13

- Regime: explicit hybrid-target decomposition from I175 full raw. All arms
  have 102,471 parameters, an active soma TCN, base coupling, matched primary
  and auxiliary losses, frozen 120k schedule, 3,750 updates, and diagnostics.
- Clipped-base/raw-final changes voltage -0.00246 mV, dynamics +0.00222,
  response -0.00646 mV, F1 area +0.01608, exact -0.00361, and 5 ms +0.02528.
  F1 area/5 ms and response improve significantly; voltage is uncertain and
  negligible. Half-base similarly changes voltage only -0.00364 mV.
- Interpretation: forced decomposition activates a meaningful tolerant-event
  specialization, unlike I177's unconstrained corrector, but supplies none of
  the requested 0.1-1 mV voltage gain. This is a narrow F1 mechanism, not a
  general performance advance.
- Decision: no promotion or coefficient tuning. Carry clipped decomposition as
  one I180 long-run F1 arm because F1 across time is a declared core metric;
  prune it if the event gain does not survive. I179 independently replicates
  pretrained raw adaptation before the long gate.
- Artifacts: `artifacts/diagnostics/iteration-178-clipped-base-raw-corrector/`
  and `manifests/iteration-178-clipped-base-raw-corrector.json`.

### Iteration 179 premise - pretrained raw adaptation replication

- Repeat I175's stable-checkpoint clipped, half-raw, and full-raw adaptation
  panel unchanged on seed 20260804 and its independent frozen 120k schedule.
  This is the second controlled test of pretraining plus target correction,
  distinct from I173-I174's scratch-training replications.
- Require half and full raw to retain >=0.1 mV voltage/response gains without
  significant F1 regression. Use I175/I179 jointly to select half versus full
  for I180; both may enter the long gate if their SERA/F1 frontier persists.
- Manifest: `manifests/iteration-179-pretrained-raw-replication.json`.

### Iteration 179 - 2026-07-13

- Regime: unchanged second pretrained-adaptation seed with 91,451 parameters,
  common cumulative-660k initialization, frozen 120k schedule, 3,750 updates,
  target interpolation, objective, and diagnostics.
- Full raw improves voltage -0.79434 mV, dynamics -0.19016 mV/ms, and response
  -0.38573 mV significantly. F1 area/exact/5 ms change
  -0.00550/-0.00219/-0.00707 uncertainly. Absolute voltage is 16.00344 mV.
- Half raw improves voltage -0.55525 mV, dynamics -0.13076, and response
  -0.28618 significantly. F1 area/exact/5 ms change
  -0.00175/+0.00274/-0.00371 uncertainly.
- Across I175/I179, full raw averages -0.79368 mV voltage, -0.18843 dynamics,
  -0.39217 response, and absolute voltage 15.99742 mV. Half raw averages
  -0.55577/-0.13074/-0.28568. No event delta is significant on either
  pretrained seed.
- Decision: full raw is the robust primary I180 arm; half raw remains the
  conservative frontier control. Run the scheduled literature scan, then the
  300k five-arm gate with exact decomposition capacity control.
- Artifacts: `artifacts/diagnostics/iteration-179-pretrained-raw-replication/`
  and `manifests/iteration-179-pretrained-raw-replication.json`.

### Iteration 180 premise - raw-target long gate

- Run the scheduled 300k-presentation gate from the exact cumulative-660k
  checkpoint. The standard clipped, half-raw, and full-raw arms use the same
  91,451 parameters, 9,375 optimizer steps, initialization, frozen schedule,
  optimizer, and diagnostics. This is the long-duration test of I175/I179's
  replicated raw-target adaptation effect.
- In parallel, compare raw-base and clipped-base variants of the I178
  decomposition with identical 102,471-parameter models, active TCN compute,
  main full-raw objective, and base auxiliary weights. Only the base target
  differs. The decomposition pair is internally controlled and is not treated
  as parameter-matched to the standard arms.
- Full raw must retain at least a 0.1 mV voltage-SERA advantage over clipped
  continuation without a significant F1 regression. Clipped-base must retain
  I178's event gain over raw-base without a material voltage regression; if it
  fails, remove the entire soma-TCN/decomposition path during I180 cleanup.
- Perform the scheduled literature scan and dead-code cleanup alongside/after
  the run, then run both projects' full test suites and verify the selected
  stable checkpoint before promotion.
- Manifest: `manifests/iteration-180-raw-target-long-gate.json`.

### Iteration 180 - 2026-07-13

- Regime: scheduled 300k promotion gate from the exact cumulative-660k stable
  checkpoint. Standard clipped, half-raw, and full-raw arms have 91,451
  parameters; raw-base and clipped-base decomposition arms have 102,471.
  Every arm receives the same frozen schedule, 300k presentations, 9,375
  optimizer steps, initialization, and diagnostics. The decomposition claim
  is made only against its exact-capacity raw-base control.
- Full raw improves voltage/dynamics/response SERA significantly by
  0.83889/0.21680/0.42889 versus clipped continuation. F1 area/exact/5 ms
  change -0.00276/+0.00326/-0.00822 and are all uncertain. Absolute metrics
  are voltage 15.95215 mV, dynamics 20.98482 mV/ms, response 16.96027 mV, F1
  area 0.41607, exact 0.21538, and 5 ms 0.49719. Half raw retains a smaller
  0.58459 mV voltage gain and no significant F1 changes.
- The clipped-base decomposition replicates its event mechanism versus the
  internally matched raw-base control: F1 area +0.00762 and 5 ms +0.01397 are
  significant, response improves 0.01023 mV, and voltage/dynamics improve only
  0.00567/0.01044 uncertainly. It remains a validated experimental F1 branch,
  but does not justify adding 11,020 parameters to the default model.
- Decision: promote the simpler full-raw model and make raw soma the reusable
  training objective. `checkpoints/axomamba.pt` is the exact I180 source state
  with 960k cumulative presentations and SHA-256
  `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`.
  Retain decomposition support for its replicated core F1 effect; remove the
  rejected I171 fixed-point, I172 horizon, and I176 soft-waveform paths.
- Compute: panel wall time 605.55 seconds. Standard arms used 70.89-75.43
  seconds of measured GPU compute; decomposition arms used 85.58-85.79.
- Artifacts: `artifacts/diagnostics/iteration-180-raw-target-long-gate/`,
  including `decomposition-pairwise/`;
  `manifests/iteration-180-raw-target-long-gate.json`; and
  `literature/iteration-180-raw-hybrid-dynamics-scan.md`.

### Iteration 180 maintenance - 2026-07-13

- Removed executable config fields, model modules, forward paths, objective
  machinery, runner parsing, and tests for rejected I171 fixed-point local
  iteration, I172 multi-horizon voltage supervision, and I176 soft waveform
  correction. Retired config keys preserve stable and historical config
  parsing. The replicated I178/I180 soma decomposition remains because it
  significantly improves two F1-across-time summaries.
- Promoted the raw-soma objective in `default_axomamba_training_objective()`
  and documented why the clipped target channel is not the scored contract.
- Verification: AxoSim passes 105/105; AxoBench passes 131 with 1 skip; the
  promoted checkpoint loads as `AxoMamba`, has 91,451 parameters, matches the
  source state exactly, reports 960k cumulative presentations, and agrees with
  `default_axomamba_config()`.

### Iteration 181 premise - causal hard event lifting

- Translate the I180 scan's top discontinuity-operator idea into a causal
  neuron surrogate. Threshold the predicted pre-coupling spike probability at
  0.5 with a straight-through estimator, then feed the event coordinate to a
  learned 21 ms causal phase kernel that emits a raw-voltage correction.
- Compare hard straight-through, soft probability, and scale-zero variants.
  All arms own and execute the same 21-parameter convolution, start
  output-identically from the cumulative-960k stable model, and have 91,472
  parameters, one frozen 120k schedule, 3,750 updates, and full-raw objective.
  The hard/soft distinction is the only active representation change.
- Require >=0.1 mV voltage gain without F1 regression and a hard-over-soft
  advantage to validate discontinuity lifting beyond I176's rejected soft
  waveform. Otherwise reject this exact event representation without tuning
  threshold or kernel length.
- Manifest: `manifests/iteration-181-hard-event-lifting.json`.

### Iteration 181 - 2026-07-13

- Regime: literature-backed causal event-coordinate test from the promoted
  cumulative-960k model. Hard, soft, and scale-zero arms all have 91,472
  parameters, identical initialization/execution, full-raw objective, frozen
  120k schedule, 3,750 updates, and diagnostics.
- Hard straight-through lifting changes voltage/dynamics/response by
  +0.00069/+0.00045/-0.00067 and F1 area/exact/5 ms by
  -0.00312/-0.00176/-0.00231 versus scale zero. F1 area significantly
  regresses; every other assessment is uncertain. Soft lifting changes
  voltage +0.00277 and significantly regresses response by 0.00285 mV.
- Interpretation: creating a hard predicted discontinuity coordinate does not
  unlock the action-potential residual. Hard is slightly less adverse than
  soft on SERA, but misses the 0.1 mV gate by two orders of magnitude and
  loses a core F1 summary.
- Decision: reject hard and soft event lifting without threshold/kernel
  tuning; queue the code for I190 cleanup. I182 tests the scan's distinct
  causal Laplace/exponential latent basis against an equal-size ordinary
  causal averaging basis and scale zero.
- Compute: panel wall time 177.75 seconds; all arms trained for 120k
  presentations and 3,750 steps.
- Artifacts: `artifacts/diagnostics/iteration-181-hard-event-lifting/` and
  `manifests/iteration-181-hard-event-lifting.json`.

### Iteration 182 premise - causal Laplace-like temporal basis

- Test the I180 scan's transform-domain fallback with a compact causal basis
  over completed AxoMamba features. Three fixed exponential filters at 2, 8,
  and 32 ms feed a learned soma residual decoder, approximating a bounded real
  Laplace basis without introducing a second recurrent trunk.
- Compare exponential filters with equal-size boxcar filters and an executed
  exponential scale-zero control. Every arm owns the same 174 decoder
  parameters for 91,625 total, starts output-identically from cumulative-960k,
  and shares the frozen 120k full-raw schedule, 3,750 updates, and diagnostics.
- Require >=0.1 mV voltage gain without F1 regression. Exponential must also
  beat boxcar to attribute any effect to the transform basis rather than a
  generic causal smoothing residual; otherwise reject without tuning taus.
- Manifest: `manifests/iteration-182-causal-temporal-basis.json`.

### Iteration 182 - 2026-07-13

- Regime: causal transform-basis test from cumulative-960k. Exponential,
  boxcar, and exponential scale-zero arms all have 91,625 parameters,
  identical initialization, three executed filters, full-raw objective,
  frozen 120k schedule, 3,750 updates, and diagnostics.
- Exponential changes voltage/dynamics/response by
  +0.00316/+0.00159/-0.00154 and F1 area/exact/5 ms by
  -0.00088/-0.00249/+0.00063 versus scale zero. Boxcar changes them by
  +0.00128/+0.00022/-0.00179 and -0.00005/-0.00110/+0.00164. Every
  assessment is uncertain, and exponential is worse than boxcar on voltage,
  dynamics, F1 area, exact F1, and 5 ms F1.
- Interpretation: additional fixed temporal memory over the completed latent
  is inactive at useful scale, while its 129-step filters add real execution
  cost. The result does not support transform-domain dynamics over ordinary
  causal smoothing.
- Decision: reject both bases without tau/kernel tuning and queue code for
  I190 cleanup. The promoted residual audit shows a 0.429 voltage gain with
  depolarized/high-dVdt/near-spike errors 2.88/3.00/4.71, motivating I183's
  post-coupling depolarized-state calibration rather than more temporal depth.
- Compute: panel wall time 180.92 seconds; all arms trained for 120k and 3,750
  steps.
- Artifacts: `artifacts/diagnostics/iteration-182-causal-temporal-basis/` and
  `manifests/iteration-182-causal-temporal-basis.json`.

### Iteration 183 premise - post-coupling depolarized calibration

- The promoted raw model has voltage gain 0.429 and concentrates error in
  depolarized/high-dVdt/near-spike states. Add a continuous piecewise-linear
  soma residual with fixed knots at -55, -40, -20, and 0 mV (normalized
  1.27/2.77/4.77/6.77). Apply it after voltage-to-spike coupling so calibration
  cannot directly change spike logits.
- Compare depolarized knots with broad state knots and an executed depolarized
  scale-zero control. Every arm owns four zero-initialized coefficients, has
  91,455 parameters, starts output-identically from cumulative-960k, and uses
  the same frozen 120k full-raw schedule, 3,750 updates, and diagnostics.
- Require >=0.1 mV voltage gain without dynamics/response regression. F1
  should be invariant apart from indirect shared-training movement; a broad-
  over-depolarized win would reject the localization premise.
- Manifest: `manifests/iteration-183-depolarized-calibration.json`.

### Iteration 183 - 2026-07-13

- Regime: diagnostic-targeted post-coupling calibration from cumulative-960k.
  Depolarized, broad, and depolarized scale-zero arms have 91,455 parameters,
  identical initialization, full-raw objective, frozen 120k schedule, 3,750
  updates, and diagnostics. Calibration cannot directly alter spike logits.
- Depolarized calibration changes voltage/dynamics/response by
  +0.00003/-0.00180/+0.00099 and F1 area/exact/5 ms by
  +0.00154/+0.00013/+0.00310. Broad calibration changes them by
  -0.00058/-0.00125/+0.00072 and +0.00144/+0.00019/+0.00106. Every
  assessment is uncertain.
- Interpretation: the raw-trained shared model absorbs the tiny hinge head
  without restoring amplitude. Neither targeted nor broad output flexibility
  approaches the 0.1 mV gate, despite isolating spike coupling correctly.
- Decision: reject hinge calibration and queue code for I190 cleanup. After
  three inactive structural output mechanisms, I184 runs one bounded,
  aggressive raw-objective balance test to distinguish gradient allocation
  from missing representation before returning to structural changes.
- Artifacts: `artifacts/diagnostics/iteration-183-depolarized-calibration/`
  and `manifests/iteration-183-depolarized-calibration.json`.

### Iteration 184 premise - aggressive raw-objective balance

- Test whether sparse AP gradients, rather than output capacity, limit the
  promoted raw model. Compare the promoted weights with a 10x near-spike
  voltage weight, a 3x direct-SERA weight, and their combination. These are
  deliberately large one-shot interventions, not a local coefficient sweep.
- All arms use the identical 91,451-parameter cumulative-960k model, frozen
  120k schedule, 3,750 updates, data, optimizer, and executed loss terms. Only
  gradient weights differ; parameter and forward compute are exact matches.
- Require >=0.1 mV voltage gain without significant F1 or response failure.
  A combined-only win supports jointly concentrating raw AP and threshold-
  relevance gradients; broad failure closes objective scaling and returns I185
  to architecture.
- Manifest: `manifests/iteration-184-raw-objective-balance.json`.

### Iteration 184 - 2026-07-13

- Regime: bounded aggressive gradient-allocation test from cumulative-960k.
  Promoted, 10x near-spike, 3x direct-SERA, and combined arms use identical
  91,451-parameter models, initialization, executed losses, frozen 120k
  schedule, 3,750 updates, and diagnostics.
- Combined improves all six point estimates: voltage/dynamics/response
  -0.07402/-0.00630/-0.01312 and F1 area/exact/5 ms
  +0.00650/+0.00682/+0.00598. Voltage, response, F1 area, and exact F1 improve
  significantly; dynamics and 5 ms are uncertain. Absolute voltage is
  15.87629 mV.
- Near-spike alone improves voltage/response significantly by 0.05207/0.00971
  mV; direct SERA alone improves them by 0.04639/0.00923. Their composition is
  larger than either component and is the only arm with significant F1 area
  and exact-F1 gains.
- Decision: this is a balanced multi-core lead but misses the standalone 0.1
  mV voltage bar. Do not tune weights. I185 repeats the combined objective
  unchanged on a fresh seed and composes it with the replicated clipped-base
  structural decomposition; only a replicated balanced effect can reach I190.
- Artifacts: `artifacts/diagnostics/iteration-184-raw-objective-balance/` and
  `manifests/iteration-184-raw-objective-balance.json`.

### Iteration 185 premise - objective/decomposition composition

- Replicate I184's combined objective unchanged on seed 20260810. In parallel,
  apply it to raw-base and clipped-base soma decompositions to test whether
  stronger raw AP gradients and the replicated event-specialized structure
  compose rather than overlap.
- Standard promoted/combined arms are exact 91,451-parameter controls. The two
  decomposition arms are exact 102,471-parameter controls with the same active
  TCN, base auxiliary weights, strong main objective, initialization, and
  compute; only the base target differs. All share one frozen 120k schedule.
- Require the standard combined arm to reproduce a balanced multi-core gain.
  Clipped-base must beat raw-base internally and supply either >=0.1 mV versus
  promoted standard or a larger replicated F1-across-time gain without SERA
  loss. No coefficient changes are permitted.
- Manifest: `manifests/iteration-185-objective-decomposition-composition.json`.

### Iteration 185 - 2026-07-13

- Regime: fresh-seed replication/composition from cumulative-960k. Standard
  promoted/combined arms have 91,451 parameters; raw-base/clipped-base strong
  decomposition arms have 102,471. Internal pairs share exact parameters,
  initialization, objective, frozen 120k schedule, 3,750 updates, and compute.
- Standard combined again improves voltage and response significantly by
  0.06767 and 0.01536 mV. Dynamics changes +0.00171; F1 area/exact/5 ms change
  -0.00356/-0.00748/-0.00120, all uncertain. Across I184-I185, mean combined
  deltas are voltage -0.07084, dynamics -0.00230, response -0.01424, F1 area
  +0.00147, exact -0.00033, and 5 ms +0.00239.
- Clipped-base composition reaches voltage -0.07490 and response -0.02680
  versus promoted standard, but its F1 changes are small and uncertain. It
  improves response 0.01144 and dynamics 0.01139 over standard combined while
  adding 11,020 parameters; voltage improves only another 0.00724 mV.
- Decision: combined weighting robustly improves SERA but does not replicate
  the I184 F1 gain and remains below 0.1 mV. Keep the unchanged standard arm as
  an I190 fallback only; do not tune weights or advance decomposition
  composition. I186 targets the separate refractory/count failure structurally.
- Schedule audit: 62.96% of 4,096 ordinary windows already contain a scored
  spike, with 1.82 spikes/window and time prevalence 0.405%. Event-window
  oversampling is therefore less discriminating than a predicted refractory
  state after the 10x near-spike loss test.
- Artifacts:
  `artifacts/diagnostics/iteration-185-objective-decomposition-composition/`
  and `manifests/iteration-185-objective-decomposition-composition.json`.

### Iteration 186 premise - causal predicted refractory state

- Target the promoted model's 36.1% refractory-violation and 30.7% spike-count
  errors with an explicit five-step biological refractory state. Subtract the
  causal sum of prior predicted spike probabilities from the current spike
  logit after voltage coupling; no teacher event enters inference.
- Compare detached-history and joint-gradient refractory states with an
  executed scale-zero control. All arms have the same 91,451 parameters,
  forward operations, cumulative-960k state, frozen 120k full-raw schedule,
  3,750 updates, and diagnostics. Active arms differ only in whether future
  loss gradients traverse prior predicted probabilities.
- Require a large F1-across-time/count/refractory gain without >=0.1 mV SERA
  regression. Voltage cannot change directly through the post-coupling
  operation, so any SERA movement measures shared-training interaction.
- Manifest: `manifests/iteration-186-causal-refractory-state.json`.

### Iteration 186 - 2026-07-13

- Regime: parameter-free causal refractory-state test from cumulative-960k.
  Detached, joint-gradient, and scale-zero arms have identical 91,451
  parameters, five-step history compute, frozen 120k full-raw schedule, 3,750
  updates, and diagnostics.
- Detached refractory improves F1 area and 5 ms F1 significantly by
  0.00704/0.00788; exact F1 rises 0.00449 uncertainly. Voltage/dynamics/
  response improve 0.00478/0.00291/0.00037 mV, all uncertain. Joint history
  significantly improves 5 ms F1 by 0.00912 and voltage by 0.00938 mV; F1
  area/exact changes +0.00527/+0.00146 are uncertain.
- Failure-mode audit: detached changes refractory violations only
  0.33979 to 0.33905 and worsens count error 0.27908 to 0.29055; joint lowers
  violations to 0.33713 and changes count error to 0.28400. The core F1 gain is
  therefore real but not explained by clearing the diagnostic trigger.
- Decision: retain detached refractory as the cleaner two-metric event lead,
  but do not tune duration/scale. I187 replicates it in a factorial with I184-
  I185's SERA-improving strong objective, testing balanced composition before
  the I190 gate.
- Artifacts: `artifacts/diagnostics/iteration-186-causal-refractory-state/`
  and `manifests/iteration-186-causal-refractory-state.json`.

### Iteration 187 premise - refractory/objective factorial

- Run a fresh-seed 2x2 factorial of promoted versus strong raw objective and
  scale-zero versus detached five-step refractory state. This tests whether
  I184-I185's robust SERA gain and I186's F1 gain compose or compete.
- All four arms have exact 91,451 parameters, execute the same refractory
  history, start from cumulative-960k, and share one frozen 120k schedule,
  3,750 updates, and diagnostics. Main effects and interaction are fully
  attributable without architecture or parameter confounds.
- Require refractory F1 gains to replicate under at least one objective and
  the combined cell to retain the strong objective's SERA gain. A combined
  >=0.1 mV voltage gain or replicated multi-F1 improvement without SERA loss
  earns an I190 arm; otherwise keep neither modification.
- Manifest: `manifests/iteration-187-refractory-objective-factorial.json`.

### Iteration 187 - 2026-07-13

- Regime: exact 2x2 fresh-seed factorial of strong objective and detached
  refractory state. All cells have 91,451 parameters, execute the same
  histories, start from cumulative-960k, and share frozen 120k training, 3,750
  updates, and diagnostics.
- Strong scale-zero replicates SERA: voltage/response improve significantly by
  0.06670/0.01371 mV, while F1 area/exact/5 ms change
  -0.00547/-0.00689/-0.00649 uncertainly. Refractory under promoted objective
  has only small uncertain changes on this seed, so I186's standalone event
  gain does not replicate directly.
- Within the strong objective, refractory significantly improves F1 area by
  0.00736, 5 ms F1 by 0.01141, and response by 0.00475 mV. Voltage/dynamics
  also improve 0.00419/0.00255 uncertainly. The combined cell versus promoted
  improves voltage/response significantly by 0.07089/0.01847, with F1 area and
  5 ms +0.00189/+0.00492 uncertainly but exact F1 -0.00564 uncertainly.
- Decision: the structural event effect is objective-dependent but composes
  cleanly and repairs the strong objective's tolerant-F1 loss. Carry the
  combined cell toward I190, but it remains below 0.1 mV. I188 tests a distinct
  one-step causal onset state aimed at exact timing, in a refractory factorial.
- Artifacts: `artifacts/diagnostics/iteration-187-refractory-objective-factorial/`
  including `strong-objective-pairwise/`, and
  `manifests/iteration-187-refractory-objective-factorial.json`.

### Iteration 188 premise - causal onset/refractory factorial

- Add a one-step causal onset coordinate from the rise in predicted spike
  probability, boosting the current logit on event onset and reducing it on
  decay. This is a parameter-free SNN-like emission mechanism aimed at exact
  timing, complementary to refractory suppression of later logits.
- Run a 2x2 onset/refractory factorial under the promoted raw objective. Every
  cell has 91,451 parameters, executes both causal states, starts from
  cumulative-960k, and shares one frozen 120k schedule, 3,750 updates, and
  diagnostics. Both histories are detached to isolate the fixed forward
  mechanism from temporal-gradient changes.
- Require onset to improve exact F1 and count/timing diagnostics without SERA
  loss. Composition must beat refractory alone; otherwise reject onset without
  lag/scale tuning and retain only the I187 candidate for I190.
- Manifest: `manifests/iteration-188-onset-refractory-factorial.json`.

### Iteration 188 - 2026-07-13

- Regime: exact parameter-free onset/refractory factorial from cumulative-960k.
  All four cells have 91,451 parameters, execute both detached histories, and
  share frozen 120k full-raw training, 3,750 updates, and diagnostics.
- Onset alone significantly improves voltage/dynamics/response by
  0.00444/0.00406/0.00245 and F1 area by 0.00364. Exact F1 changes -0.00242
  uncertainly and 5 ms +0.00419 uncertainly, so it fails its timing premise
  and remains far below the practical voltage bar.
- Refractory again significantly improves 5 ms F1, by 0.00921; its other
  changes are uncertain. Onset plus refractory improves 5 ms by 0.00688
  uncertainly, regresses exact F1 by 0.00602 uncertainly, and is worse than
  refractory alone on both F1 area and 5 ms.
- Decision: reject onset and composition without lag/scale tuning; queue onset
  code for I190 cleanup. Refractory's tolerant-F1 effect now appears on three
  seeds and remains in the I190 candidate. I189 adds the missing exact
  differentiable dynamics-SERA objective before selecting the long panel.
- Artifacts: `artifacts/diagnostics/iteration-188-onset-refractory-factorial/`
  and `manifests/iteration-188-onset-refractory-factorial.json`.

### Iteration 189 premise - exact dynamics-SERA training objective

- Add the direct training analogue of AxoBench dynamics SERA: first-difference
  error in mV/ms scored by the maximum teacher relevance across each adjacent
  pair. This closes the last missing declared core objective rather than
  introducing another proxy or coefficient sweep.
- Compare dynamic SERA at weight 0.01 under promoted and strong raw objectives,
  plus strong dynamic SERA with detached refractory. Matched controls use a
  `1e-8` monitor weight so all five arms execute the same 101-threshold loss;
  every arm has 91,451 parameters, one frozen 120k schedule, and 3,750 updates.
- Require material dynamics improvement without voltage/response/F1 failure.
  The full composition must improve the I187 candidate rather than merely
  exchange voltage for derivative fit. Only evidence-backed cells enter I190.
- Manifest: `manifests/iteration-189-dynamics-sera-objective.json`.

### Iteration 189 - 2026-07-13

- Regime: exact differentiable dynamics-SERA objective test from
  cumulative-960k. All five arms have 91,451 parameters, execute the same
  101-threshold dynamics loss through `1e-8` monitor controls, and share frozen
  120k training, 3,750 updates, and diagnostics.
- Dynamics SERA at 0.01 under the promoted objective changes diagnostic
  dynamics by -0.00590 mV/ms, voltage/response by -0.00189/-0.00325, F1
  area/5 ms by +0.00136/+0.00424, and exact F1 by -0.00389. Every assessment
  is uncertain, so direct metric training is active but practically weak.
- Under the strong objective, dynamic SERA improves dynamics only another
  0.00058 mV/ms over the matched monitor and regresses exact F1 another
  0.00441. Strong dynamic refractory reaches voltage -0.06901 and response
  -0.01989 versus promoted but significantly regresses exact F1 by 0.01211.
- Decision: reject dynamics-objective weighting and refractory composition in
  this cell; retain the loss implementation as an exact reusable metric but do
  not promote it. I190 long-gates only the evidence-backed 2x2 promoted/strong
  objective by scale-zero/detached refractory factorial.
- Artifacts: `artifacts/diagnostics/iteration-189-dynamics-sera-objective/`
  and `manifests/iteration-189-dynamics-sera-objective.json`.

### Iteration 190 premise - strong-objective/refractory long gate

- Run the exact promoted-versus-strong objective and scale-zero-versus-
  detached-refractory 2x2 factorial for 300k presentations and 9,375 updates.
  This long gate jointly resolves I184-I185's ~0.071 mV SERA lead and I186-
  I188's repeated tolerant-F1 refractory effect.
- All four cells have 91,451 parameters, execute the same five-step history,
  start from cumulative-960k, and share one frozen schedule, optimizer, and
  diagnostics. Objective and refractory main effects and interaction remain
  exactly attributable at long duration.
- Promote only if the strong objective reaches a practical balanced gain or
  refractory supplies replicated F1 value without SERA failure. Perform the
  scheduled literature scan and dead-code cleanup, then run both full suites
  and stable-checkpoint verification before I191.
- Manifest: `manifests/iteration-190-objective-refractory-long-gate.json`.

### Iteration 190 - 2026-07-13

- Regime: scheduled 300k exact 2x2 long gate from cumulative-960k. Promoted
  versus strong raw objective and scale-zero versus detached refractory cells
  all have 91,451 parameters, execute the same five-step history, and share
  one frozen schedule, 9,375 updates, initialization, optimizer, and diagnostics.
- Strong scale-zero improves voltage/response significantly by
  0.06888/0.01283 mV versus promoted scale zero. Dynamics changes -0.00365;
  F1 area/exact/5 ms change -0.00244/-0.00063/-0.00180, all uncertain. The
  voltage effect remains below 0.1 mV after three short seeds and this long run.
- Promoted refractory changes voltage/dynamics/response by
  +0.00183/+0.00031/-0.00088 and F1 area/exact/5 ms by
  +0.00003/-0.00348/+0.00310; every assessment is uncertain. Under the strong
  objective, refractory versus strong scale zero improves SERA slightly but
  lowers F1 area and 5 ms by 0.00287/0.00271. The short tolerant-F1 effect does
  not survive 300k.
- Absolute promoted-continuation metrics are voltage 15.91496, dynamics
  20.96099, response 16.94047, F1 area 0.42243, exact 0.21889, and 5 ms
  0.50531. Its extra 300k continuation gain is below the practical promotion
  bar, so the cumulative-960k I180 checkpoint remains deployed.
- Decision: reject refractory and do not promote strong weighting or plain
  continuation. Stable remains SHA
  `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`.
  I191 follows the literature scan's top distinct mechanism: learned temporal
  delays aimed at exact timing.
- Compute: panel wall time 508.95 seconds; every arm trained for 300k and
  9,375 updates.
- Artifacts:
  `artifacts/diagnostics/iteration-190-objective-refractory-long-gate/`,
  `manifests/iteration-190-objective-refractory-long-gate.json`, and
  `literature/iteration-190-temporal-alignment-scan.md`.

### Iteration 190 maintenance - 2026-07-13

- Removed executable config fields, validation, parameters/modules, forward
  paths, objective machinery, helpers, runner-visible behavior, and tests for
  failed I181 event lifting, I182 temporal bases, I183 hinge calibration, I188
  onset state, I189 dynamics objective, and I186-I190 refractory state.
  Retired config keys preserve loading of stable and historical configs.
- Retained raw-soma training support and the I178/I180 soma decomposition,
  which remains the only replicated structural F1 mechanism in this block.
  Historical manifests, diagnostics, checkpoints, and negative results remain.
- Verification: AxoSim passes 105/105, AxoBench passes 131 with 1 skip,
  `git diff --check` passes, and stable loads as a 91,451-parameter `AxoMamba`
  with 960k cumulative presentations.

### Iteration 191 premise - learned fractional spike delays

- Translate the I190 scan's top temporal-alignment mechanism into a five-tap
  causal delay bank over the final predicted spike probability. Learn five
  continuous positions in 0-4 ms by triangular interpolation and five
  zero-initialized amplitudes; no teacher spike enters the forward path.
- Compare learned fractional positions with an equal-parameter dense integer
  causal kernel and scale zero. Every arm owns the same ten parameters,
  computes both fractional and dense banks, starts output-identically from
  cumulative-960k, and has 91,461 parameters, one frozen 120k schedule, 3,750
  updates, and full diagnostics.
- Require a significant exact-F1/timing-gap gain without SERA loss. Learned
  positions must beat the dense kernel to support delay alignment rather than
  generic causal output capacity. Do not sweep bank size or maximum lag.
- Manifest: `manifests/iteration-191-fractional-spike-delays.json`.

### Iteration 191 - 2026-07-13

- Regime: controlled 120k structural timing test from cumulative-960k. Scale
  zero, dense five-lag integer kernel, and five learned fractional delays all
  have 91,461 parameters, execute both delay banks, share the frozen schedule,
  and train for 3,750 updates.
- Learned fractional versus zero changes voltage/dynamics/response by
  -0.00123/-0.00299/+0.00057 and F1 area/exact/5 ms by
  +0.00196/-0.00079/+0.00504. Only dynamics and 5 ms F1 are labeled improved;
  the practical SERA effect is negligible and exact timing worsens.
- Dense versus zero changes voltage/dynamics/response by
  -0.00120/-0.00258/+0.00060 and F1 area/exact/5 ms by
  +0.00113/-0.00145/+0.00367. Learned-over-dense differences are correspondingly
  tiny, so the data do not identify fractional delay learning over generic
  causal output capacity.
- Decision: reject the final-readout delay family without a lag/count sweep and
  queue it for I200 cleanup. I192 tests the I190 scan's distinct upstream
  sparse-impulse hypothesis before Mamba.
- Compute: panel wall time 182.99 seconds; every arm trained for 120k
  presentations and 3,750 updates.
- Artifacts: `artifacts/diagnostics/iteration-191-fractional-spike-delays/`,
  `runs/axomamba-autoresearch/iteration-191-fractional-spike-delays/`, and
  `manifests/iteration-191-fractional-spike-delays.json`.

### Iteration 192 premise - sparse delayed branch impulses

- Apply one learned fractional arrival delay and amplitude to each of the 118
  routed branch-current impulses before the existing branch nonlinearity and
  Mamba projection. This tests whether sparse signed inputs need explicit
  branchwise arrival structure rather than a correction after prediction.
- Compare the sparse operator with a dense per-branch five-lag temporal
  projection and scale zero. Every arm owns all 826 parameters, computes both
  banks, starts output-identically from cumulative-960k, and shares the same
  frozen 120k schedule, objective, optimizer, and diagnostics.
- Advance only for at least 0.1 mV voltage improvement or a large replicated
  timing or intervention gain without a core regression. Sparse must beat
  dense to support learned impulse arrivals rather than generic temporal
  capacity; no delay-span or scale tuning follows a null result.
- Manifest: `manifests/iteration-192-branch-impulse-delays.json`.

### Iteration 192 - 2026-07-13

- Regime: controlled upstream temporal-structure test from cumulative-960k.
  Scale zero, dense per-branch five-lag projection, and sparse learned
  fractional arrival have 92,277 parameters, execute both banks, and share one
  frozen 120k schedule and 3,750 updates.
- Sparse versus zero changes voltage/dynamics/response by
  -0.00300/-0.00421/-0.00185 and F1 area/exact/5 ms by
  +0.00037/-0.00164/+0.00229. Only dynamics is labeled improved; exact F1 is
  unfavorable and every practical effect is tiny.
- Dense versus zero changes voltage/dynamics/response by
  -0.00205/-0.00244/-0.00644 and F1 area/exact/5 ms by
  -0.00264/+0.00009/-0.00122. Sparse has no material learned-arrival advantage
  over the dense control.
- Decision: reject the entire branch-delay family without tuning and queue it
  for I200 cleanup. I193 tests a higher-dimensional continuous quotient
  embedding, the next distinct I190 literature mechanism.
- Compute: panel wall time 185.93 seconds; every arm trained for 120k
  presentations and 3,750 updates.
- Artifacts: `artifacts/diagnostics/iteration-192-branch-impulse-delays/`,
  `runs/axomamba-autoresearch/iteration-192-branch-impulse-delays/`, and
  `manifests/iteration-192-branch-impulse-delays.json`.

### Iteration 193 premise - continuous hybrid quotient embedding

- Add an eight-dimensional continuous residual coordinate system after the
  temporal stack. The quotient candidate maps four learned phase/radius pairs
  to periodic coordinates, allowing reset-equivalent phases to meet without
  thresholded event labels or teacher-conditioned inference.
- Compare against the same input/output projections with a generic SiLU
  embedding and scale zero. All arms own 936 parameters, compute both feature
  maps, start exactly at the stable output, and share one frozen 120k schedule,
  objective, optimizer, and diagnostics.
- Advance only for at least 0.1 mV voltage improvement or a large multi-F1 or
  intervention gain. Quotient must beat the generic residual to support hybrid
  geometry rather than added nonlinear capacity; no dimension sweep follows a
  null result.
- Manifest: `manifests/iteration-193-hybrid-quotient-embedding.json`.

### Iteration 193 - 2026-07-13

- Regime: controlled continuous-geometry test from cumulative-960k. Scale
  zero, generic SiLU residual, and periodic phase/radius quotient residual have
  92,387 parameters, execute both maps, and share one frozen 120k schedule and
  3,750 updates.
- Quotient versus zero changes voltage/dynamics/response by
  +0.00131/-0.00028/-0.00045 and F1 area/exact/5 ms by
  +0.00047/+0.00140/+0.00054. Every assessment is uncertain and voltage moves
  in the wrong direction.
- Generic versus zero changes voltage/dynamics/response by
  -0.00122/-0.00320/-0.00080 and F1 area/exact/5 ms by
  +0.00185/+0.00158/+0.00150. It is slightly better on five point estimates,
  but all are uncertain and roughly two orders below useful magnitude.
- Decision: reject quotient and generic embeddings without a dimension sweep
  and queue both for I200 cleanup. I194 tests the untried state-dependent
  damping/modulation mechanism on the already-proven local TCN.
- Compute: panel wall time 178.42 seconds; every arm trained for 120k
  presentations and 3,750 updates.
- Artifacts: `artifacts/diagnostics/iteration-193-hybrid-quotient-embedding/`,
  `runs/axomamba-autoresearch/iteration-193-hybrid-quotient-embedding/`, and
  `manifests/iteration-193-hybrid-quotient-embedding.json`.

### Iteration 194 premise - activity-gated local correction

- The promoted post-Mamba dilated TCN improves local dynamics, but diagnostics
  retain strong high-dV/dt and near-spike attenuation. Let its correction be
  modulated per time and latent channel by the current hidden state, rather
  than using one correction strength in every dynamical regime.
- Compare the adaptive gate with a learned static per-channel gate and scale
  zero. Every arm owns the same 3,480 gate parameters, computes both gates,
  begins exactly at multiplicative identity, and shares one frozen 120k
  schedule, objective, optimizer, and diagnostics.
- Require at least 0.1 mV voltage improvement or a large balanced F1/response
  gain, with preferential reduction of high-dV/dt or near-spike error. Adaptive
  must beat static to establish state dependence; no gate-scale sweep follows
  a null result.
- Manifest: `manifests/iteration-194-activity-gated-local-tcn.json`.

### Iteration 194 - 2026-07-13

- Regime: controlled state-dependent local-correction test from cumulative-
  960k. Adaptive, static, and scale-zero arms have 94,931 parameters, execute
  both gates, and share one frozen 120k schedule and 3,750 updates.
- Adaptive versus zero changes voltage/dynamics/response by
  -0.00173/+0.00196/+0.00187 and F1 area/exact/5 ms by
  +0.00172/+0.00484/+0.00147. Every assessment is uncertain; dynamics and
  response move adversely.
- Static versus zero improves voltage/dynamics/response by
  0.00495/0.00477/0.00198 and changes F1 area/exact/5 ms by
  -0.00050/+0.00061/+0.00054. SERA improvements are significant but twentyfold
  below materiality, and static beats adaptive on every SERA axis.
- Decision: reject both gates without scale tuning and queue them for I200
  cleanup. I195 revisits short-horizon supervision only because I172 optimized
  the wrong clipped future-voltage channel.
- Compute: panel wall time 178.18 seconds; every arm trained for 120k
  presentations and 3,750 updates.
- Artifacts: `artifacts/diagnostics/iteration-194-activity-gated-local-tcn/`,
  `runs/axomamba-autoresearch/iteration-194-activity-gated-local-tcn/`, and
  `manifests/iteration-194-activity-gated-local-tcn.json`.

### Iteration 195 premise - raw short-horizon physical-state supervision

- I172's 1/5/20 ms future heads improved F1 area/exact by 0.01294/0.01363 but
  slightly regressed voltage while unknowingly learning clipped future voltage.
  Repeat the mechanism after the raw-target correction using three training-
  only heads supervised on raw future soma voltage.
- Compare 1/5/20 ms heads against three identical heads predicting current raw
  voltage. Both arms have 91,628 parameters, identical inference contracts,
  one frozen 120k schedule, 3,750 updates, and the same primary raw objective;
  only auxiliary target alignment differs.
- Advance for at least 0.1 mV voltage or a replicated large multi-F1 gain with
  no SERA regression. This is the sole raw-target revisit; do not tune horizons
  or auxiliary weight after a null result.
- Manifest: `manifests/iteration-195-raw-future-supervision.json`.

### Iteration 195 - 2026-07-13

- Regime: corrected-target revisit of I172. Current-raw and 1/5/20 ms future-
  raw auxiliaries have 91,628 parameters, identical inference, one frozen 120k
  schedule, and 3,750 updates.
- Future versus current changes voltage/dynamics/response by
  +0.00113/+0.00478/-0.00180 and F1 area/exact/5 ms by
  -0.00022/-0.00070/+0.00137. Every assessment is uncertain, and the prior
  clipped-target F1 signal disappears under the scored raw target.
- Decision: reject and queue auxiliary heads/loss for I200 cleanup. I196 tests
  a direct routed-current bypass around Mamba as a distinct attenuation test.
- Compute: panel wall time 169.07 seconds; each arm trained for 120k/3,750.

### Iteration 196 premise - routed branch readout bypass

- Add zero-initialized direct output projections from saturated routed branch
  currents, bypassing the temporal trunk so fast input effects need not survive
  Mamba attenuation. Compare direct current with first-difference branch drive
  and scale zero.
- Every arm owns and executes both 118-to-2 projections, has 91,923 parameters,
  identical initial output, and shares one frozen 120k schedule and diagnostics.
- Require at least 0.1 mV voltage gain or a large balanced response/F1 gain;
  direct must beat delta to support preserved forcing rather than generic high-
  pass capacity. No skip-scale sweep follows a null result.
- Manifest: `manifests/iteration-196-branch-readout-skip.json`.

### Iteration 196 - 2026-07-13

- Direct/delta versus zero improve voltage by 0.00442/0.00431 and dynamics by
  0.00381/0.00402. Direct exact F1 rises 0.00470, but F1 area falls 0.00271
  and response regresses 0.00136. Both 91,923-parameter paths miss materiality.
- Decision: reject linear skips and queue for I200 cleanup. I197 tests one
  expressive nonlinear bypass before closing direct forcing.
- Compute: panel wall 177.60 seconds; 120k/3,750 per arm.

### Iteration 197 premise - nonlinear forcing bypass

- Compare a 32-unit nonlinear routed-branch bypass with an identically executed
  32-unit post-Mamba latent bypass and scale zero. Every arm owns both MLPs,
  has 97,275 parameters, starts output-identically, and shares 120k/3,750.
- Require >=0.1 mV or a large balanced F1/response gain; branch must beat latent
  to establish preserved forcing. No width or scale sweep follows a null.
- Manifest: `manifests/iteration-197-nonlinear-forcing-bypass.json`.

### Iteration 197 - 2026-07-13

- Nonlinear branch versus zero changes voltage by -0.00006 and exact F1 by
  +0.00147; latent control changes voltage/response by -0.00238/-0.00206.
  All practical effects are negligible. Reject and queue for I200 cleanup.
- Compute: panel wall 178.99 seconds; 97,275 parameters, 120k/3,750 each.

### Iteration 198 premise - spike/soma gradient-routing factorial

- Forward-identical 2x2: ordinary versus detached spike-readout features and
  ordinary versus detached soma contribution to spike-voltage coupling. All
  cells have 91,451 parameters, identical predictions/compute, and 120k/3,750.
- Test whether spike objectives damage the shared voltage representation.
  Advance only for >=0.1 mV or a large balanced F1/response gain.
- Manifest: `manifests/iteration-198-gradient-routing-factorial.json`.

### Iteration 198 - 2026-07-13

- Detach-readout, detach-coupling, and detach-both improve voltage by
  0.00352/0.00276/0.00773. Detach-both improves dynamics/response by
  0.01413/0.00842 but significantly regresses F1 area/5 ms by 0.01777/0.02626.
  Exact F1 rises, revealing a timing trade rather than a balanced gain.
- Decision: reject all detachments. I199 bounds whether aggressive voltage
  optimization is possible only by abandoning the event metrics.
- Compute: panel wall 204.21 seconds; 91,451 parameters, 120k/3,750 each.

### Iteration 199 premise - voltage/event Pareto boundary

- Compare the full promoted objective, a SERA-dominant objective retaining BCE
  but removing F1/near-spike terms, and voltage-only MSE+SERA. Models, seed,
  schedule, parameters, updates, and inference are identical.
- This is a boundary test, not coefficient tuning: determine whether >=0.1 mV
  is available only with a disqualifying F1 collapse. No arm is promoted
  without balanced core evidence.
- Manifest: `manifests/iteration-199-objective-pareto-boundary.json`.

### Iteration 199 - 2026-07-13

- SERA-dominant versus balanced improves voltage/dynamics/response by
  0.03616/0.01153/0.01791 but significantly regresses F1 area/5 ms by
  0.01662/0.03004. Voltage-only gains 0.03008 and loses 0.01813/0.02858.
- Decision: no balanced or >=0.1 mV arm. I200 long-confirms the less-bad SERA-
  dominant boundary against balanced, then stops. Panel wall 164.49 seconds.

### Iteration 200 premise - final long boundary and closure

- Train balanced and SERA-dominant objectives for 300k/9,375 from the unchanged
  cumulative-960k stable checkpoint. Both have 91,451 parameters and one frozen
  schedule. This confirms whether I199's disqualifying Pareto trade persists.
- Independently complete the mandatory literature scan, prune all rejected
  I191-I199 code, run both full test suites and checkpoint loading, and produce
  the final synthesis. Do not launch I201.
- Manifest: `manifests/iteration-200-final-long-boundary.json`.

### Iteration 200 - 2026-07-13

- Regime: final scheduled 300k boundary gate from cumulative-960k. Balanced
  and SERA-dominant arms have 91,451 parameters, one frozen five-morphology
  schedule, 9,375 updates, and full diagnostics.
- Balanced absolute metrics: voltage 15.91603, dynamics 20.96603, response
  16.93834, F1 area 0.43185, exact 0.22162, and 5 ms 0.51729.
- SERA-dominant versus balanced improves voltage/dynamics/response by
  0.04186/0.01762/0.01681, but significantly regresses F1 area/5 ms by
  0.03880/0.05711; exact changes -0.00088 uncertainly. It remains below the
  0.1 mV bar and is decisively worse on event utility.
- Decision: reject SERA-dominant training and do not promote balanced
  continuation. Stable remains I180 at SHA
  `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`.
  Stop; do not launch I201.
- Compute: panel wall 357.84 seconds; 300k/9,375 per arm.
- Literature: `literature/iteration-200-final-scan.md` ranks only future task-
  arc changes, led by a joint branch-time state-space neural operator.
- Cleanup: removed executable I191-I199 delays, quotient embedding, local
  gating, auxiliary horizons, branch/latent bypasses, and gradient detachment;
  historical manifests/diagnostics remain and retired keys preserve loading.
- Verification: AxoSim 105/105; AxoBench 131 passed, 1 skipped; stable loads as
  91,451 parameters with 960k metadata; `git diff --check` passes.
- Final synthesis: `artifacts/iteration-200-final-summary.md`.

### Autoresearch stop - iteration 200

- The hard stop is complete. No iteration 201 manifest or run was created.
- I180 raw-target correction remains the final promoted model. I191-I200 add
  strong negative evidence that local temporal alignment, output capacity,
  gradient separation, or more aggressive SERA weighting can supply another
  balanced 0.1 mV gain within the present formulation.

## Post-I200 Dual Baseline Fix - 2026-07-16

- This is a separate specialization/compression task arc, not iteration 201.
  It began from the immutable I180 checkpoint at 91,451 parameters and 960k
  cumulative presentations.
- A common-initialization fresh 300k ablation found large contributions from
  route occupancy, morphology embedding, the local TCN, branch saturation,
  spike-voltage coupling, and morphology-conditioned synaptic efficacy.
  Shared gain and bilinear interaction were near zero on that seed. Training
  without the soma gate improved voltage SERA by 0.02466 mV and F1 area by
  0.00984.
- Direct lesions of the mature checkpoint were intentionally reported
  separately. They showed strong co-adaptation: directly deleting the soma
  gate worsened voltage SERA by 0.51781 mV, even though an architecture trained
  without it performed better. Lesion effects are not retraining effects.
- A matched 300k mature continuation compared the full model with structural
  simplifications. Removing the soma gate and bilinear interaction produced
  voltage/dynamics/response SERA 15.91434/20.95333/16.93859 and F1
  area/exact/5 ms 0.42430/0.22168/0.50942. All six changes versus the full
  continuation were uncertain. Removing shared synaptic gain as well
  significantly worsened dynamics SERA by 0.02012 and was rejected.
- PROMOTION: the reusable dual AxoMamba now has 90,156 parameters, separate
  heads, no soma multiplicative gate, and no bilinear branch interaction. Its
  promoted checkpoint records 1.26M cumulative presentations and SHA-256
  `2e09cd058158f3e4cf504ef57087931b962f7804ab59bc37c7296066402f0222`.
- On an RTX 5090 with length-500 inputs, structural removal reduced median
  batch-1 inference from 0.93555 to 0.82824 ms, batch-32 inference from 1.48795
  to 1.43144 ms, and batch-32 forward/backward from 4.14110 to 3.93888 ms.
- Full evidence and projection assumptions are in
  `artifacts/post-i200-dual-mechanism-ablation.md`.

## Iteration Template

### Iteration NNN - YYYY-MM-DD

- Regime: scientific or blind, with justification.
- Hypothesis: falsifiable mechanism or direct downstream objective.
- Baseline: checkpoint and frozen diagnostics artifact.
- Candidates: independently attributable modifications.
- Controls: seed, frozen schedule, presentations, optimizer steps, parameters,
  batch, and relevant compute matching.
- Results: core metrics, uncertainty/deltas, failure triggers, wall time, and
  throughput.
- Interpretation: evidence, alternatives, and limitations.
- Decision: keep, discard, or inconclusive; next modification.
- Artifacts: manifests, metrics, checkpoints, and diagnostics paths.
