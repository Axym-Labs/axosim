# Progress

## Autoresearch Strategy

- Stop after iteration 200. Complete I200's scheduled long run, literature
  scan, cleanup, full verification, and consolidated result synthesis; do not
  launch iteration 201.
- Optimize the declared core metrics ruthlessly: primary voltage SERA, F1
  across time, and intervention-response SERA; use diagnostics as fast failure
  gates rather than adding unrelated evaluation burdens.
- Statistical significance is necessary but not sufficient. Voltage gains
  below `0.1` mV are not performance results worth advancing even if paired
  bootstrap intervals exclude zero. Retain a smaller delta only as mechanistic
  evidence or alongside a major non-voltage win; target structural changes in
  the `0.1-1.0` mV range.
- Start each research phase at high temperature with mechanism-changing,
  falsifiable interventions grounded in diagnostics, dataset structure, or
  adjacent literature. Move to local scalar/optimizer refinement only after a
  mechanism earns evidence. Avoid low-impact architectural wandering.
- After a bounded coefficient bracket resolves an objective mechanism, return
  to structural hypotheses from diagnostics/literature. Do not consume serial
  iterations polishing low-impact scalar optima merely because they are easy
  to run.
- One iteration contains one premise plus one or more parallel runs. Every run
  uses the manifest runner, frozen schedules, shared loading/transfers where
  the intervention permits, 120k total forward presentations by default, and
  saved diagnostics in the dedicated artifact directory.
- Match parameters, forward presentations, optimizer steps, initialization,
  data order, and relevant compute paths. When the intervention inherently
  changes one axis, expose and log that difference rather than claiming a
  fully matched comparison.
- Interpret every iteration before choosing the next modification. Promote a
  change only after paired diagnostics show no disqualifying core failure;
  replicate material leads on fresh seeds, and use longer training before
  changing the reusable AxoMamba recipe or stable checkpoint.
- Every tenth iteration uses the 300k presentation budget. Run the larger job
  with a compact candidate set selected by prior evidence, not a broad sweep.
- Every tenth iteration also performs mandatory codebase pruning: delete
  executable code, config knobs, tests, and defaults for failed modifications;
  retain manifests, diagnostics, and written negative results. Run full AxoSim
  and AxoBench verification plus stable-checkpoint loading after cleanup.
- Literature scans refresh high-temperature ideas from adjacent work. Run a
  large scan after iteration 84, next at iteration 100, then every tenth
  iteration (110, 120, ...), preferably in parallel with the scheduled long
  run. Store each scan in `docs/axomamba-autoresearch/literature/` and turn
  useful findings into ranked, falsifiable experiment proposals rather than
  blind citations.
- Keep the promoted model/training recipe conservative. Experimental machinery
  may remain temporarily between cleanup boundaries, but rejected paths must
  be queued explicitly and removed at the next tenth iteration.

## Iteration Log

- LONG GATE (2026-07-13): I200 confirms the objective Pareto failure. SERA-
  dominant versus balanced improves voltage/dynamics/response by
  0.04186/0.01762/0.01681 but significantly regresses F1 area/5 ms by
  0.03880/0.05711. Stable I180 remains deployed unchanged.
- MAINTENANCE (2026-07-13): I200 removed all rejected I191-I199 model, loss,
  config, and test paths; retained retired checkpoint keys and historical
  artifacts. AxoSim passes 105/105, AxoBench 131 with 1 skip, stable loads at
  91,451 parameters/960k with SHA `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`,
  and `git diff --check` passes.
- LITERATURE (2026-07-13): final scan concludes remaining credible work changes
  the formulation: joint branch-time neural operators, cross-morphology
  adaptation, stochastic surrogates, or signature encoders. No I201 is
  justified. See `literature/iteration-200-final-scan.md`.
- MILESTONE (2026-07-13): autoresearch stopped after I200 as requested. Final
  table is `artifacts/iteration-200-final-summary.md`; no new checkpoint was
  promoted and no I201 manifest/run exists. The selected-checkpoint trajectory
  is `artifacts/promoted-voltage-progress.png`.
- EXPERIMENT (2026-07-13): I199 finds no aggressive voltage solution on the
  objective boundary. SERA-dominant gains 0.03616 mV but significantly loses
  F1 area/5 ms by 0.01662/0.03004; voltage-only gains 0.03008 and loses
  0.01813/0.02858. Neither approaches 0.1 mV or balanced utility.
- DECISION (2026-07-13): I200 is the hard stop: 300k balanced versus SERA-
  dominant long boundary, literature scan, full I191-I199 dead-code pruning,
  complete verification, and consolidated synthesis. No I201.
- EXPERIMENT (2026-07-13): I198 rejects spike-gradient detachment. Detaching
  both improves dynamics/response by 0.01413/0.00842 but significantly loses
  F1 area/5 ms by 0.01777/0.02626; voltage improves only 0.00773.
- DECISION (2026-07-13): I199 measures the objective Pareto boundary with full,
  SERA-dominant, and voltage-only training before the final I200 closure.
- EXPERIMENT (2026-07-13): I197 rejects nonlinear forcing. Branch bypass
  voltage changes by only -0.00006; latent capacity reaches -0.00238, so direct
  forcing and generic decoder capacity are both closed.
- DECISION (2026-07-13): I198 runs an exact 2x2 gradient-routing test detaching
  spike-head latent gradients and spike-coupling soma gradients independently.
- EXPERIMENT (2026-07-13): I196 rejects linear routed-current bypasses. Direct
  and delta improve voltage by only 0.00442/0.00431; direct improves exact F1
  by 0.00470 but loses F1 area and slightly regresses response.
- DECISION (2026-07-13): I197 makes the forcing bypass nonlinear and compares
  branch-current versus latent sources under identical parameters/execution.
- EXPERIMENT (2026-07-13): I195 rejects raw short-horizon supervision.
  Future versus current raw auxiliaries changes voltage/dynamics/response by
  +0.00113/+0.00478/-0.00180 and F1 area/exact/5 ms by
  -0.00022/-0.00070/+0.00137; all assessments are uncertain. The old clipped-
  target F1 signal does not survive the corrected objective.
- DECISION (2026-07-13): I196 tests a direct routed-branch readout bypass around
  Mamba against a branch-delta bypass and scale zero, targeting measured
  high-frequency attenuation at the input-to-output path.
- EXPERIMENT (2026-07-13): I194 rejects activity-dependent local gating.
  Adaptive versus scale zero changes voltage/dynamics/response by
  -0.00173/+0.00196/+0.00187 and all assessments are uncertain. Static gating
  improves voltage by 0.00495 and dynamics by 0.00477, but remains twentyfold
  below the gate and disproves the state-dependent premise.
- DECISION (2026-07-13): I195 revisits short-horizon physical-state
  supervision using raw future voltage, because the original I172 experiment
  trained auxiliary heads on the subsequently-discovered clipped target.
- EXPERIMENT (2026-07-13): I193 rejects continuous quotient embeddings.
  Quotient versus scale zero changes voltage by +0.00131 and every core
  assessment is uncertain. The equal-capacity generic residual changes voltage
  by -0.00122 and is slightly better on five metrics, so neither magnitude nor
  quotient-specific evidence supports hybrid geometry.
- DECISION (2026-07-13): I194 tests activity-dependent modulation of the
  promoted post-Mamba local correction against a static per-channel gate and
  scale zero, directly targeting state-dependent high-frequency attenuation.
- EXPERIMENT (2026-07-13): I192 rejects sparse delayed branch impulses.
  Sparse versus scale zero changes voltage/dynamics/response by
  -0.00300/-0.00421/-0.00185 and F1 area/exact/5 ms by
  +0.00037/-0.00164/+0.00229. Dense temporal projection is similarly tiny;
  sparse does not separate from generic temporal capacity and misses the
  0.1 mV gate by two orders of magnitude.
- DECISION (2026-07-13): I193 tests the next ranked mechanism, a continuous
  periodic quotient embedding after the temporal stack, against an equal-
  dimensional generic nonlinear residual and scale zero.
- EXPERIMENT (2026-07-13): I191 rejects final-readout fractional delays.
  Learned delays improve voltage/dynamics by only 0.00123/0.00299 and 5 ms F1
  by 0.00504 versus scale zero, while exact F1 falls by 0.00079. The equal-
  parameter dense integer kernel is effectively tied, so neither practical
  magnitude nor learned-over-dense specificity supports temporal alignment.
- DECISION (2026-07-13): I192 moves the delay premise upstream to sparse
  routed branch impulses before Mamba, compared with a dense temporal
  projection and scale zero under exact parameter and executed-compute matching.
- PREMISE (2026-07-13): I191 tests the I190 scan's top mechanism: five learned
  fractional 0-4 ms delays on predicted spike probability versus an equal-
  parameter dense integer kernel and scale zero. All arms compute both banks,
  have 91,461 parameters and 120k full-raw training. Advancement requires an
  exact-timing gain and learned-over-dense evidence without SERA loss.
- MAINTENANCE (2026-07-13): I190 pruned failed event lift, temporal basis,
  hinge calibration, onset, dynamic-loss, and refractory paths while retaining
  raw training and the replicated decomposition. AxoSim passes 105 tests;
  AxoBench passes 131 with 1 skip; stable loads at 91,451 parameters/960k and
  `git diff --check` passes.
- LONG GATE (2026-07-13): I190 rejects strong weighting and refractory for
  promotion. Strong scale-zero improves voltage/response by 0.06888/0.01283
  but remains below 0.1 mV and has uncertainly negative F1 changes. Refractory
  loses its short F1 effect at 300k. Stable remains cumulative-960k at SHA
  `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`.
- LITERATURE (2026-07-13): the I190 scan ranks learned fractional delays first,
  followed by an upstream sparse delayed-impulse operator and a continuous
  hybrid quotient embedding. It explicitly deprioritizes generic temporal
  heads and long-memory retrieval. Stored at
  `literature/iteration-190-temporal-alignment-scan.md`.
- PREMISE (2026-07-13): I190 is the scheduled 300k exact 2x2 long gate for
  promoted/strong raw objective by scale-zero/detached refractory state. All
  cells have 91,451 parameters, five-step history compute, 9,375 updates, and
  one schedule. Literature scan, pruning, and full verification are mandatory
  before I191.
- EXPERIMENT (2026-07-13): I189 rejects direct dynamics-SERA weighting.
  Dynamics improves only 0.00590 mV/ms under the promoted objective with all
  assessments uncertain; under the strong objective its incremental gain is
  only 0.00058 and exact F1 worsens. Do not tune the dynamics weight.
- PREMISE (2026-07-13): I189 implements the exact differentiable analogue of
  dynamic SERA, the last declared core metric absent from training. Five arms
  test it under promoted/strong objectives and refractory composition; `1e-8`
  monitor controls execute the identical loss. All use 91,451 parameters and
  one 120k schedule. This is metric closure, not a weight sweep.
- EXPERIMENT (2026-07-13): I188 rejects causal onset sharpening. It improves
  SERA axes by only 0.00245-0.00444 and F1 area 0.00364 while exact F1 falls
  uncertainly; onset+refractory loses to refractory alone. Refractory's 5 ms
  F1 gain replicates again at +0.00921 significant.
- PREMISE (2026-07-13): I188 tests a parameter-free one-step causal spike-
  probability onset state in an exact 2x2 factorial with refractory state.
  All cells execute both histories and share 91,451 parameters and 120k data.
  Onset must improve exact timing and beat refractory alone without SERA loss;
  no lag or scale sweep follows.
- COMPOSITION (2026-07-13): I187 shows refractory is objective-dependent. It
  does not replicate alone, but under the strong objective it significantly
  improves F1 area/5 ms by 0.00736/0.01141 and response by 0.00475 mV, with
  voltage also favorable. The combined cell preserves a 0.07089 mV voltage
  and 0.01847 response gain versus promoted, earning an I190 arm.
- PREMISE (2026-07-13): I187 is an exact 2x2 factorial of detached refractory
  state and the unchanged strong raw objective. All four cells have 91,451
  parameters/compute and one 120k schedule. This tests whether the replicated
  SERA and structural F1 signals compose before I190, with no duration, scale,
  or coefficient tuning.
- EXPERIMENT (2026-07-13): I186 finds a structural F1 signal. Detached
  refractory significantly improves F1 area/5 ms by 0.00704/0.00788 with all
  SERA point estimates favorable; joint significantly improves 5 ms by
  0.00912 and voltage by 0.00938 mV. Failure ratios barely change, so the
  mechanism needs fresh-seed replication rather than interpretation by label.
- PREMISE (2026-07-13): I186 adds a parameter-free five-step causal predicted
  refractory state after voltage coupling, directly targeting refractory and
  count failures while preserving causal inference. Detached, joint-gradient,
  and scale-zero arms have exact 91,451 parameters/compute and 120k full-raw
  training. Advancement is driven by a large F1/failure-mode gain without a
  practical SERA regression.
- AUDIT (2026-07-13): 62.96% of 4,096 ordinary training windows already
  contain a scored spike, with 1.82 spikes/window and 0.405% time prevalence.
  Event-window oversampling would weakly duplicate near-spike weighting, so
  I186 instead targets the 36% refractory-violation and 31% count errors with
  a causal predicted refractory state.
- REPLICATION (2026-07-13): I185 confirms the combined objective's SERA effect
  but not I184's F1 gain. Voltage/response improve 0.06767/0.01536; all event
  changes are uncertainly negative. Two-seed mean voltage is -0.07084 mV.
  Decomposition adds only 0.00724 mV voltage over standard combined despite
  11,020 parameters, so neither branch advances beyond an I190 fallback.
- PREMISE (2026-07-13): I185 replicates I184's combined objective unchanged
  and composes it with the validated clipped-base decomposition. Standard and
  decomposition comparisons are internally parameter/compute matched; all
  arms share 120k full-raw data. This is structural composition plus fresh-
  seed replication, not another coefficient search.
- EXPERIMENT (2026-07-13): I184 finds a balanced sub-threshold lead. Combined
  3x direct-SERA plus 10x near-spike improves voltage/dynamics/response by
  0.07402/0.00630/0.01312 and F1 area/exact/5 ms by
  0.00650/0.00682/0.00598; voltage, response, area, and exact are significant.
  It needs unchanged replication because voltage remains below 0.1 mV.
- PREMISE (2026-07-13): I184 is one bounded aggressive loss-allocation test
  after three inactive structural output heads: 10x near-spike, 3x direct
  SERA, and combined versus the promoted raw objective. All four arms have
  identical 91,451 parameters and compute. A broad failure closes coefficient
  scaling; this will not turn into a local tuning sequence.
- EXPERIMENT (2026-07-13): I183 rejects post-coupling hinge calibration.
  Depolarized and broad heads change voltage by +0.00003/-0.00058 mV; all six
  assessments are uncertain. Isolating soma calibration from spike logits is
  insufficient because the shared model absorbs the tiny residual path.
- PREMISE (2026-07-13): I183 adds a four-coefficient post-coupling hinge
  calibration at -55/-40/-20/0 mV, directly targeting the promoted model's
  low voltage gain and depolarized error while leaving spike logits untouched.
  Broad-knot and scale-zero controls have the same 91,455 parameters and 120k
  full-raw compute. Advancement needs >=0.1 mV without response/dynamics loss.
- EXPERIMENT (2026-07-13): I182 rejects causal temporal bases. Exponential
  and boxcar filters regress voltage by 0.00316/0.00128 mV, all assessments
  are uncertain, and exponential loses to boxcar on five of six core point
  estimates. Do not tune taus or kernel length.
- AUDIT (2026-07-13): the promoted raw model's residual is amplitude- and
  regime-localized: voltage gain is 0.429, while depolarized, high-dVdt, and
  near-spike errors are 2.885/2.995/4.709 versus low-input error 0.321. I183
  targets post-coupling depolarized calibration while preserving spike logits.
- PREMISE (2026-07-13): I182 tests a structural causal Laplace-like residual:
  fixed 2/8/32 ms exponential filters over completed latent features plus a
  learned soma decoder, against equal-size boxcar and scale-zero controls. All
  arms have 91,625 parameters, matched compute, and 120k full-raw training;
  advancement needs >=0.1 mV and exponential-over-boxcar evidence.
- EXPERIMENT (2026-07-13): I181 rejects hard event lifting. The straight-
  through arm changes voltage by +0.00069 mV and significantly regresses F1
  area by 0.00312; soft lifting changes voltage by +0.00277 and significantly
  regresses response by 0.00285 mV. Both miss the practical gate by two orders
  of magnitude, so no threshold/kernel tuning follows.
- PREMISE (2026-07-13): I181 tests the I180 scan's top structural idea: a
  straight-through hard predicted-event coordinate feeding a learned causal
  21 ms phase kernel, against soft-probability and scale-zero controls. All
  arms have 91,472 parameters and matched execution, initialization, 120k
  full-raw presentations, and diagnostics. Advancement still requires >=0.1
  mV plus no F1 failure.
- MAINTENANCE (2026-07-13): I180 pruned the rejected I171 fixed-point, I172
  multi-horizon, and I176 soft-waveform execution paths while retaining the
  replicated decomposition mechanism. AxoSim passes 105 tests; AxoBench passes
  131 with 1 skip; the cumulative-960k promoted checkpoint loads at 91,451
  parameters and exactly matches its I180 source state.
- PROMOTION (2026-07-13): I180 confirms full-raw adaptation at 300k. Versus
  matched clipped continuation it improves voltage/dynamics/response by
  0.83889/0.21680/0.42889, with all F1 deltas uncertain and below 0.0083.
  Absolute voltage is 15.95215 mV. The exact 91,451-parameter source is now
  the cumulative-960k stable checkpoint at SHA
  `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2`,
  and the reusable objective now trains on raw soma voltage.
- EXPERIMENT (2026-07-13): I180 also confirms clipped-base decomposition as a
  narrow event mechanism. Against its exact 102,471-parameter raw-base control
  it significantly improves F1 area/5 ms by 0.00762/0.01397 and response by
  0.01023 mV, while voltage/dynamics changes are negligible and uncertain.
  Keep it available for structural F1 work, but do not add it to the simpler
  promoted model.
- PREMISE (2026-07-13): I180 is the scheduled 300k raw-target long gate. The
  91,451-parameter clipped/half/full panel and the internally matched
  102,471-parameter raw-base/clipped-base decomposition pair all start from the
  cumulative-660k checkpoint and share 9,375 updates and one frozen schedule.
  Full raw must keep >=0.1 mV voltage gain without significant F1 failure;
  clipped decomposition must replicate its large F1 gain or be pruned.
- LITERATURE (2026-07-13): the I180 structural scan ranks causal hard event
  lifting, event-lifted latent consistency, and causal Laplace residual bases
  as the next high-temperature tests. It is stored at
  `literature/iteration-180-raw-hybrid-dynamics-scan.md`.
- REPLICATION (2026-07-13): I179 confirms pretrained raw adaptation almost
  exactly. Full raw improves voltage/dynamics/response by
  0.79434/0.19016/0.38573; half raw by 0.55525/0.13076/0.28618. All F1 deltas
  are uncertain and below 0.0071. Across I175/I179, full raw's mean voltage
  gain is 0.79368 mV with absolute 15.99742 mV average; it is the primary I180
  long-gate arm.
- EXPERIMENT (2026-07-13): I178 rejects clipped-base/raw-final decomposition
  as a voltage improvement but retains it for one long F1 gate. It improves
  voltage only 0.00246 mV, while significantly improving F1 area/5 ms by
  0.01608/0.02528 and response by 0.00646 mV; dynamics regresses 0.00222 and
  exact F1 0.00361 uncertainly. The structural event gain earns an I180 arm,
  not promotion or coefficient tuning.
- EXPERIMENT (2026-07-13): I177 rejects a dedicated 11k-parameter soma TCN.
  Corrected-soma and base-soma spike coupling regress voltage by
  0.00466/0.00500 mV; all other deltas are below 0.0034 and uncertain. Base
  coupling successfully isolates direct spike-logit movement but supplies no
  performance value. Added output-specific temporal capacity is not a useful
  raw-target lever.
- EXPERIMENT (2026-07-13): I176 rejects the AP waveform head under raw
  supervision. Detached changes voltage by -0.00008 mV; joint changes it by
  +0.00056 mV, with all six assessments uncertain and negligible. The scale-
  zero full-raw continuation reaches 15.98057 mV, only 0.01084 mV beyond I175,
  so ordinary raw adaptation is already near its short-run plateau.
- EXPERIMENT (2026-07-13): I175 pretrained raw adaptation is the provisional
  best. From the same cumulative-660k checkpoint and with matched 120k
  incremental compute, full raw improves voltage/dynamics/response by
  0.79302/0.18669/0.39861 versus clipped continuation, reaching absolute
  voltage 15.99141 mV; F1 area/exact/5 ms change
  -0.00250/-0.00478/-0.00682 uncertainly. Half raw gains 0.55630 mV voltage
  and 0.28518 mV response with similarly uncertain event changes. Carry full
  raw as the I180 long-gate lead; stable checkpoint remains unchanged until
  that gate.
- REPLICATION (2026-07-13): I174 confirms raw-target SERA gains. Full raw
  improves voltage/dynamics/response by 0.57356/0.09746/0.21712 but regresses
  F1 area/5 ms by 0.01211/0.02067 on this seed. Half raw improves SERA by
  0.31646/0.03163/0.12354 with uncertain F1 changes. Across I173-I174, half
  raw's mean delta favors all six core metrics; full raw maximizes SERA/exact
  but has a small mean 5 ms tradeoff. I175 tests matched stable-checkpoint
  adaptation before selecting the I180 long-gate recipe.
- EXPERIMENT (2026-07-13): I173 finds the first new large balanced lead. Full
  raw-target training improves voltage/dynamics/response SERA by
  0.54586/0.07290/0.20185 and F1 area/exact/5 ms by
  0.01652/0.01713/0.01180 versus the compute-matched clipped target. The 50/50
  mix also improves all six by 0.35816/0.03893/0.12881 and
  0.01873/0.00985/0.02094. Full raw leads SERA and exact F1; replicate it
  unchanged before promotion.
- EXPERIMENT (2026-07-13): I172 rejects multi-horizon physical-state
  supervision. Short 1/5/20 ms horizons regress voltage by 0.00393 mV while
  improving response 0.00805 mV and F1 area/exact/5 ms by
  0.01294/0.01363/0.00910. Multiscale 5/25/100 ms horizons regress voltage and
  response by 0.02088/0.02819 mV. Neither approaches 0.1 mV; close this path.
- PREMISE AUDIT (2026-07-13): The training target clips soma voltage at -55
  mV, while all core diagnostic SERA metrics use raw soma voltage. On 40 held-
  out traces, the optimized channel differs from the scored raw target by 5.43
  mV RMSE and up to 98.66 mV. I173 will directly train on the raw scored target
  with clipped and 50/50 controls; this is a metric-protocol correction, not a
  minor hyperparameter change.
- EXPERIMENT (2026-07-13): I171 rejects shared-weight fixed-point local
  refinement. Three feedback passes improve voltage by only 0.01075 mV and F1
  area/5 ms by 0.01645/0.01565, while response regresses 0.00807 mV. Two
  passes improve voltage only 0.00246 mV and regress response 0.02050 mV. The
  two- and three-call no-feedback controls agree within 0.00000014 mV,
  confirming the matched mechanism test; neither clears the 0.1 mV bar.
- PREMISE AUDIT (2026-07-13): Do not implement observed initial-voltage state.
  All 120 balanced held-out traces start at exactly -76.0 mV (zero variance),
  and an oracle calibration/test correction from initial voltage changes
  voltage SERA by exactly 0.0 mV after the 500 ms exclusion. The proposed
  observed-state feature contains no information in this dataset.
- EXPERIMENT (2026-07-13): I170 rejects frequency-specialized output heads at
  the scheduled 300k gate. High-pass features produce a practically material
  failure: voltage/response regress by 0.11651/0.09340 mV and exact F1 falls
  0.02079. An equal-parameter raw duplicate head improves voltage by 0.04626
  mV and response by 0.02240 mV, with all six point estimates favorable, but
  misses the 0.1 mV advancement threshold. Stable remains cumulative-660k.
- MAINTENANCE (2026-07-13): I170 pruned executable support and tests for the
  rejected I161-I170 distillation, residual-target, multigrid, ConvSSM,
  chunk-reset, selective-training, local-override, and frequency-head paths.
  Historical manifests, diagnostics, and negative results remain intact. The
  scheduled literature scan is in
  `literature/iteration-170-aggressive-dynamics-scan.md`.
- EXPERIMENT (2026-07-13): I169 rejects matched-compute long state continuity.
  Continuous 2,000 ms training improves dynamics by 0.00496 and all F1 point
  estimates, but regresses voltage by 0.00929 mV; response is unchanged. The
  effect is far below 0.1 mV despite exact token/update/position matching.
- EXPERIMENT (2026-07-13): I168 closes downward capacity scaling. The 51k
  model regresses all six metrics versus 91k, including voltage by 0.12052 mV;
  the 35k model regresses response by 0.13976 and all F1 summaries. The 91k
  budget is a robust short-run sweet spot, not simple overparameterization.
- EXPERIMENT (2026-07-13): I167 rejects scaling AxoMamba from 91,451 to
  207,449 parameters at 120k. The standard model beats wide/large on all six
  point estimates, significantly improving voltage by 0.07280 mV, response by
  0.00908, and exact F1 by 0.01803. Larger capacity is slower and still misses
  the 0.1 mV bar in the adverse direction.
- EXPERIMENT (2026-07-13): I166 rejects large-budget depth/width selection.
  At exactly 207,449 parameters, narrow/deep regresses voltage/response by
  0.01291/0.03498 versus wide/shallow while improving F1 area/exact/5 ms by
  0.04362/0.01595/0.05083. The allocation delta misses 0.1 mV and is a clear
  SERA/event tradeoff; deep is also 29% slower.
- EXPERIMENT (2026-07-12): I165 rejects post-hoc amplitude calibration. A
  balanced validation fit gives scale 0.95235 and bias 0.01536, not the roughly
  3x correction naively suggested by the gain diagnostic; on held-out tests it
  regresses voltage/dynamics/response by 0.02971/0.00442/0.01532 mV.
- EXPERIMENT (2026-07-12): I164 rejects equal-parameter local ConvSSM. It
  regresses voltage by 0.05799 mV versus scale zero while improving response by
  0.01074 and exact F1 by 0.01416; it misses the new 0.1 mV bar and loses to the
  simpler dilated TCN on voltage, dynamics, response, F1 area, and 5 ms F1.
- EXPERIMENT (2026-07-12): I163 decisively rejects assigning the true
  target-minus-trunk residual to the local path. Aligned supervision regresses
  voltage/dynamics/response by 0.37378/0.10568/0.24235 and F1
  area/exact/5 ms by 0.17817/0.11966/0.18976. Shuffling the residual causes
  only small regressions, identifying sample-specific decomposition as harmful.
- DECISION (2026-07-12): The zero-weight continuation gives a borderline but
  significant 0.01044 mV voltage and 0.00429 dynamics gain over I162, with all
  other point estimates favorable but uncertain. Retain cumulative-660k stable
  at SHA `eb55f57ce90a58b0ffaa57285d9ca32d07a9be5e2caa28e7bb00667702e44238`,
  but stop treating plain continuation as a substantive research direction.
- EXPERIMENT (2026-07-12): I162 rejects causal multigrid. Versus an executed
  scale-zero second path, coarse/full regresses voltage/response by
  0.00344/0.00093 while only improving F1 area by 0.00313; dual-full capacity
  changes voltage by 0.00002 and improves response by only 0.00117. Explicit
  coarse/fine separation adds no material value.
- DECISION (2026-07-12): The scale-zero duration control itself significantly
  improves I161 voltage/dynamics by 0.02648/0.01608 and F1 area/5 ms by
  0.01337/0.01801. Promote its cumulative-540k shared state after stripping
  inactive coarse modules with exact output identity. Stable remains 91,451
  parameters at SHA-256
  `b2b8a90792beaaafcd064caf34d3631e111116d84521d6e731e330e7acc72584`.
- EXPERIMENT (2026-07-12): I161 rejects mature-checkpoint latent/output
  anchoring and isolated refiner training. Versus joint continuation, latent
  anchoring regresses voltage/dynamics/response by 0.00758/0.00445/0.00551;
  output anchoring regresses them by 0.01311/0.00500/0.00965. Local-only
  regresses all six, including voltage by 0.04066 mV; trunk-only also loses.
- DECISION (2026-07-12): Joint 120k continuation from I160 significantly
  improves the source checkpoint's voltage/dynamics/response by
  0.04143/0.01590/0.02749 and F1 area/5 ms by 0.01763/0.01734; exact F1 gains
  0.00807 uncertainly. Promote the cumulative-420k checkpoint at SHA-256
  `6fc514e4c217e021e00816ed602d07d66361cc0482fcf0c63179b7d916fd9884`.
- EXPERIMENT (2026-07-12): I160 long-gates post-TCN at 300k. Dilated improves
  voltage/response SERA by 0.05636/0.03860 mV and exact F1 by 0.01517
  significantly; dynamics improves 0.00911 uncertainly, while F1 area and
  5 ms F1 change by -0.00968/-0.01140 uncertainly. Undilated improves all
  three SERA axes significantly by 0.02554/0.02052/0.02959 mV but has no
  significant F1 gain.
- DECISION (2026-07-12): Promote post-dilated under the declared SERA-first
  policy. It adds no diagnostic trigger and improves voltage null skill and
  residual lag, though spike-count error and refractory violations worsen.
  Stable AxoMamba is now 91,451 parameters, trained for 300k presentations,
  checkpoint SHA-256 `b8b06ba1d93f7bfae9c30c297e6c5fc0e7e7d2e98dffcfa72c232a1e163bc78c`.
- CLEANUP (2026-07-12): During I160, removed executable config, modules,
  helpers, runner support, and tests for rejected I151-I154 membrane-state,
  branchwise-decay, regime-expert, and curriculum paths. Historical artifacts
  remain; AxoSim passes 101/101, AxoBench 131 with 1 skip, stable I140 loads,
  and `git diff --check` passes.
- LITERATURE (2026-07-12): The I160 scan ranks explicit residual-refiner
  optimization first after a successful long promotion, followed by a causal
  multigrid post refiner. See
  `literature/iteration-160-post-refinement-scan.md`.
- EXPERIMENT (2026-07-12): I159 validates both post-TCN variants. Dilated
  improves voltage/dynamics/response by 0.12610/0.03857/0.10343 and F1
  area/exact/5 ms by 0.04256/0.03776/0.02821. Undilated improves all six too,
  leading dynamics/F1 area/5 ms while trailing voltage/response/exact slightly.
- DECISION (2026-07-12): I160 long-gates post-dilated and post-undilated versus
  exact scale zero at 300k. During the run, prune rejected I151-I154 mechanisms
  and complete the scheduled structural literature scan.
- EXPERIMENT (2026-07-12): I158 confirms post-trunk fusion: voltage/dynamics/
  response improve significantly by 0.07419/0.01979/0.04608, with F1 area and
  exact up 0.01376/0.01471 significantly and 5 ms up 0.00517 uncertainly.
  Parallel flips to a 0.02316 mV voltage regression; interlayer regresses
  response and 5 ms F1. Two-seed post voltage mean is -0.11369 mV.
- DECISION (2026-07-12): Lock post fusion as the I160 placement. I159 tests
  post-dilated versus post-undilated equal capacity on a fresh seed to decide
  whether multiscale dilation belongs in the compact long gate.
- EXPERIMENT (2026-07-12): I157 finds a substantially stronger fusion
  placement. Post-trunk TCN significantly improves voltage/dynamics/response by
  0.15320/0.03584/0.07724 and F1 area/exact/5 ms by
  0.04241/0.03705/0.03835. Interlayer fusion also improves all six metrics,
  including voltage by 0.12806 mV; parallel replicates at 0.07586 mV.
- DECISION (2026-07-12): Provisional post fusion earns an unchanged I158
  replication against interlayer, parallel, and scale zero. No new operator or
  hyperparameter change is allowed before placement robustness is known.
- EXPERIMENT (2026-07-12): I156 replicates dilated local/global TCN: voltage
  -0.03850 mV, dynamics -0.00397 mV/ms, response -0.05729 mV, F1 area
  +0.03594, exact +0.02646, and 5 ms +0.02660, all significant except the
  already favorable dynamics assessment is uncertain. Two-seed mean voltage
  and response gains are 0.06815 and 0.04557 mV.
- DECISION (2026-07-12): Lock the local TCN family for I160 long gating.
  Dilation beats undilated on both SERA axes on both seeds. I157 tests fusion
  placement with identical operator parameters: parallel, interlayer, or
  post-Mamba refinement.
- EXPERIMENT (2026-07-12): I155 advances the local/global TCN operator. The
  dilated path significantly improves voltage by 0.09780 mV, response by
  0.03385 mV, and F1 area/exact/5 ms by 0.02560/0.01712/0.03032. Undilated
  capacity is also strong at -0.08918 mV voltage and -0.02409 mV response,
  leaving a smaller but favorable multiscale increment.
- MECHANISM (2026-07-12): Dilated TCN raises voltage gain 0.02073 and lowers
  residual lag 0.01024, but high-dV/dt and near-spike ratios worsen slightly.
  The gain is broad local trajectory modeling rather than AP-core repair. I156
  repeats the exact three-arm panel on a fresh seed before any modification.
- EXPERIMENT (2026-07-12): I154 rejects compositional objective curricula.
  Event-first and voltage-event-joint significantly regress all six metrics.
  Voltage-first significantly improves F1 area/exact/5 ms by
  0.02436/0.01079/0.03046 but regresses voltage/dynamics/response by
  0.09259/0.01305/0.01243, a severe version of the existing frontier.
- DECISION (2026-07-12): Do not tune phase lengths or subtask weights. I155
  tests a MambaNO/NeuronIO-motivated full local/global operator: causal
  multiscale dilated TCN in parallel with the stable global Mamba trunk.
- EXPERIMENT (2026-07-12): I153 rejects learned hybrid regimes. The gate is
  active and change-sensitive, but versus scale zero it regresses voltage by
  0.00311 mV and dynamics by 0.00483 mV/ms while improving response by 0.01539
  mV and exact F1 by 0.01282. A fixed expert mixture has the better, still tiny
  voltage point estimate (-0.00684 mV), so learned switching adds no value.
- DECISION (2026-07-12): Close regime experts without gate tuning. I154 tests
  literature-backed compositional pretraining within the same 120k budget:
  voltage-flow and event-timing subtasks before the unchanged joint objective.
- EXPERIMENT (2026-07-12): I152 rejects branchwise membrane integration and
  closes the family. Heterogeneous branch decays significantly regress voltage
  by 0.02172 mV while improving response by 0.00489 mV and all F1 summaries;
  shared decay regresses voltage by 0.02698 mV. The mixed I151 mechanism also
  flips to a 0.01312 mV voltage regression on the fresh seed.
- DECISION (2026-07-12): Do not tune or replicate explicit membrane states.
  I153 tests the I150 scan's next distinct model class: learned causal
  subthreshold/event regime experts between AxoMamba's two temporal layers.
- EXPERIMENT (2026-07-12): I151 validates explicit leaky membrane integration
  but misses the aggressive bar. It significantly improves voltage/dynamics/
  response by 0.02589/0.01491/0.00850 mV and exact F1 by 0.01813, while the
  matched instantaneous decoder changes voltage by only -0.00020 mV and
  significantly regresses response by 0.00111 mV.
- MECHANISM (2026-07-12): The leaky arm learns active, diverse 2.94-54.00 step
  time constants and a 0.341 decoder norm; voltage gain and residual lag improve
  slightly, but high-dV/dt and near-spike ratios do not. I152 moves temporal
  integration before branch mixing and tests heterogeneous versus shared decay.
- EXPERIMENT (2026-07-12): I150 rejects extra task-head temporal depth at the
  300k gate. Shared depth significantly regresses voltage by 0.08268 mV and
  intervention response by 0.00903 mV despite improving dynamics by 0.02302
  mV/ms and F1 area/exact/5 ms by 0.02044/0.01380/0.02405. Specialized depth
  also regresses voltage by 0.04054 mV.
- DECISION (2026-07-12): Keep deployed I140 unchanged and remove all I149
  task-head code. Because temporal depth failed at deployment budget, I151
  tests an explicit leaky membrane-state backbone plus learned residual rather
  than the literature scan's conditional fast/slow Mamba path.
- CLEANUP (2026-07-12): During I150, removed executable configuration,
  modules, forward paths, and tests for rejected I141-I148 route residual,
  forcing/corrector, route partition/moment, and morphology-readout families.
  AxoSim passes 101/101, AxoBench passes 131 with 1 skip, `git diff --check`
  passes, and stable I140 loads at 77,125 parameters and 300k presentations.
  After the gate rejected I149, its task-head path/test was also pruned;
  AxoSim then passes 100/100.
- LITERATURE (2026-07-12): The I150 scan ranks parallel learned fast/slow latent
  dynamics first if shared temporal depth passes the long gate, followed by
  trajectory-regime experts and an explicit leaky-membrane hybrid backbone.
  See `literature/iteration-150-multiscale-hybrid-dynamics-scan.md`.
- EXPERIMENT (2026-07-12): I149 rejects task specialization but advances shared
  temporal depth to the long gate. Specialized heads significantly regress
  voltage by 0.01800 mV despite improving response by 0.02294 mV and all F1
  summaries. The shared correction significantly improves voltage by 0.05033
  mV and F1 area/exact/5 ms by 0.04295/0.02279/0.04312; its 0.00337 mV response
  regression is uncertain.
- DECISION (2026-07-12): I150 long-gates shared temporal depth against exact
  scale zero and specialized heads at 300k presentations on a fresh seed. The
  shared arm must retain a material balanced lead; task specialization is only
  a response-tradeoff control and is not eligible on its I149 evidence.
- EXPERIMENT (2026-07-12): I148 rejects morphology-specific readout experts.
  Specific experts regress voltage 0.10337 mV, response 0.01854 mV, F1 area
  0.02094, exact F1 0.01606, and 5 ms F1 0.01980 significantly versus scale
  zero. Shared experts also regress voltage/response/F1.
- VALIDITY (2026-07-12): Separately trained specific and cyclic-shuffled expert
  arms are permutation-symmetric and become numerically identical, so that
  control cannot test assignment identity. This does not affect rejection:
  both active modes lose badly to the valid scale-zero control. I149 tests a
  larger task-specific temporal-head architecture to address SERA/F1 conflict.
- EXPERIMENT (2026-07-12): I147 rejects route-conditioned branch moments.
  Real occupancy/multiplicity/length moments improve dynamics 0.00972 mV/ms,
  but significantly regress voltage 0.00870 mV and response 0.01177 mV. Real
  and shuffled moments are nearly indistinguishable, so source identity is not
  preserved by the aggregated moment representation.
- DECISION (2026-07-12): Close route-residual extensions; keep only I140's
  static per-synapse occupancy operator. I148 targets persistent 1.44-1.46x
  morphology error disparity with hidden-dependent morphology-specific readout
  experts, against shared, shuffled-assignment, and scale-zero controls.
- EXPERIMENT (2026-07-12): I146 validates soft route partitioning but misses the
  aggressive voltage bar. Real used/unused separation improves all six point
  estimates versus scale zero: voltage -0.01737 mV, dynamics -0.01025 mV/ms,
  response -0.02213 mV, F1 area +0.01027, exact +0.01129, and 5 ms +0.01212;
  voltage/dynamics/response/exact are significant.
- DECISION (2026-07-12): Real partition strongly beats shuffled on response and
  F1, but voltage gain is only 0.017 mV and it does not beat duplicate-total
  capacity on voltage. Do not replicate binary partition. I147 uses the full
  I139 mechanism as branch moments: used occupancy, multiplicity, and target
  length, with duplicate-total and shuffled-feature controls.
- EXPERIMENT (2026-07-12): I145 rejects the error-driven predictor-corrector.
  Error feedback improves dynamics 0.00740 mV/ms and F1 area/exact/5 ms by
  0.02033/0.02384/0.02068, but regresses response 0.00894 mV and voltage
  0.00495 mV uncertainly. State feedback has the same response tradeoff;
  forcing-only feedback is balanced but only improves voltage 0.00576 mV.
- DECISION (2026-07-12): Do not tune correction dimension/scale. I146 returns
  to the robust route-occupancy mechanism: preserve the total generic path but
  separately project used and unused source streams, directly testing whether
  soft two-stream integration fixes hard-mask failure.
- EXPERIMENT (2026-07-12): I144 rejects a learned joint forcing basis. Joint
  change/interaction significantly improves dynamics 0.01130 mV/ms and F1
  area/exact/5 ms by 0.02694/0.02677/0.04001, but significantly regresses
  voltage 0.01267 mV and response 0.00618 mV. It fails the balanced criterion.
- DECISION (2026-07-12): Duplicated change improves all six point estimates,
  including voltage 0.02591 mV and dynamics 0.01687 mV/ms, but remains below
  the aggressive practical target and follows two source-reversal seeds. Close
  forcing-basis tuning. I145 moves to a predictor-corrector transition between
  the two existing Mamba blocks, the next literature-ranked structural change.
- REPLICATION (2026-07-12): I143 finds forcing value but rejects a fixed source.
  Interaction flips to +0.01411 mV voltage regression while significantly
  improving response 0.00827 mV and F1 area/exact/5 ms by 0.01455/0.01207/
  0.01703. Plain change flips from I142 failure to significant voltage,
  dynamics, and response gains of 0.05212/0.00859/0.01609 with uncertain F1.
- DECISION (2026-07-12): Neither hand-selected basis is seed-stable, but their
  complementary voltage/event profiles support a learned joint forcing basis.
  I144 concatenates change and change-by-state, with duplicated-change,
  duplicated-interaction, and scale-zero controls at exact parameters/compute.
- EXPERIMENT (2026-07-12): I142 finds a balanced pre-Mamba forcing lead.
  Change-by-state interaction improves voltage 0.06131 mV and response 0.02717
  mV significantly versus scale zero; dynamics improves 0.00149 mV/ms, F1 area
  0.00319, and 5 ms F1 0.00696 uncertainly, while exact F1 falls 0.00867
  uncertainly. Plain state and plain change residuals regress core SERA.
- DECISION (2026-07-12): Interaction beats the state-capacity control by
  0.07836 mV voltage and 0.04674 mV response on point estimates and is not a
  generic-width effect. I143 repeats the unchanged four-arm panel on a fresh
  seed; no width/scale tuning precedes replication.
- EXPERIMENT (2026-07-12): I141 rejects dual-route residual fusion. Real
  physical routing significantly regresses voltage 0.03851 mV versus scale
  zero while improving F1 area 0.01083 and 5 ms F1 0.02001. It is also 0.03135
  mV worse than the duplicate-generic capacity path, which improves all F1
  metrics more strongly.
- DECISION (2026-07-12): The local physical route does not beat generic
  capacity or shuffled routing and fails the predeclared 0.10 mV/balanced bar.
  Do not tune fusion width/scale. I142 tests explicit branch-change forcing
  before Mamba state evolution, motivated by graph-integrator literature and
  distinguished from the rejected output-only I134 decoder.
- PROMOTION (2026-07-12): I140 promotes the compact route-occupancy operator.
  At 300k it reaches voltage SERA 16.91837 mV, dynamics 21.25254 mV/ms,
  response 17.48370 mV, F1 area 0.35302, exact F1 0.18678, and 5 ms F1
  0.42604. It beats scale-zero on all six metrics, including voltage by
  0.19773 mV and F1 area by 0.07157, and beats shuffled occupancy on all six.
- DECISION (2026-07-12): Full geometry significantly loses 0.00839 mV voltage
  versus compact occupancy and is otherwise equivalent, confirming geometry is
  unnecessary. The exact I140 occupancy model is deployed as the 77,125-
  parameter `checkpoints/axomamba.pt`, SHA-256 `af315706...`, with output-
  identical promotion. I141 tests literature-ranked dual-route residual fusion.
- CLEANUP (2026-07-12): During I140, removed rejected executable paths and
  tests for hard route validity, tree event propagation, morphology-specific
  physical input routing, event/change and voltage-state decoders, and
  morphology Mamba adapters. Historical manifests/diagnostics remain and old
  configs load through retired-key compatibility. AxoSim passes 100/100,
  AxoBench 131 passed/1 skipped, and the stable 76,949-parameter checkpoint
  loads after cleanup.
- LITERATURE (2026-07-12): Added
  `literature/iteration-140-dual-path-dynamics-scan.md`. It ranks a dual-route
  local/global operator first: preserve the stable generic route and add exact
  physical routing as a separately projected residual, with generic-capacity,
  shuffled-route, and scale-zero controls. This directly addresses I135's
  unstable route replacement and I139's robust route-occupancy mechanism.
- EXPERIMENT (2026-07-12): I139 identifies the coordinate operator's useful
  mechanism as discrete route occupancy/multiplicity. That ablation is
  statistically indistinguishable from all nine features on five metrics and
  improves response SERA 0.00481 mV; continuous geometry instead matches the
  shuffled control, regressing voltage about 0.185 mV and F1 area about 0.083.
- DECISION (2026-07-12): I140 runs a 300k gate over real occupancy, full
  features, shuffled occupancy, and exact scale-zero compute control. Promote
  only if the real route prior remains balanced at the long budget; prefer the
  compact occupancy mechanism if it stays equivalent to full features.
- EXPERIMENT (2026-07-12): I138 rejects morphology-conditioned Mamba LoRA.
  Specific rank-4 adapters improve F1 area 0.01136, exact F1 0.00993, and
  5 ms F1 0.01169 significantly, but regress voltage 0.01163 mV significantly.
  Shared adapters regress voltage/response more, so conditioning is real but
  not balanced.
- DECISION (2026-07-12): Do not tune adapter rank/scale. I139 ablates the
  robust coordinate operator into occupancy/multiplicity, continuous geometry,
  validity, full, and shuffled feature families before I140's long gate.
- EXPERIMENT (2026-07-12): I137 rejects causal voltage-state residuals. Leaky
  state improves voltage 0.03145 mV and exact F1 0.01956 significantly but
  loses 5 ms F1 0.02144 significantly; direct residual has a similar tradeoff
  and cumulative integration regresses voltage/response severely.
- DECISION (2026-07-12): Do not tune state decay/kernel. I138 tests
  morphology-conditioned rank-4 adapters inside each Mamba layer against
  shared-adapter and scale-zero controls on the robust coordinate baseline.
- EXPERIMENT (2026-07-12): I136 rejects physical-route/change composition.
  Physical change now regresses voltage 0.05746 mV and all F1 metrics
  significantly versus generic control. Generic change also regresses voltage
  and F1; physical route alone improves response/5 ms F1 but not voltage.
- DECISION (2026-07-12): Exact source correspondence remains validated because
  shuffled routes fail catastrophically on both seeds, but route/change effects
  are seed-unstable. Close the family without normalization or scale tuning.
  I137 moves to a neural-ODE-style voltage-state residual decoder.
- EXPERIMENT (2026-07-12): I135 finds a physical-routing/change interaction.
  Both generic change and physical route alone regress SERA on seed 20260762,
  but their combination improves voltage 0.01776 mV, response 0.02164 mV,
  and all F1 metrics significantly versus generic control. Directly versus
  physical route alone, change improves five metrics significantly, including
  voltage 0.07333 mV and response 0.05189 mV.
- DECISION (2026-07-12): Source-shuffled physical routing catastrophically
  fails, validating exact segment correspondence, but the composition is an
  interaction rather than two stable main effects. I136 repeats the unchanged
  2x2-plus-shuffled panel on a fresh seed before further extension.
- EXPERIMENT (2026-07-12): I134 validates an explicit branch-change decoder.
  Versus exact coordinate compute control it significantly improves all six
  core metrics: voltage -0.10956 mV, dynamics -0.01293 mV/ms, response
  -0.06836 mV, F1 area +0.02492, exact F1 +0.01202, and 5 ms F1 +0.03070.
- DECISION (2026-07-12): Direct change-vs-state comparison shows change
  specifically improves dynamics 0.01124 mV/ms and response 0.06870 mV
  significantly, while state bypass retains a 0.00856 exact-F1 edge. I135
  replicates change and tests exact target-segment routing into branches.
- EXPERIMENT (2026-07-12): I133 rejects hard route masking. Real mask plus
  coordinates significantly regresses voltage 0.07996 mV, response 0.04847
  mV, and exact F1 0.01804 versus coordinate-only. A shuffled mask is far
  worse, confirming route validity is real but learned soft weighting is needed.
- DECISION (2026-07-12): A proposed geometry-subunit follow-up was canceled
  before training because its reconstructed mask differed from existing
  contiguous subunits in only 110/19,470 entries. I134 instead tests an
  explicit event/change decoder against a matched state-bypass decoder.
- EXPERIMENT (2026-07-12): I132 fully replicates I131 on a fresh seed. Real
  coordinates beat scale-zero significantly on all six metrics: voltage
  -0.21223 mV, dynamics -0.01339 mV/ms, response -0.05086 mV, F1 area
  +0.08737, exact F1 +0.05599, and 5 ms F1 +0.09170.
- DECISION (2026-07-12): Reject the current tree event-message residual. The
  real graph adds only 0.00448 mV voltage and 0.00395 F1 area to coordinates;
  rewiring significantly harms voltage/response, so topology matters but the
  operator is practically weak. I133 directly masks morphology-unused source
  events, motivated by the I131 feature audit.
- EXPERIMENT (2026-07-12): I131 validates normalized route coordinates. Real
  coordinates beat the exact within-type shuffled control significantly on
  all six core metrics: voltage -0.20633 mV, dynamics -0.00960 mV/ms,
  response -0.04473 mV, F1 area +0.08387, exact F1 +0.05322, and 5 ms F1
  +0.08092. The direct scale-zero comparison is likewise significant on all
  six axes, with voltage -0.20765 mV and response -0.04467 mV.
- DECISION (2026-07-12): The coordinate effect is the largest balanced recent
  structural result but only about 1.2% of remaining voltage error. I132
  replicates it on a fresh seed and adds verified SWC-tree event propagation
  against a degree-preserving rewired graph under exact parameter/compute
  controls.
- EXPERIMENT (2026-07-12): I131 reconstructed all five retained SWC route
  maps and matched segment/source summaries for every one of 1,285 metadata
  records. The frozen artifact contains real, within-type shuffled, and
  constant-coordinate tensors with shape `(5, 1278, 9)`.
- IMPLEMENTATION (2026-07-12): Added a zero-initialized shared coordinate MLP
  over routed synapses and strict hashed NPZ buffer loading in the manifest
  runner. The real/shuffled/constant/scale-zero panel has exactly 77,125
  parameters per model, matched compute, and bit-identical initial outputs.

- MILESTONE (2026-07-12): I120 promotes learned pre-aggregation synaptic
  efficacy after the exact 300k gate: voltage -0.07889 mV, dynamics -0.00343
  mV/ms, response -0.01244 mV, and 5 ms F1 +0.01586, all significant; F1 area
  +0.00745 uncertain and exact F1 -0.00123 uncertain. The deployed AxoMamba is
  now the exact 70,459-parameter I120 checkpoint with absolute voltage SERA
  17.30509 mV. I121 tests literature-ranked synapse-specific temporal bases.

- LITERATURE (2026-07-12): Added
  `literature/iteration-120-synapse-structure-scan.md`. It ranks a low-rank
  heterogeneous-synapse temporal basis first, morphology-conditioned synaptic
  dynamics second, and multiple sparse nonlinear subunits per branch third.
  The queue combines HetSyn, structured dendritic networks, NOBLE, and
  state-space neural-operator evidence with the I119 mechanism audit.
- CLEANUP (2026-07-12): During the I120 long run, removed executable config,
  modules, runner logic, and tests for rejected temporal bilinear filtering,
  bilinear modulation, branch forcing reinjection, state gating, dynamic
  routing, grouped Mamba, latent/output distillation, and latent branch tokens.
  Historical manifests and diagnostics remain. AxoSim/AxoBench tests pass 93/93
  after pruning; the promoted 69,161-parameter checkpoint loads through explicit
  retired-key compatibility.

- EXPERIMENT (2026-07-12): I119 finds a balanced exact-controlled synaptic-
  efficacy lead: voltage -0.03281 mV, response -0.00607 mV, F1 area +0.01715,
  and 5 ms F1 +0.01849, all significant. The active gains are heterogeneous
  (std 0.111, range 0.731-1.581), while the zero-scale control reproduces the
  promoted model within 3.5e-8 mV. I120 runs the unchanged 300k gate plus
  mandatory cleanup and literature scan.

- REPLICATION (2026-07-12): I118 rejects both global-depth leads. Exact-capacity
  voltage/response reverse to +0.04222/+0.00899 mV error; the large model
  reverses to +0.06459 mV voltage error. Fixed sparse latent tokens improve
  response 0.01181 mV but significantly lose 0.01431 F1 area and 0.02144 at
  5 ms. I119 moves upstream to exact-controlled learned synaptic efficacy before
  branch summation, present in the biological ELM but missing from AxoMamba.

- EXPERIMENT (2026-07-12): I117 rejects learned latent branch slicing despite
  a large active update (71% of base projection RMS): assignments collapse to
  0.996 normalized entropy and the exact-parameter global Mamba dominates it on
  voltage/response at similar F1. A 96,706-parameter three-layer global model
  improves voltage 0.01650 mV, response 0.00888 mV, F1 area 0.02243, and 5 ms
  F1 0.03851. A compute-matched 222,881-parameter model improves those by
  0.03469 mV, 0.01226 mV, 0.04528, and 0.06133. I118 replicates both global
  leads and tests fixed sparse token partitions against learned slicing.

- USER (2026-07-12 16:24): Raised the target from modest `1e-3`-scale voltage
  movements to aggressive reductions representing a visible fraction of the
  roughly 20 mV base error; explicitly requested latent regularization or
  distillation plus larger literature-backed structural changes.
- EXPERIMENT (2026-07-12): Iteration 116 rejects output, latent, and combined
  distillation from the promoted 300k teacher. Output-only improves voltage by
  0.01261 mV but loses 0.01402 F1 area and 0.02004 at 5 ms; latent matching
  regresses voltage by 0.04496 mV. The latent loss itself converges, so this is
  a mechanism failure rather than an inactive adapter. I117 moves to a
  LaMO-derived multi-token branch operator and real widened-Mamba control.
- LITERATURE (2026-07-12): Added
  `literature/iteration-116-aggressive-architecture-scan.md`, ranking a learned
  latent branch-token Mamba operator first based on LaMO, SS-NO, S-NOT, KRNO,
  and structured dendritic connectivity. The acceptance target is now a
  meaningful hundredth-to-tenth-scale change, not significance alone.

- EXPERIMENT (2026-07-12): Iteration 109 finds a real but unbalanced rank-4
  branch interaction. Active bilinear integration improves voltage 0.01643 mV,
  F1 area 0.02501, and 5 ms F1 0.02689 significantly, but response regresses
  0.01499 mV. I110 runs an unchanged 300k closure; promotion requires the
  response failure to disappear, followed by mandatory cleanup/literature.
- EXPERIMENT (2026-07-12): Iteration 108 rejects both hybrid orders.
  Mamba-then-GRU improves voltage/response but loses 0.03800 F1 area and
  0.05448 at 5 ms significantly; GRU-then-Mamba regresses voltage 0.02970 mV
  and response 0.01360 mV significantly. I109 moves to exact-budget rank-4
  bilinear branch cooperation, with an inactive state-reallocation control.
- REPLICATION (2026-07-12): Iteration 107 rejects Mamba-then-GRU: voltage
  regresses 0.01213 mV and F1 area/5 ms fall 0.02975/0.03618 significantly,
  despite response/dynamics and speed gains. I108 performs one exact-budget
  structural order test, GRU-then-Mamba, with the failed tail and promoted
  control on the same fresh schedule; no state-size tuning follows failure.
- EXPERIMENT (2026-07-12): Iteration 106 finds a strong exact-budget
  Mamba-GRU hybrid. Voltage (-0.03944 mV) and response (-0.02058 mV) improve
  significantly/materially; all F1 estimates rise, exact by 0.01080, and model
  compute falls 3.8%. The GRU update is 29.3% of hidden RMS. I107 is an
  unchanged fresh-seed replication before any refinement.
- HYPOTHESIS (2026-07-12): Iteration 106 tests an exact-budget Mamba-GRU
  hybrid: one state-54 Mamba block plus a standard GRU residual tail equals the
  promoted two-Mamba trunk at 69,377 parameters and is 14% faster in the step
  benchmark. This follows a decisive Mamba-2 rejection (voltage +0.05950 mV,
  F1 area -0.02685, 5 ms F1 -0.03047, all significant).
- HYPOTHESIS (2026-07-12): Iteration 105 tests official Mamba-2 as a genuine
  recurrence replacement. A 64/32/head-32 trunk plus 376 inert budget
  parameters exactly matches promoted Mamba-1 at 69,377; measured step latency
  is about 2x. This follows I104's failed delay-observability premise, which
  worsened held-out voltage/dynamics residual RMSE and all five morphologies.
- REPLICATION (2026-07-12): Iteration 103 rejects 5/50 ms branch memory.
  Seed-1's coherent profile reverses: response significantly regresses 0.00396
  mV, dynamics/F1 area/5 ms move adversely, and all effects are sub-material.
  No tau/scale tuning. I104 audits conditional delayed-input information before
  any high-order architecture is built.
- EXPERIMENT (2026-07-12): Iteration 102 provisionally advances the unchanged
  branch-memory bank once. All six core estimates improve; voltage (-0.00853
  mV) and response (-0.00756 mV) are significant, while F1 area rises 0.00428.
  These are individually sub-material, but learned fast/slow weights are
  heterogeneous and the overhead is only 1.21x. I103 is a strict replication,
  not a tau/scale sweep.
- HYPOTHESIS (2026-07-12): Iteration 102 replaces the slow scalar local Mamba
  with branch-specific learned mixing of fixed 5/50 ms causal memories before
  aggregation. Control/active are exact at 69,613 parameters and matched filter
  compute; measured overhead is only 1.21x. Official Mamba-3 is separately
  kernel-gated on the RTX 5090 and has no AxoBench quality result yet.
- EXPERIMENT (2026-07-12): Iteration 101 rejects scalar shared branch-local
  Mamba. It costs 5.3x step latency, leaves F1 area/5 ms exactly unchanged, and
  produces only sub-material significant voltage (+0.00362 mV) and response
  (+0.00113 mV) regressions. The learned local correction is active but only
  1.06% of branch-input RMS. Do not tune it; audit official Mamba-3 next.
- HYPOTHESIS (2026-07-12): Iteration 101 tests shared branch-local Mamba state
  before cross-branch projection. Inactive and active candidates both execute
  the same local path and have 69,552 parameters; the active path costs 5.3x
  promoted step latency, so only a material core/robustness gain can justify it.
- LITERATURE (2026-07-12): Completed the scheduled large iteration-100 scan in
  `literature/iteration-100-large-scan.md`. After excluding mechanisms already
  falsified in iterations 85-99, it ranks shared branch-local temporal state
  before cross-branch aggregation first, an official Mamba-3 recurrence/MIMO
  upgrade second, and high-order delay/state observation behind a premise
  audit. Iteration 101 starts with the compartment-local hypothesis.
- VERIFICATION (2026-07-12): Iteration-100 cleanup passes AxoSim 88/88 and the
  AxoBench evaluator 24/24. The promoted checkpoint loads as a 69,377-parameter
  five-morphology AxoMamba trained for 300k presentations on seed 20260727 with
  near-spike weight 0.002 and radius 5.
- CLEANUP (2026-07-12): Removed all rejected I91/I96-I99 executable paths:
  unsigned input-load conditioning, liquid soma state, six-state current ports,
  grouped/polarity Mamba, and their config/tests. Retained only historical
  manifests/diagnostics and explicit retired-key checkpoint compatibility.
- EXPERIMENT (2026-07-12): Iteration 100 promotes near-spike voltage weight
  0.002 after the fresh-seed 300k gate. It improves dynamics SERA by 0.00349
  mV/ms, response SERA by a material 0.01663 mV, F1 area by 0.01018, and 5 ms
  F1 by 0.01904, all significant. Voltage changes by +0.00500 mV and exact F1
  by -0.00041, both uncertain and practically negligible. The reusable default
  recipe and `checkpoints/axomamba.pt` now contain the promoted objective.
- EXPERIMENT (2026-07-12): Iteration 99 rejects exact-budget grouped Mamba and
  polarity ports. Grouping alone materially regresses voltage/5 ms F1;
  separate E/I ports regress voltage by 0.03421 mV and response by 0.00772 mV,
  while signed ports significantly lose F1 area/5 ms. Iteration 100 will
  long-gate fixed near-spike weight 0.002, then run mandatory failed-path
  cleanup and the scheduled large literature scan.
- REPLICATION (2026-07-12): Iteration 98 rejects pre+post hybrid-current
  ports. Their seed-1 response/dynamics benefit vanishes and voltage regresses
  materially by 0.02464 mV; only exact F1 improves. The standalone six-state
  BSSM also fails its predeclared speed gate at 174x AxoMamba step latency.
  Iteration 99 tests an exact-parameter fused alternative: two Mamba state
  groups with separate E/I ports versus signed and inactive grouped controls.
- EXPERIMENT (2026-07-12): Iteration 97 selects six-state pre+post current
  ports for replication. They materially improve response by 0.01440 mV and
  dynamics by 0.00523 mV/ms, keep voltage neutral, and raise F1 area/exact
  uncertainly without significant core regression. The learned port is large
  (83% of base-hidden RMS) and dominated by the interpretable 20 ms membrane
  state. Iteration 98 is an unchanged fresh-seed replication.
- EXPERIMENT (2026-07-12): Iteration 96 rejects the multi-timescale liquid
  soma readout. Adaptive mixing materially improves voltage by 0.02117 mV but
  significantly regresses response by 0.01154 mV; fixed smoothing also harms
  response. The adaptive gate was active and collapsed raw voltage occupancy
  to 0.27%, so this is a mechanism failure, not failed optimization. Iteration
  97 moves to six-state fast/slow E/I current ports before/after Mamba.
- DECISION (2026-07-12): Iteration 95 ends the near-spike tuning batch. Weight
  0.002 replicates material voltage (-0.02357 mV) and response (-0.00655 mV)
  gains, but F1 movement is tiny/mixed. Retain it unchanged as a possible
  iteration-100 long control; do not tune it further. Iteration 96 pivots to a
  literature-derived multi-timescale liquid soma state.
- EXPERIMENT (2026-07-12): Iteration 94 rejects near-spike weight 0.003 after
  a third-seed significant response regression, but identifies 0.002 as a
  balanced lower-weight lead: voltage, dynamics, and response SERA improve
  significantly while all F1 point estimates improve uncertainly. Iteration
  95 will fresh-seed bracket 0.0015/0.002/0.0025 before any long gate.
- REPLICATION (2026-07-12): Iteration 93 reverses iteration 92's F1 tradeoff
  on a fresh seed. Near-spike weight 0.003 with the unchanged F1 objective
  significantly improves voltage and response SERA plus F1 area and 5 ms F1;
  exact F1 improves uncertainly and dynamics regresses only 0.00056
  uncertainly. Stronger F1 coefficients add no balanced advantage. Iteration
  94 will run a third-seed 0.001/0.002/0.003 robustness bracket.
- EXPERIMENT (2026-07-12): Iteration 92 finds a strong but unbalanced
  near-spike-voltage objective. Weight 0.003 significantly improves voltage,
  dynamics, and response SERA by 0.02269, 0.00405, and 0.00863 respectively,
  but significantly loses 0.01605 F1 area and 0.02697 at 5 ms. Larger weights
  deepen the SERA gain and F1/count/refractory tradeoff. Iteration 93 will run
  one fresh-seed F1-compensation bracket around the smallest weight.
- EXPERIMENT (2026-07-12): Iteration 91 rejects unsigned input-load
  conditioning. A premise audit found only 6.1% E/I event-mass cancellation
  and no held-out residual value from separate E/I currents, while layer-1
  Mamba timestep selection was nearly invariant to event load. Exposing load
  to that layer produced a sharp frontier: scale 0.025 improved voltage SERA
  by 0.04992 mV but significantly regressed response SERA, F1 area, and 5 ms
  F1; every other scale also failed a core gate. Queue the conditioner for
  iteration-100 pruning and do not tune its coefficient.
- CLEANUP (2026-07-12): Iteration 90 removed the complete rejected I81-I90
  executable branch: intervention schedule mixing and paired-response
  objectives, filtered spike-train losses, parallel fast/slow Mamba, and
  frequency-band voltage losses. Their manifests, diagnostics, and written
  negative results remain as evidence. No retired symbol remains in
  `src/`, `scripts/`, `tests/`, or `configs/`.
- VERIFICATION (2026-07-12): Post-cleanup AxoSim passes 86/86, the AxoBench
  iteration evaluator passes 24/24, and `checkpoints/axomamba.pt` loads as the
  promoted 69,377-parameter model with all five morphology IDs. The large
  apparent tree delta is principally retained experiment data (`docs` 4.0 GB,
  `runs` 1.1 GB), while the reusable autoresearch implementation is 1,080
  source lines plus 337 focused test lines.
- DECISION (2026-07-12): Iteration 90 rejects broad-frequency promotion at
  300k. Only voltage improves significantly (-0.00372); response and F1 gains
  vanish, 5 ms is slightly lower, and the diagnosed amplitude defect is not
  repaired. Existing recipe/checkpoint remain promoted.
- REPLICATION (2026-07-12): Iteration 89 weight 0.005 significantly improves
  F1 area by 0.00799 on seed 3 with no core regression. Across three short
  seeds its mean voltage/response/F1-area/5 ms deltas are favorable, earning
  the scheduled 300k iteration-90 gate.
- REPLICATION (2026-07-12): Iteration 88 weight 0.005 replicates significant
  voltage and response-SERA gains without significant exact-F1 loss, but F1
  area/5 ms gains disappear. Half-weight is weaker. Advance only 0.005 to a
  third seed before the 300k gate.
- EXPERIMENT (2026-07-12): Iteration 87 broad 20-250 Hz supervision improves
  voltage, dynamics, response, F1 area, and 5 ms F1 significantly, but exact F1
  regresses by 0.01044. Its frequency-relative error improves only slightly and
  amplitude gain does not. Advance weights 0.005/0.0025 to a fresh-seed gate.
- EXPERIMENT (2026-07-12): Iteration 86 rejected exact-budget parallel
  fast/slow Mamba. Stride 2 improves voltage SERA by 0.01251 but significantly
  loses 0.02909 F1 area and 0.04502 at 5 ms; strides 4-16 also materially lose
  F1 and usually intervention response. Sequential depth is retained.
- EXPERIMENT (2026-07-12): Iteration 85 rejected causal filtered spike-train
  losses at 1, 2, 5, and blended milliseconds. All four significantly regress
  voltage/dynamics SERA, lower exact F1 and F1 area uncertainly, and improve
  5 ms F1 by at most 0.00131. No coefficient tuning is warranted.
- LITERATURE (2026-07-12): Completed the large post-iteration-84 scan in
  `literature/iteration-084-large-scan.md`. It ranks multi-timescale filtered
  spike-train supervision first and an exact-budget parallel fast/slow Mamba
  path second. ODE/operator rollout methods and the six-state BSSM are deferred
  because their key premises are not yet present in the diagnostics. The next
  scan remains scheduled for iteration 100, then every tenth iteration.
- HYPOTHESIS (2026-07-12): Iteration 29 will run a 2x2 cumulative ablation of
  voltage coupling and the promoted training recipe before the 300k checkpoint.
  This attributes gains and exposes interactions under current diagnostics.
- DECISION (2026-07-12): Coupling scale 4.0 remains optimal under the final
  recipe. Lower scales trade F1 for voltage; higher scales significantly
  regress the F1 curve.
- HYPOTHESIS (2026-07-12): Iteration 28 returns to the successful model
  mechanism by remapping voltage-coupling scale 3-6 under the finalized recipe.
  The old 4.0 optimum predates all promoted objective/burn-in changes.
- DECISION (2026-07-12): Retained 6e-4 cosine after 7.5e-4 significantly
  regressed voltage on replication. Local optimizer tuning is closed.
- REPLICATION (2026-07-12): Iteration 27 will replicate cosine rate 7.5e-4 at
  seed 20260712. Seed 1 significantly improves voltage/intervention while F1
  area/5 ms remain uncertainly lower.
- EXPERIMENT (2026-07-12): Iteration 26 provisionally favors 7.5e-4; 9e-4 has
  larger SERA gains but significantly regresses the F1 curve, and 4.5e-4
  regresses both SERA and F1.
- HYPOTHESIS (2026-07-12): Iteration 26 will recheck cosine peak rate after the
  final objective/burn-in recipe changed gradient geometry. Rates 4.5e-4,
  7.5e-4, and 9e-4 remain fully matched otherwise.
- EXPERIMENT (2026-07-12): Iteration 25 retained F1 weight 0.05. Lower weights
  regress F1, 0.06 regresses exact F1, and 0.1 regresses intervention SERA.
- HYPOTHESIS (2026-07-12): Iteration 25 will retune F1 coefficient after the
  stronger replicated SERA weight may alter the old SERA/F1 tradeoff. Nearby
  0.04-0.1 candidates keep the complete promoted recipe otherwise fixed.
- DECISION (2026-07-12): Promoted direct-SERA weight 0.0075 after seed-2
  replication significantly improves voltage, intervention, and 5 ms F1 with
  no core regression.
- REPLICATION (2026-07-12): Iteration 24 will replicate SERA weight 0.0075 at
  seed 20260712. Seed 1 improves voltage by 0.02359 and intervention by 0.00455
  significantly, with F1 area/5 ms only uncertainly lower.
- EXPERIMENT (2026-07-12): Iteration 23 provisionally favors direct-SERA weight
  0.0075. Weight 0.01 has larger SERA gains but significantly regresses the F1
  curve; 0.0025 regresses both SERA axes.
- HYPOTHESIS (2026-07-12): Iteration 23 will retune direct-SERA weight after
  exact-101 integration, double-5 ms supervision, and burn-in 50 changed the
  objective geometry. Nearby weights remain otherwise fully matched.
- DECISION (2026-07-12): Promoted burn-in 50 after replication. It improves
  voltage SERA by 0.00729 and 0.01633 across two seeds with no significant
  core regression; the marginal count-ratio crossing is the one documented
  liberalized-trigger exception, while actual count error improves.
- REPLICATION (2026-07-12): Iteration 22 will replicate burn-in 50 versus 150
  at seed 20260712. Burn-in 50 significantly improves voltage and 5 ms F1;
  its new count-ratio watch is marginal and actual count error improves.
- EXPERIMENT (2026-07-12): Iteration 21 rejects longer burn-in and provisionally
  favors 50. It improves voltage SERA by 0.00729 and 5 ms F1 by 0.00591, with
  F1 area higher and dynamics/intervention neutral.
- HYPOTHESIS (2026-07-12): Iteration 21 will test per-candidate burn-in 0-350
  on identical 500-step windows. Current training ignores 150 reset-state
  steps, while evaluation ignores 500 full-trace steps; this directly probes
  train/eval state-context mismatch with matched forward compute.
- DECISION (2026-07-12): Iteration 20 promotes exact 101-threshold SERA at
  300k. It beats 51 significantly on voltage and intervention SERA, improves
  F1 area/exact F1 uncertainly, and has matched compute.
- MAINTENANCE (2026-07-12): Iteration-20 prune removed thresholded coupling,
  spike-to-soma residuals, refractory inhibition, and soma output scaling;
  historical evidence remains in manifests/diagnostics.
- EXPERIMENT (2026-07-12): Iteration 19 provisionally selects 51 SERA
  thresholds: voltage and intervention SERA improve significantly, F1 area
  rises uncertainly, and measured compute matches 21/101 thresholds.
- HYPOTHESIS (2026-07-12): Iteration 19 will test SERA integration resolution:
  training uses 21 relevance thresholds while evaluation uses 101. Candidates
  11/51/101 retain the newly promoted double-5 ms F1 recipe, with loss compute
  measured explicitly.
- DECISION (2026-07-12): Promoted `[0,1,2,3,4,5,5]` F1 supervision after
  three seeds. Mean changes improve voltage SERA, F1 area, and 5 ms F1 with
  neutral dynamics/intervention; exact F1 is a documented mixed tradeoff.
- REPLICATION (2026-07-12): Iteration 18 will run the final double-5 ms
  objective replication at seed 20260713. Seed 20260712 significantly improves
  F1 area by 0.00615 with no significant core regression.
- EXPERIMENT (2026-07-12): Iteration 17 confirms the double-5 ms lead on a new
  balanced schedule: direct F1 area improves significantly; other core metrics
  are neutral within uncertainty and compute/parameters are matched.
- REPLICATION (2026-07-12): Iteration 17 will replicate the small iteration-16
  double-5 ms lead on seed/schedule 20260712 before promotion. The original
  paired bootstrap does not cover training-seed variance.
- EXPERIMENT (2026-07-12): Iteration 16 provisionally favors duplicating 5 ms
  F1 supervision: voltage and intervention SERA improve significantly, all F1
  points rise uncertainly, compute is matched, and no trigger is added.
- HYPOTHESIS (2026-07-12): Iteration 16 will reshape, rather than increase,
  F1-curve supervision by testing close-timing horizons and duplicate
  tolerance weights. This targets exact/tolerant F1 while preserving the
  SERA-balanced 0.05 coefficient.
- EXPERIMENT (2026-07-12): Iteration 15 retained AdamW 6e-4 cosine. Higher
  rates improve SERA at a significant F1 cost; Adam/lower rate regress; weight
  decay is inert and clipping yields no balanced gain.
- HYPOTHESIS (2026-07-12): Iteration 15 begins the lower-temperature phase by
  rechecking optimizer settings after voltage coupling changed the gradient
  path. Nearby cosine rates, Adam, weight decay, and clipping remain exactly
  matched on parameters, data, and updates.
- EXPERIMENT (2026-07-12): Iteration 14 rejected soma output scaling. Scale 4
  improves voltage SERA but regresses intervention and lowers F1; smaller
  scales largely regress SERA and significantly regress tolerant F1.
- HYPOTHESIS (2026-07-12): Iteration 14 will test fixed soma output scales
  before voltage-to-spike coupling. The promoted model has voltage gain 0.197
  with acceptable bias, a direct amplitude-compression signal; all candidates
  remain parameter matched and end-to-end trained.
- EXPERIMENT (2026-07-12): Iteration 13 rejected refractory inhibition. Scale
  2 improves the refractory/count diagnostics but not F1 and significantly
  worsens intervention SERA; larger scales regress core metrics.
- HYPOTHESIS (2026-07-12): Iteration 13 will test causal one-step refractory
  inhibition of spike logits. The promoted model has a measured 0.221 fraction
  of predicted ISIs below 2 ms, so this directly targets a fast diagnostic
  failure with no added parameters.
- EXPERIMENT (2026-07-12): Iteration 12 rejected spike-to-soma AP residuals.
  A 0.1 residual trades F1 for SERA; larger scales significantly regress F1
  and often voltage, intervention, or dynamics. One-way coupling remains best.
- HYPOTHESIS (2026-07-12): Iteration 12 will test bidirectional but acyclic
  event/voltage coupling: a fixed base spike-probability residual contributes
  to soma before the successful soma-to-spike path. It targets voltage gain and
  4.47x near-spike error without adding parameters.
- EXPERIMENT (2026-07-12): Iteration 11 rejected all rectified voltage
  coupling thresholds. Every candidate significantly regressed voltage and
  intervention SERA, showing that signed hyperpolarized suppression is part of
  the successful linear coupling mechanism.
- HYPOTHESIS (2026-07-12): Iteration 11 will restrict voltage-to-spike
  coupling to depolarized predicted soma using fixed ReLU thresholds 0.0-0.4.
  This preserves parameter matching and targets the persistent depolarized and
  near-spike diagnostic failures.
- EXPERIMENT (2026-07-12): Iteration 10 validates coupling 4.0 at 300k. It
  significantly beats uncoupled on voltage SERA, F1 area, and 5 ms F1, and
  beats coupling 2.0 on voltage, F1 area, and intervention SERA. Every model
  has 69,087 parameters and 9,375 updates.
- MAINTENANCE (2026-07-12): Iteration-10 prune removed the failed E/I tanh,
  positive-dV coupling, direct dynamics-SERA, and exact-F1-only executable
  paths while retaining their manifests and diagnostics as evidence.
- EXPERIMENT (2026-07-12): Iteration 9 retained the 0.1 initialization. Larger
  residual/branch scales significantly regress tolerant F1 or intervention;
  soma-gate 0.3 improves both SERA metrics but significantly regresses F1 area.
- EXPERIMENT (2026-07-12): Iteration 8 rejected positive-dV spike coupling.
  All scales regress voltage/intervention SERA; scales 4-8 regress tolerant F1
  and scale 16 regresses dynamics. The failed path is queued for iteration-10
  pruning.
- HYPOTHESIS (2026-07-12): Iteration 8 will add parameter-neutral positive-dV
  coupling to the spike logit. Positive teacher dV correlates 0.262 with spike
  events across 4,096 frozen windows, stronger than the successful voltage
  level premise, and directly targets event timing plus dynamics.
- EXPERIMENT (2026-07-12): Iteration 7 promoted voltage-to-spike coupling 4.0.
  Versus 2.0 it improves voltage SERA by 0.01132 mV and intervention SERA by
  0.00763 mV while preserving F1 area/exact F1; scales 6 and 8 cross the
  measured F1/voltage regression boundary.
- EXPERIMENT (2026-07-12): Iteration 6 promoted parameter-neutral
  voltage-to-spike coupling 2.0. It significantly improves voltage SERA by
  0.00348 mV, intervention SERA by 0.00114 mV, F1 area by 0.00819, and 5 ms F1
  by 0.01425, with dynamics/exact F1 uncertain and no new attention trigger.
- HYPOTHESIS (2026-07-12): Iteration 6 will test an explicit zero-parameter
  spike-logit residual from predicted soma. Across 4,096 frozen windows, mean
  normalized soma is 1.209 at spike steps versus 0.020 elsewhere; this directly
  targets the paper's weakly coupled voltage/event-head failure.
- EXPERIMENT (2026-07-12): Iteration 5 retained SERA 0.005 + F1-curve 0.05
  under the cosine schedule. Weakening SERA significantly regressed all three
  SERA axes; heavier F1 weights improved tolerant F1 but significantly
  regressed voltage, dynamics, and intervention SERA.
- EXPERIMENT (2026-07-12): Iteration 4 promoted 6e-4 cosine at batch 32. It
  improves voltage SERA by 0.35505 mV and intervention SERA by 0.00700 mV,
  leaves F1/dynamics uncertain, and clears voltage-bias plus default-threshold
  attention triggers. Constant higher rates were mixed or regressed.
- EXPERIMENT (2026-07-12): Iteration 3 rejected parameter-matched E/I tanh.
  Despite supported cancellation, it regresses voltage SERA by 0.00356 mV and
  intervention SERA by 0.00286 mV; F1 gains are uncertain and compute is 21%
  higher. Current best remains iteration 1.
- EXPERIMENT (2026-07-12): Iteration 2 did not promote a candidate. Direct
  dynamics-SERA and exact-F1 terms failed their intended metrics and cost
  22-26 s model compute versus 22.24 s control. F1-curve 0.1 improved voltage
  SERA by 0.01932 mV but left other core metrics uncertain/slightly lower.
- HYPOTHESIS (2026-07-12): On 64 frozen windows, 25.80% of active branch-time
  bins contain both E and I, 15.22% fully cancel, and signed summation removes
  33.83% of event magnitude. Iteration 3 will test parameter-matched E/I tanh.
- EXPERIMENT (2026-07-12): Iteration 1 promoted direct-SERA 0.005 + F1-curve
  0.05. Versus stripped control it improves voltage SERA by 0.05235 mV,
  intervention SERA by 0.00963 mV, F1 area by 0.02076, and 5 ms F1 by 0.04156;
  dynamics and exact F1 are uncertain, with no new attention trigger.
- VERIFICATION (2026-07-12): The stripped architecture independently beats the
  old full stack on voltage/dynamics/intervention SERA, F1 area, and 5 ms F1;
  exact F1 is uncertain. Artifact:
  `artifacts/diagnostics/iteration-001-metric-objectives/promoted-control-vs-old-full.json`.
- EXPERIMENT (2026-07-11): Completed the matched 120k batch panel. Batch 32
  uses 3,750 updates and 12.40 s model compute versus batch 16's 7,500 updates
  and 26.42 s. It regresses voltage SERA by 0.180 mV and exact F1 by 0.0114,
  but was selected over batch 64 as the requested latency compromise.
- EXPERIMENT (2026-07-11): Completed four one-addon removals at batch 32.
  Branch gain was inert. Removing the integrative mixer improved voltage,
  dynamics, and intervention SERA by 0.01696 mV, 0.00101 mV/ms, and 0.00643 mV.
  Branch tanh was retained; the soma gate was retained as a SERA/F1 tradeoff.
- IMPLEMENTATION (2026-07-11): Promoted AxoMamba defaults now disable branch
  gain and the integrative mixer. The experiment engine uses a frozen exactly
  morphology-balanced schedule, shared loading, int8 host transfer, per-model
  sub-batches, CUDA warmup, and per-candidate compute timing.
- VERIFICATION (2026-07-11): End-to-end 128-presentation smoke wrote training
  metrics, checkpoints, a baseline cache, candidate comparison, and summaries.
  Full AxoSim suite passes with 77 tests. AxoBench's focused evaluator suite
  passes with 23 tests; its unrelated full plotting suite lacks matplotlib in
  the AxoSim venv.
- USER (2026-07-11): Requested a sustained AxoMamba autoresearch loop using
  120k presentations by default, 300k every tenth iteration, diagnostics on
  every run, cached baseline outputs, shared/frozen loading, parallel candidate
  runs where latency benefits, pre-loop addon ablation, and periodic pruning.
- VERIFICATION (2026-07-11): Existing paired AxoBench evaluator tests pass in
  the AxoSim GPU environment: AxoBench `21 passed`; AxoSim adapter `4 passed`.
  The evaluator currently reruns baseline inference and therefore still needs
  cached-baseline support.
- DECISION (2026-07-11): Core optimization targets are voltage/dynamics SERA,
  F1-over-time, and intervention response SERA. High-temperature objective and
  architecture changes precede low-temperature tuning.
- INITIATION (2026-07-11): Created dedicated branches
  `codex/axomamba-autoresearch` in AxoSim and AxoBench and established this task
  arc. Existing uncommitted diagnostics work is preserved as user-owned input.
- EXPERIMENT (2026-07-12): Iteration 29's matched 2x2 cumulative ablation shows
  that coupling and the promoted recipe independently help. Removing coupling
  under the old recipe significantly worsens voltage SERA by 0.01480 mV,
  F1 area by 0.00868, 5 ms F1 by 0.00960, and intervention SERA by 0.00877 mV.
  The final coupled recipe is best on every primary metric.
- DECISION (2026-07-12): Iteration 30 will use the automatic 300k budget to
  validate the final model against the two informative single-factor
  ablations, followed by the required cleanup audit.
- EXPERIMENT (2026-07-12): Iteration 30's 300k validation confirms the promoted
  recipe: versus the old recipe it improves voltage SERA by 0.03321 mV and
  intervention SERA by 0.00512 mV significantly, with F1 area/5 ms tied. Fixed
  coupling retains a significant 0.00651 mV voltage-SERA gain over uncoupled.
- CLEANUP (2026-07-12): The iteration-30 audit found no surviving failed
  mechanisms. Direct `train_candidate_panel` calls now inherit the promoted
  burn-in/objective, and a duplicate coupling condition was removed.
- DECISION (2026-07-12): Iteration 31 starts from the dominant remaining
  intervention-gain and temporal-residual diagnostics, with a training-split
  sensitivity audit before any architecture run.
- EXPERIMENT (2026-07-12): Iteration 31's disjoint paired training improves
  voltage SERA by 0.04065 mV but response SERA by only 0.00251 mV, while
  significantly regressing F1 area by 0.01146 and 5 ms F1 by 0.02244. Direct
  response-SERA weights through 0.005 add no response benefit.
- VALIDITY (2026-07-12): Iteration 31's response-loss models spend about 29.2 s
  model compute versus 22.9 s for the control because the zero-weight path
  skips response computation. Iteration 32 will redo the objective comparison
  with response computation enabled for every candidate.
- EXPERIMENT (2026-07-12): Iteration 32's compute-matched panel confirms direct
  response-SERA does not improve held-out response. Weight 0.02 regresses
  response SERA by 0.00124 mV and F1 area by 0.00744; lower weights are neutral
  or also worsen response. A 60k-window audit rules out weak-response sampling.
- DECISION (2026-07-12): Close paired intervention training and return to the
  promoted base-only recipe. Iteration 33 tests whether the 4,500-slot
  overlapping input route itself suppresses channel-specific perturbations.
- EXPERIMENT (2026-07-12): Iteration 33 finds route 60 provisionally useful:
  versus route 100 it significantly improves dynamics SERA by 0.00276 mV/ms
  and intervention SERA by 0.00202 mV, preserves F1 area/exact point estimates,
  and reduces measured model compute by 8.9%. More aggressive overlap removal
  causes significant F1 or response regressions.
- DECISION (2026-07-12): Replicate route 60 versus route 100 at seed 20260712
  in iteration 34 before promotion.
- EXPERIMENT (2026-07-12): Iteration 34 rejects route 60 on seed 2: voltage
  SERA regresses 0.02684 mV, intervention SERA 0.00261 mV, and 5 ms F1 0.02029
  significantly. Retain route 100.
- DECISION (2026-07-12): Iteration 35 tests four exact-69,087-parameter budget
  reallocations that replace the 45-branch bottleneck with 60-143 finer
  branches while paying for it in Mamba width/state capacity.
- EXPERIMENT (2026-07-12): Iteration 35 finds exact-budget 122/66/4 promising:
  voltage SERA improves significantly by 0.02639 mV, dynamics/response and F1
  area/exact move favorably, 5 ms F1 is flat, and model compute falls 9.9%.
  Other exact-budget allocations expose F1/SERA or dynamics tradeoffs.
- DECISION (2026-07-12): Replicate 122/66/4 against 45/64/16 at seed 20260712
  in iteration 36 before promotion.
- EXPERIMENT (2026-07-12): Iteration 36 rejects 122/66/4 despite replicated
  voltage/dynamics and efficiency gains because seed-2 intervention SERA
  significantly regresses by 0.00724 mV. Retain 45/64/16.
- DECISION (2026-07-12): Iteration 37 completes the finite exact-parameter
  allocation set with 101/56/32 and 118/58/25; no unconstrained shape search.
- EXPERIMENT (2026-07-12): Iteration 37 rejects 101/56/32 on significant
  voltage/intervention SERA regressions. 118/58/25 improves voltage
  significantly and intervention uncertainly, but lowers all F1 estimates;
  replicate once because no regression is significant and compute falls 9.5%.
- DECISION (2026-07-12): Iteration 38 is the final routing-allocation
  replication. Close the family if 118/58/25 does not preserve balance.
- EXPERIMENT (2026-07-12): Iteration 38 finds 118/58/25 mixed on seed 2:
  intervention SERA improves significantly by 0.00328 mV and compute falls
  8.7%, but voltage SERA regresses significantly by 0.00689 mV. Two-seed means
  tie voltage/5 ms, improve response/F1 area, and lower exact F1.
- DECISION (2026-07-12): Iteration 39 is a predeclared third-seed tie-break for
  118/58/25 before the required iteration-40 300k run.
- EXPERIMENT (2026-07-12): Iteration 39 passes 118/58/25's third-seed gate:
  seed 3 significantly improves voltage by 0.03030 mV, response by 0.00494 mV,
  and F1 area by 0.00843. Three-seed means improve voltage, response, F1 area,
  5 ms F1, and 8.9% compute, with dynamics tied and exact F1 0.00163 lower.
- DECISION (2026-07-12): Iteration 40 is the required 300k 45/64/16 versus
  118/58/25 gate, followed by pruning of failed paired-response infrastructure.
- EXPERIMENT (2026-07-12): Iteration 40's 300k gate promotes exact-budget
  118/58/25. It significantly improves 5 ms F1 by 0.01561, improves F1 area by
  0.00824 and voltage SERA by 0.00590 mV uncertainly, has no significant core
  regression, and cuts model compute 8.3%.
- CLEANUP (2026-07-12): Removed failed paired-response objectives/loaders/views,
  retained historical artifacts, and renamed the durable default config to
  `configs/axomamba.json`. AxoMamba code/CLI/runner now use 118/58/25.
- DECISION (2026-07-12): Iteration 41 rechecks fixed coupling scale under the
  newly promoted allocation before pursuing a new mechanism.
- EXPERIMENT (2026-07-12): Iteration 41 provisionally selects coupling 2 under
  118/58/25: voltage SERA improves 0.01022 mV and response SERA 0.00350 mV
  significantly; all F1 deltas are modestly negative but uncertain. Coupling
  3 has a larger F1 tradeoff and 5-6 regress core metrics.
- DECISION (2026-07-12): Replicate coupling 2 versus 4 at seed 20260712 in
  iteration 42 before changing the durable default.
- EXPERIMENT (2026-07-12): Iteration 42 rejects coupling 2 on seed 2 because
  intervention SERA regresses significantly by 0.00376 mV and all F1 estimates
  decline. Retain coupling 4.
- DECISION (2026-07-12): Iteration 43 retunes direct-SERA and F1-curve weights
  under promoted 118/58/25 with identical loss compute for every candidate.
- EXPERIMENT (2026-07-12): Iteration 43 provisionally selects direct-SERA 0.01:
  voltage improves 0.02296 mV and response 0.00352 mV significantly, with
  neutral dynamics and uncertain 0.0028-0.0039 F1 costs. SERA 0.015 causes
  significant F1 regression; F1-weight changes and a joint point do not help.
- DECISION (2026-07-12): Replicate direct-SERA 0.01 versus 0.0075 at seed
  20260712 in iteration 44.
- EXPERIMENT (2026-07-12): Iteration 44 replicates direct-SERA 0.01. Voltage
  SERA improves 0.02836 mV and intervention SERA 0.00341 mV significantly;
  dynamics improves uncertainly and all F1 deltas are small and uncertain.
  There is no new attention trigger.
- DECISION (2026-07-12): Promote direct-SERA weight 0.01. Two-seed mean gains
  are 0.02566 mV voltage and 0.00346 mV intervention SERA, with no significant
  core regression on either seed.
- HYPOTHESIS (2026-07-12): Iteration 45 rechecks the bounded cosine peak-rate
  panel because both the promoted latent allocation and SERA gradient scale
  differ from the recipe under which 6e-4 was last selected.
- EXPERIMENT (2026-07-12): Iteration 45 provisionally selects cosine peak rate
  9e-4. It significantly improves voltage SERA by 0.01956 mV and intervention
  SERA by 0.00203 mV; dynamics, exact F1, and 5 ms F1 improve uncertainly,
  F1 area is flat, and no trigger changes. Rate 4.5e-4 significantly regresses
  F1 area and 5 ms F1.
- DECISION (2026-07-12): Iteration 46 replicates 9e-4 versus 6e-4 at seed
  20260712 before changing the durable learning rate.
- EXPERIMENT (2026-07-12): Iteration 46 rejects 9e-4. Voltage improves
  significantly by 0.00490 mV and F1 area rises 0.00249, but intervention SERA
  reverses to a significant 0.00506 mV regression. Retain 6e-4.
- DECISION (2026-07-12): Iteration 47 gives the remaining balanced 7.5e-4 lead
  one seed-2 replication; close cosine-rate tuning afterward.
- EXPERIMENT (2026-07-12): Iteration 47 rejects 7.5e-4. Voltage SERA improves
  significantly by 0.00719 mV, but intervention SERA significantly regresses
  by 0.00205 mV; F1 and dynamics are tied. Retain 6e-4 and close rate tuning.
- IMPLEMENTATION (2026-07-12): Added a tested, parameter-neutral causal
  readout lead transform. The alpha-zero control executes the same operation;
  AxoMamba model tests pass 8/8.
- HYPOTHESIS (2026-07-12): Iteration 48 tests whether latent lead compensation
  repairs measured residual lag, high-dVdt error, and near-zero intervention
  gain while preserving SERA/F1.
- EXPERIMENT (2026-07-12): Iteration 48 provisionally selects causal readout
  lead alpha 0.1. It significantly improves voltage and intervention SERA plus
  F1 area, exact F1, and 5 ms F1; dynamics improves uncertainly and no trigger
  changes. Premise diagnostics move favorably but only slightly. Larger scales
  incur significant F1 or response regressions.
- DECISION (2026-07-12): Iteration 49 replicates lead alpha 0.1 at seed
  20260712. A pass advances it to iteration 50's 300k factorial gate.
- EXPERIMENT (2026-07-12): Iteration 49 passes lead alpha 0.1. Seed 2
  significantly improves voltage SERA and improves dynamics plus all F1 point
  estimates, with response tied. Two-seed means favor the lead on all six core
  metrics.
- DECISION (2026-07-12): Iteration 50 runs the required 300k 2x2 factorial of
  SERA weight 0.0075/0.01 and lead alpha 0/0.1, followed by scheduled cleanup
  and full tests.
- EXPERIMENT (2026-07-12): Iteration 50's 300k factorial validates SERA weight
  0.01: reverting to 0.0075 significantly regresses voltage by 0.02239 mV and
  response by 0.00358 mV. Lead 0.1 significantly improves response but
  significantly regresses voltage by 0.00258 mV and lowers all F1 estimates.
- CLEANUP (2026-07-12): Retained SERA 0.01 and removed the failed readout-lead
  config/forward/test code. Historical manifests and diagnostics remain.
- VERIFICATION (2026-07-12): Post-iteration-50 cleanup passes AxoSim 81/81 and
  AxoBench evaluator 23/23. AxoMamba remains 69,087 parameters at 118/58/25;
  no readout-lead production reference remains.
- IMPLEMENTATION (2026-07-12): Added a tested parameter-neutral Mamba `A_log`
  initialization shift. The test verifies only SSM decay tensors change.
- HYPOTHESIS (2026-07-12): Iteration 51 symmetrically tests half-to-double
  initial Mamba decay rates to target residual lag, fast dynamics, and weak
  intervention memory without changing compute or capacity.
- FIDELITY (2026-07-12): Discarded iteration 51's first execution before
  interpretation because common-state matching erased constructor-time
  `A_log` shifts. Added tested post-match named-parameter offsets and rerunning
  the same predeclared panel.
- EXPERIMENT (2026-07-12): Valid iteration 51 provisionally selects half-rate
  Mamba decay initialization. Voltage SERA improves 0.01939 mV and response
  0.00620 mV significantly; exact F1 improves, while F1 area/5 ms decline
  modestly and uncertainly. Faster spectra trade response/tolerant F1 for
  exact F1. The shifted `A_log` separation persists after training.
- DECISION (2026-07-12): Iteration 52 replicates half-rate decay at seed
  20260712; require replicated SERA gains without significant F1 regression.
- EXPERIMENT (2026-07-12): Iteration 52 rejects uniform half-rate decay.
  Voltage improves significantly by 0.00389 mV, but response regresses
  significantly by 0.00151 mV and 5 ms F1 by 0.00844. Retain standard decay.
- HYPOTHESIS (2026-07-12): Iteration 53 tests layer-separated slow/fast Mamba
  spectra, using one-layer-only controls and both orderings to combine the
  SERA benefit of slow states with the timing benefit of fast states.
- EXPERIMENT (2026-07-12): Iteration 53 rejects layer-separated timescales.
  Every altered spectrum significantly regresses 5 ms F1; both mixed orderings
  also significantly regress F1 area. Retain standard Mamba initialization.
- IMPLEMENTATION (2026-07-12): Added a tested soft-OR tolerant-event F1
  surrogate and nullable blend. Non-null controls compute both legacy and
  soft-OR graphs, preserving matched loss compute.
- HYPOTHESIS (2026-07-12): Iteration 54 tests whether dense soft-OR recall
  gradients improve measured F1 across time while preserving SERA/response.
- EXPERIMENT (2026-07-12): Iteration 54 rejects soft-OR F1. Increasing its
  blend improves exact F1 but monotonically collapses F1 area/5 ms; full
  soft-OR raises refractory violations from 0.173 to 0.684 and worsens count
  calibration. Matched loss compute costs about 32% over the legacy objective.
- IMPLEMENTATION (2026-07-12): Added a tested nullable close-pair exclusion
  loss. Its scale is measured at mean 2.72 per true event on eight frozen
  batches, guiding bounded weights 0.00125-0.005.
- HYPOTHESIS (2026-07-12): Iteration 55 tests whether explicit refractory
  exclusion repairs soft-OR's diagnosed clustering failure and recovers actual
  F1 across time.
- EXPERIMENT (2026-07-12): Iteration 55 closes soft-OR/exclusion. Exclusion
  0.00125 significantly repairs all F1 metrics but significantly regresses
  voltage and response; the only balanced point collapses default-threshold
  count to 0.22x. Queue optional objective code for iteration-60 pruning.
- IMPLEMENTATION (2026-07-12): Added a tested nullable recurrent second-pass
  blend. Mix zero and one exactly match one and two shared-weight passes, all
  candidates execute two passes, and parameter count is unchanged.
- HYPOTHESIS (2026-07-12): Iteration 56 tests shared-weight iterative Mamba
  refinement at matched two-pass compute; promotion requires gains large
  enough to justify nearly doubled model compute.
- EXPERIMENT (2026-07-12): Iteration 56 provisionally selects recurrent mix
  0.25 as balanced and 0.5 as SERA-focused. Mix 0.25 improves five of six core
  point estimates and significantly improves voltage/response; mix 0.5
  significantly improves all SERA axes. Matched two-pass compute costs 24%
  over a recent one-pass control.
- DECISION (2026-07-12): Iteration 57 replicates recurrent mixes 0.25 and 0.5
  at seed 20260712 before accepting any added inference compute.
- EXPERIMENT (2026-07-12): Iteration 57 rejects recurrent mix 0.25 for
  insufficient replicated value and retains 0.5 provisionally. Mix 0.5 again
  significantly improves voltage/response with no significant F1 regression;
  two-seed means improve all SERA axes but lower tolerant-F1 point estimates.
- DECISION (2026-07-12): Iteration 58 maps recurrent mixes 0.35/0.4/0.45 to
  improve the SERA/F1 balance before iteration 60's long gate.
- EXPERIMENT (2026-07-12): Iteration 58 finds no recurrent point better than
  mix 0.5. Mixes 0.35-0.45 significantly improve all SERA axes but lower F1
  area; 0.45 gives up SERA without improving the area tradeoff over 0.5.
- DECISION (2026-07-12): Iteration 59 runs a third seed for recurrent mix 0.5
  before the required 300k gate.
- EXPERIMENT (2026-07-12): Iteration 59 rejects recurrent mix 0.5. Seed 3
  significantly regresses voltage by 0.05131 mV and response by 0.00821 mV;
  the prior SERA benefit is not robust enough for 24% extra compute.
- DECISION (2026-07-12): Iteration 60 uses the required 300k budget to
  revalidate current 118/58/25 against exact-budget historical 45/64/16 on
  seed 2 under SERA 0.01, then prunes failed optional paths and runs full tests.
- EXPERIMENT (2026-07-12): Iteration 60 strongly revalidates 118/58/25 at
  300k seed 2. Historical 45/64/16 is worse on all six core metrics, including
  significant +0.03123 mV voltage and +0.00548 mV response regressions, with
  matched compute.
- CLEANUP (2026-07-12): Removed failed recurrent-pass and soft-OR/exclusion
  production/test paths; retained generic post-match initialization offsets.
- VERIFICATION (2026-07-12): Post-iteration-60 cleanup passes AxoSim 82/82 and
  AxoBench evaluator 23/23.
- IMPLEMENTATION (2026-07-12): Added a tested nullable final-normalization
  blend. Mix 1 exactly matches the promoted normalized readout while all
  candidates retain identical LayerNorm parameters/compute.
- HYPOTHESIS (2026-07-12): Iteration 61 tests whether partial final
  normalization restores absolute voltage/intervention amplitude without
  destabilizing F1 or physiological bounds.
- EXPERIMENT (2026-07-12): Iteration 61 rejects global final-norm blending.
  Fractions 0.5-0.75 improve voltage/response but lower tolerant F1, and
  voltage gain does not improve; removing normalization regresses response and
  F1 significantly.
- IMPLEMENTATION (2026-07-12): Added a tested soma-only final-normalization
  blend through the separate soma readout/gate, leaving spike features fully
  normalized and preserving parameters/compute.
- HYPOTHESIS (2026-07-12): Iteration 62 tests whether the soma-only blend keeps
  the global blend's SERA benefit without perturbing spike F1.
- EXPERIMENT (2026-07-12): Iteration 62 provisionally selects soma-only final
  normalization fraction 0.75. It significantly improves voltage SERA by
  0.02148 mV and response SERA by 0.00522 mV with smaller, uncertain F1 losses
  than the global blend, but worsens count ratio and refractory violations.
- DECISION (2026-07-12): Iteration 63 replicates soma fraction 0.75 on seed 2
  and interpolates 0.80-0.90 for a better SERA/F1/spike-health balance.
- EXPERIMENT (2026-07-12): Iteration 63 rejects soma normalization. Every
  fraction 0.75-0.90 significantly regresses voltage and response SERA on
  seed 2, reversing seed 1 monotonically.
- FIDELITY (2026-07-12): The soma-only blend still leaked into spike logits
  through the promoted four-times-soma coupling. Iteration 64 will isolate
  reported voltage from a fully normalized spike-coupling reference under
  matched double-soma-path compute.
- IMPLEMENTATION (2026-07-12): Added tested nullable `output`/`normalized`
  soma sources for spike coupling. Both experimental modes compute the same
  two shared-weight soma paths; the isolated mode exactly preserves normalized
  spike output and mixed voltage output at initialization without parameters.
- EXPERIMENT (2026-07-12): Iteration 64 rejects isolated soma coupling. At
  fraction 0.75, the ordinary coupled candidate dominates isolation on every
  core point estimate except exact F1; raw isolated features retain the
  tolerant-F1 tradeoff despite significant SERA gains.
- HYPOTHESIS (2026-07-12): Iteration 65 targets the 4.47x near-spike voltage
  error with a matched, zero-initialized causal spike-conditioned soma
  waveform head, testing detached and joint-gradient variants.
- IMPLEMENTATION (2026-07-12): Added a tested zero-initialized causal soma
  waveform convolution over predicted spike probability. It changes only soma,
  is causal, and is included in disabled controls for exact parameter/compute
  matching.
- EXPERIMENT (2026-07-12): Iteration 65 provisionally selects detached
  waveform scale 4: voltage/response SERA and exact F1 improve significantly,
  but AP-specific diagnostics barely move and refractory violations worsen.
- DECISION (2026-07-12): Iteration 66 replicates detached scale 4 on seed 2
  and maps scales 2/8/16 under exact matched head parameters and compute.
- EXPERIMENT (2026-07-12): Iteration 66 provisionally selects detached
  waveform scale 2. On seed 2 it significantly improves all three SERA axes
  and improves all F1 point estimates with no new trigger.
- DECISION (2026-07-12): Iteration 67 runs a third seed for detached waveform
  scale 2 before any kernel refinement or long-run promotion.
- EXPERIMENT (2026-07-12): Iteration 67 rejects the waveform head. Detached
  scale 2 significantly regresses voltage SERA by 0.01546 mV on seed 3 and
  moves five of six core point estimates adversely.
- HYPOTHESIS (2026-07-12): Iteration 68 tests explicit five-morphology
  conditioning. Current training is balanced across morphologies, but the
  model/predictor API never receives morphology identity, leaving the target
  conditionally ambiguous.
- IMPLEMENTATION (2026-07-12): Added optional learned morphology embeddings
  and propagated per-sample IDs through shared training, checkpoint inference,
  cached validation, and intervention evaluation. Focused AxoSim tests pass
  29/29 and AxoBench iteration-evaluator tests pass 24/24.
- EXPERIMENT (2026-07-12): Iteration 68 finds the strongest structural result
  so far. Morphology scales 0.1 and 0.25 significantly improve all six core
  metrics; scale 0.1 is the balanced lead and reduces morphology disparity,
  though refractory violations worsen.
- DECISION (2026-07-12): Iteration 69 replicates scales 0.1/0.25 on seed 2 and
  interpolates 0.05/0.15 before the scheduled iteration-70 300k gate.
- EXPERIMENT (2026-07-12): Iteration 69 robustly replicates morphology
  conditioning. All scales 0.05-0.25 significantly improve voltage, response,
  and all F1 metrics; scale 0.1 now has two-seed support.
- DECISION (2026-07-12): Iteration 70 compares scales 0.05/0.1/0.15 with the
  inactive control at 300k, then performs scheduled failed-path cleanup and
  full two-repo verification.
- EXPERIMENT (2026-07-12): Iteration 70 validates morphology conditioning at
  300k: scales 0.05/0.1/0.15 all significantly improve all six core metrics.
  Scale 0.15 is promoted as the strongest long core/health point.
- CLEANUP (2026-07-12): Promoted scale 0.15 into reusable config/default and
  `checkpoints/axomamba.pt`; removed rejected normalization, coupling-isolation,
  and waveform paths. AxoMamba now has 69,377 parameters.
- VERIFICATION (2026-07-12): Post-iteration-70 cleanup passes AxoSim 85/85 and
  AxoBench evaluator 24/24; no rejected symbols remain and the stable
  checkpoint matches the promoted long run byte-for-byte.
- HYPOTHESIS (2026-07-12): Iteration 71 tests zero-initialized final FiLM by
  morphology on top of the promoted pre-trunk embedding, under matched added
  parameters and compute.
- IMPLEMENTATION (2026-07-12): Added a tested zero-initialized final
  morphology FiLM table. The inactive/active models start identically, active
  context changes outputs, and all iteration-71 candidates carry the same 580
  added parameters and compute.
- EXPERIMENT (2026-07-12): Iteration 71 rejects final morphology FiLM. Tiny
  SERA gains at scales 0.05/0.1 come with significant tolerant-F1 regressions;
  scales 0.25/0.5 significantly regress voltage.
- HYPOTHESIS (2026-07-12): Iteration 72 tests zero-initialized
  morphology-specific gains on routed branch currents before saturation and
  Mamba, targeting the connectome computation directly.
- IMPLEMENTATION (2026-07-12): Added a tested zero-initialized
  morphology-by-branch gain table before branch tanh/Mamba. Inactive and active
  models start identically; all iteration-72 candidates carry the same 590
  added parameters and compute.
- EXPERIMENT (2026-07-12): Iteration 72 provisionally selects morphology
  branch-gain scale 1. It significantly improves five core metrics and raises
  5 ms F1 uncertainly; intervention/refractory diagnostics improve while
  morphology disparity stays flat.
- DECISION (2026-07-12): Iteration 73 replicates branch-gain scale 1 on seed 2
  and extends the monotonic curve to scales 2/4 with a 0.75 interpolation.
- EXPERIMENT (2026-07-12): Iteration 73 partially replicates morphology branch
  gains. Scale 2 is balanced; scale 4 improves F1 area/exact significantly but
  slightly regresses response; scale 0.75 significantly regresses 5 ms F1.
- DECISION (2026-07-12): Iteration 74 runs scales 1/2/4 on a third seed before
  any promotion.
- EXPERIMENT (2026-07-12): Iteration 74 rejects morphology branch gains.
  Scales 1/2 significantly regress response on seed 3; scale 4 becomes a
  dynamics/F1 tradeoff and loses voltage.
- HYPOTHESIS (2026-07-12): Iteration 75 tests parameter-neutral reinjection of
  the promoted morphology embedding before each Mamba block.
- IMPLEMENTATION (2026-07-12): Added tested parameter-neutral morphology
  reinjection before every Mamba block. Scale zero exactly matches promoted
  AxoMamba and active reinjection changes context-conditioned outputs.
- EXPERIMENT (2026-07-12): Iteration 75 rejects morphology reinjection. Every
  scale regresses response or another core metric; larger scales produce
  significant voltage/F1 failures.
- HYPOTHESIS (2026-07-12): Iteration 76 retunes direct-SERA/F1 weights on the
  promoted conditioned architecture, revisiting pre-conditioning optima.
- EXPERIMENT (2026-07-12): Iteration 76 selects direct SERA weight 0.02 for
  replication. It significantly improves voltage SERA by 0.03901 mV and
  response SERA by 0.00726 mV while all F1 point estimates improve; refractory
  violations rise from 0.243 to 0.276. Higher F1 weights create SERA tradeoffs.
- DECISION (2026-07-12): Iteration 77 replicates the direct-SERA result on a
  fresh seed and brackets weights 0.0125-0.03 before objective promotion.
- EXPERIMENT (2026-07-12): Iteration 77 rejects stronger direct-SERA weighting
  alone. Weight 0.02 replicates voltage/response gains but significantly loses
  F1 area and 5 ms F1; all stronger SERA candidates share the tolerant-F1
  failure. Refractory rate is flat, while spike count calibration worsens.
- DECISION (2026-07-12): Iteration 78 tests a targeted joint SERA/F1
  compensation frontier, exploiting the opposing measured objective effects.
- EXPERIMENT (2026-07-12): Iteration 78 finds that SERA 0.025/F1 0.10 recovers
  the tolerant-F1 loss and significantly improves voltage, but response and
  dynamics worsen uncertainly. Refractory/count-error diagnostics improve,
  while default-threshold count ratio worsens from 1.449 to 2.303.
- DECISION (2026-07-12): Iteration 79 fresh-seed tests the joint lead and adds
  stronger spike-BCE pressure to directly correct its count overprediction.
- EXPERIMENT (2026-07-12): Iteration 79 rejects stronger BCE calibration:
  every BCE-0.35/0.50 candidate significantly harms response and/or dynamics.
  The original SERA 0.025/F1 0.10 joint point again improves voltage with no
  significant core regression and advances to the long gate.
- DECISION (2026-07-12): Iteration 80 runs a fresh-seed 300k control-versus-
  joint-objective promotion gate, then performs scheduled failed-path cleanup
  and full AxoSim/AxoBench verification.
- EXPERIMENT (2026-07-12): Iteration 80 rejects SERA 0.025/F1 0.10 at 300k.
  Voltage improves significantly, but response SERA significantly regresses
  and all F1 point estimates slip; fixed-threshold count ratio reaches 3.062.
- CLEANUP (2026-07-12): Removed failed morphology FiLM, branch-gain, and
  reinjection paths. Added explicit backward compatibility for retired config
  keys after the stable checkpoint exposed strict-load failure.
- VERIFICATION (2026-07-12): Cleaned AxoSim passes 86/86, AxoBench evaluator
  passes 24/24, and the stable 69,377-parameter five-morphology checkpoint
  loads successfully.
- HYPOTHESIS (2026-07-12): Iteration 81 targets intervention response using
  the dataset's 12,800 paired teacher examples, which current training ignores.
- IMPLEMENTATION (2026-07-12): Added leakage-safe frozen intervention mixing:
  all 25 evaluator shards are held out whole, alternate windows are morphology
  aligned, loading is shared, and exact per-candidate exposure is recorded.
- EXPERIMENT (2026-07-12): Iteration 81 finds that 25% intervention exposure
  significantly improves voltage and response SERA but significantly regresses
  F1 area; all other fractions fail to improve the full core frontier.
- DECISION (2026-07-12): Iteration 82 uses matched baseline/intervention pairs
  and a direct response-SERA objective within the same 120k forward budget.
- IMPLEMENTATION (2026-07-12): Added pair packing and differentiable
  response-SERA matching AxoBench's baseline/intervention/delta relevance,
  with exact pair and presentation accounting; AxoSim passes 90/90.
- EXPERIMENT (2026-07-12): Iteration 82 rejects direct response-SERA. Positive
  weights reduce training-pair loss but weaken or reverse held-out response;
  weights 0.01/0.02 significantly regress response and voltage. Paired data
  alone improves voltage with no significant core regression.
- DECISION (2026-07-12): Iteration 83 fresh-seed sweeps paired-data allocation
  without response loss to test robustness and locate a balanced fraction.
- EXPERIMENT (2026-07-12): Iteration 83 finds the first clean intervention
  success: paired fractions 0.125/0.25 significantly improve voltage, response,
  and 5 ms F1 together, with F1 area also higher and no significant core loss.
- DECISION (2026-07-12): Iteration 84 runs a third-seed 0.0625-0.3125 bracket
  and promotes only the smallest robust balanced paired-data fraction.
- EXPERIMENT (2026-07-12): Iteration 84 rejects paired-data promotion. On seed
  3, fractions 0.125-0.3125 significantly regress voltage and response, with
  higher fractions also losing tolerant F1; only 0.0625 improves response but
  has adverse uncertain voltage/F1 movement.
- DECISION (2026-07-12): Retain ordinary training. Run the requested large
  adjacent-literature scan now, next at iteration 100, then every tenth
  iteration; use it to select iteration 85 rather than continuing pair tuning.
- MILESTONE (2026-07-12): Iteration 100 promoted near-spike voltage weighting
  after its 300k gate; the practically small voltage movement was not counted
  as a gain, while response and tolerant F1 improved significantly.
- STRUCTURAL PHASE (2026-07-12): Iterations 101-108 rejected branch-local
  Mamba, additive branch memory, explicit delays, Mamba-2, and Mamba-GRU
  hybrids under matched controls. These failures close mechanism families and
  are not invitations for scalar tuning.
- PROMOTION (2026-07-12): Iteration 110 promoted a rank-4 low-rank interaction
  across the 118 routed branches after a fresh-seed 300k gate. Voltage SERA
  improves by 0.04746 mV and response SERA by 0.00591 mV, with all event point
  estimates favorable and no significant core regression.
- CLEANUP (2026-07-12): Removed rejected I101-I109 executable paths and tests,
  the completed delay-audit script, and isolated Mamba-3 feasibility
  environment. The deployed checkpoint contains 69,161 functional parameters;
  metadata records the exact 69,377-parameter training comparison.
- VERIFICATION (2026-07-12): Cleaned AxoSim passes 89/89, AxoBench evaluator
  passes 24/24, `git diff --check` passes, and the promoted five-morphology
  state-23/rank-4 checkpoint loads with its 300k seed metadata.
- LITERATURE (2026-07-12): The iteration-110 scan prioritizes temporal
  low-rank Volterra interaction, multiplicative dendritic modulation, and
  intervention-aware latent transitions. Iteration 111 begins with the first,
  using exact parameter and matched-compute controls rather than a scalar grid.
- STRATEGY (2026-07-12): The desired endpoint is no longer framed as another
  few thousandths of a millivolt. Continue to record small mechanistic effects,
  but prioritize literature-backed architecture changes that could reduce a
  meaningful fraction of the roughly 20 mV base SERA. Iteration 116 first tests
  strong latent/output distillation, then the loop returns to higher-capacity
  structural candidates under explicit size and compute controls.
- EXPERIMENT (2026-07-12): Iteration 121 validates a synapse-specific temporal
  signal against an exact tied control, improving voltage/dynamics/response and
  exact F1 significantly. As a full replacement it significantly loses voltage
  and 5 ms F1 to promoted I120, and mixture entropy remains 0.9978, so it is
  rejected rather than tuned.
- DECISION (2026-07-12): Iteration 122 tests the complementary interpretation:
  retain I120's instantaneous projection exactly and add the causal temporal
  basis as a zero-initialized residual, with tied/heterogeneous exact controls.
- EXPERIMENT (2026-07-12): Iteration 122 rejects residual temporal bases. They
  again improve dynamics/response significantly but significantly regress
  voltage, F1 area, and 5 ms F1 versus promoted I120 despite an active residual
  projection. This closes the fixed temporal-basis family without scale tuning.
- DECISION (2026-07-12): Iteration 123 moves to three contiguous nonlinear
  subunits inside each routed branch, against an exact dense-subunit control,
  testing whether pre-tanh spatial information is the missing bottleneck.
- EXPERIMENT (2026-07-12): Iteration 123's contiguous sparse subunits improve
  all six core metrics significantly versus promoted I120, including voltage
  by 0.08208 mV and response by 0.02646 mV. Against exact dense subunits they
  significantly improve voltage and all F1 metrics; a parameter-matched global
  model cannot explain the voltage lead but retains a response edge.
- DECISION (2026-07-12): Iteration 124 repeats sparse, exact dense, promoted,
  and global controls unchanged on a fresh seed before any architectural tuning.
- EXPERIMENT (2026-07-12): Iteration 124 replicates sparse subunits' balanced
  advantage over promoted I120, significantly improving voltage and response.
  The unique sparse effect narrows to a small voltage/response gain over dense
  subunits and disappears against generic global capacity, which has better
  response. No long gate or biological promotion is justified.
- DECISION (2026-07-12): Iteration 125 tests morphology-conditioned per-synapse
  efficacy under an exact scale-zero control, motivated by the dataset's
  morphology-dependent source-site remapping.
- EXPERIMENT (2026-07-12): Iteration 125's morphology-conditioned synapses
  significantly improve all six core metrics versus the exact scale-zero path,
  including voltage by 0.08575 mV and response by 0.02308 mV. They beat a
  near-parameter-matched global model on voltage and tolerant F1 but lose
  response by 0.00445 mV; morphology disparity does not improve.
- DECISION (2026-07-12): Iteration 126 repeats the exact panel unchanged on a
  fresh seed before any promotion or combination with subunits.
- EXPERIMENT (2026-07-12): Iteration 126 fully replicates morphology-conditioned
  synaptic efficacy: all six core metrics improve significantly versus the
  exact control, while generic capacity regresses most metrics on this seed.
- AUDIT (2026-07-12): Leakage-safe raw-voltage affine calibration improves
  SERA by 0.035 mV globally and 0.070 mV per morphology, but voltage gain stays
  near 0.25. Calibration is useful but not the missing fraction-of-error fix.
- DECISION (2026-07-12): Iteration 127 runs an exact 2x2 composition of the
  replicated morphology-synapse mechanism and contiguous subunits.
- EXPERIMENT (2026-07-12): Iteration 127 finds additive structure. Combined
  morphology-conditioned synapses and sparse subunits improve voltage by
  0.03515 mV and response by 0.00820 mV significantly over morphology-only;
  dynamics/F1 changes are uncertain and no core regression is demonstrated.
- DECISION (2026-07-12): Iteration 128 replicates combined versus morphology-
  only and exact global capacity unchanged before the I130 long gate.
- EXPERIMENT (2026-07-12): Iteration 128 strongly replicates composition. The
  combined model improves all six metrics over morphology-only, significantly
  for voltage/dynamics/response/5 ms F1, and dominates an exact-parameter
  global model. It is locked for the I130 long gate.
- DECISION (2026-07-12): Iteration 129 tests only ten post-coupling morphology-
  specific soma affine parameters under an exact scale-zero control.
- EXPERIMENT (2026-07-12): Iteration 129 rejects learned morphology soma
  calibration. It improves voltage by 0.01625 mV but significantly regresses
  intervention response by 0.02641 mV; spike metrics remain tied as designed.
- DECISION (2026-07-12): Iteration 130 long-gates the structural stack without
  the affine, then performs required cleanup, verification, and literature scan.
- CLEANUP (2026-07-12): During I130, removed rejected temporal-basis and soma-
  affine execution paths/tests and repaired deterministic subunit-mask loading
  for old checkpoints. Cleaned suites pass AxoSim 97/97 and AxoBench 24/24.
- LITERATURE (2026-07-12): The I130 scan ranks a verified normalized-coordinate
  synaptic operator and SWC tree encoder above more temporal filtering. I131
  should reconstruct and validate exact morphology route maps before graph use.
- PROMOTION (2026-07-12): Iteration 130 promotes morphology-conditioned
  synaptic efficacy. At 300k it significantly improves all six core metrics
  versus exact both-off control; deployed absolute metrics are voltage 17.15971,
  dynamics 21.30351, response 17.57034, F1 area 0.28866, exact 0.11421, and
  5 ms 0.36600. The stripped deployed model has 76,949 functional parameters.
- DECISION (2026-07-12): Do not promote subunits: they improve voltage and F1
  but significantly regress response at long budget. I131 reconstructs and
  validates exact morphology routing before coordinate/graph experiments.
