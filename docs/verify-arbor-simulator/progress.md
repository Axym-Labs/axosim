# Progress Log — Publishable Mamba Paper / NEURO-BENCH

Reverse-chronological execution log. Historical entries from the pre-port file
use `00:00` when no exact time was recorded.

- IMPLEMENTATION (2026-06-06 12:06): Updated `BenchmarkSuite.compare_models`
  from the stale MVP/legacy comparator to a publishable-schema comparator.
  It now summarizes current `m1_trace_fidelity` fields (`soma_rmse_mv`,
  `soma_mae_mv`, state-weighted trace distance, dynamic distance,
  spike AUC/Brier, raw and calibrated spike-count MAE), lightweight M1b feature
  MAEs, state-conditioned per-state RMSE/support fractions, repaired M2 regime
  RMSEs, and paired structured-M4 teacher/model delta summaries. Legacy
  Gaussian perturbation stability remains as a debug-only compatibility block.
  Added a regression test so model comparison cannot silently collapse back to
  global RMSE/AUC. Verification:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed (`27` tests); py_compile passed for `suite.py`.

- MILESTONE (2026-06-06 12:04): Hardened the M3 morphology-transfer scaffold
  into an explicit morphology/routing compatibility diagnostic rather than a
  bare branch-count placeholder. Fixed SWC branch extraction so soma-started
  basal/apical dendrites are retained, added whole-morphology feature summaries
  (`n_branches`, dendritic length, basal/apical length, apical fraction,
  branch-length summaries, radial extent, bounding box), and separated
  whole-morphology features from the selected branch subset used for model
  input routing. `run_benchmark --run-morphology-transfer` now records this
  comparison under `m3_morphology_transfer` instead of only listing shapes.
  Local SWC check: the available candidates are not defensible same-cell-type
  M3 targets for the current L5/Hay path. Compared with
  `allen_l5_template.swc`, `specimen_327962063.swc` has only `43` extracted
  basal/apical branches, dendritic-length ratio `0.376`, apical fraction delta
  `-0.650`, radial extent ratio `0.399`, and routing shape mismatch;
  `specimen_474267418.swc` is smaller still (`26` branches, length ratio
  `0.212`, radial extent ratio `0.347`). Interpretation: this is useful
  incompatibility evidence, not a successful M3 transfer benchmark. Publishable
  M3 still needs a same-cell-type/closely matched L5 morphology set with
  comparable apical/basal structure and a simulator/routing protocol per
  morphology. Tracked report:
  `artifacts/m3_morphology_feature_comparison_20260606.md`. Verification:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed (`26` tests); py_compile passed for M3/CLI modules.

- MILESTONE (2026-06-06 11:56): Moved the benchmark target from MVP scaffolding
  toward publishable evaluation per user update. Updated `spec.md` to require
  a primary metric upgrade beyond global RMSE, behavior/state-conditioned
  metrics, roughly five structured M4 interventions, and morphology transfer
  unless explicitly scoped out. Added reusable state-conditioned metric code in
  `src/bnn_sim/benchmark/state_metrics.py` and wired it into
  `BenchmarkSuite` as `state_conditioned_fidelity`; M1 now also reports
  `state_weighted_trace_distance_mv` and `dynamic_distance`. Current
  CoreNEURON/Hay 16-trace eval: RNN remains best on global RMSE and
  state-weighted trace distance (`0.7987 mV`, `1.8707 mV`), while prior
  `mamba_exc0925_80k` is best on composite dynamic distance (`0.5249`) because
  its calibrated spike-count behavior is better. Tracked report:
  `artifacts/coreneuron_state_metrics_pair16_20260606.md`.

- MILESTONE (2026-06-06 11:56): Expanded M4 from one event-dropout probe to a
  five-intervention CoreNEURON/Hay suite using the same `16` validation traces:
  `event_dropout`, `exc_dropout`, `inh_dropout`, `temporal_jitter`, and
  `site_silence`. Added reusable input transforms in
  `structured_intervention.py` and a resumable suite runner
  `artifacts/run_coreneuron_m4_intervention_suite.sh`. Also fixed a
  reproducibility hole: the old Hay source default depended on vanished
  `/tmp/neuron-as-deep-net`; the runner now defaults to an ignored durable
  checkout of `SelfishGene/neuron_as_deep_net` and the suite script fetches it
  if missing. Teacher sanity: `event_dropout` soma/spike deltas
  `-2.4102 mV / -4.6250 spikes`, `exc_dropout` `-4.6864 / -4.8125`,
  `inh_dropout` `+3.0588 / +12.8750`, `site_silence` `-2.5980 / -4.5000`.
  `temporal_jitter` is weaker (`-0.0448 / -0.1250`, spike-count interval
  crosses zero) and should be framed as timing sensitivity, not strong OOD.
  Tracked report:
  `artifacts/coreneuron_m4_intervention_suite_20260606.md`.

- IMPLEMENTATION (2026-06-06 00:06): Added per-trace uncertainty summaries
  to paired M4 event-intervention metrics. Teacher reports now include
  `n_traces`, trace-mean soma-delta intervals, and trace-mean spike-count
  delta intervals; model reports include per-trace M4 soma-delta RMSE and
  spike-delta absolute-error intervals. Updated the CoreNEURON MVP runner to
  surface these intervals and to make the M4 limitation trace-count aware
  rather than hardcoding the old `4`-trace smoke. Refreshed the `16`-pair MVP
  report: teacher soma delta interval `[-2.4792, -2.3412] mV`, teacher
  spike-count delta interval `[-6.6562, -2.5938]`, M4 soma-delta intervals
  `soma075 [0.6282, 0.8482]`, `mamba_exc0925 [0.6551, 0.8726]`, `rnn [0.5905,
  0.8345]`; spike-delta MAE intervals `soma075 [2.0602, 5.3148]`,
  `mamba_exc0925 [1.1801, 2.8199]`, `rnn [2.1876, 7.4374]`. Verification:
  py_compile for the runner and M4 module passed, and
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed (`23` tests).

- MILESTONE (2026-06-06 00:03): Expanded the current-teacher M4 event-dropout
  benchmark from `4` paired traces to all `16` validation traces. Generated
  `artifacts/coreneuron_event_dropout_pair16_20260605/` with the accepted
  Hay/CoreNEURON teacher, `35%` nonzero-event dropout, and the same clipped
  soma/local-peak spike target policy. Tracked summary:
  `artifacts/coreneuron_event_dropout_pair16_20260605.md`. Teacher response
  remains interpretable: event fraction ratio `0.6500`, delta spike rate
  `-0.8409 / 1000 steps`, delta soma mean `-2.4102 mV`, direction
  `lower_event_drive/lower_spiking/hyperpolarized`, response detected `true`;
  generation elapsed `401.0 s`.

- EXPERIMENT (2026-06-06 00:03): Re-ran the CoreNEURON/Hay NEURO-BENCH MVP
  on the expanded `16`-trace M4 pair using the same three saved `80k`
  checkpoints and train-split calibrated thresholds. Tracked report:
  `artifacts/coreneuron_neuro_bench_mvp_pair16_20260605.md`. M1/M2 metrics
  reproduce the prior eval. Expanded M4 results: `soma075_calibrated_80k`
  M4 soma-delta RMSE/spike-delta MAE `0.7695 / 3.6875`;
  `mamba_exc0925_80k` `0.7935 / 2.0000`; `rnn_exc0925_80k`
  `0.7522 / 4.8125`. All preserve qualitative M4 soma/spike direction.
  Interpretation: RNN remains strongest for trace and M4 voltage delta,
  `soma075` remains current Mamba voltage/RMSE path, and prior
  `mamba_exc0925_80k` remains better for calibrated spike-count/M4
  spike-delta behavior. This removes the immediate `4`-pair smoke limitation,
  but broader robustness still needs additional intervention types or seeds.

- IMPLEMENTATION (2026-06-05 23:55): Tightened NEURO-BENCH MVP report
  guardrails after the current CoreNEURON/Hay smoke comparison. Fixed the
  top-level `limitations` field so paired M4 reports no longer claim
  structured M4 is unimplemented; they now state that M4 is limited to the
  supplied paired traces and needs expansion before robustness claims. Added
  explicit M2 `what_failure_means` notes for far-subthreshold, near-spike,
  refractory, post-input-event, and spike-timing failures, plus low-support
  warnings for underrepresented MVP regimes. Refreshed the current CUDA MVP
  evaluator on the same three `80k` checkpoints; metrics reproduced the
  previous report, paired-M4 limitations were corrected, and current M2 support
  warnings were empty on the `16`-trace validation split. Verification:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed (`23` tests).

- MILESTONE (2026-06-05 23:50): Returned to the NEURO-BENCH MVP benchmark
  after the simulator/model-selection detour. Added a current-teacher
  CoreNEURON/Hay event-dropout M4 pair generator,
  `artifacts/run_coreneuron_event_dropout_pair.py`, fixing output paths to be
  resolved before NEURON mechanism-directory `chdir`. Generated a `4`-trace
  accepted-teacher event-dropout pair by dropping `35%` of nonzero input
  events and re-simulating the Hay/CoreNEURON teacher. Tracked summary:
  `artifacts/coreneuron_event_dropout_pair_20260605.md`. Teacher response is
  interpretable and qualitatively expected: event fraction ratio `0.6493`,
  delta spike rate `-0.6818 / 1000 steps`, delta soma mean `-2.3395 mV`,
  direction `lower_event_drive/lower_spiking/hyperpolarized`, response
  detected `true`.

- EXPERIMENT (2026-06-05 23:50): Added and ran the current CoreNEURON/Hay
  NEURO-BENCH MVP runner, `artifacts/run_coreneuron_neuro_bench_mvp.py`.
  It evaluates M1, lightweight M1b, repaired M2, and the new M4 event-dropout
  pair using calibrated spike thresholds learned on the `32`-trace training
  split and frozen for validation/M4. Tracked report:
  `artifacts/coreneuron_neuro_bench_mvp_20260605.md`. Model summary on the
  accepted `16`-trace validation split plus `4`-trace M4 pair:
  `soma075_calibrated_80k` RMSE/AUC `0.8507 / 0.9666`, calibrated count MAE
  `3.8125`, calibrated timing F1 `0.1272`, M4 soma-delta RMSE `0.6803`,
  M4 spike-delta MAE `3.5000`; prior Mamba baseline RMSE/AUC
  `0.8763 / 0.9632`, calibrated count MAE `2.0625`, M4 spike-delta MAE
  `1.5000`; RNN RMSE/AUC `0.7987 / 0.9723`, near-spike RMSE `3.9798`, M4
  soma-delta RMSE `0.6495`. All three models preserve the qualitative
  M4 event-dropout direction for soma and spike count. Interpretation:
  `soma075` remains the current Mamba voltage/RMSE path, RNN remains the best
  trace model, and prior Mamba has stronger calibrated spike-count/M4
  spike-delta behavior in this small smoke test. Next benchmark priority:
  expand M4 beyond `4` pairs or add uncertainty/error bars before making
  robustness claims.

- EXPERIMENT (2026-06-05 23:26): Selected and validated the current Mamba
  metric path: `soma075` Mamba with a learned spike threshold. Artifact:
  `artifacts/coreneuron_soma075_calibrated_20260605.md`. Fresh training used
  the accepted CoreNEURON/Hay `exc_event_keep_prob=0.925` teacher, exactly
  `80k` presentations, `spike_loss_weight=0.25`, `soma_loss_weight=0.75`,
  and CUDA. The spike threshold was learned on the `32` training traces by
  matching calibration-set spike prevalence, then frozen for the `16`-trace
  validation evaluation; this avoids validation-target threshold leakage.
  Learned threshold: `-2.6941`. Result: RMSE/AUC `0.8507 mV / 0.9666`,
  near-spike RMSE `4.6228`, calibrated spike-count MAE `3.8125`, calibrated
  timing F1 `0.1272`, diagnostic plots produced. Raw zero-threshold spike
  sanity still flags `threshold_zero_predicts_no_spikes`, but calibrated
  threshold flags are none. Interpretation: accept this as the current
  voltage/RMSE-oriented Mamba path with explicit reporting of calibrated and
  raw spike operating points; do not claim spike timing is solved.

- DECISION (2026-06-05 23:26): Current simulator-path framing for paper notes:
  CoreNEURON/Hay is preferable to the old NeuronIO-only path because it keeps
  the source Hay morphology/HOC/channel lineage while giving an explicit
  NEURON/CoreNEURON execution path, reproducible deterministic synapse
  mechanisms, validated CPU-vs-GPU equivalence on the deterministic subset,
  much faster data loading/training throughput in the current shard policy,
  and a clear route to future population-level parallelism. It is preferable
  to the abandoned Arbor path for Paper 1 because it avoids the unresolved
  Arbor morphology/mechanism/resting-regime mismatch while basically
  replicating the old NeuronIO clipped-voltage dynamic regime. Supporting
  evidence is in `artifacts/simulator_mamba_comparison_20260605.md`: input
  density, clipped voltage std, and clipped `|dV/dt|` p95/p99 are close to old
  NMDA (`0.010600` vs `0.010554`; `3.2606` vs `3.2363 mV`;
  `0.4667/0.6706` vs `0.4688/0.6875 mV/step`). Remaining caveat: spike
  distribution is not a full replication and must be reported explicitly.

- ANALYSIS (2026-06-05 22:53): Compared accepted CoreNEURON/Hay
  `exc_event_keep_prob=0.925` against the old NeuronIO/NMDA setup for both
  simulator statistics and Mamba metrics. Artifact:
  `artifacts/simulator_mamba_comparison_20260605.md`. Summary: current
  CoreNEURON/Hay matches old NMDA closely in input density (`0.010600` vs
  `0.010554` canonical old), clipped-voltage std (`3.2606` vs `3.2363 mV`),
  and clipped `|dV/dt|` p95/p99 (`0.4667/0.6706` vs `0.4688/0.6875
  mV/step`). It is not a full statistical replication: current spike rate is
  lower than old canonical NMDA (`0.8438` vs `0.9896` spikes/1000), spike
  counts are less variable (`5.06 +/- 4.34` vs `5.94 +/- 7.88` spikes/trace), and
  fewer traces are non-spiking (`0.125` vs `0.438`). Mamba `80k` voltage RMSE
  is better on current CoreNEURON/Hay (`0.8763 mV`) than old NMDA canonical
  (`0.9286 mV`) and old first-16 sensitivity (`0.9886 mV`), but AUC and
  prevalence-matched timing are worse (`0.9632` / `0.1468` vs old canonical
  `0.9777` / `0.2193`). Interpretation: current setup is a similar
  clipped-voltage dynamic regime and better for Mamba RMSE, but old NeuronIO
  remains stronger on spike ranking/timing and the spike-distribution mismatch
  should be reported explicitly.

- EXPERIMENT (2026-06-05 22:35): Corrected the RMSE autoresearch loop after
  the prior conductance-only loop was too narrow. Added
  `artifacts/run_coreneuron_training_autoresearch.sh`, a resumable,
  parallelizable runner with `GPU_IDS`/`MAX_PARALLEL` controls, CPU refusal,
  and an `80k` presentation cap per candidate. This loop keeps the accepted
  CoreNEURON/Hay teacher fixed (`exc_event_keep_prob=0.925`,
  `inh_event_keep_prob=1.0`) and tests training/data-policy interventions, so
  positive results are metric/training evidence rather than simulator-fidelity
  evidence. Six Mamba candidates completed on the accepted `32` train / `16`
  validation split, all with diagnostic plots. Same-loop baseline reproduced
  prior metrics exactly: RMSE/AUC `0.8763 mV / 0.9632`, near-spike RMSE
  `4.6860`, timing F1 matched `0.1468`, no sanity flags. Best RMSE was
  `soma075` (`spike_loss_weight=0.25`, `soma_loss_weight=0.75`) at
  `0.8507 mV`, AUC `0.9666`, near-spike RMSE `4.6228`, timing F1 `0.1598`,
  but it triggered `threshold_zero_predicts_no_spikes`, so it is a candidate
  for fresh-seed validation rather than a default. Cleanest improvement was
  `longctx1000` (`1000`-step context, `300` burn-in): RMSE `0.8586 mV`,
  AUC `0.9631`, near-spike RMSE `4.4535`, no sanity flags, but timing F1 fell
  to `0.1070`. Negative/secondary results: slope loss `0.8774 mV` with a
  sanity flag; sparse soma loss `0.8808 mV` but clean and slightly better
  timing; official-style sampling `0.9025 mV` but best AUC/timing in the loop.
  Consolidated report:
  `artifacts/coreneuron_training_autoresearch_20260605.md`. Next priority:
  validate `soma075` and `longctx1000` on a fresh seed/split, then consider a
  combined long-context plus modest voltage-weight run only if it preserves
  calibration sanity.

- DECISION (2026-06-05 22:35): Clarified CPU-vs-GPU simulator usage. We chose
  CPU for the current single-cell CoreNEURON/Hay dataset generation because
  CPU and GPU CoreNEURON had already matched to floating-point noise on the
  deterministic Hay validation probe, while the single-cell GPU path was
  slower on this machine due to CoreNEURON/NVHPC launch/setup overhead
  (`8 x 500 ms`: CPU `24.9 s`, GPU `131.3 s`). That was a throughput decision
  after equivalence validation, not a correctness fallback. GPU remains the
  intended path for population/batched-cell follow-up work where Arbor-like or
  CoreNEURON parallelism can amortize launch overhead.

- EXPERIMENT (2026-06-05 21:22): Completed a bounded conductance-calibration
  autoresearch loop targeting RMSE improvement without degrading simulator
  defensibility. Added explicit `--exc-gmax-scale`/`--inh-gmax-scale`
  controls to the CoreNEURON/Hay dataset generator and
  `artifacts/run_coreneuron_gmax_autoresearch.sh`. Seven teacher candidates
  were ranked on fresh matched `16 x 6s` validation traces using only
  predeclared teacher statistics, not model metrics. The teacher-stat winner
  was `exc_gmax_scale=0.925`, `inh_gmax_scale=1.0`, with unchanged event
  distribution (`0.010959` event fraction), spike rate `0.8958 Hz`, clipped
  soma mean/std `-68.2684/3.3319 mV`, and clipped `|dV/dt|` p95/p99
  `0.4465/0.6884`; it slightly beat the existing `exc_event_keep_prob=0.925`
  teacher objective because it preserves events and matches dV/dt better.
  However, `80k` Mamba model evaluation on this setting produced worse
  RMSE/AUC than the current accepted run: `0.9769 mV / 0.9564`, near-spike
  RMSE `5.2105`, timing F1 `0.1627`, no sanity flags, plots produced. A
  hotter but still small `exc_gmax_scale=0.95` candidate also failed:
  RMSE/AUC `1.0438 mV / 0.9592`, near-spike RMSE `4.8532`, no sanity flags,
  plots produced. Decision: do not replace the provisional default for metric
  reasons. Conductance scaling is a defensible alternative lever, but this
  loop is negative evidence for RMSE improvement under the current `80k`
  protocol.

- MILESTONE (2026-06-05 20:10): Validated the CoreNEURON/Hay GPU execution
  path on this machine using NEURON `9.0.1`, NVIDIA HPC SDK `26.3`, CUDA
  `13.1`, and the RTX `5090`. The installed conda NEURON `8.2.7` remains
  CPU-only (`CORENRN_ENABLE_GPU OFF`), and a source NEURON `8.2.7` GPU build
  failed at device linkage for `coreneuron::celsius`, so the working GPU path
  is explicitly NEURON `9.0.1`. Mechanisms compiled for CPU NEURON and GPU
  CoreNEURON with the deterministic Hay AMPA/NMDA/GABA subset; unused
  `epsp.mod` was excluded because it fails 9.x NMODL translation and is not
  part of the current deterministic generator. CPU-vs-GPU validation on
  `8 x 500 ms` seeded traces matched to floating-point noise: spike/event
  metrics exactly equal, maximum row-wise soma-stat deltas `<=7.6e-11 mV`
  and other voltage-summary deltas `~1e-13` to `1e-12`. Single-cell GPU was
  slower than CPU (`131.3 s` vs `24.9 s` elapsed), as expected; this is a
  correctness and future population-scaling milestone, not a single-cell speed
  claim. Added
  `artifacts/coreneuron_hay_port_probe_20260605/run_coreneuron_hay_gpu_validation.sh`
  to reproduce the CPU/GPU comparison from the local NEURON 9 build.

- MILESTONE (2026-06-05 18:50): Completed the first bounded CoreNEURON/Hay
  teacher-distribution calibration pass under the `<=80k` presentation rule.
  Consolidated
  artifact: `artifacts/coreneuron_exc0925_results_20260605.md`. Accepted
  provisional simulator setting is deterministic Hay/CoreNEURON with
  `exc_event_keep_prob=0.925`, `inh_event_keep_prob=1.0` because it better
  matches the historical NeuronIO/NMDA teacher distribution, not because it
  was the best RMSE setting. Teacher stats move from an over-hot uncalibrated
  CoreNEURON split (`1.5625` spikes/1000, `-67.186/3.452 mV` soma mean/std)
  to `0.8438` spikes/1000 and `-68.016/3.261 mV`, close to the historical
  NMDA validation aggregate (`0.9896`, `-67.595/3.236`). Downstream Mamba
  checkpoint evaluation also improves from RMSE/AUC `1.1556 mV / 0.9648` to
  `0.8493 mV / 0.9804`, and near-spike RMSE improves from `5.6565` to
  `4.5663 mV`, but those model metrics are secondary evidence. The more
  aggressive `exc_event_keep_prob=0.90` produced better RMSE `0.7869 mV` and
  AUC `0.9863`, but was rejected for the default because it overcorrected the
  teacher distribution (`0.6563` spikes/1000, `-68.352 mV` mean). This is a
  teacher-distribution-first, defensibility-over-metric decision. For
  paper-grade use, rerun selection on fresh seeds with predeclared simulator
  statistic distance before reporting any model result.

- EXPERIMENT (2026-06-05 18:50): Trained Mamba and RNN on the accepted
  CoreNEURON/Hay `exc=0.925` setting with exactly `80k` presentations each.
  Train data: `data/hay-coreneuron-source-train32-exc0925-20260605/train`;
  validation data: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
  Combined NEURO-BENCH report:
  `artifacts/hay_coreneuron_exc0925_80k_combined_neuro_bench_20260605/report.md`.
  Mamba `80k`: RMSE/AUC `0.8763 mV / 0.9632`, near-spike RMSE `4.6860 mV`,
  spike-count MAE@0 `4.7500`, timing F1 matched `0.1468`, no sanity flags.
  RNN `80k`: RMSE/AUC `0.7987 mV / 0.9723`, near-spike RMSE `3.9798 mV`,
  spike-count MAE@0 `4.1250`, timing F1 matched `0.0779`, no sanity flags.
  Both produced diagnostic SVG plots. Interpretation: RNN wins RMSE on this
  small accepted split, while Mamba remains much faster for iteration; spike
  timing is still weak for both and remains a higher-value metric target than
  saturated AUC.

- IMPLEMENTATION (2026-06-05 18:50): Added
  `artifacts/coreneuron_hay_port_probe_20260605/generate_coreneuron_hay_dataset.py`
  to write NeuronIO-compatible shards from the CoreNEURON-compatible Hay path,
  with explicit `--exc-event-keep-prob` and `--inh-event-keep-prob` input
  calibration knobs. Defaults preserve the exact CoreNEURON/Hay generator.
  Added `artifacts/run_coreneuron_exc0925_80k_models.sh`, which refuses CPU
  training and refuses any configuration above `80k` presentations before
  training/evaluating the accepted setting.

- VERIFICATION (2026-06-05 18:50): Verified the richer metric answer: current
  NEURO-BENCH already includes metrics beyond RMSE. M1 reports spike AUC,
  Brier/log-loss, count and prevalence-matched spike diagnostics; M1b reports
  electrophysiological feature fidelity; M2 reports regime-aware RMSE/MAE and
  spike-timing F1. `tests/test_benchmark_suite.py` passed earlier in this
  session (`7 passed`); final focused verification
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py
  tests/test_arbor_sim.py tests/test_cli.py -q` passed (`24 passed`). Python
  compile checks and shell syntax checks also passed, and all combined reports
  produced SVG diagnostic plots. Recommendation for paper framing: headline
  RMSE for comparability, but use M1b/M2 to explain biological/temporal
  failure modes.

- MILESTONE (2026-06-05 17:58): Selected and validated the NEURON-family
  replacement path for the current time-constrained benchmark arc:
  CoreNEURON/Hay, not Arbor. Artifact:
  `artifacts/coreneuron_hay_port_probe_20260605/report.md`. Summary:
  fixed-step NEURON differs negligibly from CVode on the 2x3 s source-Hay
  probe (`0.0 Hz` spike-rate delta; clipped mean/std deltas about
  `-0.0027 mV`; clipped `|dV/dt|` p95/p99 deltas below `0.003 mV/step`).
  Deterministic Hay synapse subsets (`DetAMPANMDA`, `DetGABA`) plus original
  Hay channel MOD files compile for NEURON and CPU CoreNEURON. NEURON
  fixed-step and CoreNEURON fixed-step match exactly on `1x100ms`, `1x1s`,
  `1x3s`, and spiking `2x3s` PatternStim probes (`max aggregate diff 0.0`,
  `max row diff 0.0`). The spiking `2x3s` probe has event fraction
  `0.010593`, spike count mean/std `6.5/4.5`, clipped soma mean/std
  `-67.9793/3.7700 mV`, and clipped `|dV/dt|` p95/p99
  `0.5085/0.8386 mV/step`, i.e. the NeuronIO-like dynamic regime. GPU status:
  CoreNEURON exposes GPU controls locally, but actual GPU execution is not yet
  validated because `nvc++` is missing; install NVIDIA HPC SDK before claiming
  GPU-run evidence. Next priority: convert the probe into the replacement
  dataset generator and run Mamba-first model metrics with no run over `80k`
  presentations.

- IMPLEMENTATION (2026-06-05 17:58): Added
  `artifacts/coreneuron_hay_port_probe_20260605/run_coreneuron_hay_probe.py`
  and deterministic CoreNEURON-compatible synapse mechanisms. Important
  implementation detail: `PatternStim`/`ParallelContext.gid_connect` is needed
  for CoreNEURON event delivery; direct `NetCon(None).event(...)` aborted.
  Also fixed the probe runner so relative `--output` paths resolve before
  changing into the mechanism directory.

- DECISION (2026-06-05 17:42): Due to time constraints, do not spend a
  `2-3` day Arbor feasibility sprint on the current paper path. Rationale:
  evidence so far shows the Arbor mismatch is not a simple configuration gap
  but a coupled morphology/mechanism/resting-regime/active-dynamics gap; a
  faithful Hay-to-Arbor port may still be valuable, but it is too large and
  uncertain for the current benchmark milestone. Leave Arbor open for the
  population-level simulation follow-up project, where its GPU/HPC scaling,
  many-morphology support, and heterogeneous-network design are more central.
  Current replacement path: prioritize a NEURON-family GPU-parallelizable
  route, starting with CoreNEURON because it preserves the Hay/NEURON source
  lineage better than a separate simulator rewrite.

- ARTIFACT (2026-06-05 16:16): Added executable batch script
  `artifacts/run_hay_source_benchmark_batch.sh` for larger source-Hay runs.
  Defaults: generate `32` train + `32` val direct Hay/NEURON traces, then run
  Branch-ELM evaluation in conda `bnn_sim` and official-Mamba evaluation in
  `.venv-mamba`. The observed `4+4` generation time was `315.7 s`, so the
  default `32+32` generation is expected to take roughly `40-45 min` before
  the short benchmark passes. Verified with `bash -n` and `chmod +x`.

- IMPLEMENTATION (2026-06-05 16:14): Added a direct Hay/NEURON dataset writer
  `artifacts/generate_hay_neuron_dataset.py`. It writes NeuronIO-compatible
  `inputs`/`targets` shards with the historical channel layout (first `639`
  excitatory `+1`, second `639` inhibitory `-1`), local-peak spike labels,
  and `-55 mV` clipped/normalized soma targets. Two smoke failures were fixed
  before use: relative output paths after `os.chdir(source_root)`, and
  duplicate `nrn_load_dll` calls across train/val splits in one process.

- EXPERIMENT (2026-06-05 16:14): Generated a small source-teacher dataset with
  the original Hay morphology/biophysics/MOD mechanisms:
  `data/hay-neuron-source-sanity-4_4-20260605/` (`4` train + `4` val traces,
  `6 s`, CVode, NEURON `8.2.7`). Val split stats: event fraction `0.011145`,
  spike rate `1.8333 / 1000 steps`, clipped soma mean/std
  `-67.4122 / 3.9298 mV`, soma min/max `-77.27/-55.00 mV`. This split is
  benchmark-compatible and in the NeuronIO-like dynamic regime, though one val
  trace is relatively hot; treat it as source-teacher sanity evidence rather
  than final ranking.

- EXPERIMENT (2026-06-05 16:14): Evaluated trained NMDA checkpoints on the
  direct Hay source split with NEURO-BENCH M1/M1b/M2 and diagnostic plots.
  Consolidated artifact:
  `artifacts/hay_source_neuro_bench_eval_4_4_20260605.md`. Branch-ELM results
  used conda `bnn_sim` GPU: `branch_elm_official80k` RMSE/AUC
  `1.1734 mV / 0.9514`, `branch_elm_random800k` `0.9840 mV / 0.9707`.
  Official-Mamba results used `.venv-mamba` GPU: `mamba_compact80k`
  `1.1912 mV / 0.9424`, `mamba_compact480k_partial`
  `1.1427 mV / 0.9666`. All produced finite outputs and plots. Shared sanity
  flag: `threshold_zero_predicts_no_spikes`, so prevalence-matched spike
  metrics remain more informative than zero-logit threshold counts.

- VERIFICATION (2026-06-05 16:14): Verified environment constraints and report
  generation. Conda `bnn_sim` has Torch `2.12.0.dev20260408+cu128` with
  `sm_120`, but its `mamba_ssm` extension has a Torch ABI mismatch; `.venv`
  Torch lacks `sm_120` CUDA support and no `mamba_ssm`; `.venv-mamba` has
  Torch `2.7.1+cu128`, `sm_120`, and `mamba_ssm 2.3.2.post1`, so official
  Mamba evaluation must use `.venv-mamba`. Updated the benchmark report title
  from NMDA-specific wording to generic `NEURO-BENCH Evaluation` for non-NMDA
  teacher datasets.

- IMPLEMENTATION (2026-06-05 16:01): Installed NEURON `8.2.7` into the
  existing `/home/davwis/miniconda3/envs/bnn_sim` environment and successfully
  compiled the original Beniaguev/Hay MOD mechanisms with `nrnivmodl mods`
  under `/tmp/neuron-as-deep-net/L5PC_NEURON_simulation`. Added reusable
  artifacts `run_hay_neuron_baseline.py` and `run_hay_neuron_driven_probe.py`
  for direct source-teacher diagnostics.

- EXPERIMENT (2026-06-05 16:01): Direct NEURON/Hay no-input baseline is now
  measured, not inferred. With original `cell1.asc`, `L5PCbiophys5b.hoc`,
  compiled mechanisms, `v_init=-76`, and CVode, soma settles to
  `-81.128 mV` over `3 s` with no spikes
  (`artifacts/hay_neuron_baseline_cvode_20260605.json`). Fixed a runner bug
  where `h.run()` reused NEURON's default global `v_init=-65`; corrected script
  now sets `h.v_init` and uses `h.continuerun` after `h.finitialize`.

- EXPERIMENT (2026-06-05 16:01): Ran direct driven NEURON/Hay probes using the
  original segment-level NMDA/GABA_A setup and NeuronIO-like smoothed input
  generator, without dendritic voltage recording. A 3 s smoke trace matched
  the target regime immediately: event fraction `0.011894`, spike rate
  `0.6667 Hz`, clipped soma mean/std `-67.4962/2.6761 mV`, clipped `|dV/dt|`
  p95/p99 `0.5337/0.7719`
  (`artifacts/hay_neuron_driven_probe_3s_20260605.json`). A 6 s, 2-sample
  CVode probe gave aggregate event fraction `0.011076`, spike rate
  `0.5833 +/- 0.4167 Hz`, clipped soma mean/std
  `-66.6061 +/- 0.1640 / 2.9004 +/- 0.6505 mV`, and clipped `|dV/dt|`
  p95/p99 `0.4913/0.6883`
  (`artifacts/hay_neuron_driven_probe_6s_2sample_20260605.json`). This
  directly reproduces the historical NeuronIO dynamic regime far better than
  Arbor tuning did.

- DECISION (2026-06-05 16:01): Update priority: direct NEURON/Hay is runnable
  locally and should be the reference for NeuronIO-equivalent claims. Arbor
  should be treated as a distinct teacher distribution unless we undertake a
  real Hay-to-Arbor mechanism/morphology port. The highest-success path for
  publishable benchmarking is to use existing/direct Hay/NeuronIO data for
  NeuronIO-comparable model metrics, and use corrected Arbor routing/labels as
  a separate simulator benchmark with explicit limitations, not as a drop-in
  replacement for NeuronIO.

- EXPERIMENT (2026-06-05 15:46): Ran a small mechanism-scale probe based on the
  Allen-vs-Hay source-level mismatch. Artifact:
  `artifacts/arbor_mechanism_scale_probe_20260605/mechanism_scale_probe.md`.
  Source comparison: Hay basal dendrites are passive+Ih only, while current
  Allen Arbor dendrites include `NaV`, `Kv3_1`, and `Im_v2`; Allen apical
  `Kv3_1`/`Im_v2` densities also differ strongly from Hay apical settings.
  Probe used distributed `639` sites/polarity, NeuronIO-smoothed input, local
  peak labels, weight `0.3`, all-region `e_pas` shift `10`, and five mechanism
  variants. Baseline bracket gave spike rate `1.1667 Hz`, clipped mean/std
  `-77.4940/2.3556 mV`, clipped `|dV/dt|` p95/p99 `0.8021/2.7340`.
  Suppressing basal/dendritic `NaV/Kv/Im` improved `dV/dt` tails
  (`0.5081/0.9187`) and std (`3.5040 mV`) but killed spikes and stayed
  hyperpolarized (`-76.8787 mV`). More aggressive dendrite+apical K/M
  suppression made `dV/dt` close to NeuronIO (`~0.40-0.59`) but produced no
  spikes and very low voltage variance. Verdict: the remaining gap is a
  coupled mechanism/regime mismatch, not a single scalar tuning problem.

- DECISION (2026-06-05 15:46): Stop piling on scalar Arbor tuning for now.
  High-success next options are (a) install/run NEURON locally to measure the
  Hay no-input and controlled-input baselines directly, or (b) explicitly frame
  Arbor as a separate teacher distribution with corrected routing/label
  semantics and report why it is not NeuronIO-equivalent. Porting Hay
  mechanisms/morphology into Arbor may be possible but is a larger simulator
  port, not a quick calibration modification.

- VERIFICATION (2026-06-05 15:43): Checked for the reported hanging workflow.
  No active Python/Arbor/training/evaluation process was running, and GPU
  utilization was idle (`0%`, `42 MiB` used). Only unrelated system Python
  daemons and the local `monitor_cv` server were present. No compute job was
  left hanging from the previous simulator sweeps.

- EXPERIMENT (2026-06-05 15:43): Ran a zero-input Arbor resting diagnostic to
  isolate baseline cell dynamics from synaptic drive. Artifact:
  `artifacts/arbor_zero_input_resting_diagnostic_20260605.json`. With current
  Allen active/passive mechanisms and fixed nominal parameters, no-input soma
  settles far below the NeuronIO driven voltage regime: all-region `e_pas`
  shift `0/3/6/10/15 mV` gives last-500 ms mean soma
  `-82.60/-80.17/-77.86/-74.92/-71.07 mV`; `20 mV` causes spontaneous
  spiking (`26` spikes in `3 s`). A naive Hay-like passive conductance/cm
  scaling made rest even more hyperpolarized (`-87.24 mV` final), so copying
  only Hay passive scalars onto the Allen active model is not a sensible fix.

- EXPERIMENT (2026-06-05 15:43): Used the zero-input result to run one narrow
  distributed-site high-`e_pas` refinement, instead of another broad sweep:
  `artifacts/arbor_distributed_sites_high_epas_refine_20260605/` (`9`
  settings, `2` traces each, NeuronIO-smoothed input, distributed `639`
  sites/polarity, `101` branches, local-peak labels). Best row was weight
  `0.3`, inhibition `1`, `e_pas` shift `10`: event fraction `0.010466`,
  spike rate `0.6667 Hz`, clipped mean/std `-77.6351/2.6889 mV`, clipped
  `|dV/dt|` p95/p99 `1.1184/3.3157`. Weight `0.4`, shift `12` overfired
  (`2.1667 Hz`) while still staying hyperpolarized (`-76.3457 mV`). Verdict:
  higher passive reversal shifts improve no-input rest but do not close the
  driven voltage-level gap under the current Allen mechanism set; further
  scalar weight/passive sweeps have low expected value.

- DECISION (2026-06-05 15:43): Next sensible path is mechanism-level diagnosis,
  not surrogate training or more scalar tuning. Compare Allen-vs-Hay active and
  passive mechanism families and consider whether the Arbor teacher should be
  framed as a distinct, defensible distribution. If continuing to tune Arbor
  toward NeuronIO, do it via mechanism-aware hypotheses (e.g. active current
  densities, region-specific passive properties, synapse kinetics), not broad
  objective search.

- USER (2026-06-05 15:43): User asked to continue and noted that something
  apparently hung in the workflow.

- IMPLEMENTATION (2026-06-05 15:14): Investigated and repaired the most
  obvious morphology/input-site mismatch in the Arbor proxy. The prior Arbor
  path routed `1278` logical channels onto only `45` physical branch-level
  synapse sites, with excitation at branch position `0.65` and inhibition at
  `0.35`. Added opt-in `synapse_routing="distributed"` with
  `distributed_sites_per_polarity=639`, separating stimulus branch routing
  from physical Arbor event-target routing and distributing sites along
  selected Allen branches by branch length. Also exposed this in the CLI and
  pilot/calibration scripts. Added `spike_label_mode="local_peak"` with
  `spike_peak_threshold_mv=-25` to support NeuronIO-style spike-label
  semantics for future Arbor datasets.

- EXPERIMENT (2026-06-05 15:14): Distributed-site morphology/routing test was
  partially successful but did not close the simulator gap. A first
  NeuronIO-smoothed + distributed-site + `101`-branch pilot with old calibrated
  weights (`data/arbor-neuronio-smoothed-distributed-sites-pilot-4_4-20260605/`)
  badly overfired: val spike rate `29.9167 Hz`, soma mean/std
  `-62.3110/10.3368 mV`. This shows old branch-collapsed weights are invalid
  once synapses are no longer collapsed into one conductance state per branch.
  A low-weight distributed sweep
  (`artifacts/arbor_distributed_sites_low_weight_calibration_20260605/`) found
  a better bracket: weight `0.35`, inhibition `1`, passive shift `0` gave
  event fraction `0.010144`, spike rate `1.3333 Hz`, clipped mean/std
  `-77.4393/2.5517 mV`, clipped `|dV/dt|` p95/p99 `0.9742/2.8771`.
  A passive-refinement sweep
  (`artifacts/arbor_distributed_sites_passive_refine_20260605/`) gave best
  row weight `0.4`, inhibition `1`, passive shift `6`: spike rate
  `1.5000 Hz`, clipped mean/std `-76.9948/2.7635 mV`, clipped `|dV/dt|`
  `1.3174/3.3590`. Verdict: distributed sites improve the implementation
  fidelity and can bring `dV/dt` closer than the overexcited pilot, but simple
  weight/passive tuning still leaves the Arbor voltage level too hyperpolarized
  or overfired.

- VERIFICATION (2026-06-05 15:14): Verified morphology/routing and spike-label
  changes with `PYTHONPATH=src .venv/bin/python -m pytest tests/test_arbor_sim.py
  tests/test_cli.py -q` (`17 passed`) and `py_compile` for touched source and
  experiment scripts. The new tests check NeuronIO-smoothed slow-drive
  structure, `639` distributed physical synapse sites per polarity, ex/inh
  pairing on the same site lattice, and local-peak vs threshold-crossing spike
  label semantics.

- DECISION (2026-06-05 15:14): Keep the distributed-site routing as a
  correctness improvement and a higher-priority simulator variant, but do not
  claim success on matching NeuronIO yet. The next sensible modifications are
  not broad surrogate training or more arbitrary weight sweeps; they are
  simulator diagnostics around the remaining voltage-level gap: isolate
  resting/passive membrane behavior, compare Allen-vs-Hay active/passive
  mechanisms, and decide whether an Arbor teacher should be presented as a new
  distribution rather than tuned toward NeuronIO.

- USER (2026-06-05 15:14): User asked to investigate morphology mismatch now,
  report success, continue with sensible modifications, and spend more time
  thinking before choosing next actions because the branch-count mismatch
  appears higher priority than previous calibration attempts.

- DECISION (2026-06-05 14:59): After the failed weight-only and observation-only
  Arbor calibration attempts, update expected success rates downward for
  "simple tuning closes the NeuronIO-Arbor RMSE gap". Prefer higher-success,
  correctness-preserving steps: (1) quantify/repair morphology and input-site
  mismatch, (2) verify spike-label and observation semantics, (3) keep
  simulator claims scoped to the actual teacher distribution, and (4) postpone
  additional surrogate training until the teacher distribution is defensible.
  Reports do not need continuous polishing during execution; generate or
  polish consolidated reports at task boundaries unless an intermediate
  artifact is needed for decision-making.

- VERIFICATION (2026-06-05 14:59): Confirmed that morphology/input-site
  mismatch is a plausible high-priority contributor to the simulator gap:
  historical NeuronIO uses `639` dendritic segments/input sites (`262` basal,
  `377` apical), whereas the current Arbor proxy routes `1278` input channels
  onto only `45` selected branches (`20` dend, `25` apic) from a `104`-branch
  Allen morphology. This compresses spatial targeting and branch-specific
  cable filtering substantially, so it is a stronger candidate than further
  global weight tuning for explaining why the same event statistics can produce
  different voltage/spike dynamics.

- DECISION (2026-06-05 14:59): Record non-simulator improvements relative to
  the upstream Branch-ELM setup for later paper/report writing. Our pipeline
  appears to improve on Branch-ELM through faster shard-based data loading and
  a better data policy: full-trace/window sampling, random/full-trace or
  official-style sampling controls, shard-local shuffling/cache controls, and
  explicit validation/benchmark semantics. A likely reason for some performance
  improvements is that our training sees more of the dataset and less
  repetitive fixed-window structure, but this remains an attribution hypothesis
  until measured with a controlled data-policy ablation.

- USER (2026-06-05 14:59): User asked to focus on higher-success-rate paths
  after failed calibration results, said reports can be generated at task end
  rather than continuously, flagged morphology branch-count differences as a
  possible metric-gap contributor, and asked to note data-loading/data-policy
  improvements over Branch-ELM in `progress.md`.

- EXPERIMENT (2026-06-05 14:55): Ran a NeuronIO-smoothed Arbor stimulus pilot
  with the same fixed calibrated Arbor cell/synapse parameters as the earlier
  iid-input pilot (`32` train / `16` val traces, CUDA Arbor, `6 s`, artifact
  `data/arbor-neuronio-smoothed-calibrated-pilot-32_16-20260605/`). The input
  distribution now matches NeuronIO much better: event fraction `0.010892`
  vs NMDA `0.010618`, event-count autocorr at 100 ms `0.168` vs `0.196`,
  100 ms total events p50/p95/p99 `1386/1845/2249` vs `1362/1842/2176`,
  and 100 ms E-I diff p50/p95/p99 `155/360/552` vs `144/387/543`.
  However, the same Arbor weights become over-excited under structured drive:
  validation spike rate `9.9896 Hz`, soma std `6.9468 mV`, `|dV/dt|`
  p95/p99 `6.428/39.444 mV/step`, and spike-triggered mean peak
  `-17.0 mV`, versus NMDA `0.7533 Hz`, `3.036 mV`, `0.438/0.687`, and
  clipped peak `-55.4 mV`. Artifact:
  `artifacts/teacher_protocol_comparison_neuronio_smoothed_arbor_20260605.md`.

- EXPERIMENT (2026-06-05 14:55): Ran a small CUDA calibration sweep under the
  new `neuronio_smoothed` Arbor input protocol (`12` settings, `4` traces,
  `3 s` each) to test whether synaptic weights alone can restore the NMDA
  regime. Best rows recover spike rate only by landing in a hyperpolarized,
  too-sharp voltage regime: weight `0.75`, inhibition `1.0` gives event
  fraction `0.009877`, spike rate `0.7500 Hz`, clipped mean/std
  `-74.9309/2.4654 mV`, clipped `|dV/dt|` p95/p99 `2.2833/4.2333`;
  weight `0.5`, inhibition `1.0` gives `0.8333 Hz` but clipped mean/std
  `-77.1411/2.4881 mV`. Stronger inhibition suppresses spikes but does not
  fix the voltage regime. Artifact:
  `artifacts/arbor_neuronio_smoothed_weight_calibration_20260605/calibration_summary.md`.

- IMPLEMENTATION (2026-06-05 14:55): Added an opt-in `neuronio_smoothed`
  Arbor input protocol in `src/bnn_sim/arbor_sim.py`, exposed it through
  `bnn_sim make-arbor-hetero-data --input-protocol`, and extended the pilot
  generation/calibration scripts to accept it. The mode approximates the
  Beniaguev/NeuronIO stimulus on the selected Arbor branches using coarse
  basal/apical E/I count fields, jittered hold intervals, Gaussian smoothing,
  exponential instantaneous probabilities, and branch-length weighting. It is
  explicitly an Arbor-morphology protocol approximation, not NeuronIO simulator
  equivalence.

- VERIFICATION (2026-06-05 14:55): Added `tests/test_arbor_sim.py` and verified
  that `neuronio_smoothed` inputs have the intended slow-drive structure:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_arbor_sim.py -q`
  passed (`1 passed`). Also ran `py_compile` for `src/bnn_sim/arbor_sim.py`,
  `src/bnn_sim/cli.py`, and the touched artifact scripts. A first pilot
  attempt with `.venv/bin/python` failed because that environment has
  CPU-only Arbor (`gpu: None`); reran simulator generation with
  `/home/davwis/miniconda3/envs/bnn_sim/bin/python`, which has CUDA Arbor.
  No CPU fallback was used.

- ARTIFACT (2026-06-05 14:55): Added
  `artifacts/compare_teacher_protocols.py` and generated
  `artifacts/teacher_protocol_comparison_20260605.md/json`. This confirms the
  original Arbor iid-input pilot matched event fraction/spike rate roughly but
  not NeuronIO input dynamics: event-count autocorr at 100 ms was `-0.0003`
  vs NMDA `0.1964`, and 100 ms total-event std was `35.93` vs `291.47`.
  Source-level diff records that NeuronIO used Hay L5PC/NEURON, smoothed
  basal/apical segment-length-weighted stimulus, high-res spike-peak labels,
  and historical `-55 mV` clipped voltage targets, while current Arbor uses
  Allen L5 morphology/fit, Arbor mechanisms, and threshold-crossing labels.
  The NeuronIO MOD files support release/STP machinery, but the dataset
  generator sets `Use=1`, `Dep=0`, `Fac=0`, so this should not be overstated
  as an active probabilistic-release difference for the historical dataset.

- DECISION (2026-06-05 14:55): Do not proceed to surrogate training on the
  NeuronIO-smoothed Arbor pilot yet. The input-process discrepancy is real and
  fixable, but once fixed it exposes a different Arbor neuronal regime; closing
  the remaining RMSE gap without touching the model likely requires a more
  defensible simulator/protocol calibration family (passive reversal/resting
  regime, synaptic conductance kinetics/weights, and spike-label semantics),
  not just matching mean event rate or clipping the observation.

- USER (2026-06-05 14:55): User said "good. now continue as planned"; continued
  the correctness-first simulator/protocol path before additional benchmark
  training.

- VERIFICATION (2026-06-05 14:14): Confirmed that hard soma-voltage clipping
  was explicitly part of the ELM / Branch-ELM NeuronIO setup. The local ELM
  paper PDF (`.untracked/papers/elm_neuron.pdf`, Appendix B) says NeuronIO was
  preprocessed for both training and evaluation by capping somatic membrane
  voltage at `-55 mV`, subtracting `-67.7 mV`, and scaling by `1/10` for
  training, following Beniaguev et al. (2021). The cloned upstream
  `elmneuron` repo confirms this in code: `neuronio_data_loader.py` thresholds
  `y_soma` before bias/scale for training, and `neuronio_eval_utils.py`
  thresholds `y_soma_test` before evaluation. Local code mirrors this in
  `src/bnn_sim/neuronio_raw.py::normalize_soma` and
  `src/bnn_sim/official/constants.py`. Also verified the lineage in the
  original Beniaguev et al. repo (`SelfishGene/neuron_as_deep_net`, commit
  `074c4666300a`): `fit_CNN.py` clips `y_soma` above `-55` before subtracting
  `-67.7`, and `evaluate_CNN_test.py` clips test soma voltage before
  evaluation. The ELM paper's NeuronIO results include Branch-ELM (e.g. Figure
  2 text and Table S8), and the upstream model docs describe the included
  NeuronIO pretrained models as Branch-ELM models, so the caveat directly
  applies to the Branch-ELM comparison.

- DECISION (2026-06-05 14:14): Flag the clipping result as paper-writing
  important. Prior ELM / Branch-ELM sub-`1 mV` voltage RMSE is RMSE against a
  clipped somatic-voltage target, not full action-potential peak-shape
  fidelity. Our Arbor diagnostic therefore matters: the convention is sparse
  globally (`0.2094%` of masked timesteps) but large near spikes (`12.923%`
  affected near spikes, `8.9514 mV` near-spike raw-vs-clipped RMSE, max removed
  AP amplitude `50.21 mV`). Added
  `artifacts/paper_writing_notes.md` so this caveat is available during paper
  drafting.

- DECISION (2026-06-05 13:55): Do not treat low-pass or slew-rate observation
  transforms as the default Arbor target. They can match NeuronIO-like
  `dV/dt` tails, but only by visibly distorting the target distribution:
  best transform score examples shrink clipped Arbor voltage std to
  `2.19 mV` (`clip_slew_limit0.7`) or `1.72 mV`
  (`clip_lowpass_tau10`) versus NeuronIO `3.05 mV`, and add raw-soma
  distortion RMSE `2.01-2.46 mV`. Hard clipping at `-55 mV` is different:
  it is explicitly part of the historical `convert_neuronio_pickles`
  normalization path, so it is defensible for NeuronIO comparability, but raw
  Arbor `soma_mv` must remain available and raw AP-shape fidelity must be
  reported separately.

- EXPERIMENT (2026-06-05 13:55): Ran a simulator-only clipped-dynamics
  calibration sweep over `48` Arbor configurations at `8` samples / `3000`
  steps with CUDA Arbor. Artifact:
  `artifacts/arbor_teacher_calibration_clipped_dynamics_20260605/`. Best
  balanced row was weight `2.6`, inhibition `1.2`, passive shift `0`:
  event fraction `0.010001`, spike rate `0.7917 Hz`, clipped mean/std
  `-69.4674/2.9916 mV`, clipped `|dV/dt|` p95/p99 `3.3976/5.3912`
  against NeuronIO targets `0.010616`, `0.8815 Hz`, `-67.6158/3.1697 mV`,
  and `0.4375/0.6875`. Mean-matched rows exist, but only by overfiring:
  e.g. weight `2.0`, inhibition `0.8`, passive shift `3` gives clipped mean
  `-67.5465 mV` but spike rate `4.9167 Hz`; weight `2.6`, inhibition `1.0`,
  passive shift `3` gives `-67.3839 mV` but `3.5000 Hz`. Verdict: this simple
  static weight/inhibition/passive-shift family does not close the gap without
  changing the neuronal regime.

- VERIFICATION (2026-06-05 13:55): Completed the longer Branch-ELM control on
  the same `256/64` clipped Arbor scale split for `245,760` presentations
  (`60` epochs, CUDA, `3844.3s`). Training artifact:
  `artifacts/arbor_scale256_64_branch_elm_clipped_4096x60_20260605/`.
  Best-RMSE checkpoint at epoch `59` reached validation RMSE/AUC
  `1.2921 mV` / `0.9848`. Corrected NEURO-BENCH artifact:
  `artifacts/arbor_scale256_64_branch_elm_clipped_4096x60_neuro_bench_20260605/`;
  it produced `3` SVG plots. Compared with the `81,920` presentation baseline
  (`1.3668 mV` / `0.9897`), extra training improves global RMSE but degrades
  AUC and only nudges near-spike RMSE (`5.2764 -> 5.1803 mV`) and timing F1
  (`0.2703 -> 0.2756`). Voltage-gain diagnostic
  `artifacts/arbor_scale256_64_branch_elm_clipped_4096x60_voltage_gain_20260605.json`
  again shows affine calibration is irrelevant (`1.2921 -> 1.2915 mV`).

- ARTIFACT (2026-06-05 13:55): Added
  `artifacts/analyze_observation_and_dynamics.py` and
  `artifacts/analyze_observation_transforms.py`. Outputs:
  `artifacts/observation_dynamics_gap_20260605.json`,
  `artifacts/observation_dynamics_gap_20260605.md`,
  `artifacts/observation_transform_sensitivity_20260605.json`, and
  `artifacts/observation_transform_sensitivity_20260605.md`. Key evidence:
  Arbor raw -> `-55 mV` clipping affects only `0.2094%` of masked timesteps
  globally but `12.923%` near spikes; global raw-vs-clipped distortion RMSE is
  `0.9248 mV`, near-spike distortion RMSE `8.9514 mV`, and max removed AP
  amplitude `50.21 mV`. NeuronIO vs clipped Arbor still differs dynamically:
  spike-triggered waveform RMSE `7.1007 mV`, spike rate delta `-0.2045`
  spikes / `1000` steps, mean delta `-2.4810 mV`, while the high-drive
  response kernel is close (`0.0942 mV` RMSE). This points to spike/threshold
  dynamics and observation semantics, not input-response gain.

- USER (2026-06-05 13:55): User agreed that the observation convention path
  sounds best but raised the clipping distortion concern, endorsed dynamic
  calibration, and said protocol matching only makes sense if NeuronIO has
  enough fidelity/correctness to justify tuning toward it; asked to continue.

- DECISION (2026-06-05 11:45): The next modeling priority should not be a
  stronger version of the current sparse-soma weighting. Corrected
  NEURO-BENCH diagnostics show the calibrated clipped Arbor scale split still
  fails mainly in high-voltage/near-spike/timing regimes, but voltage-gain
  diagnostics show post-hoc affine calibration improves baseline RMSE by only
  `0.0021 mV` (`1.3668 -> 1.3647 mV`). This rules out simple global
  gain/bias compression as the main cause. Next useful options are a more
  faithful temporal/spike-shape target or simulator/teacher dynamics
  refinement, not repeating the same 80k sparse-loss recipe.

- EXPERIMENT (2026-06-05 11:45): Ran a controlled Branch-ELM sparse-soma-loss
  experiment on the same `256/64` clipped Arbor scale split for `81,920`
  presentations. Training artifact:
  `artifacts/arbor_scale256_64_branch_elm_clipped_sparse_4096x20_20260605/`.
  Best-RMSE checkpoint reached validation RMSE/AUC `1.3708 mV` / `0.9899`;
  final checkpoint reached AUC `0.9903` with RMSE `1.3734 mV`. Corrected
  NEURO-BENCH artifacts:
  `artifacts/arbor_scale256_64_branch_elm_clipped_sparse_neuro_bench_20260605/`
  and
  `artifacts/arbor_scale256_64_branch_elm_clipped_sparse_final_neuro_bench_20260605/`.
  Compared with the baseline scale checkpoint (`1.3668 mV` / `0.9897`), sparse
  loss slightly improves the high-voltage bin (`2.7866 -> 2.7276 mV` best-RMSE,
  `2.6772 mV` final) and the final checkpoint crosses AUC `0.99`, but it
  worsens global RMSE and timing F1 (`0.2703 -> 0.2664` best-RMSE,
  `0.2614` final). Verdict: useful diagnostic, not the fix.

- ARTIFACT (2026-06-05 11:45): Added
  `artifacts/diagnose_voltage_gain.py` and produced voltage-gain diagnostic
  JSONs for the baseline and sparse Branch-ELM scale checkpoints:
  `artifacts/arbor_scale256_64_branch_elm_clipped_voltage_gain_20260605.json`
  and
  `artifacts/arbor_scale256_64_branch_elm_clipped_sparse_voltage_gain_20260605.json`.
  Baseline prediction/target stats over the masked val split:
  mean `-69.98/-70.05 mV`, std `2.47/2.83 mV`, correlation `0.8758`,
  affine slope `1.0009`, affine RMSE `1.3647 mV`. Sparse-best stats:
  std `2.51/2.83 mV`, correlation `0.8749`, affine RMSE `1.3692 mV`.

- IMPLEMENTATION (2026-06-05 11:45): Fixed two NEURO-BENCH reporting bugs
  before interpreting new results. First, M1b voltage extrema and M2 voltage
  bin labels now apply the canonical soma bias after inverse scaling, so
  clipped targets report peak `-55.0 mV` rather than normalized-offset values
  such as `12.7`. Second, M2 post-input-event semantics now use recent
  high-input-drive timesteps via an event-count quantile (`p95`, threshold
  `19` events/timestep on the scale val split), because "any event" across
  `1278` sparse input channels covered effectively the entire trace and made
  `far_from_spike_subthreshold` empty. Corrected baseline artifact:
  `artifacts/arbor_scale256_64_branch_elm_clipped_neuro_bench_20260605_bias_m2fix/`.

- VERIFICATION (2026-06-05 11:45): Focused benchmark tests pass after the
  reporting fixes: `.venv/bin/python -m pytest tests/test_benchmark_suite.py -q`
  -> `7 passed`; py_compile also passed for the touched benchmark modules and
  `artifacts/diagnose_voltage_gain.py`. Corrected baseline NEURO-BENCH remains
  M1 RMSE/AUC `1.3668 mV` / `0.9897`; corrected high-voltage bin RMSE is
  `2.7866 mV`, near-spike RMSE `5.2764 mV`, far-subthreshold RMSE
  `1.3321 mV`, and post-high-input-drive RMSE `1.4013 mV`.

- USER (2026-06-05 11:45): User asked to continue.

- EXPERIMENT (2026-06-05 01:15): Added
  `artifacts/run_calibrated_arbor_scaleup.sh` and ran the full calibrated
  Arbor scale-up path at `256` train / `64` val full traces. Raw data:
  `data/arbor-calibrated-nmda-proxy-scale256_64-20260605/`; clipped target
  view: `data/arbor-calibrated-nmda-proxy-scale256_64-clipped-20260605/`.
  CUDA Arbor generation completed in split order; train generation took
  `897.3s`. Scale-up clipped stats: train event fraction/spike rate/clipped
  soma mean/std `0.010003` / `0.6380 Hz` / `-70.0698 mV` / `2.8251 mV`; val
  `0.009997` / `0.7292 Hz` / `-70.0668 mV` / `2.8503 mV`. This confirms the
  calibrated simulator remains in the intended input/voltage regime when
  scaled from `64/32` to `256/64`, though spike rate is still slightly below
  the NMDA val10 target `0.8815 Hz`.

- VERIFICATION (2026-06-05 01:15): Trained Branch-ELM on the `256/64`
  clipped Arbor scale-up for `81,920` presentations. Artifacts:
  `artifacts/arbor_scale256_64_branch_elm_clipped_4096x20_20260605/` and
  `artifacts/arbor_scale256_64_branch_elm_clipped_neuro_bench_20260605/`.
  Best epoch `18`, validation RMSE/AUC `1.3668 mV` / `0.9897`; NEURO-BENCH
  on all `64` val traces: M1 RMSE `1.3668 mV`, AUC `0.9897`, near-spike RMSE
  `5.2764 mV`, threshold-zero spike-count MAE `3.0781`, prevalence-matched
  spike-count MAE `2.1562`, timing F1@2 steps `0.2703`; `3` SVG trace plots
  were produced. Interpretation: scaling improved AUC substantially and
  voltage RMSE modestly versus the `64/32` clipped pilot (`1.4211 mV` /
  `0.9765`), but it still does not reach the historical NMDA sub-1mV regime.
  Diagnostic sanity shows predicted mean is close (`-69.98` vs `-70.05 mV`)
  but predicted voltage std is compressed (`2.47` vs `2.83 mV`), so the next
  fix should target gain/timing or simulator/target dynamics, not affine mean
  calibration.

- VERIFICATION (2026-06-05 00:22): Trained and benchmarked models directly on
  the calibrated Arbor pilot to separate teacher shift from trainability.
  Raw-Arbor target Branch-ELM 20k (`artifacts/arbor_pilot_branch_elm_20k_20260604/`)
  improved validation RMSE/AUC to `2.0743 mV` / `0.9591`, and NEURO-BENCH
  near-spike RMSE remained high at `19.97 mV` because AP peaks are raw and
  unclipped. Compact Mamba 40k on raw targets
  (`artifacts/arbor_pilot_mamba_40k_20260604/`) was worse: best
  `2.6467 mV` / `0.8801`; sparse/slope-loss Mamba 80k
  (`artifacts/arbor_pilot_mamba_sparse_80k_20260604/`) was also worse:
  best `2.6960 mV` / `0.7757`. Conclusion: the calibrated raw-Arbor pilot is
  learnable, but raw AP voltage is a genuine benchmark stressor and the quick
  sparse-loss tweak did not fix it.

- ARTIFACT (2026-06-05 00:22): Added
  `artifacts/derive_clipped_arbor_targets.py` and derived a NeuronIO-compatible
  clipped-target view at `data/arbor-calibrated-nmda-proxy-clipped-20260604/`.
  This uses the exact same Arbor simulations and preserves raw `soma_mv`, but
  replaces only `targets[..., 1]` with `min(raw_soma_mv, -55 mV)` under the
  historical NeuronIO affine normalization. Clipped val stats: event fraction
  `0.009997`, spike rate `0.8333 Hz`, clipped soma mean/std
  `-70.0455/2.8673 mV`, clipped max `-55 mV`, raw max still `2.64 mV`.
  Use this view only for historical metric comparability; use the raw view for
  AP-shape fidelity.

- EXPERIMENT (2026-06-05 00:22): Trained Branch-ELM on the clipped Arbor pilot.
  The 20k run (`artifacts/arbor_pilot_clipped_branch_elm_20k_20260604/`)
  reached validation RMSE/AUC `1.6816 mV` / `0.9574`; NEURO-BENCH near-spike
  RMSE dropped to `6.28 mV`. The 80k run
  (`artifacts/arbor_pilot_clipped_branch_elm_80k_20260605/`) reached best epoch
  `19`, validation RMSE/AUC `1.4211 mV` / `0.9765`, and NEURO-BENCH
  near-spike RMSE `5.51 mV`; it produced SVG plots at
  `artifacts/arbor_pilot_clipped_branch_elm_80k_neuro_bench_20260605/`.
  Improvement from 20k to 80k is real but still not the historical NMDA
  sub-1mV / high-AUC regime, even under clipped target convention.

- EXPERIMENT (2026-06-05 00:22): Trained compact Mamba on the clipped Arbor
  pilot for 80k presentations
  (`artifacts/arbor_pilot_clipped_mamba_80k_20260604/`). It was very fast
  (`~1674 samples/s`) but underperformed Branch-ELM: best epoch `3`,
  validation RMSE/AUC `2.2773 mV` / `0.8848`; NEURO-BENCH near-spike RMSE
  `8.71 mV`. This suggests the current tiny calibrated pilot is better served
  by Branch-ELM for benchmark sanity checks; do not interpret this as a final
  Mamba architecture result.

- VERIFICATION (2026-06-05 00:22): Verification after clipped-target and
  Arbor-pilot training/evaluation changes: py_compile passed for
  `artifacts/derive_clipped_arbor_targets.py`,
  `artifacts/evaluate_nmda_checkpoints_neuro_bench.py`, and
  `artifacts/run_neuro_bench_mvp.py`; focused tests
  `tests/test_benchmark_suite.py tests/test_cli.py -q` passed (`19 passed`).

- EXPERIMENT (2026-06-04 23:34): Completed an additional refined Arbor
  calibration sweep at `artifacts/arbor_teacher_calibration_gpu_refine_20260604/`
  around the pilot candidate (`base_event_rate_hz=10`, weights `2.0-2.4`,
  inhibition `1.0-1.5`, passive shifts `0/3/6`, `8` samples, `3000` steps).
  Best objective was weight `2.0`, inhibition `1.0`, passive shift `0`:
  event fraction `0.010005`, spike rate `0.7500 Hz`, soma mean/std
  `-70.3218/3.1911 mV`. Weight `2.2` moved mean to about `-69.1 mV` but
  increased spike rate to `1.08-1.33 Hz` and std to `3.63-3.65 mV`; weight
  `2.4` moved mean near `-68.2 mV` but overfired (`2.33-2.79 Hz`) and inflated
  std (`4.12-4.35 mV`). Extra inhibition suppressed spikes and hyperpolarized.
  Conclusion: the current pilot's `~2.4 mV` hyperpolarized mean is a real
  residual mismatch under this simulator/protocol, not a simple parameter
  tweak to hide.

- VERIFICATION (2026-06-04 23:34): Evaluated local trained NMDA checkpoints on
  NEURO-BENCH M1/M1b/M2 with diagnostic SVG plots. Artifact:
  `artifacts/nmda_neuro_bench_eval_20260604/` with `results.json`,
  `report.md`, and `12` SVG trace overlays. On `32` corrected-NMDA val traces:
  Branch-ELM official 80k RMSE/AUC `0.93825 mV` / `0.97888`;
  Branch-ELM random 800k `0.70895 mV` / `0.98628`; compact Mamba 80k
  `0.92863 mV` / `0.97767`; partial compact Mamba 480k `0.85366 mV` /
  `0.98512`. Outputs have finite voltage ranges close to the teacher; spike
  logits rank events but are not calibrated at threshold zero, so benchmark
  now reports both threshold-zero and prevalence-matched spike event metrics.

- IMPLEMENTATION (2026-06-04 23:34): Fixed NEURO-BENCH spike operating-point
  semantics. `BenchmarkSuite` now preserves threshold-zero spike-count/timing
  checks as calibration diagnostics and adds prevalence-matched spike-count,
  M1b feature, and M2 timing diagnostics. Updated
  `artifacts/evaluate_nmda_checkpoints_neuro_bench.py` to surface those
  fields and generate dependency-free SVG trace plots. Verification:
  `tests/test_benchmark_suite.py tests/test_cli.py -q` passed (`19 passed`);
  relevant benchmark/evaluator scripts compile.

- EXPERIMENT (2026-06-04 23:34): Calibrated Arbor teacher toward NMDA/NeuronIO
  stats using CUDA-enabled Arbor from the conda `bnn_sim` environment. The old
  default `.venv` Arbor remains CPU-only, but conda Arbor reports `gpu:
  cuda`. The previous source-medium split was rejected as not comparable
  (`71.16 Hz`, event fraction `0.00365`, soma std `16.79 mV`). New calibration
  artifacts: `artifacts/arbor_teacher_calibration_gpu_bracket_20260604/` and
  `artifacts/arbor_teacher_calibration_gpu_epas_20260604/`. Best base-10
  candidate in the e_pas sweep: event fraction `0.010012`, spike rate
  `0.8333 Hz`, soma mean/std `-70.0674/3.1638 mV` at AMPA/NMDA weight scale
  `2`, inhibition scale `1`, passive reversal shift `+3 mV`, fixed nominal
  physiology. This matches input fraction, spike rate, and variance but remains
  about `2.45 mV` hyperpolarized relative to NMDA, so it is a defensible pilot
  teacher, not final publication data.

- ARTIFACT (2026-06-04 23:34): Added
  `artifacts/generate_calibrated_arbor_pilot.py` and generated pilot Arbor
  replacement data at `data/arbor-calibrated-nmda-proxy-20260604/` (`64`
  train, `32` val, `6000` steps, CUDA Arbor, fixed calibrated nominal
  physiology). Pilot stats: train event fraction/spike rate/soma mean/std
  `0.010001` / `0.6641 Hz` / `-70.0346 mV` / `3.1820 mV`; val
  `0.009997` / `0.8333 Hz` / `-70.0038 mV` / `3.2776 mV`. Important target
  convention: Arbor saves raw soma voltage normalized with the NeuronIO affine
  scale; unlike historical NMDA targets, AP peaks are not clipped to `-55 mV`.

- VERIFICATION (2026-06-04 23:34): Ran a cross-teacher NEURO-BENCH sanity pass
  on the calibrated Arbor pilot val split using the NMDA-trained checkpoints.
  Artifact: `artifacts/arbor_pilot_cross_teacher_neuro_bench_20260604/` with
  `12` SVG plots. RMSE rises to about `3.0 mV` and near-spike RMSE to about
  `24 mV`; AUC falls to `0.75-0.82`. This is expected because these models
  were trained on clipped NMDA/NeuronIO targets and are being evaluated on raw
  Arbor AP peaks; it is teacher/data shift evidence, not an Arbor-trained model
  result.

- IMPLEMENTATION (2026-06-04 23:34): Updated
  `artifacts/run_neuro_bench_mvp.py` so its default validation data is the
  calibrated Arbor pilot split and M4 is skipped unless
  `--allow-mixed-teacher-m4` is passed. Verified the runner on
  `data/arbor-calibrated-nmda-proxy-20260604/val` with Branch-ELM checkpoints:
  M4 reports `not_run` with warning "skipped to avoid mixing calibrated-Arbor
  M1 with old-teacher M4 pairs"; Branch-ELM official80k cross-teacher M1
  RMSE/AUC on `8` pilot traces was `3.0720 mV` / `0.7938`.

- USER (2026-06-04 23:34): User requested evaluation of trained models on the
  benchmark including Branch-ELM if local/downloadable, sanity inspection of
  outputs and plots, then replacement of the NeuronIO dataset with Arbor data
  with simulator correctness and defensibility prioritized over matching old
  metrics.

- EXPERIMENT (2026-06-04 22:59): Stopped the corrected-NMDA long compact-Mamba
  run because the immediate question was whether the smaller canonical runs
  matched, and a full `2.4M`-presentation run was not needed. Partial artifact
  directory:
  `artifacts/nmda_mamba_compact_long_b16_2p4m_20260604/`. It completed two
  epochs before interruption; best checkpoint metadata reports epoch 2,
  `480k` presentations, RMSE `0.8327195383 mV`, AUC `0.9880110666`, validation
  samples `1280`, training throughput `476.1 samples/s`. No final metrics JSON
  was written because the run was interrupted. Interpretation: the smaller
  corrected-NMDA Mamba checks already matched the expected short-run regime;
  the partial long run confirms continued improvement but is not required for
  the current benchmark sanity check.

- EXPERIMENT (2026-06-04 22:45): Started the meaningful corrected-NMDA long
  compact-Mamba curve after the 40k/80k probes showed Mamba learns the fixed
  data but does not reach `>0.99` AUC at 80k. Command uses
  `configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json`,
  `--preset mamba-official-probe`, random full-trace windows, burn-in `150`,
  batch `16`, `10 x 240000 = 2.4M` presentations, `shard-reuse-batches 64`,
  CUDA, and val batch `8`. Artifact directory:
  `artifacts/nmda_mamba_compact_long_b16_2p4m_20260604/`. First update rows
  show healthy throughput around `475 samples/s`; expected finish from observed
  training throughput plus full validations is roughly `00:25-00:45 CEST`.

- EXPERIMENT (2026-06-04 22:36): Resolved the "80k vs >0.99 AUC" question on
  the corrected NMDA dataset and tested compact Mamba. Historical
  `progress.md` shows the canonical short Mamba architecture-selection probes
  were usually `40k` presentations (`2 x 20000`, random full-trace, batch 8),
  while the canonical Branch-ELM official-style reference probe was `80k`.
  Historical `>0.99` AUC belonged to default/random-full-trace longer or
  pre-windowed/full-context evaluation curves, not to the 80k official-style
  sampler. Current corrected-NMDA runs confirm this: Branch-ELM
  random-full-trace 800k crossed `0.99` first at epoch 5 / `400k`
  presentations (`0.990018` AUC), with best epoch 9 RMSE/AUC `0.65010 mV` /
  `0.99133`; its epoch 1 / `80k` point was `0.87891 mV` / `0.98294`.
  Compact official Mamba (`branch_gain + integrative_pre + branch_tanh +
  soma_gate`, config
  `configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json`)
  on corrected NMDA reached `0.94751 mV` / `0.97148` at `40k`
  (`artifacts/nmda_mamba_compact_40k_20260604/`) and best-RMSE `0.92916 mV`
  / best-AUC `0.98071` at `80k`
  (`artifacts/nmda_mamba_compact_80k_20260604/`). Therefore Mamba is learning
  the corrected data but does not reach `>0.99` AUC at 80k; a meaningful Mamba
  comparison requires a longer compact-Mamba curve.

- EXPERIMENT (2026-06-04 15:45): Completed the NMDA/NeuronIO 80k
  official-style Branch-ELM reproduction on the rebuilt historical dataset,
  outside the failed Arbor-medium run path. Artifacts:
  `artifacts/nmda_branch_elm_reproduction_20260604/`. Data integrity before
  training: train `150` shards / `38400` unique full traces, val `5` shards /
  `1280` full traces, shapes `(6000, 1278)` inputs and `(6000, 2)` targets,
  no duplicate sample IDs, soma targets clipped at `-55 mV`. Command used
  `--preset official-fulltrace-reference`, `4` epochs, `20000` samples/epoch,
  batch `8`, CUDA, validation ignore-start `500`, and affine soma
  calibration. Result: best epoch `4`, validation RMSE `0.8225879021 mV`,
  AUC `0.9845350714`, `80000` train presentations in `1736.0s`. Per-epoch:
  epoch 1 `0.9742 mV` / `0.9686` AUC; epoch 2 `0.8958` / `0.9800`; epoch 3
  `0.8472` / `0.9801`; epoch 4 `0.8226` / `0.9845`. This confirms the
  previous ~20 mV / ~0.6 AUC numbers were caused by using the Arbor-medium
  teacher/data distribution, not by an inability of Branch-ELM to learn the
  NMDA task.

- ARTIFACT (2026-06-04 15:45): Added
  `artifacts/calibrate_arbor_teacher_against_nmda.py`, an objective-ranked
  Arbor teacher calibration sweep against the rebuilt NMDA val10 target stats
  (`0.01062` event fraction, `0.8815 Hz` spike rate, soma mean/std
  `-67.62/3.17 mV`). The script fixes all heterogeneity fields by default so
  event-rate / synaptic-weight / E-I calibration can be debugged before
  reintroducing heterogeneous cells. Smoke run
  `artifacts/arbor_teacher_calibration_smoke_20260604/` completed with three
  tiny CPU-only diagnostic conditions; low fixed weights produced `0 Hz` and
  hyperpolarized traces, confirming the diagnostic can detect the opposite
  failure direction from the current hot Arbor-medium dataset.

- VERIFICATION (2026-06-04 15:45): Small Arbor no-input diagnostic on 8 random
  heterogeneous cells produced zero spikes and a quiet resting trace
  (event fraction `0`, spike counts all `0`, soma mean/std `-83.36/1.76 mV`).
  This narrows the Arbor source-medium mismatch: the present excessive firing
  is not spontaneous cell instability by itself; it arises from the current
  input/weight/excitability calibration and target convention under synaptic
  drive.

- USER (2026-06-04 15:45): User identified the likely root cause of the bad
  model numbers: runs used Arbor instead of the NMDA dataset used by
  Branch-ELM. Required actions: confirm local NMDA data, reproduce the last
  progress metrics with that dataset, then continue the Arbor path only if the
  Arbor simulator is as correct as the previous dataset in simulation fidelity
  and neuronal dynamics; otherwise report why. Correctness is the priority.

- EXPERIMENT (2026-06-04 14:45): Root-cause confirmation for the failed
  2026-06-03/04 model numbers: the run path had switched from the historical
  NeuronIO/NMDA full-trace dataset to
  `data/arbor-allen-nmda-source-medium-20260510`. Rebuilt the historical
  10-file in-distribution NMDA validation split from local raw Kaggle assets
  into `data/full-trace-shards/nmda-val`: `5` shards, `1280` full traces,
  sample shape `(6000, 1278)` inputs and `(6000, 2)` targets. Full NMDA train
  conversion from `data/raw/nmda-train` to `data/full-trace-shards/nmda-train`
  is running; expected final shape is `38400` traces / `150` shards at shard
  size `256`. Added reproducibility script
  `artifacts/reproduce_nmda_branch_elm_metrics.sh` to rebuild the old shards
  and rerun the official-style 80k plus paper-random Branch-ELM checks.

- VERIFICATION (2026-06-04 14:45): Current Arbor source-medium data is not
  dynamics-equivalent to the historical NMDA teacher. Artifact:
  `artifacts/teacher_dataset_stats_nmda_vs_arbor_20260604.json`, generated by
  `artifacts/compare_teacher_dataset_stats.py`. Target-derived stats: NMDA
  val10 spike rate `0.8815 Hz`, event fraction `0.01062`, soma mean/std
  `-67.62/3.17 mV`, target max `-55.0 mV` from NeuronIO clipping; Arbor
  source val spike rate `71.16 Hz`, event fraction `0.00365`, soma mean/std
  `-62.38/16.79 mV`, target max `22.88 mV`. Therefore the current Arbor split
  is a much hotter, higher-variance, unclipped task, and unconditioned
  Branch-ELM/Mamba/RNN metrics on it cannot be compared to the old sub-1 mV /
  >0.99 AUC results.

- EXPERIMENT (2026-06-04 04:03): Corrected-GPU RNN comparison completed and
  also failed the expected regime. Artifacts:
  `artifacts/rnn_gpu_200k_20260604/`. Training: `200,000` random-window
  presentations, batch `16`, LR `5e-4`, grad clip `1.0`, CUDA throughput
  `89.1 samples/s`, validation RMSE `20.0070 mV`, validation AUC `0.6326`.
  NEURO-BENCH M1 RMSE `16.8215 mV`, AUC `0.6454`; dropout soma-delta RMSE
  `21.6847`, dropout spike-count-delta MAE `66.25`. Since official Mamba,
  PyTorch fallback Mamba, Branch-ELM smoke checkpoint, and RNN all sit near
  the same target-variance scale on the current source-medium split, the next
  priority is data/protocol/conditioning debugging, not another blind model
  training run.

- VERIFICATION (2026-06-04 04:03): Code verification after Mamba/RNN setup
  changes: `.venv-gpu/bin/python -m pytest tests/test_mamba_pytorch.py
  tests/test_training.py::test_cli_train_writes_checkpoint_and_metrics -q`
  passed (`5 passed`); compile checks passed for `src/bnn_sim` and relevant
  artifact scripts; `bash -n run_long_model_benchmark_batch.sh` passed;
  `.venv-mamba` official `BranchOfficialMamba` CUDA smoke passed. Full
  `.venv-gpu` suite has `3` expected environment failures because Arbor is not
  installed there; full original `.venv` suite passed (`64 passed, 1 skipped`).

- EXPERIMENT (2026-06-04 03:26): Long official-Mamba full-trace run completed
  and failed validation. Command used `.venv-mamba`, `mamba-official`,
  branchgain/integrative/pre/branchtanh/somagate config, 500 full-trace epochs,
  batch `16`, LR `1e-3`, no weight decay, grad clip `1.0`. Artifacts:
  `artifacts/official_mamba_fulltrace_gpu_500epoch_lr1e3_20260604/`. Best
  epoch `46`, validation RMSE `20.2446 mV`, validation AUC `0.6144`; final
  epoch `500`, validation RMSE `21.6453 mV`, AUC `0.5766`. NEURO-BENCH on
  the best checkpoint: M1 RMSE `17.2448 mV`, AUC `0.6360`, dropout soma-delta
  RMSE `21.5012`, dropout spike-count-delta MAE `59.875`. This confirms
  fused Mamba speed is fixed but model quality is not.

- EXPERIMENT (2026-06-04 03:26): Started corrected-GPU RNN comparison to test
  whether the remaining failure is Mamba-specific or dataset/protocol-wide:
  `.venv-mamba`, `RUN_RNN=1 RUN_MAMBA=0`, `RNN_EPOCH_SAMPLES=200000`,
  batch `16`, LR `5e-4`, grad clip `1.0`, output
  `artifacts/rnn_gpu_200k_20260604/`. Early throughput is about `87
  samples/s`; expected completion including NEURO-BENCH is around `04:05 CEST`.

- EXPERIMENT (2026-06-04 02:56): Fixed the Mamba setup path enough to run the
  actual fused/official backend on RTX 5090. Created `.venv-mamba` with Python
  `3.11`, Torch `2.7.1+cu128` (`sm_120` supported), `causal_conv1d
  1.6.2.post1`, and `mamba_ssm 2.3.2.post1`; CUDA Mamba smoke passed. A
  Torch `2.12.0+cu130` attempt imported Torch but produced a
  `causal_conv1d` ABI mismatch, and source rebuild was blocked by system
  `nvcc` CUDA `12.4` vs Torch CUDA `13.0`, so the viable official-Mamba env is
  `.venv-mamba` pinned back to Torch `2.7.1+cu128`.

- IMPLEMENTATION (2026-06-04 02:56): Added `branch-mamba-pytorch` as a
  tested fallback wrapper using the existing branch-routed/integrative/soma-gate
  `BranchOfficialMamba` shell with the differentiable PyTorch `MambaBlock`.
  Added gradient-clipping support to the trainer/CLI and exposed
  `MAMBA_GRAD_CLIP_NORM`, `RNN_GRAD_CLIP_NORM`, and `MAMBA_WEIGHT_DECAY` in
  `run_long_model_benchmark_batch.sh`. Focused verification passed:
  `.venv-gpu/bin/python -m pytest tests/test_mamba_pytorch.py
  tests/test_training.py::test_cli_train_writes_checkpoint_and_metrics -q`
  (`5 passed`), `bash -n` passed, and compile checks passed.

- EXPERIMENT (2026-06-04 02:56): The CPU fallback fixed-Mamba diagnostic
  finished training before the edited wrapper exited with an unbound shell
  variable. It saved `mamba_pytorch_fixed_8k.pt`,
  `mamba_pytorch_fixed_8k_best.pt`, and metrics under
  `artifacts/mamba_fixed_debug_8k_20260603/`. Result: `8,000` presentations
  took `11,304s` (`0.708 samples/s`), validation RMSE `19.90 mV`, AUC
  `0.633`. Per user clarification this checkpoint is usable for
  model-loading/NEURO-BENCH plumbing only, not speed or final-quality evidence.

- EXPERIMENT (2026-06-04 02:56): Official Mamba short distribution runs are
  fast but still wrong on quality. Random-window 200k with LR `3e-4` reached
  validation RMSE `20.87 mV`, AUC `0.595`, NEURO-BENCH M1 RMSE `18.12 mV`,
  AUC `0.568`. Random-window 200k with LR `1e-3`, no weight decay, grad clip
  `1.0` reached NEURO-BENCH M1 RMSE `17.82 mV`, AUC `0.599`. Full-trace
  30-epoch LR `1e-3` run reached best validation RMSE `20.76 mV`, AUC
  `0.5895`. These are failures relative to the expected sub-1 mV / >0.99 AUC
  scale.

- VERIFICATION (2026-06-04 02:56): Added
  `artifacts/debug_overfit_fixed_batch.py` to separate backend/autograd from
  distribution-learning issues. Official branch-Mamba on one fixed real batch
  at LR `1e-3`, no weight decay, grad clip `1.0` reached AUC `1.0` and RMSE
  `0.92 mV` by step `400`, `0.52 mV` by step `500`, and `0.015 mV` by step
  `1000`. Therefore the model/backend can learn real examples; the remaining
  failure is in data/protocol/generalization, not Mamba autograd or CUDA
  availability.

- EXPERIMENT (2026-06-04 02:56): Baseline/oracle scale check on
  `data/arbor-allen-nmda-source-medium-20260510` showed the failed models sit
  only slightly above global target variance: val global-mean RMSE `16.80 mV`,
  per-trace mean oracle RMSE `14.57 mV`, spike rate `71.27/1000` steps, no
  constant conditioning channels. This supports the current working hypothesis
  that the source-medium heterogeneous dataset may be under-conditioned or
  materially harder than the old healthy checkpoints referenced in top-level
  `progress.md`, which are not present in this workspace (`/workspace/runs/...`
  paths are absent).

- VERIFICATION (2026-06-04 00:57): Tightened the corrected Mamba run path before
  starting any new benchmark training. Added `MAMBA_UPDATE_LOG_INTERVAL` wiring
  to `run_long_model_benchmark_batch.sh`; `bash -n` passed. Verified
  `.venv-gpu` with `verify_training_environment.py --require-gpu`: RTX 5090
  `sm_120` is supported by Torch `2.11.0+cu128`, CUDA matmul smoke passed,
  while `mamba_ssm`/`causal_conv1d` remain unavailable. Fixed pure-PyTorch
  Mamba GPU forward benchmark at batch `8`, length `500` averaged `0.0385s`;
  focused GPU tests
  `tests/test_mamba_pytorch.py tests/test_training.py::test_cli_train_writes_checkpoint_and_metrics`
  passed (`3 passed`). This is the valid setup for the next Mamba debug run;
  the still-running CPU fallback run is retained only as a model/plumbing
  artifact per user clarification.

- USER (2026-06-04 00:34): User clarified that the CPU fallback Mamba run can
  still be used as a model/testing artifact for code-path and NEURO-BENCH
  plumbing checks. It must not be treated as acceptable setup for speed claims
  or final model-quality evidence.

- USER (2026-06-04 00:08): User corrected another execution failure: accepting
  CPU fallback and 17 mV / 0.6 AUC numbers violated the project priorities.
  Correct expected scale is sub-1 mV RMSE and >0.99 AUC quickly in training.
  Mamba should be used for debugging first because it is faster, then RNN.
  Do not proceed with NEURO-BENCH until setup is correct.

- BLOCKER (2026-06-04 00:08): The current default `.venv` is not a valid GPU
  training environment on this machine. It has Torch `2.6.0+cu124`, whose arch
  list stops at `sm_90`; RTX 5090 is `sm_120`, and a CUDA kernel smoke fails
  with "no kernel image is available for execution on the device". CPU fallback
  training is diagnostic-only and must not be accepted for benchmark runs.

- IMPLEMENTATION (2026-06-04 00:08): Fixed the pure-PyTorch Mamba fallback root
  cause. `src/bnn_sim/mamba_pytorch.py` used a custom `autograd.Function` whose
  `backward()` returned `None` for every input, so Mamba scan/conv/projection
  parameters (`A_log`, `D`, `conv1d`, `x_proj`, `dt_proj`) did not train.
  Replaced it with ordinary differentiable PyTorch scan operations and added
  `tests/test_mamba_pytorch.py` to require nonzero gradients for those trunk
  parameters. Added `mamba-pytorch` checkpoint/CLI support and a serializable
  config file `configs/mamba_pytorch_1278_d64_l2.json`.

- VERIFICATION (2026-06-04 00:08): Fixed-Mamba sanity checks passed before any
  new benchmark claim. Tiny voltage overfit on real Arbor windows reached
  `0.019 mV` by step 200. Spike-containing overfit slice reached AUC `0.998`
  by step 50 and AUC `1.0` / RMSE `0.50 mV` by step 100. This proves the
  corrected fallback can learn voltage and spike ranking; the previous
  `runs/mamba_best_trained.pt` checkpoint is from a broken non-training Mamba
  path and should not be used as model evidence.

- IMPLEMENTATION (2026-06-04 00:08): Created `.venv-gpu` with Torch
  `2.11.0+cu128`. `verify_training_environment.py --require-gpu` confirms the
  RTX 5090 as `sm_120`, Torch arch list includes `sm_120`, and CUDA matmul
  smoke passes. Added `docs/verify-arbor-simulator/artifacts/verify_training_environment.py`
  and patched `run_long_model_benchmark_batch.sh` to refuse CPU training by
  default unless `ALLOW_CPU_TRAINING=1` is explicitly set. Guard verification:
  `.venv` training is refused with exit code `2`; `.venv-gpu` resolves
  `DEVICE=cuda` and runs evaluation-only NEURO-BENCH on CUDA.

- BLOCKER (2026-06-04 00:08): Official/fused `mamba_ssm` is not yet installed.
  Build attempt in `.venv-gpu` failed because no prebuilt wheel exists for the
  local Torch `2.11.0+cu128`/Python 3.12 combo, and source build is blocked by
  the machine CUDA toolkit/compiler mismatch (`nvcc` CUDA `12.4`, GCC/G++ `15`,
  while CUDA 12.4 rejects compiler versions >=14). Current valid path is GPU
  PyTorch fallback Mamba; official Mamba requires a compatible wheel matrix or
  a compiler/toolkit fix before use.

- VERIFICATION (2026-06-03 22:55): Final checks passed after the 200k RNN
  experiment and runner/logging changes: `.venv/bin/python -m pytest -q`
  (`60 passed, 1 skipped`); compile check passed for `src/bnn_sim` and all
  benchmark artifact scripts; `bash -n` passed for
  `run_long_model_benchmark_batch.sh`.

- EXPERIMENT (2026-06-03 22:53): The 200k branch-trace RNN run completed and
  automatically evaluated on NEURO-BENCH. Training budget was `200,000`
  presentations (`12,500` batches) on CPU, total training time `6,450.2s`
  (`31.0` samples/s). Saved artifacts under
  `docs/verify-arbor-simulator/artifacts/rnn_experiment_200k_20260603/`:
  `rnn_branchtanh_fixed_200k.pt`, `rnn_branchtanh_fixed_200k_best.pt`,
  `rnn_branchtanh_fixed_200k_metrics.json`, update log JSONL, registry entry,
  and NEURO-BENCH JSON/Markdown reports. Validation improved versus the 8k
  smoke run: soma RMSE `19.9184 mV`, spike AUC `0.6303`. NEURO-BENCH comparison:
  | model | M1 RMSE mV | M1 AUC | rate soma-delta RMSE | rate spike-delta MAE | dropout soma-delta RMSE | dropout spike-delta MAE |
  |---|---:|---:|---:|---:|---:|---:|
  | `branch_elm_30_trained` | `17.9833` | `0.3888` | `21.2951` | `317.0` | `21.8621` | `66.625` |
  | `mamba_pytorch_best` | `17.5602` | `0.4970` | `23.7367` | `318.25` | `22.2460` | `66.375` |
  | `rnn_branchtanh_fixed_200k` | `17.3010` | `0.6259` | `17.6073` | `318.75` | `21.7904` | `65.625` |
  Interpretation: the RNN has now learned enough to improve M1 discrimination
  and rate-shift soma-delta fidelity, but it still fails M4 spike-delta
  preservation. Per spec, this is a real benchmark finding rather than a model
  success claim; next model work should address spike/intervention response
  rather than only optimizing trace RMSE.

- USER (2026-06-03 21:05): User corrected the execution mistake: an 8k
  presentation run is insufficient for a model-learning benchmark and should
  have been treated only as a smoke test. Continue execution with a run long
  enough for the model to learn the data, save the models, and evaluate them on
  NEURO-BENCH.

- IMPLEMENTATION (2026-06-03 21:05): Parameterized
  `docs/verify-arbor-simulator/artifacts/run_long_model_benchmark_batch.sh`
  so RNN/Mamba budgets and checkpoint prefixes are explicit via environment
  variables (`RNN_EPOCH_SAMPLES`, `RNN_EPOCHS`, `RNN_BATCH_SIZE`,
  `RNN_LEARNING_RATE`, `RNN_PREFIX`, `RNN_UPDATE_LOG_INTERVAL`, and analogous
  Mamba variables). Also patched `src/bnn_sim/train.py` to flush JSONL update
  logs after each write for future long-run monitoring.

- EXPERIMENT (2026-06-03 21:05): Started a materially larger branch-trace RNN
  training/evaluation job:
  `RUN_RNN=1 RUN_MAMBA=0 RNN_EPOCH_SAMPLES=200000
  RNN_PREFIX=rnn_branchtanh_fixed_200k RNN_UPDATE_LOG_INTERVAL=500
  OUT_DIR=docs/verify-arbor-simulator/artifacts/rnn_experiment_200k_20260603
  docs/verify-arbor-simulator/artifacts/run_long_model_benchmark_batch.sh`.
  It runs on CPU because the local Torch build still does not support RTX 5090
  `sm_120`. Expected runtime from measured throughput is roughly two hours,
  after which the same job saves checkpoints and evaluates Branch-ELM, existing
  pure-PyTorch Mamba, and the new RNN on NEURO-BENCH.

- EXPERIMENT (2026-06-03 18:51): Confirmed local Branch-ELM is our BranchELM
  surrogate implementation (`model_kind: baseline`) and reused
  `runs/branch_elm_30_trained.pt` rather than retraining it. Started and
  completed a bounded branch-trace RNN training/evaluation run with
  `RUN_RNN=1 RUN_MAMBA=0 OUT_DIR=docs/verify-arbor-simulator/artifacts/rnn_experiment_20260603
  docs/verify-arbor-simulator/artifacts/run_long_model_benchmark_batch.sh`.
  Device auto-resolved to CPU because local Torch does not support RTX 5090
  `sm_120`. Training saved
  `rnn_branchtanh_fixed_8k.pt` and `rnn_branchtanh_fixed_8k_best.pt` after
  `8,000` train presentations (`500` batches, `272.5s` total,
  validation soma RMSE `20.8280 mV`, validation spike AUC `0.5563`). The
  appended NEURO-BENCH report compared Branch-ELM, existing pure-PyTorch Mamba,
  and the new RNN. RNN smoke metrics: M1 RMSE `17.4947 mV`, M1 AUC `0.5616`,
  rate-shift soma-delta RMSE `19.7532`, rate-shift spike-delta MAE `317.0`,
  dropout soma-delta RMSE `21.5935`, dropout spike-delta MAE `66.625`. This is
  a useful benchmark-learning signal but still not a successful intervention
  model, because spike-delta errors remain essentially unchanged.

- VERIFICATION (2026-06-03 12:15): Final checks passed after adding the long
  benchmark batch runner and extra-checkpoint MVP reporting:
  `.venv/bin/python -m pytest -q` (`60 passed, 1 skipped`);
  `.venv/bin/python -m compileall -q src/bnn_sim
  docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py
  docs/verify-arbor-simulator/artifacts/run_teacher_rate_sweep.py
  docs/verify-arbor-simulator/artifacts/run_event_dropout_teacher_pair.py`
  (passed); `bash -n` passed for `run_long_model_benchmark_batch.sh` and
  `run_neuro_bench_mvp_experiments.sh`.

- EXPERIMENT (2026-06-03 12:15): Added
  `docs/verify-arbor-simulator/artifacts/run_long_model_benchmark_batch.sh` as
  the runnable long-experiment entrypoint. It can train the branch-trace RNN
  (`RUN_RNN=1`, 8k presentations) and official Mamba (`RUN_MAMBA=1`, 40k
  presentations), then evaluates all available checkpoints with the MVP
  benchmark runner using `--extra-checkpoint NAME=PATH`. The script now defaults
  `DEVICE=auto`, resolving to CPU on this workstation because the current Torch
  build does not support the visible RTX 5090 `sm_120` CUDA architecture.
  Smoke command `RUN_RNN=0 RUN_MAMBA=0
  docs/verify-arbor-simulator/artifacts/run_long_model_benchmark_batch.sh`
  completed successfully in CPU mode and wrote
  `docs/verify-arbor-simulator/artifacts/long_model_runs/neuro_bench_long_model_results.json`
  plus `.md`.

- IMPLEMENTATION (2026-06-03 12:15): Extended
  `docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py` with
  `--extra-checkpoint NAME=PATH` support using the repo checkpoint loader. This
  lets newly trained comparison models be included in the same M1/M1b/M2/M4
  report without editing the runner.

- VERIFICATION (2026-06-03 12:10): Full local test suite passed again after
  adding explicit-input Arbor re-simulation and the dropout-pair experiment:
  `.venv/bin/python -m pytest -q` (`60 passed, 1 skipped`). Compile check passed
  for `src/bnn_sim` and all benchmark artifact scripts.

- EXPERIMENT (2026-06-03 12:10): Added and ran the cleaner M4 event-dropout
  teacher pair via
  `docs/verify-arbor-simulator/artifacts/run_event_dropout_teacher_pair.py`.
  Source was the nominal teacher-rate-sweep traces (`n=8`, `3000` steps);
  intervention dropped `48,005 / 95,715` baseline events (`p=0.5`) and
  re-simulated the same heterogeneity with Arbor CPU. Artifacts:
  `docs/verify-arbor-simulator/artifacts/event_dropout_pair/event_dropout_pair.npz`,
  `event_dropout_report.json`, and `event_dropout_report.md`. Teacher response:
  event fraction `0.003067 -> 0.001529`, spike rate delta `-26.65` per
  1000 steps, soma mean delta `-5.343 mV`, direction
  `lower_event_drive/lower_spiking/hyperpolarized`, response detected. This is
  cleaner than the low→high balanced-rate shift and is the preferred MVP M4
  intervention for next model comparisons.

- EXPERIMENT (2026-06-03 12:10): Re-ran the MVP smoke comparison after adding
  the dropout pair to `run_neuro_bench_mvp.py`. Updated artifacts:
  `neuro_bench_mvp_smoke_results.json` and `.md`. Current comparison:
  | model | M1 RMSE mV | M1 AUC | rate soma-delta RMSE | rate spike-delta MAE | dropout soma-delta RMSE | dropout spike-delta MAE |
  |---|---:|---:|---:|---:|---:|---:|
  | `branch_elm_30_trained` | `17.9833` | `0.3888` | `21.2951` | `317.0` | `21.8621` | `66.625` |
  | `mamba_pytorch_best` | `17.5602` | `0.4970` | `23.7367` | `318.25` | `22.2460` | `66.375` |
  The available checkpoints still fail the intervention-delta task badly, so
  this is a benchmark validation result, not a model success result.

- IMPLEMENTATION (2026-06-03 12:10): Added
  `simulate_arbor_targets_for_inputs` to `src/bnn_sim/arbor_sim.py` so benchmark
  artifacts can re-simulate explicit input interventions with stored
  heterogeneity vectors. Updated the MVP runner to include an optional
  event-dropout pair when present.

- VERIFICATION (2026-06-03 12:07): Full local test suite passed:
  `.venv/bin/python -m pytest -q` (`60 passed, 1 skipped`). Compile check also
  passed: `.venv/bin/python -m compileall -q src/bnn_sim
  docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py
  docs/verify-arbor-simulator/artifacts/run_teacher_rate_sweep.py`. Installed
  missing local test dependencies into `.venv`: `arbor==0.11.0` CPU wheel and
  `requests>=2.32`. The local PyTorch build cannot execute CUDA kernels on the
  visible RTX 5090 (`sm_120` unsupported), so the MVP model smoke ran on CPU.

- EXPERIMENT (2026-06-03 12:07): Ran the NEURO-BENCH MVP smoke experiment via
  `docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp_experiments.sh`.
  Artifacts:
  `docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_results.json`
  and `docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_report.md`.
  Teacher M4 low→high event-rate pair (`n=4`, ignore first `500` steps): event
  fraction increased `0.001133 -> 0.008048` (`7.11x`), soma mean increased by
  `+14.98 mV`, but spike rate decreased by `-23.91` spikes / 1000 steps. This
  is a response-detected intervention but not a clean biological robustness
  claim; use it as teacher-fidelity/stress testing until the teacher protocol is
  calibrated. Model smoke comparison:
  | model | M1 RMSE mV | M1 AUC | M4 soma-delta RMSE mV | M4 spike-count-delta MAE |
  |---|---:|---:|---:|---:|
  | `branch_elm_30_trained` | `17.9833` | `0.3888` | `21.2951` | `317.0` |
  | `mamba_pytorch_best` | `17.5602` | `0.4970` | `23.7367` | `318.25` |
  Both checkpoints preserved soma-delta sign but failed spike-count-delta
  direction; current checkpoints are not useful as final model evidence.

- EXPERIMENT (2026-06-03 12:07): Ran a bounded Arbor teacher rate sweep using
  the new artifact script
  `docs/verify-arbor-simulator/artifacts/run_teacher_rate_sweep.py`
  (`samples=8`, `time_steps=3000`, CPU-only Arbor, elapsed `89.70s`). Artifacts:
  `docs/verify-arbor-simulator/artifacts/teacher_rate_sweep/rate_sweep_summary.json`
  and `.md`. Observed:
  | protocol | event fraction | spike rate Hz | soma mean mV |
  |---|---:|---:|---:|
  | `low` | `0.000719` | `32.04` | `-73.89` |
  | `nominal` | `0.003121` | `94.04` | `-62.48` |
  | `high` | `0.014364` | `68.58` | `-55.95` |
  Event fraction and soma mean are monotone increasing, but spike rate is not.
  This supports keeping M4 but blocks any simple "higher rate should increase
  spiking" publishable intervention claim under the current balanced E/I
  protocol.

- IMPLEMENTATION (2026-06-03 12:07): Implemented NEURO-BENCH M4 structured
  event-intervention metrics in
  `src/bnn_sim/benchmark/structured_intervention.py`: deterministic event
  dropout helper, paired teacher response summaries, model-vs-teacher delta
  RMSE/MAE/correlation, spike-count-delta error, direction agreement, and
  representative trace IDs. `BenchmarkSuite.evaluate_model` now accepts optional
  paired M4 intervention data while preserving the debug-only Gaussian
  perturbation alias. Added tests covering event dropout, paired M4 delta
  alignment, and suite integration.

- IMPLEMENTATION (2026-06-03 12:07): Added reusable experiment artifacts:
  `run_neuro_bench_mvp.py`, `run_neuro_bench_mvp_experiments.sh`, and
  `run_teacher_rate_sweep.py` under
  `docs/verify-arbor-simulator/artifacts/`. The MVP runner falls back to CPU
  when local CUDA is visible but unsupported by PyTorch. The shell wrapper can
  regenerate paired low/high Arbor data before running the smoke comparison.

- IMPLEMENTATION (2026-06-03 12:07): Fixed verification blockers discovered
  during broader testing. `make-arbor-hetero-data` now falls back to CPU when
  Arbor is CPU-only and the user did not explicitly request GPU. Official-style
  full-trace window IDs now preserve parseable source grouping for demo IDs
  without `:` separators. Full-trace samplers clamp requested sequence length to
  actual shard length to avoid empty windows. Training validation clamps an
  impossible default `500`-step ignore to `0` for tiny validation traces.

- USER (2026-06-03 12:07): User asked to continue the benchmarking work beyond
  implementation into validation and experiments, to script particularly long
  experiments, and to assume enough time to run most needed runs.

- VERIFICATION (2026-05-31 21:27): Focused checks passed after the benchmark
  MVP schema/M2 repair: `.venv/bin/python -m pytest tests/test_benchmark_suite.py
  tests/test_cli.py::test_cli_benchmark_writes_json_report -q` (`3 passed`);
  `.venv/bin/python -m compileall -q src/bnn_sim` (passed);
  `.venv/bin/python -m pytest tests/test_benchmark_suite.py
  tests/test_cli.py::test_cli_benchmark_writes_json_report tests/test_metrics.py
  -q` (`11 passed`). Base Python and the repo venv initially lacked `pytest`;
  installed `pytest>=8` into `.venv` with `uv pip install --python
  .venv/bin/python 'pytest>=8'`.

- IMPLEMENTATION (2026-05-31 21:27): Completed the first Milestone A/B/C slice
  of the NEURO-BENCH MVP. `BenchmarkSuite.evaluate_model` now emits the spec
  report keys `metadata`, `m1_trace_fidelity`, `m1b_feature_fidelity`,
  `m2_regimes`, `m4_intervention`, and `limitations` while preserving legacy
  aliases for existing scripts. M1 now applies the stated per-trace
  `ignore_start` mask before RMSE/AUC/Brier/log-loss/spike-count metrics. M1b
  adds dependency-light feature summaries for spike count/rate, ISI summaries,
  adaptation proxy, voltage extrema, and non-spiking traces. M4 is explicitly
  marked `debug_only`/`not_run` rather than publishable robustness evidence.

- IMPLEMENTATION (2026-05-31 21:27): Repaired M2 trace-boundary handling in
  `src/bnn_sim/benchmark/regime_stratified.py`. Slope, near-spike,
  post-spike, and post-event masks are now computed independently per trace, so
  spike/event windows cannot leak across sample boundaries. Added MVP regime
  rows: `far_from_spike_subthreshold`, `near_spike`, `post_spike_refractory`,
  and `post_input_event`, with support counts and skipped-regime `None` metrics.

- DECISION (2026-05-31 21:27): Preserved existing benchmark output aliases
  (`m1_open_loop_fidelity`, `m2_regime_stratified`,
  `m4_perturbation_stability`) to avoid breaking current runner/printing code
  while moving new work to the spec schema. Updated `run_benchmark.py` to merge
  runner metadata into the suite metadata instead of overwriting mask/model/data
  provenance, and to pass `model_name`/`dataset_path` into the suite directly.

- IMPLEMENTATION (2026-05-31 21:21): Ported this task-arc progress file to the
  execution-skill format: reverse-chronological typed entries with timestamp,
  material outcomes, verification status, blockers, decisions, and artifact
  links. Preserved prior benchmark, training, and pivot notes below.

- USER (2026-05-31 21:21): User asked to work on the benchmarking task set for
  `bnn_sim`, using `spec.md` as the task source, and to start by porting
  `progress.md` to the execution-skill layout.

- BLOCKER (2026-05-31 00:00): Benchmark slop test found a hard confound for
  practitioner-facing benchmark claims: the local simulator verification report
  shows pathological firing rates and failed input-rate monotonicity. Until this
  is resolved or explicitly scoped, benchmark results should be framed as
  teacher-fidelity diagnostics rather than biological validation.

- DECISION (2026-05-31 00:00): Revised NEURO-BENCH scope to an MVP:
  M1 cleanup, lightweight M1b electrophysiological feature fidelity, repaired
  M2 trace-aware regimes, one structured M4 intervention, and a two-model smoke
  comparison. M3 morphology transfer, M5 closed-loop stability, and M6 population
  statistics are deferred unless needed for the immediate model decision.

- DECISION (2026-05-31 00:00): Corrected M4 framing. Gaussian input noise and
  Lipschitz-like regularity are not publishable robustness evidence for spiking
  neuron surrogates. Gaussian perturbation may remain only as a numerical sanity
  check; paper-facing robustness should use simulator-plausible event, branch,
  morphology, and heterogeneity interventions with regime-specific
  interpretation.

- ARTIFACT (2026-05-31 00:00): Benchmark slop-test report saved at
  `docs/verify-arbor-simulator/artifacts/benchmark_slop_test_20260531.md`.
  Verdict: M1 is necessary but weak without electrophysiological features; M2 is
  useful only with trace-aware biophysical regimes; M3 is valuable but confounded
  unless split into same-cell-type morphology transfer and cross-cell-type
  transfer; M4 requires controlled teacher intervention-response checks before
  surrogate comparison.

- IMPLEMENTATION (2026-05-31 00:00): Updated `spec.md` to add source-backed M1b
  electrophysiological feature fidelity, revise M2/M3/M4, add anti-slop
  validation gates, and require critical evaluation before BSSM, noise
  calibration, baseline, and timeline work.

- PIVOT (2026-05-30 00:00): Archived the simulator-verification spec as
  `spec_archive.md` and replaced the active task spec with `spec.md`,
  "Publishable Mamba Surrogate Paper". Contribution order is now
  NEURO-BENCH, then BSSM, then noise-calibrated surrogates.

- MILESTONE (2026-05-30 00:00): Initial NEURO-BENCH implementation phase was
  marked complete before the later scope correction. Implemented suite
  integration in `src/bnn_sim/benchmark/suite.py`, combining the available
  benchmark modules into a single evaluation call.

- IMPLEMENTATION (2026-05-30 00:00): Implemented the initial M2
  regime-stratified error module at `src/bnn_sim/benchmark/regime_stratified.py`.
  Metrics included voltage regime, slope regime, spike proximity, post-event
  periods, and spike timing precision. Current spec now says this needs MVP
  repair around trace boundaries and four robust regimes before being treated as
  complete.

- VERIFICATION (2026-05-30 00:00): Tested the initial M2 implementation on 32
  real validation samples with an untrained Branch-ELM. Results: RMSE
  `16.86 mV`, AUC `0.496`, near-spike RMSE `26.48 mV`, and far-from-spike RMSE
  `8.75 mV`. The near-spike/far-from-spike gap is useful diagnostic evidence,
  but the untrained model and current teacher confounds limit interpretation.

- IMPLEMENTATION (2026-05-30 00:00): Implemented initial M3 morphology-transfer
  utilities in `morphology_transfer.py`, `swc_utils.py`, and
  `branch_adapter.py`. Downloaded two L2/3 pyramidal morphologies from Allen
  Cell Types.

- DECISION (2026-05-30 00:00): Initial M3 work found L2/3 morphologies have
  `24-40` branches versus the current `45`-branch models. Implemented a
  padding-based `BranchAdapter`, but the active spec now defers M3 unless the
  immediate model decision requires it.

- IMPLEMENTATION (2026-05-30 00:00): Implemented the initial M4 perturbation
  stability module at `src/bnn_sim/benchmark/perturbation_stability.py`.
  Validation on an untrained model produced stability coefficients `0.033`
  (soma) and `0.019` (spike).

- DECISION (2026-05-30 00:00): The initial M4 perturbation stability module is
  superseded for paper claims. Treat it as debug-only numerical smoke testing
  until a structured intervention probe replaces it.

- EXPERIMENT (2026-05-30 00:00): Branch-ELM-30 baseline training was reported in
  progress at epoch `3/30`, with epoch-1 RMSE `16.79 mV`, AUC `0.472`, GPU
  utilization around `99%`, and roughly `52s` per epoch. This status needs
  re-verification before any comparison is reported as current.

- IMPLEMENTATION (2026-05-30 00:00): BSSM training scaffold was reported ready
  at `src/bnn_sim/train_bssm.py`. Architecture: 6-state biophysical SSM with
  `6,161` parameters. It was waiting for Branch-ELM baseline completion, but the
  active spec now requires a critical evaluation gate before BSSM work proceeds.

- DECISION (2026-05-30 00:00): Previous next actions were to wait for
  Branch-ELM training, train BSSM, evaluate both on the full suite, and compare.
  Superseded by the active benchmark MVP guardrails: complete Milestone A/B/C/D
  first, avoid treating Gaussian perturbation as M4, and re-verify any stale
  training runs before using their metrics.
