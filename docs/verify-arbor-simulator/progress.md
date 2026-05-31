# Progress Log — Publishable Mamba Paper / NEURO-BENCH

Reverse-chronological execution log. Historical entries from the pre-port file
use `00:00` when no exact time was recorded.

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
