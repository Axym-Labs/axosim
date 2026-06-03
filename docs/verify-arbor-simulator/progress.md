# Progress Log — Publishable Mamba Paper / NEURO-BENCH

Reverse-chronological execution log. Historical entries from the pre-port file
use `00:00` when no exact time was recorded.

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
