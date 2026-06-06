# M1/M2 Audit Closeout

Date: 2026-06-06 13:46 CEST

## Change

M1/M2 are now closer to paper-facing rather than only aggregate benchmark
scores.

- M1 now records per-trace voltage RMSE/MAE/max error and spike-count errors
  at zero, prevalence-matched, and calibrated thresholds.
- M1 exposes worst trace IDs for voltage and spike-count failures.
- M2 now exposes representative trace IDs for each MVP regime:
  far-from-spike/subthreshold, near-spike, post-spike/refractory, and
  post-input-event.
- The CoreNEURON/Hay benchmark markdown report now includes an audit section
  with these trace IDs.

## Real Validation

Regenerated ignored report directory:
`docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_audit_m1m2_pair16_20260606/`

Examples surfaced in the compact summary:

| model | worst soma RMSE examples | near-spike examples | post-input examples |
| --- | --- | --- | --- |
| `mamba_exc0925_80k` | `000008`, `000004`, `000005` | `000012`, `000002`, `000005` | `000008`, `000004`, `000011` |
| `rnn_exc0925_80k` | `000008`, `000012`, `000004` | `000012`, `000005`, `000002` | `000008`, `000004`, `000003` |
| `soma075_calibrated_80k` | `000008`, `000005`, `000004` | `000012`, `000002`, `000005` | `000008`, `000004`, `000011` |

Full trace IDs are in the generated JSON/Markdown report. The repeated
appearance of traces such as `000008`, `000004`, and `000012` gives concrete
targets for visual inspection instead of leaving M1/M2 as aggregate-only
scores.

## Verification

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed: `29` tests.
- `py_compile` passed for `suite.py`, `regime_stratified.py`, and the
  CoreNEURON/Hay benchmark runner.
