# Publishable Schema Comparator Validation

Date: 2026-06-06

Purpose: verify that `BenchmarkSuite.compare_models()` can summarize the
current NEURO-BENCH report schema on real CoreNEURON/Hay model reports, not
only synthetic tests.

Input reports:

- `artifacts/coreneuron_neuro_bench_state_metrics_pair16_20260606/results.json`
- `artifacts/coreneuron_neuro_bench_m4_event_dropout_16_20260606/results.json`

The two reports agree on the shared event-dropout comparison.

| model | RMSE mV | state-weighted trace distance mV | dynamic distance | AUC | calibrated count MAE | spiking-state RMSE mV | M4 soma-delta RMSE | M4 spike-delta MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rnn_exc0925_80k` | 0.7987 | 1.8707 | 0.6561 | 0.9723 | 4.9375 | 4.6232 | 0.7522 | 4.8125 |
| `soma075_calibrated_80k` | 0.8507 | 2.0343 | 0.5993 | 0.9666 | 3.8125 | 5.3794 | 0.7695 | 3.6875 |
| `mamba_exc0925_80k` | 0.8763 | 2.0808 | 0.5249 | 0.9632 | 2.0625 | 5.4295 | 0.7935 | 2.0000 |

## Interpretation

- `rnn_exc0925_80k` remains the best voltage/state-weighted trace model and
  has the best M4 voltage-delta error on event dropout.
- `mamba_exc0925_80k` is worse on voltage RMSE but best on the composite
  dynamic distance, calibrated spike-count MAE, and M4 spike-delta MAE.
- `soma075_calibrated_80k` remains the current Mamba voltage/RMSE compromise,
  with intermediate dynamic and spike-delta behavior.

The comparator is therefore doing something materially more useful than
ranking by global RMSE alone: it exposes the voltage-vs-spike/dynamics tradeoff
that would be hidden by a single scalar.

## Verification

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed after the comparator change: `27` tests.
- Real-report smoke used `BenchmarkSuite().compare_models(...)` on the two
  result files listed above and produced consistent model summaries.

