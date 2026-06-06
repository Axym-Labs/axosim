# Trace-Shape Metric Validation

Date: 2026-06-06

Purpose: add and validate a trace metric that is more informative than global
pointwise RMSE, even when RMSE is already state-conditioned.

## Metric

Implemented `trace_shape_fidelity` in `src/bnn_sim/benchmark/trace_shape.py`
and wired it into `BenchmarkSuite`.

The metric reports:

- pointwise voltage and dV/dt RMSE/MAE;
- symmetric lag-tolerant voltage RMSE over a local +/- 2 step window;
- multi-scale low-pass voltage RMSE at 5 and 20 steps;
- voltage-distribution quantile distance;
- spike-triggered waveform RMSE after nearest-spike alignment;
- spike waveform coverage, so the waveform component cannot look good by
  comparing only a tiny subset of aligned spikes.

The headline `trace_shape_distance` is a normalized component average. It is
not meant to replace all M1/M2/M4 metrics; it is a compact trace-shape summary
that separates slow voltage, local timing tolerance, dV/dt, distribution, and
AP-neighborhood failures.

## Real 16-Trace CoreNEURON/Hay Validation

Run:

`PYTHONPATH=src .venv-mamba/bin/python docs/verify-arbor-simulator/artifacts/run_coreneuron_neuro_bench_mvp.py ...`

Output directory:
`artifacts/coreneuron_neuro_bench_trace_shape_pair16_20260606/` (ignored,
regenerable).

| model | RMSE mV | lag-tolerant RMSE mV | trace-shape distance | dV/dt comp. | low-pass comp. | voltage-quantile comp. | spike waveform comp. | waveform coverage miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `soma075_calibrated_80k` | 0.8507 | 0.6508 | 0.2408 | 0.1851 | 0.0706 | 0.0163 | 0.2049 | 0.9079 |
| `rnn_exc0925_80k` | 0.7987 | 0.6159 | 0.2418 | 0.1443 | 0.0663 | 0.0184 | 0.2178 | 0.9474 |
| `mamba_exc0925_80k` | 0.8763 | 0.6656 | 0.2473 | 0.1994 | 0.0724 | 0.0202 | 0.2094 | 0.9211 |

Spike-triggered waveform support:

| model | target spikes | evaluable target windows | matched windows | match fraction | waveform RMSE mV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soma075_calibrated_80k` | 77 | 76 | 7 | 0.0921 | 2.2307 |
| `rnn_exc0925_80k` | 77 | 76 | 4 | 0.0526 | 2.3707 |
| `mamba_exc0925_80k` | 77 | 76 | 6 | 0.0789 | 2.2789 |

## Interpretation

Lag-tolerant RMSE is substantially lower than pointwise RMSE for all models,
which means part of the voltage error is local timing/fast-event alignment.
But waveform coverage is poor for all three models, so the metric does not
hide the spike-alignment problem. The ranking differs slightly from pure RMSE:
`soma075_calibrated_80k` and `rnn_exc0925_80k` are nearly tied on trace-shape
distance, with `soma075` helped by better spike-window coverage and voltage
distribution, while RNN remains best on pointwise voltage and dV/dt.

This complements the existing dynamic-distance result: `mamba_exc0925_80k`
still wins calibrated spike-count/dynamic distance, while RNN/soma075 are
stronger trace-shape paths.

## Verification

- Synthetic regression: one-step shifted AP-like waveform has lower
  lag-tolerant distance than pointwise RMSE and zero aligned waveform RMSE.
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed: `28` tests.
- py_compile passed for `trace_shape.py`, `suite.py`, benchmark exports, and
  the CoreNEURON/Hay benchmark runner.

