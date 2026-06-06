# Trace-Shape Metric v2 Validation

Date: 2026-06-06 12:56 CEST

## Change

Extended the benchmark trace metric beyond global RMSE with two additions:

- `state_conditioned_trace_shape`: equal-weight behavioral-state trace-shape
  rows using the shared neuron-state masks. Each state reports pointwise RMSE,
  lag-tolerant RMSE, low-pass RMSE, dV/dt MAE, support, and spike counts.
- `spike_train_distance`: a dependency-light van-Rossum-like spike-train
  distance over exponential time constants `2`, `10`, and `50` steps.

The spike-train distance is reported as a sibling metric rather than folded
into the headline `trace_shape_distance`. In the real validation run its
natural scale was much larger than the voltage-shape components, so including
it in the headline would silently turn the metric into a spike-count/timing
score. This separation is more defensible for paper use.

## Real 16-Trace CoreNEURON/Hay Validation

Command output directory:
`docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_trace_shape_v3_pair16_20260606/`

| model | RMSE mV | trace-shape | state trace-shape | spike-train distance | lag-tolerant RMSE mV | calibrated count MAE | M4 spike-delta MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rnn_exc0925_80k` | `0.7987` | `0.2427` | `0.2479` | `3.9906` | `0.6159` | `4.9375` | `4.8125` |
| `soma075_calibrated_80k` | `0.8507` | `0.2447` | `0.2685` | `3.2872` | `0.6508` | `3.8125` | `3.6875` |
| `mamba_exc0925_80k` | `0.8763` | `0.2518` | `0.2793` | `2.8244` | `0.6656` | `2.0625` | `2.0000` |

Interpretation:

- RNN remains best for voltage-level and state-conditioned trace-shape
  fidelity.
- Prior `mamba_exc0925_80k` remains best for spike-train distance,
  calibrated spike-count MAE, and M4 spike-count delta.
- `soma075_calibrated_80k` remains the current Mamba voltage compromise.
- The metric now makes the tradeoff explicit: lower RMSE does not imply better
  spike-train behavior.

## Verification

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed: `29` tests.
- `py_compile` passed for touched benchmark modules and runners.
- Synthetic tests cover timing-shifted spike waveforms and verify that a nearby
  predicted spike receives a lower spike-train distance than a missed spike.
