# Pretrained Literature Model Benchmark Smoke — 2026-06-17

## Scope

Evaluated the downloaded/cached literature pretrained models through the
dataframe-first `benchmark-v1` bundle with plotting/Markdown summaries disabled.
This is a stored smoke benchmark on the existing all-five-morphology v1 pilot
dataset, not a final paper-scale evaluation.

- Dataset:
  `docs/verify-arbor-simulator/artifacts/v1_swc_cpu_core_batched_smoke_20260607`
- Validation cap: `20` traces
- Intervention cap: `2` paired examples per morphology
- Suite: `paper-summary`
- Storage mode: JSONL rows + per-model `manifest.json` + combined
  `summary.json`/`summary.csv`; no Markdown summaries or plots
- TCN adapter: original receptive-field-derived overlapping windows, discard
  leading overlap on non-initial windows, zero-pad only the final short window,
  trim padded timesteps, no clipping, and no test-target mean/std alignment.
- Output root:
  `docs/verify-arbor-simulator/artifacts/pretrained_benchmark_eval_20260617`

The larger in-progress v1 GPU dataset was not used for this sweep because its
current compressed full-trace shards are large (`128 x 6000 x 1278`), so even a
small `max_val_samples` request decompresses a full large shard. That is a
benchmark I/O issue to fix with rechunking/streaming before full-dataset model
evaluation.

## Code Path Changes

- `benchmark-v1` now supports storage-only execution via `--no-summaries`.
- `run_v1_benchmark_bundle(..., write_summaries=False)` writes canonical JSONL
  diagnostic rows and manifests without Markdown summaries.
- Sharded intervention pairs are now grouped by `(intervention, morphology)`
  before evaluation. This fixes two issues in the earlier path:
  `max_intervention_samples` no longer multiplies by the number of shards, and
  repeated morphology filenames are no longer overwritten shard-by-shard.
- Package imports are lazier so TensorFlow-only TCN evaluation can import the
  benchmark bundle without requiring PyTorch.
- Added
  `docs/verify-arbor-simulator/artifacts/evaluate_pretrained_literature_benchmarks.py`
  for split-env reproducible evaluation:
  - PyTorch official ELM/Branch-ELM in `.venv`
  - Keras TCN `.h5` models in `.venv-tcn`
- Implemented the fair TCN window path from the original source's acceptable
  mechanics: overlap is computed from the architecture metadata as
  `min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)`, where
  `time_window_T = sum(filter_size - 1) + 1`. The adapter intentionally omits
  the original test-target soma mean/std alignment and does not add clipping.

## Headline Stored Metrics

| model | family | rows | RMSE mV | state-weighted dist mV | dynamic dist | lag RMSE mV | AUC | Brier | spike-count MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `branch_elm_m100` | official | 177 | 3.2185 | 4.9719 | 0.7745 | 2.8028 | 0.8648 | 0.0027 | 1.35 |
| `branch_elm_m30` | official | 177 | 3.2235 | 4.9712 | 0.7744 | 2.8055 | 0.8851 | 0.0027 | 1.35 |
| `branch_elm_best` | official | 177 | 3.2371 | 4.9897 | 0.7745 | 2.8216 | 0.8628 | 0.0027 | 1.35 |
| `branch_elm_m1` | official | 177 | 3.4885 | 5.1018 | 0.7934 | 3.0511 | 0.8885 | 0.0027 | 1.35 |
| `tcn_dwt8_224_217` | Keras TCN | 177 | 68.9019 | 74.4432 | 94.4948 | 65.6238 | 0.7775 | 0.2565 | 498.65 |
| `tcn_dwt7_128_153` | Keras TCN | 177 | 204.3985 | 170.6360 | 96.9738 | 197.2438 | 0.5284 | 0.2506 | 456.00 |
| `tcn_dwt9_256_241` | Keras TCN | 177 | 1293.7712 | 1491.3781 | 119.1607 | 1280.7919 | 0.4658 | 0.2500 | 35.45 |

## Sanity Interpretation

The upstream Branch-ELM/ELM checkpoints run through the benchmark and produce
finite, plausible v1-pilot metrics. These numbers are close to the earlier
Branch-ELM v1 smoke scale and should be treated as compatibility/domain-shift
smoke metrics, not final performance claims.

The TCN models load and run, but their somatic outputs remain not sane under
the fair v1 benchmark adapter. The adapter now implements the original
overlapping-window mechanics, but deliberately does not apply the original
test-target soma distribution alignment. Direct range checks from the prior
non-overlap smoke already showed the underlying scale issue: finite spike
probabilities but somatic predictions far outside the normalized target range.
The updated overlap pass preserves that interpretation; it improves the
strongest TCN AUC/RMSE slightly but does not make the TCN scores paper-facing.

| model | spike min/mean/max | soma min/mean/max | target soma min/mean/max |
| --- | --- | --- | --- |
| `tcn_dwt7_128_153` | `0.0000 / 0.0317 / 0.9997` | `-43.30 / 2.50 / 46.98` | `-0.90 / 0.11 / 1.27` |
| `tcn_dwt8_224_217` | `0.0000 / 0.0265 / 0.6016` | `-21.88 / -1.31 / 15.43` | `-0.90 / 0.11 / 1.27` |
| `tcn_dwt9_256_241` | `0.0000 / 0.0157 / 1.0000` | `-308.02 / -44.07 / 246.76` | `-0.90 / 0.11 / 1.27` |

So the stored TCN rows are valuable as adapter/domain-compatibility evidence,
but not as fair final TCN benchmark scores. The next defensible TCN step is to
recover the exact NeuronIO TCN target scaling/window preprocessing from the
original code before making paper-facing comparisons.

## Verification

- `PYTHONPATH=src .venv/bin/python -m py_compile ...` passed for touched
  modules and the pretrained evaluator script.
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_bundle.py
  tests/test_benchmark_runner.py tests/test_cli.py -q` passed:
  `35 passed in 1.74s`.
- TensorFlow-only import check passed:
  `.venv-tcn` can import `bnn_sim.benchmark.bundle` without PyTorch.
- Storage-only check: `find ... -name '*.md' | wc -l` returned `0`.
