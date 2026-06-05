# CoreNEURON/Hay NEURO-BENCH MVP

Date: 2026-06-05

## Scope

This is the current-teacher MVP benchmark smoke comparison requested by the
spec: M1, lightweight M1b, repaired M2, and one M4 event-dropout intervention.
It uses calibrated spike thresholds learned on the training split and frozen
for validation/M4.

Validation split:
`data/hay-coreneuron-source-val16-exc0925-20260605/val`.

Calibration split:
`data/hay-coreneuron-source-train32-exc0925-20260605/train`.

M4 pair:
`docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605/event_dropout_pair.npz`.

## Teacher M4 Event Dropout

| metric | value |
| --- | ---: |
| baseline event fraction | `0.010676` |
| intervention event fraction | `0.006932` |
| event fraction ratio | `0.6493` |
| delta spike rate / 1000 steps | `-0.6818` |
| delta soma mean mV | `-2.3395` |
| direction | `lower_event_drive/lower_spiking/hyperpolarized` |
| response detected | `true` |

## Model Smoke Comparison

| model | RMSE mV | AUC | threshold | count MAE calibrated | timing F1 calibrated | near-spike RMSE | M4 soma-delta RMSE | M4 spike-delta MAE | M4 direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `soma075_calibrated_80k` | `0.8507` | `0.9666` | `-2.6941` | `3.8125` | `0.1272` | `4.6228` | `0.6803` | `3.5000` | `True/True` |
| `mamba_exc0925_80k` | `0.8763` | `0.9632` | `-2.8015` | `2.0625` | `0.0972` | `4.6860` | `0.6971` | `1.5000` | `True/True` |
| `rnn_exc0925_80k` | `0.7987` | `0.9723` | `-1.6836` | `4.9375` | `0.0762` | `3.9798` | `0.6495` | `5.2500` | `True/True` |

## Interpretation

- RNN remains the strongest trace model on RMSE/AUC/near-spike RMSE.
- `soma075_calibrated_80k` remains the current Mamba voltage/RMSE path.
- The prior Mamba baseline has the best calibrated spike-count MAE and best
  M4 spike-delta MAE in this small smoke comparison.
- All three models preserve the qualitative M4 event-dropout direction for
  soma mean and spike count.
- The M4 result is useful but small: only `4` paired traces. It should be
  expanded before paper-level robustness claims.

## Artifacts

- Ignored detailed report:
  `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_20260605/report.md`.
- Ignored JSON:
  `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_20260605/results.json`.
