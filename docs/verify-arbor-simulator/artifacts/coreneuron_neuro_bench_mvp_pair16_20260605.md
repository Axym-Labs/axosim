# CoreNEURON/Hay NEURO-BENCH MVP With 16-Pair M4

Date: 2026-06-06

## Scope

This repeats the current-teacher MVP benchmark comparison using the expanded
`16`-trace CoreNEURON/Hay event-dropout pair. M1, lightweight M1b, repaired M2,
and M4 are evaluated with calibrated spike thresholds learned on the `32`-trace
training split and frozen for validation/M4.

Validation split:
`data/hay-coreneuron-source-val16-exc0925-20260605/val`.

Calibration split:
`data/hay-coreneuron-source-train32-exc0925-20260605/train`.

M4 pair:
`docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair16_20260605/event_dropout_pair.npz`.

## Teacher M4 Event Dropout

| metric | value |
| --- | ---: |
| baseline event fraction | `0.010626` |
| intervention event fraction | `0.006907` |
| event fraction ratio | `0.6500` |
| delta spike rate / 1000 steps | `-0.8409` |
| delta soma mean mV | `-2.4102` |
| direction | `lower_event_drive/lower_spiking/hyperpolarized` |
| response detected | `true` |

## Model Comparison

| model | RMSE mV | AUC | threshold | count MAE calibrated | timing F1 calibrated | near-spike RMSE | M4 soma-delta RMSE | M4 spike-delta MAE | M4 direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `soma075_calibrated_80k` | `0.8507` | `0.9666` | `-2.6941` | `3.8125` | `0.1272` | `4.6228` | `0.7695` (`0.6282`, `0.8482`) | `3.6875` (`2.0602`, `5.3148`) | `True/True` |
| `mamba_exc0925_80k` | `0.8763` | `0.9632` | `-2.8015` | `2.0625` | `0.0972` | `4.6860` | `0.7935` (`0.6551`, `0.8726`) | `2.0000` (`1.1801`, `2.8199`) | `True/True` |
| `rnn_exc0925_80k` | `0.7987` | `0.9723` | `-1.6836` | `4.9375` | `0.0762` | `3.9798` | `0.7522` (`0.5905`, `0.8345`) | `4.8125` (`2.1876`, `7.4374`) | `True/True` |

## Interpretation

- All three models preserve the qualitative M4 event-dropout direction for
  soma mean and spike count on all `16` validation traces.
- RNN remains strongest on ordinary trace RMSE/AUC/near-spike RMSE and has the
  best M4 soma-delta RMSE.
- `soma075_calibrated_80k` remains the current Mamba voltage/RMSE path.
- Prior `mamba_exc0925_80k` remains best on calibrated spike-count MAE and M4
  spike-delta MAE in this expanded M4 smoke.
- Parenthesized M4 values are approximate per-trace `95%` intervals.
- This is materially stronger than the initial `4`-trace M4 check, but still a
  single intervention type. Further robustness claims need additional
  interventions or seeds.

## Artifacts

- Ignored detailed report:
  `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_pair16_20260605/report.md`.
- Ignored JSON:
  `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_pair16_20260605/results.json`.
