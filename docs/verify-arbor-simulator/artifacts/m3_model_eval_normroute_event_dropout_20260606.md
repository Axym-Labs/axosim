# M3 Held-Out SWC Model Smoke

Date: 2026-06-06 13:24 CEST

## Setup

- Pair:
  `docs/verify-arbor-simulator/artifacts/coreneuron_m3_swc_normroute_event_dropout_simbase8_20260606/intervention_pair.npz`
- Teacher baseline and intervention are both simulated under held-out Allen
  `specimen_487664663.swc`.
- Routing: normalized basal/apical arclength source-site routing.
- Models: existing source-trained `80k` Mamba/RNN checkpoints.
- Spike thresholds: learned on the original source training split and frozen
  before held-out evaluation.
- Raw output directory:
  `docs/verify-arbor-simulator/artifacts/m3_model_eval_normroute_event_dropout_simbase8_20260606/`
  (ignored/regenerable).

## Results

| model | RMSE mV | trace-shape | state trace-shape | spike-train dist | AUC | count MAE | timing F1 | M3 soma-delta RMSE | M3 spike-delta MAE | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `soma075_calibrated_80k` | `2.7701` | `0.4155` | `0.4269` | `3.4772` | `0.9219` | `8.0000` | `0.0377` | `2.4299` | `6.6250` | `true/true` |
| `mamba_exc0925_80k` | `2.7900` | `0.4319` | `0.4302` | `3.2202` | `0.9171` | `7.6250` | `0.0614` | `2.4328` | `4.7500` | `true/true` |
| `rnn_exc0925_80k` | `2.8524` | `0.4423` | `0.4277` | `4.0392` | `0.9020` | `9.1250` | `0.0443` | `2.4817` | `7.5000` | `true/true` |

## Interpretation

This is a real morphology-transfer stress signal, not just another source-cell
M4 check. RMSE degrades substantially compared with the source validation
split (`~0.8-0.9 mV` source vs `~2.8 mV` held-out SWC), so the current
source-trained models do not solve morphology transfer. Still, all three
models preserve qualitative event-dropout direction for soma mean and spike
count on this small pair.

Current best readings:

- `soma075_calibrated_80k` is best on held-out baseline voltage and
  trace-shape.
- `mamba_exc0925_80k` is best on held-out spike-train distance, calibrated
  count MAE, and event-dropout spike-delta MAE.
- RNN is not the clear winner under this held-out morphology despite being the
  strongest source-voltage model.

Limitations: `8` traces, one held-out dendrite-only morphology, one event
dropout intervention for model comparison, and no morphology-conditioned model
inputs.
