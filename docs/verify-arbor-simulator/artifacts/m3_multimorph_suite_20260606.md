# M3 Multi-Morphology Suite

Date: 2026-06-06 14:13 CEST

## Setup

- Held-out Allen L5 spiny candidates: `487664663`, `515249852`, `479770916`.
- All are dendrite-only reconstructions; this is a dendritic-routing transfer
  diagnostic, not a full-cell generalization claim.
- Routing: normalized basal/apical arclength source-site routing.
- Baseline policy: both baseline and intervention targets are simulated under
  the held-out morphology/template.
- Teacher interventions: event dropout and inhibitory dropout, `8` traces per
  morphology.
- Model evaluation: source-trained `80k` checkpoints on event-dropout pairs.

## Teacher Sanity

| specimen | intervention | traces | segments | delta soma mV | delta spike rate / 1000 | spike-count delta | spike-count interval | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `479770916` | `event_dropout` | `8` | `302` | `-1.0742` | `-3.6136` | `-19.8750` | `[-22.8018, -16.9482]` | lower event drive / lower spiking / hyperpolarized |
| `479770916` | `inh_dropout` | `8` | `302` | `+1.1778` | `+6.3182` | `+34.7500` | `[31.2802, 38.2198]` | lower event drive / higher spiking / depolarized |
| `487664663` | `event_dropout` | `8` | `265` | `-1.8843` | `-1.2500` | `-6.8750` | `[-9.1109, -4.6391]` | lower event drive / lower spiking / hyperpolarized |
| `487664663` | `inh_dropout` | `8` | `265` | `+1.5334` | `+4.0000` | `+22.0000` | `[18.5056, 25.4944]` | lower event drive / higher spiking / depolarized |
| `515249852` | `event_dropout` | `8` | `294` | `-1.8660` | `-1.1818` | `-6.5000` | `[-8.9005, -4.0995]` | lower event drive / lower spiking / hyperpolarized |
| `515249852` | `inh_dropout` | `8` | `294` | `+2.1452` | `+2.9091` | `+16.0000` | `[14.1854, 17.8146]` | lower event drive / higher spiking / depolarized |

Interpretation: normalized routing now passes the basic M3 teacher sanity
checks across three held-out morphologies. Event dropout consistently
hyperpolarizes and reduces spiking; inhibitory dropout consistently depolarizes
and increases spiking.

## Model Transfer

| model | mean RMSE mV | mean trace-shape | mean spike-train dist | mean count MAE | mean M3 soma-delta RMSE | mean M3 spike-delta MAE | all directions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `soma075_calibrated_80k` | `3.0228` | `0.4141` | `3.4997` | `17.2917` | `2.7011` | `10.0833` | `true` |
| `mamba_exc0925_80k` | `3.0421` | `0.4194` | `3.2809` | `16.7500` | `2.7026` | `8.9583` | `true` |
| `rnn_exc0925_80k` | `3.0994` | `0.4334` | `4.0031` | `17.6667` | `2.7607` | `10.9583` | `true` |

Interpretation: source-trained models preserve qualitative event-dropout
direction across all three held-out morphologies, but voltage fidelity degrades
from source-validation `~0.8-0.9 mV` to M3 `~3.0 mV`. M3 therefore exposes a
real morphology-transfer failure mode. `soma075_calibrated_80k` is the best
voltage/trace-shape transfer model; `mamba_exc0925_80k` is better on spike
train distance and spike-delta MAE.

## Artifacts

- Teacher suite script:
  `docs/verify-arbor-simulator/artifacts/run_m3_normalized_routing_suite.sh`
- Model suite evaluator:
  `docs/verify-arbor-simulator/artifacts/evaluate_m3_model_suite.py`
- Ignored teacher raw output:
  `docs/verify-arbor-simulator/artifacts/m3_normroute_suite_20260606/`
- Ignored model raw output:
  `docs/verify-arbor-simulator/artifacts/m3_model_suite_20260606/`

## Remaining M3 Gaps

- More morphologies would improve confidence, but top-three coverage is enough
  for a first publishable benchmark scaffold.
- The Allen reconstructions are dendrite-only; phrase claims accordingly.
- The models are not morphology-conditioned, so transfer failure is expected
  and should be reported as a benchmark finding, not a model bug.
