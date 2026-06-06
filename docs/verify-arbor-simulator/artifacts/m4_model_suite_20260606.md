# M4 Structured Intervention Model Suite

Date: 2026-06-06 14:17 CEST

## Setup

- Source teacher: accepted CoreNEURON/Hay generator.
- Traces: `16` validation traces per intervention.
- Interventions: `event_dropout`, `exc_dropout`, `inh_dropout`,
  `temporal_jitter`, `site_silence`.
- Models: current source-trained `80k` checkpoints with source-train calibrated
  spike thresholds.
- Added metric: M4 now reports soma and spike delta gain in addition to
  direction agreement and delta error.

## Teacher Sanity

| intervention | traces | soma delta mV | spike-count delta | direction |
| --- | ---: | ---: | ---: | --- |
| `event_dropout` | `16` | `-2.4102` | `-4.6250` | lower event drive / lower spiking / hyperpolarized |
| `exc_dropout` | `16` | `-4.6864` | `-4.8125` | lower event drive / lower spiking / hyperpolarized |
| `inh_dropout` | `16` | `+3.0588` | `+12.8750` | lower event drive / higher spiking / depolarized |
| `temporal_jitter` | `16` | `-0.0448` | `-0.1250` | lower event drive / lower spiking / hyperpolarized |
| `site_silence` | `16` | `-2.5980` | `-4.5000` | lower event drive / lower spiking / hyperpolarized |

Four probes have strong teacher responses. `temporal_jitter` remains a weak
timing/sensitivity probe; its gain values are unstable because teacher mean
spike delta is close to zero.

## Aggregate Model Behavior

| model | mean soma RMSE | mean spike MAE | soma direction frac | spike direction frac | mean soma gain error | mean spike gain error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soma075_calibrated_80k` | `0.8527` | `5.5125` | `1.000` | `0.800` | `0.0617` | `2.0415` |
| `mamba_exc0925_80k` | `0.8821` | `4.3375` | `1.000` | `1.000` | `0.1184` | `0.6448` |
| `rnn_exc0925_80k` | `0.8857` | `5.2875` | `1.000` | `0.800` | `0.1103` | `1.8077` |

## Interpretation

M4 is now a genuine structured behavior battery rather than a single
event-dropout probe. All models preserve soma-direction changes across all
interventions. Prior `mamba_exc0925_80k` is the most robust spike-behavior
model by this battery: it preserves spike direction on all five probes and has
the lowest mean spike-delta MAE. `soma075_calibrated_80k` has the best average
soma-delta RMSE and soma gain, so it remains the voltage-oriented Mamba path.
RNN is competitive on voltage and best on inhibitory-dropout spike MAE, but it
fails spike direction on the weak temporal-jitter probe.

## Remaining M4 Caveats

- `temporal_jitter` should be reported as a weak sensitivity probe unless it is
  redesigned with stronger timing-specific metrics.
- `site_silence` is spatial/site-level, not a true branch morphology probe.
- M4 now covers event, E/I, timing, and site perturbations; morphology-dependent
  perturbations are partly covered by M3 rather than source-cell M4.

## Artifacts

- Evaluator:
  `docs/verify-arbor-simulator/artifacts/evaluate_m4_model_suite.py`
- Ignored raw output:
  `docs/verify-arbor-simulator/artifacts/m4_model_suite_20260606/`
