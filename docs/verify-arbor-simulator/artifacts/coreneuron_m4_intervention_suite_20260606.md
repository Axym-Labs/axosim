# CoreNEURON/Hay M4 Structured Intervention Suite

Date: 2026-06-06

## Scope

This is the first publishable-direction M4 battery beyond the original single
event-dropout MVP. All interventions use the accepted CoreNEURON/Hay teacher,
the same `16` validation traces, the same clipped soma/local-peak target
policy, and the same three `80k` checkpoints with train-split calibrated spike
thresholds.

Interventions:

- `event_dropout`: Bernoulli drop `35%` of all nonzero events.
- `exc_dropout`: Bernoulli drop `35%` of excitatory events only.
- `inh_dropout`: Bernoulli drop `35%` of inhibitory events only.
- `temporal_jitter`: move existing events by at most `2` ms.
- `site_silence`: silence a contiguous `25%` synaptic-site band in both E/I
  channel halves.

## Teacher Sanity Checks

| intervention | event ratio | soma delta mV | spike-count delta / trace | teacher direction | status |
| --- | ---: | ---: | ---: | --- | --- |
| `event_dropout` | `0.6500` | `-2.4102` | `-4.6250` | `lower_event_drive/lower_spiking/hyperpolarized` | strong |
| `exc_dropout` | `0.8098` | `-4.6864` | `-4.8125` | `lower_event_drive/lower_spiking/hyperpolarized` | strong |
| `inh_dropout` | `0.8402` | `+3.0588` | `+12.8750` | `lower_event_drive/higher_spiking/depolarized` | strong |
| `temporal_jitter` | `0.9948` | `-0.0448` | `-0.1250` | `lower_event_drive/lower_spiking/hyperpolarized` | weak/sensitivity |
| `site_silence` | `0.7291` | `-2.5980` | `-4.5000` | `lower_event_drive/lower_spiking/hyperpolarized` | strong |

Interpretation: four interventions have strong, interpretable teacher
responses. `temporal_jitter` is a weaker sensitivity probe: its absolute soma
response is nonzero (`0.3008 mV`), but the mean soma and spike-count deltas are
small and the spike-count interval crosses zero.

## Model M4 Results

| intervention | model | M4 soma-delta RMSE | M4 spike-delta MAE | soma/spike direction |
| --- | --- | ---: | ---: | --- |
| `event_dropout` | `soma075_calibrated_80k` | `0.7695` | `3.6875` | `True/True` |
| `event_dropout` | `mamba_exc0925_80k` | `0.7935` | `2.0000` | `True/True` |
| `event_dropout` | `rnn_exc0925_80k` | `0.7522` | `4.8125` | `True/True` |
| `exc_dropout` | `soma075_calibrated_80k` | `0.7817` | `3.8125` | `True/True` |
| `exc_dropout` | `mamba_exc0925_80k` | `0.8206` | `2.0000` | `True/True` |
| `exc_dropout` | `rnn_exc0925_80k` | `0.9825` | `4.9375` | `True/True` |
| `inh_dropout` | `soma075_calibrated_80k` | `1.3665` | `14.6250` | `True/True` |
| `inh_dropout` | `mamba_exc0925_80k` | `1.3973` | `14.4375` | `True/True` |
| `inh_dropout` | `rnn_exc0925_80k` | `1.3316` | `10.1250` | `True/True` |
| `temporal_jitter` | `soma075_calibrated_80k` | `0.5854` | `1.8125` | `True/False` |
| `temporal_jitter` | `mamba_exc0925_80k` | `0.6049` | `1.4375` | `True/True` |
| `temporal_jitter` | `rnn_exc0925_80k` | `0.4452` | `1.8125` | `True/False` |
| `site_silence` | `soma075_calibrated_80k` | `0.7603` | `3.6250` | `True/True` |
| `site_silence` | `mamba_exc0925_80k` | `0.7941` | `1.8125` | `True/True` |
| `site_silence` | `rnn_exc0925_80k` | `0.9169` | `4.7500` | `True/True` |

## Interpretation

- The suite now has a real structured-intervention battery rather than a
  single event-dropout probe.
- `inh_dropout` is the hardest and most mechanistically informative probe so
  far: reducing inhibition depolarizes and increases spiking; all models get
  direction right, but spike-delta errors are large.
- Prior `mamba_exc0925_80k` is consistently best on spike-delta MAE for
  event/exc/site/jitter probes, while RNN is best on `inh_dropout` spike delta
  and often best on voltage-delta RMSE.
- `temporal_jitter` should be reported as a timing/sensitivity probe, not a
  strong OOD perturbation, unless strengthened with a larger jitter or a
  timing-specific teacher metric.
- `site_silence` is spatial/site-level, not yet a true morphology-transfer
  result. It should not be oversold as branch morphology generalization.

## Artifacts

Ignored generated pair/eval directories:

- `docs/verify-arbor-simulator/artifacts/coreneuron_m4_event_dropout_16_20260606/`
- `docs/verify-arbor-simulator/artifacts/coreneuron_m4_exc_dropout_16_20260606/`
- `docs/verify-arbor-simulator/artifacts/coreneuron_m4_inh_dropout_16_20260606/`
- `docs/verify-arbor-simulator/artifacts/coreneuron_m4_temporal_jitter_16_20260606/`
- `docs/verify-arbor-simulator/artifacts/coreneuron_m4_site_silence_16_20260606/`
- `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_m4_*_16_20260606/`
