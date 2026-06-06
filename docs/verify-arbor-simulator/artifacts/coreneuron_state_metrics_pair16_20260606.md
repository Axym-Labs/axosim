# CoreNEURON/Hay State-Conditioned Metrics

Date: 2026-06-06

## Scope

This report adds the first publishable-direction replacement for global RMSE:
a reusable neuron-state atlas plus state-conditioned dynamic fidelity metrics.
The same `16` validation traces and `16`-pair M4 event-dropout artifact are
used as in the current CoreNEURON/Hay benchmark report.

State labels are mutually exclusive and derived from teacher voltage/spikes and
signed input drive: `spiking`, `refractory`, `near_threshold`,
`excited_drive`, `inhibited_drive`, `hyperpolarized_quiet`, and `resting`.

## Headline Metrics

| model | RMSE mV | state-weighted trace distance mV | dynamic distance | AUC | calibrated count MAE | calibrated timing F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soma075_calibrated_80k` | `0.8507` | `2.0343` | `0.5993` | `0.9666` | `3.8125` | `0.1272` |
| `mamba_exc0925_80k` | `0.8763` | `2.0808` | `0.5249` | `0.9632` | `2.0625` | `0.0972` |
| `rnn_exc0925_80k` | `0.7987` | `1.8707` | `0.6561` | `0.9723` | `4.9375` | `0.0762` |

## Interpretation

- Global RMSE still ranks RNN best, and state-weighted trace distance agrees:
  RNN is best on voltage shape across behavioral states.
- The composite dynamic distance ranks prior `mamba_exc0925_80k` best because
  it has substantially better calibrated spike-count behavior; this exposes a
  real voltage-vs-spike tradeoff hidden by global RMSE.
- Rare states are where the voltage failures live. On `mamba_exc0925_80k`,
  `spiking` is only `0.09%` of evaluated timesteps but has `5.43 mV` RMSE;
  `refractory` is `0.53%` with `3.63 mV`; `resting` is `67.63%` with
  `0.77 mV`. This is the central reason global RMSE is not enough.

## Artifact

- Ignored detailed JSON:
  `docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_state_metrics_pair16_20260606/results.json`.
