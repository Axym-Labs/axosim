# CoreNEURON/Hay Exc0925 Results

Date: 2026-06-05

## Simulator Decision

Arbor is deferred to the population-level follow-up. The current benchmark path
uses CoreNEURON/Hay because it preserves the source Hay morphology/HOC/channel
lineage and reproduced fixed-step NEURON exactly in the probe suite. GPU
execution is still pending because this machine lacks `nvc++`, but the
simulator path is CoreNEURON-compatible.

## Accepted Simulator Setting

Use deterministic Hay synapses plus `exc_event_keep_prob=0.925`,
`inh_event_keep_prob=1.0`.

Rationale: uncalibrated CoreNEURON/Hay was somewhat hotter than the historical
NMDA validation aggregate. A small, explicit excitatory event-retention
calibration was selected to better match the teacher distribution first:
spike-rate, clipped soma mean/std, and dV/dt regime. The Mamba RMSE improvement
is a downstream consequence, not the selection criterion. The more aggressive
`0.90` setting is not the default despite lower RMSE because it overcorrected
the teacher distribution.

| setting | val samples | spike / 1000 | soma mean/std mV | Mamba RMSE mV | Mamba AUC | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| historical NMDA val | 32 | `0.9896` | `-67.595 / 3.236` | `0.9286` | `0.9777` | reference |
| CoreNEURON uncalibrated | 16 | `1.5625` | `-67.186 / 3.452` | `1.1556` | `0.9648` | reject as too hot |
| CoreNEURON `exc=0.925` | 16 | `0.8438` | `-68.016 / 3.261` | `0.8493` | `0.9804` | accepted provisional default |
| CoreNEURON `exc=0.90` | 16 | `0.6563` | `-68.352 / 3.201` | `0.7869` | `0.9863` | reject for default; metric temptation |

Paper-grade caveat: this setting was discovered during exploratory work. Before
using it as a paper claim, rerun selection on fresh seeds with a predeclared
teacher-statistic distance and only then evaluate models on the selected
setting.

## 80k Model Runs On Accepted Setting

Training data: `data/hay-coreneuron-source-train32-exc0925-20260605/train`.
Validation data: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
Both model runs used `80k` presentations.

Combined report:
`artifacts/hay_coreneuron_exc0925_80k_combined_neuro_bench_20260605/report.md`.

| model | M1 RMSE mV | M1 AUC | near-spike RMSE | spike-count MAE @0 | timing F1 matched | sanity flags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Mamba `80k` | `0.8763` | `0.9632` | `4.6860` | `4.7500` | `0.1468` | none |
| RNN `80k` | `0.7987` | `0.9723` | `3.9798` | `4.1250` | `0.0779` | none |

Interpretation: the accepted simulator setting gives a teacher distribution
closer to the historical NeuronIO/NMDA regime than the uncalibrated CoreNEURON
split and also yields substantially better RMSE. RNN wins RMSE in this pass;
Mamba remains much faster for iteration.

## Richer Metrics Than RMSE

Yes. NEURO-BENCH already reports metrics more sophisticated than RMSE:

- M1: voltage RMSE/MAE plus spike AUC, Brier/log-loss, spike-count error, and
  prevalence-matched spike diagnostics.
- M1b: electrophysiological feature fidelity, including spike count/rate,
  non-spiking fraction, peak/trough voltage, ISI/adaptation summaries when
  present.
- M2: regime-aware errors, including far-from-spike, near-spike,
  post-spike/refractory, post-input-event RMSE/MAE and spike-timing F1.

For paper writing, RMSE should stay a headline metric for comparability, but
M1b/M2 should be the defensibility metrics that explain what a given RMSE means.
