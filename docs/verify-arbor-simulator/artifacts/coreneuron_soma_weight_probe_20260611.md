# Stronger Soma Weight Probe

Date: 2026-06-11

Baseline: `soma075_calibrated_80k` had NEURO-BENCH RMSE/AUC `0.8507 / 0.9666` on the same accepted CoreNEURON/Hay train32/val16 setup.

| candidate | spike loss | soma loss | train best val RMSE | NEURO-BENCH RMSE | AUC | lag RMSE | prevalence count MAE | timing F1 | max trace RMSE | flags | plots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `soma085` | `0.15` | `0.85` | `0.8435` | `0.8600` | `0.9647` | `0.6571` | `4.6250` | `0.1417` | `1.2278` | none | True |
| `soma090` | `0.1` | `0.9` | `0.8360` | `0.8463` | `0.9643` | `0.6458` | `5.1250` | `0.1648` | `1.2116` | none | True |
| `soma095` | `0.05` | `0.95` | `0.8395` | `0.8514` | `0.9610` | `0.6483` | `4.7500` | `0.1247` | `1.2257` | none | True |

Interpretation:

- Stronger soma weighting gave at most a small voltage-RMSE improvement over `soma075` on this small split.
- The improvement is not monotonic enough, and spike-ranking/count metrics do not clearly improve, so no preset change is made from this probe.
- Treat the result as suggestive only; full v1 training should compare `soma075` against the best stronger point with uncertainty before any paper-facing default change.
