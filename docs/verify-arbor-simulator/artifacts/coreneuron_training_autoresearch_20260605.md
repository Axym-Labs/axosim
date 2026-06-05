# CoreNEURON/Hay Training Autoresearch

Created: `2026-06-05 22:35:20 CEST`.

## Setup

- Train data: `data/hay-coreneuron-source-train32-exc0925-20260605/train`.
- Validation data: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
- Presentation cap: `80000` presentations per candidate.
- Simulator/teacher fixed: accepted CoreNEURON/Hay `exc_event_keep_prob=0.925`, `inh_event_keep_prob=1.0`.
- Interpretation guard: these are training/data-policy interventions, not simulator-fidelity improvements.

## Results

| candidate | intervention | RMSE mV | AUC | near-spike RMSE | count MAE @0 | timing F1 matched | worst trace RMSE | sanity flags | plots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `baseline_repro` | control rerun | `0.8763` | `0.9632` | `4.6860` | `4.7500` | `0.1468` | `1.2450` | `none` | `True` |
| `longctx1000` | 1000-step context, 300 burn-in | `0.8586` | `0.9631` | `4.4535` | `4.5000` | `0.1070` | `1.2269` | `none` | `True` |
| `soma075` | 0.25 spike / 0.75 soma loss | `0.8507` | `0.9666` | `4.6228` | `4.8125` | `0.1598` | `1.2070` | `threshold_zero_predicts_no_spikes` | `True` |
| `slope005` | soma-slope loss 0.05 | `0.8774` | `0.9631` | `4.7162` | `4.8125` | `0.1154` | `1.2471` | `threshold_zero_predicts_no_spikes` | `True` |
| `sparse005` | sparse soma loss 0.05 | `0.8808` | `0.9643` | `4.6584` | `4.7500` | `0.1648` | `1.2637` | `none` | `True` |
| `official_sampling` | official-style window sampling | `0.9025` | `0.9683` | `5.0615` | `4.7500` | `0.1693` | `1.2360` | `none` | `True` |

## Decision

Best completed RMSE was `soma075` at `0.8507 mV`, versus same-loop baseline `0.8763 mV` (`-0.0255` mV delta).
This is worth a follow-up validation rerun on a fresh seed/split before accepting as a paper default.

## Artifacts

- Candidate directories: `docs/verify-arbor-simulator/artifacts/coreneuron_training_autoresearch_20260605/candidates/<candidate>/`.
- Each completed candidate has `neuro_bench_eval/results.json`, `report.md`, and diagnostic SVG trace plots.
