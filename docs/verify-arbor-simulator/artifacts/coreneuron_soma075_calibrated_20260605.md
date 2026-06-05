# CoreNEURON/Hay Soma075 With Calibrated Spike Threshold

Date: 2026-06-05

## Setup

- Teacher/data: accepted CoreNEURON/Hay `exc_event_keep_prob=0.925`,
  `inh_event_keep_prob=1.0`.
- Train split: `data/hay-coreneuron-source-train32-exc0925-20260605/train`.
- Validation split: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
- Model: Mamba official wrapper,
  `configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json`.
- Training: `80,000` presentations, batch `16`, `500`-step windows,
  `150`-step burn-in, AdamW `3e-4`, weight decay `1e-4`,
  `spike_loss_weight=0.25`, `soma_loss_weight=0.75`.
- Spike threshold calibration: global logit threshold learned on the `32`
  training traces by matching calibration-set spike prevalence, then frozen
  before evaluation on the `16` validation traces.

## Result

| model | threshold source | threshold | RMSE mV | AUC | near-spike RMSE | count MAE @0 | count MAE calibrated | timing F1 calibrated | raw flags | calibrated flags | plots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `soma075_calibrated_80k` | train prevalence | `-2.6941` | `0.8507` | `0.9666` | `4.6228` | `4.8125` | `3.8125` | `0.1272` | `threshold_zero_predicts_no_spikes` | none | yes |

Training metrics: best validation RMSE during training was `0.8446 mV` at
epoch `2`; NEURO-BENCH full-trace validation RMSE for the saved best checkpoint
is `0.8507 mV`.

## Interpretation

This is now the current Mamba metric path, with an explicit caveat:

- Voltage metrics improve versus the accepted current Mamba baseline
  (`0.8507 mV` vs `0.8763 mV` RMSE).
- AUC also improves modestly (`0.9666` vs `0.9632`).
- The raw zero-threshold spike head remains miscalibrated and predicts no
  spikes, so raw zero-threshold sanity still flags it.
- The learned threshold is fit without validation leakage and emits validation
  spikes without sanity flags; calibrated count MAE improves over zero-threshold
  count MAE (`3.8125` vs `4.8125`).
- Calibrated timing F1 (`0.1272`) remains worse than the old NeuronIO/NMDA
  Mamba canonical timing F1 and worse than the current baseline's
  prevalence-matched timing. This path is a voltage/RMSE improvement, not a
  solved spike-timing result.

## Artifacts

- Ignored run directory:
  `docs/verify-arbor-simulator/artifacts/coreneuron_soma075_calibrated_20260605/`.
- Evaluation report:
  `docs/verify-arbor-simulator/artifacts/coreneuron_soma075_calibrated_20260605/neuro_bench_eval/report.md`.
- The ignored evaluation directory contains `results.json` and diagnostic SVG
  plots.
