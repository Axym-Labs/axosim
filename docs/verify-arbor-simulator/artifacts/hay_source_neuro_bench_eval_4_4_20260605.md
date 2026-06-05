# Direct Hay Source NEURO-BENCH Evaluation

## Purpose

This is a source-teacher sanity evaluation after the Arbor-vs-NeuronIO mismatch
diagnosis. The dataset was generated with the original Hay/NEURON morphology,
biophysics, MOD mechanisms, segment-level NMDA/GABA_A synapses, NeuronIO-like
input generator, local-peak spike labels, and historical `-55 mV` soma clipping.

## Dataset

- Path: `data/hay-neuron-source-sanity-4_4-20260605/`
- Splits: `4` train traces, `4` validation traces
- Duration: `6000 ms`
- Environment: `/home/davwis/miniconda3/envs/bnn_sim` with NEURON `8.2.7`
- Generator: `docs/verify-arbor-simulator/artifacts/generate_hay_neuron_dataset.py`
- Val teacher subset: event fraction `0.011145`, spike rate `1.8333 / 1000 steps`,
  clipped soma mean/std `-67.4122 / 3.9298 mV`, soma min/max `-77.27/-55.00 mV`

## Results

| model | environment | RMSE mV | AUC | near-spike RMSE mV | plots |
|---|---|---:|---:|---:|---|
| `branch_elm_official80k` | conda `bnn_sim` GPU | 1.1734 | 0.9514 | 5.0334 | yes |
| `branch_elm_random800k` | conda `bnn_sim` GPU | 0.9840 | 0.9707 | 4.2381 | yes |
| `mamba_compact80k` | `.venv-mamba` GPU | 1.1912 | 0.9424 | 5.4113 | yes |
| `mamba_compact480k_partial` | `.venv-mamba` GPU | 1.1427 | 0.9666 | 5.1106 | yes |

## Interpretation

- Direct Hay/NEURON is benchmark-compatible and lands in the NeuronIO-like
  voltage/input regime, unlike the current Arbor proxy.
- The small validation subset is somewhat hotter than the earlier 2-sample
  probe because one trace fired strongly, so these numbers are source-teacher
  sanity evidence, not final model ranking.
- All evaluated models produced finite outputs and trace plots. The shared
  `threshold_zero_predicts_no_spikes` flag remains a calibration/decision-threshold
  issue; prevalence-matched spike metrics are more informative for this pass.
- Official-Mamba evaluation requires `.venv-mamba`; the conda `bnn_sim` Torch
  supports RTX 5090 but has an incompatible `mamba_ssm` extension, while `.venv`
  lacks `mamba_ssm`.

## Artifacts

- Branch-ELM report: `hay_source_neuro_bench_eval_branch_4_4_20260605/report.md`
- Branch-ELM JSON: `hay_source_neuro_bench_eval_branch_4_4_20260605/results.json`
- Mamba report: `hay_source_neuro_bench_eval_mamba_4_4_20260605/report.md`
- Mamba JSON: `hay_source_neuro_bench_eval_mamba_4_4_20260605/results.json`
