# V1 Benchmark Bundle Validation

Date: 2026-06-06

## Commands

- Teacher-only bundle:
  `PYTHONPATH=src .venv/bin/python -m bnn_sim.cli benchmark-v1 --dataset-root data/hay-coreneuron-source-val16-exc0925-20260605 --output-dir docs/verify-arbor-simulator/artifacts/v1_benchmark_bundle_teacher_pilot_20260606 --intervention-pair-glob 'docs/verify-arbor-simulator/artifacts/v1_teacher_qc_pilot_20260606/*/intervention_pair.npz' --max-val-samples 4 --max-intervention-samples 4 --morphology-id source_hay --ignore-start 500`
- Mamba bundle:
  `PYTHONPATH=src .venv-mamba/bin/python -m bnn_sim.cli benchmark-v1 --dataset-root data/hay-coreneuron-source-val16-exc0925-20260605 --output-dir docs/verify-arbor-simulator/artifacts/v1_benchmark_bundle_mamba_pilot_20260606 --intervention-pair-glob 'docs/verify-arbor-simulator/artifacts/v1_teacher_qc_pilot_20260606/*/intervention_pair.npz' --max-val-samples 2 --max-intervention-samples 2 --morphology-id source_hay --ignore-start 500 --checkpoint docs/verify-arbor-simulator/artifacts/hay_coreneuron_exc0925_mamba80k_20260605/mamba_exc0925_80k_best.pt --model-id mamba_exc0925_80k --device cuda --batch-size 1`

## Outputs

| run | output dir | artifacts | tracked here |
| --- | --- | ---: | --- |
| teacher-only | `artifacts/v1_benchmark_bundle_teacher_pilot_20260606/` | 13 | no, ignored bundle dir |
| Mamba | `artifacts/v1_benchmark_bundle_mamba_pilot_20260606/` | 14 | no, ignored bundle dir |

The directories contain per-condition `.jsonl` rows, `.md` summaries, and a
`manifest.json`. This report is the tracked compact validation record.

## Sanity Results

Teacher-only bundle:

| check | result |
| --- | --- |
| validation teacher-QC rows | 18 |
| validation clipping rows | 9 |
| clipped fraction on 4-trace pilot | 0.00218 |
| near-spike clipped fraction on 4-trace pilot | 0.395 |
| near-spike distortion RMSE on 4-trace pilot | 20.97 mV |
| morphology-routing QC rows | 35 |
| intervention pairs covered | 10 |
| v1 morphologies covered | 5 |
| interventions covered | `event_dropout`, `inh_dropout` |
| event-dropout teacher direction | 5/5 `lower_event_drive/lower_spiking/hyperpolarized` |
| inhibitory-dropout teacher direction | 5/5 `lower_event_drive/higher_spiking/depolarized` |
| morphology-contrast rows | 70 |

Mamba bundle:

| check | result |
| --- | --- |
| validation teacher-QC rows | 14 |
| validation clipping rows | 7 |
| clipped fraction on 2-trace pilot | 0.000182 |
| near-spike clipped fraction on 2-trace pilot | 0.286 |
| near-spike distortion RMSE on 2-trace pilot | 29.49 mV |
| morphology-routing QC rows | 35 |
| validation model rows | 105 |
| validation trace-core soma RMSE on 2-trace pilot | 0.5805 mV summary mean |
| validation trace-core spike AUC on 2-trace pilot | 0.9966 |
| intervention model rows per pair | 13 |
| morphology-contrast rows | 90 |
| soma direction agreement | 10/10 |
| spike direction agreement | 5/10 |

Intervention detail for Mamba at `max_intervention_samples=2`:

| intervention | morphologies | soma direction agreement | spike direction agreement | soma-delta RMSE range |
| --- | ---: | ---: | ---: | ---: |
| `event_dropout` | 5 | 5/5 | 0/5 | 1.1353-3.2267 mV |
| `inh_dropout` | 5 | 5/5 | 5/5 | 1.7428-4.5496 mV |

Selected morphology-contrast ranges:

| run | condition | metric | range |
| --- | --- | --- | ---: |
| teacher-only | `event_dropout` | teacher `delta_soma_mean_mv` | 1.2215 mV |
| teacher-only | `inh_dropout` | teacher `delta_soma_mean_mv` | 1.2794 mV |
| teacher-only | `event_dropout` | teacher `delta_spike_rate_per_1000_steps` | 2.8636 |
| teacher-only | `inh_dropout` | teacher `delta_spike_rate_per_1000_steps` | 3.9091 |
| Mamba | `event_dropout` | model `soma_delta_rmse_mv` | 2.0914 mV |
| Mamba | `inh_dropout` | model `soma_delta_rmse_mv` | 2.8067 mV |
| Mamba | `event_dropout` | model `spike_count_delta_mae_per_trace` | 14.0 |
| Mamba | `inh_dropout` | model `spike_count_delta_mae_per_trace` | 25.5 |

## Interpretation

The promoted bundle command is now validated as a package-facing path for
running validation-set teacher/model diagnostics plus paired intervention
diagnostics from the existing v1 pilot artifacts. The Mamba intervention
results are not a final scientific estimate because the bundle used only two
paired traces per morphology/intervention; they are a smoke validation of the
benchmark path and an early signal that voltage-delta behavior is more stable
than spike-delta behavior for event dropout in this tiny pilot.
