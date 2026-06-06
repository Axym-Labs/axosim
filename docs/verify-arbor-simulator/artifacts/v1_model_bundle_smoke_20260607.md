# v1 Model Bundle Smoke - 2026-06-07

Dataset: `artifacts/v1_swc_cpu_core_batched_smoke_20260607/`

Scope: package-facing validation that Mamba, RNN, and Branch-ELM checkpoints
can run through `bnn-sim benchmark-v1` on a v1-shaped all-five-morphology SWC
pilot. This is not a paper-facing performance estimate: the pilot uses only
`4` validation traces per morphology, `2` event-dropout pairs per morphology,
and `1000 ms` traces, while the checkpoints were trained on earlier source-Hay
or NeuronIO-like distributions.

| model | env/device | model rows | soma RMSE mV | state-weighted trace distance mV | spike AUC | spike-count MAE/trace |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RNN `rnn_exc0925_80k_best` | `.venv`, CPU | `177` | `3.165` | `4.995` | `0.833` | `1.35` |
| Mamba `mamba_exc0925_80k_best` | `.venv-mamba`, CUDA | `177` | `3.119` | `4.914` | `0.854` | `1.35` |
| Branch-ELM `branch_elm_official_fulltrace_80k_best` | `.venv`, CPU | `177` | `3.237` | `4.996` | `0.863` | `1.35` |

State-specific voltage RMSE on the smoke is highest near spikes for all three
models:

| model | spiking | refractory | near-threshold | excited-drive | inhibited-drive | hyperpolarized-quiet | resting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNN | `9.843` | `7.713` | `7.193` | `2.825` | `2.367` | `1.883` | `3.142` |
| Mamba | `9.763` | `7.776` | `6.979` | `2.515` | `2.372` | `1.904` | `3.092` |
| Branch-ELM | `10.134` | `7.506` | `7.466` | `2.517` | `2.346` | `1.720` | `3.285` |

Intervention smoke signal: for sampled event-dropout pairs, teacher soma
direction is preserved (`soma_mean=True`) while spike-count direction is not
preserved in the small paired samples (`spike_count=False` in inspected
specimen rows). This matches the intended role of intervention diagnostics as
behavioral tests, not merely trace-RMSE repeats.

Operational notes:
- `.venv` PyTorch cannot run CUDA kernels on the RTX 5090 (`sm_120`); RNN and
  Branch-ELM smokes used CPU.
- Mamba used `.venv-mamba`, which has Torch `2.7.1+cu128` and `mamba_ssm`.
- All three bundle commands wrote `9` artifacts: teacher QC, routing QC,
  model validation rows, five specimen-specific intervention reports, and
  morphology-contrast rows.
