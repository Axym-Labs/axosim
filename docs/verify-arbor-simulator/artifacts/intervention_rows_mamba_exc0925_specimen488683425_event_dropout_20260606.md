# Benchmark Diagnostic Summary

- rows: `13`
- diagnostics: `intervention_response`
- suites: `intervention-response, paper-summary, teacher-qc`
- models: `mamba_exc0925_80k`
- morphologies: `specimen_488683425`
- conditions: `interventions/event_dropout`

## Numeric Metrics

| diagnostic | metric | unit | n | mean | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| intervention_response | baseline_event_fraction | fraction | 1 | 0.0106767 | 0.0106767 | 0.0106767 |
| intervention_response | delta_soma_abs_mean_mv | mV | 1 | 2.22861 | 2.22861 | 2.22861 |
| intervention_response | delta_soma_mean_mv | mV | 1 | -1.91233 | -1.91233 | -1.91233 |
| intervention_response | delta_spike_rate_per_1000_steps | spikes_per_1000_steps | 1 | -1.63636 | -1.63636 | -1.63636 |
| intervention_response | event_fraction_delta | fraction | 1 | -0.00374036 | -0.00374036 | -0.00374036 |
| intervention_response | intervention_event_fraction | fraction | 1 | 0.00693634 | 0.00693634 | 0.00693634 |
| intervention_response | response_detected | bool | 1 | 1 | 1 | 1 |
| intervention_response | soma_delta_rmse_mv | mV | 1 | 1.91648 | 1.91648 | 1.91648 |
| intervention_response | spike_count_delta_mae_per_trace | count | 1 | 9 | 9 | 9 |

## Label Metrics

| diagnostic | metric | counts |
| --- | --- | --- |
| intervention_response | direction_agreement | {'soma_mean': True, 'spike_count': False}: 1 |
| intervention_response | teacher_direction | lower_event_drive/lower_spiking/hyperpolarized: 1 |
