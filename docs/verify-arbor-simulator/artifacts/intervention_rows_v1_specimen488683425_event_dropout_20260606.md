# Benchmark Diagnostic Summary

- rows: `8`
- diagnostics: `intervention_response`
- suites: `intervention-response, paper-summary, teacher-qc`
- models: `model`
- morphologies: `specimen_488683425`
- conditions: `interventions/event_dropout`

## Numeric Metrics

| diagnostic | metric | unit | n | mean | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| intervention_response | baseline_event_fraction | fraction | 1 | 0.0106757 | 0.0106757 | 0.0106757 |
| intervention_response | delta_soma_abs_mean_mv | mV | 1 | 2.20627 | 2.20627 | 2.20627 |
| intervention_response | delta_soma_mean_mv | mV | 1 | -1.71628 | -1.71628 | -1.71628 |
| intervention_response | delta_spike_rate_per_1000_steps | spikes_per_1000_steps | 1 | -1.95455 | -1.95455 | -1.95455 |
| intervention_response | event_fraction_delta | fraction | 1 | -0.00374043 | -0.00374043 | -0.00374043 |
| intervention_response | intervention_event_fraction | fraction | 1 | 0.00693523 | 0.00693523 | 0.00693523 |
| intervention_response | response_detected | bool | 1 | 1 | 1 | 1 |

## Label Metrics

| diagnostic | metric | counts |
| --- | --- | --- |
| intervention_response | teacher_direction | lower_event_drive/lower_spiking/hyperpolarized: 1 |
