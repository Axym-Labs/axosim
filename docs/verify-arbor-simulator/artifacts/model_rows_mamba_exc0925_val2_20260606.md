# Benchmark Diagnostic Summary

- rows: `105`
- diagnostics: `feature_fidelity, spike_behavior, state_dynamics, trace_core`
- suites: `feature-fidelity, paper-summary, spike-behavior, state-dynamics, trace-core`
- models: `mamba_exc0925_80k`
- morphologies: `source_hay`
- conditions: `val, val:spike_prevalence_matched, val:spike_threshold_zero`
- sample trace IDs: `hay_l5pc_coreneuron:det_syn:000000:seed1055729794, hay_l5pc_coreneuron:det_syn:000001:seed1058259185`

## Numeric Metrics

| diagnostic | metric | unit | n | mean | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| feature_fidelity | firing_rate_per_1000_steps_prediction_mean | spikes_per_1000_steps | 1 | 0 | 0 | 0 |
| feature_fidelity | firing_rate_per_1000_steps_teacher_mean | spikes_per_1000_steps | 1 | 0.0909091 | 0.0909091 | 0.0909091 |
| feature_fidelity | non_spiking_prediction_mean | unitless | 1 | 1 | 1 | 1 |
| feature_fidelity | non_spiking_teacher_mean | unitless | 1 | 0.5 | 0.5 | 0.5 |
| feature_fidelity | peak_voltage_mv_prediction_mean | mV | 1 | -60.2164 | -60.2164 | -60.2164 |
| feature_fidelity | peak_voltage_mv_teacher_mean | mV | 1 | -58.7649 | -58.7649 | -58.7649 |
| feature_fidelity | spike_count_prediction_mean | count | 1 | 0 | 0 | 0 |
| feature_fidelity | spike_count_teacher_mean | count | 1 | 0.5 | 0.5 | 0.5 |
| feature_fidelity | trough_voltage_mv_prediction_mean | mV | 1 | -73.9492 | -73.9492 | -73.9492 |
| feature_fidelity | trough_voltage_mv_teacher_mean | mV | 1 | -74.3984 | -74.3984 | -74.3984 |
| spike_behavior | count_mae_per_trace | count | 2 | 0.25 | 0 | 0.5 |
| spike_behavior | predicted_count | count | 2 | 0.5 | 0 | 1 |
| spike_behavior | spike_auc | fraction | 1 | 0.996636 | 0.996636 | 0.996636 |
| spike_behavior | spike_brier | fraction | 1 | 9.15952e-05 | 9.15952e-05 | 9.15952e-05 |
| spike_behavior | spike_log_loss | unitless | 1 | 0.000794954 | 0.000794954 | 0.000794954 |
| spike_behavior | threshold | logit | 2 | -1.13219 | -2.26438 | 0 |
| state_dynamics | dvdt_mae_mv_per_step | mV | 7 | 0.72429 | 0.163154 | 2.01803 |
| state_dynamics | fraction | fraction | 7 | 0.142857 | 9.09091e-05 | 0.688182 |
| state_dynamics | spike_count_error | count | 7 | -0.142857 | -1 | 0 |
| state_dynamics | spike_predicted_count | count | 7 | 0 | 0 | 0 |
| state_dynamics | spike_target_count | count | 7 | 0.142857 | 0 | 1 |
| state_dynamics | voltage_mae_mv | mV | 7 | 1.30601 | 0.445401 | 3.4738 |
| state_dynamics | voltage_rmse_mv | mV | 7 | 1.44237 | 0.548167 | 3.4738 |
| trace_core | dynamic_distance | unitless | 1 | 0.59853 | 0.59853 | 0.59853 |
| trace_core | lag_tolerant_voltage_rmse_mv | mV | 1 | 0.385916 | 0.385916 | 0.385916 |
| trace_core | soma_mae_mv | mV | 3 | 0.45231 | 0.449266 | 0.455355 |
| trace_core | soma_max_abs_error_mv | mV | 2 | 2.79843 | 2.00984 | 3.58702 |
| trace_core | soma_rmse_mv | mV | 3 | 0.580466 | 0.561964 | 0.598774 |
| trace_core | spike_auc | fraction | 1 | 0.996636 | 0.996636 | 0.996636 |
| trace_core | spike_brier | fraction | 1 | 9.15952e-05 | 9.15952e-05 | 9.15952e-05 |
| trace_core | spike_count_mae_per_trace | count | 1 | 0.5 | 0.5 | 0.5 |
| trace_core | spike_log_loss | unitless | 1 | 0.000794954 | 0.000794954 | 0.000794954 |
| trace_core | spike_train_distance | unitless | 1 | 1.44277 | 1.44277 | 1.44277 |
| trace_core | state_trace_shape_distance | unitless | 1 | 0.290307 | 0.290307 | 0.290307 |
| trace_core | state_weighted_trace_distance_mv | mV | 1 | 1.44237 | 1.44237 | 1.44237 |
| trace_core | target_spike_count | count | 2 | 0.5 | 0 | 1 |
| trace_core | trace_shape_distance | unitless | 1 | 0.267159 | 0.267159 | 0.267159 |
