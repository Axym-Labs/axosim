# Benchmark Diagnostic Summary

- rows: `105`
- diagnostics: `feature_fidelity, spike_behavior, state_dynamics, trace_core`
- suites: `feature-fidelity, paper-summary, spike-behavior, state-dynamics, trace-core`
- models: `rnn_exc0925_80k`
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
| feature_fidelity | peak_voltage_mv_prediction_mean | mV | 1 | -59.1796 | -59.1796 | -59.1796 |
| feature_fidelity | peak_voltage_mv_teacher_mean | mV | 1 | -58.7649 | -58.7649 | -58.7649 |
| feature_fidelity | spike_count_prediction_mean | count | 1 | 0 | 0 | 0 |
| feature_fidelity | spike_count_teacher_mean | count | 1 | 0.5 | 0.5 | 0.5 |
| feature_fidelity | trough_voltage_mv_prediction_mean | mV | 1 | -74.0004 | -74.0004 | -74.0004 |
| feature_fidelity | trough_voltage_mv_teacher_mean | mV | 1 | -74.3984 | -74.3984 | -74.3984 |
| spike_behavior | count_mae_per_trace | count | 2 | 0.25 | 0 | 0.5 |
| spike_behavior | predicted_count | count | 2 | 0.5 | 0 | 1 |
| spike_behavior | spike_auc | fraction | 1 | 0.959815 | 0.959815 | 0.959815 |
| spike_behavior | spike_brier | fraction | 1 | 9.09584e-05 | 9.09584e-05 | 9.09584e-05 |
| spike_behavior | spike_log_loss | unitless | 1 | 0.00120691 | 0.00120691 | 0.00120691 |
| spike_behavior | threshold | logit | 2 | -2.00227 | -4.00454 | 0 |
| state_dynamics | dvdt_mae_mv_per_step | mV | 7 | 0.488721 | 0.121662 | 1.42824 |
| state_dynamics | fraction | fraction | 7 | 0.142857 | 9.09091e-05 | 0.688182 |
| state_dynamics | spike_count_error | count | 7 | -0.142857 | -1 | 0 |
| state_dynamics | spike_predicted_count | count | 7 | 0 | 0 | 0 |
| state_dynamics | spike_target_count | count | 7 | 0.142857 | 0 | 1 |
| state_dynamics | voltage_mae_mv | mV | 7 | 1.31334 | 0.309402 | 4.18137 |
| state_dynamics | voltage_rmse_mv | mV | 7 | 1.50991 | 0.389984 | 4.18137 |
| trace_core | dynamic_distance | unitless | 1 | 0.539875 | 0.539875 | 0.539875 |
| trace_core | lag_tolerant_voltage_rmse_mv | mV | 1 | 0.347889 | 0.347889 | 0.347889 |
| trace_core | soma_mae_mv | mV | 3 | 0.377041 | 0.360766 | 0.393316 |
| trace_core | soma_max_abs_error_mv | mV | 2 | 4.49258 | 4.17935 | 4.80581 |
| trace_core | soma_rmse_mv | mV | 3 | 0.51515 | 0.489609 | 0.540275 |
| trace_core | spike_auc | fraction | 1 | 0.959815 | 0.959815 | 0.959815 |
| trace_core | spike_brier | fraction | 1 | 9.09584e-05 | 9.09584e-05 | 9.09584e-05 |
| trace_core | spike_count_mae_per_trace | count | 1 | 0.5 | 0.5 | 0.5 |
| trace_core | spike_log_loss | unitless | 1 | 0.00120691 | 0.00120691 | 0.00120691 |
| trace_core | spike_train_distance | unitless | 1 | 1.44277 | 1.44277 | 1.44277 |
| trace_core | state_trace_shape_distance | unitless | 1 | 0.245326 | 0.245326 | 0.245326 |
| trace_core | state_weighted_trace_distance_mv | mV | 1 | 1.50991 | 1.50991 | 1.50991 |
| trace_core | target_spike_count | count | 2 | 0.5 | 0 | 1 |
| trace_core | trace_shape_distance | unitless | 1 | 0.25262 | 0.25262 | 0.25262 |
