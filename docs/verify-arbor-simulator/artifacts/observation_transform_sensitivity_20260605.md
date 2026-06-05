# Observation Transform Sensitivity

## Summaries

| view | mean mV | std mV | min/max mV | |dV/dt| p95/p99 |
|---|---:|---:|---:|---:|
| `reference` | -67.5733 | 3.0545 | -80.38/-55.00 | 0.4375/0.6875 |
| `arbor_raw` | -70.0238 | 3.1248 | -78.58/-4.79 | 3.5416/5.7729 |

## Transforms

| transform | std mV | max mV | |dV/dt| p95/p99 | raw RMSE mV | score |
|---|---:|---:|---:|---:|---:|
| `clip_slew_limit0.7` | 2.1943 | -57.68 | 0.7000/0.7000 | 2.0086 | 1.1352 |
| `clip_slew_limit0.5` | 2.0242 | -59.50 | 0.5000/0.5000 | 2.2201 | 1.2803 |
| `clip_lowpass_tau10` | 1.7195 | -59.78 | 0.4315/0.6891 | 2.4566 | 1.3426 |
| `clip_lowpass_tau8` | 1.8291 | -59.01 | 0.5193/0.8314 | 2.3525 | 1.4512 |
| `clip_slew_limit1` | 2.3570 | -55.88 | 1.0000/1.0000 | 1.7731 | 1.5725 |
| `clip_lowpass_tau5` | 2.0490 | -57.66 | 0.7548/1.2262 | 2.1057 | 1.8615 |
| `clip_lowpass_tau15` | 1.5176 | -61.45 | 0.3046/0.4867 | 2.6217 | 1.8706 |
| `clip_slew_limit1.5` | 2.5171 | -55.00 | 1.5000/1.5000 | 1.5168 | 2.4124 |
| `clip_lowpass_tau3` | 2.2625 | -55.95 | 1.1045/1.8246 | 1.8050 | 2.5961 |
| `clip_slew_limit2` | 2.6154 | -55.00 | 2.0000/2.0000 | 1.3531 | 3.3141 |
| `clip_lowpass_tau2` | 2.4096 | -55.28 | 1.4622/2.4081 | 1.5559 | 3.3902 |
| `clip_slew_limit3` | 2.7268 | -55.00 | 3.0000/3.0000 | 1.1555 | 5.2027 |
| `clip_only` | 2.8271 | -55.00 | 3.5092/5.5447 | 0.9248 | 8.1563 |

## Interpretation

- A target transform is only defensible if it corresponds to an observation convention, not if it merely hides simulator dynamics.
- Low-pass or slew transforms that match NeuronIO dV/dt while strongly shrinking voltage variance should be treated as distortive unless NeuronIO measurement filtering justifies them.
- For Arbor replacement data, keep raw soma voltage available even when a clipped training target is used for historical Branch-ELM comparability.
