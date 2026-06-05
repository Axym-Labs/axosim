# Observation And Dynamics Comparison

## Dataset Views

| view | spike / 1000 steps | soma mean/std mV | soma min/max mV | |dV/dt| p95/p99 |
|---|---:|---:|---:|---:|
| `nmda` | 0.7898 | -67.5733/3.0545 | -80.38/-55.00 | 0.4375/0.6875 |
| `arbor_raw` | 0.5852 | -70.0238/3.1248 | -78.58/-4.79 | 3.5416/5.7729 |
| `arbor_clipped` | 0.5852 | -70.0543/2.8271 | -78.58/-55.00 | 3.5092/5.5447 |

## Clipping Distortion

- Arbor raw -> `-55 mV` clip affects `0.2094%` of masked timesteps.
- Global raw-vs-clipped distortion RMSE: `0.9248 mV`; near-spike distortion RMSE: `8.9514 mV`.
- Maximum removed AP peak amplitude above threshold: `50.21 mV`.

| clip mV | affected % | global RMSE mV | near-spike RMSE mV | max removed mV |
|---:|---:|---:|---:|---:|
| -65.0 | 4.8455 | 1.4248 | 12.1017 | 60.21 |
| -60.0 | 0.6219 | 1.1095 | 10.4085 | 55.21 |
| -55.0 | 0.2094 | 0.9248 | 8.9514 | 50.21 |
| -50.0 | 0.1307 | 0.7706 | 7.6429 | 45.21 |
| -45.0 | 0.0980 | 0.6321 | 6.4081 | 40.21 |

## Dynamic Gap After Clipping

- Arbor clipped minus NMDA spike-rate delta: `-0.2045` spikes / 1000 steps.
- Arbor clipped minus NMDA soma mean/std deltas: `-2.4810` / `-0.2274 mV`.
- Spike-triggered average waveform RMSE: `7.1007 mV`.
- High-drive response-kernel RMSE: `0.0942 mV`.

## Interpretation

- Hard clipping is an observation convention: it discards raw action-potential peak amplitude while preserving the spike-label channel.
- It is defensible for NeuronIO comparability only if raw AP-shape fidelity is reported separately on the raw-soma view.
- For simulator calibration, match dynamic features such as spike-triggered waveform, dV/dt distribution, and high-drive response kernels rather than only mean/std/spike rate.
