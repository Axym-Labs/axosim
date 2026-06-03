# NEURO-BENCH MVP Smoke Report

## Teacher M4 Event-Rate Shift
- Baseline event fraction: `0.001133`
- Intervention event fraction: `0.008048`
- Delta spike rate / 1000 steps: `-23.9091`
- Delta soma mean mV: `14.9830`
- Direction: `higher_event_drive/lower_spiking/depolarized`
- Warning: Higher event density reduced teacher spike rate in this small pair. Treat M4 as teacher-fidelity/stress testing, not biological robustness, until the teacher protocol is calibrated.

## Model Smoke Comparison

| model | M1 RMSE mV | M1 AUC | M4 soma-delta RMSE mV | M4 spike-count-delta MAE |
|---|---:|---:|---:|---:|
| `branch_elm_30_trained` | 17.9833 | 0.3888 | 21.2951 | 317.0000 |
| `mamba_pytorch_best` | 17.5602 | 0.4970 | 23.7367 | 318.2500 |

## Limitations

- This is an MVP smoke comparison on a small intervention pair.
- The M4 teacher response is not yet a biological-validity claim.
- If Mamba is included, this uses the slow pure-PyTorch checkpoint available in `runs/`.
