# CoreNEURON/Hay Exc0925 80k Combined Evaluation

- Dataset: `data/hay-coreneuron-source-val16-exc0925-20260605/val`
- Samples: `16`
- Ignore start: `500`
- Plots produced: `True`

## Teacher Subset

| event fraction | spike rate / 1000 steps | soma mean mV | soma std mV | soma min/max mV |
|---:|---:|---:|---:|---:|
| 0.010600 | 0.8438 | -68.0156 | 3.2606 | -77.23/-55.00 |

## Models

| model | params | M1 RMSE mV | M1 AUC | spike-count MAE @0 | spike-count MAE rate-matched | timing F1 rate-matched | near-spike RMSE | worst trace RMSE | sanity flags | plots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mamba_exc0925_80k` | 74188 | 0.8763 | 0.9632 | 4.7500 | 2.3750 | 0.1468 | 4.6860 | 1.2450 | none | [plot 1](plots/mamba_exc0925_80k_trace013.svg), [plot 2](plots/mamba_exc0925_80k_trace002.svg), [plot 3](plots/mamba_exc0925_80k_trace008.svg) |
| `rnn_exc0925_80k` | 27785 | 0.7987 | 0.9723 | 4.1250 | 4.6250 | 0.0779 | 3.9798 | 1.1868 | none | [plot 1](plots/rnn_exc0925_80k_trace013.svg), [plot 2](plots/rnn_exc0925_80k_trace002.svg), [plot 3](plots/rnn_exc0925_80k_trace008.svg) |

## Limitations

- This evaluator runs M1/M1b/M2 only; it does not mix in Arbor M4 intervention pairs.
- Trace plots are diagnostic sanity plots, not publication figures.
- Metrics are over the selected validation subset unless max_samples covers the full split.
