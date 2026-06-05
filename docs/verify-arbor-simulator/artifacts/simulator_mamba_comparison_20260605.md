# Current CoreNEURON/Hay vs Old NeuronIO/NMDA

Date: 2026-06-05

## Scope

This compares the accepted current setup, CoreNEURON/Hay with
`exc_event_keep_prob=0.925`, against the old NeuronIO/NMDA setup using the
validation subsets and Mamba checkpoints that were actually evaluated.

Important caveat: the canonical old NMDA Mamba report used the first `32`
validation traces, while the current CoreNEURON/Hay report uses `16` generated
validation traces. A first-`16` old-NMDA sensitivity evaluation was added for a
closer subset-size comparison, but it is not a replacement for a matched
fresh-seed study.

## Simulator / Teacher Statistics

| setup | eval traces | event fraction | spike / 1000 steps | spikes / trace mean+/-std | non-spiking frac | soma mean+/-std mV | soma min/max mV | abs dV/dt p95/p99 mV/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old NeuronIO/NMDA, canonical | 32 | `0.010554` | `0.9896` | `5.94 +/- 7.88` | `0.4375` | `-67.595 +/- 3.236` | `-80.38 / -55.00` | `0.4688 / 0.6875` |
| old NeuronIO/NMDA, first-16 sensitivity | 16 | `0.010563` | `1.2292` | `7.38 +/- 8.93` | `0.3750` | `-67.434 +/- 3.502` | `-80.38 / -55.00` | `0.4688 / 0.6875` |
| current CoreNEURON/Hay `exc=0.925` | 16 | `0.010600` | `0.8438` | `5.06 +/- 4.34` | `0.1250` | `-68.016 +/- 3.261` | `-77.23 / -55.00` | `0.4667 / 0.6706` |

Interpretation:

- Input density is very close across old and current eval subsets.
- Clipped voltage standard deviation and dV/dt percentiles are also close,
  which supports the claim that the current setup is in the old dynamic regime.
- The current subset is slightly more hyperpolarized than old NMDA
  (`-0.42 mV` vs old canonical mean; `-0.58 mV` vs old first-16 mean).
- Current CoreNEURON/Hay has lower spike rate than old NMDA canonical and much
  lower than the old first-16 sensitivity subset.
- Current spike counts are less variable and fewer traces are non-spiking. So
  the current teacher is similar in input density and clipped-voltage scale, but
  not identical in spike-count distribution.

## Mamba Model Metrics

All rows are Mamba-family checkpoints evaluated by the same NEURO-BENCH
M1/M1b/M2 evaluator. `80k` means `80,000` training presentations.

| setup / checkpoint | eval traces | RMSE mV | AUC | near-spike RMSE | spike-count MAE @0 | spike-count MAE matched | timing F1 matched | worst trace RMSE | sanity flags | plots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| old NMDA `mamba_compact80k`, canonical | 32 | `0.9286` | `0.9777` | `4.8703` | `5.4688` | `3.9375` | `0.2193` | `2.1097` | `threshold_zero_predicts_no_spikes` | yes |
| old NMDA `mamba_compact80k`, first-16 sensitivity | 16 | `0.9886` | `0.9757` | `4.7243` | `6.8750` | `4.8750` | `0.1838` | `2.1097` | `threshold_zero_predicts_no_spikes` | yes |
| current CoreNEURON/Hay `mamba_exc0925_80k` | 16 | `0.8763` | `0.9632` | `4.6860` | `4.7500` | `2.3750` | `0.1468` | `1.2450` | none | yes |
| current `longctx1000` 80k candidate | 16 | `0.8586` | `0.9631` | `4.4535` | `4.5000` | `2.7500` | `0.1070` | `1.2269` | none | yes |
| current `soma075` 80k candidate | 16 | `0.8507` | `0.9666` | `4.6228` | `4.8125` | `4.1250` | `0.1598` | `1.2070` | `threshold_zero_predicts_no_spikes` | yes |

Interpretation:

- Current CoreNEURON/Hay Mamba has lower RMSE than old NMDA Mamba in both
  comparisons: `0.8763` vs old canonical `0.9286`, and `0.8763` vs old
  first-16 `0.9886`.
- Current AUC is lower: `0.9632` vs old canonical `0.9777` and old first-16
  `0.9757`.
- Current baseline has cleaner sanity behavior at threshold zero and much
  lower worst-trace RMSE, but lower prevalence-matched spike-timing F1.
- The best current RMSE candidates improve voltage error further, but they do
  not close the AUC/timing gap; `soma075` also reintroduces the threshold-zero
  sanity flag.
- Therefore the current setup is better for voltage RMSE under the evaluated
  80k Mamba protocol, but old NeuronIO/NMDA still looks better for spike
  ranking/timing metrics.

## Training Throughput Note

The old NMDA `mamba_compact80k` run reported total training throughput
`180.34` samples/s, with per-epoch training around `275-285` samples/s. The
current CoreNEURON/Hay `mamba_exc0925_80k` run reported total throughput
`1077.20` samples/s, with per-epoch training around `860-1217` samples/s.
This supports the earlier note that our current data path is substantially
faster, but this is an engineering property, not a simulator-fidelity metric.

## Bottom Line

The current CoreNEURON/Hay setup is close to the old NeuronIO/NMDA teacher in
input density, clipped-voltage standard deviation, and dV/dt scale. It differs
meaningfully in spike distribution: lower spike rate than old canonical NMDA,
less spike-count variance, and fewer non-spiking traces. Mamba voltage RMSE is
better on the current setup, but spike AUC and timing are worse. For paper use,
we should not frame the current setup as a full statistical replication of
NeuronIO/NMDA; the defensible claim is that it is a source-Hay/CoreNEURON
replacement in a similar clipped-voltage dynamic regime, with a different
spike distribution that should be reported explicitly.
