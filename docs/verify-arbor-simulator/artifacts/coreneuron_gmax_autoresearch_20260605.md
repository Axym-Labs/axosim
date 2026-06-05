# CoreNEURON/Hay Conductance Autoresearch

Date: 2026-06-05

## Question

Can we reduce RMSE without weakening simulator defensibility by calibrating
deterministic synaptic conductance instead of deleting input events?

This is a plausible lever because the CoreNEURON-compatible path replaces the
original probabilistic/STP synapse MOD files with deterministic AMPA/NMDA/GABA
subsets. A small efficacy calibration preserves the stimulus distribution,
unlike event-retention calibration.

## Teacher Selection

Selection used teacher statistics only, not model RMSE. Target statistics were
the historical NMDA validation aggregate: spike rate `0.9896 Hz`, clipped soma
mean/std `-67.595/3.236 mV`, and clipped `|dV/dt|` p95/p99 `0.438/0.687`.

| rank | candidate | objective | exc keep | exc gmax | inh gmax | event frac | spike Hz | mean/std mV | dV/dt p95/p99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `gexc0925` | `0.5746` | `1` | `0.925` | `1` | `0.010959` | `0.8958` | `-68.2684/3.3319` | `0.4465/0.6884` |
| 2 | `keep0925` | `0.6778` | `0.925` | `1` | `1` | `0.010504` | `0.9062` | `-68.2931/3.3353` | `0.4588/0.6996` |
| 3 | `gexc095_ginh105` | `0.7261` | `1` | `0.95` | `1.05` | `0.010959` | `0.9687` | `-68.2714/3.3611` | `0.4593/0.7279` |
| 4 | `gexc095` | `0.7468` | `1` | `0.95` | `1` | `0.010959` | `1.0833` | `-67.9349/3.4227` | `0.4616/0.7427` |
| 5 | `gexc090` | `0.9684` | `1` | `0.9` | `1` | `0.010959` | `0.6979` | `-68.5843/3.2673` | `0.4350/0.6576` |
| 6 | `ginh110` | `1.1577` | `1` | `1` | `1.1` | `0.010959` | `1.2083` | `-67.9640/3.4410` | `0.4836/0.7918` |
| 7 | `ginh105` | `1.4089` | `1` | `1` | `1.05` | `0.010959` | `1.3958` | `-67.6553/3.4933` | `0.4874/0.8504` |

## Model Evaluation

Both evaluated model runs used Mamba, CUDA, `80k` presentations, `32` training
traces, `16` validation traces, and the current NEURO-BENCH M1/M1b/M2 report.

| setting | teacher rationale | RMSE mV | AUC | near-spike RMSE | count MAE matched | timing F1 matched | sanity flags | plots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gexc0925` | best teacher objective; preserves events | `0.9769` | `0.9564` | `5.2105` | `2.7500` | `0.1627` | none | yes |
| `gexc095` | hotter small-scale candidate, plausible RMSE bet | `1.0438` | `0.9592` | `4.8532` | `2.7500` | `0.1254` | none | yes |
| prior `keep0925` | current provisional default | `0.8763` | `0.9632` | `4.6860` | `4.7500` | `0.1468` | none | yes |

## Decision

Do not switch the default to conductance scaling for metric reasons. The best
teacher-stat conductance candidate is defensible and input-policy cleaner, but
it did not improve RMSE under the bounded `80k` Mamba protocol. The existing
`exc_event_keep_prob=0.925` default remains the current metric default, with
the caveat that it should be framed as matching the teacher distribution rather
than as an innocuous no-op.

The useful conclusion is negative but actionable: small scalar conductance
calibration is not an obvious free lunch. Future simulator improvements should
look at higher-fidelity deterministic synapse translation, richer seed splits,
or model/training changes, not more blind scalar sweeps.
