# CoreNEURON/Hay Event Dropout Teacher Pair16

Date: 2026-06-06

## Setup

- Source data: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
- Teacher backend: CoreNEURON/Hay CPU path.
- Samples: all `16` validation traces.
- Intervention: drop a Bernoulli `35%` subset of nonzero input events, then
  re-simulate the same Hay/CoreNEURON teacher on the intervened inputs.
- Target policy: same clipped/normalized soma and local-peak spike labels as
  the accepted CoreNEURON/Hay dataset.
- Generation elapsed time: `401.0 s`.

## Teacher Response

| metric | value |
| --- | ---: |
| baseline event fraction | `0.010626` |
| intervention event fraction | `0.006907` |
| event fraction ratio | `0.6500` |
| delta spike rate / 1000 steps | `-0.8409` |
| delta soma mean mV | `-2.4102` |
| delta absolute soma mean mV | `2.4246` |
| direction | `lower_event_drive/lower_spiking/hyperpolarized` |
| response detected | `true` |

Interpretation: this expanded M4 teacher pair is qualitatively expected and
stronger than the initial `4`-trace smoke. Lower event drive hyperpolarizes the
teacher and substantially lowers spiking across the full validation subset.

## Artifacts

- Ignored pair directory:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair16_20260605/`.
- Pair file:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair16_20260605/event_dropout_pair.npz`.
- Detailed ignored report:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair16_20260605/event_dropout_report.md`.
