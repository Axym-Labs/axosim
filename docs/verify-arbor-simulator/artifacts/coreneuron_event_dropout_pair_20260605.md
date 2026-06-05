# CoreNEURON/Hay Event Dropout Teacher Pair

Date: 2026-06-05

## Setup

- Source data: `data/hay-coreneuron-source-val16-exc0925-20260605/val`.
- Teacher backend: CoreNEURON/Hay CPU path.
- Samples: `4` validation traces.
- Intervention: drop a Bernoulli `35%` subset of nonzero input events, then
  re-simulate the same Hay/CoreNEURON teacher on the intervened inputs.
- Target policy: same clipped/normalized soma and local-peak spike labels as
  the accepted CoreNEURON/Hay dataset.

## Teacher Response

| metric | value |
| --- | ---: |
| baseline event fraction | `0.010676` |
| intervention event fraction | `0.006932` |
| event fraction ratio | `0.6493` |
| delta spike rate / 1000 steps | `-0.6818` |
| delta soma mean mV | `-2.3395` |
| delta absolute soma mean mV | `2.3468` |
| direction | `lower_event_drive/lower_spiking/hyperpolarized` |
| response detected | `true` |

Interpretation: this is a usable MVP M4 teacher pair for the current
CoreNEURON/Hay path. The direction is qualitatively expected: lower event
drive hyperpolarizes the teacher and lowers spiking. The pair is small, so it
is a smoke benchmark rather than a final robustness claim.

## Artifacts

- Ignored pair directory:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605/`.
- Pair file:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605/event_dropout_pair.npz`.
- Detailed ignored report:
  `docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605/event_dropout_report.md`.
