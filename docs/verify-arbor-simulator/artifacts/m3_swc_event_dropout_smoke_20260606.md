# M3 Held-Out SWC Teacher Smoke

Date: 2026-06-06 12:56 CEST

## Question

Can the provisional SWC-aware Hay/NEURON path generate a structured teacher
pair for a held-out Allen L5 morphology, and does the teacher response look
consistent enough to support an M3 morphology-transfer benchmark?

## Setup

- Source/control morphology: original Hay `cell1.asc`.
- Held-out morphology: Allen `specimen_487664663.swc`.
- Backend: CPU NEURON with the local SWC-aware Hay template and NEURON 9
  compiled MOD files.
- Intervention: `35%` event dropout on the same first `8` validation traces.
- Source input protocol: `639` excitatory and `639` inhibitory source sites.

## Routing Policy

The runner now records the source-site-to-morphology routing policy.

| morphology | morphology segments | used source sites/sign | unused source sites/sign | policy |
| --- | ---: | ---: | ---: | --- |
| original `cell1.asc` | `639` | `639` | `0` | ordered source site routing |
| held-out `specimen_487664663.swc` | `265` | `265` | `374` | ordered source site truncation |

For the held-out SWC, this is a runtime smoke policy, not a publishable final
M3 routing protocol. It preserves source order but drops many source sites.

## Teacher Response

| morphology | delta soma mean mV | delta spike rate / 1000 steps | spike-count delta mean | direction |
| --- | ---: | ---: | ---: | --- |
| original `cell1.asc` | `-2.3968` | `-0.6136` | `-3.3750` | lower event drive / lower spiking / hyperpolarized |
| held-out `specimen_487664663.swc` | `-1.7247` | `+0.9773` | `+5.3750` | lower event drive / higher spiking / hyperpolarized |

## Interpretation

Runtime compatibility is achieved: the held-out SWC instantiates, integrates,
accepts deterministic synapses, and produces paired intervention targets.

Correctness replication is not achieved yet. The source morphology control
preserves the expected event-dropout response under the same runner, while the
held-out SWC hyperpolarizes but spikes more. Because the held-out route uses
only `265/639` source sites per sign, this result is best treated as evidence
that M3 needs a morphology-normalized routing policy before model transfer
scores are publishable.

## Next Defensible Path

Use this implementation as the M3 runtime scaffold, but do not claim M3
teacher correctness until one of these is implemented and sanity-checked:

- morphology-normalized synapse placement over apical/basal arclength;
- matched per-branch/site sampling with comparable E/I density across
  morphologies;
- or a restricted held-out morphology set whose segment/site count closely
  matches the source protocol.
