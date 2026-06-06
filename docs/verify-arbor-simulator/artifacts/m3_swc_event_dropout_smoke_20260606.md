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
- Follow-up: valid simulated-baseline normalized-routing event dropout and
  inhibitory dropout checks on the same held-out SWC.
- Source input protocol: `639` excitatory and `639` inhibitory source sites.

## Routing Policies

The runner now records the source-site-to-morphology routing policy.

| morphology | morphology segments | used source sites/sign | unique source sites/sign | unused source sites/sign | policy |
| --- | ---: | ---: | ---: | ---: | --- |
| original `cell1.asc` | `639` | `639` | `639` | `0` | ordered source site routing |
| held-out `specimen_487664663.swc` | `265` | `265` | `265` | `374` | ordered source site truncation |
| held-out `specimen_487664663.swc` | `265` | `265` | `264` | `375` | normalized basal/apical arclength routing |

The normalized routing maps each target basal/apical segment to the nearest
source site in the same compartment by normalized arclength. It still uses
fewer source sites because the target morphology has fewer segments, but it no
longer drops a contiguous suffix of the source route.

## Source-Baseline Diagnostic

These rows compare held-out SWC interventions against source-morphology
baseline targets. They are invalid for M3 transfer correctness, but useful for
showing why the runner needed a simulated-baseline mode.

| morphology/routing | traces | delta soma mean mV | delta spike rate / 1000 steps | spike-count delta mean | direction |
| --- | ---: | ---: | ---: | ---: | --- |
| original `cell1.asc`, ordered | `8` | `-2.3968` | `-0.6136` | `-3.3750` | lower event drive / lower spiking / hyperpolarized |
| held-out SWC, ordered truncation | `8` | `-1.7247` | `+0.9773` | `+5.3750` | lower event drive / higher spiking / hyperpolarized |
| held-out SWC, normalized by type | `8` | `-1.7377` | `-0.0227` | `-0.1250` | lower event drive / lower spiking / hyperpolarized |

## Valid Simulated-Baseline Follow-Up

Both baseline and intervention targets in this table are simulated under the
held-out SWC morphology, the same normalized basal/apical arclength routing,
and the same mechanism/template path.

| intervention | traces | delta soma mean mV | delta spike rate / 1000 steps | spike-count delta mean | spike-count interval | direction |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| event dropout | `8` | `-1.8843` | `-1.2500` | `-6.8750` | `[-9.1109, -4.6391]` | lower event drive / lower spiking / hyperpolarized |
| inhibitory dropout | `8` | `+1.5334` | `+4.0000` | `+22.0000` | `[18.5056, 25.4944]` | lower event drive / higher spiking / depolarized |

## Interpretation

Runtime compatibility is achieved: the held-out SWC instantiates, integrates,
accepts deterministic synapses, and produces paired intervention targets.

Correctness replication is substantially improved after fixing the baseline
comparison. The valid simulated-baseline checks show the expected signs for
both event dropout and inhibitory dropout: reducing event drive hyperpolarizes
and lowers spiking, while reducing inhibitory drive depolarizes and increases
spiking. This supports normalized routing as a plausible M3 teacher-generation
scaffold.

The result is still not a completed publishable transfer benchmark: it covers
one held-out morphology, `8` traces, two interventions, and a dendrite-only
Allen reconstruction with Hay source-fit active parameters.

## Next Defensible Path

Use this implementation as the M3 runtime scaffold, but do not claim M3
teacher correctness until normalized routing is expanded and sanity-checked:

- matched per-branch/site sampling with comparable E/I density across
  morphologies;
- stronger teacher-response checks across more traces and at least one more
  intervention type;
- or a restricted held-out morphology set whose segment/site count closely
  matches the source protocol.
