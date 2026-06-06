# V1 Teacher-QC Pilot

Date: 2026-06-06

Purpose: run the first v1 teacher sanity pilot over the provisional `N=5`
morphology set before benchmark implementation and before any full dataset
generation.

## Setup

- Backend used for successful pilot: NEURON/Hay via `.venv/bin/python`.
- Samples: `8` paired traces per morphology/intervention.
- Morphologies: `488683425`, `487664663`, `479770916`, `589427435`,
  `488679042`.
- Interventions: `event_dropout`, `inh_dropout`.
- Routing: `normalized_by_type_arclength`.
- Baseline policy: simulate baseline under requested morphology/template.
- Output payloads are in ignored directory
  `docs/verify-arbor-simulator/artifacts/v1_teacher_qc_pilot_20260606/`.

## Results

All `10` morphology/intervention pairs produced a detected response in the
expected direction.

| specimen | intervention | traces | segments | delta soma mV | delta spike rate / 1000 | spike-count delta | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `479770916` | `event_dropout` | 8 | 302 | -1.0742 | -3.6136 | -19.8750 | lower drive / lower spiking / hyperpolarized |
| `479770916` | `inh_dropout` | 8 | 302 | 1.1778 | 6.3182 | 34.7500 | lower inhibitory drive / higher spiking / depolarized |
| `487664663` | `event_dropout` | 8 | 265 | -1.8843 | -1.2500 | -6.8750 | lower drive / lower spiking / hyperpolarized |
| `487664663` | `inh_dropout` | 8 | 265 | 1.5334 | 4.0000 | 22.0000 | lower inhibitory drive / higher spiking / depolarized |
| `488679042` | `event_dropout` | 8 | 377 | -1.3678 | -2.5455 | -14.0000 | lower drive / lower spiking / hyperpolarized |
| `488679042` | `inh_dropout` | 8 | 377 | 1.3055 | 5.4545 | 30.0000 | lower inhibitory drive / higher spiking / depolarized |
| `488683425` | `event_dropout` | 8 | 283 | -1.7704 | -1.7955 | -9.8750 | lower drive / lower spiking / hyperpolarized |
| `488683425` | `inh_dropout` | 8 | 283 | 1.9357 | 4.0682 | 22.3750 | lower inhibitory drive / higher spiking / depolarized |
| `589427435` | `event_dropout` | 8 | 340 | -2.2319 | -0.6818 | -3.7500 | lower drive / lower spiking / hyperpolarized |
| `589427435` | `inh_dropout` | 8 | 340 | 2.5304 | 2.6364 | 14.5000 | lower inhibitory drive / higher spiking / depolarized |

## Interpretation

This supports keeping the provisional `N=5` morphology set for the next
benchmark implementation pass. It is not a full teacher freeze: the pilot
checks intervention polarity and rough dynamic sanity, but the benchmark still
needs dataframe-first teacher-QC diagnostics for source-regime statistics,
raw-vs-clipped distortion, routing metadata, and target-view semantics.

## Backend Note

The first smoke used the wrong NEURON environment and failed because
`hay_swc_template/x86_64/libnrnmech.so` was compiled for NEURON `9.0.1` while
the conda environment had NEURON `8.2.7`. Using `.venv/bin/python` fixed the
plain NEURON/Hay smoke. CoreNEURON with the SWC template still aborted under
the matching `.venv`, so the current correctness pilot uses plain NEURON/Hay.

Full dataset generation must not use this slow correctness-only path by
default. The generation-throughput gate must resolve GPU/batched simulation or
produce a measured, defensible backend choice before full data generation.

