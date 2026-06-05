# CoreNEURON Hay Port Probe

Date: 2026-06-05

## Decision

Use CoreNEURON as the current NEURON-family replacement path for the paper
benchmark, and leave Arbor for the population-level follow-up. CoreNEURON keeps
the Hay morphology/HOC/ion-channel lineage and gives a credible path to
GPU-parallel execution without requiring an Arbor reimplementation under the
current time constraint.

## What Was Tested

- Source morphology/HOC: `/tmp/neuron-as-deep-net/L5PC_NEURON_simulation`
  (`cell1.asc`, `L5PCbiophys5b.hoc`, `L5PCtemplate_2.hoc`).
- Ion channels: original Hay active/passive MOD files, compiled for both NEURON
  and CoreNEURON under this artifact directory.
- Synapses: deterministic `DetAMPANMDA` and `DetGABA` subsets of the original
  probabilistic synapse mechanisms. This matches the current source-teacher
  use case where `Use=1`, `Dep=0`, `Fac=0`, `u0=0`, so events are effectively
  always released and the RNG path is not a meaningful model ingredient.
- Event delivery: `PatternStim` plus `ParallelContext.gid_connect`. Direct
  scheduled `NetCon(None).event(...)` is not CoreNEURON-compatible for this
  probe.
- Integration: fixed-step `dt=0.025 ms`, `CVode().cache_efficient(1)`,
  low-resolution output at `1 ms`.

## Compile Evidence

- The original stochastic synapse MOD files do not compile cleanly for
  CoreNEURON because they use `POINTER`, `VERBATIM`, and non-thread-safe RNG
  paths. Treating those files as the GPU path would be incorrect.
- The deterministic synapse subset and original Hay channel MOD files compile
  for NEURON and CPU CoreNEURON:
  - `x86_64/libnrnmech.so`
  - `x86_64/libcorenrnmech.so`
  - `x86_64/special`
  - `x86_64/special-core`
- GPU update: the installed conda NEURON `8.2.7` package is CPU-only
  (`CORENRN_ENABLE_GPU OFF`). A source NEURON `8.2.7` GPU build configured
  successfully with NVIDIA HPC SDK but failed device linking on
  `coreneuron::celsius` for Hay ion mechanisms. The validated GPU path is
  therefore NEURON `9.0.1` from source with NVIDIA HPC SDK `26.3`, CUDA
  `13.1`, and the RTX `5090`.
- The NEURON `9.0.1` GPU path compiles the deterministic Hay AMPA/NMDA/GABA
  generator. Unused `epsp.mod` is excluded because it fails 9.x NMODL
  translation and is not part of the deterministic source-teacher path.
- Local toolchain note: CUDA `13.1` on this Ubuntu/GCC/glibc stack required a
  local NVIDIA SDK header workaround for `rsqrt`/`rsqrtf` exception-spec
  declarations. The patched file is under the user-local NVIDIA SDK install,
  not in this repository.

## Numerical Correctness

First, fixed-step NEURON was compared with the original CVode source-teacher
path on the same 2x3 s probe. Fixed-step changed spike rate by `0.0 Hz`; the
main clipped-voltage differences were tiny:

| metric | fixed - CVode |
| --- | ---: |
| clipped soma mean | `-0.002714 mV` |
| clipped soma std | `-0.002672 mV` |
| clipped `|dV/dt|` p95 | `-0.000897 mV/step` |
| clipped `|dV/dt|` p99 | `-0.002512 mV/step` |

Then NEURON fixed-step and CoreNEURON fixed-step were compared with identical
inputs and deterministic synapses:

| probe | spikes? | max aggregate diff | max row diff |
| --- | --- | ---: | ---: |
| `1x100ms` | no | `0.0` | `0.0` |
| `1x1s` | no | `0.0` | `0.0` |
| `1x3s` | no | `0.0` | `0.0` |
| `2x3s` | yes | `0.0` | `0.0` |

The spiking `2x3s` probe matched exactly and produced a NeuronIO-like dynamic
regime:

| metric | value |
| --- | ---: |
| event fraction | `0.010593` |
| spike count mean/std | `6.5 / 4.5` |
| spike rate mean/std | `2.1667 / 1.5 Hz` |
| clipped soma mean/std | `-67.9793 / 3.7700 mV` |
| clipped `|dV/dt|` p95/p99 | `0.5085 / 0.8386 mV/step` |

NEURON `9.0.1` CPU and GPU CoreNEURON were then compared on the same
deterministic mechanism set with `8 x 500 ms` seeded traces:

| metric | value |
| --- | ---: |
| CPU elapsed | `24.85 s` |
| GPU elapsed | `131.30 s` |
| max spike/event delta | `0.0` |
| max soma-stat delta | `7.6e-11 mV` |
| max other voltage-summary delta | `3.4e-12` |

The GPU run reports one RTX `5090` device and uses CoreNEURON GPU execution.
The single-cell GPU path is slower than CPU at this scale, so the evidence is
correctness evidence and a path to population-level scaling, not a speed claim
for one cell.

## Interpretation

This is a successful simulator-correctness replication for the NEURON-family
path: CoreNEURON reproduces the fixed-step Hay/NEURON data generator exactly
on the tested traces, including spiking traces. GPU execution is now validated
for the NEURON `9.0.1` deterministic-Hay subset and matches CPU summaries to
floating-point noise. The exact-version NEURON `8.2.7` GPU path is not
validated and should not be claimed.

The next implementation step is to turn this probe into the replacement dataset
generator, preserving the source-Hay target policy and capping model metric
runs at `80k` presentations for now.
