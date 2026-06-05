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
- The local NEURON Python package exposes CoreNEURON GPU controls, but this
  machine currently lacks `nvc++`, so GPU MOD compilation/execution was not
  validated in this pass.

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

## Interpretation

This is a successful simulator-correctness replication for the NEURON-family
path: CoreNEURON reproduces the fixed-step Hay/NEURON data generator exactly
on the tested traces, including spiking traces. It is not yet a completed GPU
benchmark run because the machine is missing the NVIDIA HPC SDK compiler needed
for GPU mechanism compilation.

The next implementation step is to turn this probe into the replacement dataset
generator, preserving the source-Hay target policy and capping model metric
runs at `80k` presentations for now.
