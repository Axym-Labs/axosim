# NEURO-BENCH Axis Composition Profile

Date: 2026-06-06

## Decision

Use an explicit benchmark profile layer instead of hard-coding the cartesian
product of metrics, states, interventions, and morphologies in experiment
scripts.  The importable default is `paper-default`; it can be inspected with:

```bash
PYTHONPATH=src python -m bnn_sim.cli benchmark-profile --name paper-default --plan
```

## Paper-Default Axis Policy

| Axis area | Default | Rationale |
| --- | ---: | --- |
| Source traces | 64 | Main in-distribution M1/M1b/M2/state/trace-shape estimate. |
| Metric x state product | Dense: 5 metric suites x 7 states | This is cheap once predictions exist and is central to behavior diagnosis. |
| Morphology transfer | 15 morphologies x 16 traces | M3 is a headline generalization axis; the earlier `N=3` suite is only a smoke/initial validation result. |
| Source interventions | 5 interventions x 16 traces | M4 behavior is measured as paired teacher/model response deltas. |
| Morphology-intervention sanity | 5 morphologies x 2 interventions x 8 traces | Avoids the full 15 x 5 morphology/intervention product while checking that intervention direction remains meaningful under transfer. |
| Training budget | <=80k presentations | Matches the current paper constraint; no metric run should silently exceed it. |

## What Interventions Measure

Interventions are not plain M1 repeats by default.  They measure paired
teacher/model deltas:

- teacher-response sanity and direction;
- soma delta RMSE and gain;
- spike-count delta MAE and gain;
- direction preservation;
- representative trace IDs for inspection.

Full M1/M1b/M2/state metrics on intervention traces remain an extended
analysis mode, not the default.  This keeps the benchmark tractable and keeps
M4 focused on whether the surrogate preserves the teacher's response to a
controlled perturbation.

## CPU/GPU Simulator Policy

The current default single-cell teacher backend is `coreneuron_hay_cpu`.  This
does not mean CPU is always faster.  It means our local CoreNEURON/Hay
validation already showed CPU and GPU matching to floating-point noise, while
the single-cell GPU path was slower on this machine (`8 x 500 ms`: CPU
`24.9 s`, GPU `131.3 s`) because launch/setup overhead and single-cell
under-occupancy dominate.

The GPU path is therefore a validated future-scaling path, not the default for
small single-cell generation.  Before switching the standalone package default
to GPU, run a dedicated batch/cell-scale benchmark that records CPU/GPU
equivalence, elapsed time, GPU utilization, and whether CoreNEURON is actually
using the GPU backend.

## Package Implication

The profile API is path-free and package-facing:

- `bnn_sim.benchmark.get_benchmark_profile(name)`
- `bnn_sim.benchmark.build_axis_plan(profile)`
- `bnn_sim.benchmark.list_benchmark_profiles()`

This gives the future standalone benchmark package a clean place to expose
default budgets while letting local scripts decide which datasets, checkpoints,
and simulator runners instantiate the plan.
