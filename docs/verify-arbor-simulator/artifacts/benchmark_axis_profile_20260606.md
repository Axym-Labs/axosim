# NEURO-BENCH Axis Composition Profile

Date: 2026-06-06

## Decision

Use an explicit benchmark profile layer instead of hard-coding the cartesian
product of metrics, states, interventions, and morphologies in experiment
scripts.  The profile values are provisional, capacity-audited hypotheses,
not final truths.  The importable default candidate is `paper-default`; it can
be inspected with:

```bash
PYTHONPATH=src python -m bnn_sim.cli benchmark-profile --name paper-default --plan
```

The local cost estimate can be inspected with:

```bash
PYTHONPATH=src python -m bnn_sim.cli benchmark-profile --name paper-default --cost
```

## Heuristic Basis

This profile is now framed as a staged, budget-constrained experimental design:

- Dense metric/state evaluation is retained after predictions exist because it
  is cheap and diagnoses behavior-specific failures.
- Expensive simulator axes use staged/fractional coverage instead of a full
  factorial grid.
- Morphology transfer is treated as a main benchmark task, analogous to
  evaluating across multiple datasets/environments, not as a single smoke
  example.
- Interventions are paired-delta probes; they are closer to controlled
  screening experiments than to ordinary IID held-out trace metrics.
- Report uncertainty intervals and rank stability because small benchmark
  budgets can make model rankings fragile.

Relevant external anchors:

- Demsar 2006 recommends robust non-parametric comparisons over multiple data
  sets for ML algorithm comparison, supporting the idea that morphology/task
  breadth matters for model ranking:
  <https://www.jmlr.org/papers/v7/demsar06a.html>
- Fractional/orthogonal experimental-design ideas are used to improve sample
  efficiency when full grids are too expensive; MOFA is one ML/HPO example
  using orthogonal Latin hypercube and fractional-factorial analysis:
  <https://openreview.net/forum?id=OpUJ46CNv43>
- Method-comparison literature argues for combining benchmark datasets and
  simulations rather than treating either alone as sufficient:
  <https://arxiv.org/abs/2208.01457>
- Recent RL benchmarking analysis shows that aggregate rankings can need many
  seeds/environments for tight confidence intervals; our benchmark should
  therefore expose uncertainty rather than overclaiming from small profiles:
  <https://people.cs.umass.edu/~pthomas/papers/Jordan2024b.pdf>

## Paper-Default Axis Policy

| Axis area | Default candidate | Rationale |
| --- | ---: | --- |
| Source traces | 64 | Main in-distribution M1/M1b/M2/state/trace-shape estimate. |
| Metric x state product | Dense: 5 metric suites x 7 states | This is cheap once predictions exist and is central to behavior diagnosis. |
| Morphology transfer | 15 morphologies x 16 traces | M3 is a headline generalization axis; the earlier `N=3` suite is only a smoke/initial validation result. |
| Source interventions | 5 interventions x 16 traces | M4 behavior is measured as paired teacher/model response deltas. |
| Morphology-intervention sanity | 5 morphologies x 2 interventions x 8 traces | Avoids the full 15 x 5 morphology/intervention product while checking that intervention direction remains meaningful under transfer. |
| Training budget | <=80k presentations | Matches the current paper constraint; no metric run should silently exceed it. |

## Capacity Audit

Local cost model: `local_coreneuron_hay_cpu_20260606`.

Assumptions:

- Source validation traces are cached, so source teacher generation is not part
  of the default evaluation cost.
- Source intervention teacher generation costs about `25 s` per 6 s trace,
  anchored to the measured `16`-trace event-dropout run (`401 s`).
- Held-out morphology baseline+intervention generation costs about `28 s` per
  trace pair, anchored to the measured `8`-trace held-out morphology run
  (`223 s`).
- A held-out morphology baseline-only trace is estimated at half that pair
  cost (`14 s`).
- Model prediction/metric work is estimated at `0.15 s` per trace per model.

Current `paper-default` estimate with `3` models:

| Component | Estimated seconds |
| --- | ---: |
| Cached source distribution | `0` |
| Morphology baselines | `3360` |
| Source interventions | `2000` |
| Morphology-intervention sanity | `2240` |
| Model eval and metrics | `280.8` |
| Total | `7880.8` (`2.19 h`) |

The target sequential wall-clock budget is `4 h`, so the current default
candidate is feasible under the local estimate.  The selected morphology-
intervention slice avoids `520` expensive trace pairs relative to the full
`15 x 5 x 8` morphology/intervention grid, about `4.04` local teacher-hours.
The recommended local parallelism is about `5` independent teacher jobs.

This audit is a guardrail, not proof.  The standalone package should replace
these constants with measured local timings and should fail or warn when a
chosen profile exceeds a declared capacity budget.

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
and simulator runners instantiate the plan.  The package should expose the
profile status as provisional until a final capacity and uncertainty analysis
is completed.
