# AxoMamba Model Autoresearch Specification

## Objective

Iteratively improve the current best AxoMamba model on AxoBench v1 through a
diagnostics-driven autoresearch loop. The work is model-focused; the accepted
teacher dataset and evaluation definitions are fixed inputs.

## Required Setup

1. Directly measure batch-size behavior on the exact experimental setup before
   selecting the new higher-batch default. Report presentation count,
   optimizer-step count, wall time, and core diagnostics.
2. Before iteration 1, ablate every auxiliary addon in the promoted AxoMamba
   stack: branch gain, branch tanh, pre-Mamba short/long integrative mixer, and
   multiplicative soma gate. Use matched data, initialization policy, budget,
   and evaluation.
3. Provide one compact experiment runner for the standard 120k-presentation
   setup and a 300k long option used on every tenth iteration. Keep the CLI
   small; candidate details belong in a manifest.
4. Run one or more attributable candidates per iteration. Use concurrency or a
   shared in-process panel only when measured iteration latency benefits.
5. Freeze the 120k presentation schedule and share loading/tensor transfer
   across candidates trained together. Optimize loading only when profiling
   shows a material gain.
6. Run AxoBench diagnostics for every new run against the current baseline.
   Cache the baseline evaluation in an output artifact instead of rerunning
   baseline inference. Save all diagnostic outputs under a dedicated directory.
7. Log every iteration in `autoresearch_progress.md`, including hypothesis,
   configuration, raw metrics, diagnostic interpretation, failure triggers,
   and keep/discard decision.
8. Every tenth iteration uses the 300k budget and also prunes dead code and
   rejected modifications, with tests before and after cleanup.

## Optimization Targets

Primary targets, in order of practical emphasis:

- voltage/dynamics SERA;
- F1 across the diagnostic time-tolerance curve;
- intervention response SERA.

Secondary evidence includes the remaining AxoBench attention/failure-mode
panel, runtime, memory, firing/count calibration, morphology disparity, and
numerical/physiological validity. One overly restrictive failure trigger may be
lightened only with a recorded rationale and before using the relaxed trigger
to accept a candidate.

## Search Policy

- Start high-temperature: objective alignment, output parameterization,
  temporal architecture, routing/aggregation, and other changes likely to
  materially alter the setup.
- Move to low-temperature tuning only after the main structural hypotheses are
  tested.
- The first objective candidates should train directly toward SERA and toward
  a differentiable analogue of F1 across time, consistent with the paper's
  revised evaluation critique.
- Do not run blind modifications without a plausible direct link to a core
  metric. Metric optimization is allowed and encouraged; fast failure-mode
  diagnostics guard against pathological solutions.

## Experimental Validity

- Matched comparisons use the same frozen examples, seed policy, budget,
  preprocessing, and diagnostics.
- Parameter counts, optimizer steps, and compute are recorded. Unmatched
  comparisons are labeled and are not used for causal addon claims.
- The addon ablation uses one-at-a-time removals from the full promoted stack.
- Candidate promotion requires core-metric evidence without a material new
  failure trigger. Negative and inconclusive results remain in the ledger.
- Diagnostics are run on held-out validation/intervention traces; training
  schedules never include them.

## Deliverables

- reusable experiment runner and tests;
- cached-baseline diagnostics support and tests;
- frozen sample schedule and measured loading/concurrency policy;
- pre-loop batch-size benchmark and addon-ablation table;
- `autoresearch_progress.md` plus machine-readable result artifacts;
- ongoing retained AxoMamba configuration and checkpoints.
