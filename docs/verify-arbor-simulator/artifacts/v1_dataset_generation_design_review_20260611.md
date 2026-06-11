# v1 Dataset Generation Gate and Design Review - 2026-06-11

## Claim Under Review

The v1 dataset can be generated with the actual SWC morphology CoreNEURON GPU
path without falling back to CPU, while preserving the intended dataset
semantics: `5` morphologies, `100,000` train traces, `37,500` ordinary
non-train traces, and `12,500` paired intervention examples total.

## Throughput Gate

All measurements used actual v1 SWC morphology `specimen_488683425`, duration
`6000 ms`, durable NEURON `9.0.1` CoreNEURON GPU runtime, and the durable SWC
mechanism root.

| case | batch | semantic units/hour | simulation traces/hour | max RSS |
| --- | ---: | ---: | ---: | ---: |
| split | 8 | 193.1 | 193.1 | 0.98 GB |
| split | 16 | 343.1 | 343.1 | 1.20 GB |
| split | 32 | 632.0 | 632.0 | 1.65 GB |
| split | 64 | 1089.1 | 1089.1 | 2.56 GB |
| split | 128 | 1609.1 | 1609.1 | 4.34 GB |
| paired `event_dropout` | 32 | 337.5 pairs/hour | 674.9 trace-sims/hour | 2.40 GB |
| paired `event_dropout` | 64 | 601.8 pairs/hour | 1203.5 trace-sims/hour | 4.05 GB |

Measured launch estimate using split batch `128` and intervention batch `64`:

- ordinary traces: `137,500` -> `85.45 h` at the reference morphology rate;
- paired examples: `12,500` -> `20.77 h` at the reference morphology rate;
- reference-morphology total: `106.23 h` (`4.43 days`).

The five v1 morphologies have routed segment counts `265`, `283`, `302`,
`340`, and `377`. Because the throughput gate used the `283`-segment reference
morphology, a conservative segment-count correction puts the expected full run
closer to roughly `5-6 days`. That estimate is acceptable but not loose.

## Critical Design Review

Supported:

- GPU correctness: actual SWC split and paired workers matched CPU
  CoreNEURON exactly in prior checks, and full-trace GPU workers now generate
  valid outputs.
- Batch sizing: split batch `128` is the best measured ordinary-trace point
  with modest memory use; paired batch `64` is validated and avoids relying on
  untested pair batch `128`.
- Parallelism: one GPU worker already saturates the GPU during solve. Use
  `--parallelism 1` for the full run to avoid multi-process GPU contention.
- Storage: compressed shard size is far below the `<200 GB` cap on the local
  filesystem, and the working disk has enough headroom.

Corrected before full generation:

- Intervention-count semantics: the dataset target is `2500` paired examples
  per morphology across all interventions, not `2500` per intervention. The
  generator now supports `--intervention-total-samples-per-morphology`; the
  full run should use `2500`, which distributes to `500` examples per
  intervention for the five selected interventions.
- Shard sizing: the generator now supports
  `--intervention-samples-per-shard`, so ordinary and paired workloads can use
  their measured-safe shard sizes.
- Resume safety: the generator now supports `--resume` and skips existing
  valid shard+metadata pairs, which is necessary for a multi-day run.
- Worker paths: hidden worker output paths are resolved before NEURON changes
  cwd to the mechanism root, preventing relative-path shards from being written
  under the mechanism directory.

Remaining risks:

- Throughput was measured deeply on one morphology and smoke-tested on all
  five. Segment-count differences imply slower morphologies; the run should be
  monitored after the first full shards of the larger morphologies.
- HOC template code prints branch-length diagnostics to stdout, so scripts
  should read `generation_summary.json` rather than parse stdout as JSON.
- `temporal_jitter` and `site_silence` intervention teacher-response quality
  still needs dataset-level QC after generation; this does not block
  generation, but it can affect which intervention claims are paper-facing.

## Launch Decision

Proceed with full generation using:

- output root: `data/v1-neuronio-like-gpu-20260611`;
- `--backend coreneuron --coreneuron-gpu`;
- `--parallelism 1`;
- `--samples-per-shard 128`;
- `--intervention-samples-per-shard 64`;
- `--train-samples-per-morphology 20000`;
- `--val-samples-per-morphology 2500`;
- `--test-samples-per-morphology 2500`;
- `--private-test-samples-per-morphology 2500`;
- `--intervention-total-samples-per-morphology 2500`;
- `--resume`.
