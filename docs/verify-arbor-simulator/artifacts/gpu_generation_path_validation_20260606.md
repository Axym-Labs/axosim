# GPU Generation Path Validation - 2026-06-06

## Durable Runtime

- Built and installed durable NEURON `9.0.1` + CoreNEURON GPU runtime at `/home/davwis/.local/neuron-9.0.1-gpu-runtime`.
- Verified Python import reports NEURON `9.0.1` and exposes `coreneuron.gpu`.
- Rebuilt source-Hay and SWC-template mechanisms against the durable runtime.
- `ldd` for the rebuilt mechanism resolves `libcoreneuron-cuda.so`, NVHPC OpenACC libraries, and `libcudart.so.13` when the scripted runtime environment is active.

## Source-Hay GPU Result

Source-Hay CoreNEURON GPU is valid and batching is necessary.

| run | elapsed s | simulated trace s / wall s | result |
| --- | ---: | ---: | --- |
| CPU vs GPU validation, `8 x 500 ms`, one trace per solve | CPU `23.80`, GPU `131.16` | GPU slow | CPU/GPU metrics match to numerical precision, but launch overhead dominates |
| GPU batch `32 x 500 ms` | `30.23` | `0.529` | useful crossover vs single-process CPU CoreNEURON |
| GPU batch `64 x 500 ms` | `37.57` | `0.852` | stable, higher throughput |
| GPU batch `128 x 500 ms` | `53.47` | `1.197` | stable, higher throughput |
| GPU batch `64 x 2000 ms` | `121.70` | `1.052` | stable longer-trace path |
| GPU batch `128 x 2000 ms` | `152.86` | `1.675` | best measured source-Hay GPU point so far |

Batch equivalence check: source-Hay `batch_size=1` vs `batch_size=4` on NEURON produced identical inputs, targets, soma voltage, and sample IDs.

Root-cause fix: CoreNEURON GPU aborts if a batch creates one `PatternStim` per trace; the working batched path uses one global `PatternStim` carrying all unique gids/events for the solve.

## v1 SWC Morphology Path

Update 2026-06-11: the actual v1 `N=5` morphology path is now GPU-valid for
generation under the durable NEURON `9.0.1` CoreNEURON GPU runtime.

Verified:
- SWC template mechanisms compile against the durable NEURON/CoreNEURON GPU runtime.
- Source-Hay runs on GPU using the SWC-compiled mechanism root, so the SWC mechanism bundle itself is not the failure.
- v1 SWC worker on plain NEURON and CPU CoreNEURON writes valid shards and exits cleanly after worker hard-exit.
- Plain NEURON vs CPU CoreNEURON SWC smoke (`4 x 250 ms`) matched exactly: identical inputs, `targets_max_abs=0.0`, `soma_max_abs=0.0`.
- Bare actual Allen SWC cells advance on CoreNEURON GPU; the morphology itself
  was not GPU-incompatible.
- Actual SWC split worker CPU CoreNEURON vs GPU CoreNEURON (`2 x 250 ms`)
  matched exactly: `inputs_equal=True`, `targets_max_abs=0.0`,
  `soma_max_abs=0.0`, and spike counts `7/7`.
- Actual SWC event-dropout worker CPU CoreNEURON vs GPU CoreNEURON
  (`2 x 250 ms`) matched exactly for baseline and intervention targets/soma:
  all max absolute differences `0.0`, baseline spike counts `7/7`, and
  intervention spike counts `4/4`.
- All-five v1 morphology GPU orchestrator smoke wrote the real v1 layout:
  `5` validation shards plus `5` `event_dropout` paired intervention shards at
  `100 ms` under
  `artifacts/v1_swc_gpu_generation_smoke_20260611/`.
- A longer actual-SWC GPU worker (`4 x 1000 ms`) wrote valid outputs with
  `46,232` input events, `13` spike labels, and elapsed worker time
  `23.64 s`.

Root cause of the old failure:
- `run_v1_parallel_generation.py` imported benchmark/model modules before
  importing NEURON. That package graph can import torch/CUDA modules first; in
  that import order, the NVHPC/OpenACC CoreNEURON runtime later reported
  `Enabled GPU execution but couldn't find NVIDIA GPU!`.
- Importing NEURON/CoreNEURON before torch-bearing modules fixes GPU discovery
  for the exact previously failing worker command. The generation scripts now
  perform this early import explicitly.

Interpretation:
- There is no remaining SWC GPU correctness blocker in the tested v1 generation
  path.
- Full dataset generation should use the actual v1 SWC CoreNEURON GPU route
  with the durable runtime environment and early NEURON import. Throughput
  tuning can still improve batch/worker sizing, but that is no longer a
  correctness substitute or CPU fallback.

## Code Changes

- `generate_coreneuron_hay_dataset.py` supports `--batch-size` and batched simulator solves.
- `attach_deterministic_synapses` supports unique gid offsets and event collection into one global `PatternStim`.
- `run_coreneuron_event_dropout_pair.py` supports batched signed-input simulation and imports NEURON/CoreNEURON before torch-bearing benchmark modules.
- `run_v1_parallel_generation.py` batches per worker, supports bounded `--samples-per-shard`, loads hoc files from the durable runtime, imports NEURON/CoreNEURON before torch-bearing benchmark modules, sets GPU cell permutation explicitly, and hard-exits successful workers to avoid NEURON finalization crashes after successful SWC output writes.
- `run_generation_throughput_gate.py` now points at the durable GPU runtime and can reproduce source-Hay GPU batch timings.
