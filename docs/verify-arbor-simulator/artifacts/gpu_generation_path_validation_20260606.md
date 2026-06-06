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

The actual v1 `N=5` morphology path is not yet GPU-valid.

Verified:
- SWC template mechanisms compile against the durable NEURON/CoreNEURON GPU runtime.
- Source-Hay runs on GPU using the SWC-compiled mechanism root, so the SWC mechanism bundle itself is not the failure.
- v1 SWC worker on plain NEURON and CPU CoreNEURON writes valid shards and exits cleanly after worker hard-exit.
- Plain NEURON vs CPU CoreNEURON SWC smoke (`4 x 250 ms`) matched exactly: identical inputs, `targets_max_abs=0.0`, `soma_max_abs=0.0`.

Failed:
- v1 SWC CoreNEURON GPU worker aborts before writing output, even for `1 x 100 ms`.
- GDB backtrace places the abort in CoreNEURON GPU initialization during `pc.psolve`, before the solve proper.
- Debug notifications did not produce an additional diagnostic message.

Interpretation:
- There is no user-side blocker, but SWC/CoreNEURON GPU is currently a simulator compatibility failure for this v1 morphology path.
- For v1 dataset generation, the defensible path today is batched CPU CoreNEURON/SWC with bounded shard sizes.
- Source-Hay GPU batching remains useful as a proven GPU route and as a reference for future work, but it cannot be claimed as the v1 `N=5` generation backend until the SWC GPU abort is fixed.

## Code Changes

- `generate_coreneuron_hay_dataset.py` supports `--batch-size` and batched simulator solves.
- `attach_deterministic_synapses` supports unique gid offsets and event collection into one global `PatternStim`.
- `run_coreneuron_event_dropout_pair.py` supports batched signed-input simulation.
- `run_v1_parallel_generation.py` batches per worker, supports bounded `--samples-per-shard`, loads hoc files from the durable runtime, and hard-exits successful workers to avoid NEURON finalization crashes after successful SWC output writes.
- `run_generation_throughput_gate.py` now points at the durable GPU runtime and can reproduce source-Hay GPU batch timings.
