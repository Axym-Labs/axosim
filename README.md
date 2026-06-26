# AxoSim

AxoSim is a deterministic NeuronIO training and evaluation harness for
biologically inspired surrogate models, including the official ELM /
Branch-ELM family, recurrent comparison cells, and the promoted `AxoMamba`
sequence surrogate. The small legacy BranchELM path remains available for
toy-data smoke tests.

## Install

```bash
python -m pip install -e .[dev]
```

For a new machine, use the setup script. The default creates local directories
and installs the package without downloading the large public datasets:

```bash
python scripts/setup.py
```

To inspect the full public-data workflow before running it:

```bash
python scripts/setup.py --dry-run --install-kaggle --download-data --convert-raw
```

To download the public NeuronIO Kaggle data, configure Kaggle credentials first
with `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`, then run:

```bash
python scripts/setup.py --install-kaggle --download-data
```

The dataset download uses a resumable `.zip.part` file. If a large transfer is
interrupted, rerun the same command and it will continue from the partial file
when Kaggle's server honors byte ranges.

## Recommended Workflow

List the named presets:

```bash
axosim list-presets
```

Train with the current recommended setup:

```bash
axosim train \
  --preset paper-random-fulltrace \
  --data data/full-trace-shards/nmda-train \
  --val-data data/full-trace-shards/nmda-val \
  --epochs 10 \
  --epoch-samples 80000 \
  --shard-reuse-batches 58 \
  --checkpoint runs/paper-like.pt \
  --metrics runs/paper-like-metrics.json
```

Train the promoted AxoMamba configuration:

```bash
axosim train \
  --preset axomamba-probe \
  --data data/full-trace-shards/nmda-train \
  --val-data data/full-trace-shards/nmda-val \
  --epochs 10 \
  --epoch-samples 80000 \
  --shard-reuse-batches 58 \
  --checkpoint runs/axomamba.pt \
  --metrics runs/axomamba-metrics.json
```

Evaluate with the corrected full-trace metric path:

```bash
axosim evaluate \
  --preset fulltrace-paper-metric \
  --data data/full-trace-shards/nmda-val \
  --checkpoint runs/paper-like.pt \
  --output runs/paper-like-val.json
```

Each `train` and `evaluate` run writes a tiny `experiment-registry.jsonl` next
to the metrics file by default. Disable that with `--no-registry`.

## Smoke Workflow

Create deterministic demo NeuronIO-style shards:

```bash
axosim make-demo-data --output runs/demo-data --samples 8 --time-steps 32 --input-dim 64
```

Train the small legacy smoke path on toy data:

```bash
axosim train \
  --preset legacy-baseline-smoke \
  --data runs/demo-data \
  --input-dim 64 \
  --memory-units 30 \
  --branches 8 \
  --epochs 1 \
  --checkpoint runs/branch-elm.pt \
  --metrics runs/train-metrics.json
```

Train the official paper-repo ELM shape on toy data:

```bash
axosim make-demo-data --output runs/official-demo-data --samples 2 --time-steps 8 --input-dim 1278

axosim train \
  --preset paper-random-fulltrace \
  --data runs/official-demo-data \
  --model-config configs/branch_elm_30_official.json \
  --window-size 8 \
  --ignore-start 0 \
  --burn-in 0 \
  --epoch-samples 2 \
  --epochs 1 \
  --checkpoint runs/official-elm.pt \
  --metrics runs/official-train-metrics.json
```

Evaluate a checkpoint:

```bash
axosim evaluate \
  --preset legacy-baseline-smoke \
  --data runs/demo-data \
  --checkpoint runs/branch-elm.pt \
  --output runs/eval-metrics.json
```

Evaluate a raw upstream official checkpoint state dict by pairing it with its
model config:

```bash
axosim evaluate \
  --preset fulltrace-paper-metric \
  --data runs/neuronio-shards \
  --model-config configs/branch_elm_30_official.json \
  --checkpoint .external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt \
  --output runs/official-eval-metrics.json
```

Benchmark recurrent forward latency:

```bash
axosim benchmark \
  --input-dim 1278 \
  --memory-units 30 \
  --branches 32 \
  --batch-size 1 \
  --time-steps 500 \
  --runs 10 \
  --output runs/benchmark.json
```

## Implemented From The Report

- Deterministic pre-sharded loader for NeuronIO-style `(samples, time, channels)` inputs and two-output spike/soma targets.
- Official paper-repo ELM / Branch-ELM runtime, routing, synaptic trace state, memory state, and spike/soma readout.
- `AxoMamba`, the reusable promoted Mamba surrogate configuration.
- Legacy simplified BranchELM path kept for smoke tests.
- BCE + MSE training loop with burn-in support.
- Checkpoint save/load with model config metadata.
- Evaluation metrics: soma RMSE and spike AUC.
- Benchmark command with optional `torch.compile`.
- Named train/evaluate presets and tiny JSONL experiment registry.

## Not Included

The public NeuronIO Kaggle assets are large and are not downloaded by this
scaffold. Convert those files into `.npz` shards containing `inputs`, `targets`,
and optionally `sample_ids`, then point `train` or `evaluate` at that directory.

Raw NeuronIO pickle files matching the upstream `*sim__*` naming pattern can be
converted directly:

```bash
axosim convert-neuronio --input path/to/raw-neuronio-folder --output runs/neuronio-shards
```

The recommended training path is the preset-based full-trace workflow above. If
you need the older fixed-window path for a compatibility check:

```bash
axosim train \
  --data runs/neuronio-shards \
  --model-kind official \
  --model-config configs/branch_elm_30_official.json \
  --window-size 500 \
  --window-stride 350 \
  --ignore-start 500 \
  --burn-in 150 \
  --shuffle-mode shard \
  --cache-shards 1 \
  --checkpoint runs/branch-elm-30.pt \
  --metrics runs/branch-elm-30-train.json
```

`--shuffle-mode shard` keeps training deterministic while shuffling shard order
and within-shard sample order. Pair it with `--cache-shards 1` for compressed
NeuronIO shards so each shard is inflated once while its local samples are used.
