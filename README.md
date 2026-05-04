# bnn_sim

Deterministic Branch-ELM / official ELM neuron harness inspired by
`deep-research-report.md`.

This includes the earlier small Branch-ELM-style baseline plus a minimal port of
the paper repository's Expressive Leaky Memory (ELM) neuron model for NeuronIO.

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

## Smoke Workflow

Create deterministic demo NeuronIO-style shards:

```bash
python -m bnn_sim.cli make-demo-data --output runs/demo-data --samples 8 --time-steps 32 --input-dim 64
```

Train a small reproducible Branch-ELM baseline:

```bash
python -m bnn_sim.cli train \
  --data runs/demo-data \
  --input-dim 64 \
  --memory-units 30 \
  --branches 8 \
  --epochs 1 \
  --checkpoint runs/branch-elm.pt \
  --metrics runs/train-metrics.json
```

Train the official paper-repo ELM shape with the included `num_memory_30`
configuration:

```bash
python -m bnn_sim.cli make-demo-data --output runs/official-demo-data --samples 2 --time-steps 8 --input-dim 1278

python -m bnn_sim.cli train \
  --data runs/official-demo-data \
  --model-kind official \
  --model-config configs/branch_elm_30_official.json \
  --epochs 1 \
  --checkpoint runs/official-elm.pt \
  --metrics runs/official-train-metrics.json
```

Evaluate a checkpoint:

```bash
python -m bnn_sim.cli evaluate \
  --data runs/demo-data \
  --checkpoint runs/branch-elm.pt \
  --output runs/eval-metrics.json
```

Evaluate a raw upstream official checkpoint state dict by pairing it with its
model config:

```bash
python -m bnn_sim.cli evaluate \
  --data runs/neuronio-shards \
  --model-kind official \
  --model-config configs/branch_elm_30_official.json \
  --checkpoint .external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt \
  --output runs/official-eval-metrics.json
```

Benchmark recurrent forward latency:

```bash
python -m bnn_sim.cli benchmark \
  --input-dim 1278 \
  --memory-units 30 \
  --branches 32 \
  --batch-size 1 \
  --time-steps 500 \
  --runs 10 \
  --output runs/benchmark.json
```

For the official model, replace the model sizing flags with:

```bash
--model-kind official --model-config configs/branch_elm_30_official.json
```

## Implemented From The Report

- Deterministic pre-sharded loader for NeuronIO-style `(samples, time, channels)` inputs and two-output spike/soma targets.
- Branch-ELM recurrent runtime with configurable memory units, branch routing, synaptic trace state, memory state, and spike/soma readout.
- Branch-ELM-30 default configuration.
- BCE + MSE training loop with burn-in support.
- Checkpoint save/load with model config metadata.
- Evaluation metrics: soma RMSE and spike AUC.
- Benchmark command with optional `torch.compile`.
- Official paper-repo ELM model port, `configs/branch_elm_30_official.json`,
  official checkpoint-compatible loading, and CLI selection via
  `--model-kind official`.

## Not Included

The public NeuronIO Kaggle assets are large and are not downloaded by this
scaffold. Convert those files into `.npz` shards containing `inputs`, `targets`,
and optionally `sample_ids`, then point `train` or `evaluate` at that directory.

Raw NeuronIO pickle files matching the upstream `*sim__*` naming pattern can be
converted directly:

```bash
python -m bnn_sim.cli convert-neuronio --input path/to/raw-neuronio-folder --output runs/neuronio-shards
```

Use deterministic fixed windows for the report's 500 ms training/evaluation
setup:

```bash
python -m bnn_sim.cli train \
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
