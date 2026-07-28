# AxoSim

AxoSim trains and executes feed-forward surrogates of biological neurons, from
single-neuron AxoBench evaluation to connected million-neuron populations. The
project has three public neuron models:

- **AxoSim-Mamba** is the accuracy-oriented sequence model.
- **AxoSim-GRU** is the recurrent full model.
- **AxoSim-Lite** is the compact adaptive P4 model used by large populations.

**AxoSim Population** combines neuron execution, threshold firing, delayed
connectome routing, external drive, and population state. **AxoRuntime** is the
lower-level compute layer: fused kernels, sparse event rings, delay queues,
quantization, memory layouts, and scheduling.

## Install

```bash
python -m pip install -e .[dev]
```

The setup script creates local directories without downloading the large
public datasets:

```bash
python scripts/setup.py
```

Inspect the complete public-data setup before running it:

```bash
python scripts/setup.py --dry-run --install-kaggle --download-data --convert-raw
```

## Python Interface

The public imports use the stable model names. Historical internal class names
remain checkpoint-compatible.

```python
from axosim import (
    AXOSIM_POPULATION_ACCURATE_PROFILE,
    AXOSIM_POPULATION_THROUGHPUT_PROFILE,
    AxoSimGRU,
    AxoSimLite,
    AxoSimMamba,
    AxoSimPopulation,
    CudaGraphAdaptationStep,
)
```

`AxoSimPopulation` is an ordinary `torch.nn.Module`. It keeps three adaptation
mechanisms separate:

- a direct behavior adapter for each neuron;
- a morphology adapter shared only within a morphology class; and
- a positive efficacy for every retained incoming contact.

```python
population = AxoSimPopulation(
    lite_neuron,
    morphology_indices=morphology_indices,
    contact_branch_indices=contact_branch_indices,
)
population.enable_adaptation_training(
    behavior=True,
    morphology=False,
    synaptic=True,
)

prediction = population(contact_inputs)
loss = objective(prediction, target)
loss.backward()
```

`contact_inputs` has shape `(population, time, contacts)`.
`forward_batch()` accepts
`(batch, persistent_population, time, contacts)` when independent examples
share the same neuron population and adaptation parameters. Both paths
preserve full temporal gradients. `forward_tokens()` and
`forward_token_batch()` expose pre-encoded P4 paths for full-BPTT adaptation
without rebuilding exact contact inputs.

Fixed-shape online or test-time updates can capture the complete forward,
loss, backward, gradient transform, and optimizer step:

```python
optimizer = torch.optim.Adam(
    population.trainable_adaptation_parameters(),
    lr=1e-3,
    capturable=True,
)
step = CudaGraphAdaptationStep.capture(
    population,
    optimizer,
    example_inputs,
    example_targets,
    objective,
    gradient_transform=lambda: torch.nn.utils.clip_grad_norm_(
        list(population.trainable_adaptation_parameters()),
        1.0,
    ),
)
loss = step(next_inputs, next_targets, synchronize=True)
```

Capture does not consume an optimizer update. Replays require the same tensor
shapes and dtypes as the examples used for capture.

## Population Profiles

The authoritative workload is `MILLION_NEURON_REALTIME_CONTRACT`. It fixes
1,048,576 neurons, exact native 1 ms inputs and outputs, causal P4 forecasting,
threshold firing without a quota or top-k, 80/20 E/I roles, delayed
morphology-aware routing, and complete-stack timing on an RTX 5090.

| Profile | Behavior values / neuron | Synaptic efficacies / neuron | p90 neuron-steps/s | Voltage SERA | F1 area |
|---|---:|---:|---:|---:|---:|
| AxoSim Population-Throughput | 88 W8 | 320 W8 | 1.358B | 15.942 mV | 0.360 |
| AxoSim Population-Accurate | 228 W8 | 320 W8 | 1.135B | 15.742 mV | 0.386 |

These measurements include exact synaptic efficacy and 10% native-voltage
telemetry to pinned host memory. Full GPU-resident output remains available.
Durably recording every FP16 voltage to the measured local NVMe device is
outside the real-time contract because its bandwidth is insufficient.

The benchmark initializes five fitted morphology classes and deterministic
per-neuron behavior residuals. It proves that unique adaptation rows execute;
it does not claim that one million neurons were fitted independently. The
population connectome is procedural and biologically structured, not an
empirical connectome.

The compatibility constants
`MILLION_NEURON_MAX_THROUGHPUT_PROFILE` and
`MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE` refer to the same profile
objects as the stable public names.

## Train And Evaluate

List the named presets:

```bash
axosim list-presets
```

Train AxoSim-Mamba on full-trace shards:

```bash
axosim train \
  --preset axomamba-probe \
  --data data/full-trace-shards/nmda-train \
  --val-data data/full-trace-shards/nmda-val \
  --epochs 10 \
  --epoch-samples 80000 \
  --shard-reuse-batches 58 \
  --checkpoint runs/axosim-mamba.pt \
  --metrics runs/axosim-mamba-metrics.json
```

Evaluate with the AxoBench full-trace metric path:

```bash
axosim evaluate \
  --preset fulltrace-paper-metric \
  --data data/full-trace-shards/nmda-val \
  --checkpoint runs/axosim-mamba.pt \
  --output runs/axosim-mamba-val.json
```

Run the bounded AxoBench diagnostic panel:

```bash
axosim-evaluate-model runs/axosim-mamba.pt \
  --dataset-root data/v1-neuronio-like-gpu-20260611 \
  --device cuda
```

Compare a candidate and frozen baseline on exactly paired traces:

```bash
axosim-evaluate-model runs/candidate.pt \
  --baseline-checkpoint runs/baseline.pt \
  --dataset-root data/axobench-v1 \
  --device cuda \
  --output runs/candidate-vs-baseline.json
```

AxoBench owns metric and diagnostic definitions. AxoSim supplies checkpoint
loading, training, adaptation, and GPU prediction.

The role-correct population-trained AxoSim-Mamba uses
`configs/axomamba-population.json` and
`checkpoints/axomamba-population.pt`. It retains all six excitatory/inhibitory
morphology IDs and is separate from the compact SERA-only and spike-focused
checkpoints.

## Data

Public NeuronIO assets are not included. Configure Kaggle credentials through
its standard credential file or environment variables, then run:

```bash
python scripts/setup.py --install-kaggle --download-data
```

Raw NeuronIO files matching the upstream `*sim__*` naming convention can be
converted into deterministic AxoSim shards:

```bash
axosim convert-neuronio \
  --input path/to/raw-neuronio-folder \
  --output runs/neuronio-shards
```

## Smoke Test

Create toy data, train the small compatibility baseline, and evaluate it:

```bash
axosim make-demo-data \
  --output runs/demo-data \
  --samples 8 \
  --time-steps 32 \
  --input-dim 64

axosim train \
  --preset legacy-baseline-smoke \
  --data runs/demo-data \
  --input-dim 64 \
  --memory-units 30 \
  --branches 8 \
  --epochs 1 \
  --checkpoint runs/branch-elm.pt \
  --metrics runs/train-metrics.json

axosim evaluate \
  --preset legacy-baseline-smoke \
  --data runs/demo-data \
  --checkpoint runs/branch-elm.pt \
  --output runs/eval-metrics.json
```

The Branch-ELM implementation remains as a baseline and checkpoint-compatibility
path. It is not one of the AxoSim deployment models.
