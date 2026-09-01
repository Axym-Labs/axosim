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

Released AxoSim-Mamba checkpoints require the fused official backend:

```bash
python -m pip install -e '.[mamba]'
```

The dependency-free `AxoPyTorchMamba` backend is intended for tests and smoke
runs; official Mamba checkpoints use a different learned-timestep rank and are
not state-dict compatible with that fallback.

Deep AxoSim-Mamba configurations can condition each temporal-branch update to
unit RMS before the learned residual multiplier:

```python
from axosim.axomamba import default_axomamba_config

config = default_axomamba_config(
    model_dim=400,
    state_dim=100,
    num_layers=12,
    head_dim=100,
    mamba_update_normalization="fixed_rms",
)
```

This is an optimization-stability option.
 The default is `"none"` so existing checkpoint behavior is unchanged;
the selected mode is stored in new checkpoint configurations and is applied
identically in full-sequence and streaming execution.

The setup command creates local directories without downloading the large
public datasets:

```bash
axosim-setup
```

Inspect the complete public-data setup before running it:

```bash
axosim-setup --dry-run --install-kaggle --download-data --convert-raw
```

## Python Interface

The public imports use the stable model names. Historical internal class names
remain checkpoint-compatible.

```python
from axosim import (
    AXOSIM_POPULATION_PROFILE,
    AxoSimGRU,
    AxoSimLite,
    AxoSimMamba,
    AxoSimPopulation,
    CudaGraphAdaptationStep,
    HomeostaticThresholdController,
)
```

`AxoSimPopulation` is an ordinary `torch.nn.Module`. It keeps three adaptation
mechanisms separate:

- a direct behavior adapter for each neuron;
- a morphology adapter shared only within a morphology class; and
- a positive efficacy for every retained incoming contact.

For AxoSim-Lite, the behavior group acts at seven explicit sites: encoded-input
offset/gain, state-decay delta, recurrent-proposal offset/gain, and native
output offset/gain. It specializes temporal response and readout calibration;
the separate homeostatic threshold controller regulates firing rates.

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

When contact histories are sparse, `forward_sparse_contacts()` avoids a dense
`population × time × contacts` tensor while retaining one independently
trainable efficacy per contact:

```python
prediction = population.forward_sparse_contacts(
    event_summary_indices,
    event_contact_indices,
    event_values,
    time_steps=256,
)
```

`event_summary_indices` address flattened `(population, time)` rows;
`event_contact_indices` address flattened `(population, contact)` rows. Each
paired index must identify the same target neuron. Gradients propagate through
event values, behavior and morphology adapters, and selected contact
efficacies.

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

Connected simulations can also apply causal firing-rate homeostasis without
changing the shared neuron model. The controller observes thresholded events
and returns one bounded threshold offset per declared population group:

```python
controller = HomeostaticThresholdController(
    group_ids=role_morphology_group_ids,
    target_rates_hz=role_morphology_target_rates_hz,
    dt_ms=1.0,
    update_interval_ms=100.0,
    time_constant_ms=500.0,
    learning_rate=0.07,
    max_abs_offset=3.0,
)
if controller.observe(firing):
    effective_thresholds = (
        fixed_thresholds + controller.threshold_offsets_per_neuron()
    )
```

The update is causal, group-stratified, and bounded. Its response time is a
simulation setting; compressed validation settings are not biological
timescale estimates.

## Population Runtime

The authoritative workload is `MILLION_NEURON_REALTIME_CONTRACT`. It fixes
1,048,576 neurons, exact native 1 ms inputs and outputs, causal P4 forecasting,
threshold firing without a quota or top-k, 80/20 E/I roles, delayed
morphology-aware routing, and complete-stack timing on an RTX 5090.

`AXOSIM_POPULATION_PROFILE` uses the compact 24-token, 8-state population
model and 88 W8 behavior values per neuron. Its densest directly measured
real-time operating point retains 362 independently mutable contact
efficacies per neuron. Across three isolated, target-rate-valid runs, its
median p90 throughput is 1.177B neuron-steps/s and its slowest run reaches
1.174B. The same model provides a measured contact/throughput frontier: its
replicated maximum-throughput point reaches a median 2.339B at 100 contacts,
while target-rate observations at 376, 400, and 426 contacts do not establish
a denser reproducible real-time operating point.

For the exact 1,048,576-neuron population, real time is 1.048576B
neuron-steps/s. The publication curve uses a monotone fit through replicated
target-rate medians and stops at 362 contacts. It does not extrapolate an
unmeasured crossing through the activity-unstable and resource-limited
region; 362 contacts is the densest conservative replicated exact pass.

These measurements include native-voltage telemetry for 1% of neurons to
pinned host memory. Full GPU-resident output remains available.
Durably recording every FP16 voltage to the measured local NVMe device is
outside the real-time contract because its bandwidth is insufficient.

The benchmark initializes five fitted morphology classes and deterministic
per-neuron behavior residuals. It proves that unique adaptation rows execute;
it does not claim that one million neurons were fitted independently. The
population connectome is procedural and biologically structured, not an
empirical connectome.

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

The role-correct development-corpus AxoSim-Mamba uses
`configs/axomamba-population.json` and
`checkpoints/axomamba-population.pt`. It was trained on the 7,500-trace
development corpus, retains all six excitatory/inhibitory morphology IDs, and
is separate from the compact SERA-only and spike-focused checkpoints. It is
not presented as a checkpoint trained on the completed 50,000-trace release.

## Data

Public NeuronIO assets are not included. Configure Kaggle credentials through
its standard credential file or environment variables, then run:

```bash
axosim-setup --install-kaggle --download-data
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

## Verify the package

```bash
python -m pytest -q
```

## Evidence boundaries

The million-neuron rates are hardware- and contract-specific complete-stack
measurements, not inferred speedups over Arbor, CoreNEURON, or another
biophysical simulator. AxoSim's trainable behavior, morphology, threshold, and
contact-efficacy coordinates are learned operational state; they are not
interpretable ion-channel or compartment parameters. The population-context
dataset uses biophysical teachers on connected-model input histories rather
than a fully biophysical closed loop. A matched fixed-recipe run on the
completed 50,000-trace corpus improved held-out trace metrics but did not
pass the morphology-conditioned firing-rate criterion on three held-out
connectomes, so that checkpoint is not promoted here.
