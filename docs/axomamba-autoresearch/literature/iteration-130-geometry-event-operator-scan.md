# Literature Scan After Iteration 130

Date: 2026-07-12
Decision target: identify structural changes capable of moving substantially
more than the 0.05-0.15 mV class while preserving the replicated gains from
morphology-conditioned synaptic efficacy and nonlinear branch subunits.

## Executive Decision

The next high-temperature direction should make morphology conditioning
**continuous and structural**, rather than adding a larger categorical table.
The v1 generator maps common source sites to each target morphology by
compartment type and normalized arclength, and the five SWCs plus routing code
remain available locally. Reconstruct those exact maps, verify them against the
shard summaries, and feed source-site coordinates/tree context into a compact
spatial operator before the retained Mamba trunk.

The first implementation should be less ambitious than a full graph simulator:

1. regenerate, version, and test the five source-to-segment maps;
2. represent each valid synapse by sign, compartment type, normalized
   arclength, target multiplicity, and distance-to-soma/tree-depth features;
3. use one shared coordinate-conditioned efficacy/subunit network across all
   morphologies, with the promoted categorical gain table as a residual;
4. compare against a parameter/compute-matched model receiving shuffled or
   constant coordinates and against the promoted categorical table;
5. advance to graph message passing only if coordinate structure has signal.

This directly tests geometry rather than cell identity and can eventually
generalize to unseen morphologies. It is also the most faithful extension of
the strongest I125-I130 mechanism.

## Evidence

### Geometry-independent electrophysiology operators

Zhou et al. express 147 patient-specific atrial geometries in a universal
coordinate system, then learn an electrophysiology operator over a common
fixed-resolution representation. Across 308,700 simulations, their operator
predicts local activation-time fields with 5.1 ms mean error and 0.12 ms
inference. The relevant mechanism is not the vision-transformer backbone; it is
the explicit decoupling of biophysical location from mesh identity.

Source: [geometry-independent cardiac electrophysiology operator](https://arxiv.org/abs/2512.01702).

**AxoMamba implication.** The v1 normalized-by-type-arclength policy already
defines an analogous universal coordinate. Categorical morphology embeddings
discard that known correspondence. Coordinate-conditioned synaptic efficacy is
the smallest faithful test, and should precede a generic graph network.

### Latent dynamics on neural trees

GALDS encodes neural-tree geometry and physical fields with graph autoencoders,
assembles those latents into a global graph, and evolves them with a Neural-ODE-
inspired latent dynamics model. It reports evaluation on eight unseen geometries,
roughly 3% mean relative error, maximum error below 8%, and a ten-fold speedup
over prior surrogate approaches.

Source: [GALDS neural-tree surrogate](https://arxiv.org/abs/2507.10871).

**AxoMamba implication.** Tree topology can support out-of-geometry transfer,
but GALDS predicts smooth material transport rather than thresholded membrane
voltage. Use its geometry encoder/global latent graph separation, while keeping
Mamba as the causal temporal solver and AxoBench's event diagnostics as gates.

### Multi-compartment integration in long-sequence models

Sun et al. use nonlinear multi-compartment integration in a spiking world model.
Their model is competitive with GRU world models on DeepMind Control Suite and
outperforms SNN baselines on SHD, TIMIT, and LibriSpeech 100h. This is adjacent
task evidence that cooperative dendritic sources can be useful beyond neuron
simulation.

Source: [multi-compartment spiking world model](https://arxiv.org/abs/2503.00713).

**AxoMamba implication.** I127-I128 already give a more relevant within-task
premise test: nonlinear subunits add to morphology-conditioned synapses under
exact controls. Keep that mechanism, but organize subunits using reconstructed
tree geometry instead of arbitrary contiguous source-slot groups.

### Explicit handling of events and changes

Eventax combines differentiable ODE solving with event handling, providing
forward-exact gradients for arbitrary ODE neuron models and demonstrating a
multi-compartment human L2/3 dendritic-spike model. CP-DSA separately uses
adjacent-step changes in dendritic potential to enhance or weaken its somatic
update and reports low-latency gains on event datasets.

Sources: [Eventax](https://arxiv.org/abs/2603.08146) and
[change-perceptive DSA neuron](https://arxiv.org/abs/2512.16259).

**AxoMamba implication.** The persistent failure modes are high-dV/dt,
near-spike error, and low voltage gain. A future decoder should expose both
continuous voltage state and an event/change state, rather than relying on one
smooth hidden representation for both. A causal change feature is more
targeted than revisiting fixed exponential bases, which I121-I122 rejected.

### Adaptive dendritic timing and fractional memory

Recent dendritic-heterogeneity work dynamically adjusts timing factors and uses
continuous modulation to preserve nonlinear feature expression. Fractional-
order SNNs likewise motivate non-Markovian dynamics from long-range neural
correlations and fractal dendrites.

Sources: [adaptive dendritic plasticity](https://doi.org/10.1016/j.neunet.2025.108191)
and [fractional-order SNN](https://openreview.net/forum?id=NJhBSLJ0nL).

**AxoMamba implication.** These are lower priority. I121-I122 showed that fixed
synapse-specific temporal bases improve dynamics/response but create a
voltage/F1 frontier even as an active residual. Revisit temporal heterogeneity
only after geometry-aware routing, and require a control that distinguishes
adaptive state-dependent timing from another fixed filter bank.

## Dataset Feasibility Findings

- All five training SWCs are present under
  `data/morphologies/allen_mouse_l5_spiny/`.
- The AxoBench generator retains `_build_source_site_route` and the exact
  normalized-by-type routing implementation.
- Shards record morphology ID and aggregate routing checks, but not the full
  source-to-segment map. The map must therefore be regenerated and checked
  against segment counts, unique/unused source counts, and duplication summaries
  before it can be treated as experimental input.
- For a recorded 265-segment morphology, only 264 of 639 source sites per sign
  are unique after mapping; 375 are unused. This makes target multiplicity and
  coordinate identity plausible high-value features.
- A full graph operator should not be claimed to use biological topology until
  SWC parent relations, compartment labels, and source-route alignment all pass
  deterministic round-trip tests.

## Ranked Queue

| rank | experiment | mechanism | decisive control |
| ---: | --- | --- | --- |
| 1 | normalized-coordinate synaptic operator | shared efficacy depends on sign, compartment, arclength, depth, and multiplicity | shuffled/constant coordinates with exact parameters and compute |
| 2 | morphology-tree graph encoder into AxoMamba | message passing follows verified SWC parent edges before temporal integration | degree-matched rewired tree plus categorical-gain promoted model |
| 3 | geometry-organized nonlinear subunits | subunits group synapses by tree neighborhoods instead of source-index contiguity | contiguous and random degree-matched subunits |
| 4 | event/change-conditioned voltage decoder | explicit change state targets high-dV/dt and near-spike failures | same decoder receiving detached/zero change features |
| 5 | latent graph dynamics plus causal Mamba | tree autoencoder compresses variable geometry before temporal operator | coordinate encoder with no graph propagation |
| 6 | adaptive/fractional dendritic timing | state-dependent non-Markovian memory | fixed-basis temporal model; only revisit after geometry |

## Rejections Carried Forward

- Do not retry fixed synaptic temporal bases or tune their taus/scales.
- Do not retry generic global depth/width; exact global controls remain
  seed-unstable and are dominated by the combined structural model.
- Do not use a categorical morphology table as evidence of topology or unseen-
  morphology generalization.
- Do not build a graph from source-channel order alone. Reconstruct and verify
  the actual generator mapping first.
- Do not optimize only the voltage output affine: I129 showed a small voltage
  gain with a substantially larger intervention-response regression.

## Next Experiment Rule

After the I130 long-gate decision and promotion/rollback, iteration 131 should
be an infrastructure-plus-premise iteration: regenerate the exact route maps,
write verification artifacts, and run a compact coordinate-conditioned efficacy
operator against shuffled-coordinate and categorical-gain controls. If route
reconstruction fails validation, skip graph claims and test the event/change
decoder instead.
