# Literature Scan After Iteration 120

Date: 2026-07-12
Decision target: find structural changes capable of reducing a visible fraction
of AxoMamba's roughly 20 mV voltage SERA after latent-token operators and global
scale-up failed to replicate, while learned pre-aggregation synaptic efficacy
produced a balanced 120k lead.

## Executive Decision

The next phase should preserve and model **synapse-level temporal structure
before branch aggregation**. AxoMamba's most promising recent result came from
restoring one degree of freedom that the biological ELM already had: individual
synaptic efficacy. The more aggressive extension is not another global Mamba
variant. It is a compact heterogeneous-synapse operator in which each synapse
selects among several causal temporal bases before its branch is collapsed.

The preferred implementation is efficient and attributable:

1. learn per-synapse weights into four fixed or learnable causal timescale
   bases;
2. aggregate synapses within each branch separately for each basis;
3. filter the resulting 118 x 4 branch streams, then project them into the
   retained Mamba trunk;
4. compare against an exact-parameter/compute model that executes the same
   filters but ties every synapse to the same temporal mixture;
5. include a genuine global-capacity control and report latency/memory.

This preserves fine-grained synaptic dynamics without materializing 1,278 full
per-synapse state trajectories. It is distinct from the rejected branch-memory
bank, which filtered only after all synapses in a branch had already been summed.

## Evidence

### Synapse-specific temporal heterogeneity

HetSyn (2025) places distinct temporal dynamics at individual synapses instead
of applying one decay after inputs are aggregated. It reports benefits on
pattern generation, delayed match-to-sample, speech, vision, noise robustness,
and limited-neuron settings. Its central equation maintains a synaptic current
with its own efficacy and decay for each afferent. The transferable mechanism
is exact: temporal response can differ by input pathway, and aggregation before
that response destroys the distinction.

Source: [HetSyn arXiv paper](https://arxiv.org/abs/2508.11644).

**AxoMamba implication.** A literal 1,278-state Python recurrence would be too
slow. A low-rank temporal basis gives each synapse a learned mixture over a few
causal decays, then uses linearity to aggregate before filtering. This tests the
same premise at practical GPU cost.

### Structured dendritic connectivity and restricted sampling

Chavlis and Poirazi (Nature Communications, 2025) use fixed sparse
synapse-to-dendrite and dendrite-to-soma structure with restricted receptive
fields. Their dendritic networks resist overfitting and can match or outperform
dense controls with far fewer parameters on classification tasks. Their task
results do not transfer directly, but their architecture preserves multiple
learned subunits inside each restricted dendritic input set.

Source: [Nature Communications article](https://www.nature.com/articles/s41467-025-56297-9).

**AxoMamba implication.** If heterogeneous temporal bases fail, replace the
single scalar branch sum with several learned nonlinear subunits per branch.
Each subunit receives only that branch's 11 routed synapses; the 118 x H outputs
are projected to the temporal trunk. Match against a dense/global capacity
control rather than attributing gains to extra parameters.

### NOBLE and neuron-specific continuous embeddings

NOBLE (2025) uses a Fourier neural operator and biologically informed continuous
embeddings to map current-injection functions and neuron features to somatic
voltage. It trains on 75,600 simulated examples, validates on experimental human
neurons, and uses relative L4 loss because a preliminary study found it better
preserved spike widths and amplitudes than relative L2.

Source: [NOBLE arXiv paper](https://arxiv.org/abs/2506.04536).

**AxoMamba implication.** The useful lesson is conditioning the operator on
interpretable neuron properties before dynamics, not merely adding a late
morphology embedding. AxoBench's five morphologies map common source channels
to different target segments by normalized arclength; morphology-conditioned
synaptic gains or temporal mixtures are therefore a stronger next test than a
larger generic morphology vector. NOBLE's full temporal FNO is noncausal and
uses square-pulse currents, so it is not an immediate replacement for the causal
event-driven benchmark.

### State-space models as neural operators

Hu et al. (Neural Networks, 2026) benchmark Mamba state-space models as neural
operators for dynamical systems and find strong accuracy/efficiency and
extrapolation behavior across diverse systems. This supports retaining Mamba as
the causal temporal solver while improving the spatial/input operator.

Source: [paper record and abstract](https://pubmed.ncbi.nlm.nih.gov/41447828/)
and [arXiv preprint](https://arxiv.org/abs/2409.03231).

**AxoMamba implication.** I117-I118 already show that more global Mamba depth
or width is seed-unstable. The literature does not justify repeating scale-up;
it supports composing Mamba with a better forcing representation.

### Learnable synaptic weight and delay

Recent work on joint synaptic weight-delay learning treats pathway efficacy and
timing as coupled rather than interchangeable. The 2026 SWDSL paper learns both
through a kernel representation of spike trains. Its downstream SNN setting is
different, but it independently prioritizes synapse-specific delay after weight
heterogeneity.

Source: [Neurocomputing article](https://doi.org/10.1016/j.neucom.2026.133430).

**AxoMamba implication.** Do not add free per-channel integer delays first.
Test stable positive temporal-basis mixtures, which are differentiable, causal,
and can express delayed/slow influence without discontinuous indexing.

## Dataset-Specific Findings

- Inputs are 1,278 signed event channels: the first half is excitatory and the
  second inhibitory before interlocking.
- The promoted router covers all 1,278 channels in 117 nonempty 11-slot groups;
  one of the nominal 118 branches is empty because ceil-stride routing starts
  its window beyond the input. This wastes one branch column but does not drop
  an input channel.
- Before I119, every valid slot had fixed efficacy one. Thus each branch exposed
  only a signed event sum to the learned model.
- I119 learns all 1,278 valid gains with standard deviation 0.111 and range
  0.731-1.581, while improving all six core point estimates. That is stronger
  direct evidence for synaptic heterogeneity than the noisy observational
  single-event correlation audit.
- Morphology routing is normalized by compartment type and arclength, so one
  source channel can correspond to different physical segments across the five
  morphologies. A shared efficacy is therefore only a first approximation.

## Ranked Structural Queue

| rank | experiment | mechanism | decisive gate |
| ---: | --- | --- | --- |
| 1 | low-rank heterogeneous synaptic time bases | each site has distinct causal efficacy and timescale before branch sum | material balanced gain over tied-mixture exact control and global-capacity control |
| 2 | morphology-conditioned synaptic efficacy/time mixtures | shared source channels map to different physical compartments across morphologies | improve morphology disparity and intervention SERA without average/F1 regression |
| 3 | multiple sparse nonlinear subunits per branch | one scalar branch sum cannot retain within-branch input patterns | beat parameter-matched dense projection and promoted trunk at acceptable latency |
| 4 | morphology-graph neural operator feeding causal Mamba | actual tree topology should determine spatial propagation before soma dynamics | only after reconstructing exact channel-to-segment maps and validating graph fidelity |
| 5 | causal multi-resolution temporal operator | local spike transients and slower membrane response need distinct resolutions | must beat Mamba with matched synaptic encoder, not a weaker old baseline |
| 6 | NOBLE-style full temporal FNO | global spectral operator may capture voltage/spike waveform | low priority because standard FNO is noncausal and NOBLE's stimulus regime is much simpler |

## Rejections Carried Forward

- Do not revisit unconstrained latent tokens: learned assignments collapsed and
  fixed sparse assignments created a SERA/F1 tradeoff.
- Do not revisit global depth/width: both balanced seed-1 leads reversed on the
  unchanged replication.
- Do not reintroduce post-aggregation branch memory: it cannot test
  synapse-specific temporal heterogeneity because identity is already lost.
- Do not treat another 0.01-0.03 mV movement as the endpoint. It may justify a
  mechanistic replication, but the stated research target remains a visible
  fraction of base error.

## Next Experiment Rule

If I120 validates learned synaptic efficacy at 300k, retain it and test the
four-basis heterogeneous-synapse operator at I121. If I120 reverses, do not tune
gain scales; test multiple sparse nonlinear subunits directly, because the
single scalar branch bottleneck remains even when scalar efficacy is unstable.
