# Spec: Publishable Mamba Surrogate Paper

> **Scope:** Single-neuron surrogate models. Population-level simulation is Paper 2.
> **Date:** 2026-05-30
> **Order of contributions:** 1) Deep NeuronIO-like Dataset + Diagnostic Benchmark → 2) Biophysically Structured SSMs, only after a positive bounded feasibility sprint → 3) Noise Calibration

---

## Contribution 1: NEURO-BENCH — Neuron Surrogate Benchmark Suite

**Status:** Completed as a first implementation and superseded by the v1
dataset + diagnostic benchmark task arc below. The original metric-partition
language in this section is historical context; future work should use
diagnostic suites and dataset conditions rather than the old numbered metric
partition.

### Core Claim
Current surrogate evaluation (RMSE + AUC on 500-step windows) is insufficient. We propose a lean, reusable benchmark that exposes where models fail. The first implementation should be an MVP for model iteration, not a full publication-grade benchmark.

### Implementation Scope
- **MVP target:** M1, a lightweight M1b subset, repaired M2, and one simple M4 intervention.
- **Deferred:** M3 morphology transfer, M5 closed-loop stability, and M6 population statistics unless they become necessary for the immediate model decision.
- **Known weakness:** If the current teacher simulator/data generator still has pathological firing rates or failed input-response checks, the benchmark can only claim fidelity to that teacher/data distribution, not biological validity.

### Publishable Benchmark Target Update
- **Primary metric upgrade:** Do not rely on global RMSE as the main trace
  metric. Report a state-conditioned trace/dynamics metric that separates
  voltage level, voltage derivative, spike timing, and spike-count behavior.
- **Behavioral state resolution:** Compute headline metrics separately over
  meaningful neuron states such as spiking, refractory, near-threshold,
  excited-drive, inhibited-drive, hyperpolarized-quiet, and resting.
- **Structured robustness/OOD:** M4 should become a small battery, targeting
  roughly five well-motivated interventions/probes rather than one MVP probe.
  Each intervention must first pass a teacher-response sanity check and should
  be framed by direction/gain/timing preservation, not generic smoothness.
- **Morphology transfer:** M3 is no longer merely optional for the final
  publishable benchmark. Same-cell-type morphology transfer should be
  implemented or explicitly justified as out of scope with a defensible reason.
- **Leanness constraint:** Reuse shared state definitions and metric utilities
  across M1/M1b/M2/M4 instead of duplicating state-dependent calculations.

### Metric Suites

#### M1: Open-Loop Trace Fidelity (baseline)
- **What:** Direct trace-prediction quality on held-out traces.
  - Soma voltage RMSE/MAE with explicit burn-in and masking semantics.
  - Spike probability quality: AUC for discrimination plus Brier score / log loss when spike outputs are interpreted probabilistically.
  - Spike-count and spike-timing precision over the same trace windows.
- **Credibility sources:** Spike-train similarity is a standard neuroscience problem, and metric choice should match the neural feature under analysis ([Victor 2005 spike train metrics review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2713191/)); time-scale-dependent spike distances such as Victor-Purpura and van Rossum are implemented in neuroscience tooling ([Elephant spike train dissimilarity docs](https://elephant.readthedocs.io/en/stable/reference/_spike_train_processing.html)).
- **Why:** Matches Branch-ELM/TCN for direct comparison, but remains only a baseline. It cannot by itself establish biological validity.
- **Status:** Already implemented

#### M1b: Electrophysiological Feature Fidelity
- **What:** Compare model and teacher traces using electrophysiological features, not only pointwise error.
  - Firing rate, first/last inter-spike interval, adaptation index.
  - Spike threshold, AP amplitude, AP width, AHP depth/time.
  - Burst and refractory summaries where the protocol produces them.
- **Credibility sources:** eFEL was built to standardize electrophysiological feature extraction for in vitro and in silico neuron traces and exposes many such features ([eFEL documentation](https://efel.readthedocs.io/en/latest/index.html), [eFEL paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btag328/8690922)); neuron model validation papers use eFEL features such as first/last ISI when comparing model variants ([CA1 model validation example](https://pmc.ncbi.nlm.nih.gov/articles/PMC9679150/)).
- **Why:** A model can have acceptable RMSE while getting spike shape, adaptation, or firing-rate response wrong. These are practitioner-recognizable failure modes.
- **MVP subset:** Start with dependency-light features: spike count/rate, first/last/mean ISI when spikes exist, rough adaptation ratio, peak voltage, trough/AHP proxy, and non-spiking trace counts. Full eFEL integration is a later extension.
- **Status:** Not implemented; MVP subset should be enough for the first benchmark pass.

#### M2: Trace- and Regime-Aware Failure Diagnostics
- **What:** Decompose errors by trace-aware, biophysical regimes rather than arbitrary flattened bins.
  - Subthreshold/resting, threshold approach, spike, refractory/AHP, burst, and post-event periods.
  - High vs low `dV/dt` periods, computed within each trace.
  - Regime-specific RMSE/MAE, spike-timing error, support counts, and uncertainty.
- **Credibility sources:** eFEL-style feature extraction explicitly treats voltage traces as time-indexed voltage vectors with stimulus metadata, and feature definitions can depend on stimulus start/end and spike events ([eFEL feature documentation](https://bluebrain.github.io/eFEL/efeature-documentation.pdf)); neuroscience spike-train analysis treats all-or-none events differently from continuous signals, which supports separating spike/event regimes from subthreshold voltage regimes ([Victor 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC2713191/)).
- **Why:** Reveals whether errors are concentrated in critical regimes and makes failures actionable.
- **MVP subset:** Repair trace boundaries and report four robust regimes: far-from-spike/subthreshold, near-spike, post-spike/refractory, and post-input-event. Include support counts and skip regimes with too little support.
- **Status:** Existing implementation is only partial. MVP repair should prioritize correctness over adding many regimes.

#### M3a: Same-Cell-Type Morphology Transfer
- **What:** Train on one set of morphologies from a cell type and test on held-out morphologies from the same or closely matched cell type, while holding stimulation protocol and mechanism distributions fixed where possible.
- **Metrics:** M1/M1b/M2 deltas, transfer degradation by morphology features such as branch count, total dendritic length, apical/basal distribution, and route mismatch.
- **Credibility sources:** Allen Cell Types provides electrophysiology, downloadable morphology reconstructions, and neuronal models in a common resource ([Allen Cell Types overview](https://celltypes.brain-map.org/), [AllenSDK Cell Types docs](https://alleninstitute.github.io/AllenSDK/cell_types.html)); Allen data explicitly combines electrophysiology and morphology for cell-type characterization ([Gouwens et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31209381/)).
- **Why:** Morphology generalization is a real surrogate-simulator concern, but same-type transfer avoids confounding morphology with cell identity.
- **Status:** Not part of the MVP unless data are already ready. Keep as a protocol/design item for now.

#### M3b: Cross-Cell-Type Transfer
- **What:** Test broader transfer, e.g. L5 pyramidal to L2/3 pyramidal or interneuron-like morphologies, only after same-type transfer is understood.
- **Metrics:** Same as M3a, but reported as cross-cell-type transfer, not pure morphology transfer.
- **Credibility sources:** Allen Cell Types was designed around multimodal cell-type characterization, including electrophysiological and morphological diversity ([Allen Cell Types overview](https://celltypes.brain-map.org/), [AllenSDK docs](https://alleninstitute.github.io/AllenSDK/cell_types.html)); morpho-electric classification work shows that cell classes differ across electrophysiological and morphological axes ([Gouwens et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31209381/)).
- **Why:** Useful for scoping surrogate generality, but failure is expected unless models are conditioned on relevant physiology and morphology.
- **Status:** Deferred until M3a is interpretable.

#### M4: Structured Intervention and Robustness Probes
- **What:** Test model behavior under biologically meaningful input and parameter interventions, not generic smoothness.
  - Event-level interventions: spike-time jitter, event dropout, burst compression/expansion, rate scaling, and E/I balance shifts.
  - Branch-level interventions: branch-specific drive changes, silent branches, apical/basal redistribution, and synapse-weight perturbations within simulator-plausible ranges.
  - Heterogeneity interventions: evaluate error as simulator parameters move across held-out ranges, especially parameters represented in branch/soma embeddings.
  - Distribution-shift summaries: report whether the surrogate preserves direction, gain, timing, and saturation behavior under each intervention.
- **Credibility sources:** Mechanistic neuron-model validation often uses controlled perturbations and ground-truth electrophysiology rather than generic noise robustness ([mechanistic microcircuit validation with input perturbations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6821748/)); scientific model validation frameworks emphasize experimental-data-driven tests rather than generic scores ([SciUnit docs](https://sciunit.readthedocs.io/en/latest/setup.html)).
- **Why:** Publishable robustness needs to reflect how neuron inputs, morphology, and heterogeneity vary. A global Lipschitz or "stability coefficient" is a poor target because spiking neurons are thresholded, event-driven systems with real discontinuities and high-gain regimes.
- **MVP subset:** Implement one controlled intervention only, preferably event dropout or event-rate scaling. First verify the teacher/data response is interpretable on a small subset; then compare surrogates on direction/gain preservation.
- **Status:** Existing `perturbation_stability.py` is only a sanity-check prototype. Gaussian noise on event inputs may be retained as a smoke test for numerical pathologies, but it is not a publishable benchmark and must not be used as evidence of biological robustness.

#### M5: Closed-Loop Stability (for Paper 2)
- **What:** Run a concrete downstream closed-loop task, such as recurrent spike feedback, conductance feedback, or population rollout, and measure divergence from the teacher under matched feedback rules and seeds.
- **Metrics:** Time-to-divergence, spike-count drift, voltage-distribution drift, synchrony/correlation drift, and runtime.
- **Credibility sources:** Spike-train model goodness-of-fit is commonly assessed with point-process and rescaling ideas rather than only trace RMSE ([population spike-train goodness-of-fit framework](https://link.springer.com/article/10.1007/s10827-018-0698-4)); spike-train similarity and reliability are established concerns in neural coding and model comparison ([Victor 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC2713191/)).
- **Why:** Critical for population simulation validity
- **Status:** Deferred to Paper 2 unless Paper 1 explicitly claims closed-loop simulator replacement.

#### M6: Population Statistics (for Paper 2)
- **What:** Compare teacher and surrogate population behavior under calibrated heterogeneity/noise distributions.
  - Spike-count distribution, firing-rate distribution, pairwise correlations, synchrony, burst statistics, response variability across seeds/heterogeneity.
  - Population-level runtime and memory cost for the target simulator use case.
- **Credibility sources:** Multi-neuron spike-train dissimilarity and synchrony measures are standard enough to be implemented in neuroscience toolkits ([Elephant spike train processing and synchrony docs](https://elephant.readthedocs.io/en/stable/reference/_spike_train_processing.html)); population spike-train model checking has dedicated goodness-of-fit methods ([Tao et al. 2018](https://link.springer.com/article/10.1007/s10827-018-0698-4)).
- **Why:** Population-level validity
- **Status:** Deferred to Paper 2. Do not use until single-cell firing-rate distributions, heterogeneity, and stochasticity are calibrated.

### Design Decisions
- **Lean:** The MVP evaluates M1, lightweight M1b, repaired M2, and one M4 intervention. M3a is deferred unless morphology data are ready. M3b, M5, and M6 are deferred unless they become necessary for the paper's central claim.
- **Reusable:** Benchmark code is a standalone package that can be imported by other groups
- **Allen morphologies:** Use Allen Cell Types Database SWC files; preprocess to match our branch-routing format
- **No global smoothness claim:** Do not frame the benchmark around Lipschitz regularity, global input-output smoothness, or Gaussian noise robustness. Spikes and threshold crossings make those metrics easy to misinterpret.
- **Sanity checks vs claims:** Gaussian perturbation tests can remain in CI or debug reports to catch numerical brittleness. Paper-facing robustness must use simulator-plausible interventions and regime-specific interpretation.
- **Experimental-test framing:** Treat metric suites as data-driven validation tests, closer to SciUnit/NeuronUnit style, not as a generic leaderboard ([SciUnit](https://sciunit.readthedocs.io/en/latest/setup.html), [NeuronUnit](https://neuronunit.readthedocs.io/en/docs/)).

### MVP Guardrails
- [ ] **G1:** If the teacher simulator/data generator still has implausible firing rates or failed input-response checks, label results as teacher-fidelity diagnostics, not biological validation.
- [ ] **G2:** Each MVP metric needs a short "what failure means" note.
- [ ] **G3:** The MVP M4 intervention needs a teacher/data response sanity check before surrogate comparison.
- [ ] **G4:** Aggregate reports should include representative trace IDs or examples for follow-up inspection.
- [ ] **G5:** Anything that takes more than roughly one focused implementation milestone should be deferred unless it is necessary for the immediate model decision.

### Premise Validation Required
- [ ] **P1.1:** Allen L2/3 morphologies are available and compatible with our branch-routing pipeline
- [ ] **P1.2:** Branch-ELM baselines can be evaluated on M1b, M2, M3a, and M4 without architecture-specific hacks
- [ ] **P1.3:** M1b-M4 metrics are computationally feasible at evaluation time (<2x overhead vs M1 for the default benchmark subset)
- [ ] **P1.4:** M4 interventions are defined from simulator/data-generating quantities, with plausible ranges and clear failure criteria before model comparison
- [ ] **P1.5:** M1b feature extraction is robust on the teacher traces and does not fail silently on non-spiking traces

### Benchmark MVP Milestones
- [x] **Milestone A: Report schema and M1 cleanup**
  - Define one JSON/Markdown report format with `metadata`, `m1_trace_fidelity`, `m1b_feature_fidelity`, `m2_regimes`, `m4_intervention`, and `limitations`.
  - Ensure M1 records mask/burn-in semantics, model/config identity, dataset path, and trace IDs.
- [x] **Milestone B: Lightweight M1b**
  - Implement the MVP feature subset without requiring full eFEL integration.
  - Report feature distribution errors plus a few worst trace IDs.
- [x] **Milestone C: M2 repair**
  - Fix trace-boundary handling.
  - Report only the four MVP regimes with support counts and skip/flag under-supported regimes.
- [x] **Milestone D: One M4 intervention**
  - Choose one fast intervention, preferably event dropout or event-rate scaling.
  - Verify teacher/data response direction on a small subset before comparing models.
- [x] **Milestone E: Smoke comparison**
  - Run the MVP benchmark on two representative models/checkpoints.
  - Produce a concise report with metrics, limitations, and the next model decision.

---

## Contribution 1 v1: Deep NeuronIO-like Dataset + Diagnostic Benchmark

### Core Claim
NeuronIO-style surrogate modeling needs a trainable dataset and benchmark that
preserves the old dataset's usability while adding morphology diversity,
teacher provenance, state-aware diagnostics, intervention response tests, and
raw-vs-clipped target transparency. Version 1 should be deep, trainable,
diagnostic-rich, and lean; it should not try to be the broad morphology release.

### Dataset Framing
- **Artifact:** dataset + diagnostic benchmark + generation recipe, packaged
  as one contribution.
- **Scope:** single-neuron surrogate modeling; population simulation and broad
  morphology scaling are future work.
- **Teacher:** CoreNEURON/Hay in the NeuronIO-like dynamic regime, with frozen
  simulator version, mechanisms, routing policy, spike-label policy, clipping
  policy, seed policy, and provenance hashes before public generation.
- **Compatibility:** preserve NeuronIO-style shard access for training
  (`inputs`, `targets`, `sample_ids`, `manifest.json`) while adding richer
  metadata and optional target views. The dataset should be NeuronIO-compatible,
  not NeuronIO-limited.
- **Morphologies:** use `N=5` hand-selected deep morphologies. They should be
  selected for insight and teacher sanity, not for broad generalization claims:
  include a close/control morphology, intermediate shifts, and at least one
  stronger but still sane morphology shift.
- **Broad morphology:** explicitly defer the many-morphology dataset to a
  future v2. A short future pass should try to increase the candidate pool
  beyond the current Allen L5 spiny set; if not, the local `71` dendrite-only
  Allen candidates are acceptable for a tightly framed dendritic-transfer v2.
- **Dendrite-only caveat:** dendrite-only Allen reconstructions are acceptable
  for dendritic input-routing and soma-output surrogate diagnostics, but not
  for full-cell morphology, axonal effects, or full biological reconstruction
  claims.

### Dataset Layout
Use ordinary splits at the top level and keep interventions as an equally
top-level condition category:

```text
dataset/
  train/
  val/
  test/
  private-test/
  interventions/
    event_dropout/
    exc_dropout/
    inh_dropout/
    temporal_jitter/
    site_silence/
  metadata/
```

Default diagnostics run on `val/` across all `5` morphologies. Interventions
are the necessary extra rule: intervention diagnostics read paired traces under
`interventions/<name>/`. Private evaluation should require only an alternate
data path, not a separate benchmark architecture.

### Dataset Size Target
A trace means one full independent `6000 ms` teacher simulation sampled at
`1 ms`, stored as NeuronIO-style `(time, input_channels)` input events and
two-output targets. Local NeuronIO conversion is `38,400` train traces
(`5.4 GB`) and `1,280` validation traces (`220 MB`) for one source morphology.

Version 1 should work backward from the `<200 GB` machine constraint while
reserving headroom for a future broad morphology and heterogeneity/noise
extension. Assume each non-train ordinary split is `1/8` of train, and the
entire intervention category is also `1/8` of train measured in paired examples.

Provisional v1 target:

| category | morphologies | semantic units per morphology | total semantic units |
| --- | ---: | ---: | ---: |
| `train` | `5` | `20,000` traces | `100,000` traces |
| `val` | `5` | `2,500` traces | `12,500` traces |
| `test` | `5` | `2,500` traces | `12,500` traces |
| `private-test` | `5` | `2,500` traces | `12,500` traces |
| `interventions` | `5` | `2,500` paired examples | `12,500` pairs |

At roughly `0.20-0.25 MB` per compressed full trace, ordinary splits are
approximately `25-35 GB`. If paired interventions store both baseline and
intervened traces, `12,500` pairs add roughly `5-7 GB`. Raw soma targets,
metadata, manifests, and provenance should still leave v1 well below `100 GB`,
with `<200 GB` as the hard local cap and `<150 GB` preferred after future
extensions.

### Teacher Validation
Parts of the benchmark should validate the teacher before model claims. These
diagnostics can falsify or sanity-check the teacher; they do not prove
biological correctness by themselves.

Teacher-facing diagnostics should include:
- source-regime match against historical NeuronIO-like statistics;
- voltage/spike distribution sanity per morphology;
- excitatory/inhibitory polarity response;
- intervention response direction and effect size;
- routing and morphology feature sanity;
- raw-vs-clipped target distortion;
- numerical backend agreement where applicable.

Every deep morphology must pass baseline teacher sanity and intervention
teacher sanity before the dataset is frozen.

### Clipping Diagnostic
The dataset must treat voltage clipping as an explicit diagnostic, not a hidden
preprocessing detail. Historical NeuronIO/ELM/Branch-ELM evaluation clips soma
voltage above `-55 mV`, subtracts `-67.7 mV`, and scales by `1/10`. Local
diagnostics found that hard `-55 mV` clipping affects only about `0.21%` of
masked timesteps globally but about `13%` of near-spike timesteps, with
near-spike raw-vs-clipped RMSE about `8.95 mV` and up to about `50 mV` removed
from raw action-potential peaks.

Writing and dataset implications:
- include raw soma voltage and the historical clipped NeuronIO target view;
- report clipped-target RMSE for historical comparability;
- report raw-soma and spike-shape/state diagnostics separately;
- never imply that sub-`1 mV` clipped RMSE proves raw AP peak-shape fidelity.

### Diagnostics Architecture
Diagnostics are atomic computations that emit tidy tabular evidence. Suites
are overlapping bundles of diagnostics, not partitions. A diagnostic can appear
in teacher validation, model evaluation, and paper-summary suites.

Each diagnostic should:
- compute from a declared data selector, usually `split=val` or
  `condition=interventions/<name>`;
- emit a dataframe-compatible table with stable columns such as
  `diagnostic`, `suite`, `model_id`, `teacher_id`, `morphology_id`, `split`,
  `condition`, `intervention`, `trace_id`, `state`, `metric`, `value`, `unit`,
  and `n`;
- include enough metadata to reproduce mask, threshold, clipping, calibration,
  and target-view semantics;
- expose plotters as a variable-length list of atomic evidence views attached
  to the diagnostic, not as fixed attributes such as `plot_main` or
  `plot_breakdown` and not as separate plot suites.

Suite-level plotters are allowed only when they synthesize multiple diagnostic
tables into a specific scientific figure. Example and worst-case figures are a
special case: they should combine multiple per-diagnostic views for the same
trace(s) because their purpose is qualitative inspection. Generic
`main`/`breakdown` plotter categories should be avoided.

Plot quality requires a separate pass after implementation. First implement
correct dataframes and diagnostic semantics; then design data-rich,
publishable figures with a clear reason to exist.

Recommended diagnostic suites:

| suite | default selector | purpose |
| --- | --- | --- |
| `teacher-qc` | `val` + `interventions/*` | verify teacher sanity before model claims |
| `trace-core` | `val` | trace fidelity, spike discrimination, count/timing basics |
| `state-dynamics` | `val` | dense state/regime-conditioned behavior |
| `spike-behavior` | `val` | calibration, timing, count, threshold behavior |
| `feature-fidelity` | `val` | electrophysiology-style feature summaries |
| `morphology-contrast` | `val`, all `N=5` | per-morphology differences and rough uncertainty |
| `intervention-response` | `interventions/*` | paired teacher/model response deltas |
| `paper-summary` | `val` + `interventions/*` | compact paper-facing subset |
| `full` | `val` + `interventions/*` | all practical diagnostics |

Diagnostics should remain dense across metrics, states, and the `N=5`
morphologies by default. Since broad morphology is deferred, pruning is not a
primary concern for v1. Interventions are interpreted as alternative
conditions to the default validation situation, which should keep the code path
cleaner than special-casing old robustness tests.

### Benchmark Completion Criteria
The benchmark is not done when the schema exists. It is done only when the
following package-facing pieces work on real or pilot v1 data:

1. **Data selectors:** one promoted default selector for `val/` across all
   `5` morphologies, plus intervention selectors for
   `interventions/<name>/` paired traces.
2. **Dataframe diagnostics:** every atomic diagnostic in the default suite
   emits tidy dataframe-compatible rows with the declared columns, stable
   mask/threshold/target-view metadata, and no hidden plot-only outputs.
3. **Suite composition:** `teacher-qc`, `paper-summary`, and `full` can be
   run independently and can share diagnostics without treating suites as a
   partition.
4. **Teacher-QC diagnostics:** teacher distribution, polarity response,
   intervention direction/effect size, routing sanity, and raw-vs-clipped
   distortion are computed before model claims.
5. **Model diagnostics:** Mamba, RNN, and Branch-ELM baselines can be evaluated
   through the same dataframe outputs, including learned/calibrated spike
   threshold metadata where used.
6. **Reports:** command-line output writes numeric tables and compact
   Markdown/JSON summaries with trace IDs for follow-up inspection.
7. **Plot hooks:** diagnostics expose variable-length plotter lists, but
   publishable plot quality is explicitly a later pass after numeric semantics
   are correct.

Scheduling: after the current teacher-QC pilot and generation-throughput gate,
finish this benchmark implementation block before full dataset generation,
BSSM, or noise-calibration work. Pilot data are sufficient for implementing and
validating the benchmark code; the full dataset is not required for that step.

### Implementation Plan
1. **Dataset schema and manifest v2**
   - Define trace, morphology, split, condition, intervention-pair, target-view,
     and provenance metadata.
   - Preserve existing NeuronIO-style loading for `inputs` and `targets`.
2. **Morphology selection**
   - Select `N=5` deep morphologies by diversity and teacher sanity.
   - Record dendrite-only/full-cell limitations and routing metadata.
3. **Teacher generator freeze**
   - Freeze CoreNEURON/Hay config, mechanisms, routing, spike labels, clipping,
     seed policy, and provenance hashes.
4. **Teacher QC pilot**
   - Generate small baseline and intervention pilots for all `5` morphologies.
   - Run teacher-facing diagnostics; reject or replace morphologies that fail.
5. **Generation throughput gate**
   - Before generating the full dataset, make the generation path fast enough
     for the planned scale. Prefer GPU/batched simulation where it preserves
     teacher correctness; otherwise produce a measured throughput report and a
     defensible reason for the chosen backend.
   - Do not start full generation on a slow correctness-only path.
   - Current source-Hay/CoreNEURON gate result: promote `8` independent CPU
     CoreNEURON shard workers as the durable local default because measured
     `8 x 6000 ms` generation finished in `28.08 s`, within `10%` of the
     measured `16`-worker peak while preserving headroom. The GPU route is not
     rejected on correctness; it is blocked in the current environment because
     the compiled GPU mechanism requires `libcoreneuron-cuda.so` and no durable
     NEURON GPU runtime library is present. Evidence:
     `artifacts/generation_throughput_gate_20260606.md`.
   - `N=5` morphology generation path status: `artifacts/run_v1_parallel_generation.py`
     now generates v1-shaped ordinary split shards and specimen-coded
     intervention pair shards. All-five smoke validation passed at `1000 ms`
     and `6000 ms`, and `benchmark-v1` consumes the generated layout. Remaining
     backend caveat: the SWC morphology route currently uses plain NEURON
     workers; CoreNEURON/SWC remains a separate backend fix, while the accepted
     near-term acceleration is independent shard parallelism.
6. **Full v1 dataset generation**
   - Generate ordinary splits and intervention pairs under the size target.
   - Store raw and clipped target views plus metadata.
7. **Diagnostic framework**
   - Implement dataframe-first diagnostic outputs, suite composition, selectors,
     and report rendering.
   - Default command: run dense diagnostics on `val` plus intervention
     diagnostics on `interventions/`.
8. **Model evaluation**
   - Evaluate current Mamba, RNN, and Branch-ELM baselines using the same
     diagnostic suite outputs.
9. **Plot-quality pass**
   - After correct outputs exist, design diagnostic-attached and suite-level
     publishable plots. Do not block core implementation on plot polish.
10. **Packaging path**
   - Optimize the future package for ease of use: one promoted default command,
     composable diagnostic suites for researchers, and dataframes as primary
     numeric outputs.

### Remaining Publishability Gaps
- Morphology-transfer evidence is currently a scaffold: normalized routing and
  three held-out morphologies showed useful failure modes, but v1 must validate
  all `5` deep morphologies and phrase claims as deep multi-morphology
  trainability/diagnostics, not broad morphology generalization.
- Intervention evidence is better than the initial smoke test but still needs
  v1-scale paired data and stronger framing. Weak timing perturbations and
  site-level silence should either be redesigned, explicitly labeled as weak
  probes, or excluded from paper-summary claims.
- Teacher correctness claims must remain bounded: diagnostics establish a
  NeuronIO-like CoreNEURON/Hay teacher distribution and sanity under
  perturbations, not biological truth.

---

## Contribution 2: BSSM — Biophysically Structured State-Space Model

**Gate:** Do not start full BSSM implementation until a bounded feasibility
sprint gives positive evidence that the proposed state structure targets a
real failure mode exposed by the v1 diagnostics. The sprint should be small,
predeclared, and allowed to kill or defer BSSM if it does not beat a simpler
alternative.

### Core Claim
State-space models match RNN accuracy when the state transition is structured by biophysical priors (fast/slow E/I currents, membrane integration, adaptation) rather than learned from scratch.

### Architecture Design

```
State vector z_t = [I_fast^E, I_slow^E, I_fast^I, I_slow^I, V_m, adaptation]
  
Evolution equations (discretized):
  I_fast^E[t] = alpha_fast^E * I_fast^E[t-1] + (1-alpha_fast^E) * w_fast^E * sum_E(inputs)
  I_slow^E[t] = alpha_slow^E * I_slow^E[t-1] + (1-alpha_slow^E) * w_slow^E * sum_E(inputs)
  I_fast^I[t] = alpha_fast^I * I_fast^I[t-1] + (1-alpha_fast^I) * w_fast^I * sum_I(inputs)
  I_slow^I[t] = alpha_slow^I * I_slow^I[t-1] + (1-alpha_slow^I) * w_slow^I * sum_I(inputs)
  V_m[t] = f(V_m[t-1], I_fast^E, I_slow^E, I_fast^I, I_slow^I, adaptation, params)
  adaptation[t] = alpha_adapt * adaptation[t-1] + beta_adapt * spike_indicator[t-1]
```

Key design choices:
1. **Learnable timescales:** `alpha_* = exp(-dt / tau_*)` where `tau_*` is learned per state (not fixed log-spaced)
2. **Branch-routing preserved:** Inputs are routed to E/I pools by branch type (same as current)
3. **Small state dimension:** 6 states vs 30-64 in generic Mamba
4. **Nonlinearities are biologically placed:** Voltage-dependent conductances, not arbitrary MLPs
5. **Skip connections:** Direct current-to-readout paths

### Mamba Package Integration Strategy

**Blocker awareness:** We depend on the `mamba_ssm` package for fused kernels. BSSM may not fit their API.

**Options to validate:**
- **Option A (preferred):** Implement BSSM as a custom CUDA kernel or pure PyTorch. If the state dimension is small (6), the unfused cost is acceptable.
- **Option B:** Implement within `mamba_ssm` by passing a custom `A` matrix and `D` skip connection. The package supports learned `A` (diagonal) but not full structured state matrices.
- **Option C:** Hybrid — use `mamba_ssm` for the mixing layer after BSSM state construction.

**Validation needed:** Benchmark forward speed of pure-PyTorch 6-state BSSM vs compact Mamba (74k params). If BSSM forward is within 2x of Mamba at batch 8, Option A is viable.

### Portability to RNN
- The same state structure can be added to our `BranchTraceRNN` as an alternative memory core
- Replace the 4500 synapse traces + 30 hidden units with the 6-state biophysical core
- Test: Does structured-state RNN match or exceed current RNN accuracy with fewer parameters?

### Premise Validation Required
- [ ] **P2.1:** Pure-PyTorch 6-state BSSM forward speed is acceptable (<2x compact Mamba)
- [ ] **P2.2:** Learnable timescales converge to biologically plausible values (ms range)
- [ ] **P2.3:** In the bounded sprint, BSSM improves a predeclared v1
  diagnostic failure mode under the `80k` presentation cap without merely
  improving convergence speed.
- [ ] **P2.4:** State interpretability holds: I_E correlates with excitatory input, I_I with inhibitory, V_m with soma voltage
- [ ] **P2.5:** Structured-state RNN variant improves over current BranchTraceRNN

### Open Design Questions
1. Should state coupling be fully specified (hard-coded ODEs) or partially learned (sparse learnable coupling matrix)?
2. Where to place the MLP/linear readout from state to outputs?
3. Should the soma gate (from Mamba wrappers) be preserved or replaced by the structured V_m state?

---

## Contribution 3: NCS — Noise-Calibrated Surrogates

### Core Claim
Training on mean traces only creates systematic overconfidence. Calibrating output distributions to match biological trial-to-trial variability improves robustness and enables sampling for inference.

### Approach

**Step 1: Characterize biological noise**
- Run Arbor simulator with identical parameters but 10 different random seeds
- Record: voltage variance, spike-time jitter, trial-to-trial correlation
- Estimate noise model parameters as function of regime (resting, spiking, post-input)

**Step 2: Distributional training**
- Model outputs a parametric distribution: p(voltage_t), p(spike_t)
- For voltage: Gaussian with learned mean and variance
- For spike: Bernoulli with learned probability (already done) + calibration
- Loss: negative log-likelihood under the estimated biological noise model

**Step 3: Inference-time sampling**
- At test time, sample from the calibrated distribution
- Enables Monte Carlo estimates of population statistics for Paper 2

### Data Pipeline Blocker

**Problem:** Our current dataset has one trace per parameter set. NCS needs multiple traces per parameter set.

**Solutions to validate:**
- **Option A (preferred):** Generate a small "distributional dataset" with 10 seeds per parameter set. Use this only for noise characterization, not for main training.
- **Option B:** Add noise injection to the simulator (stochastic ion channels, synaptic failures). More realistic but requires mechanism changes.
- **Option C:** Use the existing dataset's heterogeneity variance as a proxy for noise. Less principled but no new data needed.

**Validation needed:** Does Option A's 10-seed dataset capture meaningful trial-to-trial variance, or is the simulator deterministic enough that seeds don't matter?

### Premise Validation Required
- [ ] **P3.1:** Arbor simulator produces non-trivial trial-to-trial variance across seeds
- [ ] **P3.2:** Noise model can be characterized with <100 extra simulation runs
- [ ] **P3.3:** Distributional training improves over point-estimate training
  on a predeclared v1 diagnostic suite.
- [ ] **P3.4:** Calibrated distributions can be sampled at inference time without slowdown

---

## Baselines

1. **Branch-ELM-30** (replicated in our codebase) — primary baseline for accuracy comparison
2. **TCN** — from literature, if we can reproduce or cite
3. **Our BranchTraceRNN** — secondary baseline; improvements from BSSM structure are ported here too
4. **Compact Mamba-1** (with wrappers) — our previous best Mamba variant

---

## Execution Order

### Phase 1: Deep Dataset + Diagnostic Benchmark
1. Finalize dataset schema, manifest v2, and diagnostic dataframe schema.
2. Hand-select `N=5` deep morphologies and generate small teacher-QC pilots
   for baseline and intervention conditions.
3. Freeze the CoreNEURON/Hay teacher config after all five morphologies pass
   baseline and intervention sanity checks.
4. Establish a fast generation path before full generation: GPU/batched if
   correct, or a measured throughput report plus defensible backend choice.
5. Generate v1 ordinary splits and intervention pairs within the size target.
6. Implement dataframe-first diagnostics, overlapping diagnostic suites, and
   the promoted default path: dense diagnostics on `val` plus intervention
   diagnostics under `interventions/`.
7. Evaluate Mamba, RNN, and Branch-ELM baselines with the same diagnostic
   outputs.
8. Run a separate plot-quality pass after the numeric outputs are correct.
9. Write dataset card, benchmark report, limitations, and paper framing.

### Phase 2: BSSM Feasibility Sprint
1. Use v1 diagnostics to identify a concrete failure mode that the proposed
   state structure should fix.
2. Predeclare a bounded sprint: minimal pure-PyTorch prototype, one or two
   baselines, fixed data subset, fixed success/failure criteria, and a fixed
   runtime budget.
3. Continue to full BSSM implementation only if the sprint gives positive
   evidence on accuracy, interpretability, or efficiency that a simpler model
   does not already provide.
4. Port state structure to RNN only if the BSSM result survives this gate.

### Phase 3: Noise Calibration (2-3 weeks)
1. Run a critical evaluation pass before implementation: is there measurable trial-to-trial variability from a plausible stochastic teacher, and is it relevant to the paper's use case?
2. Generate 10-seed distributional dataset only if the teacher/noise premise is credible (validate P3.1)
3. Characterize noise model
4. Implement distributional training only if the predeclared diagnostic target
   is credible.
5. Add sampling inference path only if it improves an identified downstream metric without unacceptable slowdown (P3.4)

### Phase 4: Integration & Paper Writing (2-3 weeks)
1. Full benchmark evaluation of all models
2. Ablation studies
3. Write paper

**Total estimated timeline:** 9-13 weeks

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mamba package incompatible with BSSM | High | Fallback to pure PyTorch; small state dimension keeps cost acceptable |
| Simulator noise is negligible across seeds | Medium | Pivot to Option C (heterogeneity variance as proxy) or reduce NCS scope |
| Deep morphologies fail teacher sanity | High | Replace morphology before full generation; do not freeze a dataset with unsound teacher behavior |
| Dataset exceeds storage/runtime target | High | Reduce train traces per morphology first; preserve all five morphologies and validation/intervention coverage if possible |
| Dendrite-only morphologies are overclaimed | Medium | Phrase v1 as deep multi-morphology surrogate diagnostics, not full-cell biological morphology generalization |
| Intervention probes are weak or ambiguous | Medium | Exclude weak probes from paper-summary claims or redesign them before full generation |
| BSSM accuracy still below Branch-ELM | High | Dataset/benchmark paper stands alone; structured SSM becomes a methodological exploration |
| Scope creep to population sim | High | Strictly defer closed-loop and population statistics to Paper 2 |
| Benchmark code becomes bloated | Medium | Keep diagnostics dataframe-first, suites compositional, and plotting as a separate layer after implementation |

## Critical Evaluation Requirements For Later Batches

The benchmark slop test should be repeated in lighter form before each remaining task batch, but the standard should stay practical: the goal is to catch obvious slop before spending compute, not to require publication-grade validation before every implementation step.

1. **BSSM:** Required. The current BSSM section risks being
   architecture-first. Before implementation, verify that the 6-state
   structure targets an observed failure mode in the v1 diagnostic suites, has
   an interpretable state-to-behavior test, and has clear falsification
   criteria.
2. **Noise calibration:** Required. This is especially vulnerable to sounding biologically sophisticated while measuring simulator artifacts. Before implementation, verify that repeated-seed variability exists, is plausible, and affects a metric practitioners care about.
3. **Baselines:** Required. Each baseline must have a role: accuracy reference, speed reference, morphology/heterogeneity reference, or ablation control. Do not include a baseline only because it is familiar.
4. **Execution/timeline:** Required but lighter. Re-evaluate ordering after the dataset/diagnostic benchmark is resolved, because BSSM/NCS work is premature if the teacher dataset or diagnostic suite is not credible.
