# Spec: Publishable Mamba Surrogate Paper

> **Scope:** Single-neuron surrogate models. Population-level simulation is Paper 2.
> **Date:** 2026-05-30
> **Order of contributions:** 1) Benchmark/Metric Suite → 2) Biophysically Structured SSMs → 3) Noise Calibration

---

## Contribution 1: NEURO-BENCH — Neuron Surrogate Benchmark Suite

### Core Claim
Current surrogate evaluation (RMSE + AUC on 500-step windows) is insufficient. We propose a lean, reusable benchmark that exposes where models fail. The first implementation should be an MVP for model iteration, not a full publication-grade benchmark.

### Implementation Scope
- **MVP target:** M1, a lightweight M1b subset, repaired M2, and one simple M4 intervention.
- **Deferred:** M3 morphology transfer, M5 closed-loop stability, and M6 population statistics unless they become necessary for the immediate model decision.
- **Known weakness:** If the current teacher simulator/data generator still has pathological firing rates or failed input-response checks, the benchmark can only claim fidelity to that teacher/data distribution, not biological validity.

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
- [ ] **Milestone A: Report schema and M1 cleanup**
  - Define one JSON/Markdown report format with `metadata`, `m1_trace_fidelity`, `m1b_feature_fidelity`, `m2_regimes`, `m4_intervention`, and `limitations`.
  - Ensure M1 records mask/burn-in semantics, model/config identity, dataset path, and trace IDs.
- [ ] **Milestone B: Lightweight M1b**
  - Implement the MVP feature subset without requiring full eFEL integration.
  - Report feature distribution errors plus a few worst trace IDs.
- [ ] **Milestone C: M2 repair**
  - Fix trace-boundary handling.
  - Report only the four MVP regimes with support counts and skip/flag under-supported regimes.
- [ ] **Milestone D: One M4 intervention**
  - Choose one fast intervention, preferably event dropout or event-rate scaling.
  - Verify teacher/data response direction on a small subset before comparing models.
- [ ] **Milestone E: Smoke comparison**
  - Run the MVP benchmark on two representative models/checkpoints.
  - Produce a concise report with metrics, limitations, and the next model decision.

---

## Contribution 2: BSSM — Biophysically Structured State-Space Model

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
- [ ] **P2.3:** BSSM reaches RMSE <1.0 mV at 800k presentations ( Branch-ELM territory )
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
- [ ] **P3.3:** Distributional training improves over point-estimate training on M1 metrics
- [ ] **P3.4:** Calibrated distributions can be sampled at inference time without slowdown

---

## Baselines

1. **Branch-ELM-30** (replicated in our codebase) — primary baseline for accuracy comparison
2. **TCN** — from literature, if we can reproduce or cite
3. **Our BranchTraceRNN** — secondary baseline; improvements from BSSM structure are ported here too
4. **Compact Mamba-1** (with wrappers) — our previous best Mamba variant

---

## Execution Order

### Phase 1: Benchmark MVP
1. Complete Milestone A: report schema and M1 cleanup
2. Complete Milestone B: lightweight M1b feature fidelity
3. Complete Milestone C: M2 trace-boundary repair and four-regime diagnostic
4. Complete Milestone D: one controlled M4 intervention with teacher/data sanity check
5. Complete Milestone E: smoke comparison on two representative models/checkpoints
6. Record limitations clearly, especially whether results are biological claims or only teacher-fidelity diagnostics
7. Defer M3/M5/M6 unless the smoke comparison shows they are needed for the immediate model decision

### Phase 2: BSSM Design & Validation (3-4 weeks)
1. Run a critical evaluation pass before implementation: does the proposed 6-state structure measure a real bottleneck in current models, and what result would falsify it?
2. Implement pure-PyTorch 6-state BSSM core
3. Benchmark forward speed (validate P2.1)
4. Train on standard dataset, check convergence and interpretability (P2.2-P2.4)
5. Port state structure to RNN only if the BSSM result survives the critical evaluation gate (P2.5)

### Phase 3: Noise Calibration (2-3 weeks)
1. Run a critical evaluation pass before implementation: is there measurable trial-to-trial variability from a plausible stochastic teacher, and is it relevant to the paper's use case?
2. Generate 10-seed distributional dataset only if the teacher/noise premise is credible (validate P3.1)
3. Characterize noise model
4. Implement distributional training (validate P3.3)
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
| Allen morphologies incompatible with branch router | Medium | Simplify morphology to compatible subset; document limitation |
| BSSM accuracy still below Branch-ELM | High | Benchmark paper stands alone; structured SSM becomes a methodological exploration |
| Scope creep to population sim | High | Strictly defer M5-M6 to Paper 2; include in benchmark release only |
| Benchmark MVP grows into full validation suite | High | Stop at M1/lightweight M1b/M2/one M4 intervention unless a milestone fails for a specific reason |

## Critical Evaluation Requirements For Later Batches

The benchmark slop test should be repeated in lighter form before each remaining task batch, but the standard should stay practical: the goal is to catch obvious slop before spending compute, not to require publication-grade validation before every implementation step.

1. **BSSM:** Required. The current BSSM section risks being architecture-first. Before implementation, verify that the 6-state structure targets an observed failure mode in M1b/M2/M4, has an interpretable state-to-behavior test, and has clear falsification criteria.
2. **Noise calibration:** Required. This is especially vulnerable to sounding biologically sophisticated while measuring simulator artifacts. Before implementation, verify that repeated-seed variability exists, is plausible, and affects a metric practitioners care about.
3. **Baselines:** Required. Each baseline must have a role: accuracy reference, speed reference, morphology/heterogeneity reference, or ablation control. Do not include a baseline only because it is familiar.
4. **Execution/timeline:** Required but lighter. Re-evaluate ordering after the benchmark blocker is resolved, because BSSM/NCS work is premature if the teacher simulator or metric suite is not credible.
