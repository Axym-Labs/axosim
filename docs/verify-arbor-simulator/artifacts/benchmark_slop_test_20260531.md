# Benchmark Slop Test — 2026-05-31

## Question

Would the proposed `NEURO-BENCH` metric suites look meaningful to a skeptical computational neuroscientist or practitioner, or do they mostly sound like plausible ML benchmark language?

## Rubric

A metric suite passes only if it has:

1. **Practitioner-facing construct:** it measures something a simulator user or neuroscientist would recognize as relevant.
2. **Observable anchor:** it can be tied to electrophysiology, morphology, stimulation protocols, or simulator-controlled mechanisms.
3. **Actionability:** a failure tells us what to fix in the surrogate or data generator.
4. **Confound control:** it does not mix morphology, cell type, stimulus distribution, and simulator pathologies into one uninterpretable score.
5. **Claim discipline:** it supports only the claim actually measured.

## Source Grounding

- Allen Cell Types is a useful anchor because it exposes electrophysiology, morphologies/SWC files, and neuronal models under common access patterns.
- eFEL/Blue Brain-style workflows matter because neuron models are commonly validated with electrophysiological features such as spike threshold, AP width, AHP, adaptation, firing rate, and related trace features, not only pointwise RMSE.
- Spike train metrics are a recognized family of analysis tools; metric choice depends on whether timing, rate, synchrony, bursts, or population structure is the target.
- SciUnit/NeuronUnit-style validation is a better conceptual model than a generic leaderboard: tests should be explicit, data-driven, and tied to observations.
- Local simulator verification currently reports pathologically high population spike rates and failed input-response monotonicity. That is a blocking confound for any benchmark claiming biological relevance on the current generated data.

## Verdict By Suite

### M1: Open-Loop Trace Fidelity

**Verdict:** Useful but not sufficient. Keep it, but do not sell it as neuroscience-validating by itself.

**Why it passes partly:** Voltage trace fidelity, spike AUC, and spike timing are directly relevant to surrogate accuracy.

**Why it is currently slop-prone:** RMSE and AUC can hide whether the model gets electrophysiological behavior wrong. A low RMSE model can still have wrong spike threshold, AP width, adaptation, firing-rate response, afterhyperpolarization, or burst structure. AUC is also not the same as calibrated spike timing or spike-count fidelity.

**Repair:** Add an `M1b` feature-fidelity layer:
- eFEL-style feature errors: firing rate, first/last ISI, spike threshold, AP width/amplitude, AHP depth, adaptation index.
- Spike-train distance at relevant time scales: Victor-Purpura / van Rossum / SPIKE-distance style summaries, chosen by the question.
- Calibration: reliability diagram or Brier/log loss for spike probabilities if probabilities are interpreted probabilistically.

### M2: Regime-Stratified Errors

**Verdict:** Potentially meaningful, but only if regimes are biophysically defined.

**Why it passes partly:** Practitioners do care whether a model fails near spikes, during high slope periods, after input events, or in subthreshold regimes.

**Why it is currently slop-prone:** Quantile voltage bins and flattened trace windows can become arbitrary buckets. "Near spike" errors need trace-aware windows, not flattened indices that accidentally cross samples. Regimes also need to reflect absolute voltage/time/stimulus concepts, not only distribution-dependent bins.

**Repair:**
- Define regimes in physical units: subthreshold, threshold approach, spike, refractory/AHP, burst, high-conductance/post-event.
- Keep trace boundaries explicit.
- Report support/counts and uncertainty per regime.
- Use M2 to diagnose model failure, not as a single leaderboard score.

### M3: Morphology Transfer

**Verdict:** High value if done carefully; otherwise highly confounded.

**Why it passes:** Generalization across morphology is a real practitioner concern for surrogate simulators. Allen SWC morphologies provide a credible source of morphology variation.

**Why it is currently slop-prone:** "Train L5, test L2/3" mixes morphology, cell type, channel distributions, input statistics, and target firing regime. A padding adapter that makes branch counts match may create an ML compatibility score rather than a neuroscience transfer test.

**Repair:**
- Separate same-cell-type morphology variation from cross-cell-type transfer.
- Hold stimulation protocol and mechanism distributions fixed where possible.
- Report transfer by interpretable morphology features: branch count, total length, apical/basal distribution, electrotonic distance proxies.
- If testing L5 to L2/3, call it cross-cell-type transfer and treat failure as expected unless the model is explicitly conditioned on physiology.

### M4: Structured Intervention and Robustness Probes

**Verdict:** Good direction after the revision, but not yet a metric until interventions have expected-response checks.

**Why it passes:** Event timing/dropout/rate, E/I balance, branch-specific input, synaptic weights, and heterogeneity parameters are all closer to practitioner-relevant perturbations than Gaussian noise or Lipschitz coefficients.

**Why it is still slop-prone:** A list of interventions is not automatically meaningful. Each intervention needs a simulator-grounded expected effect. The local simulator currently failed a simple input-rate monotonicity check, so some intervention tests may be invalid until the data generator is calibrated.

**Repair:**
- For each intervention, state expected qualitative direction before model comparison.
- Include teacher-simulator response curves, not just surrogate response curves.
- Start with controlled one-factor interventions where all other heterogeneity is fixed.
- Use multi-factor interventions only after one-factor checks pass.

### M5: Closed-Loop Stability

**Verdict:** Potentially the most practitioner-relevant metric for simulator replacement, but dangerous if underspecified.

**Why it passes:** If surrogates are used in recurrent or population simulations, closed-loop error accumulation matters more than one-window open-loop RMSE.

**Why it is slop-prone:** "Feed output back as input with probability p" is artificial unless it corresponds to a real downstream simulator wiring or use case. Time-to-divergence depends strongly on feedback rule, threshold, and stochasticity.

**Repair:**
- Define one concrete closed-loop use case first: recurrent spike feedback, conductance feedback, or population rollout.
- Compare teacher vs surrogate under identical feedback and random seeds.
- Report spike-count drift, voltage distribution drift, synchrony/correlation drift, and runtime.
- Treat this as Paper 2 only unless the closed-loop task is already central to Paper 1.

### M6: Population Statistics

**Verdict:** Meaningful for the stated long-term goal, but too broad for the current single-neuron paper.

**Why it passes:** Spike-count variance, pairwise correlations, synchrony, population rates, and response distributions are real practitioner concerns.

**Why it is slop-prone:** "100 identical-input neurons" may measure stochastic sampling behavior rather than model validity if the surrogate is deterministic or if heterogeneity/noise is not calibrated. Population metrics require validated single-cell and noise/heterogeneity distributions first.

**Repair:**
- Defer as planned.
- Before using it, calibrate single-cell firing-rate distributions and heterogeneity ranges.
- Use population metrics only after teacher simulator statistics are themselves plausible.

## Overall Slop Test Result

**M1:** Keep, but upgrade with electrophysiological feature fidelity and spike-train distances.

**M2:** Keep, but replace arbitrary bins with trace-aware, biophysically defined regimes.

**M3:** Keep, but narrow and deconfound. Same-type morphology transfer is stronger than broad L5 to L2/3 transfer.

**M4:** Keep the revised direction, but require controlled simulator response curves and expected-effect checks.

**M5:** High value, but only with a concrete downstream closed-loop task.

**M6:** High value for Paper 2, not ready for Paper 1 claims.

## Recommended Spec Changes

1. Add `M1b: Electrophysiological Feature Fidelity`.
2. Rename M2 to `Trace- and Regime-Aware Failure Diagnostics`, emphasizing diagnostic use rather than a scalar score.
3. Split M3 into `M3a same-cell-type morphology transfer` and `M3b cross-cell-type transfer`.
4. Add a gate before M4: teacher simulator intervention responses must be plausible and monotonic/qualitatively expected under controlled one-factor changes.
5. Keep M5-M6 deferred unless Paper 1 explicitly targets simulator replacement in closed-loop population use.

## Hard Blocker

The current local simulator/data generator cannot support practitioner-facing benchmark claims until the pathological firing-rate distribution and failed input-rate response check are resolved or scoped out. Otherwise the benchmark may measure fidelity to a broken teacher rather than useful neuron behavior.
