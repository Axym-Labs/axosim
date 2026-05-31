# ARCHIVED — Simulator specs deferred to future work

> **Note:** This specification is archived. The Arbor simulator verification work is paused. The simulator specs remain valid for when we return to heterogeneous neuron simulation, but our current priority is publishable results on the Mamba surrogate model.

---

# Verify Arbor Simulator (Step 2 Closure)

## Objective
Verify the Arbor heterogeneous neuron simulator is working correctly and not in a failure mode before proceeding to Step 3 (model heterogeneity integration).

## Background
The simulator uses Arbor 0.11.0 (CUDA-built) with:
- Allen L5 morphology (`allen_l5_template.swc`)
- Allen active channel catalogue (NaV, K families, Ih, Im, Ca, SK, CaDynamics)
- Custom NMDA mechanism from ModelDB 2019343 (`NMDA_CA1_pyr_SC`)
- 46 heterogeneity parameters controlling physiology and stimulus
- 45 selected branches (basal/apical) with AMPA+NMDA excitatory and GABA-like inhibitory synapses

## Verification Tasks

### T1. Smoke Test
- Generate a small dataset on GPU (8 samples, 6000 steps)
- Verify no crashes, reasonable runtime, GPU utilization
- Check output shapes and dtypes

### T2. Biophysical Plausibility
- Inspect soma voltage traces for:
  - Resting potential range (-70 to -50 mV typical)
  - Spike amplitudes (>0 mV peak typical)
  - No pathological values (e.g., all -100 mV or all +50 mV)
  - Spike rates in biologically plausible range (0.5–100 Hz)
- Compare with known Allen L5 pyramidal cell behavior

### T3. Heterogeneity Sensitivity
- Generate pairs of cells with identical seeds but one parameter changed:
  - High vs low input rate
  - High vs low leak conductance
  - High vs low NaV scale
  - Excitatory vs inhibitory dominant
- Verify traces are meaningfully different

### T4. Mechanism Verification
- Test with NMDA disabled (weight = 0) vs enabled
- Test with active channels disabled (all scales = 0) → passive cell
- Verify expected behavioral changes

### T5. Input-Response Consistency
- Generate cells with varying input rates
- Verify spike rate increases monotonically with input rate
- Verify E/I balance shifts spike probability correctly

### T6. Edge Cases
- Extreme parameters (very high/low conductances)
- Very sparse vs dense inputs
- Verify no NaN/Inf in outputs

## Acceptance Criteria
- All smoke tests pass on GPU
- Voltage traces are biophysically plausible
- Heterogeneity parameters have measurable effects on outputs
- Mechanism disabling produces expected changes
- No silent failures, NaNs, or pathological edge-case behavior
- Documented findings with plots/artifacts in `docs/verify-arbor-simulator/artifacts/`
