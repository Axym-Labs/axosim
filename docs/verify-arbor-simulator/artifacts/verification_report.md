# Arbor Simulator Verification Report

## Date: 2026-05-30

---

## T1: Smoke Test

**Status: PASSED**

- Built Arbor 0.11.0 from source with CUDA support (sm_90, CUDA 12.4)
- GPU context: has_gpu=True
- Generated 2 samples, 100 steps, 8 branches in 5.6s
- No crashes, output shapes correct

---

## T2: Biophysical Plausibility

**Status: PARTIAL - Individual traces plausible, population rates pathological**

### Individual Trace Analysis
- Resting potential: ~-67.6 mV (biologically reasonable)
- Spike peaks: up to +3.8 mV (sample 3), with clear threshold crossings
- Spike detection works correctly: voltage rises from ~-67 mV to positive values at spike times
- No pathological voltage values (all within -92 to +18 mV range)

### Population Spike Rate Distribution (train split, 256 samples)
- Mean: 70.6 Hz
- Median: 70.6 Hz
- Min: 0.0 Hz, Max: 176.2 Hz
- 95th percentile: 141.0 Hz
- 62.9% of samples fire >50 Hz
- 28.9% of samples fire >100 Hz

**Concern:** These rates are pathologically high for cortical pyramidal neurons, which typically fire at 0.5-20 Hz. The Allen L5 template should not produce such high rates under normal conditions.

### Parameter Analysis
High-firing cells (>100 Hz) vs low-firing cells (<10 Hz):
- High firing: lower inhibition (inh_weight_scale: 0.0057 vs 0.0090), more positive E/I balance (0.0485 vs -0.1231)
- Input rate scale is NOT the main driver (correlation with spike rate: -0.005)

---

## T3: Heterogeneity Sensitivity

**Status: PASSED - Parameters produce measurable differences**

Compared high vs low input rate scale (1.75 vs 0.55):
- High input: 246,169 input events, mean spike rate 49.2 Hz
- Low input: 34,741 input events, mean spike rate 72.8 Hz
- Input event counts differ by 7x as expected
- Traces are meaningfully different (different spike patterns)

Note: The relationship between input rate and spike rate is not monotonic due to the interaction with other random parameters (E/I balance, synaptic weights, etc.).

---

## T4: Mechanism Verification

**Status: PASSED - NMDA has measurable effect**

NMDA on vs NMDA off (nmda_weight_scale=0):
- Mean absolute difference: 15.42 mV
- Maximum difference: 84.20 mV
- NMDA clearly affects membrane potential dynamics

---

## T5: Input-Response Consistency

**Status: FAILED - No monotonic relationship**

Correlation between input_rate_scale and spike_rate across 256 samples: **-0.005**

This near-zero correlation is unexpected. Possible causes:
1. Other parameters (E/I balance, synaptic weights) dominate over input rate
2. The input generation may not properly scale with input_rate_scale
3. The cell may be in a saturated regime where additional input doesn't increase firing

**This needs investigation before proceeding to Step 3.**

---

## T6: Edge Cases

**Status: PASSED**

- No NaN values in outputs
- No Inf values in outputs
- Voltage range: -92.5 mV to +18.3 mV (extreme but not pathological)
- Spike signal: properly bounded [0, 1]

---

## Summary and Recommendations

### Blockers for Step 3
1. **Pathological spike rates:** 63% of cells fire >50 Hz. This suggests the simulator is in a failure mode or the parameter distributions are not biologically calibrated.
2. **Input-response non-monotonicity:** The near-zero correlation between input_rate_scale and spike_rate indicates a potential issue with how inputs drive the cell.

### Recommendations
1. Investigate why spike rates are so high - check if the Allen L5 template with these synaptic weights and input rates is expected to fire at 70+ Hz
2. Verify that input_rate_scale actually affects the Poisson event generation rate
3. Consider narrowing the parameter ranges or adding stronger homeostatic constraints
4. Run a controlled experiment: fix all parameters except input_rate_scale and verify monotonicity

### Non-blockers
- GPU build and execution work correctly
- Individual traces show biophysically plausible spike shapes
- NMDA mechanism has expected effect
- No crashes, NaNs, or silent failures
