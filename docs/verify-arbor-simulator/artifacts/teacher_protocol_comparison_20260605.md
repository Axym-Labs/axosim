# Teacher Protocol Comparison

## Source Protocol Diff

| axis | NeuronIO/Beniaguev | current Arbor |
|---|---|---|
| cell template | Hay et al. L5PC NEURON model, morphology `cell1.asc`, active dendrites, Ih shift 0, SK_E2 scale 1. | Allen L5 template SWC/fit from Arbor single-cell example. |
| synapse mechanisms | ProbAMPANMDA2 for NMDA condition, ProbUDFsyn2 for GABA_A; generator uses Use=1, Dep=0, Fac=0, so the MOD files' release/STP machinery is present but effectively disabled for this dataset. | Separate deterministic Exp2Syn AMPA, sourced NMDA_CA1_pyr_SC mechanism, and Exp2Syn inhibitory synapse. |
| input process | Coarse basal/apical E/I rates per 100 ms, 25-450 ms rate-hold intervals, 25-600 ms Gaussian smoothing, exponential instantaneous probabilities, segment-length weighting. | Per-ms Poisson event count over all channels with branch weights, E/I balance, apic fraction, and occasional same-branch synchrony. |
| spike labeling | Detect high-resolution local maxima above -25 mV; low-res labels use `(outputSpikeTimes - 0.5).astype(int)`. | Detect threshold crossings of raw soma voltage at a configurable threshold, currently -32 mV in calibrated fixed params. |
| voltage observation | CVode trace interpolated to 1 ms low-res soma voltage, stored as float16, then historical training/eval clips above -55 mV. | Sampled every 1 ms from a fixed-step Arbor integration trace; raw soma stored, clipped target view derived later for comparability. |

## Split Statistics

Diagnostic sample cap: `256` (`null` means all samples).

| metric | NeuronIO/NMDA | Arbor `arbor_poisson` target view |
|---|---:|---:|
| samples | 256 | 64 |
| event fraction | 0.010618 | 0.009997 |
| spike rate / 1000 steps | 0.7533 | 0.7292 |
| event count / ms p50/p95/p99 | 13.00/22.00/26.00 | 13.00/19.00/22.00 |
| 100ms total events p50/p95/p99 | 1362.00/1842.00/2176.41 | 1277.00/1336.00/1359.00 |
| 100ms E-I diff p50/p95/p99 | 144.00/387.00/543.41 | 0.00/60.00/82.00 |
| event-count Fano / ms | 1.671 | 0.999 |
| event autocorr lag 100ms | 0.196 | -0.000 |
| channel-rate Gini | 0.133 | 0.048 |
| soma mean/std mV | -67.604/3.036 | -70.067/2.850 |
| |dV/dt| p95/p99 mV/step | 0.438/0.687 | 3.531/5.638 |
| spike-triggered mean peak mV | -55.402 | -55.000 |

## Region Counts

| view | region event counts |
|---|---|
| NeuronIO/NMDA | `{"apical": 10798757.0, "basal": 10044199.0}` |
| current Arbor | `{"apic": 2901336.0, "dend": 2004904.0}` |

## Interpretation

- The current Arbor teacher is not a direct reimplementation of the NeuronIO/Beniaguev simulator; it is a calibrated proxy with different morphology, channel fit, synapse mechanisms, input process, and spike-label semantics.
- Clipped voltage comparability is valid for old-model RMSE reporting, but it does not by itself make Arbor dynamically equivalent to NeuronIO.
- If the goal is to reduce the Arbor RMSE gap without touching the model, the most defensible path is to port NeuronIO-like protocol pieces into Arbor one at a time: input process/routing first, then spike-label semantics, then closer synapse mechanisms if feasible.
- If those pieces cannot be faithfully ported, the benchmark should report Arbor as a new teacher distribution and treat the RMSE gap as expected teacher/protocol shift.
