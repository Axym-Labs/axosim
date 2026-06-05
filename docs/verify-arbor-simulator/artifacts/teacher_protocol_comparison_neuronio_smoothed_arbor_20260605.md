# Teacher Protocol Comparison

## Source Protocol Diff

| axis | NeuronIO/Beniaguev | current Arbor |
|---|---|---|
| cell template | Hay et al. L5PC NEURON model, morphology `cell1.asc`, active dendrites, Ih shift 0, SK_E2 scale 1. | Allen L5 template SWC/fit from Arbor single-cell example. |
| synapse mechanisms | ProbAMPANMDA2 for NMDA condition, ProbUDFsyn2 for GABA_A; generator uses Use=1, Dep=0, Fac=0, so the MOD files' release/STP machinery is present but effectively disabled for this dataset. | Separate deterministic Exp2Syn AMPA, sourced NMDA_CA1_pyr_SC mechanism, and Exp2Syn inhibitory synapse. |
| input process | Coarse basal/apical E/I rates per 100 ms, 25-450 ms rate-hold intervals, 25-600 ms Gaussian smoothing, exponential instantaneous probabilities, segment-length weighting. | NeuronIO-like smoothed basal/apical E/I rate-field approximation on the selected Arbor branches, with branch-length weighting. |
| spike labeling | Detect high-resolution local maxima above -25 mV; low-res labels use `(outputSpikeTimes - 0.5).astype(int)`. | Detect threshold crossings of raw soma voltage at a configurable threshold, currently -32 mV in calibrated fixed params. |
| voltage observation | CVode trace interpolated to 1 ms low-res soma voltage, stored as float16, then historical training/eval clips above -55 mV. | Sampled every 1 ms from a fixed-step Arbor integration trace; raw soma stored, clipped target view derived later for comparability. |

## Split Statistics

Diagnostic sample cap: `256` (`null` means all samples).

| metric | NeuronIO/NMDA | Arbor `neuronio_smoothed` target view |
|---|---:|---:|
| samples | 256 | 16 |
| event fraction | 0.010618 | 0.010892 |
| spike rate / 1000 steps | 0.7533 | 9.9896 |
| event count / ms p50/p95/p99 | 13.00/22.00/26.00 | 14.00/22.00/27.00 |
| 100ms total events p50/p95/p99 | 1362.00/1842.00/2176.41 | 1386.00/1845.05/2249.46 |
| 100ms E-I diff p50/p95/p99 | 144.00/387.00/543.41 | 155.00/360.05/551.64 |
| event-count Fano / ms | 1.671 | 1.586 |
| event autocorr lag 100ms | 0.196 | 0.168 |
| channel-rate Gini | 0.133 | 0.381 |
| soma mean/std mV | -67.604/3.036 | -67.706/6.947 |
| |dV/dt| p95/p99 mV/step | 0.438/0.687 | 6.428/39.444 |
| spike-triggered mean peak mV | -55.402 | -17.005 |

## Region Counts

| view | region event counts |
|---|---|
| NeuronIO/NMDA | `{"apical": 10798757.0, "basal": 10044199.0}` |
| current Arbor | `{"apic": 674214.0, "dend": 662105.0}` |

## Interpretation

- The current Arbor teacher is not a direct reimplementation of the NeuronIO/Beniaguev simulator; it is a calibrated proxy with different morphology, channel fit, synapse mechanisms, input process, and spike-label semantics.
- Clipped voltage comparability is valid for old-model RMSE reporting, but it does not by itself make Arbor dynamically equivalent to NeuronIO.
- If the goal is to reduce the Arbor RMSE gap without touching the model, the most defensible path is to port NeuronIO-like protocol pieces into Arbor one at a time: input process/routing first, then spike-label semantics, then closer synapse mechanisms if feasible.
- If those pieces cannot be faithfully ported, the benchmark should report Arbor as a new teacher distribution and treat the RMSE gap as expected teacher/protocol shift.
