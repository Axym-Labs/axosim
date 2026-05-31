# Publishable Direction Brainstorm: Mamba Surrogate for Biological Neurons

## Context

- **Current best:** Branch-ELM (RNN-based) achieves ~0.76 mV RMSE, ~0.99 AUC on open-loop trace prediction
- **Mamba gap:** Mamba-1 with wrappers reaches ~0.85 mV RMSE but requires 3-5x more presentations. Diagnostic evidence shows Mamba's final features collapse to a low-rank soma-like manifold (participation rank ~1.8 vs RNN memory rank ~4.6).
- **Root cause identified:** Mamba lacks explicit recurrent state construction. It smooths and mixes but doesn't maintain separate fast/slow excitatory/inhibitory current states or a true membrane-like integration variable.
- **User novel idea:** Introducing varying noise levels to match biological neuron noise and correct behavioral biases.

---

## Assessment: What Would Be Publishable?

The Branch-ELM paper established that RNNs can surrogate biophysical neurons. The TCN paper showed temporal convolutions work. To publish, we need a **genuine advance** beyond "we tried Mamba and it works too."

### Current State: Not Yet Publishable

- Mamba is faster but less accurate → engineering observation, not a contribution
- Mamba state collapses → diagnostic finding, but needs a solution
- Extensive ablations → useful internally, but ablation studies alone rarely drive publications

---

## Direction 1: Biophysically Structured State-Space Models (Strongest)

### Core Claim
**State-space models can match RNN accuracy *if* the state transition is structured by biophysical priors, not learned from scratch.**

### Why This Is Novel
- Prior SSM/Mamba work uses generic diagonal/HiPPO transitions
- Prior neuron surrogates use black-box RNNs (Branch-ELM) or convolutions (TCN)
- No prior work has designed SSM state matrices to enforce fast/slow currents, E/I separation, and membrane integration

### Proposed Architecture
Replace generic Mamba with a **Biophysical State-Space Model (BSSM)**:

```
State vector z_t = [I_fast^E, I_slow^E, I_fast^I, I_slow^I, V_m, adaptation]

Each state evolves as:
  dI_fast^E/dt = -I_fast^E / tau_fast^E + w_fast^E * input^E(t)
  dI_slow^E/dt = -I_slow^E / tau_slow^E + w_slow^E * input^E(t)
  dI_fast^I/dt = -I_fast^I / tau_fast^I + w_fast^I * input^I(t)
  dI_slow^I/dt = -I_slow^I / tau_slow^I + w_slow^I * input^I(t)
  dV_m/dt = -(V_m - E_leak)/tau_m + g_E*(E_E - V_m) + g_I*(E_I - V_m) + I_inject
```

Key design choices:
- **Timescales are learnable** (not fixed like Branch-ELM's log-spaced grid)
- **State dimension is small** (6-10 states, not 30-64)
- **Nonlinearities are placed where biology places them** (voltage-dependent conductances, not arbitrary MLPs)
- **Skip connections from currents to readout** (direct branch-current readout, not just hidden state)

### What Makes This Different from Previous Attempts
- Previous Mamba wrappers added branch-gain, tanh, soma-gate as **post-hoc corrections**
- BSSM builds the structure **into the recurrence itself**
- Previous RNN work used 4500 synapse traces + 30 memory units
- BSSM uses 6-10 physically interpretable states

### Evaluation Hook
- Match Branch-ELM accuracy with **fewer parameters and faster inference**
- Ablation: generic SSM vs structured SSM vs RNN → structured SSM wins on accuracy/parameter tradeoff
- Interpretability: inspect learned timescales, E/I balance, adaptation → should correlate with biophysical parameters

### Risk
- The structure might be too rigid → need learnable coupling between states
- If it still underperforms RNN, we learn something about what's truly needed

---

## Direction 2: Noise-Calibrated Surrogates for Closed-Loop Stability

### Core Claim
**Training surrogates on mean traces only creates a systematic bias: the model is overconfident and unstable in closed-loop feedback. Calibrating output noise to match biological variability fixes this.**

### Why This Is Novel
- All prior surrogate work (Branch-ELM, TCN, our Mamba) minimizes mean-squared error on deterministic traces
- Biological neurons are stochastic: repeated identical inputs produce different spike times
- No prior work has trained surrogates to match the **distribution** of responses, not just the mean

### Proposed Method

**Step 1: Quantify biological noise**
- Run the Arbor simulator with multiple seeds for identical inputs
- Measure spike-time jitter, voltage variance, and trial-to-trial correlation
- Characterize noise as function of regime (resting, spiking, post-input)

**Step 2: Noise-aware training**
- Model outputs a distribution: p(spike_t), p(voltage_t) rather than point estimates
- Loss becomes negative log-likelihood under the biological noise model
- Train to match both mean and variance

**Step 3: Behavioral bias correction**
- If the model systematically underestimates spike probability near threshold (diagnostic evidence suggests this), add a learned bias correction
- Use the noise model to detect when the model is "too certain"

### What Makes This Different from User's Raw Idea
- Raw idea: "add noise to match biological noise"
- Refined idea: **learn the noise model jointly with the surrogate**, use it for training and inference
- Raw idea alone is just data augmentation; refined idea is a methodological contribution

### Evaluation Hook
- Open-loop: mean RMSE/AUC (should be comparable)
- **Closed-loop:** feedback a fraction of output spikes back as input. Measure divergence from ground truth over time.
- **Population coherence:** simulate 1000 neurons with shared input. Measure population statistics vs ground truth.
- The closed-loop test is the real contribution — no prior surrogate paper evaluates this.

### Risk
- Need multiple stochastic simulator runs → compute cost
- May need simulator to support noise injection (ion channel stochasticity, synaptic variability)
- If closed-loop doesn't diverge, the problem is less severe than hypothesized

---

## Direction 3: A New Benchmark Task

### Core Claim
**Current benchmarks (RMSE, AUC on 500-step windows) are insufficient. We propose a multi-scale fidelity benchmark that exposes where surrogates fail.**

### Why This Is Novel
- Branch-ELM, TCN, and all our work evaluate on the same task: predict 500-step traces
- No standard benchmark evaluates: closed-loop stability, population statistics, spike-timing precision, or regime-specific errors

### Proposed Benchmark Components

1. **Single-neuron trace fidelity** (existing)
   - RMSE, AUC, spike timing precision

2. **Regime-stratified errors** (partially done in our diagnostics)
   - Error near spikes vs far from spikes
   - Error during high dV/dt vs low dV/dt
   - Error in different voltage ranges

3. **Closed-loop stability** (novel)
   - Feed surrogate output back as input with some probability
   - Measure how long before trajectory diverges from ground truth
   - Report mean time-to-divergence

4. **Population statistics** (novel)
   - Simulate 100 identical-input neurons, measure spike count variance
   - Compare surrogate population vs simulator population

5. **Transfer to unseen morphologies** (novel)
   - Train on L5 pyramidal, test on L2/3 or interneuron morphologies
   - Measure generalization

### What Makes This Publishable
- Even if the model is only moderately improved, a rigorous benchmark is a genuine contribution
- The community needs this
- Positions us as methodological leaders

### Risk
- Building the benchmark takes time
- Need to ensure the benchmark is fair and not gamed

---

## Direction 4: Hybrid Architecture (Pragmatic)

### Core Claim
**The best surrogate uses Mamba for fast open-loop inference and an explicit-state RNN for closed-loop accuracy, with a learned distillation path between them.**

### Architecture
- **Teacher:** Branch-ELM RNN (accurate, slow in open-loop)
- **Student:** Mamba variant (fast, less accurate)
- **Distillation loss:** Student mimics not just final output but also RNN hidden state trajectory
- **Switching:** For open-loop batch inference, use Mamba. For closed-loop online simulation, use RNN.

### Why This Is Publishable
- Demonstrates a deployment-aware design: different modes need different architectures
- The distillation from RNN state to Mamba features is a technical contribution
- Addresses the throughput vs accuracy tradeoff explicitly

### Risk
- Might be seen as engineering, not science
- Need to show the hybrid outperforms either alone in some metric

---

## Recommendation

**Primary direction: Direction 1 (Biophysically Structured SSM) + Direction 2 (Noise Calibration)**

These two combine naturally:
1. The structured SSM gives us an interpretable, efficient model
2. The noise calibration makes it robust and biologically realistic
3. Together they address the core limitation of current approaches: black-box models that don't know about currents, membranes, or variability

### Concrete Next Steps

1. **Design the BSSM core**
   - Implement the 6-state fast/slow E/I + membrane + adaptation model
   - Make timescales learnable
   - Add branch-routing (same as current)
   - Test against compact Mamba and RNN at 40k presentations

2. **Build the noise model**
   - Run Arbor with multiple seeds for identical parameters
   - Characterize voltage variance and spike-time jitter
   - Implement noise-aware loss

3. **Closed-loop validation**
   - Implement feedback loop: surrogate output → input with probability p
   - Measure divergence time vs ground truth
   - Compare deterministic vs noise-calibrated surrogates

4. **If BSSM works:** paper is "Biophysically Structured State-Space Models for Efficient and Accurate Neuron Surrogates"
5. **If BSSM doesn't close gap:** pivot to benchmark paper with noise calibration as the advance

---

## Open Questions for Discussion

1. Should the state structure be fully specified (hard-coded ODEs) or partially learned (state matrix with sparsity constraints)?
2. Is the noise model Gaussian, or do we need something heavier-tailed for spike-time jitter?
3. Do we have access to real biological recordings for validation, or only simulator outputs?
4. Should we compare against the official Branch-ELM-30 or our improved RNN variant?
