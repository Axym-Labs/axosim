# Iteration 200 literature scan: conclusions and post-loop directions

Date: 2026-07-13

## Evidence boundary

I191-I199 falsify learned output delays, sparse branch delays, continuous
quotient residuals, activity-gated local correction, raw future-state
supervision, linear/nonlinear forcing bypasses, and spike-gradient separation
as practical 0.1 mV levers. This scan therefore does not propose I201. It asks
which adjacent 2025-2026 ideas remain genuinely different enough for a future
project, after the current stable checkpoint and evaluation are frozen.

## Selected work

- [State Space Neural Operator](https://openreview.net/forum?id=SwLxxz0x58)
  (TMLR 2026) combines SSM memory with spatial operator structure, adaptive
  damping, and learnable frequency modulation. AxoSim tested a weak local-gate
  translation, not the paper's joint field operator. A faithful future test
  would replace the branch projection/trunk with a joint branch-time operator,
  under an exact global-SSM control; it is too large a change for I200.
- [Expanding the Chaos](https://openreview.net/forum?id=uZWpKAqjEg) (ICML 2026)
  represents stochastic solution operators through orthogonal chaos features.
  AxoBench currently treats each input trace deterministically. This becomes
  relevant only if repeated-noise simulations expose irreducible conditional
  voltage variance; without that audit it would be unjustified capacity.
- [Learning non-Markovian Dynamical Systems with Signature-based
  Encoders](https://proceedings.mlr.press/v277/pradeleix25a.html) (2025) uses
  path signatures for history-dependent dynamics. AxoSim's matched continuity
  and delay tests argue against missing ordinary context, but not against a
  compact nonlinear path summary. A future premise test should first measure
  conditional information in signatures beyond the deployed hidden state.
- [ENMA](https://openreview.net/forum?id=3CYXSMFv55) (NeurIPS 2025 spotlight)
  models parametric PDE trajectories with a masked autoregressive neural
  operator and supports irregular conditioning. Its useful implication is not
  another decoder: evaluate cross-morphology few-shot adaptation as an operator
  family, rather than averaging five systems in one fixed surrogate.
- [LR-RaNN](https://proceedings.mlr.press/v321/liao26a.html) (PMLR 2026)
  emphasizes Lipschitz regularization and explicit generalization bounds for
  system identification. This is relevant to intervention robustness, but it
  should be evaluated as a robustness/generalization protocol, not expected to
  lower in-distribution SERA directly.
- [Flexible and Scalable Deep Dendritic Spiking Neural Networks](https://arxiv.org/abs/2412.06355)
  develops trainable nonlinear dendritic branches for downstream SNNs. It
  motivates transferring AxoSim's validated synaptic efficacy/occupancy
  structure into a new ML architecture, not adding more heads to this neuron
  surrogate.

## Ranked post-loop work

| rank | future project | premise gate | decisive comparison |
| ---: | --- | --- | --- |
| 1 | Joint branch-time state-space neural operator | branch-local residual information remains after the deployed hidden state | joint operator vs equal-budget global SSM and shuffled branch geometry |
| 2 | Cross-morphology operator adaptation/meta-learning | held-out morphology adaptation is a real deployment need | shared model vs few-shot operator conditioning on held-out cells |
| 3 | Stochastic conditional neuron surrogate | repeated simulations show irreducible conditional variance | probabilistic operator vs deterministic AxoMamba on calibration and intervention likelihood |
| 4 | Signature/path-summary encoder | signatures add conditional target information beyond hidden state | signature encoder vs equal-size learned lag encoder |
| 5 | Export validated dendritic structure to downstream SNNs | separate task arc; not a surrogate-SERA claim | AxoSim-derived branch prior vs degree-matched and learned dense SNNs |

## Conclusion

The current loop should stop at I200. The credible remaining ideas change the
problem formulation or the core joint branch-time operator; they are not small
extensions of AxoMamba and cannot be tested honestly as one more 120k addon.
Within the present formulation, raw-target correction remains the only large
late-stage gain, and I191-I199 provide strong negative evidence against local
temporal/output refinements as the route to another 0.1 mV improvement.
