# Paper Writing Notes

## NeuronIO Voltage Target Clipping

Important paper-writing caveat: the ELM / Branch-ELM NeuronIO setup reports
voltage RMSE on a clipped somatic-voltage target, not on the full raw
action-potential peak waveform.

Evidence:

- The local copy of the ELM paper, `.untracked/papers/elm_neuron.pdf`,
  Appendix B, states that NeuronIO training and evaluation cap somatic voltage
  at `-55 mV`, subtract `-67.7 mV`, and scale voltage by `1/10` for training.
  The paragraph says this follows Beniaguev et al. (2021).
- The same paper's NeuronIO figures/tables include Branch-ELM results
  (e.g. Figure 2 text and Table S8), so this is directly relevant to the
  Branch-ELM comparison, not only to vanilla ELM.
- The public upstream `elmneuron` code uses the same constants in
  `src/neuronio/neuronio_data_utils.py`:
  `DEFAULT_Y_TRAIN_SOMA_BIAS = -67.7`,
  `DEFAULT_Y_SOMA_THRESHOLD = -55.0`, and
  `DEFAULT_Y_TRAIN_SOMA_SCALE = 1 / 10`.
- The upstream repository README/model docs state that the included pretrained
  NeuronIO models are various-sized Branch-ELM models.
- Upstream training preprocessing applies
  `y_soma[y_soma > y_soma_threshold] = y_soma_threshold` before the bias and
  scale transform.
- Upstream evaluation also thresholds `y_soma_test` above the same voltage
  threshold before computing predictions and RMSE.
- The original Beniaguev et al. codebase
  (`SelfishGene/neuron_as_deep_net`, checked at commit `074c4666300a`) uses
  the same convention: `fit_CNN.py` defines `y_train_soma_bias=-67.7` and
  `y_soma_threshold=-55.0`, thresholds `y_soma` before subtracting the bias,
  and `evaluate_CNN_test.py` clips test soma voltage at `v_threshold = -55`
  before subtracting `-67.7` for evaluation.
- Our local NeuronIO converter mirrors this convention in
  `src/bnn_sim/neuronio_raw.py::normalize_soma`.

Writing implication:

- When comparing against ELM / Branch-ELM, describe voltage RMSE as RMSE on the
  historical clipped NeuronIO somatic-voltage target.
- Do not imply that sub-`1 mV` RMSE directly means accurate action-potential
  peak-shape fidelity. Spike timing/probability is evaluated separately by AUC.
- Our Arbor diagnostic quantifies why this matters: hard `-55 mV` clipping
  affects only `0.2094%` of masked timesteps globally, but `12.923%` of
  near-spike timesteps, with near-spike raw-vs-clipped RMSE `8.9514 mV` and
  maximum removed AP amplitude `50.21 mV`.
- If we use a clipped Arbor target, state that this preserves comparability to
  prior NeuronIO/ELM metrics. If we use raw Arbor soma voltage, report raw
  waveform/spike-shape metrics separately and do not expect direct RMSE
  equivalence to the clipped NeuronIO setup.
- This is best framed as an inherited benchmark-semantics choice, not as a
  claim that clipping is biophysically harmless.
