# Low-Compute Replication Success Note

This note records the evidence that our local official-style implementation reproduces the same low-compute training behavior as the official repository code when both are run on the same rebuilt full-trace data and evaluated with the same metric path.

## Compared runs

- Local repo official-style sampler probe:
  `/workspace/runs/official_fulltrace_probe_4x2500_b8_burn150_20260504_130016/metrics.json`
- Official repo compatibility probe:
  `/workspace/runs/elmneuron_repo_probe_4x2500_b8_burn150_progress_20260504_164234/out/metrics.json`

Both runs used:

- 4 epochs
- 2,500 batches per epoch
- batch size 8
- 80,000 total presentations
- burn-in 150 for training loss
- all-5 full-trace validation with ignore-start 500 and affine soma calibration

## Matching evidence

The per-epoch validation metrics matched exactly:

| epoch | val RMSE mV | val AUC |
|---:|---:|---:|
| 1 | 1.145837151394279 | 0.9726728226942521 |
| 2 | 1.0168520884413812 | 0.9806191151969379 |
| 3 | 0.9681284892480015 | 0.9826034240356828 |
| 4 | 0.9991507541355389 | 0.981937216600557 |

Best epoch matched as well:

- best epoch: 3
- best val RMSE: 0.9681284892480015 mV
- best val AUC: 0.9826034240356828

## Interpretation

This does not mean we matched the paper's final RMSE. It means something narrower and still very useful:

- our local official-style model/loss path behaves the same as the official repo code under the same low-compute setup
- the poorer convergence of the official-style sampler is therefore not explained by a hidden bug in our local model or loss port
- the remaining quality gap is more likely due to training setup, sampling regime, data split/provenance, or compute budget
