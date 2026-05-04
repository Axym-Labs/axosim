# Progress Log

Concise record of runs and decisions. Metrics JSON paths are the source of truth for plotting.

## 2026-05-02

- **Pilot smoke, 32 real NMDA windows/split**: verified official Branch-ELM-30 train/eval/checkpoint path. Artifacts: `/workspace/runs/branch_elm_pilot/`.
- **Pilot 256 windows/split**: 1 epoch, batch 8, LR `5e-4`, burn-in 150. Val RMSE `0.3883`, val AUC `0.1938`; test RMSE `0.3377`, test AUC `0.3028`. Metrics: `/workspace/runs/branch_elm_pilot/*-256.json`.
- **Loader optimization**: shard-local shuffle + shard cache changed one compressed 256-window shard from >225s interrupted to ~17-19s at batch 8. Main insight: repeated `.npz` decompression was dominant.
- **Recurrence storage optimization**: preallocated TorchScript output tensors. Batch-8 forward improved from ~`0.116s` to ~`0.08-0.09s` per 500-step pass.
- **First subset learning run**: 10 train shards, 10 val shards, 10 test shards, 3 epochs, batch 8. Best val at epoch 3: RMSE `0.2046`, AUC `0.8372`; test RMSE `0.1373`, AUC `0.8049`. Metrics: `/workspace/runs/replication_branch_elm30_subset/branch-elm30-3epoch-metrics.json`.
- **Throughput screening**: concurrent jobs help because single runs underutilize GPU. Best stable sweep mode found: ~10 jobs at batch 64, ~`303` aggregate presentations/s. Throughput artifacts: `/workspace/runs/throughput_sweep/`.
- **Hyperparameter screen**: 10 concurrent variations, 5 epochs on 10-shard train/val subset, 12,800 presentations/job. Summary: `/workspace/runs/hparam_screen_90min/summary.tsv`.
  | rank | variation | best val AUC | best val RMSE | notes |
  |---:|---|---:|---:|---|
  | 1 | `v07_adam_b32_lr5e4` | `0.7456` | `0.2925` | batch 32, Adam, cosine |
  | 2 | `v08_adamw_b32_lr5e4_wd1e4` | `0.7456` | `0.2925` | AdamW nearly identical |
  | 3 | `v09_adam_b64_lr5e4_constant` | `0.7224` | `0.2948` | constant LR beat b64 cosine on AUC |
  | 4 | `v04_adam_b64_lr1e3` | `0.6848` | `0.2835` | better RMSE, worse AUC |
  | 5 | `v05_adamw_b64_lr1e3_wd1e4` | `0.6848` | `0.2835` | AdamW nearly identical |
  | 6 | `v06_adam_b64_lr2e4` | `0.6802` | `0.3744` | lower LR underfit |
  | 7 | `v10_adamw_b64_lr2e4_wd1e4` | `0.6802` | `0.3744` | same as Adam at low LR |
  | 8 | `v01_adam_b64_lr5e4` | `0.6475` | `0.3092` | batch 64 cosine baseline |
  | 9 | `v02_adamw_b64_lr5e4_wd1e4` | `0.6475` | `0.3092` | AdamW nearly identical |
  | 10 | `v03_adamw_b64_lr5e4_wd1e3` | `0.6475` | `0.3092` | stronger weight decay did not help |
- **Confound notes**: model and loss are official-style; loader is ours. Paper training budget is `30 * 11400 * 8 = 2,736,000` presentations. For partial runs, cosine should use the full-paper horizon (`342000` scheduler steps) rather than decay to zero early.
- **Official loss scale check**: paper repository `NeuronioLoss` uses `0.5 * BCE(spikes) + 0.5 * MSE(soma)`. Training defaults now match this exactly; earlier runs used equal relative weighting but 2x larger absolute loss/gradient scale.
- **Started detached replication-oriented run**: full NMDA train pool sampled with paper epoch definition, 7 epochs x 11,400 batches x batch 8 = `638,400` presentations (`23.3%` of paper training presentation budget). Validation uses 10-shard val subset to keep runtime training-focused. Run dir: `/workspace/runs/branch_elm30_replication_b8_7epoch_val10_20260502_212245`.
- **Voltage RMSE scaling correction**: previous validation RMSE values were normalized training units. Paper-scale voltage RMSE divides the normalized error by `0.1`, so the 7-epoch run is best RMSE `1.463 mV` at epoch 1 and best AUC `0.9817` at epoch 6. Corrected metrics: `/workspace/runs/branch_elm30_replication_b8_7epoch_val10_20260502_212245/metrics-paper-soma.json`.
- **Full-val checkpoint check**: epoch-1/best-RMSE checkpoint on all 19,200 val windows: RMSE `1.491 mV`, AUC `0.9683`; final epoch-7 checkpoint: RMSE `1.674 mV`, AUC `0.9819`. This confirms the subset trend: AUC improves while voltage RMSE worsens, so the gap to paper RMSE is unlikely to be solved by simply extending the same run. Metrics: `/workspace/runs/branch_elm30_replication_b8_7epoch_val10_20260502_212245/*-fullval-paper-eval.json`.
- **Evaluation speed fix**: AUC tied-rank calculation no longer loops once per unique score. Full 19,200-window val eval now takes about `155s` with `cache_shards=1`; large full-shard caches are counterproductive because compressed shards expand to multi-GB dense arrays in RAM.
- **Official semantics check**: target preprocessing matches official constants/operations (`clip soma > -55 mV`, subtract `-67.7`, scale by `0.1`; signed input types are applied as `+1/-1`). Official training uses random 500-step windows from raw 6000-step simulations after `ignore_time_from_start=500`, batch 8, Adam, cosine stepped every batch, 10 workers, 100 prefetched batches, and loss over the full 500-step window. Their `burn_in_time=150` is used in evaluation stitching, not in the training loss.
- **Metric semantics update**: on current pre-windowed full val, final epoch-7 checkpoint with first 150 timesteps ignored gives RMSE `0.809 mV`, AUC `0.9908`; adding official-style soma affine calibration gives RMSE `0.805 mV`, AUC unchanged. This is much closer to paper Branch-ELM-30 (`0.628 mV`, `0.9927`) than the all-timestep metric. Metrics: `/workspace/runs/branch_elm30_replication_b8_7epoch_val10_20260502_212245/*ignore150*paper-eval.json`.
- **I/O optimization**: added `.npy` memmap shard repacking. `nmda-val` repacked with int8 inputs at `data/shards/nmda-val-npy` (`12G`); full-val eval improved from ~`155s` on compressed shards to ~`49s` at batch 128. One-step train prefetch on compressed train was only a small win in a 100-batch benchmark (`16.87` to `17.10` presentations/s).
- **500 ms eval-mask check**: local val shards already start windows at original time `500` for all 1,280 simulations, so "ignore first 500 ms of the full simulation" is mostly already baked into the converted split. Added an `official-overlap` mask that keeps the first available window per simulation and drops 150-step burn-in only from later overlapping windows; final epoch-7 with affine calibration gives RMSE `0.925 mV`, AUC `0.9899`. This is worse than dropping 150 from every local window (`0.805 mV`, `0.9908`) because our pre-window starts (`500,850,...`) are not the official eval starts (`0,350,700,...`). Exact reproduction needs raw 6000-step simulations or a reconverted eval set with official chunk starts.
- **Throughput prep while shards rebuild**: validation/training now expose transfer knobs (`pin_memory`, non-blocking copies) and validation can use `official-overlap` masks. On `nmda-val-npy`, sequential batch-128 eval with official-overlap+affine improved from `50.8s`/`378 samples/s` to `46.0s`/`417 samples/s` with pinned memory; concurrent full-val evals were much worse (`~384s`) from CPU/memory contention. On compressed train, 200 batch-8 steps with official loss and `burn_in=0` were `17.25 samples/s` plain vs `17.07 samples/s` with pin+1 prefetch, so keep training plain unless the rebuilt loader changes the bottleneck. Artifacts: `/workspace/runs/throughput_prep_20260503/`.
- **Full-trace shards moved into repo**: rebuilt shards are now at `data/full-trace-shards` (`train 5.2G`, `val 182M`, `test 888M`) and contain full `(256, 6000, 1278)` int8 inputs plus `(256, 6000, 2)` float32 targets. Direct `.npz` full-shard decompression costs ~`4.5-5.4s` per shard, but cached random 500-step windows from one loaded shard are fast: batch 8 mean `0.0016s` (`~5,040 windows/s`), batch 64 mean `0.027s` (`~2,366 windows/s`). This suggests the next loader should cache one/few full-trace shards and sample windows from memory rather than call `get_batch` on full traces.
- **Full-trace training cache implemented**: added `RandomFullTraceWindowDataset` and CLI `--window-sampling random-full-trace`. It caches full-trace shards in raw input dtype (`int8`) and samples 500-step windows with starts `>=500`; training loss should use `--burn-in 0` for repo-faithful runs. Benchmark on 200 batch-8 steps: old pre-windowed compressed path `17.25 samples/s`; full-trace cache with shard reuse 32 `12.15 samples/s` due to repeated full-shard decompression; full-trace cache with shard reuse 200 `18.29 samples/s`. Artifacts: `/workspace/runs/full_trace_cache_benchmark/`.
- **Full-trace prefetch update**: training iterator now overlaps one CPU batch load/decompression with GPU work. 200 batch-8 steps: reuse 32 improved from `12.15` to `13.29 samples/s`; reuse 200 improved from `18.29` to `18.80 samples/s`. A tiny 200-step from-scratch run is not accurate yet (1 val shard RMSE `2.10 mV`, AUC `0.081`), but the previous 7-epoch checkpoint evaluated on the same full-trace val shard with original `<500 ms` ignored gives RMSE `0.765 mV`, AUC `0.9911`, showing the corrected full-context eval is much closer to paper. Metrics: `/workspace/runs/full_trace_cache_benchmark/*val1shard*.json`.

## 2026-05-03

- **Default validation semantics updated**: training validation now defaults to full-trace evaluation when `--val-window-size` is unset, ignores original `<500 ms` for metrics, and enables official-style soma affine calibration. Smoke check on all 5 full-trace val shards completed end-to-end. Metrics: `/workspace/runs/full_trace_replication_smoke.json`.
- **Started bounded full-trace replication run**: 3 epochs x 5,000 batches x batch 8 = `120,000` presentations (`4.39%` of paper `2,736,000`). Training samples random 500-step windows with starts `>=500` from all full-trace train shards, using shard reuse 64 and one-step prefetch; validation uses all full-trace val shards with default full-context metric semantics. Run dir: `/workspace/runs/full_trace_replication_3x5000_b8_reuse64_20260503_170636`.
- **Started paper-like long full-trace run**: stopped the burn-in-0 launch after confirming paper Appendix B says NeuronIO loss was calculated after a `150 ms` burn-in. Relaunched 10 epochs x 10,000 batches x batch 8 = `800,000` presentations (`29.24%` of paper-style `2,736,000`; `28.57%` of repo `35*10000*8`) to use the 16-20h budget. Uses 500-step random windows with starts `>=500`, Adam LR `5e-4`, cosine full horizon, training loss after `burn_in=150`, and shard reuse 58 to approximate official `file_load_fraction=0.3` for 128-simulation files. Run dir: `/workspace/runs/paper_like_fulltrace_10x10000_b8_reuse58_burn150_20260503_202435`.

## 2026-05-04

- **Paper-like long run finished**: `/workspace/runs/paper_like_fulltrace_10x10000_b8_reuse58_burn150_20260503_202435/metrics.json`. Best epoch `10`; best val RMSE `0.7620 mV`; best val AUC `0.99095`; total runtime `16.35 h`; total compute `800,000` presentations. This is materially closer to paper (`0.628`, `0.9927`) but still leaves a real RMSE gap.
- **Official-style sampler added**: new `official-full-trace` mode groups by original raw source file and approximates official `file_load_fraction` sampling semantics while keeping deterministic rebuilt full-trace shards. Code paths: `src/bnn_sim/data.py`, `src/bnn_sim/cli.py`; smoke-tested successfully.
- **Official-style sampler probe underperformed early**: `/workspace/runs/official_fulltrace_probe_4x2500_b8_burn150_20260504_130016/metrics.json`. At `80,000` presentations, best val RMSE `0.9681`, AUC `0.9826`, clearly worse than the earlier `random-full-trace` curve at matched compute.
- **Official repo compatibility probe matched our local implementation**: cloned repo at `/workspace/elmneuron`; compatibility runner uses official `ELM` + `NeuronioLoss` with our rebuilt full-trace data. Run `/workspace/runs/elmneuron_repo_probe_4x2500_b8_burn150_progress_20260504_164234/out/metrics.json` reproduced the same per-epoch RMSE/AUC as our `official-full-trace` probe. Main insight: model/loss port is not the reason for the poorer official-style convergence.
- **Working defaults after replication checks**:
  - training sampler default: `random-full-trace`
  - training burn-in: `150 ms`
  - validation: full-trace recurrence, ignore original `<500 ms`, affine soma calibration
  - official-style sampler remains a reference path, not the default, unless a longer run shows it overtakes `random-full-trace`
- **Low-compute replication evidence saved in-repo**: [notes/low_compute_replication_success.md](/workspace/bnn_sim/notes/low_compute_replication_success.md) records that our local official-style implementation and the official repo compatibility runner matched exactly on the same `80,000`-presentation probe. This is the main evidence that model/loss porting is not the source of the official-style convergence gap.
- **Architecture / efficiency ideas queued for next phase**:
  - replace or hybridize the recurrent memory core with a more parallelizable state-space / Mamba-style core
  - learn memory time constants instead of fixing them
  - replace the simple memory-update MLP with a more structured gated update block
  - revisit activation choices and other lightweight inductive-bias improvements in the update rule
  - explore a small engram-style latent memory mechanism
  - learn branch routing instead of relying only on fixed morphology-based routing
  - try multi-scale memory with explicit short / medium / long latent groups
  - consider separate spike and voltage heads earlier in the network
  - keep event-driven / sparse computation ideas on the efficiency track
