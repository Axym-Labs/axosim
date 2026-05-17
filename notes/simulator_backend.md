# Simulator Backend

The Arbor teacher simulator is in the repo. The implementation lives in
`src/bnn_sim/arbor_sim.py`, with the Allen L5 template assets in
`src/bnn_sim/assets/` and the source-matched NMDA mechanism in
`src/bnn_sim/mechanisms/nmda.mod`.

The generated datasets are intentionally not tracked because `data/` is ignored.
Regenerate them from checked-in configs instead:

```bash
bnn-sim make-arbor-hetero-data --config configs/sim/arbor_allen_nmda_source_smoke.json
bnn-sim make-arbor-hetero-data --config configs/sim/arbor_allen_nmda_source_medium.json
```

The medium config reproduces the local `data/arbor-allen-nmda-source-medium-20260510`
split layout: train/val/test `256/64/64`, four shards per split, `6000` steps,
`1278` input channels, 45 selected Allen morphology branches, Arbor integration
`dt=0.05 ms`, and batches of 32 independent cells.

Use `pip install -e ".[sim]"` only as a dependency marker. On this workstation,
the reproducible GPU path was a CUDA-enabled Arbor `0.11.0` source build; CPU-only
or incompatible Arbor builds will be rejected by the generator unless `--cpu-only`
is explicitly passed.
