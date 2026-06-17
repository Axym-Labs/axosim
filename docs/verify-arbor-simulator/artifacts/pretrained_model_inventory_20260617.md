# Pretrained Model Inventory

Date: 2026-06-17

Purpose: collect literature pretrained models for later benchmark evaluation
without retraining. Scope is at most three models per requested architecture or
paper family.

## Downloaded / Local Models

| family | candidate | local path | source | verification |
| --- | --- | --- | --- | --- |
| TCN | `NMDA_TCN__DWT_7_128_153` | `data/raw/nmda-test/Models/NMDA_TCN__DWT_7_128_153__model.h5` | Beniaguev et al. Kaggle NMDA test/pretrained cache | loaded with TensorFlow CPU; zero-input forward finite |
| TCN | `NMDA_TCN__DWT_8_224_217` | `data/raw/nmda-test/Models/NMDA_TCN__DWT_8_224_217__model.h5` | Beniaguev et al. Kaggle NMDA test/pretrained cache | loaded with TensorFlow CPU; zero-input forward finite |
| TCN | `NMDA_TCN__DWT_9_256_241` | `data/raw/nmda-test/Models/NMDA_TCN__DWT_9_256_241__model.h5` | Beniaguev et al. Kaggle NMDA test/pretrained cache | loaded with TensorFlow CPU; zero-input forward finite |
| ELM | `best_elm_neuron` | `.external/elmneuron/models/best_elm_neuron/neuronio_best_model_state.pt` | `AaronSpieler/elmneuron` | strict PyTorch state load; zero-input forward finite |
| Branch-ELM | `num_memory_1` | `.external/elmneuron/models/num_memory_1/neuronio_best_model_state.pt` | `AaronSpieler/elmneuron` | strict PyTorch state load; zero-input forward finite |
| Branch-ELM | `num_memory_30` | `.external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt` | `AaronSpieler/elmneuron` | strict PyTorch state load; zero-input forward finite |
| Branch-ELM | `num_memory_100` | `.external/elmneuron/models/num_memory_100/neuronio_best_model_state.pt` | `AaronSpieler/elmneuron` | strict PyTorch state load; zero-input forward finite |

Also downloaded for the ELM/Branch-ELM cache: `model_config.json`,
`model_stats.json`, `eval_results.json`, `train_config.json`, plus the
top-level `models/README.md` and `eval_results.csv`.

## Verification Details

TCN verification used an isolated local CPU TensorFlow environment
`.venv-tcn` with `CUDA_VISIBLE_DEVICES=-1`. This environment is local/ignored
and should not be part of the package. Smoke outputs:

| TCN | input shape | output shapes | finite | output means |
| --- | --- | --- | --- | --- |
| `7_128_153` | `(None, 400, 1278)` | `(None, 400, 1)`, `(None, 400, 1)`, `(None, 400, 20)` | yes | `1.08e-06`, `-17.4646`, `-0.3218` |
| `8_224_217` | `(None, 500, 1278)` | `(None, 500, 1)`, `(None, 500, 1)`, `(None, 500, 64)` | yes | `3.90e-06`, `-13.1270`, `0.00748` |
| `9_256_241` | `(None, 500, 1278)` | `(None, 500, 1)`, `(None, 500, 1)`, `(None, 500, 64)` | yes | `9.20e-06`, `-14.0593`, `-0.0193` |

ELM/Branch-ELM verification used `PYTHONPATH=src .venv/bin/python` and the
repo's official ELM loader. All selected checkpoints loaded with
`strict=True` and produced finite `(2, 16, 2)` outputs on zero input:

| model | local port parameter count | finite | output mean |
| --- | ---: | --- | ---: |
| `num_memory_1` | `18102` | yes | `-6.7275` |
| `num_memory_30` | `24482` | yes | `-7.0033` |
| `num_memory_100` | `67602` | yes | `-6.4570` |
| `best_elm_neuron` | `67602` | yes | `-6.4229` |

The upstream ELM `model_stats.json` reports smaller counts for the same
families (`4601`, `10952`, and `54002` respectively); strict loading confirms
checkpoint compatibility with the local port despite that counting convention
difference.

## Checksums

```text
541c71bcf78875354cc32acbb7e15c2495415fd3f85cc683eba4f75fbe7ac4e0  data/raw/nmda-test/Models/NMDA_TCN__DWT_7_128_153__model.h5
9533ca70114b3396ead678206fe9a1e6f1725a26e342796b8ae23641e738c9fd  data/raw/nmda-test/Models/NMDA_TCN__DWT_8_224_217__model.h5
950327afb6e3c3c2ad0aceab3539213e958e2201575b0f4ae489c4c9e94b9e85  data/raw/nmda-test/Models/NMDA_TCN__DWT_9_256_241__model.h5
ee1252616cd2ad7dd60786e304a56e6d9d13b4f85ce9974fbef909efa9f43812  .external/elmneuron/models/best_elm_neuron/neuronio_best_model_state.pt
2fde82f06143fee92dc1e76d20a16aa69cfb0f3fed3876f6e321e643a6846a7e  .external/elmneuron/models/num_memory_1/neuronio_best_model_state.pt
444dd697c7f0f9ea13dd82bcd46f56b757d2ef4dfdc367e1d63520ea625303f7  .external/elmneuron/models/num_memory_100/neuronio_best_model_state.pt
90e81b6a970f536d18f99abfeec8baa7b46a007b0ce06e70c118c38622937491  .external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt
```
