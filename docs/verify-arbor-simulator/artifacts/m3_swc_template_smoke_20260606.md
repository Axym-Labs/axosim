# M3 SWC-Aware Hay Template Smoke

Date: 2026-06-06

Purpose: verify that the M3 held-out Allen SWC path can instantiate and
integrate with the Hay L5PC biophysics, rather than only passing static
morphology-feature checks.

## Setup

- Installed `neuron==9.0.1` into `.venv` with `uv pip install --python .venv/bin/python neuron`.
- Generated a SWC-aware template with
  `docs/verify-arbor-simulator/artifacts/prepare_hay_swc_template.py`.
- Compiled the existing deterministic CoreNEURON/Hay mechanism set for this
  NEURON runtime under ignored
  `docs/verify-arbor-simulator/artifacts/hay_swc_template/x86_64/`.
- Source biophysics: original `L5PCbiophys5b.hoc`.
- Generated template: ignored
  `docs/verify-arbor-simulator/artifacts/hay_swc_template/L5PCtemplate_swc.hoc`.
- Held-out morphology smoke target:
  `data/morphologies/allen_mouse_l5_spiny/specimen_487664663.swc`
  (`Rbp4-Cre_KL100`, held out from source specimen `488683425`).

## Instantiation Smoke

Using the generated template:

| morphology | sections | basal | apical | axon |
| --- | ---: | ---: | ---: | ---: |
| original `cell1.asc` | 195 | 84 | 109 | 2 |
| held-out `specimen_487664663.swc` | 101 | 32 | 67 | 2 |

The generated template selects `Import3d_SWC_read` for `.swc` and preserves the
original `Import3d_Neurolucida3` path for `.asc`.

## Integration Smoke

No-input NEURON fixed-step run on `specimen_487664663.swc`:

- duration: `10 ms`
- `dt`: `0.025 ms`
- samples: `401`
- initial soma voltage: `-76.0 mV`
- final soma voltage: `-79.1941 mV`
- min/max: `-79.1941 / -76.0 mV`

## Interpretation

This clears the immediate M3 feasibility blocker: a held-out Allen mouse L5
spiny SWC can be instantiated with the Hay biophysics and integrated locally.
It does not yet prove simulator correctness for M3 because the held-out Allen
reconstructions are `dendrite-only`, the Hay active-parameter distribution was
fit to the source morphology, and synapse placement/input routing still needs a
paired teacher-generation smoke. The next M3 step is to run a small
intervention/data-generation pair on one held-out SWC using
`run_coreneuron_event_dropout_pair.py --template-hoc ... --morphology ...`,
then compare teacher response statistics against the source-morphology teacher.

