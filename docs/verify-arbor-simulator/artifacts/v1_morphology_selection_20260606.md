# V1 Morphology Selection

Date: 2026-06-06

Purpose: choose the provisional `N=5` deep morphology set for the v1
NeuronIO-like dataset and diagnostic benchmark before teacher-QC pilots.

## Inputs

- Candidate source: `docs/verify-arbor-simulator/artifacts/allen_l5_m3_candidates_20260606.json`
- Local SWCs: `data/morphologies/allen_mouse_l5_spiny/`
- Reference/source morphology: `src/bnn_sim/assets/allen_l5_template.swc`
- Branch-routing target: `45` routed basal/apical dendritic branches.

All selected held-out candidates are Allen mouse layer-5 spiny reconstructions
with route-shape compatibility under the current 45-branch routing policy.
Allen metadata marks this candidate pool as `dendrite-only`, so claims must be
framed as dendritic-routing and soma-output surrogate diagnostics, not full-cell
biological morphology generalization.

## Selected Set

| role | specimen | line | score | branches | routed | length ratio | routed length | apical delta | radial ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference/control | `488683425` | `Rbp4-Cre_KL100` | 0.000 | 99 | 45 | 1.000 | 1.000 | 0.000 | 1.000 |
| closest held-out control | `487664663` | `Rbp4-Cre_KL100` | 0.203 | 98 | 45 | 0.930 | 0.946 | 0.053 | 1.022 |
| intermediate line shift | `479770916` | `Rorb-IRES2-Cre` | 0.247 | 110 | 45 | 1.116 | 1.037 | -0.029 | 0.930 |
| line-diversity shift | `589427435` | `Sim1-Cre_KJ18` | 0.284 | 120 | 45 | 1.187 | 1.033 | 0.054 | 0.975 |
| stronger sane length shift | `488679042` | `Rorb-IRES2-Cre` | 0.405 | 132 | 45 | 1.325 | 1.095 | 0.019 | 0.986 |

## Rationale

The set is not simply the top five by compatibility score. It keeps the exact
reference morphology, includes the nearest same-line held-out control, adds two
moderate shifts from different Cre lines, and reserves one stronger but still
routable length/branch-count shift. This matches the v1 framing: deep,
trainable, diagnostic-rich, and lean, with enough morphology contrast for
diagnostics but without broad morphology-generalization claims.

## Replacement Rule

This set is provisional until teacher-QC pilots pass. A failed morphology should
be replaced by the nearest available candidate that preserves its role:
`515249852` for close/apical shift, `485835055` for same-line moderate length
shift, `488695444` for stronger apical-fraction shift, or the next ranked
route-shape-compatible Allen L5 spiny candidate.

## Next Check

Run baseline and intervention teacher-QC pilots for all five morphologies before
freezing the public dataset. Required sanity checks: source-regime statistics,
voltage/spike distribution, excitatory/inhibitory polarity, intervention effect
direction and size, routing/morphology features, and raw-vs-clipped distortion.
