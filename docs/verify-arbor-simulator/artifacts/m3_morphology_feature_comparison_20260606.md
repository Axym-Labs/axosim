# M3 Morphology Feature Comparison

Date: 2026-06-06

Purpose: validate whether the currently available local SWC morphologies are
usable held-out targets for publishable M3 same-cell-type morphology transfer.

## Method

- Training/reference morphology:
  `src/bnn_sim/assets/allen_l5_template.swc`
- Local candidate held-out morphologies:
  `data/morphologies/specimen_327962063.swc`,
  `data/morphologies/specimen_474267418.swc`
- Branch routing target: `45` selected basal/apical branches, matching the
  current surrogate input-routing assumption.
- New diagnostic split:
  - `morphology_features`: whole extracted basal/apical morphology features.
  - `routing_features`: features of the branch subset actually routed to model
    inputs.

## Results

| morphology | extracted branches | routed branches | dendritic length ratio vs train | routed length ratio vs train | apical fraction delta | radial extent ratio | routing shape match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `allen_l5_template.swc` | 99 | 45 | 1.000 | 1.000 | 0.000 | 1.000 | reference |
| `specimen_327962063.swc` | 43 | 43 | 0.376 | 0.484 | -0.650 | 0.399 | false |
| `specimen_474267418.swc` | 26 | 26 | 0.212 | 0.273 | -0.650 | 0.347 | false |

## Interpretation

These two local held-out SWCs are not defensible same-cell-type transfer
targets for the current L5/Hay/CoreNEURON path. Under the SWC tags available
locally, both candidates are basal-only after basal/apical extraction, have no
apical dendritic length, have substantially fewer extracted/routed branches,
and have much smaller total dendritic length and radial extent than the
training/reference L5 template.

This is a useful M3 negative-control/compatibility result, not a successful
M3 morphology-transfer benchmark. A later transfer score on these morphologies
would mix model generalization with cell-type/morphology distribution shift and
routing incompatibility. For a publishable M3 claim, we still need a same-cell
type or closely matched L5 morphology set with comparable apical/basal
structure and an explicit simulator/routing protocol for each morphology.

## Verification

- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_suite.py tests/test_cli.py -q`
  passed: `26` tests.
- `PYTHONPATH=src .venv/bin/python -m py_compile src/bnn_sim/benchmark/swc_utils.py src/bnn_sim/benchmark/morphology_transfer.py src/bnn_sim/run_benchmark.py`
  passed.

