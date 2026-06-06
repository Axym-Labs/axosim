# Allen L5 M3 Candidate Ranking

- Source filter: `Mus musculus, layer 5, spiny, non-null reconstruction`
- Reference SWC: `src/bnn_sim/assets/allen_l5_template.swc`
- Allen API basis: `ApiCellTypesSpecimenDetail` records with non-null
  reconstructions, then `NeuronReconstruction` records with SWC well-known
  files.
- Candidate rows: `71`
- Ranked rows: `71`
- Exact reference specimen ID: `488683425`

All candidates found for this filter are marked `dendrite-only` in the Allen metadata. This is acceptable for a dendritic-routing M3 diagnostic, but not a full-cell morphology-transfer claim.

| rank | specimen | line | type | score | branches | routed | length ratio | routed length | apical delta | radial ratio | route shape |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `487664663` | `Rbp4-Cre_KL100` | `dendrite-only` | 0.203 | 98 | 45 | 0.930 | 0.946 | 0.053 | 1.022 | `True` |
| 2 | `515249852` | `Rbp4-Cre_KL100` | `dendrite-only` | 0.214 | 113 | 45 | 1.019 | 0.931 | 0.106 | 0.983 | `True` |
| 3 | `479770916` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.247 | 110 | 45 | 1.116 | 1.037 | -0.029 | 0.930 | `True` |
| 4 | `485835055` | `Rbp4-Cre_KL100` | `dendrite-only` | 0.268 | 128 | 45 | 1.232 | 0.992 | 0.012 | 0.961 | `True` |
| 5 | `589427435` | `Sim1-Cre_KJ18` | `dendrite-only` | 0.284 | 120 | 45 | 1.187 | 1.033 | 0.054 | 0.975 | `True` |
| 6 | `485838981` | `Rbp4-Cre_KL100` | `dendrite-only` | 0.284 | 130 | 45 | 1.002 | 0.877 | -0.014 | 0.872 | `True` |
| 7 | `502366405` | `Ctgf-T2A-dgCre` | `dendrite-only` | 0.284 | 143 | 45 | 1.193 | 0.969 | -0.021 | 0.947 | `True` |
| 8 | `589760138` | `Sim1-Cre_KJ18` | `dendrite-only` | 0.288 | 101 | 45 | 0.916 | 0.904 | 0.020 | 0.924 | `True` |
| 9 | `479225052` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.315 | 117 | 45 | 0.977 | 0.892 | 0.021 | 0.855 | `True` |
| 10 | `485880739` | `Rbp4-Cre_KL100` | `dendrite-only` | 0.322 | 90 | 45 | 0.911 | 0.928 | 0.010 | 0.867 | `True` |
| 11 | `488698341` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.323 | 121 | 45 | 1.063 | 0.921 | 0.014 | 0.847 | `True` |
| 12 | `488680917` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.358 | 106 | 45 | 1.077 | 1.038 | 0.054 | 0.825 | `True` |
| 13 | `557252022` | `Sim1-Cre_KJ18` | `dendrite-only` | 0.361 | 126 | 45 | 1.135 | 1.016 | 0.074 | 0.866 | `True` |
| 14 | `488695444` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.388 | 85 | 45 | 0.892 | 0.992 | -0.128 | 0.871 | `True` |
| 15 | `488679042` | `Rorb-IRES2-Cre` | `dendrite-only` | 0.405 | 132 | 45 | 1.325 | 1.095 | 0.019 | 0.986 | `True` |

Interpretation: the bundled Allen L5 template is exactly specimen `488683425`, so that specimen must be excluded from held-out transfer. Several held-out mouse L5 spiny dendrite-only morphologies remain close enough on branch count, routed length, apical fraction, and radial extent to support a defensible M3a dendritic-routing feasibility benchmark.

Source documentation: Allen Cell Types API documents `ApiCellTypesSpecimenDetail`
metadata, `NeuronReconstruction`, and `well_known_file_download` reconstruction
downloads; AllenSDK documents `get_cells(require_reconstruction=True)` and
`get_reconstruction(specimen_id)` for the same underlying resource.
