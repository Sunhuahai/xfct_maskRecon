# Corrected Grid9 Stage 1 Report

## Matrix Provenance

Final mask reconstruction uses only the clipped physical-support grid9 matrix:

`data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz`

This matrix was produced by clipping the original padded grid9 CSR rows to the real
physical detector support: detector z `80`, padded detector x `160`, physical
padded x columns `40..119`, virtual columns `0..39` and `120..159`.

Direct matrix check in this Stage 1 run:

| field | value |
| --- | ---: |
| shape | `(64000, 144000)` |
| nnz | 184,933,778 |
| total row sum | 61.90858427187911 |
| physical row sum | 61.90858427187911 |
| virtual row sum | 0.0 |
| virtual fraction | 0.0 |

The original unclipped grid9 matrix is not used for the final `mask_5_model` row.

## Detector Support Validation

Validation source: `results/forward_model_validation_clipped/validation_summary.md`
and `validation_summary.json`.

| check | status | key result |
| --- | --- | --- |
| detector padding | PASS | virtual fraction `0.0`, virtual row sum `0.0` |
| delta voxel tests | PASS | max virtual fraction `0.0`, max centroid shift `0.000 px` |
| multi-hole linearity | PASS | relative L2 `5.072e-16` |
| adjoint tests | PASS | matrix-free rel `2.996e-16`, explicit CSR PASS |
| known phantom residual | PASS | self deviance `0.000e+00`, physical-vs-A deviance `0.000e+00` |

## Non-Quick EM-TV Comparison

Command used 35 EM-TV iterations and did not pass `--quick`. Output root:
`results/corrected_grid9_stage1/`.

| run | projection sum | final NLL | final relative change | matrix |
| --- | ---: | ---: | ---: | --- |
| `traditional_5` | 219,811.179195 | -419,783.889968 | 0.0307549 | `data/system_matrix/cij_5_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` |
| `traditional_15` | 657,307.573813 | -1,253,519.706771 | 0.0112100 | `data/system_matrix/cij_15_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` |
| `traditional_45` | 1,968,367.735310 | -3,755,497.277611 | 0.00737305 | `data/system_matrix/cij_45_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` |
| `mask_5_model` | 10,913,099.560160 | -54,920,270.512371 | 0.0146426 | clipped grid9 physical-support matrix |

## ROI And CNR Metrics

ROI means are the six simulation ROIs with nominal concentrations
`[0, 0.5, 1.0, 1.5, 2.0, 3.0] mg/ml`.

| run | ROI means | CNR values | DL flag | raw DL mg/ml | R2 | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| `traditional_5` | `0.023105, 0.084695, 1.187260, 1.502750, 1.854470, 3.000000` | `0.000000, 3.986750, 75.356100, 95.778400, 118.545000, 192.696000` | valid | 0.262452 | 0.959957 | valid |
| `traditional_15` | `0.010068, 0.415908, 1.040630, 1.487650, 2.139330, 3.000000` | `0.000000, 90.490700, 229.787000, 329.458000, 474.765000, 666.669000` | valid | 0.063257 | 0.994301 | valid |
| `traditional_45` | `0.004534, 0.495444, 1.008570, 1.524680, 2.059980, 3.000000` | `0.000000, 248.041000, 507.307000, 768.078000, 1038.550000, 1513.510000` | valid | 0.003945 | 0.999305 | valid |
| `mask_5_model` | `2.208054, 1.667770, 0.676627, 0.815704, 1.377950, 3.000000` | `0.000000, -0.193889, -0.549580, -0.499669, -0.297897, 0.284204` | invalid | 19.644854 | 0.463720 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response |

The invalid `mask_5_model` raw DL is a diagnostic fit output only and is not a
meaningful numerical detection limit.

## Figures And Artifacts

- Reconstruction panel: `results/corrected_grid9_stage1/reconstruction_panel.png`
- Comparison CSV: `results/corrected_grid9_stage1/effect_comparison.csv`
- Comparison Markdown: `results/corrected_grid9_stage1/effect_comparison.md`
- Metric quality report: `results/corrected_grid9_stage1/metric_quality_report.md`
- Per-run reconstruction and ROI figures: `results/corrected_grid9_stage1/*_em_tv/`

## Conclusion Scope

This Stage 1 evidence supports only this statement: under the corrected physical
detector support and this non-quick EM-TV setup, the `mask_5_model` corrected
grid9 reconstruction does not produce a valid CNR/DL fit, while the three
single-pinhole baselines do. It does not establish that multi-hole XFCT is
generally better or worse, and it does not decide final sparse-mask candidates
or final protocol-level performance.
