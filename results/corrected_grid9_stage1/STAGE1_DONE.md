# Stage 1 Done

Completed on 2026-05-10 Asia/Shanghai.

## Objective Restated

Produce a corrected, non-quick, physically consistent EM-TV baseline evidence
package comparing:

1. `traditional_5`
2. `traditional_15`
3. `traditional_45`
4. corrected `mask_5_model` using
   `data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz`

The package must validate detector support, avoid the original unclipped grid9
matrix for final mask conclusions, add DL/ROI quality flags, and avoid broad
claims about multi-hole XFCT.

## Prompt-To-Artifact Checklist

| requirement | evidence | status |
| --- | --- | --- |
| Create `results/corrected_grid9_stage1/` and progress log | `progress_log.md` exists and records checkpoints | PASS |
| Verify clipped grid9 detector support and validation checks | `results/forward_model_validation_clipped/validation_summary.json` is PASS for detector padding, delta voxel tests, multi-hole linearity, adjoint tests, and known phantom residual | PASS |
| Confirm virtual padded detector row sum is zero | Direct CSR audit: shape `(64000, 144000)`, nnz `184,933,778`, virtual row sum `0.0`, virtual fraction `0.0` | PASS |
| Run non-quick EM-TV comparison with required runs | `effect_comparison.csv` contains exactly `traditional_5`, `traditional_15`, `traditional_45`, `mask_5_model`; each `reconstruction_results.npz` has 35 NLL entries | PASS |
| Use clipped grid9 matrix for final `mask_5_model` | CSV `mask_5_model.system_matrix_path` contains `clipx80pad40` and is not the original unclipped matrix filename | PASS |
| Do not use quick-mode outputs as final evidence | Final comparison command did not use `--quick`; output root is `results/corrected_grid9_stage1/`; each run has 35 iterations | PASS |
| Add DL/CNR/ROI quality control | `src/reporting_roi.py`, `src/reporting_reconstruction.py`, and `experiments/run_effect_comparison.py` write validity fields; `metric_quality_report.md` documents the rules | PASS |
| Mark invalid DL values explicitly | `mask_5_model` has `detection_limit_valid=False`, reason `poor CNR linear fit R2<0.80; non-monotonic CNR concentration response` | PASS |
| Generate corrected baseline report | `corrected_grid9_report.md` includes matrix provenance, detector validation, projection sums, NLL/rel convergence, ROI/CNR/DL metrics, figures, and conclusion scope | PASS |
| Save command log | `commands_run.md` exists | PASS |
| Preserve existing result directories | Final outputs were routed to `results/corrected_grid9_stage1/`; no earlier quick or original-confounded result directory was overwritten | PASS |
| Avoid broad multi-hole conclusion | `corrected_grid9_report.md` states only the corrected grid9 Stage 1 finding and says this does not establish that multi-hole XFCT is generally better or worse | PASS |

## Key Results

| run | projection sum | DL flag | raw DL mg/ml | R2 | final NLL | final relative change |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `traditional_5` | 219,811.179195 | valid | 0.262452 | 0.959957 | -419,783.889968 | 0.0307549 |
| `traditional_15` | 657,307.573813 | valid | 0.063257 | 0.994301 | -1,253,519.706771 | 0.0112100 |
| `traditional_45` | 1,968,367.735310 | valid | 0.003945 | 0.999305 | -3,755,497.277611 | 0.00737305 |
| `mask_5_model` | 10,913,099.560160 | invalid | 19.644854 | 0.463720 | -54,920,270.512371 | 0.0146426 |

The invalid `mask_5_model` raw DL is not interpreted numerically.

## Completion Audit

Final audit command printed `COMPLETION_AUDIT_PASS` after checking required
artifacts, validation coverage, direct matrix virtual support, run set, 35
iterations, corrected mask matrix path, DL validity flags, report scope, and
non-quick command usage.

No Stage 1 requirement remains missing, incomplete, weakly verified, or blocked.
