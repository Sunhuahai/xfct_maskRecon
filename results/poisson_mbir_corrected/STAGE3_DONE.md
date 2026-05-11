# Stage 3 Done

Completed on 2026-05-10 Asia/Shanghai.

## Objective Restated

Harden and run the raw-domain Poisson-TV PDHG reconstruction path under matched
`physical_padded` support. Reconstruct corrected grid9 and selected sparse
candidates, record a documented parameter grid, save projection-fit and residual
diagnostics, report ROI/CNR/DL validity, and distinguish real grid9 projection
data from matched synthetic sparse-candidate data.

## Prompt-To-Artifact Checklist

| requirement | evidence | status |
| --- | --- | --- |
| Add/verify numerical checks | `recon/poisson_tv_pdhg.py` checks finite/positive lambda, finite objective/deviance/relative change, nonnegative reconstruction, and records objective behavior diagnostics | PASS |
| Validate operators before reconstruction | Each summary row has `operator_validation_status=PASS`, adjoint relative error `<1e-10`, and `operator_delta_virtual_sum=0.0` | PASS |
| Tune only within documented grid | `parameter_grid.csv` records six candidate/beta/iteration rows: beta `1e-5` and `1e-4`, 30 iterations, seed `20260509` | PASS |
| Run corrected grid9 | `poisson_mbir_summary.csv` includes `grid9_p6_d1p25_5` using `real_grid9_projection_padded_physical_detector` | PASS |
| Run selected sparse candidates | Summary includes `blue_noise_n3_d1d25_mind3_s1` and `ring_n7_d1d25_mind3` using `synthetic_matched_poisson` | PASS |
| Generate residual maps and reconstruction panels | `reconstruction_panels/` and `residual_maps/` each contain 6 PNGs | PASS |
| Report ROI/CNR/DL with invalid-DL flags | Summary CSV includes `detection_limit_valid`, `detection_limit_invalid`, and `detection_limit_invalid_reason`; grid9 DL rows are invalid and not interpreted numerically | PASS |
| Preserve matched projection domain/support | Every row has `support_mode=physical_padded`; real grid9 rows cite the physical 5-view grid9 projection path, synthetic rows have matched operator-generated Poisson data | PASS |
| Compare against corrected EM-TV baseline without overstatement | `mbir_report.md` cites the Stage 1 corrected EM-TV invalid DL flag and states these rows do not establish final protocol-level superiority or inferiority | PASS |
| Save commands | `commands_run.md` exists | PASS |

## Key Rows

| candidate | data domain | beta values | DL status summary |
| --- | --- | --- | --- |
| `grid9_p6_d1p25_5` | real padded grid9 projection | `1e-5`, `1e-4` | invalid DL: poor CNR linear fit and non-monotonic CNR response |
| `blue_noise_n3_d1d25_mind3_s1` | matched synthetic Poisson | `1e-5`, `1e-4` | valid DL in this synthetic matched run |
| `ring_n7_d1d25_mind3` | matched synthetic Poisson | `1e-5`, `1e-4` | valid DL in this synthetic matched run |

Synthetic matched sparse-candidate results are not real projection measurements
and are not final protocol-comparison evidence.

## Completion Audit

Final audit command printed `STAGE3_COMPLETION_AUDIT_PASS` after checking required
artifacts, parameter-grid coverage, support mode, data-domain distinction,
finite numerical diagnostics, operator validation, reconstruction/residual paths,
DL validity flags, and report scope.

No Stage 3 requirement remains missing, incomplete, weakly verified, or blocked.
