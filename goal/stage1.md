# Stage 1 Goal: Corrected Grid9 Evidence Package

This stage is unlocked immediately. It is the only detailed stage prompt that Codex may read at the start of the gated program.

## Purpose

Answer what the corrected clipped grid9 matrix actually supports in a non-quick EM-TV comparison against single-pinhole baselines.

## Expected Artifacts

```text
results/corrected_grid9_stage1/
  effect_comparison.csv
  effect_comparison.md
  reconstruction_panel.png
  metric_quality_report.md
  commands_run.md
  corrected_grid9_report.md
  STAGE1_DONE.md
```

## Codex `/goal` Prompt

```text
/goal Complete the corrected-grid9 XFCT baseline evidence package without stopping until results/corrected_grid9_stage1/STAGE1_DONE.md exists and states that all acceptance checks below passed.
Context to read first:
- docs/current_state_handoff_gpt_pro.md
- experiments/validate_mask_forward_consistency.py
- experiments/run_effect_comparison.py
- algorithm/em_tv.py
- algorithm/recon_common.py
- src/reporting_roi.py
- src/reporting_reconstruction.py
- scripts/clip_system_matrix_to_physical_detector.py
- results/forward_model_validation_clipped/validation_summary.md
- results/forward_model_validation_clipped/validation_summary.json
- results/effect_comparison/effect_comparison.csv
- results/effect_comparison_clipped_quick/effect_comparison.csv
Scientific objective:
Produce a corrected, non-quick, physically consistent XFCT reconstruction baseline comparing:
1. traditional_5
2. traditional_15
3. traditional_45
4. mask_5_model using the clipped physical-support grid9 matrix:
   data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz
Do not use the original unclipped grid9 matrix for any final conclusion. It may appear only as a historical/confounded reference, clearly labeled.
Detector-domain invariants:
- Projection data are physical 80 x 80 detector arrays.
- Reconstruction pads detector x by 40 columns on each side to 80 x 160.
- Physical padded x columns are 40..119.
- Virtual padded x columns are 0..39 and 120..159.
- Any final mask matrix used for reconstruction must have zero support in virtual padded detector columns.
- Re-run or verify forward validation before trusting reconstruction results.
Required work loop:
1. Create results/corrected_grid9_stage1/ and a short progress log at results/corrected_grid9_stage1/progress_log.md.
2. Verify that the clipped grid9 matrix passes detector_padding, delta_voxel_tests, multi_hole_linearity, adjoint_tests, and known_phantom_residual checks under support_mode="physical_padded".
3. Run the non-quick EM-TV comparison for traditional_5, traditional_15, traditional_45, and mask_5_model using the clipped grid9 matrix. Use conda run -n xfct. Preserve existing results directories; do not overwrite earlier quick or original-confounded outputs unless you first copy or route outputs to results/corrected_grid9_stage1/.
4. If the existing script cannot directly write to the target directory or cannot select this exact run set, make the smallest safe code change needed, document it, and keep backward compatibility.
5. Add metric quality control around DL/CNR/ROI reporting. Mark DL invalid rather than reporting it as meaningful when any of these occur: nonpositive slope, negative DL, very poor linear fit, NaN/inf, non-monotonic concentration response, or unstable background estimate. Save the rule implementation and a metric_quality_report.md.
6. Generate a concise corrected baseline report at results/corrected_grid9_stage1/corrected_grid9_report.md. It must include:
   - matrix provenance
   - detector support validation summary
   - projection sums
   - final nll / rel convergence metrics
   - ROI recovery and CNR/DL metrics with valid/invalid flags
   - reconstruction panel paths
   - a clear statement of what is concluded and what is not concluded
7. Save every command run to results/corrected_grid9_stage1/commands_run.md.
8. Write STAGE1_DONE.md only after all acceptance checks pass.
Acceptance checks:
- The clipped grid9 validation has virtual row-sum fraction 0.0, or the report explains why an equivalent detector-support check passed.
- No final result uses data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz as the mask_5_model matrix.
- The comparison CSV contains traditional_5, traditional_15, traditional_45, and corrected mask_5_model.
- Quick-mode outputs are not used as final corrected evidence.
- DL values that fail quality checks are explicitly marked invalid; do not interpret invalid DL numerically.
- The report does not claim that multi-hole XFCT is generally better or worse. It only states what the corrected grid9 baseline supports.
- Existing files outside the new results/corrected_grid9_stage1/ evidence package are not destructively modified.
- If a run is blocked by missing data, memory, runtime, or environment issues, stop only after writing BLOCKED.md with exact failed command, traceback, partial artifacts, and the smallest next action needed.
Keep progress reports compact:
- current checkpoint
- command or artifact verified
- next action
- blocked or not blocked
```

## Unlocks

When `results/corrected_grid9_stage1/STAGE1_DONE.md` exists and documents passed acceptance checks, Codex may read `goal/stage2.md`.
