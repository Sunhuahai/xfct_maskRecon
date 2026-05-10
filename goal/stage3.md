# Stage 3 Goal: Raw-Domain Poisson-TV MBIR

This stage is locked until `results/mask_design_corrected/STAGE2_DONE.md` exists.

Codex must not read or execute this file before Stage 2 is complete unless the user explicitly overrides the gate.

## Purpose

Make the raw Poisson-TV reconstruction path scientifically usable, not merely runnable. Reconstruct corrected grid9 and selected sparse candidates using a matched raw-domain Poisson model.

## Model

```text
min over f >= 0:
    sum_i [lambda_i(f) - y_i log(lambda_i(f) + eps)] + beta TV(f)

lambda(f) = A f + b
```

## Codex `/goal` Prompt

```text
/goal Harden raw-domain Poisson-TV XFCT mask reconstruction and produce validated real/properly matched reconstruction results for the corrected baseline candidate set without stopping until results/poisson_mbir_corrected/STAGE3_DONE.md exists or BLOCKED.md documents an irreducible data/runtime blocker.
Read first:
- docs/current_state_handoff_gpt_pro.md
- recon/poisson_tv_pdhg.py
- experiments/run_poisson_mbir_mask_recon.py
- src/mask_xfct_model.py
- src/reporting_roi.py
- results/corrected_grid9_stage1/corrected_grid9_report.md
- results/mask_design_corrected/pareto_candidates.json
Objective:
Make the raw-domain Poisson-TV PDHG pipeline scientifically usable, not just runnable. Reconstruct from raw/properly matched Poisson mask-domain data for corrected grid9 and selected sparse candidates. Report projection fit, residual structure, ROI bias, CNR linearity, valid/invalid DL, and convergence diagnostics.
Model:
min over f >= 0:
    sum_i [lambda_i(f) - y_i log(lambda_i(f) + eps)] + beta TV(f)
lambda(f) = A f + b
Required loop:
1. Add or verify numerical checks for positivity, finite lambda, finite objective, stable primal/dual residuals, and monotone or explainable objective behavior.
2. Validate the operator with adjoint tests and small synthetic tests before real/proper projection runs.
3. Tune only within documented parameter grids. Do not silently cherry-pick beta or iteration count.
4. Run corrected grid9 and the selected sparse candidates.
5. Generate residual maps and ROI/CNR/DL reports with invalid-DL flags.
6. Compare against corrected EM-TV baseline without overstating conclusions.
Required artifacts:
- results/poisson_mbir_corrected/poisson_mbir_summary.csv
- results/poisson_mbir_corrected/parameter_grid.csv
- results/poisson_mbir_corrected/reconstruction_panels/
- results/poisson_mbir_corrected/residual_maps/
- results/poisson_mbir_corrected/mbir_report.md
- results/poisson_mbir_corrected/commands_run.md
- results/poisson_mbir_corrected/STAGE3_DONE.md
Acceptance checks:
- No negative or nonsensical DL is interpreted as valid.
- Every reported reconstruction has matching forward model support and projection domain.
- The report distinguishes synthetic smoke tests from real/properly matched projection results.
- At least corrected grid9 and one sparse candidate are reconstructed, unless BLOCKED.md explains the precise blocker.
```

## Unlocks

When `results/poisson_mbir_corrected/STAGE3_DONE.md` exists and documents passed acceptance checks, Codex may read `goal/stage4.md`.
