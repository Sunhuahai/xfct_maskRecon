# Stage 2 Goal: Corrected Physical-Support Candidate Screening

This stage is locked until `results/corrected_grid9_stage1/STAGE1_DONE.md` exists.

Codex must not read or execute this file before Stage 1 is complete unless the user explicitly overrides the gate.

## Purpose

Re-screen mask candidates under physical detector support after corrected grid9 validation. Produce a small Pareto candidate set, not a single opaque winner.

## Candidate Anchors

- `single_center`
- corrected `grid9`
- `blue_noise_n3_d1d25_mind3_s1`
- one low-overlap 5-hole blue-noise candidate
- one low-truncation ring or ring_two candidate

## Codex `/goal` Prompt

```text
/goal Complete corrected physical-support mask candidate screening and select a small Pareto candidate set without stopping until results/mask_design_corrected/STAGE2_DONE.md exists and all acceptance checks pass.
Read first:
- docs/current_state_handoff_gpt_pro.md
- src/mask_xfct_model.py
- scripts/generate_mask_candidates.py
- experiments/screen_mask_candidates.py
- results/forward_model_validation_clipped/validation_summary.json
- results/mask_design/candidate_manifest.csv
- results/mask_design/candidate_screening.csv
- results/mask_design/top_candidates.json
- results/corrected_grid9_stage1/corrected_grid9_report.md
Objective:
Re-run or harden candidate screening under support_mode="physical_padded" after the clipped explicit grid9 validation has passed. Produce a scientifically defensible shortlist of sparse detector-side multi-pinhole masks for later explicit matrix generation and full reconstruction.
Constraints:
- Do not treat old quick/provisional screening rankings as final.
- Do not select candidates only by throughput or total open area.
- Penalize physical detector truncation, overlap, high coherence, sensitivity nonuniformity, and poor task Fisher / CRLB.
- Include corrected grid9 and single_center as anchors.
- Prefer a small Pareto set over a single opaque winner.
Required artifacts:
- results/mask_design_corrected/candidate_screening_corrected.csv
- results/mask_design_corrected/pareto_candidates.json
- results/mask_design_corrected/screening_report.md
- results/mask_design_corrected/commands_run.md
- results/mask_design_corrected/STAGE2_DONE.md
Acceptance checks:
- Screening uses physical_padded support or an equivalent true physical 80 x 80 detector support.
- Report includes sensitivity mean/CV, ROI sensitivity, physical truncation mean/p95, overlap mean/max, coherence max, task Fisher d2, task CRLB.
- Shortlist includes no more than 5 primary candidates plus baselines.
- The report explicitly says candidate rankings are design-screening hypotheses, not final reconstruction conclusions.
- If any script needed modification, changes are minimal, backward compatible, and documented.
```

## Unlocks

When `results/mask_design_corrected/STAGE2_DONE.md` exists and documents passed acceptance checks, Codex may read `goal/stage3.md`.
