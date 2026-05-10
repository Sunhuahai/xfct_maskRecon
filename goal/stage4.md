# Stage 4 Goal: Final Multi-Seed Protocol Comparison

This stage is locked until `results/poisson_mbir_corrected/STAGE3_DONE.md` exists.

Codex must not read or execute this file before Stage 3 is complete unless the user explicitly overrides the gate.

## Purpose

Run final protocol-level evidence only after corrected baseline, candidate screening, and MBIR hardening are complete.

## Protocols

1. equal acquisition time
2. equal incident dose
3. equal detected counts

The equal detected-counts protocol is critical because it removes raw throughput advantage and tests coding/inverse-problem quality.

## Codex `/goal` Prompt

```text
/goal Complete final multi-seed XFCT mask protocol comparison under equal acquisition time, equal incident dose, and equal detected counts without stopping until results/final_protocol_comparison/STAGE4_DONE.md exists and all acceptance checks pass.
Read first:
- docs/current_state_handoff_gpt_pro.md
- experiments/run_mask_protocol_comparison.py
- experiments/run_mask_pose_sensitivity.py
- results/corrected_grid9_stage1/corrected_grid9_report.md
- results/mask_design_corrected/pareto_candidates.json
- results/poisson_mbir_corrected/mbir_report.md
Objective:
Produce final protocol-level evidence comparing single-pinhole 5/15/45-view baselines, corrected grid9, and selected sparse mask candidates under:
1. equal_acquisition_time
2. equal_incident_dose
3. equal_detected_counts
Use multiple seeds and report mean +/- std with valid-DL seed counts.
Required artifacts:
- results/final_protocol_comparison/protocol_summary.csv
- results/final_protocol_comparison/protocol_summary.md
- results/final_protocol_comparison/pose_sensitivity_summary.csv
- results/final_protocol_comparison/final_report.md
- results/final_protocol_comparison/commands_run.md
- results/final_protocol_comparison/STAGE4_DONE.md
Acceptance checks:
- Each protocol has all required baselines and selected mask candidates.
- Counts normalization is documented for each protocol.
- DL validity is reported seed-by-seed and aggregated.
- Equal detected counts conclusions are separated from equal time / equal dose conclusions.
- Robustness conclusions are based on stable reconstruction outputs, not quick synthetic smoke tests.
- The final report explicitly lists non-conclusions and limitations.
```

## Completion

When `results/final_protocol_comparison/STAGE4_DONE.md` exists and documents passed acceptance checks, the staged goal program is complete.
