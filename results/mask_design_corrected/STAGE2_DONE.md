# Stage 2 Done

Completed on 2026-05-10 Asia/Shanghai.

## Objective Restated

Re-screen the mask candidates under true `physical_padded` detector support after
the corrected clipped grid9 validation passed, then produce a small Pareto
shortlist for later explicit matrix generation and reconstruction. The shortlist
is a design-screening hypothesis set, not a final reconstruction conclusion.

## Prompt-To-Artifact Checklist

| requirement | evidence | status |
| --- | --- | --- |
| Use corrected physical support | `candidate_screening_corrected.csv` has 90 rows and every row has `support_mode=physical_padded` | PASS |
| Use clipped validation provenance | `top_candidates.json` records `Forward validation passed.` from `results/forward_model_validation_clipped/validation_summary.json` | PASS |
| Do not use old quick/provisional ranking as final | Corrected command used no `--quick`, screened current 90 candidate JSONs, and wrote new outputs under `results/mask_design_corrected/` | PASS |
| Include required metrics | CSV/report include sensitivity mean/CV, ROI sensitivity, physical truncation mean/p95, overlap mean/max, coherence max, task Fisher d2, and task CRLB | PASS |
| Include single and corrected grid9 anchors | `pareto_candidates.json` baselines include `single_center_n1_d1d25_mind0` and `grid3x3_n9_d1d25_mind6` | PASS |
| Include specified blue-noise anchor | Primary shortlist includes `blue_noise_n3_d1d25_mind3_s1` | PASS |
| Include low-overlap 5-hole blue-noise candidate | Primary shortlist includes `blue_noise_n5_d1d25_mind6_s0` | PASS |
| Include low-truncation ring/ring_two candidate | Primary shortlist includes `ring_n7_d1d25_mind3` | PASS |
| Keep shortlist small | `pareto_candidates.json` has 5 primary candidates plus 2 baselines | PASS |
| Avoid throughput-only selection | Ranking note and report state penalties for truncation, overlap, weighted coherence, and sensitivity nonuniformity alongside task Fisher metrics | PASS |
| State screening is not final reconstruction evidence | `screening_report.md` explicitly says rankings are design-screening hypotheses and not final reconstruction conclusions | PASS |
| Save commands | `commands_run.md` exists | PASS |

## Shortlist

Baselines:

- `single_center_n1_d1d25_mind0`
- `grid3x3_n9_d1d25_mind6`

Primary candidates:

- `blue_noise_n3_d1d25_mind3_s1`
- `blue_noise_n5_d1d25_mind6_s0`
- `ring_n7_d1d25_mind3`
- `cross_plus_center_n5_d1d25_mind3`
- `cross_plus_center_n5_d0d75_mind3`

## Completion Audit

Final audit command printed `STAGE2_COMPLETION_AUDIT_PASS` after checking
required artifacts, row count, `physical_padded` support, required metric
columns, clipped-validation provenance, baseline anchors, primary shortlist
contents, shortlist size, report scope, and non-quick command usage.

No Stage 2 requirement remains missing, incomplete, weakly verified, or blocked.
