# Next-Stage Multi-Pinhole XFCT Experiment Report

## Repository Changes and Commands

Added scripts/modules:

- `src/mask_xfct_model.py`: depth-dependent matrix-free mask operator and task phantoms.
- `experiments/validate_mask_forward_consistency.py`: forward consistency validation.
- `scripts/generate_mask_candidates.py`: sparse mask candidate generator.
- `experiments/screen_mask_candidates.py`: task-based candidate screening.
- `recon/poisson_tv_pdhg.py`: raw-domain Poisson-TV MBIR solver.
- `experiments/run_poisson_mbir_mask_recon.py`: raw mask-domain MBIR experiment.
- `experiments/run_mask_protocol_comparison.py`: equal time/dose/count comparison.
- `experiments/run_mask_pose_sensitivity.py`: pose/geometry robustness study.
- `experiments/make_next_experiments_report.py`: this report generator.

Primary commands:

```bash
conda run -n xfct python experiments/run_effect_comparison.py --quick
conda run -n xfct python experiments/validate_mask_forward_consistency.py --quick
conda run -n xfct python scripts/generate_mask_candidates.py --quick
conda run -n xfct python experiments/screen_mask_candidates.py --quick --candidate-limit 20
conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --quick
conda run -n xfct python experiments/run_mask_protocol_comparison.py --quick --num-seeds 3
conda run -n xfct python experiments/run_mask_pose_sensitivity.py --quick
conda run -n xfct python experiments/make_next_experiments_report.py
```

## Forward-Model Validation

Overall validation status: **NOT_RUN**.
80x80 to 80x160 padding result: **not assessed**.

Validation has not been run.

## Candidate Mask Ranking

No screened candidate available.

| rank | candidate | family | score | truncation | overlap | Fisher d2 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| NA | not run | NA | NA | NA | NA | NA |

## Raw Poisson MBIR

| candidate | family | counts | DL | R2 | deviance | residual structure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| not run | NA | NA | NA | NA | NA | NA |

## Fair Protocols

Protocol comparison has not been run yet.

Interpretation:

Protocol comparison has not been run.

## Pose Robustness

Pose sensitivity has not been run yet.

## Failure Modes

- FOV truncation: reported as physical-detector truncation before padding and remains a central risk.
- Hole overlap: reported by isolated-hole footprint inner products; high overlap is penalized.
- Forward mismatch: validation explicitly checks physical 80-column support versus padded 160-column rows.
- Poor conditioning: approximated with weighted mutual coherence and task Fisher/CRLB metrics.
- Regularization bias: reported through ROI bias, CNR slope/intercept/R2, and invalid DL flags.
- Pose sensitivity: reported as DL, ROI bias, deviance, and residual-structure change under mask/geometry perturbations.

## Recommendation

Fix forward-model support first. The current evidence should not be used to choose hardware: regenerate the mask matrix with physical 80-column clipping before padding, or regenerate projection data for a true 160-column detector. After that, rerun screening and protocol comparison.

Scientific rule: if multi-hole wins only under equal time/dose but loses under equal detected counts, treat the benefit as throughput rather than coding information.