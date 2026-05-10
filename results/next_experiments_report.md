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

Overall validation status: **FAIL**.
80x80 to 80x160 padding result: **FAIL; virtual fraction=0.3183024686313407**.

Stop before expensive mask sweeps. Fail-critical tests failed: detector_padding, delta_voxel_tests. Recommended first fix: make the projection generator and system matrix use the same detector support (either clip all matrix rows to the physical 80-column detector before padding, or regenerate projections for a true 160-column detector).

## Candidate Mask Ranking

`blue_noise_n3_d1d25_mind3_s1` (blue_noise), holes=3, diameter=1.25 mm, min distance=3.0 mm, score=-1.1914. Warning: Forward validation did not pass; screening rankings are provisional and should not drive final mask selection.

| rank | candidate | family | score | truncation | overlap | Fisher d2 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | blue_noise_n3_d1d25_mind3_s1 | blue_noise | -1.1914 | 0.4889 | 0.0384 | 8.7342e-05 |
| 2 | blue_noise_n3_d0d75_mind3_s1 | blue_noise | -1.3047 | 0.4913 | 0.0384 | 2.5525e-05 |
| 3 | blue_noise_n3_d1d25_mind3_s0 | blue_noise | -1.5602 | 0.5252 | 0.0395 | 7.2367e-05 |
| 4 | blue_noise_n3_d1d25_mind6_s0 | blue_noise | -1.6397 | 0.5076 | 0.0988 | 6.7967e-05 |
| 5 | blue_noise_n3_d0d75_mind3_s0 | blue_noise | -1.6825 | 0.5271 | 0.0395 | 2.1268e-05 |

## Raw Poisson MBIR

| candidate | family | counts | DL | R2 | deviance | residual structure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| grid9_p6_d1p25_5 | grid3x3 | 2.004e+05 | -27.8236 | 0.1861 | 2.189e+08 | 0.9873 |
| blue_noise_n3_d1d25_mind3_s1 | blue_noise | 2.002e+05 | -54.0630 | 0.0121 | 7.600e+06 | 0.2277 |

## Fair Protocols

| protocol | best valid single DL | best valid multi DL | best valid multi run | interpretation |
| --- | ---: | ---: | --- | --- |
| equal_acquisition_time | 23.9492 | 26.9142 | grid9_p6_d1p25_5 | multi-hole not lower valid DL |
| equal_detected_counts | 4.2601 | nan | NA | no valid DL comparison |
| equal_incident_dose | 23.9492 | 26.9142 | grid9_p6_d1p25_5 | multi-hole not lower valid DL |

Interpretation:

The quick equal-detected-count runs do not provide a reliable valid-DL comparison because the CNR fits are invalid or nonsensical for the relevant rows. Treat these protocol results as smoke-test outputs only.

## Improvement Status

| question | status |
| --- | --- |
| raw throughput | provisional; validation failed; best multi/single count ratio about 6.88 in quick equal-time runs. |
| Fisher/task detectability | provisional; validation failed; top screened candidate `blue_noise_n3_d1d25_mind3_s1` has task Fisher d2=8.734e-05, but quick screening did not establish a validated grid-vs-new improvement. |
| reconstruction DL | provisional; validation failed; best valid multi does not improve over best valid single in quick rows. |
| ROI bias | provisional; validation failed; best abs multi bias=1.0573, best abs single bias=7.8746 in quick protocol rows. |
| projection fit | provisional; validation failed; lowest raw-MBIR deviance `blue_noise_n3_d1d25_mind3_s1`, lowest residual structure `blue_noise_n3_d1d25_mind3_s1`. |
| robustness | provisional; validation failed; lowest mean DL perturbation is `grid9_p6_d1p25_5` (0.4876 mg/ml). |

## Pose Robustness

| candidate | mean | max | note |
| --- | ---: | ---: | --- |
| blue_noise_n3_d1d25_mind3_s1 | 0.8572 | 3.5787 | mean/max absolute DL change |
| grid9_p6_d1p25_5 | 0.4876 | 1.4816 | mean/max absolute DL change |

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