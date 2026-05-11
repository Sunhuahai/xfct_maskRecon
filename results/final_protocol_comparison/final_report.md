# Final Protocol Comparison Report

## Artifacts

- Seed-level protocol CSV: `results/final_protocol_comparison/protocol_summary.csv`
- Protocol summary Markdown: `results/final_protocol_comparison/protocol_summary.md`
- Pose sensitivity CSV: `results/final_protocol_comparison/pose_sensitivity_summary.csv`
- Pose sensitivity Markdown: `results/final_protocol_comparison/pose_sensitivity_summary.md`
- Command log: `results/final_protocol_comparison/commands_run.md`

## Protocol Design

The protocol comparison used the corrected physical-padded support path, finite-aperture PMMA operators, two seeds (`20260509`, `20260510`), and 15 Poisson-TV PDHG iterations. Each protocol contains `traditional_5`, `traditional_15`, `traditional_45`, corrected grid9 (`grid9_p6_d1p25_5`), `blue_noise_n3_d1d25_mind3_s1`, and `ring_n7_d1d25_mind3`.

Counts normalization:

- `equal_acquisition_time`: exposure is calibrated from `traditional_5`; multi-hole masks are allowed to collect different detected counts.
- `equal_incident_dose`: identical to equal acquisition time in this synthetic incident-flux model; rows are cloned from equal acquisition and marked by `protocol_reused_from=equal_acquisition_time`.
- `equal_detected_counts`: exposure is normalized per run to roughly `2e5` expected detected counts. These conclusions are separate from equal-time/equal-dose conclusions.

## Protocol Results

Under equal acquisition time and equal incident dose, no selected mask candidate had valid DL in either seed. `traditional_15` was valid in both seeds with mean valid DL `0.1764 mg/ml`; `traditional_5` was valid in one of two seeds; `traditional_45` had invalid negative DL despite higher counts. Grid9 and ring also had high residual structure (`0.9958` and `0.9908` mean), so their raw DL values are diagnostic only.

Under equal detected counts, `traditional_45` was valid in both seeds with mean valid DL `0.1280 mg/ml`, `traditional_15` was valid with mean `0.4874 mg/ml`, and blue-noise was valid with mean `1.1022 mg/ml`. Corrected grid9 and ring were DL-invalid in both seeds. This equal-count result must not be read as an equal-dose or equal-time result.

## Pose Sensitivity

Pose sensitivity was run as a finite-aperture PMMA diagnostic with a minimal perturbation profile: nominal, `mask_dx_p0.1`, `detector_distance_p0.5`, `angle_p0.5`, and `center_jitter_sigma0.05` for grid9 and blue-noise. All pose rows were DL-invalid under the shared CNR quality gate; in other words, all pose rows were DL-invalid and no DL-based robustness ranking is made from this sweep. The diagnostic residual scores remained high for grid9 (`~0.995`) and ranged from `0.808` to `0.954` for blue-noise across the minimal perturbations.

## Non-Conclusions And Limitations

- This stage does not establish that a multi-hole mask improves detection limit over traditional acquisition under equal time or equal dose.
- Equal detected-count results are reported separately and do not imply equal dose or equal acquisition time performance.
- Invalid DL values are retained as raw CNR-fit diagnostics only and are not interpreted numerically.
- The protocol comparison used two seeds and 15 iterations to keep the 45-view baseline tractable.
- Pose sensitivity used two candidates, five perturbation cases per candidate, and three iterations; it is a finite-aperture diagnostic, not a complete robustness proof.
- Sparse candidates used synthetic matched projections in these protocol studies; the corrected real raw-domain evidence remains limited to the grid9 projection from earlier stages.
