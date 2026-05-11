# Stage 4 Done

Stage 4 completed with corrected finite-aperture protocol comparison artifacts in `results/final_protocol_comparison`.

Required artifacts:

- `protocol_summary.csv`
- `protocol_summary.md`
- `pose_sensitivity_summary.csv`
- `final_report.md`
- `commands_run.md`
- `STAGE4_DONE.md`

Completion notes:

- `protocol_summary.csv` has 36 rows: 3 protocols x 6 runs x 2 seeds.
- All protocols include the traditional 5/15/45 baselines, corrected grid9, blue-noise, and ring candidates.
- `equal_incident_dose` rows are cloned from `equal_acquisition_time` and explicitly marked with `protocol_reused_from=equal_acquisition_time`.
- Seed-by-seed DL validity and invalidity reasons are recorded in the CSV.
- Equal detected-count conclusions are separated in `final_report.md`.
- Pose sensitivity was completed as a finite-aperture minimal diagnostic and is not used for DL-based robustness ranking because all pose DL rows are invalid.
