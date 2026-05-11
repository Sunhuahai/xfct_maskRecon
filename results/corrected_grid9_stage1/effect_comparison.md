# Mask Reconstruction Effect Comparison

| run | method | angles | decode | DL status | R2 | counts | note |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| traditional_5 | em_tv | 5 | none | valid 0.2625 mg/ml | 0.9600 | 2.1981e+05 | Single-pinhole 5-angle simulation baseline. |
| traditional_15 | em_tv | 15 | none | valid 0.0633 mg/ml | 0.9943 | 6.5731e+05 | Single-pinhole 15-angle conventional comparison. |
| traditional_45 | em_tv | 45 | none | valid 0.0039 mg/ml | 0.9993 | 1.9684e+06 | Single-pinhole 45-angle conventional upper reference. |
| mask_5_model | em_tv | 5 | none | invalid; raw=19.6449; poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 0.4637 | 1.0913e+07 | 9-hole mask projection reconstructed directly with a matching multi-hole system matrix. |

CSV: `results/corrected_grid9_stage1/effect_comparison.csv`