# Mask Reconstruction Effect Comparison

| run | method | angles | decode | DL (mg/ml) | R2 | counts | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| traditional_5 | em_tv | 5 | none | 0.5496 | 0.9470 | 2.1981e+05 | Single-pinhole 5-angle simulation baseline. |
| mask_5_model | em_tv | 5 | none | 8.4869 | 0.7999 | 1.0913e+07 | 9-hole mask projection reconstructed directly with a matching multi-hole system matrix. |

CSV: `results/effect_comparison_clipped_quick/effect_comparison.csv`