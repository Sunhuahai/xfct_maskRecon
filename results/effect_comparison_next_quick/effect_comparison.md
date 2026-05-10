# Mask Reconstruction Effect Comparison

| run | method | angles | decode | DL (mg/ml) | R2 | counts | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| traditional_5 | em_tv | 5 | none | 0.5496 | 0.9470 | 2.1981e+05 | Single-pinhole 5-angle simulation baseline. |
| mask_5_naive | em_tv | 5 | none | 6.4159 | 0.7887 | 1.0913e+07 | 9-hole mask projection reconstructed with the single-pinhole matrix. |
| mask_5_wiener | em_tv | 5 | wiener | -13.7837 | 0.4732 | 1.2523e+07 | 9-hole mask projection decoded by a fixed-shift Wiener baseline before reconstruction. |
| mask_5_model | em_tv | 5 | none | 8.0417 | 0.9929 | 1.0913e+07 | 9-hole mask projection reconstructed directly with a matching multi-hole system matrix. |
| traditional_15 | em_tv | 15 | none | 0.5494 | 0.9963 | 6.5731e+05 | Single-pinhole 15-angle conventional comparison. |
| traditional_45 | em_tv | 45 | none | 0.4907 | 0.9977 | 1.9684e+06 | Single-pinhole 45-angle conventional upper reference. |

CSV: `results/effect_comparison_next_quick/effect_comparison.csv`