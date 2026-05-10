# Mask Reconstruction Effect Comparison

| run | method | angles | decode | DL (mg/ml) | R2 | counts | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| traditional_5 | em_tv | 5 | none | 0.5496 | 0.9470 | 2.1981e+05 | Single-pinhole 5-angle simulation baseline. |
| mask_5_naive | em_tv | 5 | none | 6.4159 | 0.7887 | 1.0913e+07 | 9-hole mask projection reconstructed with the single-pinhole matrix. |
| mask_5_wiener | em_tv | 5 | wiener | -13.7837 | 0.4732 | 1.2523e+07 | 9-hole mask projection decoded by a fixed-shift Wiener baseline before reconstruction. |

CSV: `results/smoke/effect_comparison.csv`