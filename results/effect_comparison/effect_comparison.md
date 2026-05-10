# Mask Reconstruction Effect Comparison

| run | method | angles | decode | DL (mg/ml) | R2 | counts | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| traditional_5 | em_tv | 5 | none | 0.2625 | 0.9600 | 2.1981e+05 | Single-pinhole 5-angle simulation baseline. |
| mask_5_naive | em_tv | 5 | none | 10.6846 | 0.4798 | 1.0913e+07 | 9-hole mask projection reconstructed with the single-pinhole matrix. |
| mask_5_wiener | em_tv | 5 | wiener | -29.9987 | 0.5131 | 1.2523e+07 | 9-hole mask projection decoded by a fixed-shift Wiener baseline before reconstruction. |
| mask_5_model | em_tv | 5 | none | 9.7767 | 0.4530 | 1.0913e+07 | 9-hole mask projection reconstructed directly with a matching multi-hole system matrix. |
| traditional_15 | em_tv | 15 | none | 0.0633 | 0.9943 | 6.5731e+05 | Single-pinhole 15-angle conventional comparison. |
| traditional_45 | em_tv | 45 | none | 0.0039 | 0.9993 | 1.9684e+06 | Single-pinhole 45-angle conventional upper reference. |

CSV: `results/effect_comparison/effect_comparison.csv`