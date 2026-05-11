# Raw-Domain Poisson MBIR Mask Reconstruction

All reconstructions use raw padded detector measurements with physical 80-column support; no fixed-shift decoding is used.

| candidate | domain | beta | iter | seed | counts | DL flag | raw DL | R2 | deviance | residual structure |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| grid9_p6_d1p25_5 | real_grid9_projection_padded_physical_detector | 1.0e-05 | 30 | 20260509 | 1.091e+07 | invalid: poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 23.2873 | 0.1005 | 4.787e+10 | 0.9961 |
| grid9_p6_d1p25_5 | real_grid9_projection_padded_physical_detector | 1.0e-04 | 30 | 20260509 | 1.091e+07 | invalid: poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 23.2873 | 0.1005 | 4.787e+10 | 0.9961 |
| blue_noise_n3_d1d25_mind3_s1 | synthetic_matched_poisson | 1.0e-05 | 30 | 20260509 | 1.998e+05 | valid | 0.9413 | 0.9663 | 1.954e+04 | 0.1835 |
| blue_noise_n3_d1d25_mind3_s1 | synthetic_matched_poisson | 1.0e-04 | 30 | 20260509 | 1.998e+05 | valid | 0.9412 | 0.9663 | 1.954e+04 | 0.1834 |
| ring_n7_d1d25_mind3 | synthetic_matched_poisson | 1.0e-05 | 30 | 20260509 | 2.003e+05 | valid | 0.2659 | 0.9383 | 2.492e+04 | 0.1963 |
| ring_n7_d1d25_mind3 | synthetic_matched_poisson | 1.0e-04 | 30 | 20260509 | 2.003e+05 | valid | 0.2658 | 0.9383 | 2.492e+04 | 0.1963 |