# Mask Pose and Geometry Sensitivity

| candidate | perturbation | DL change | ROI bias change | deviance increase | residual structure |
| --- | --- | ---: | ---: | ---: | ---: |
| grid9_p6_d1p25_5 | nominal | 0.0000 | 0.0000 | 0.000e+00 | 0.9879 |
| grid9_p6_d1p25_5 | mask_dx_p0.1 | -0.4507 | 0.0222 | 1.327e+05 | 0.9879 |
| grid9_p6_d1p25_5 | mask_dx_m0.1 | -1.0527 | 0.0106 | -1.395e+04 | 0.9879 |
| grid9_p6_d1p25_5 | mask_dz_p0.1 | -1.4108 | 0.7883 | -2.169e+05 | 0.9879 |
| grid9_p6_d1p25_5 | mask_dz_m0.1 | -1.0150 | 1.0274 | 1.500e+06 | 0.9878 |
| grid9_p6_d1p25_5 | detector_distance_p0.1 | 0.1621 | 0.1540 | 1.521e+06 | 0.9878 |
| grid9_p6_d1p25_5 | detector_distance_m0.1 | -0.3322 | -0.1867 | -1.842e+06 | 0.9880 |
| grid9_p6_d1p25_5 | detector_distance_p0.5 | 1.0922 | 0.7263 | 7.520e+06 | 0.9876 |
| grid9_p6_d1p25_5 | detector_distance_m0.5 | -1.4816 | -0.8132 | -7.864e+06 | 0.9881 |
| grid9_p6_d1p25_5 | detector_offset_p0.1 | -0.1658 | 0.0181 | 7.884e+04 | 0.9879 |
| grid9_p6_d1p25_5 | detector_offset_m0.1 | -0.0515 | 0.0169 | 1.093e+05 | 0.9879 |
| grid9_p6_d1p25_5 | rotation_center_p0.1 | 0.0782 | 0.1011 | 1.057e+06 | 0.9879 |
| grid9_p6_d1p25_5 | rotation_center_m0.1 | -0.1957 | -0.0731 | -8.335e+05 | 0.9879 |
| grid9_p6_d1p25_5 | angle_p0.1 | -0.1312 | -0.0020 | -3.702e+04 | 0.9879 |
| grid9_p6_d1p25_5 | angle_m0.1 | 0.0134 | 0.0056 | 4.183e+04 | 0.9879 |
| grid9_p6_d1p25_5 | angle_p0.5 | -0.2811 | 0.0281 | 2.036e+05 | 0.9879 |
| grid9_p6_d1p25_5 | angle_m0.5 | 0.0222 | 0.0000 | -3.451e+04 | 0.9879 |
| grid9_p6_d1p25_5 | center_jitter_sigma0.05 | 0.3526 | 0.5212 | 3.896e+06 | 0.9877 |
| blue_noise_n3_d1d25_mind3_s1 | nominal | 0.0000 | 0.0000 | 0.000e+00 | 0.2274 |
| blue_noise_n3_d1d25_mind3_s1 | mask_dx_p0.1 | 0.5304 | 0.0088 | 3.774e+04 | 0.5454 |
| blue_noise_n3_d1d25_mind3_s1 | mask_dx_m0.1 | -0.0646 | -0.0366 | -1.177e+05 | 0.2683 |
| blue_noise_n3_d1d25_mind3_s1 | mask_dz_p0.1 | 0.9865 | 0.1093 | -3.783e+05 | 0.2178 |
| blue_noise_n3_d1d25_mind3_s1 | mask_dz_m0.1 | -3.4002 | 0.6688 | 8.888e+05 | 0.2437 |
| blue_noise_n3_d1d25_mind3_s1 | detector_distance_p0.1 | 0.5236 | 0.0783 | 3.013e+05 | 0.2382 |
| blue_noise_n3_d1d25_mind3_s1 | detector_distance_m0.1 | -0.6162 | -0.1053 | -2.998e+05 | 0.2208 |
| blue_noise_n3_d1d25_mind3_s1 | detector_distance_p0.5 | 1.8653 | 0.3946 | 2.058e+06 | 0.4979 |
| blue_noise_n3_d1d25_mind3_s1 | detector_distance_m0.5 | -3.5787 | -0.4602 | -3.747e+05 | 0.2476 |
| blue_noise_n3_d1d25_mind3_s1 | detector_offset_p0.1 | -0.0119 | -0.0191 | -6.893e+04 | 0.2534 |
| blue_noise_n3_d1d25_mind3_s1 | detector_offset_m0.1 | 0.1753 | -0.0158 | -4.938e+04 | 0.2196 |
| blue_noise_n3_d1d25_mind3_s1 | rotation_center_p0.1 | 0.2948 | 0.0686 | 2.631e+05 | 0.2503 |
| blue_noise_n3_d1d25_mind3_s1 | rotation_center_m0.1 | -0.4652 | -0.0973 | -2.848e+05 | 0.2097 |
| blue_noise_n3_d1d25_mind3_s1 | angle_p0.1 | -0.1076 | -0.0166 | -6.503e+04 | 0.2183 |
| blue_noise_n3_d1d25_mind3_s1 | angle_m0.1 | -0.0254 | -0.0079 | -3.410e+04 | 0.2246 |
| blue_noise_n3_d1d25_mind3_s1 | angle_p0.5 | -0.1993 | -0.0240 | -9.342e+04 | 0.2473 |
| blue_noise_n3_d1d25_mind3_s1 | angle_m0.5 | 0.0402 | -0.0242 | -8.432e+04 | 0.2220 |
| blue_noise_n3_d1d25_mind3_s1 | center_jitter_sigma0.05 | 1.6880 | 0.5164 | 3.234e+06 | 0.2997 |

## Robustness Comparison

- `grid9_p6_d1p25_5` mean absolute DL change: 0.4876 mg/ml
- `blue_noise_n3_d1d25_mind3_s1` mean absolute DL change: 0.8572 mg/ml