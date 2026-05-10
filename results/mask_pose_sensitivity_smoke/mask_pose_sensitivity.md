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
| blue_noise_n3_d0d75_mind3_s0 | nominal | 0.0000 | 0.0000 | 0.000e+00 | 0.3674 |
| blue_noise_n3_d0d75_mind3_s0 | mask_dx_p0.1 | 1.4135 | 0.2514 | 1.499e+04 | 0.5159 |
| blue_noise_n3_d0d75_mind3_s0 | mask_dx_m0.1 | 1.2756 | -0.1644 | 2.544e+04 | 0.3366 |
| blue_noise_n3_d0d75_mind3_s0 | mask_dz_p0.1 | -5.1919 | 5.2584 | 1.423e+06 | 0.2769 |
| blue_noise_n3_d0d75_mind3_s0 | mask_dz_m0.1 | -9.4883 | 0.1088 | 5.406e+05 | 0.2890 |
| blue_noise_n3_d0d75_mind3_s0 | detector_distance_p0.1 | 0.8349 | 0.1161 | -2.283e+03 | 0.3320 |
| blue_noise_n3_d0d75_mind3_s0 | detector_distance_m0.1 | -0.6965 | -0.1001 | 1.914e+04 | 0.3660 |
| blue_noise_n3_d0d75_mind3_s0 | detector_distance_p0.5 | 3.5084 | 0.7902 | 8.572e+04 | 0.5030 |
| blue_noise_n3_d0d75_mind3_s0 | detector_distance_m0.5 | -37.0281 | 0.5029 | 7.313e+04 | 0.3361 |
| blue_noise_n3_d0d75_mind3_s0 | detector_offset_p0.1 | -0.6344 | -0.1035 | 1.412e+04 | 0.3218 |
| blue_noise_n3_d0d75_mind3_s0 | detector_offset_m0.1 | 0.6473 | 0.0738 | -9.224e+03 | 0.3172 |
| blue_noise_n3_d0d75_mind3_s0 | rotation_center_p0.1 | 1.6264 | 0.2449 | 2.331e+04 | 0.3559 |
| blue_noise_n3_d0d75_mind3_s0 | rotation_center_m0.1 | -0.3533 | -0.1279 | -1.050e+04 | 0.3535 |
| blue_noise_n3_d0d75_mind3_s0 | angle_p0.1 | -0.1321 | -0.0171 | 4.197e+03 | 0.3617 |
| blue_noise_n3_d0d75_mind3_s0 | angle_m0.1 | -0.0111 | -0.0040 | 1.582e+03 | 0.3573 |
| blue_noise_n3_d0d75_mind3_s0 | angle_p0.5 | -0.2142 | -0.0295 | 1.359e+04 | 0.3215 |
| blue_noise_n3_d0d75_mind3_s0 | angle_m0.5 | 0.0266 | -0.0111 | 4.229e+03 | 0.3267 |
| blue_noise_n3_d0d75_mind3_s0 | center_jitter_sigma0.05 | 4.6721 | 1.0986 | 2.704e+05 | 0.3194 |

## Robustness Comparison

- `grid9_p6_d1p25_5` mean absolute DL change: 0.4876 mg/ml
- `blue_noise_n3_d0d75_mind3_s0` mean absolute DL change: 3.9856 mg/ml