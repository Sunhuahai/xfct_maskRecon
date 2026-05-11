# Mask Pose and Geometry Sensitivity

Reconstructions use the nominal forward model while expected projection data are generated with each listed perturbation. DL changes are interpreted only when both nominal and perturbed rows pass the shared CNR quality gate.

| candidate | perturbation | DL change | ROI bias change | deviance increase | residual structure |
| --- | --- | ---: | ---: | ---: | ---: |
| grid9_p6_d1p25_5 | nominal | invalid | 0.0000 | 0.000e+00 | 0.9948 |
| grid9_p6_d1p25_5 | mask_dx_p0.1 | invalid | -0.0056 | -2.494e+04 | 0.9948 |
| grid9_p6_d1p25_5 | detector_distance_p0.5 | invalid | -0.0525 | -2.600e+05 | 0.9948 |
| grid9_p6_d1p25_5 | angle_p0.5 | invalid | -0.0009 | -2.961e+03 | 0.9948 |
| grid9_p6_d1p25_5 | center_jitter_sigma0.05 | invalid | -0.0211 | -1.020e+05 | 0.9948 |
| blue_noise_n3_d1d25_mind3_s1 | nominal | invalid | 0.0000 | 0.000e+00 | 0.9536 |
| blue_noise_n3_d1d25_mind3_s1 | mask_dx_p0.1 | invalid | -0.0080 | -3.546e+04 | 0.8768 |
| blue_noise_n3_d1d25_mind3_s1 | detector_distance_p0.5 | invalid | -0.0361 | -1.784e+05 | 0.8077 |
| blue_noise_n3_d1d25_mind3_s1 | angle_p0.5 | invalid | -0.0003 | 1.017e+03 | 0.8992 |
| blue_noise_n3_d1d25_mind3_s1 | center_jitter_sigma0.05 | invalid | -0.0486 | -2.468e+05 | 0.9346 |