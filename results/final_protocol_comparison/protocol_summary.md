# Mask Protocol Comparison

Protocols: equal acquisition time, equal incident dose, and equal detected counts.

Counts normalization:

- `equal_acquisition_time`: all runs use the exposure scale calibrated from `traditional_5` to the target count level; multi-hole masks are allowed to produce different detected counts through throughput.
- `equal_incident_dose`: identical to equal acquisition time in this synthetic incident-flux model; these rows are cloned from the equal-acquisition reconstruction rows and marked in `protocol_reused_from`.
- `equal_detected_counts`: each run receives its own exposure scale so expected detected counts are normalized to the requested target; conclusions from this protocol are reported separately from equal time/dose.

DL validity uses the shared CNR quality gate from `src.reporting_roi`: finite positive-slope fit, R2 >= 0.80, monotonic CNR response, and stable background noise.

## Aggregate Summary

| protocol | run | family | seeds | valid DL | counts mean | valid DL mean | valid DL std | ROI bias mean | residual structure mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_acquisition_time | blue_noise_n3_d1d25_mind3_s1 | blue_noise | 2 | 0/2 | 4.985e+05 | nan | nan | -0.5463 | 0.2962 |
| equal_acquisition_time | grid9_p6_d1p25_5 | grid3x3 | 2 | 0/2 | 1.326e+06 | nan | nan | 12.9221 | 0.9958 |
| equal_acquisition_time | ring_n7_d1d25_mind3 | ring | 2 | 0/2 | 1.611e+06 | nan | nan | 10.2674 | 0.9908 |
| equal_acquisition_time | traditional_15 | single_pinhole | 2 | 2/2 | 5.989e+05 | 0.1764 | 0.1399 | -0.3384 | 0.4018 |
| equal_acquisition_time | traditional_45 | single_pinhole | 2 | 0/2 | 1.797e+06 | nan | nan | -0.1059 | 0.4701 |
| equal_acquisition_time | traditional_5 | single_pinhole | 2 | 1/2 | 2.000e+05 | 0.5336 | 0.0000 | -0.6674 | 0.3595 |
| equal_detected_counts | blue_noise_n3_d1d25_mind3_s1 | blue_noise | 2 | 2/2 | 1.998e+05 | 1.1022 | 0.0313 | -0.5964 | 0.2535 |
| equal_detected_counts | grid9_p6_d1p25_5 | grid3x3 | 2 | 0/2 | 1.999e+05 | nan | nan | -0.7147 | 0.6015 |
| equal_detected_counts | ring_n7_d1d25_mind3 | ring | 2 | 0/2 | 2.000e+05 | nan | nan | -1.1109 | 0.4046 |
| equal_detected_counts | traditional_15 | single_pinhole | 2 | 2/2 | 2.000e+05 | 0.4874 | 0.1268 | -0.2943 | 0.2209 |
| equal_detected_counts | traditional_45 | single_pinhole | 2 | 2/2 | 1.995e+05 | 0.1280 | 0.0206 | -0.0589 | 0.1452 |
| equal_detected_counts | traditional_5 | single_pinhole | 2 | 1/2 | 2.000e+05 | 0.5336 | 0.0000 | -0.6674 | 0.3595 |
| equal_incident_dose | blue_noise_n3_d1d25_mind3_s1 | blue_noise | 2 | 0/2 | 4.985e+05 | nan | nan | -0.5463 | 0.2962 |
| equal_incident_dose | grid9_p6_d1p25_5 | grid3x3 | 2 | 0/2 | 1.326e+06 | nan | nan | 12.9221 | 0.9958 |
| equal_incident_dose | ring_n7_d1d25_mind3 | ring | 2 | 0/2 | 1.611e+06 | nan | nan | 10.2674 | 0.9908 |
| equal_incident_dose | traditional_15 | single_pinhole | 2 | 2/2 | 5.989e+05 | 0.1764 | 0.1399 | -0.3384 | 0.4018 |
| equal_incident_dose | traditional_45 | single_pinhole | 2 | 0/2 | 1.797e+06 | nan | nan | -0.1059 | 0.4701 |
| equal_incident_dose | traditional_5 | single_pinhole | 2 | 1/2 | 2.000e+05 | 0.5336 | 0.0000 | -0.6674 | 0.3595 |

## Seed-Level Rows

| protocol | run | seed | counts | DL | invalid DL | ROI bias | deviance | residual structure |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| equal_acquisition_time | blue_noise_n3_d1d25_mind3_s1 | 20260509 | 4.983e+05 | 6.3055 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -0.7422 | 1.275e+07 | 0.3112 |
| equal_acquisition_time | blue_noise_n3_d1d25_mind3_s1 | 20260510 | 4.986e+05 | 7.0494 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -0.3504 | 1.788e+07 | 0.2812 |
| equal_acquisition_time | grid9_p6_d1p25_5 | 20260509 | 1.325e+06 | 22.0423 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 13.0390 | 5.075e+08 | 0.9958 |
| equal_acquisition_time | grid9_p6_d1p25_5 | 20260510 | 1.327e+06 | 21.9951 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 12.8052 | 4.986e+08 | 0.9958 |
| equal_acquisition_time | ring_n7_d1d25_mind3 | 20260509 | 1.611e+06 | 25.1924 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 10.5312 | 7.182e+08 | 0.9909 |
| equal_acquisition_time | ring_n7_d1d25_mind3 | 20260510 | 1.610e+06 | 25.0261 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 10.0035 | 6.987e+08 | 0.9907 |
| equal_acquisition_time | traditional_15 | 20260509 | 5.990e+05 | 0.2754 | valid | -0.3295 | 4.129e+05 | 0.3894 |
| equal_acquisition_time | traditional_15 | 20260510 | 5.989e+05 | 0.0775 | valid | -0.3474 | 5.155e+05 | 0.4142 |
| equal_acquisition_time | traditional_45 | 20260509 | 1.795e+06 | -0.1576 | negative detection limit | -0.1036 | 2.103e+06 | 0.4550 |
| equal_acquisition_time | traditional_45 | 20260510 | 1.799e+06 | -0.1301 | negative detection limit | -0.1082 | 2.623e+06 | 0.4852 |
| equal_acquisition_time | traditional_5 | 20260509 | 2.002e+05 | 0.5336 | valid | -0.6619 | 8.854e+04 | 0.3390 |
| equal_acquisition_time | traditional_5 | 20260510 | 1.998e+05 | 0.4994 | non-monotonic CNR concentration response | -0.6728 | 1.340e+05 | 0.3800 |
| equal_incident_dose | blue_noise_n3_d1d25_mind3_s1 | 20260509 | 4.983e+05 | 6.3055 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -0.7422 | 1.275e+07 | 0.3112 |
| equal_incident_dose | blue_noise_n3_d1d25_mind3_s1 | 20260510 | 4.986e+05 | 7.0494 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -0.3504 | 1.788e+07 | 0.2812 |
| equal_incident_dose | grid9_p6_d1p25_5 | 20260509 | 1.325e+06 | 22.0423 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 13.0390 | 5.075e+08 | 0.9958 |
| equal_incident_dose | grid9_p6_d1p25_5 | 20260510 | 1.327e+06 | 21.9951 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 12.8052 | 4.986e+08 | 0.9958 |
| equal_incident_dose | ring_n7_d1d25_mind3 | 20260509 | 1.611e+06 | 25.1924 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 10.5312 | 7.182e+08 | 0.9909 |
| equal_incident_dose | ring_n7_d1d25_mind3 | 20260510 | 1.610e+06 | 25.0261 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | 10.0035 | 6.987e+08 | 0.9907 |
| equal_incident_dose | traditional_15 | 20260509 | 5.990e+05 | 0.2754 | valid | -0.3295 | 4.129e+05 | 0.3894 |
| equal_incident_dose | traditional_15 | 20260510 | 5.989e+05 | 0.0775 | valid | -0.3474 | 5.155e+05 | 0.4142 |
| equal_incident_dose | traditional_45 | 20260509 | 1.795e+06 | -0.1576 | negative detection limit | -0.1036 | 2.103e+06 | 0.4550 |
| equal_incident_dose | traditional_45 | 20260510 | 1.799e+06 | -0.1301 | negative detection limit | -0.1082 | 2.623e+06 | 0.4852 |
| equal_incident_dose | traditional_5 | 20260509 | 2.002e+05 | 0.5336 | valid | -0.6619 | 8.854e+04 | 0.3390 |
| equal_incident_dose | traditional_5 | 20260510 | 1.998e+05 | 0.4994 | non-monotonic CNR concentration response | -0.6728 | 1.340e+05 | 0.3800 |
| equal_detected_counts | blue_noise_n3_d1d25_mind3_s1 | 20260509 | 1.998e+05 | 1.1244 | valid | -0.5819 | 5.305e+04 | 0.2557 |
| equal_detected_counts | blue_noise_n3_d1d25_mind3_s1 | 20260510 | 1.998e+05 | 1.0801 | valid | -0.6108 | 8.380e+04 | 0.2512 |
| equal_detected_counts | grid9_p6_d1p25_5 | 20260509 | 1.997e+05 | 3.4822 | non-monotonic CNR concentration response | -0.7278 | 7.119e+04 | 0.6510 |
| equal_detected_counts | grid9_p6_d1p25_5 | 20260510 | 2.002e+05 | 3.0386 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -0.7017 | 5.392e+04 | 0.5520 |
| equal_detected_counts | ring_n7_d1d25_mind3 | 20260509 | 2.003e+05 | 10.9710 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response; unstable background estimate std<=1.0e-08 | -1.1308 | 5.045e+04 | 0.4503 |
| equal_detected_counts | ring_n7_d1d25_mind3 | 20260510 | 1.996e+05 | -0.1502 | negative detection limit; poor CNR linear fit R2<0.80; non-monotonic CNR concentration response | -1.0910 | 4.391e+04 | 0.3589 |
| equal_detected_counts | traditional_15 | 20260509 | 2.001e+05 | 0.3977 | valid | -0.2880 | 1.002e+05 | 0.2111 |
| equal_detected_counts | traditional_15 | 20260510 | 1.998e+05 | 0.5771 | valid | -0.3005 | 1.158e+05 | 0.2308 |
| equal_detected_counts | traditional_45 | 20260509 | 1.991e+05 | 0.1426 | valid | -0.0429 | 2.236e+05 | 0.1434 |
| equal_detected_counts | traditional_45 | 20260510 | 1.999e+05 | 0.1134 | valid | -0.0749 | 2.245e+05 | 0.1470 |
| equal_detected_counts | traditional_5 | 20260509 | 2.002e+05 | 0.5336 | valid | -0.6619 | 8.854e+04 | 0.3390 |
| equal_detected_counts | traditional_5 | 20260510 | 1.998e+05 | 0.4994 | non-monotonic CNR concentration response | -0.6728 | 1.340e+05 | 0.3800 |