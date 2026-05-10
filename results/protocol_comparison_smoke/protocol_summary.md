# Mask Protocol Comparison

Protocols: equal acquisition time, equal incident dose, and equal detected counts.
Dose is controlled by incident exposure scale, not by detected fluorescence count.

| protocol | run | seed | counts | DL | invalid DL | ROI bias | deviance | residual structure |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| equal_acquisition_time | traditional_5 | 20260509 | 2.000e+05 | -8.4146 | True | 7.8904 | 8.882e+06 | 0.2897 |
| equal_acquisition_time | traditional_15 | 20260509 | 6.012e+05 | -3.9697 | True | 33.2090 | 3.346e+08 | 0.9139 |
| equal_acquisition_time | traditional_45 | 20260509 | 1.799e+06 | 23.7736 | True | 73.1226 | 2.042e+09 | 0.9190 |
| equal_acquisition_time | grid9_p6_d1p25_5 | 20260509 | 1.375e+06 | 26.9611 | False | 4.6591 | 3.027e+08 | 0.9904 |
| equal_acquisition_time | best_5hole_sparse_blue_noise_n5_d0d75_mind3_s0 | 20260509 | 2.397e+05 | -2.1385 | True | 0.8995 | 2.516e+07 | 0.9648 |
| equal_incident_dose | traditional_5 | 20260509 | 2.000e+05 | -8.4146 | True | 7.8904 | 8.882e+06 | 0.2897 |
| equal_incident_dose | traditional_15 | 20260509 | 6.012e+05 | -3.9697 | True | 33.2090 | 3.346e+08 | 0.9139 |
| equal_incident_dose | traditional_45 | 20260509 | 1.799e+06 | 23.7736 | True | 73.1226 | 2.042e+09 | 0.9190 |
| equal_incident_dose | grid9_p6_d1p25_5 | 20260509 | 1.375e+06 | 26.9611 | False | 4.6591 | 3.027e+08 | 0.9904 |
| equal_incident_dose | best_5hole_sparse_blue_noise_n5_d0d75_mind3_s0 | 20260509 | 2.397e+05 | -2.1385 | True | 0.8995 | 2.516e+07 | 0.9648 |
| equal_detected_counts | traditional_5 | 20260509 | 2.000e+05 | -8.4146 | True | 7.8904 | 8.882e+06 | 0.2897 |
| equal_detected_counts | traditional_15 | 20260509 | 2.005e+05 | 4.2601 | False | 13.4505 | 3.040e+07 | 0.2529 |
| equal_detected_counts | traditional_45 | 20260509 | 1.995e+05 | 4.5679 | False | 71.5562 | 1.884e+08 | 0.9061 |
| equal_detected_counts | grid9_p6_d1p25_5 | 20260509 | 2.004e+05 | -26.9795 | True | 39.2768 | 2.325e+08 | 0.9879 |
| equal_detected_counts | best_5hole_sparse_blue_noise_n5_d0d75_mind3_s0 | 20260509 | 2.006e+05 | 7.5055 | False | 4.7669 | 4.673e+07 | 0.9575 |