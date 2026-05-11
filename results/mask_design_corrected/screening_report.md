# Corrected Physical-Support Candidate Screening

Screening support mode: `physical_padded` true physical 80 x 80 detector support embedded in the 80 x 160 padded detector.

Candidate rankings are design-screening hypotheses for later explicit matrix generation and full reconstruction; they are not final reconstruction conclusions.

The combined ranking score is not throughput-only. It combines task Fisher information with penalties for physical detector truncation, isolated-hole footprint overlap, weighted coherence, and sensitivity nonuniformity.

## Shortlist

| role | candidate | family | holes | sensitivity mean | sensitivity CV | ROI sensitivity mean | trunc mean/p95 | overlap mean/max | coherence max | task Fisher d2 | task CRLB | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| baseline | `single_center_n1_d1d25_mind0` | single_center | 1 | 1.020057e-04 | 0.0831 | 9.924224e-05 | 0.0000/0.0000 | 0.0000/0.0000 | 0.6341 | 1.444491e-06 | 737645.0196 | single-center pinhole anchor |
| baseline | `grid3x3_n9_d1d25_mind6` | grid3x3 | 9 | 4.301661e-04 | 0.1213 | 4.293733e-04 | 0.4582/0.5000 | 0.1091/0.5460 | 0.4195 | 2.624233e-06 | 383941.5329 | corrected grid9 geometry anchor |
| primary | `blue_noise_n3_d1d25_mind3_s1` | blue_noise | 3 | 1.589283e-04 | 0.1633 | 1.675269e-04 | 0.3927/0.4250 | 0.0426/0.0648 | 0.4524 | 2.174354e-06 | 485033.2071 | specified blue-noise anchor from the Stage 2 prompt |
| primary | `blue_noise_n5_d1d25_mind6_s0` | blue_noise | 5 | 2.144026e-04 | 0.1643 | 1.792638e-04 | 0.6322/0.6550 | 0.0563/0.1735 | 0.5141 | 1.877126e-06 | 556921.5936 | lowest-overlap 5-hole blue-noise candidate at 1.25 mm diameter |
| primary | `ring_n7_d1d25_mind3` | ring | 7 | 4.924372e-04 | 0.1774 | 5.925547e-04 | 0.1320/0.2071 | 0.1889/0.5695 | 0.5352 | 3.193811e-06 | 315068.1258 | lowest physical-truncation ring/ring_two candidate at 1.25 mm diameter |
| primary | `cross_plus_center_n5_d1d25_mind3` | cross_plus_center | 5 | 4.490529e-04 | 0.1096 | 4.707355e-04 | 0.0444/0.0900 | 0.3774/0.6232 | 0.5357 | 1.675936e-06 | 598012.3132 | non-dominated Pareto-front candidate |
| primary | `cross_plus_center_n5_d0d75_mind3` | cross_plus_center | 5 | 1.619242e-04 | 0.1113 | 1.695662e-04 | 0.0436/0.0900 | 0.3411/0.5836 | 0.5430 | 8.633279e-07 | 1159700.1865 | non-dominated Pareto-front candidate |

## Top Ranking Rows

| rank | candidate | family | holes | score | Fisher d2 | CRLB | trunc mean | overlap max | ROI coherence p95 | sensitivity CV | throughput |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `single_center_n1_d1d25_mind0` | single_center | 1 | -0.0341 | 1.444491e-06 | 737645.0196 | 0.0000 | 0.0000 | 0.1185 | 0.0831 | 1.1038e+00 |
| 2 | `single_center_n1_d0d75_mind0` | single_center | 1 | -0.1127 | 7.058670e-07 | 1495321.1943 | 0.0000 | 0.0000 | 0.1631 | 0.0849 | 4.3873e-01 |
| 3 | `cross_plus_center_n5_d1d25_mind3` | cross_plus_center | 5 | -0.6281 | 1.675936e-06 | 598012.3132 | 0.0444 | 0.6232 | 0.0836 | 0.1096 | 4.7463e+00 |
| 4 | `cross_plus_center_n5_d0d75_mind3` | cross_plus_center | 5 | -0.7293 | 8.633279e-07 | 1159700.1865 | 0.0436 | 0.5836 | 0.1419 | 0.1113 | 1.7519e+00 |
| 5 | `ura_mura_inspired_n3_d1d25_mind3` | ura_mura_inspired | 3 | -0.7699 | 1.390601e-06 | 725890.6111 | 0.0739 | 0.6232 | 0.1104 | 0.1070 | 2.8614e+00 |
| 6 | `ura_mura_inspired_n3_d0d75_mind3` | ura_mura_inspired | 3 | -0.8666 | 7.395141e-07 | 1366562.7719 | 0.0727 | 0.5836 | 0.1738 | 0.1092 | 1.0720e+00 |
| 7 | `ring_n7_d1d25_mind3` | ring | 7 | -0.9527 | 3.193811e-06 | 315068.1258 | 0.1320 | 0.5695 | 0.0725 | 0.1774 | 5.4246e+00 |
| 8 | `ring_n5_d1d25_mind3` | ring | 5 | -0.9901 | 3.273581e-06 | 311312.1463 | 0.1491 | 0.5695 | 0.0761 | 0.1226 | 3.9531e+00 |
| 9 | `ring_n9_d1d25_mind3` | ring | 9 | -0.9916 | 2.627169e-06 | 381523.9817 | 0.1674 | 0.5695 | 0.0650 | 0.1707 | 6.8102e+00 |
| 10 | `sparse_random_n3_d1d25_mind3_s0` | sparse_random | 3 | -1.0339 | 1.778557e-06 | 570903.7904 | 0.2144 | 0.3562 | 0.0860 | 0.1646 | 2.1668e+00 |

## Artifacts

- Corrected CSV: `results/mask_design_corrected/candidate_screening_corrected.csv`
- Pareto JSON: `results/mask_design_corrected/pareto_candidates.json`
- Full top-candidate diagnostics: `results/mask_design_corrected/top_candidates.json`