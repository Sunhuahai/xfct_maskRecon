# Forward Model Validation Summary

Overall status: **PASS**

| test | status | fail-critical | key result |
| --- | --- | ---: | --- |
| detector_padding | PASS | True | virtual fraction=0.0000e+00; virtual sum=0.0000e+00 |
| single_center_pinhole_regression | WARN | False | One-hole matrix-free model compared with existing single-pinhole matrix. |
| delta_voxel_tests | PASS | True | max virtual fraction=0.0000e+00; max centroid shift=0.000 px |
| multi_hole_linearity | PASS | True | rel L2=5.072e-16 |
| adjoint_tests | PASS | True | matrix-free rel=2.996e-16; explicit status=PASS |
| known_phantom_residual | PASS | False | self dev=0.000e+00; physical-vs-A dev=0.000e+00 |

## Stop Condition

No fail-critical mismatch was found; mask screening can proceed.

Diagnostic PNGs are saved in this directory.