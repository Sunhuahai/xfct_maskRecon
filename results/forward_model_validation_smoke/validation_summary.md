# Forward Model Validation Summary

Overall status: **FAIL**

| test | status | fail-critical | key result |
| --- | --- | ---: | --- |
| detector_padding | SKIPPED | True | virtual fraction=nan; virtual sum=nan |
| single_center_pinhole_regression | SKIPPED | False | Explicit single-pinhole matrix was not loaded. |
| delta_voxel_tests | FAIL | True | max virtual fraction=1.0000e+00; max centroid shift=19.059 px |
| multi_hole_linearity | PASS | True | rel L2=4.697e-16 |
| adjoint_tests | PASS | True | matrix-free rel=2.260e-15; explicit status=SKIPPED |
| known_phantom_residual | PASS | False | self dev=0.000e+00; physical-vs-A dev=2.892e+00 |

## Stop Condition

Stop before expensive mask sweeps. Fail-critical tests failed: delta_voxel_tests. Recommended first fix: make the projection generator and system matrix use the same detector support (either clip all matrix rows to the physical 80-column detector before padding, or regenerate projections for a true 160-column detector).

Diagnostic PNGs are saved in this directory.