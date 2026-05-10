# XFCT Mask Reconstruction Current State Handoff

Last updated: 2026-05-10, Asia/Shanghai.

This document is self-contained by design. It is written for a fresh GPT Pro instance or a new developer who may not inspect files immediately. Paths are included for provenance, but the key facts and numeric results are embedded here.

This document only states the current project situation. It does not prescribe follow-up experiments or decide the next research plan.

## One-Sentence State

The repository now contains a next-stage XFCT mask reconstruction framework centered on forward-model validation, sparse mask candidate generation/screening, and raw-domain Poisson reconstruction; the strongest current finding is that the original 5-view grid9 multi-hole system matrix was physically inconsistent because it wrote signal into virtual 80x160 padded detector columns, while the actual data are clipped to a real 80x80 detector and only then padded.

## Scientific Target

The project goal is not simply to show that multi-pinhole XFCT increases photon throughput.

The intended scientific framing is:

> Task-driven depth-coded sparse multi-pinhole XFCT: jointly design a sparse detector-side multi-pinhole mask and few-view angle set under the true depth-dependent XFCT forward model, reconstruct directly from raw Poisson mask-domain data, and compare against single-pinhole 5/15/45-view baselines under equal time, equal incident dose, and equal detected-count protocols.

The current work should not be interpreted as:

- a conventional fixed 3x3 multi-hole throughput experiment,
- a URA/MURA decoding experiment,
- a fixed-shift deconvolution experiment,
- or a visual-quality-only comparison.

The relevant quantities in this project are detection limit, ROI bias, CNR linearity, projection fit, residual structure, FOV truncation, hole overlap, conditioning/task Fisher metrics, sensitivity uniformity, and pose/geometry robustness.

## Geometry and Data Convention

Known geometry currently used by baseline scripts:

| item | value |
| --- | --- |
| physical detector | 80 x 80 pixels |
| detector pixel size | 0.25 mm |
| detector plane distance behind mask | 30 mm |
| rotation center to mask distance | 50 mm |
| source to rotation center distance | 300 mm |
| reconstruction grid | 40 x 60 x 60 voxels |
| voxel size | 0.5 mm |
| default 45-view scan | 8 degree steps |
| default 5-view indices | 0, 9, 18, 27, 36 |
| default 5-view angles | 0, 72, 144, 216, 288 deg |
| default 15-view indices | 0, 3, 6, ..., 42 |
| default 15-view angular step | 24 deg |

Important detector convention:

- Stored projection data are physical `80 x 80` arrays.
- Reconstruction pads detector x by 40 columns on each side.
- Padded detector shape used by existing system matrices is `80 x 160`.
- Physical detector x support inside padded coordinates is columns `40..119`.
- Virtual padded x columns are `0..39` and `120..159`.

Current grid mask baseline:

| item | value |
| --- | --- |
| layout | 3 x 3 grid |
| holes | 9 |
| pitch | 6 mm |
| hole diameter | 1.25 mm |
| projection file | `data/projections/mask/geometry_45_proj_cmask9_grid_p6_d1d25.npy` |
| original 5-view matrix | `data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz` |
| corrected clipped 5-view matrix | `data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz` |

## Repository Layout

Top-level directory roles:

| path | role |
| --- | --- |
| `projection/` | projection-generation code from the XFCT workflow |
| `scripts/` | system-matrix generation, candidate generation, projection utilities, matrix clipping |
| `algorithm/` | existing reconstruction algorithms, especially EM-TV |
| `recon/` | newly added raw-domain Poisson-TV reconstruction code |
| `src/` | reporting utilities, ROI metrics, mask decoding, matrix-free mask model |
| `data/` | projection data, system matrices, mask candidates, attenuation/physics/phantom files |
| `experiments/` | runnable experiment entrypoints |
| `results/` | generated outputs and result summaries |
| `docs/` | documentation and handoff notes |

The workspace at `/Users/huahai/PARA/Project/XFCT/xfct_maskRecon` is not currently a Git repository.

## Important Code Files

### Baseline Reconstruction and Metrics

| file | current role |
| --- | --- |
| `experiments/run_effect_comparison.py` | existing comparison driver for traditional and mask runs |
| `algorithm/em_tv.py` | current EM-TV implementation |
| `algorithm/recon_common.py` | projection loading, background handling, detector padding, CSR matrix loading |
| `src/reporting_roi.py` | ROI/CNR/detection-limit metrics; nominal ROI concentrations are `[0, 0.5, 1.0, 1.5, 2.0, 3.0] mg/ml` |
| `src/reporting_reconstruction.py` | reconstruction wrappers and figure/ROI result saving |
| `src/mask_decode.py` | fixed-shift/Wiener decoding diagnostic baseline |

### Projection and System Matrix Code

| file | current role |
| --- | --- |
| `projection/fluorescence.py` | single-pinhole projection generator from GEANT4 event data |
| `projection/fluorescence_cmask.py` | multi-hole coded-mask projection generator from GEANT4 event data |
| `projection/mask_geometry.py` | mask layout utilities for single, grid, cross, random, and custom masks |
| `scripts/build_mask_system_matrix.py` | explicit sparse multi-hole system-matrix builder |
| `scripts/clip_system_matrix_to_physical_detector.py` | utility that zeroes rows outside physical 80-column detector support in an existing padded CSR matrix |

`scripts/build_mask_system_matrix.py` has been updated with these detector-support options:

| option | purpose |
| --- | --- |
| `--clip-to-physical-detector` | clip samples to the real physical detector before writing padded rows |
| `--physical-detector-x` | physical detector x width, currently 80 |
| `--pad-x` | x-padding amount, currently 40 |

### Newly Added Next-Stage Framework Code

| file | current role |
| --- | --- |
| `src/mask_xfct_model.py` | matrix-free depth-dependent multi-pinhole forward model and helper phantoms/metrics |
| `experiments/validate_mask_forward_consistency.py` | forward-model validation suite |
| `scripts/generate_mask_candidates.py` | sparse mask candidate generator |
| `experiments/screen_mask_candidates.py` | task-based candidate screening |
| `recon/poisson_tv_pdhg.py` | raw-domain Poisson-TV PDHG solver |
| `experiments/run_poisson_mbir_mask_recon.py` | raw Poisson MBIR experiment driver |
| `experiments/run_mask_protocol_comparison.py` | equal time/dose/count protocol comparison driver |
| `experiments/run_mask_pose_sensitivity.py` | mask pose and geometry perturbation study |
| `experiments/make_next_experiments_report.py` | report assembly script for `results/next_experiments_report.md` |

`src/mask_xfct_model.py` supports two detector modes:

| mode | meaning |
| --- | --- |
| `support_mode="padded"` | artificial full `80 x 160` detector support |
| `support_mode="physical_padded"` | true physical `80 x 80` detector embedded in padded columns `40..119` |

## Data Distribution

### Projection Data

Single-pinhole simulation projections:

| file | meaning |
| --- | --- |
| `data/projections/simulation/PMMA_3d0_5.npy` | 5-view single-pinhole simulation projection |
| `data/projections/simulation/PMMA_3d0_15.npy` | 15-view single-pinhole simulation projection |
| `data/projections/simulation/PMMA_3d0_45.npy` | 45-view single-pinhole simulation projection |

Generated/copied single-pinhole geometry projections:

| file | meaning |
| --- | --- |
| `data/projections/single/geometry_5_proj.npy` | 5-view geometry projection |
| `data/projections/single/geometry_15_proj.npy` | 15-view geometry projection |
| `data/projections/single/geometry_45_proj.npy` | 45-view geometry projection |

Mask projections:

| file | meaning |
| --- | --- |
| `data/projections/mask/geometry_5_proj_cmask9_grid_p6_d1d25.npy` | 5-view grid9 mask projection |
| `data/projections/mask/geometry_15_proj_cmask9_grid_p6_d1d25.npy` | 15-view grid9 mask projection |
| `data/projections/mask/geometry_45_proj_cmask9_grid_p6_d1d25.npy` | 45-view grid9 mask projection |

All projection arrays above are physical detector data with shape `(angle, 80, 80)`. They are not native `80 x 160` detector measurements.

### System Matrices

Single-pinhole matrices:

| file | views |
| --- | ---: |
| `data/system_matrix/cij_5_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` | 5 |
| `data/system_matrix/cij_15_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` | 15 |
| `data/system_matrix/cij_45_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz` | 45 |

Mask matrices:

| file | status |
| --- | --- |
| `data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz` | original grid9 matrix; failed detector-support validation |
| `data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz` | clipped grid9 matrix; passes detector-support validation |

Corrected clipped matrix dimensions:

| item | value |
| --- | --- |
| matrix shape | `(64000, 144000)` |
| rows | `5 angles x 80 z x 160 padded x` |
| columns | `40 x 60 x 60 voxels` |
| physical padded columns | `40..119` |
| virtual padded columns | `0..39` and `120..159` |
| nonzeros after clipping | `184,933,778` |

### Mask Candidate Data

Candidate files:

| item | value |
| --- | --- |
| JSON folder | `data/masks/candidates/*.json` |
| current JSON file count | 90 |
| manifest | `results/mask_design/candidate_manifest.csv` |
| manifest line count | 91 including header |
| screening CSV | `results/mask_design/candidate_screening.csv` |
| screening line count | 101 including header |

Generated families include:

- `single_center`
- `grid3x3`
- `sparse_random`
- `blue_noise`
- `ring`
- `ring_two`
- `cross_plus_center`
- `ura_mura_inspired` as diagnostic baseline only

### Physics and Phantom Data

| file | role |
| --- | --- |
| `data/attenuation_map/PMMA.csv` | attenuation map |
| `data/phantom/pmma_gd_hex_80_120_120.txt` | phantom |
| `data/projection_physics/spec_150kVp.mat` | source spectrum data |
| `data/projection_physics/miu.npz` | attenuation/physics coefficients |
| `data/projection_physics/miu3.mat` | attenuation/physics coefficients |

## Forward-Model Validation Results

### Original Unclipped Grid9 Matrix

Validation files:

- `results/forward_model_validation/validation_summary.json`
- `results/forward_model_validation/validation_summary.md`
- `results/forward_model_validation/detector_support.png`

Overall result:

| test | status |
| --- | --- |
| overall | FAIL |
| detector_padding | FAIL |
| delta_voxel_tests | FAIL |
| multi_hole_linearity | PASS |
| adjoint_tests | PASS |
| known_phantom_residual | PASS |
| single_center_pinhole_regression | WARN |

Detector-support numbers for the original matrix:

| quantity | value |
| --- | ---: |
| total row sum | 90.8153270668061 |
| physical row sum | 61.90858427187911 |
| virtual row sum | 28.906742794927 |
| virtual fraction | 0.3183024686313407 |
| virtual max row sum | 0.0027938083461124183 |
| physical columns | 40..119 |
| padded detector shape | 80 x 160 |
| physical detector shape | 80 x 80 |

Delta-test failure numbers:

| quantity | value |
| --- | ---: |
| max virtual fraction | 1.0 |
| max centroid delta | 19.059090639687234 px |
| max relative L2 after scalar scaling | 0.9693031579714333 |

Interpretation as current status:

The original explicit grid9 mask matrix writes nonzero signal into virtual padded detector pixels. The projection pipeline produces real `80 x 80` detector data and pads afterward. Therefore the original `mask_5_model` full comparison used a physically inconsistent mask system matrix.

### Corrected Physical-Support Clipped Grid9 Matrix

Clipping summary file:

- `results/forward_model_validation_clipped/clip_summary.json`

Validation files:

- `results/forward_model_validation_clipped/validation_summary.json`
- `results/forward_model_validation_clipped/validation_summary.md`
- `results/forward_model_validation_clipped/detector_support.png`
- `results/forward_model_validation_clipped/known_phantom_physical_vs_padded_residual.png`
- `results/forward_model_validation_clipped/known_phantom_self_residual.png`
- `results/forward_model_validation_clipped/multi_hole_linearity_residual.png`

Clipping result:

| quantity | before clipping | after clipping |
| --- | ---: | ---: |
| total row sum | 90.8153270668061 | 61.90858427187911 |
| virtual row sum | 28.906742794927 | 0.0 |
| virtual fraction | 0.3183024686313407 | 0.0 |

Corrected matrix metadata:

| quantity | value |
| --- | --- |
| shape | `[64000, 144000]` |
| nnz after | `184933778` |
| detector z | `80` |
| padded detector x | `160` |
| physical detector x | `80` |
| pad x | `40` |
| physical column range inclusive | `[40, 119]` |

Validation result after clipping:

| test | status | key value |
| --- | --- | --- |
| overall | PASS | physical-support validation passed |
| detector_padding | PASS | virtual fraction `0.0` |
| delta_voxel_tests | PASS | max virtual fraction `0.0` |
| multi_hole_linearity | PASS | output plot saved |
| adjoint_tests | PASS | explicit/matrix-free adjoint checks available in JSON |
| known_phantom_residual | PASS | residual maps saved |
| single_center_pinhole_regression | WARN | non-fail-critical |

Single-center regression warning numbers:

| quantity | value |
| --- | ---: |
| scalar fit | 0.20400000000000001 |
| relative L2 after scaling | 0.9080420895865016 |
| explicit centroid x | 80.214 px |
| explicit centroid z | 38.966 px |
| matrix-free centroid x | 79.875 px |
| matrix-free centroid z | 38.75 px |

Interpretation as current status:

The detector-support mismatch is fixed for the clipped grid9 matrix. The one-hole single-center regression still has a warning, but the fail-critical detector padding, delta footprint, linearity, adjoint, and known-phantom checks pass for the corrected matrix.

## Reconstruction Methods Currently Present

### EM-TV Baseline

Code:

- `algorithm/em_tv.py`
- called by `experiments/run_effect_comparison.py`

The update structure is EM-like:

```text
f <- f * A^T (y / (A f)) / (A^T 1)
f <- max(f, 0)
```

A TV smoothing/penalty step is applied after warmup.

Current real projection comparison status:

| run type | matrix | reconstruction |
| --- | --- | --- |
| traditional_5/15/45 | single-pinhole CSR matrices | EM-TV |
| original mask_5_model | original grid9 mask CSR matrix | EM-TV |
| clipped quick mask_5_model | clipped grid9 mask CSR matrix | EM-TV |

### Raw-Domain Poisson-TV PDHG

Code:

- `recon/poisson_tv_pdhg.py`
- called by `experiments/run_poisson_mbir_mask_recon.py`

Objective:

```text
min over f >= 0:
    sum_i [lambda_i(f) - y_i log(lambda_i(f) + eps)]
    + beta TV(f)

lambda(f) = A f + b
```

Current status:

- Code runs in quick synthetic/matrix-free mode.
- Current quick output is not a final real-projection result.
- Current quick detection-limit fits are invalid/nonsensical.

## Existing Full Baseline Result With Original Unclipped Matrix

Output files:

- `results/effect_comparison/effect_comparison.csv`
- `results/effect_comparison/reconstruction_panel.png`

Important full baseline rows:

| run | method | projection sum | DL mg/ml | R2 | final nll | final rel | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| traditional_5 | EM-TV | 219811.1792 | 0.2624586488 | 0.9599576732 | -419798.9081 | 0.0307465079 | valid single-pinhole baseline |
| mask_5_naive | EM-TV with single matrix | 10913099.5602 | 10.6846167120 | 0.4798225165 | -28755527.6305 | 0.0103649015 | invalid physical model |
| mask_5_wiener | Wiener decode + EM-TV | 12522828.7700 | -29.9987300983 | 0.5131226123 | -34603710.3927 | 0.0137218053 | diagnostic only |
| mask_5_model | EM-TV with original grid9 mask matrix | 10913099.5602 | 9.7766556813 | 0.4529584598 | -51117115.8240 | 0.0169795336 | affected by detector-support mismatch |
| traditional_15 | EM-TV | 657307.5738 | 0.0632584145 | 0.9943013350 | -1253522.4406 | 0.0112109133 | valid single-pinhole baseline |
| traditional_45 | EM-TV | 1968367.7353 | 0.0039451179 | 0.9993047237 | -3755498.0162 | 0.0073731242 | valid single-pinhole baseline |

Current interpretation:

The original full `mask_5_model` result is not a physically reliable mask reconstruction conclusion because it used the original matrix that wrote signal into virtual padded detector pixels.

## Corrected Clipped Quick EM-TV Smoke Result

Output files:

- `results/effect_comparison_clipped_quick/effect_comparison.csv`
- `results/effect_comparison_clipped_quick/effect_comparison.md`
- `results/effect_comparison_clipped_quick/reconstruction_panel.png`
- `results/effect_comparison_clipped_quick/traditional_5_em_tv/reconstruction.png`
- `results/effect_comparison_clipped_quick/traditional_5_em_tv/reconstruction_roi_dl.png`
- `results/effect_comparison_clipped_quick/mask_5_model_em_tv/reconstruction.png`
- `results/effect_comparison_clipped_quick/mask_5_model_em_tv/reconstruction_roi_dl.png`

Rows:

| run | method | projection sum | DL mg/ml | R2 | final nll | final rel | matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| traditional_5 | EM-TV quick | 219811.1792 | 0.5495999563 | 0.9470275783 | -395025.4724 | 0.2355821483 | `cij_5_3d_mod30_p1...npz` |
| mask_5_model | EM-TV quick | 10913099.5602 | 8.4868506378 | 0.7999205363 | -54012068.4646 | 0.1586554848 | clipped grid9 matrix |

Current status of this result:

- This run used the corrected clipped grid9 matrix.
- It was run in `--quick` mode.
- It is a smoke test, not a final comparison.
- It shows that the corrected matrix is runnable in the existing EM-TV pipeline.
- In this quick run, grid9 mask reconstruction remains much worse than the single-pinhole 5-view quick baseline.

## Candidate Mask Screening Current Data

Output files:

- `results/mask_design/candidate_manifest.csv`
- `results/mask_design/candidate_screening.csv`
- `results/mask_design/top_candidates.json`

Important status notes:

- Current candidate generation created 90 JSON candidates.
- Current screening CSV has 100 result rows plus header.
- Screening is matrix-free and uses physical detector support.
- The `top_candidates.json` currently includes a warning: `"Forward validation did not pass; screening rankings are provisional and should not drive final mask selection."`
- That warning reflects the state when screening was generated; since then the clipped explicit grid9 matrix has passed physical-support validation.
- The screening results remain quick/provisional and are not final scientific mask-selection results.

Top candidate recorded in `top_candidates.json`:

| quantity | value |
| --- | --- |
| candidate id | `blue_noise_n3_d1d25_mind3_s1` |
| family | `blue_noise` |
| holes | 3 |
| hole diameter | 1.25 mm |
| minimum distance | 3.0 mm |
| total open area | 3.6815538909255388 mm2 |
| angle set | `phase0_default` |
| angle indices | `0,9,18,27,36` |
| comments | `blue-noise sparse 3-hole candidate` |
| ranking score | -1.1913896383491724 |
| sensitivity mean | 0.0003026233215472655 |
| sensitivity CV | 0.13446078583813986 |
| sensitivity min/mean | 0.6082094637777654 |
| ROI sensitivity mean | 0.0003568663509121364 |
| ROI sensitivity CV | 0.06452453157164643 |
| global physical truncation mean | 0.4889166666666667 |
| global physical truncation p95 | 0.6 |
| global padded truncation mean | 0.2986666666666667 |
| ROI physical truncation mean | 0.3824444444444444 |
| ROI physical truncation p95 | 0.4 |
| ROI padded truncation mean | 0.086 |
| overlap mean | 0.023287153411039593 |
| overlap max | 0.03835679183306404 |
| overlap high fraction | 0.0 |
| global coherence max | 0.08219219744205475 |
| ROI coherence max | 0.44975513219833374 |
| task Fisher d2 mean | 8.734187514615749e-05 |
| task Fisher d2 min | 6.999879326179085e-05 |
| task CRLB mean | 11670.271547316763 |
| task CRLB max | 14285.960563063798 |

Other high-ranked rows visible in current screening output:

| candidate id | family | holes | diameter mm | min distance mm | angle set | ranking score | task Fisher d2 mean | ROI physical truncation mean | overlap mean |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `blue_noise_n3_d1d25_mind3_s1` | blue_noise | 3 | 1.25 | 3.0 | phase0_default | -1.1913896383 | 8.7341875e-05 | 0.3824444444 | 0.0232871534 |
| `blue_noise_n3_d0d75_mind3_s1` | blue_noise | 3 | 0.75 | 3.0 | phase0_default | -1.3047244881 | 2.5524560e-05 | 0.3804444444 | 0.0232871534 |
| `blue_noise_n3_d1d25_mind3_s0` | blue_noise | 3 | 1.25 | 3.0 | phase0_default | value present in JSON after the shown excerpt | 7.2367018e-05 | 0.5382222222 | 0.0180260080 |

Current interpretation:

The quick screening suggests sparse blue-noise-like 3-hole candidates can reduce overlap relative to grid-like layouts, but the truncation remains substantial and the current screening output is provisional.

## Raw Poisson MBIR Quick Result

Output file:

- `results/poisson_mbir_mask_recon/poisson_mbir_summary.csv`

Rows:

| candidate | family | holes | diameter mm | raw counts | expected counts | DL mg/ml | R2 | final deviance | residual structure | runtime s | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `grid9_p6_d1p25_5` | grid3x3 | 9 | 1.25 | 200408 | 200000.064 | -27.8235984461 | 0.1861371385 | 218931300.1499 | 0.9872582958 | 2.4164 | invalid DL / smoke test |
| `blue_noise_n3_d1d25_mind3_s1` | blue_noise | 3 | 1.25 | 200174 | 200000.064 | -54.0630499124 | 0.0120861832 | 7599868.1117 | 0.2276742008 | 0.8135 | invalid DL / smoke test |

Output image paths:

| candidate | reconstruction | residual map |
| --- | --- | --- |
| grid9 | `results/poisson_mbir_mask_recon/grid9_p6_d1p25_5/seed_20260509/reconstruction.npy` | `results/poisson_mbir_mask_recon/grid9_p6_d1p25_5/seed_20260509/residual_map.png` |
| blue_noise | `results/poisson_mbir_mask_recon/blue_noise_n3_d1d25_mind3_s1/seed_20260509/reconstruction.npy` | `results/poisson_mbir_mask_recon/blue_noise_n3_d1d25_mind3_s1/seed_20260509/residual_map.png` |

Current interpretation:

The raw-domain Poisson-TV code executes, but the current quick MBIR rows do not provide meaningful detection-limit conclusions.

## Protocol Comparison Quick Result

Output files:

- `results/protocol_comparison/protocol_summary.csv`
- `results/protocol_comparison/protocol_summary.md`

The current protocol CSV has 63 result rows plus header.

Current protocol run set:

- `traditional_5`
- `traditional_15`
- `traditional_45`
- `grid9_p6_d1p25_5`
- `best_5hole_sparse_blue_noise_n5_d1d25_mind6_s0`
- `best_7hole_sparse_blue_noise_n7_d0d75_mind3_s0`
- `best_ring_ring_n3_d0d75_mind3`

The table below aggregates current quick rows by protocol and run. `valid_dl` is the number of seeds whose `detection_limit_invalid` flag is false.

| protocol | run | seeds | valid DL seeds | mean DL mg/ml | mean R2 | mean counts | truncation | overlap | residual structure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_acquisition_time | traditional_5 | 3 | 0 | -7.11669 | 0.851627 | 199812 | 0.0368 | 0 | 0.281801 |
| equal_acquisition_time | traditional_15 | 3 | 0 | -3.98808 | 0.777324 | 600032 | 0.0357333 | 0 | 0.913919 |
| equal_acquisition_time | traditional_45 | 3 | 1 | 23.9792 | 0.248039 | 1798370 | 0.0346667 | 0 | 0.918812 |
| equal_acquisition_time | grid9_p6_d1p25_5 | 3 | 3 | 26.9359 | 0.261627 | 1374290 | 0.534756 | 0.392314 | 0.990452 |
| equal_acquisition_time | best_5hole_sparse_blue_noise_n5_d1d25_mind6_s0 | 3 | 0 | -2.85400 | 0.0955182 | 639377 | 0.58432 | 0.0944352 | 0.917723 |
| equal_acquisition_time | best_7hole_sparse_blue_noise_n7_d0d75_mind3_s0 | 3 | 0 | -16.1193 | 0.456742 | 328844 | 0.567429 | 0.373196 | 0.980839 |
| equal_acquisition_time | best_ring_ring_n3_d0d75_mind3 | 3 | 0 | -7.41319 | 0.76329 | 255702 | 0.260133 | 0.406906 | 0.898159 |
| equal_incident_dose | traditional_5 | 3 | 0 | -7.11669 | 0.851627 | 199812 | 0.0368 | 0 | 0.281801 |
| equal_incident_dose | traditional_15 | 3 | 0 | -3.98808 | 0.777324 | 600032 | 0.0357333 | 0 | 0.913919 |
| equal_incident_dose | traditional_45 | 3 | 1 | 23.9792 | 0.248039 | 1798370 | 0.0346667 | 0 | 0.918812 |
| equal_incident_dose | grid9_p6_d1p25_5 | 3 | 3 | 26.9359 | 0.261627 | 1374290 | 0.534756 | 0.392314 | 0.990452 |
| equal_incident_dose | best_5hole_sparse_blue_noise_n5_d1d25_mind6_s0 | 3 | 0 | -2.85400 | 0.0955182 | 639377 | 0.58432 | 0.0944352 | 0.917723 |
| equal_incident_dose | best_7hole_sparse_blue_noise_n7_d0d75_mind3_s0 | 3 | 0 | -16.1193 | 0.456742 | 328844 | 0.567429 | 0.373196 | 0.980839 |
| equal_incident_dose | best_ring_ring_n3_d0d75_mind3 | 3 | 0 | -7.41319 | 0.76329 | 255702 | 0.260133 | 0.406906 | 0.898159 |
| equal_detected_counts | traditional_5 | 3 | 0 | -7.11669 | 0.851627 | 199812 | 0.0368 | 0 | 0.281801 |
| equal_detected_counts | traditional_15 | 3 | 3 | 4.41633 | 0.658001 | 200038 | 0.0357333 | 0 | 0.256476 |
| equal_detected_counts | traditional_45 | 3 | 3 | 4.45895 | 0.59779 | 199613 | 0.0346667 | 0 | 0.904631 |
| equal_detected_counts | grid9_p6_d1p25_5 | 3 | 0 | -27.2002 | 0.196313 | 200108 | 0.534756 | 0.392314 | 0.987937 |
| equal_detected_counts | best_5hole_sparse_blue_noise_n5_d1d25_mind6_s0 | 3 | 0 | 53.0874 | 0.0204973 | 199843 | 0.58432 | 0.0944352 | 0.934183 |
| equal_detected_counts | best_7hole_sparse_blue_noise_n7_d0d75_mind3_s0 | 3 | 0 | -14.4696 | 0.450374 | 199709 | 0.567429 | 0.373196 | 0.984359 |
| equal_detected_counts | best_ring_ring_n3_d0d75_mind3 | 3 | 0 | -134.885 | 0.00365897 | 200040 | 0.260133 | 0.406906 | 0.311825 |

Current interpretation:

This protocol result is a quick synthetic/matrix-free smoke test. Many DL fits are invalid. It confirms that the protocol-comparison code can produce all three protocols, but it does not establish a final equal-time/equal-dose/equal-count scientific conclusion.

## Pose Sensitivity Quick Result

Output files:

- `results/mask_pose_sensitivity/mask_pose_sensitivity.csv`
- `results/mask_pose_sensitivity/mask_pose_sensitivity.md`

Summary currently reported:

| candidate | mean abs DL perturbation | max abs DL perturbation |
| --- | ---: | ---: |
| `grid9_p6_d1p25_5` | 0.4876 mg/ml | 1.4816 mg/ml |
| `blue_noise_n3_d1d25_mind3_s1` | 0.8572 mg/ml | 3.5787 mg/ml |

Current interpretation:

These numbers are from quick synthetic perturbation tests. They are framework smoke-test outputs, not final robustness conclusions.

## Result Image Locations

Forward validation images:

| image | path |
| --- | --- |
| original detector support | `results/forward_model_validation/detector_support.png` |
| clipped detector support | `results/forward_model_validation_clipped/detector_support.png` |
| clipped multi-hole linearity residual | `results/forward_model_validation_clipped/multi_hole_linearity_residual.png` |
| clipped known phantom residual | `results/forward_model_validation_clipped/known_phantom_physical_vs_padded_residual.png` |
| clipped single-center residual | `results/forward_model_validation_clipped/single_center_residual.png` |

Original comparison images:

| image | path |
| --- | --- |
| full original panel | `results/effect_comparison/reconstruction_panel.png` |
| original mask model reconstruction | `results/effect_comparison/mask_5_model_em_tv/reconstruction.png` |
| original mask model ROI/DL | `results/effect_comparison/mask_5_model_em_tv/reconstruction_roi_dl.png` |

Corrected clipped quick images:

| image | path |
| --- | --- |
| clipped quick panel | `results/effect_comparison_clipped_quick/reconstruction_panel.png` |
| clipped mask reconstruction | `results/effect_comparison_clipped_quick/mask_5_model_em_tv/reconstruction.png` |
| clipped mask ROI/DL | `results/effect_comparison_clipped_quick/mask_5_model_em_tv/reconstruction_roi_dl.png` |
| clipped traditional 5 reconstruction | `results/effect_comparison_clipped_quick/traditional_5_em_tv/reconstruction.png` |
| clipped traditional 5 ROI/DL | `results/effect_comparison_clipped_quick/traditional_5_em_tv/reconstruction_roi_dl.png` |

Poisson MBIR quick images:

| image | path |
| --- | --- |
| grid9 residual | `results/poisson_mbir_mask_recon/grid9_p6_d1p25_5/seed_20260509/residual_map.png` |
| blue-noise residual | `results/poisson_mbir_mask_recon/blue_noise_n3_d1d25_mind3_s1/seed_20260509/residual_map.png` |

## Current Facts and Non-Conclusions

Current strong facts:

- The actual projection data are physical `80 x 80` detector arrays.
- The existing reconstruction pipeline pads those data into `80 x 160`.
- The original grid9 mask system matrix had nonzero support in virtual padded detector columns.
- The original matrix had virtual row-sum fraction `0.3183024686313407`.
- The clipped matrix has virtual row-sum fraction `0.0`.
- The clipped matrix passes fail-critical detector padding, delta voxel, multi-hole linearity, adjoint, and known-phantom validation tests.
- The original full `mask_5_model` DL of about `9.7767 mg/ml` is affected by the detector-support mismatch.
- The clipped quick `mask_5_model` DL is about `8.4869 mg/ml`, but this is only a 3-iteration quick EM-TV smoke test.
- The current quick raw Poisson MBIR and protocol comparison outputs contain many invalid/nonsensical DL fits.

Current non-conclusions:

- The current data do not prove that multi-hole XFCT is worse in general.
- The current data do not prove that sparse/blue-noise/ring masks win.
- The current data do not prove that the grid9 failure is only a throughput issue.
- The current data do not prove a coding-information advantage under equal detected counts.
- The current data do not validate Wiener/fixed-shift decoding as a main reconstruction route.

## Current Missing Final Results

These are absent from the current repository state:

- A non-quick full EM-TV comparison using the clipped grid9 matrix.
- A final corrected candidate-screening run after the clipped explicit matrix validation.
- A final raw-domain Poisson MBIR reconstruction on real/properly matched projection data.
- Large explicit clipped system matrices for new sparse/blue-noise/ring candidates.
- A final multi-seed protocol comparison with reliable DL fits.
- A final robustness conclusion based on stable reconstruction outputs.

## Provenance: Commands Already Run or Represented by Existing Outputs

Original matrix validation represented by `results/forward_model_validation/`:

```bash
conda run -n xfct python experiments/validate_mask_forward_consistency.py \
  --quick \
  --support-mode physical_padded \
  --explicit-mask-matrix data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz \
  --output-root results/forward_model_validation
```

Existing matrix clipping represented by `results/forward_model_validation_clipped/clip_summary.json`:

```bash
conda run -n xfct python scripts/clip_system_matrix_to_physical_detector.py
```

Clipped matrix validation represented by `results/forward_model_validation_clipped/`:

```bash
conda run -n xfct python experiments/validate_mask_forward_consistency.py \
  --quick \
  --support-mode physical_padded \
  --explicit-mask-matrix data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz \
  --output-root results/forward_model_validation_clipped
```

Clipped quick EM-TV smoke represented by `results/effect_comparison_clipped_quick/`:

```bash
conda run -n xfct python experiments/run_effect_comparison.py \
  --quick \
  --runs traditional_5,mask_5_model \
  --mask-system-matrix data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz \
  --output-root results/effect_comparison_clipped_quick
```

Candidate generation represented by `data/masks/candidates/` and `results/mask_design/candidate_manifest.csv`:

```bash
conda run -n xfct python scripts/generate_mask_candidates.py --quick
```

Candidate screening represented by `results/mask_design/candidate_screening.csv`:

```bash
conda run -n xfct python experiments/screen_mask_candidates.py \
  --quick \
  --candidate-limit 20
```

Raw Poisson MBIR quick output represented by `results/poisson_mbir_mask_recon/`:

```bash
conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --quick
```

Protocol quick output represented by `results/protocol_comparison/`:

```bash
conda run -n xfct python experiments/run_mask_protocol_comparison.py \
  --quick \
  --num-seeds 3
```

Report assembly represented by `results/next_experiments_report.md`:

```bash
conda run -n xfct python experiments/make_next_experiments_report.py
```

## Compact Status Summary

The present repository state is best summarized as follows: the original mask reconstruction result was confounded by a real detector-support bug in the explicit grid9 mask matrix; a corrected physical 80x80-clipped, 80x160-padded matrix now exists and passes fail-critical forward validation; the current corrected grid9 quick EM-TV reconstruction still looks poor; candidate generation, screening, raw Poisson MBIR, protocol comparison, and pose sensitivity frameworks exist, but their current non-validation outputs are quick smoke tests rather than final scientific results.
