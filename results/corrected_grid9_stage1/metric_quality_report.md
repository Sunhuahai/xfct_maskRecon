# Metric Quality Report

## Rule Implementation

DL/CNR quality control is implemented in `src/reporting_roi.py` by
`evaluate_detection_limit_quality()` and is saved by `src/reporting_reconstruction.py`
into each `reconstruction_results.npz`. The comparison driver
`experiments/run_effect_comparison.py` writes the same flags to
`effect_comparison.csv`.

A CNR-derived DL is marked invalid when any of these checks fail:

- DL, CNR slope/intercept, R2, ROI means/stds, or CNR values are NaN or inf.
- CNR fit slope is nonpositive.
- DL is negative.
- CNR linear fit has `R2 < 0.80`.
- Fitted CNR response over nonzero concentration ROIs is non-monotonic.
- Background ROI estimate is unstable: non-finite mean/std or std `<= 1.0e-8`.

Invalid DL values are retained as raw diagnostic fit outputs but are not interpreted
as meaningful detection limits.

## Stage 1 Results

| run | DL flag | raw DL mg/ml | R2 | slope | monotonic CNR | background mean | background std | reason |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `traditional_5` | valid | 0.262452 | 0.959957 | 69.733917 | True | 0.023105 | 0.015449 | valid |
| `traditional_15` | valid | 0.063257 | 0.994301 | 230.509542 | True | 0.010068 | 0.004485 | valid |
| `traditional_45` | valid | 0.003945 | 0.999305 | 508.188697 | True | 0.004534 | 0.001979 | valid |
| `mask_5_model` | invalid | 19.644854 | 0.463720 | 0.235600 | False | 2.208054 | 2.786540 | poor CNR linear fit R2<0.80; non-monotonic CNR concentration response |

## Files

- CSV with quality flags: `results/corrected_grid9_stage1/effect_comparison.csv`
- Markdown comparison: `results/corrected_grid9_stage1/effect_comparison.md`
- Per-run result archives: `results/corrected_grid9_stage1/*_em_tv/reconstruction_results.npz`
