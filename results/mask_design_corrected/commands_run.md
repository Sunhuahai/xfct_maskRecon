# Commands Run

Commands were run from `/home/venti/PARA/Project/XFCT/xfct_maskRecon`.

## Gate And Context

```bash
sed -n '1,280p' goal/stage2.md
find results/mask_design_corrected -maxdepth 3 -type f | sort 2>/dev/null || true
sed -n '1,360p' src/mask_xfct_model.py
sed -n '320,620p' src/mask_xfct_model.py
sed -n '1,360p' experiments/screen_mask_candidates.py
sed -n '360,760p' experiments/screen_mask_candidates.py
sed -n '1,320p' scripts/generate_mask_candidates.py
sed -n '320,620p' scripts/generate_mask_candidates.py
sed -n '1,80p' results/mask_design/candidate_manifest.csv
sed -n '1,80p' results/mask_design/candidate_screening.csv
sed -n '1,220p' results/mask_design/top_candidates.json
sed -n '1,180p' results/corrected_grid9_stage1/corrected_grid9_report.md
ls data/masks/candidates/single_center*.json data/masks/candidates/grid3x3_n9_d1d25_mind6.json data/masks/candidates/blue_noise_n3_d1d25_mind3_s1.json
```

## Code Verification

```bash
conda run -n xfct python -m py_compile experiments/screen_mask_candidates.py
git diff -- experiments/screen_mask_candidates.py
```

## Smoke Check

```bash
conda run -n xfct python experiments/screen_mask_candidates.py --candidate-limit 2 --angle-phase-count 0 --output-root /tmp/mask_design_corrected_smoke --validation-summary results/forward_model_validation_clipped/validation_summary.json --screening-csv-name candidate_screening_corrected.csv --top-json-name top_candidates.json --pareto-json-name pareto_candidates.json --screening-report-name screening_report.md
```

## Corrected Screening

```bash
mkdir -p results/mask_design_corrected
conda run -n xfct python experiments/screen_mask_candidates.py --angle-phase-count 0 --output-root results/mask_design_corrected --validation-summary results/forward_model_validation_clipped/validation_summary.json --screening-csv-name candidate_screening_corrected.csv --top-json-name top_candidates.json --pareto-json-name pareto_candidates.json --screening-report-name screening_report.md --max-primary-candidates 5
```

Result:

```text
Forward validation passed.
Screened 90 candidates across 1 angle sets.
CSV: results/mask_design_corrected/candidate_screening_corrected.csv
Pareto candidates: results/mask_design_corrected/pareto_candidates.json
Top candidates: results/mask_design_corrected/top_candidates.json
```

## Artifact Inspection

```bash
find results/mask_design_corrected -maxdepth 2 -type f | sort
head -n 5 results/mask_design_corrected/candidate_screening_corrected.csv
sed -n '1,260p' results/mask_design_corrected/screening_report.md
sed -n '1,260p' results/mask_design_corrected/pareto_candidates.json
wc -l results/mask_design_corrected/candidate_screening_corrected.csv
sed -n '1,80p' results/mask_design_corrected/top_candidates.json
```
