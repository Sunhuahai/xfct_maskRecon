# Commands Run

Commands were run from `/home/venti/PARA/Project/XFCT/xfct_maskRecon`.

## Gate And Context

```bash
sed -n '1,320p' goal/stage3.md
find results/poisson_mbir_corrected -maxdepth 3 -type f | sort 2>/dev/null || true
sed -n '1,360p' experiments/run_poisson_mbir_mask_recon.py
sed -n '220,520p' recon/poisson_tv_pdhg.py
sed -n '1,260p' results/mask_design_corrected/pareto_candidates.json
sed -n '1,160p' results/poisson_mbir_mask_recon/poisson_mbir_summary.csv
sed -n '1,160p' results/poisson_mbir_mask_recon/poisson_mbir_summary.md
find results/poisson_mbir_mask_recon -maxdepth 3 -type f | sort | sed -n '1,120p'
```

## Code Verification And Smoke

```bash
conda run -n xfct python -m py_compile recon/poisson_tv_pdhg.py experiments/run_poisson_mbir_mask_recon.py
git diff -- recon/poisson_tv_pdhg.py experiments/run_poisson_mbir_mask_recon.py
conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --output-root /tmp/poisson_mbir_corrected_smoke --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1 --grid9-data-mode real --betas 1e-4 --iteration-grid 2 --num-seeds 1 --target-counts 200000 --background 1e-6
conda run -n xfct python -m py_compile experiments/run_poisson_mbir_mask_recon.py
conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --output-root /tmp/poisson_mbir_corrected_smoke2 --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1 --grid9-data-mode real --betas 1e-4 --iteration-grid 2 --num-seeds 1 --target-counts 200000 --background 1e-6
```

The first smoke exposed a missing parent directory for aggregate residual maps.
After fixing `_save_residual_map()`, the second smoke completed and wrote summary
and report artifacts under `/tmp/poisson_mbir_corrected_smoke2`.

## Corrected Stage 3 Grid

```bash
mkdir -p results/poisson_mbir_corrected
conda run -n xfct python experiments/run_poisson_mbir_mask_recon.py --output-root results/poisson_mbir_corrected --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1,ring_n7_d1d25_mind3 --grid9-data-mode real --betas 1e-5,1e-4 --iteration-grid 30 --num-seeds 1 --target-counts 200000 --background 1e-6 --pareto-candidates results/mask_design_corrected/pareto_candidates.json
```

Result:

```text
Running raw Poisson-TV MBIR: grid9_p6_d1p25_5 seed=20260509 beta=1e-05 iter=30
Running raw Poisson-TV MBIR: grid9_p6_d1p25_5 seed=20260509 beta=0.0001 iter=30
Running raw Poisson-TV MBIR: blue_noise_n3_d1d25_mind3_s1 seed=20260509 beta=1e-05 iter=30
Running raw Poisson-TV MBIR: blue_noise_n3_d1d25_mind3_s1 seed=20260509 beta=0.0001 iter=30
Running raw Poisson-TV MBIR: ring_n7_d1d25_mind3 seed=20260509 beta=1e-05 iter=30
Running raw Poisson-TV MBIR: ring_n7_d1d25_mind3 seed=20260509 beta=0.0001 iter=30
Summary: results/poisson_mbir_corrected/poisson_mbir_summary.csv
Report: results/poisson_mbir_corrected/mbir_report.md
```

## Artifact Inspection

```bash
find results/poisson_mbir_corrected -maxdepth 3 -type f | sort | sed -n '1,220p'
sed -n '1,120p' results/poisson_mbir_corrected/parameter_grid.csv
sed -n '1,160p' results/poisson_mbir_corrected/poisson_mbir_summary.csv
sed -n '1,260p' results/poisson_mbir_corrected/mbir_report.md
ls -lh results/poisson_mbir_corrected/reconstruction_panels results/poisson_mbir_corrected/residual_maps
```
