# Stage 4 Commands Run

Working directory: `/home/venti/PARA/Project/XFCT/xfct_maskRecon`

## Checks

```bash
python -m py_compile experiments/run_mask_protocol_comparison.py experiments/run_mask_pose_sensitivity.py
conda run -n xfct python experiments/run_mask_protocol_comparison.py --quick --output-root /tmp/final_protocol_comparison_smoke --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1 --num-seeds 1 --num-iterations 2 --protocols equal_acquisition_time --target-counts 50000 --background 1e-6
conda run -n xfct python experiments/run_mask_pose_sensitivity.py --quick --output-root /tmp/final_pose_sensitivity_smoke --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1 --num-iterations 2 --target-counts 50000 --background 1e-6 --summary-csv-name pose_sensitivity_summary.csv --summary-md-name pose_sensitivity_summary.md
```

## Final Protocol Comparison

Two longer 30- and 20-iteration protocol attempts were interrupted because the 45-view baseline dominated runtime. The completed final protocol run used finite-aperture PMMA, two seeds, 15 PDHG iterations, and cloned `equal_incident_dose` rows from `equal_acquisition_time` because those protocols are identical in this incident-flux model.

```bash
conda run -n xfct python experiments/run_mask_protocol_comparison.py --final --output-root results/final_protocol_comparison --pareto-candidates results/mask_design_corrected/pareto_candidates.json --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1,ring_n7_d1d25_mind3 --num-seeds 2 --num-iterations 15 --target-counts 200000 --background 1e-6
```

## Final Pose Sensitivity

The full and focused pose sweeps were interrupted because finite-aperture grid9 perturbations were too slow for the full perturbation set. The completed final pose run used finite-aperture PMMA and a minimal diagnostic perturbation profile.

```bash
conda run -n xfct python experiments/run_mask_pose_sensitivity.py --final --output-root results/final_protocol_comparison --pareto-candidates results/mask_design_corrected/pareto_candidates.json --candidate-ids grid9_p6_d1p25_5,blue_noise_n3_d1d25_mind3_s1 --num-iterations 3 --target-counts 200000 --background 1e-6 --perturbation-profile minimal --summary-csv-name pose_sensitivity_summary.csv --summary-md-name pose_sensitivity_summary.md
```
