# Commands Run

Commands were run from `/home/venti/PARA/Project/XFCT/xfct_maskRecon`.

## Setup And Inspection

```bash
sed -n '1,220p' docs/goal.md
sed -n '1,260p' docs/current_state_handoff_gpt_pro.md
sed -n '1,280p' goal/stage1.md
find results -maxdepth 3 -type f | sort | sed -n '1,220p'
sed -n '1,280p' experiments/run_effect_comparison.py
sed -n '280,620p' experiments/run_effect_comparison.py
sed -n '1,260p' src/reporting_roi.py
sed -n '1,320p' src/reporting_figures.py
sed -n '1,260p' src/reporting_reconstruction.py
sed -n '260,620p' src/reporting_reconstruction.py
sed -n '1,260p' algorithm/recon_common.py
sed -n '1,320p' algorithm/em_tv.py
sed -n '1,280p' experiments/validate_mask_forward_consistency.py
sed -n '280,760p' experiments/validate_mask_forward_consistency.py
sed -n '1,220p' scripts/clip_system_matrix_to_physical_detector.py
sed -n '1,220p' results/forward_model_validation_clipped/validation_summary.md
sed -n '1,160p' results/effect_comparison/effect_comparison.csv
sed -n '1,160p' results/effect_comparison_clipped_quick/effect_comparison.csv
conda run -n xfct python --version
mkdir -p results/corrected_grid9_stage1
```

## Code Verification

```bash
conda run -n xfct python -m py_compile src/reporting_roi.py src/reporting_figures.py src/reporting_reconstruction.py experiments/run_effect_comparison.py
git diff -- src/reporting_roi.py src/reporting_figures.py src/reporting_reconstruction.py experiments/run_effect_comparison.py
```

## Validation Checks

```bash
ls -lh data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz results/forward_model_validation_clipped/validation_summary.json results/forward_model_validation_clipped/validation_summary.md
conda run -n xfct python -c "import json; from pathlib import Path; s=json.loads(Path('results/forward_model_validation_clipped/validation_summary.json').read_text()); required={'detector_padding','delta_voxel_tests','multi_hole_linearity','adjoint_tests','known_phantom_residual'}; tests={t['name']:t for t in s['tests']}; missing=required-set(tests); failed=[name for name in required if tests[name].get('status')!='PASS']; print('overall', s['overall_status']); print('missing', sorted(missing)); print('failed', failed); print('virtual_fraction', tests['detector_padding'].get('virtual_fraction')); assert s['overall_status']=='PASS'; assert not missing; assert not failed; assert float(tests['detector_padding'].get('virtual_fraction'))==0.0"
conda run -n xfct python -c "import numpy as np; from scipy.sparse import load_npz; path='data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz'; A=load_npz(path); row_sums=np.asarray(A.sum(axis=1)).ravel(); grid=row_sums.reshape(5,80,160); support=np.zeros(160,dtype=bool); support[40:120]=True; virtual=float(np.sum(grid[:,:,~support])); total=float(np.sum(grid)); physical=float(np.sum(grid[:,:,support])); print('path', path); print('shape', A.shape); print('nnz', A.nnz); print('total_row_sum', total); print('physical_row_sum', physical); print('virtual_row_sum', virtual); print('virtual_fraction', virtual/max(total,1e-300)); assert virtual == 0.0"
```

## Non-Quick Corrected Comparison

```bash
conda run -n xfct python experiments/run_effect_comparison.py --runs traditional_5,traditional_15,traditional_45,mask_5_model --methods em_tv --output-root results/corrected_grid9_stage1 --num-iterations 35 --mask-system-matrix data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz
```

## Result Audits

```bash
sed -n '1,120p' results/corrected_grid9_stage1/effect_comparison.csv
sed -n '1,140p' results/corrected_grid9_stage1/effect_comparison.md
find results/corrected_grid9_stage1 -maxdepth 2 -type f | sort
conda run -n xfct python -c "import csv, numpy as np
from pathlib import Path
root=Path('results/corrected_grid9_stage1')
rows=list(csv.DictReader((root/'effect_comparison.csv').open()))
print('rows', [r['run'] for r in rows])
assert [r['run'] for r in rows]==['traditional_5','traditional_15','traditional_45','mask_5_model']
mask=[r for r in rows if r['run']=='mask_5_model'][0]
print('mask_matrix', mask['system_matrix_path'])
assert 'clipx80pad40' in mask['system_matrix_path']
assert 'cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz' not in Path(mask['system_matrix_path']).name
assert mask['detection_limit_valid']=='False'
assert mask['detection_limit_invalid']=='True'
assert 'poor CNR linear fit' in mask['detection_limit_invalid_reason']
expected={'traditional_5':True,'traditional_15':True,'traditional_45':True,'mask_5_model':False}
for r in rows:
    p=root/f\"{r['run']}_em_tv/reconstruction_results.npz\"
    z=np.load(p)
    print(r['run'], 'iters', z['nll_history'].shape[0], 'dl_valid', bool(z['detection_limit_valid']), 'final_nll', float(z['nll_history'][-1]), 'final_rel', float(z['relative_change'][-1]))
    assert z['nll_history'].shape[0]==35
    assert bool(z['detection_limit_valid']) is expected[r['run']]
print('stage1 result audit ok')"
conda run -n xfct python -c "import numpy as np; from pathlib import Path; root=Path('results/corrected_grid9_stage1'); [print(run, np.load(root/f'{run}_em_tv/reconstruction_results.npz')['V'], np.load(root/f'{run}_em_tv/reconstruction_results.npz')['CNR']) for run in ['traditional_5','traditional_15','traditional_45','mask_5_model']]"
git status --short
```
