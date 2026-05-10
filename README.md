# XFCT Mask Reconstruction Baseline

This folder is an independent baseline workspace for testing whether the current
single-pinhole XFCT geometry can be extended to a multi-hole coded mask.

The original reference folders are not modified:

- projection generation reference: `../xfct_geant4/read_make_proj`
- reconstruction reference: `../xfct_fastRecon`

## Contents

- `projection/`: copied and adapted XRF projection generation code, including a
  configurable multi-hole mask generator.
- `algorithm/`, `src/`: copied reconstruction code from `xfct_fastRecon`, plus
  `src/mask_decode.py` for a first coded-mask decoding baseline.
- `scripts/build_mask_system_matrix.py`: multi-hole coded-mask system-matrix
  generator derived from the previous Monte Carlo `sm_cal.py` structure.
- `data/projections/simulation/`: traditional 5/15/45-angle single-pinhole
  simulation projections.
- `data/projections/mask/`: copied 45-angle 3x3 mask projection and generated
  5/15-angle subsamples.
- `data/system_matrix/`: copied 5/15/45 single-pinhole system matrices plus the
  generated 5-angle 3x3 mask system matrix.
- `experiments/run_effect_comparison.py`: default effect comparison entrypoint.
- `docs/mask_recon_gpt_pro_brief.md`: geometry and research brief for deeper
  investigation.

## Default Baseline

From this directory:

```bash
conda run -n xfct python scripts/subsample_mask_projection.py
conda run -n xfct python experiments/run_effect_comparison.py
```

The default comparison runs:

- `traditional_5`: single-pinhole 5-angle EM-TV reconstruction.
- `mask_5_naive`: 3x3 mask 5-angle data reconstructed with the single-pinhole
  system matrix. This is the intentional negative-control baseline.
- `mask_5_wiener`: 3x3 mask 5-angle data decoded by a fixed-shift Wiener filter,
  then reconstructed with the single-pinhole system matrix.
- `mask_5_model`: 3x3 mask 5-angle data reconstructed directly with the matching
  multi-hole system matrix. This is the current mask baseline.
- `traditional_15`: single-pinhole 15-angle EM-TV comparison.
- `traditional_45`: single-pinhole 45-angle upper reference.

Outputs are written to `results/effect_comparison/`:

- `effect_comparison.csv`
- `effect_comparison.md`
- `reconstruction_panel.png`
- per-run reconstruction volumes and ROI figures

For a quick smoke test:

```bash
conda run -n xfct python experiments/run_effect_comparison.py --quick --runs traditional_5,mask_5_model --output-root results/smoke
```

To include the copied pseudo-MBIR path:

```bash
conda run -n xfct python experiments/run_effect_comparison.py --methods em_tv,pseudo_mbir
```

## Generate A Matching Multi-Hole System Matrix

The previous single-pinhole system-matrix generator is directly extensible to a
multi-hole mask: compute each hole's detector footprint and solid-angle weight,
then accumulate all holes into the same detector rows. The standalone version is:

```bash
conda run -n xfct python scripts/build_mask_system_matrix.py \
  --angle-indices 0,9,18,27,36 \
  --mask-layout grid3x3 \
  --mask-pitch-mm 6 \
  --mask-hole-diameter-mm 1.25 \
  --detector-to-pinhole 30 \
  --center-to-pinhole 50 \
  --detector-x 160 \
  --detector-z 80 \
  --image-xy 60 \
  --image-z 40 \
  --voxel-size 0.5 \
  --n-sample 500
```

Generated output:

```text
data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz
```

The generated matrix is `2.0 GB`, shape `(64000, 144000)`, with `273,186,535`
nonzeros. It is now included in the default comparison.

Run only the direct mask-model baseline:

```bash
conda run -n xfct python experiments/run_effect_comparison.py \
  --runs traditional_5,mask_5_model,traditional_15,traditional_45
```

## Generate New Mask Projections

The copied 45-angle mask projection is already present. To regenerate from
GEANT4 event txt files:

```bash
conda run -n xfct python projection/fluorescence_cmask.py \
  --input-dir ../xfct_geant4/read_make_proj/output/geometry \
  --phantom-map data/phantom/pmma_gd_hex_80_120_120.txt \
  --mask-layout grid3x3 \
  --mask-pitch-mm 6 \
  --mask-hole-diameter-mm 1.25 \
  --angle-indices 0,9,18,27,36
```

Useful alternatives:

```bash
conda run -n xfct python projection/fluorescence_cmask.py --help
```

Supported layouts include `single`, `grid3x3`, `grid`, `cross5`, `random`, and
custom two-column mask-center files.
