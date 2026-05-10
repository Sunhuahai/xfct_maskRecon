# XFCT Mask Reconstruction Master Goal

Last updated: 2026-05-10, Asia/Shanghai.

This is the master Codex goal document. It defines the staged objective and the gating rule for reading the detailed stage prompts under `goal/`.

## Master Objective

Complete the XFCT mask reconstruction evidence program in a gated sequence:

1. Corrected grid9 full baseline evidence package.
2. DL / ROI metric validity and corrected physical-support candidate screening.
3. Raw-domain Poisson-TV MBIR hardening and reconstruction.
4. Final multi-seed equal-time, equal-dose, equal-count protocol comparison and robustness summary.

The project must preserve the scientific framing:

> Task-driven depth-coded sparse multi-pinhole XFCT under the true depth-dependent forward model, reconstructed from raw Poisson mask-domain data, compared against single-pinhole 5/15/45-view baselines under fair protocols.

Do not frame the project as a simple multi-hole throughput experiment, fixed 3x3 grid experiment, URA/MURA decoding experiment, or fixed-shift deconvolution experiment.

## Current Reliable Facts

- Real projection data are physical `80 x 80` detector arrays.
- Existing reconstruction pads detector x by 40 columns on each side to `80 x 160`.
- Physical padded detector x columns are `40..119`.
- Virtual padded x columns are `0..39` and `120..159`.
- The original grid9 5-view mask matrix wrote about `31.8%` of its row sum into virtual padded detector columns.
- The corrected clipped grid9 matrix has virtual-column row sum `0.0`.
- The corrected clipped matrix passes fail-critical forward validation.
- Existing downstream reconstruction, raw MBIR, protocol comparison, and robustness outputs are mostly quick smoke tests.
- Quick smoke tests are not final scientific evidence.

## Gated Reading Rule

Codex must not read or execute a later stage prompt until the previous stage has completed and written its `STAGE*_DONE.md` artifact.

Allowed reads at the start:

- `docs/goal.md`
- `docs/current_state_handoff_gpt_pro.md`
- `goal/stage1.md`

Forbidden until Stage 1 is complete:

- `goal/stage2.md`
- `goal/stage3.md`
- `goal/stage4.md`

Stage unlock conditions:

| stage to unlock | required prior artifact |
| --- | --- |
| `goal/stage2.md` | `results/corrected_grid9_stage1/STAGE1_DONE.md` |
| `goal/stage3.md` | `results/mask_design_corrected/STAGE2_DONE.md` |
| `goal/stage4.md` | `results/poisson_mbir_corrected/STAGE3_DONE.md` |

If a stage is blocked, Codex may stop only after writing the stage-specific `BLOCKED.md` requested in that stage prompt. A `BLOCKED.md` does not unlock later stages unless the user explicitly says to proceed.

## Master `/goal` Prompt

Use this as the top-level goal if running Codex in goal mode. It delegates the detailed work to the stage files while enforcing the reading gate.

```text
/goal Manage the XFCT mask reconstruction evidence program as a gated sequence. Start by reading only docs/goal.md, docs/current_state_handoff_gpt_pro.md, and goal/stage1.md. Do not read goal/stage2.md, goal/stage3.md, or goal/stage4.md until the required prior STAGE*_DONE.md artifact exists. Complete Stage 1 first: the corrected physical-support grid9 matrix must be validated, a non-quick EM-TV comparison against traditional_5/15/45 must be run, DL/ROI metrics must have valid/invalid quality flags, and results/corrected_grid9_stage1/STAGE1_DONE.md must document all passing checks. Only after Stage 1 is complete may Stage 2 be read and executed. Only after Stage 2 is complete may Stage 3 be read and executed. Only after Stage 3 is complete may Stage 4 be read and executed. Preserve detector-domain invariants, never use the original unclipped grid9 matrix for final mask conclusions, do not interpret invalid DL numerically, and do not claim that multi-hole XFCT is generally better or worse unless the completed evidence supports that exact statement.
```

## Stage Summary

### Stage 1

Detailed prompt:

- `goal/stage1.md`

Target artifact:

- `results/corrected_grid9_stage1/STAGE1_DONE.md`

Purpose:

Produce a corrected, non-quick, physically consistent EM-TV baseline comparing `traditional_5`, `traditional_15`, `traditional_45`, and corrected `mask_5_model`.

### Stage 2

Detailed prompt:

- `goal/stage2.md`

Locked until:

- `results/corrected_grid9_stage1/STAGE1_DONE.md` exists.

Target artifact:

- `results/mask_design_corrected/STAGE2_DONE.md`

Purpose:

Re-screen candidates under physical detector support and produce a small Pareto candidate set.

### Stage 3

Detailed prompt:

- `goal/stage3.md`

Locked until:

- `results/mask_design_corrected/STAGE2_DONE.md` exists.

Target artifact:

- `results/poisson_mbir_corrected/STAGE3_DONE.md`

Purpose:

Harden raw-domain Poisson-TV MBIR and reconstruct corrected grid9 plus selected sparse candidates with matched projection/operator support.

### Stage 4

Detailed prompt:

- `goal/stage4.md`

Locked until:

- `results/poisson_mbir_corrected/STAGE3_DONE.md` exists.

Target artifact:

- `results/final_protocol_comparison/STAGE4_DONE.md`

Purpose:

Run final multi-seed protocol comparison under equal acquisition time, equal incident dose, and equal detected counts, with robustness reporting.

## Global Non-Conclusions

Until the gated evidence is complete, do not conclude:

- that multi-hole XFCT is generally better or worse,
- that grid9 is a final failure,
- that blue-noise or ring candidates are final winners,
- that throughput alone explains performance,
- that fixed-shift decoded-domain reconstruction is the main method,
- or that invalid DL values are meaningful numerical outcomes.
