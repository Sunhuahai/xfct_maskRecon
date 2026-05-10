from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = PROJECT_ROOT / (
    "data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "data/system_matrix/"
    "cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma_clipx80pad40.npz"
)


def _physical_row_mask(
    *,
    row_count: int,
    detector_z: int,
    detector_x: int,
    physical_detector_x: int,
    pad_x: int,
) -> np.ndarray:
    rows_per_angle = int(detector_z) * int(detector_x)
    if row_count % rows_per_angle != 0:
        raise ValueError(
            f"Matrix row count {row_count} is not divisible by detector rows/angle {rows_per_angle}."
        )
    if int(pad_x) < 0 or int(pad_x) + int(physical_detector_x) > int(detector_x):
        raise ValueError(
            "Invalid physical detector embedding: "
            f"pad_x={pad_x}, physical_detector_x={physical_detector_x}, detector_x={detector_x}."
        )
    support_x = np.zeros(int(detector_x), dtype=bool)
    support_x[int(pad_x) : int(pad_x) + int(physical_detector_x)] = True
    support_one_angle = np.tile(support_x, int(detector_z))
    angle_count = row_count // rows_per_angle
    return np.tile(support_one_angle, int(angle_count))


def clip_matrix(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_path = Path(args.output)
    A = load_npz(input_path).tocsr(copy=True)
    keep_rows = _physical_row_mask(
        row_count=A.shape[0],
        detector_z=int(args.detector_z),
        detector_x=int(args.detector_x),
        physical_detector_x=int(args.physical_detector_x),
        pad_x=int(args.pad_x),
    )
    row_sums_before = np.asarray(A.sum(axis=1)).ravel()
    total_before = float(np.sum(row_sums_before))
    virtual_before = float(np.sum(row_sums_before[~keep_rows]))

    zero_rows = np.flatnonzero(~keep_rows)
    for row in zero_rows:
        start = int(A.indptr[row])
        end = int(A.indptr[row + 1])
        if end > start:
            A.data[start:end] = 0.0
    A.eliminate_zeros()
    A.sort_indices()

    row_sums_after = np.asarray(A.sum(axis=1)).ravel()
    total_after = float(np.sum(row_sums_after))
    virtual_after = float(np.sum(row_sums_after[~keep_rows]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, A)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "shape": [int(A.shape[0]), int(A.shape[1])],
        "nnz_after": int(A.nnz),
        "detector_z": int(args.detector_z),
        "detector_x": int(args.detector_x),
        "physical_detector_x": int(args.physical_detector_x),
        "pad_x": int(args.pad_x),
        "physical_column_range_inclusive": [
            int(args.pad_x),
            int(args.pad_x) + int(args.physical_detector_x) - 1,
        ],
        "total_row_sum_before": total_before,
        "virtual_row_sum_before": virtual_before,
        "virtual_fraction_before": virtual_before / max(total_before, 1.0e-300),
        "total_row_sum_after": total_after,
        "virtual_row_sum_after": virtual_after,
        "virtual_fraction_after": virtual_after / max(total_after, 1.0e-300),
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip a padded XFCT CSR system matrix to the physical detector support."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "results/forward_model_validation_clipped/clip_summary.json"))
    parser.add_argument("--detector-z", type=int, default=80)
    parser.add_argument("--detector-x", type=int, default=160)
    parser.add_argument("--physical-detector-x", type=int, default=80)
    parser.add_argument("--pad-x", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    summary = clip_matrix(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
