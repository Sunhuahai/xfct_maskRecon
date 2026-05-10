from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mask_xfct_model import (
    EPS,
    XFCTForwardConfig,
    XFCTMaskOperator,
    detector_moments,
    make_roi_detection_phantom,
    physical_support_columns,
    poisson_deviance,
    residual_map,
    scalar_fit_and_relative_error,
)


DEFAULT_MASK_MATRIX = PROJECT_ROOT / (
    "data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz"
)
DEFAULT_SINGLE_MATRIX = PROJECT_ROOT / "data/system_matrix/cij_5_3d_mod30_p1_lim0d5_xy60_z40_att_pmma.npz"
DEFAULT_MASK_PROJECTION = PROJECT_ROOT / "data/projections/mask/geometry_5_proj_cmask9_grid_p6_d1d25.npy"

RECON_SHAPE = (40, 60, 60)
DET_Z = 80
DET_X_PADDED = 160
PHYSICAL_DET_X = 80
PAD_X = 40
DEFAULT_ANGLE_INDICES = (0, 9, 18, 27, 36)
GRID9_CENTERS = np.array(
    [
        [-6.0, -6.0],
        [0.0, -6.0],
        [6.0, -6.0],
        [-6.0, 0.0],
        [0.0, 0.0],
        [6.0, 0.0],
        [-6.0, 6.0],
        [0.0, 6.0],
        [6.0, 6.0],
    ],
    dtype=np.float64,
)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _voxel_index(z: int, y: int, x: int) -> int:
    return int(np.ravel_multi_index((int(z), int(y), int(x)), RECON_SHAPE))


def _save_detector_support_plot(row_sums: np.ndarray | None, output_root: Path) -> Path:
    output_path = output_root / "detector_support.png"
    support = physical_support_columns(DET_X_PADDED, PHYSICAL_DET_X, PAD_X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    support_img = np.tile(support.astype(float), (DET_Z, 1))
    axes[0].imshow(support_img, cmap="gray", origin="upper", aspect="auto")
    axes[0].set_title("Physical 80-column support in 80x160 padded grid")
    axes[0].set_xlabel("padded detector x")
    axes[0].set_ylabel("detector z")
    axes[0].axvline(PAD_X - 0.5, color="tab:red", linestyle="--", linewidth=1)
    axes[0].axvline(PAD_X + PHYSICAL_DET_X - 0.5, color="tab:red", linestyle="--", linewidth=1)
    if row_sums is not None:
        support_sum = np.sum(row_sums.reshape(-1, DET_Z, DET_X_PADDED), axis=(0, 1))
        axes[1].plot(np.arange(DET_X_PADDED), support_sum, color="tab:blue")
        axes[1].axvspan(PAD_X, PAD_X + PHYSICAL_DET_X - 1, color="tab:green", alpha=0.18)
        axes[1].set_yscale("symlog", linthresh=1.0e-12)
        axes[1].set_title("Explicit matrix row-sum support")
        axes[1].set_xlabel("padded detector x")
        axes[1].set_ylabel("sum(A rows)")
    else:
        axes[1].text(0.5, 0.5, "Explicit matrix not loaded", ha="center", va="center")
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_projection_image(vector: np.ndarray, output_path: Path, title: str) -> None:
    arr = np.asarray(vector, dtype=float).reshape(-1, DET_Z, DET_X_PADDED)
    image = np.sum(arr, axis=0)
    fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.2))
    im = axis.imshow(image, cmap="magma", origin="upper", aspect="auto")
    axis.axvline(PAD_X - 0.5, color="cyan", linestyle="--", linewidth=1)
    axis.axvline(PAD_X + PHYSICAL_DET_X - 0.5, color="cyan", linestyle="--", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("padded detector x")
    axis.set_ylabel("detector z")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_explicit_matrix(path: Path, enabled: bool):
    if not enabled:
        return None
    if not path.exists():
        return None
    return load_npz(path)


def detector_padding_test(A_mask, output_root: Path) -> dict:
    result: dict[str, object] = {
        "name": "detector_padding",
        "status": "SKIPPED",
        "fail_critical": True,
        "message": "Explicit mask matrix was not loaded.",
    }
    row_sums = None
    if A_mask is not None:
        row_sums = np.asarray(A_mask.sum(axis=1)).ravel()
        if row_sums.size != len(DEFAULT_ANGLE_INDICES) * DET_Z * DET_X_PADDED:
            result.update(
                status="FAIL",
                message=f"Unexpected matrix row count {row_sums.size}.",
            )
        else:
            grid = row_sums.reshape(len(DEFAULT_ANGLE_INDICES), DET_Z, DET_X_PADDED)
            support = physical_support_columns(DET_X_PADDED, PHYSICAL_DET_X, PAD_X)
            virtual = grid[:, :, ~support]
            physical = grid[:, :, support]
            total = float(np.sum(grid))
            virtual_sum = float(np.sum(virtual))
            physical_sum = float(np.sum(physical))
            virtual_max = float(np.max(virtual)) if virtual.size else 0.0
            virtual_fraction = virtual_sum / max(total, EPS)
            result.update(
                status="PASS" if virtual_sum <= 1.0e-10 * max(total, 1.0) else "FAIL",
                message=(
                    "Matrix rows are confined to the physical detector support."
                    if virtual_sum <= 1.0e-10 * max(total, 1.0)
                    else "Matrix writes nonzero signal into virtual padded detector pixels."
                ),
                total_row_sum=total,
                physical_row_sum=physical_sum,
                virtual_row_sum=virtual_sum,
                virtual_fraction=virtual_fraction,
                virtual_max_row_sum=virtual_max,
                physical_columns=[PAD_X, PAD_X + PHYSICAL_DET_X - 1],
                padded_detector_shape=[DET_Z, DET_X_PADDED],
                physical_detector_shape=[DET_Z, PHYSICAL_DET_X],
            )
    plot_path = _save_detector_support_plot(row_sums, output_root)
    result["diagnostic_plot"] = str(plot_path)
    return result


def single_center_regression(A_single, output_root: Path, aperture_samples: int) -> dict:
    result: dict[str, object] = {
        "name": "single_center_pinhole_regression",
        "status": "SKIPPED",
        "message": "Explicit single-pinhole matrix was not loaded.",
    }
    if A_single is None:
        return result
    center_voxel = _voxel_index(20, 30, 30)
    reference = np.asarray(A_single[: DET_Z * DET_X_PADDED, center_voxel].todense()).ravel()
    op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=np.array([[0.0, 0.0]], dtype=np.float64),
            hole_diameter_mm=1.0,
            angle_indices=(0,),
            aperture_mode="finite",
            aperture_samples=int(aperture_samples),
            attenuation="pmma",
        )
    )
    estimate = op.forward_delta(center_voxel, support_mode="padded")
    scale, rel = scalar_fit_and_relative_error(reference, estimate)
    ref_mom = detector_moments(reference)
    est_mom = detector_moments(scale * estimate)
    _save_projection_image(reference, output_root / "single_center_explicit_footprint.png", "Single explicit matrix footprint")
    _save_projection_image(scale * estimate, output_root / "single_center_matrix_free_footprint.png", "Single matrix-free footprint after scalar fit")
    _save_projection_image(reference - scale * estimate, output_root / "single_center_residual.png", "Single regression residual")
    result.update(
        status="PASS" if rel < 0.25 else "WARN",
        message="One-hole matrix-free model compared with existing single-pinhole matrix.",
        scalar_fit=scale,
        relative_l2_after_scaling=rel,
        explicit_moments=ref_mom,
        matrix_free_moments=est_mom,
        voxel_index=center_voxel,
        angle_index=0,
    )
    return result


def delta_voxel_tests(output_root: Path, quick: bool, support_mode: str) -> dict:
    tests = {
        "center": _voxel_index(20, 30, 30),
        "near_edge_x": _voxel_index(20, 30, 55),
        "near_edge_z": _voxel_index(3, 30, 30),
        "shallow_y": _voxel_index(20, 5, 30),
        "deep_y": _voxel_index(20, 55, 30),
    }
    modes = [
        ("point_no_att", "point", 1, "none"),
        ("finite_no_att", "finite", 8 if quick else 32, "none"),
        ("finite_pmma", "finite", 4 if quick else 16, "pmma"),
    ]
    records = []
    max_centroid_delta = 0.0
    max_virtual_fraction = 0.0
    max_rel = 0.0
    angle_indices = (0, 9)
    for mode_name, aperture_mode, samples, attenuation in modes:
        op = XFCTMaskOperator(
            XFCTForwardConfig(
                hole_centers_mm=GRID9_CENTERS,
                hole_diameter_mm=1.25,
                angle_indices=angle_indices,
                aperture_mode=aperture_mode,
                aperture_samples=samples,
                attenuation=attenuation,
            )
        )
        hole_groups = {"all": None, "center_hole": [4]}
        if not quick:
            hole_groups.update({f"hole_{idx}": [idx] for idx in range(GRID9_CENTERS.shape[0])})
        else:
            hole_groups.update({"hole_0": [0], "hole_8": [8]})
        for voxel_name, voxel_idx in tests.items():
            for hole_name, hole_indices in hole_groups.items():
                padded = op.forward_delta(voxel_idx, support_mode=support_mode, hole_indices=hole_indices)
                physical = op.forward_delta(voxel_idx, support_mode="physical_padded", hole_indices=hole_indices)
                scale, rel = scalar_fit_and_relative_error(physical, padded)
                padded_mom = detector_moments(padded)
                physical_mom = detector_moments(physical)
                total_padded = float(np.sum(padded))
                total_physical = float(np.sum(physical))
                virtual_fraction = max(total_padded - total_physical, 0.0) / max(total_padded, EPS)
                centroid_delta = float(
                    np.hypot(
                        padded_mom["centroid_x_px"] - physical_mom["centroid_x_px"],
                        padded_mom["centroid_z_px"] - physical_mom["centroid_z_px"],
                    )
                )
                max_centroid_delta = max(max_centroid_delta, centroid_delta if np.isfinite(centroid_delta) else 0.0)
                max_virtual_fraction = max(max_virtual_fraction, virtual_fraction)
                max_rel = max(max_rel, rel)
                records.append(
                    {
                        "mode": mode_name,
                        "voxel": voxel_name,
                        "hole_group": hole_name,
                        "voxel_index": voxel_idx,
                        "padded_total": total_padded,
                        "physical_total": total_physical,
                        "virtual_fraction": virtual_fraction,
                        "centroid_delta_px": centroid_delta,
                        "relative_l2_after_scaling": rel,
                        "padded_centroid_x_px": padded_mom["centroid_x_px"],
                        "physical_centroid_x_px": physical_mom["centroid_x_px"],
                        "padded_sigma_x_px": padded_mom["sigma_x_px"],
                        "physical_sigma_x_px": physical_mom["sigma_x_px"],
                    }
                )
                if mode_name == "point_no_att" and voxel_name in {"center", "near_edge_x"} and hole_name in {"all", "hole_0", "hole_8"}:
                    stem = f"delta_{mode_name}_{voxel_name}_{hole_name}"
                    _save_projection_image(padded, output_root / f"{stem}_padded.png", f"{stem}: padded A")
                    _save_projection_image(physical, output_root / f"{stem}_physical.png", f"{stem}: physical generator support")
                    _save_projection_image(physical - padded, output_root / f"{stem}_residual.png", f"{stem}: physical - padded")
    status = "PASS"
    message = f"Delta footprints agree between physical support and A support_mode={support_mode}."
    if max_virtual_fraction > 1.0e-6 or max_centroid_delta > 0.1:
        status = "FAIL"
        message = f"Delta tests show physical-detector clipping differs from A support_mode={support_mode}."
    return {
        "name": "delta_voxel_tests",
        "status": status,
        "fail_critical": True,
        "message": message,
        "max_virtual_fraction": max_virtual_fraction,
        "max_centroid_delta_px": max_centroid_delta,
        "max_relative_l2_after_scaling": max_rel,
        "records": records,
    }


def multi_hole_linearity_test(output_root: Path, quick: bool, support_mode: str) -> dict:
    op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=GRID9_CENTERS,
            hole_diameter_mm=1.25,
            angle_indices=(0, 9),
            aperture_mode="finite",
            aperture_samples=4 if quick else 16,
            attenuation="none",
        )
    )
    phantom = make_roi_detection_phantom(RECON_SHAPE).reshape(-1)
    all_holes = op.forward(phantom, support_mode=support_mode)
    sum_holes = np.zeros_like(all_holes)
    for hole_idx in range(GRID9_CENTERS.shape[0]):
        sum_holes += op.forward(phantom, support_mode=support_mode, hole_indices=[hole_idx])
    diff = all_holes - sum_holes
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(all_holes) + EPS))
    max_abs = float(np.max(np.abs(diff)))
    total_err = float(abs(np.sum(all_holes) - np.sum(sum_holes)) / max(abs(np.sum(all_holes)), EPS))
    _save_projection_image(diff, output_root / "multi_hole_linearity_residual.png", "All holes minus sum of isolated holes")
    return {
        "name": "multi_hole_linearity",
        "status": "PASS" if rel < 1.0e-10 and max_abs < 1.0e-10 else "FAIL",
        "fail_critical": True,
        "relative_l2_error": rel,
        "max_absolute_error": max_abs,
        "total_count_error": total_err,
        "message": "All-hole forward projection equals the pixelwise isolated-hole sum.",
    }


def adjoint_tests(A_mask, quick: bool, support_mode: str) -> dict:
    rng = np.random.default_rng(20260509)
    out: dict[str, object] = {"name": "adjoint_tests", "status": "PASS", "fail_critical": True}
    op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=GRID9_CENTERS,
            hole_diameter_mm=1.25,
            angle_indices=(0, 9) if quick else DEFAULT_ANGLE_INDICES,
            aperture_mode="point",
            attenuation="none",
        )
    )
    x = rng.random(op.shape[1])
    y = rng.normal(size=op.shape[0])
    ax = op.forward(x, support_mode=support_mode)
    aty = op.adjoint(y, support_mode=support_mode)
    lhs = float(np.dot(ax, y))
    rhs = float(np.dot(x, aty))
    mf_rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), EPS)
    out["matrix_free"] = {"lhs": lhs, "rhs": rhs, "relative_error": mf_rel, "status": "PASS" if mf_rel < 1.0e-10 else "FAIL"}
    if A_mask is not None:
        x2 = rng.random(A_mask.shape[1])
        y2 = rng.normal(size=A_mask.shape[0])
        lhs2 = float(np.dot(A_mask @ x2, y2))
        rhs2 = float(np.dot(x2, A_mask.T @ y2))
        ex_rel = abs(lhs2 - rhs2) / max(abs(lhs2), abs(rhs2), EPS)
        out["explicit_csr"] = {"lhs": lhs2, "rhs": rhs2, "relative_error": ex_rel, "status": "PASS" if ex_rel < 1.0e-10 else "FAIL"}
    else:
        out["explicit_csr"] = {"status": "SKIPPED", "message": "Explicit mask matrix was not loaded."}
    if out["matrix_free"]["status"] != "PASS" or out["explicit_csr"].get("status") == "FAIL":
        out["status"] = "FAIL"
        out["message"] = "Adjoint identity failed."
    else:
        out["message"] = "Adjoint identity passed for available operator modes."
    return out


def residual_known_phantom_test(output_root: Path, quick: bool, support_mode: str) -> dict:
    op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=GRID9_CENTERS,
            hole_diameter_mm=1.25,
            angle_indices=(0, 9) if quick else DEFAULT_ANGLE_INDICES,
            aperture_mode="finite",
            aperture_samples=4 if quick else 16,
            attenuation="none",
        )
    )
    f_true = make_roi_detection_phantom(RECON_SHAPE).reshape(-1)
    background = 1.0e-6
    y = op.forward(f_true, support_mode=support_mode) + background
    lam = op.forward(f_true, support_mode=support_mode) + background
    residual_self = residual_map(y, lam)
    dev_self = poisson_deviance(y, lam)
    sens = np.maximum(op.sensitivity(support_mode), EPS)
    ratio = y / np.maximum(lam, EPS)
    f_after_one_em = f_true * op.adjoint(ratio, support_mode=support_mode) / sens
    lam_after = op.forward(f_after_one_em, support_mode=support_mode) + background
    dev_after = poisson_deviance(y, lam_after)
    y_physical = op.forward(f_true, support_mode="physical_padded") + background
    residual_physical_vs_a = residual_map(y_physical, lam)
    dev_physical_vs_a = poisson_deviance(y_physical, lam)
    _save_projection_image(residual_self, output_root / "known_phantom_self_residual.png", "Self-generated residual")
    _save_projection_image(residual_physical_vs_a, output_root / "known_phantom_physical_vs_padded_residual.png", "Physical generator support vs padded A residual")
    return {
        "name": "known_phantom_residual",
        "status": "PASS" if dev_self < 1.0e-8 and dev_after < 1.0e-6 else "FAIL",
        "message": "A-generated noiseless data are self-consistent; physical-support generator comparison is reported separately.",
        "self_deviance": dev_self,
        "one_em_step_deviance": dev_after,
        "one_em_relative_change": float(np.linalg.norm(f_after_one_em - f_true) / (np.linalg.norm(f_true) + EPS)),
        "physical_generator_vs_A_deviance": dev_physical_vs_a,
        "physical_generator_vs_A_residual_l2": float(np.linalg.norm(residual_physical_vs_a)),
        "physical_generator_vs_A_total_count_error": float(abs(np.sum(y_physical) - np.sum(lam)) / max(np.sum(lam), EPS)),
    }


def _write_outputs(summary: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    lines = [
        "# Forward Model Validation Summary",
        "",
        f"Overall status: **{summary['overall_status']}**",
        "",
        "| test | status | fail-critical | key result |",
        "| --- | --- | ---: | --- |",
    ]
    for test in summary["tests"]:
        key = test.get("message", "")
        if test["name"] == "detector_padding":
            key = (
                f"virtual fraction={test.get('virtual_fraction', float('nan')):.4e}; "
                f"virtual sum={test.get('virtual_row_sum', float('nan')):.4e}"
            )
        elif test["name"] == "delta_voxel_tests":
            key = (
                f"max virtual fraction={test.get('max_virtual_fraction', float('nan')):.4e}; "
                f"max centroid shift={test.get('max_centroid_delta_px', float('nan')):.3f} px"
            )
        elif test["name"] == "multi_hole_linearity":
            key = f"rel L2={test.get('relative_l2_error', float('nan')):.3e}"
        elif test["name"] == "adjoint_tests":
            key = (
                f"matrix-free rel={test.get('matrix_free', {}).get('relative_error', float('nan')):.3e}; "
                f"explicit status={test.get('explicit_csr', {}).get('status', 'NA')}"
            )
        elif test["name"] == "known_phantom_residual":
            key = (
                f"self dev={test.get('self_deviance', float('nan')):.3e}; "
                f"physical-vs-A dev={test.get('physical_generator_vs_A_deviance', float('nan')):.3e}"
            )
        lines.append(
            f"| {test['name']} | {test.get('status', 'NA')} | "
            f"{bool(test.get('fail_critical', False))} | {key} |"
        )
    lines.extend(
        [
            "",
            "## Stop Condition",
            "",
            summary["stop_condition_message"],
            "",
            "Diagnostic PNGs are saved in this directory.",
        ]
    )
    (output_root / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate consistency between the physical mask projection support and XFCT system matrices."
    )
    parser.add_argument("--quick", action="store_true", help="Use reduced aperture samples and fewer delta-hole cases.")
    parser.add_argument("--final", action="store_true", help="Use higher aperture sampling for finite-aperture tests.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Accepted for workflow compatibility; validation is deterministic.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Accepted for workflow compatibility.")
    parser.add_argument("--protocols", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--recon-methods", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="auto")
    parser.add_argument(
        "--support-mode",
        choices=["physical_padded", "padded"],
        default="physical_padded",
        help="Detector support expected for A; physical_padded matches 80x80 clipping followed by x padding.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "forward_model_validation"))
    parser.add_argument("--explicit-mask-matrix", default=str(DEFAULT_MASK_MATRIX))
    parser.add_argument("--explicit-single-matrix", default=str(DEFAULT_SINGLE_MATRIX))
    parser.add_argument(
        "--skip-explicit",
        action="store_true",
        help="Skip loading large CSR matrices. This leaves explicit-only checks marked SKIPPED.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    load_explicit = (args.matrix_mode in {"explicit", "auto"}) and not args.skip_explicit
    A_mask = _load_explicit_matrix(Path(args.explicit_mask_matrix), load_explicit)
    A_single = _load_explicit_matrix(Path(args.explicit_single_matrix), load_explicit)
    aperture_samples = 8 if args.quick else 32
    if args.final:
        aperture_samples = 64

    tests = [
        detector_padding_test(A_mask, output_root),
        single_center_regression(A_single, output_root, aperture_samples=aperture_samples),
        delta_voxel_tests(output_root, quick=bool(args.quick), support_mode=args.support_mode),
        multi_hole_linearity_test(output_root, quick=bool(args.quick), support_mode=args.support_mode),
        adjoint_tests(A_mask, quick=bool(args.quick), support_mode=args.support_mode),
        residual_known_phantom_test(output_root, quick=bool(args.quick), support_mode=args.support_mode),
    ]
    critical_failures = [
        test["name"]
        for test in tests
        if bool(test.get("fail_critical", False)) and str(test.get("status")) == "FAIL"
    ]
    overall = "FAIL" if critical_failures else "PASS"
    if critical_failures:
        stop_message = (
            "Stop before expensive mask sweeps. Fail-critical tests failed: "
            + ", ".join(critical_failures)
            + ". Recommended first fix: make the projection generator and system matrix use the same detector support "
            "(either clip all matrix rows to the physical 80-column detector before padding, or regenerate projections for "
            "a true 160-column detector)."
        )
    else:
        stop_message = "No fail-critical mismatch was found; mask screening can proceed."
    summary = {
        "overall_status": overall,
        "output_root": str(output_root),
        "matrix_mode": args.matrix_mode,
        "support_mode": args.support_mode,
        "explicit_mask_matrix": str(args.explicit_mask_matrix),
        "explicit_single_matrix": str(args.explicit_single_matrix),
        "tests": tests,
        "critical_failures": critical_failures,
        "stop_condition_message": stop_message,
    }
    _write_outputs(summary, output_root)
    print(f"Forward validation overall status: {overall}")
    print(stop_message)
    print(f"Summary: {output_root / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
