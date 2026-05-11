from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_poisson_mbir_mask_recon import ScaledOperator, _grid9_candidate, _load_candidate_pool
from recon.poisson_tv_pdhg import run_poisson_tv_pdhg
from src.mask_xfct_model import (
    EPS,
    XFCTForwardConfig,
    XFCTMaskOperator,
    candidate_to_mask_config,
    load_candidate_json,
    make_roi_detection_phantom,
    poisson_deviance,
    residual_structure_score,
)
from src.reporting_roi import roi_analysis


RECON_SHAPE = (40, 60, 60)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "mask_pose_sensitivity"
TOP_CANDIDATES = PROJECT_ROOT / "results" / "mask_design" / "top_candidates.json"
PARETO_CANDIDATES = PROJECT_ROOT / "results" / "mask_design_corrected" / "pareto_candidates.json"
CANDIDATE_DIR = PROJECT_ROOT / "data" / "masks" / "candidates"


def _candidate_lookup(candidate_dir: Path, pareto_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for candidate in _load_candidate_pool(candidate_dir):
        lookup[candidate["candidate_id"]] = candidate
    if pareto_path.exists():
        payload = json.loads(pareto_path.read_text(encoding="utf-8"))
        for section in ("baselines", "primary_candidates"):
            for row in payload.get(section, []):
                path = row.get("json_path")
                if path and Path(path).exists():
                    candidate = load_candidate_json(path) | {"json_path": path}
                    lookup[candidate["candidate_id"]] = candidate
    return lookup


def _add_candidate(candidates: list[dict], candidate: dict, seen: set[str]) -> None:
    candidate = dict(candidate)
    candidate["angle_count"] = 5
    candidate_id = str(candidate["candidate_id"])
    if candidate_id not in seen:
        candidates.append(candidate)
        seen.add(candidate_id)


def _select_candidates(args: argparse.Namespace) -> list[dict]:
    candidate_dir = Path(args.candidate_dir)
    pareto_path = Path(args.pareto_candidates)
    lookup = _candidate_lookup(candidate_dir, pareto_path)
    selected: list[dict] = []
    seen: set[str] = set()
    if args.candidate_ids:
        for item in [value.strip() for value in str(args.candidate_ids).split(",") if value.strip()]:
            if item in {"grid9", "grid9_p6_d1p25_5", "grid3x3_n9_d1d25_mind6"}:
                candidate = _grid9_candidate()
            elif item in lookup:
                candidate = lookup[item]
            else:
                raise KeyError(f"Unknown candidate id {item!r}.")
            _add_candidate(selected, candidate, seen)
        return selected

    _add_candidate(selected, _grid9_candidate(), seen)
    if pareto_path.exists():
        payload = json.loads(pareto_path.read_text(encoding="utf-8"))
        for row in payload.get("primary_candidates", []):
            path = row.get("json_path")
            if path and Path(path).exists():
                _add_candidate(selected, load_candidate_json(path) | {"json_path": path}, seen)
            if args.quick and len(selected) >= 2:
                return selected
            if args.candidate_limit is not None and len(selected) >= int(args.candidate_limit):
                return selected
        return selected

    top_candidates = Path(args.top_candidates)
    if top_candidates.exists():
        payload = json.loads(top_candidates.read_text(encoding="utf-8"))
        for row in payload.get("top_candidates", []):
            if row.get("family") not in {"grid3x3", "single_center", "single_pinhole"}:
                path = row.get("json_path")
                if path and Path(path).exists():
                    _add_candidate(selected, load_candidate_json(path) | {"json_path": path}, seen)
                    return selected
    for candidate in _load_candidate_pool(candidate_dir):
        if candidate["family"] in {"blue_noise", "sparse_random", "ring", "ring_two"}:
            _add_candidate(selected, candidate, seen)
            break
    return selected


def _candidate_label(candidate: dict) -> str:
    return str(candidate["candidate_id"]).replace("/", "_")


def _operator(candidate: dict, args: argparse.Namespace, perturb: dict | None = None) -> XFCTMaskOperator:
    perturb = {} if perturb is None else dict(perturb)
    centers, diameter = candidate_to_mask_config(candidate)
    centers = np.asarray(centers, dtype=float).copy()
    centers[:, 0] += float(perturb.get("mask_dx_mm", 0.0))
    centers[:, 1] += float(perturb.get("mask_dz_mm", 0.0))
    if "center_jitter_sigma_mm" in perturb:
        rng = np.random.default_rng(int(args.seed) + int(abs(float(perturb["center_jitter_sigma_mm"])) * 1.0e6))
        centers += rng.normal(scale=float(perturb["center_jitter_sigma_mm"]), size=centers.shape)
    diameter = max(0.05, float(diameter) + float(perturb.get("hole_diameter_delta_mm", 0.0)))
    return XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=centers,
            hole_diameter_mm=diameter,
            angle_indices=(0, 9, 18, 27, 36),
            aperture_mode="point" if args.quick else "finite",
            aperture_samples=1 if args.quick else int(args.aperture_samples),
            attenuation="none" if args.quick else args.attenuation,
            detector_to_pinhole_mm=30.0 + float(perturb.get("detector_distance_delta_mm", 0.0)),
            detector_offset_x_mm=-0.5 + float(perturb.get("detector_offset_delta_mm", 0.0)),
            center_to_pinhole_mm=50.0 + float(perturb.get("rotation_center_delta_mm", 0.0)),
            angle_offset_deg=float(perturb.get("angle_delta_deg", 0.0)),
        )
    )


def _perturbations(quick: bool, profile: str = "full") -> list[dict]:
    items = [{"name": "nominal"}]
    for delta in [0.05, 0.1, 0.2]:
        items.append({"name": f"mask_dx_p{delta:g}", "mask_dx_mm": delta})
        items.append({"name": f"mask_dx_m{delta:g}", "mask_dx_mm": -delta})
        items.append({"name": f"mask_dz_p{delta:g}", "mask_dz_mm": delta})
        items.append({"name": f"mask_dz_m{delta:g}", "mask_dz_mm": -delta})
    for delta in [0.1, 0.5]:
        items.append({"name": f"detector_distance_p{delta:g}", "detector_distance_delta_mm": delta})
        items.append({"name": f"detector_distance_m{delta:g}", "detector_distance_delta_mm": -delta})
    for delta in [0.1]:
        items.append({"name": f"detector_offset_p{delta:g}", "detector_offset_delta_mm": delta})
        items.append({"name": f"detector_offset_m{delta:g}", "detector_offset_delta_mm": -delta})
        items.append({"name": f"rotation_center_p{delta:g}", "rotation_center_delta_mm": delta})
        items.append({"name": f"rotation_center_m{delta:g}", "rotation_center_delta_mm": -delta})
    for delta in [0.1, 0.5]:
        items.append({"name": f"angle_p{delta:g}", "angle_delta_deg": delta})
        items.append({"name": f"angle_m{delta:g}", "angle_delta_deg": -delta})
    for delta in [0.02, 0.05]:
        items.append({"name": f"hole_diameter_p{delta:g}", "hole_diameter_delta_mm": delta})
        items.append({"name": f"hole_diameter_m{delta:g}", "hole_diameter_delta_mm": -delta})
    for sigma in [0.02, 0.05]:
        items.append({"name": f"center_jitter_sigma{sigma:g}", "center_jitter_sigma_mm": sigma})
    if profile == "focused":
        keep = {
            "nominal",
            "mask_dx_p0.1",
            "mask_dx_m0.1",
            "mask_dz_p0.1",
            "mask_dz_m0.1",
            "detector_distance_p0.5",
            "detector_distance_m0.5",
            "angle_p0.5",
            "angle_m0.5",
            "hole_diameter_p0.05",
            "hole_diameter_m0.05",
            "center_jitter_sigma0.05",
        }
        return [item for item in items if item["name"] in keep]
    if profile == "minimal":
        keep = {
            "nominal",
            "mask_dx_p0.1",
            "detector_distance_p0.5",
            "angle_p0.5",
            "center_jitter_sigma0.05",
        }
        return [item for item in items if item["name"] in keep]
    if quick:
        # Keep the required perturbation families while limiting repeated directions.
        keep = {"nominal"}
        keep.update(name for name in [i["name"] for i in items] if any(token in name for token in ["0.1", "0.5", "sigma0.05"]))
        return [item for item in items if item["name"] in keep]
    return items


def _roi_metrics(volume: np.ndarray) -> dict[str, float | bool | str]:
    roi = roi_analysis(volume, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    polyf = np.asarray(roi["polyf"], dtype=float)
    v = np.asarray(roi["V"], dtype=float)
    return {
        "detection_limit_mgml": float(roi["DL"]),
        "detection_limit_valid": bool(roi.get("detection_limit_valid", False)),
        "detection_limit_invalid": bool(roi.get("detection_limit_invalid", True)),
        "detection_limit_invalid_reason": str(roi.get("detection_limit_invalid_reason", "")),
        "detection_limit_quality": str(roi.get("detection_limit_quality", "invalid")),
        "roi_r_squared": float(roi["r_squared"]),
        "cnr_slope": float(polyf[0]),
        "cnr_monotonic": bool(roi.get("cnr_monotonic", False)),
        "roi_bias_proxy": float(np.mean(v) - 1.3333333333),
    }


def _save_residual(residual: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = residual.reshape(5, 80, 160)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.0), squeeze=False)
    vmax = max(float(np.percentile(np.abs(arr), 99)), 1.0)
    for idx, axis in enumerate(axes.ravel()):
        im = axis.imshow(arr[idx], cmap="coolwarm", origin="upper", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_title(f"angle {idx}")
        axis.axvline(39.5, color="k", linestyle="--", linewidth=0.6)
        axis.axvline(119.5, color="k", linestyle="--", linewidth=0.6)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_case(candidate: dict, perturb: dict, args: argparse.Namespace, nominal_op: ScaledOperator, truth: np.ndarray) -> dict:
    pert_op_base = _operator(candidate, args, perturb)
    # Use the same incident exposure scale as the nominal mask to isolate pose/calibration mismatch.
    pert_op = ScaledOperator(pert_op_base, nominal_op.scale)
    y = pert_op.forward(truth.reshape(-1), support_mode="physical_padded") + float(args.background)
    result = run_poisson_tv_pdhg(
        nominal_op,
        y,
        recon_shape=RECON_SHAPE,
        background=float(args.background),
        beta=float(args.beta),
        num_iterations=int(args.num_iterations),
        support_mode="physical_padded",
        norm_power_iterations=1 if args.quick else 4,
        seed=int(args.seed),
        roi_every=0,
    )
    metrics = _roi_metrics(result.reconstruction)
    dev = poisson_deviance(y, result.lambda_hat)
    residual_score = residual_structure_score(result.residual)
    out_dir = Path(args.output_root) / _candidate_label(candidate)
    if perturb["name"] in {"nominal", "mask_dx_p0.1", "detector_distance_p0.5", "angle_p0.5", "center_jitter_sigma0.05"}:
        _save_residual(
            result.residual,
            out_dir / f"residual_{perturb['name']}.png",
            f"{candidate['candidate_id']} {perturb['name']}",
        )
    metrics.update(
        {
            "candidate_id": candidate["candidate_id"],
            "family": candidate["family"],
            "perturbation": perturb["name"],
            "projection_deviance": dev,
            "residual_structure_score": residual_score,
            "total_counts": float(np.sum(y)),
            "num_iterations": int(args.num_iterations),
            "perturbation_profile": str(args.perturbation_profile),
        }
    )
    return metrics


def _write_outputs(rows: list[dict], output_root: Path, csv_name: str, md_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / csv_name
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    by_candidate: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["perturbation"] == "nominal":
            by_candidate[row["candidate_id"]] = {
                "dl": float(row["detection_limit_mgml"]),
                "bias": float(row["roi_bias_proxy"]),
                "dev": float(row["projection_deviance"]),
            }
    lines = [
        "# Mask Pose and Geometry Sensitivity",
        "",
        "Reconstructions use the nominal forward model while expected projection data are generated with each listed perturbation. DL changes are interpreted only when both nominal and perturbed rows pass the shared CNR quality gate.",
        "",
        "| candidate | perturbation | DL change | ROI bias change | deviance increase | residual structure |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    aggregates = {}
    for row in rows:
        base = by_candidate.get(row["candidate_id"], {"dl": np.nan, "bias": np.nan, "dev": np.nan})
        dl_change = float(row["detection_limit_mgml"]) - base["dl"]
        bias_change = float(row["roi_bias_proxy"]) - base["bias"]
        dev_increase = float(row["projection_deviance"]) - base["dev"]
        dl_valid = bool(row.get("detection_limit_valid", False))
        nominal_valid = bool(next((r.get("detection_limit_valid", False) for r in rows if r["candidate_id"] == row["candidate_id"] and r["perturbation"] == "nominal"), False))
        dl_change_text = f"{dl_change:.4f}" if dl_valid and nominal_valid else "invalid"
        if row["perturbation"] != "nominal" and dl_valid and nominal_valid:
            aggregates.setdefault(row["candidate_id"], []).append(abs(dl_change))
        lines.append(
            f"| {row['candidate_id']} | {row['perturbation']} | {dl_change_text} | "
            f"{bias_change:.4f} | {dev_increase:.3e} | {row['residual_structure_score']:.4f} |"
        )
    if len(aggregates) >= 2:
        lines.extend(["", "## Robustness Comparison", ""])
        for candidate_id, values in aggregates.items():
            lines.append(f"- `{candidate_id}` mean absolute DL change: {float(np.mean(values)):.4f} mg/ml")
    output_root.joinpath(md_name).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess mask pose and geometry robustness under nominal-model reconstruction.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--num-seeds", type=int, default=1, help="Accepted for workflow compatibility; deterministic expected data are used.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Accepted for workflow compatibility.")
    parser.add_argument("--protocols", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--recon-methods", default="poisson_tv")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="matrix_free")
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR))
    parser.add_argument("--top-candidates", default=str(TOP_CANDIDATES))
    parser.add_argument("--pareto-candidates", default=str(PARETO_CANDIDATES))
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Comma-separated corrected mask candidate IDs to test.",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--summary-csv-name", default="mask_pose_sensitivity.csv")
    parser.add_argument("--summary-md-name", default="mask_pose_sensitivity.md")
    parser.add_argument("--perturbation-profile", choices=["full", "focused", "minimal"], default="full")
    parser.add_argument("--num-iterations", type=int, default=15)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--target-counts", type=float, default=2.0e5)
    parser.add_argument("--background", type=float, default=1.0e-6)
    parser.add_argument("--aperture-samples", type=int, default=8)
    parser.add_argument("--attenuation", choices=["none", "pmma"], default="pmma")
    parser.add_argument("--seed", type=int, default=20260509)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_iterations = min(int(args.num_iterations), 5)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = _select_candidates(args)
    truth = make_roi_detection_phantom(RECON_SHAPE)
    rows: list[dict] = []
    for candidate in candidates:
        nominal_base = _operator(candidate, args, None)
        lam0 = nominal_base.forward(truth.reshape(-1), support_mode="physical_padded")
        scale = float(args.target_counts) / max(float(np.sum(lam0)), EPS)
        nominal = ScaledOperator(nominal_base, scale)
        for perturb in _perturbations(bool(args.quick), str(args.perturbation_profile)):
            print(f"{candidate['candidate_id']}: {perturb['name']}")
            rows.append(run_case(candidate, perturb, args, nominal, truth))
    _write_outputs(rows, output_root, str(args.summary_csv_name), str(args.summary_md_name))
    print(f"Pose sensitivity summary: {output_root / str(args.summary_csv_name)}")


if __name__ == "__main__":
    main()
