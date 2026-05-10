from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
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
    default_angle_indices,
    load_candidate_json,
    make_roi_detection_phantom,
    poisson_deviance,
    residual_structure_score,
)
from src.reporting_roi import roi_analysis


RECON_SHAPE = (40, 60, 60)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "protocol_comparison"
TOP_CANDIDATES = PROJECT_ROOT / "results" / "mask_design" / "top_candidates.json"
CANDIDATE_DIR = PROJECT_ROOT / "data" / "masks" / "candidates"


def _single_candidate(name: str, angle_count: int) -> dict:
    return {
        "candidate_id": name,
        "family": "single_pinhole",
        "hole_centers_mm": [[0.0, 0.0]],
        "hole_diameter_mm": 1.0,
        "min_distance_mm": 0.0,
        "angle_count": int(angle_count),
        "comments": f"single-pinhole {angle_count}-view baseline",
    }


def _load_top_candidate_by(predicate) -> dict | None:
    if TOP_CANDIDATES.exists():
        payload = json.loads(TOP_CANDIDATES.read_text(encoding="utf-8"))
        for row in payload.get("top_candidates", []):
            if predicate(row):
                path = row.get("json_path")
                if path and Path(path).exists():
                    return load_candidate_json(path) | {"json_path": path}
    for candidate in _load_candidate_pool(CANDIDATE_DIR):
        if predicate(candidate):
            return candidate
    return None


def _select_runs(args: argparse.Namespace) -> list[dict]:
    runs = [
        _single_candidate("traditional_5", 5),
        _single_candidate("traditional_15", 15),
        _single_candidate("traditional_45", 45),
        _grid9_candidate(),
    ]
    runs[-1]["candidate_id"] = "grid9_p6_d1p25_5"
    runs[-1]["angle_count"] = 5
    selectors = [
        lambda c: c.get("family") in {"blue_noise", "sparse_random"} and int(c.get("hole_count", len(c.get("hole_centers_mm", [])))) == 5,
        lambda c: c.get("family") in {"blue_noise", "sparse_random"} and int(c.get("hole_count", len(c.get("hole_centers_mm", [])))) == 7,
        lambda c: c.get("family") in {"ring", "ring_two"},
        lambda c: c.get("family") == "ura_mura_inspired",
    ]
    labels = ["best_5hole_sparse", "best_7hole_sparse", "best_ring", "ura_mura_diagnostic"]
    seen = {r["candidate_id"] for r in runs}
    for label, selector in zip(labels, selectors):
        cand = _load_top_candidate_by(selector)
        if cand is None or cand["candidate_id"] in seen:
            continue
        cand = dict(cand)
        cand["candidate_id"] = f"{label}_{cand['candidate_id']}"
        cand["angle_count"] = 5
        runs.append(cand)
        seen.add(cand["candidate_id"])
        if args.quick and len(runs) >= 7:
            break
    if args.candidate_limit is not None:
        # Always keep single baselines and grid; limit only added masks when possible.
        base = runs[:4]
        extra = runs[4 : 4 + max(0, int(args.candidate_limit))]
        runs = base + extra
    return runs


def _operator_for_run(run: dict, args: argparse.Namespace) -> XFCTMaskOperator:
    centers, diameter = candidate_to_mask_config(run)
    angle_count = int(run.get("angle_count", 5))
    return XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=centers,
            hole_diameter_mm=diameter,
            angle_indices=tuple(default_angle_indices(angle_count)),
            aperture_mode="point" if args.quick else "finite",
            aperture_samples=1 if args.quick else int(args.aperture_samples),
            attenuation="none" if args.quick else args.attenuation,
        )
    )


def _roi_metrics(volume: np.ndarray, truth: np.ndarray) -> dict[str, float | bool]:
    roi = roi_analysis(volume, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    truth_roi = roi_analysis(truth, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    v = np.asarray(roi["V"], dtype=float)
    vt = np.asarray(truth_roi["V"], dtype=float)
    s = np.asarray(roi["S"], dtype=float)
    cnr = np.asarray(roi["CNR"], dtype=float)
    polyf = np.asarray(roi["polyf"], dtype=float)
    rmse = float(np.sqrt(np.mean((volume - truth) ** 2)))
    nrmse = float(rmse / max(np.max(truth) - np.min(truth), EPS))
    slope = float(polyf[0])
    r2 = float(roi["r_squared"])
    dl = float(roi["DL"])
    invalid = bool((not np.isfinite(dl)) or slope <= 0.0 or r2 < 0.25)
    return {
        "roi_mean": float(np.mean(v)),
        "roi_bias": float(np.mean(v - vt)),
        "background_mean": float(v[0]),
        "background_std": float(s[0]),
        "cnr_mean": float(np.mean(cnr[1:])),
        "cnr_slope": slope,
        "cnr_intercept": float(polyf[1]),
        "cnr_r_squared": r2,
        "detection_limit_mgml": dl,
        "detection_limit_invalid": invalid,
        "rmse": rmse,
        "nrmse": nrmse,
        "ssim": float("nan"),
    }


def _sensitivity_uniformity(op: XFCTMaskOperator) -> tuple[float, float]:
    sens = op.sensitivity("physical_padded")
    positive = sens[sens > 0.0]
    if positive.size == 0:
        return float("inf"), 0.0
    return float(np.std(positive) / max(np.mean(positive), EPS)), float(np.min(positive) / max(np.mean(positive), EPS))


def _truncation_ratio(op: XFCTMaskOperator, quick: bool) -> float:
    rng = np.random.default_rng(20260509)
    all_indices = np.arange(np.prod(RECON_SHAPE), dtype=np.int64)
    count = 500 if quick else 2500
    idx = rng.choice(all_indices, size=min(count, all_indices.size), replace=False)
    return float(np.mean(op.truncation_fraction(idx, physical=True)))


def _overlap_score(op: XFCTMaskOperator, phantom_flat: np.ndarray) -> float:
    n_holes = op.hole_centers_mm.shape[0]
    if n_holes <= 1:
        return 0.0
    footprints = []
    for hole_idx in range(n_holes):
        fp = op.forward(phantom_flat, support_mode="physical_padded", hole_indices=[hole_idx])
        footprints.append(fp / max(np.linalg.norm(fp), EPS))
    corr = np.abs(np.asarray(footprints) @ np.asarray(footprints).T)
    offdiag = corr[~np.eye(n_holes, dtype=bool)]
    return float(np.max(offdiag)) if offdiag.size else 0.0


def _save_image(volume: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(5.0, 4.6))
    im = axis.imshow(volume[20], cmap="jet", origin="upper", vmin=0.0, vmax=3.0)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_curve(values: np.ndarray, path: Path, ylabel: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(5.2, 3.8))
    axis.plot(values)
    axis.set_xlabel("iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_residual(residual: np.ndarray, path: Path, angle_count: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = residual.reshape(int(angle_count), 80, 160)
    shown = min(int(angle_count), 5)
    fig, axes = plt.subplots(1, shown, figsize=(3.2 * shown, 3.0), squeeze=False)
    vmax = max(float(np.percentile(np.abs(arr), 99)), 1.0)
    for idx, axis in enumerate(axes.ravel()):
        im = axis.imshow(arr[idx], cmap="coolwarm", origin="upper", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_title(f"angle {idx}")
        axis.axvline(39.5, color="k", linestyle="--", linewidth=0.6)
        axis.axvline(119.5, color="k", linestyle="--", linewidth=0.6)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_one(run: dict, protocol: str, seed: int, args: argparse.Namespace, truth: np.ndarray, base_exposure: float, target_counts: float) -> dict:
    base_op = _operator_for_run(run, args)
    truth_flat = truth.reshape(-1)
    lam_base = base_op.forward(truth_flat, support_mode="physical_padded")
    if protocol in {"equal_acquisition_time", "equal_incident_dose"}:
        exposure_scale = base_exposure
        dose_scale = base_exposure
    elif protocol == "equal_detected_counts":
        exposure_scale = target_counts / max(float(np.sum(lam_base)), EPS)
        dose_scale = exposure_scale
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    op = ScaledOperator(base_op, exposure_scale)
    lam = op.forward(truth_flat, support_mode="physical_padded") + float(args.background)
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.maximum(lam, 0.0)).astype(np.float64)
    start = time.time()
    result = run_poisson_tv_pdhg(
        op,
        y,
        recon_shape=RECON_SHAPE,
        background=float(args.background),
        beta=float(args.beta),
        num_iterations=int(args.num_iterations),
        support_mode="physical_padded",
        norm_power_iterations=1 if args.quick else 4,
        seed=seed,
        roi_every=0,
    )
    runtime_s = time.time() - start
    volume = np.asarray(result.reconstruction, dtype=float)
    out_dir = Path(args.output_root) / protocol / run["candidate_id"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reconstruction.npy", volume)
    angle_count = len(default_angle_indices(int(run.get("angle_count", 5))))
    np.save(out_dir / "raw_projection.npy", y.reshape(angle_count, 80, 160))
    _save_image(volume, Path(args.output_root) / "reconstruction_panels" / f"{protocol}_{run['candidate_id']}_seed{seed}.png", f"{protocol} {run['candidate_id']}")
    _save_curve(result.deviance_history, Path(args.output_root) / "convergence_curves" / f"{protocol}_{run['candidate_id']}_seed{seed}.png", "deviance", run["candidate_id"])
    _save_residual(result.residual, Path(args.output_root) / "residual_maps" / f"{protocol}_{run['candidate_id']}_seed{seed}.png", int(run.get("angle_count", 5)), run["candidate_id"])
    sens_cv, sens_min = _sensitivity_uniformity(base_op)
    trunc = _truncation_ratio(base_op, quick=bool(args.quick))
    overlap = _overlap_score(base_op, truth_flat)
    metrics = _roi_metrics(volume, truth)
    metrics.update(
        {
            "protocol": protocol,
            "run": run["candidate_id"],
            "family": run["family"],
            "seed": seed,
            "angle_count": int(run.get("angle_count", 5)),
            "hole_count": len(run["hole_centers_mm"]),
            "hole_diameter_mm": float(run["hole_diameter_mm"]),
            "total_detected_counts": float(np.sum(y)),
            "expected_detected_counts": float(np.sum(lam)),
            "estimated_dose_or_exposure_scale": float(dose_scale),
            "projection_poisson_deviance": poisson_deviance(y, result.lambda_hat),
            "residual_structure_score": residual_structure_score(result.residual),
            "sensitivity_uniformity_cv": sens_cv,
            "sensitivity_min_over_mean": sens_min,
            "fov_truncation_ratio": trunc,
            "overlap_score": overlap,
            "final_objective": float(result.objective_history[-1]),
            "final_deviance": float(result.deviance_history[-1]),
            "final_relative_change": float(result.relative_change[-1]),
            "runtime_s": runtime_s,
            "memory_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        }
    )
    return metrics


def _save_cnr_curves(rows: list[dict], output_root: Path) -> None:
    # Summary DL plot by protocol; individual ROI CNR data are in the CSV metrics.
    output_dir = output_root / "cnr_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = sorted(set(row["protocol"] for row in rows))
    for protocol in protocols:
        subset = [r for r in rows if r["protocol"] == protocol]
        labels = [r["run"] for r in subset]
        dl = np.array([float(r["detection_limit_mgml"]) for r in subset])
        fig, axis = plt.subplots(1, 1, figsize=(9.0, 4.5))
        axis.bar(np.arange(len(subset)), dl, color="tab:blue")
        axis.set_xticks(np.arange(len(subset)))
        axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        axis.set_ylabel("DL (mg/ml)")
        axis.set_title(f"{protocol}: detection-limit summary")
        fig.tight_layout()
        fig.savefig(output_dir / f"{protocol}_dl_summary.png", dpi=170)
        plt.close(fig)


def _write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "protocol_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Mask Protocol Comparison",
        "",
        "Protocols: equal acquisition time, equal incident dose, and equal detected counts.",
        "Dose is controlled by incident exposure scale, not by detected fluorescence count.",
        "",
        "| protocol | run | seed | counts | DL | invalid DL | ROI bias | deviance | residual structure |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {protocol} | {run} | {seed} | {total_detected_counts:.3e} | "
            "{detection_limit_mgml:.4f} | {detection_limit_invalid} | {roi_bias:.4f} | "
            "{projection_poisson_deviance:.3e} | {residual_structure_score:.4f} |".format(**row)
        )
    output_root.joinpath("protocol_summary.md").write_text("\n".join(lines), encoding="utf-8")
    _save_cnr_curves(rows, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-pinhole and sparse mask XFCT under fair protocols.")
    parser.add_argument("--quick", action="store_true", help="Use point-pinhole and few PDHG iterations.")
    parser.add_argument("--final", action="store_true", help="Use configured final settings.")
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--protocols", default="equal_acquisition_time,equal_incident_dose,equal_detected_counts")
    parser.add_argument("--recon-methods", default="poisson_tv", help="Currently supports poisson_tv; EM-TV remains available in run_effect_comparison.py.")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="matrix_free")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--num-iterations", type=int, default=20)
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
        args.num_seeds = min(int(args.num_seeds), max(1, int(args.num_seeds)))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    truth = make_roi_detection_phantom(RECON_SHAPE)
    traditional5 = _operator_for_run(_single_candidate("traditional_5", 5), args)
    lam_trad5 = traditional5.forward(truth.reshape(-1), support_mode="physical_padded")
    base_exposure = float(args.target_counts) / max(float(np.sum(lam_trad5)), EPS)
    runs = _select_runs(args)
    protocols = [item.strip() for item in str(args.protocols).split(",") if item.strip()]
    rows: list[dict] = []
    for protocol in protocols:
        for run in runs:
            for seed_idx in range(int(args.num_seeds)):
                seed = int(args.seed) + seed_idx
                print(f"{protocol}: {run['candidate_id']} seed={seed}")
                rows.append(run_one(run, protocol, seed, args, truth, base_exposure, float(args.target_counts)))
    _write_summary(rows, output_root)
    print(f"Protocol summary: {output_root / 'protocol_summary.csv'}")


if __name__ == "__main__":
    main()
