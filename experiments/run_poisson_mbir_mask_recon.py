from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "poisson_mbir_mask_recon"
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "data" / "masks" / "candidates"
DEFAULT_TOP_CANDIDATES = PROJECT_ROOT / "results" / "mask_design" / "top_candidates.json"


class ScaledOperator:
    def __init__(self, base: XFCTMaskOperator, scale: float):
        self.base = base
        self.scale = float(scale)
        self.shape = base.shape
        self.hole_centers_mm = base.hole_centers_mm

    def forward(self, x, support_mode: str = "physical_padded"):
        return self.scale * self.base.forward(x, support_mode=support_mode)

    def adjoint(self, y, support_mode: str = "physical_padded"):
        return self.scale * self.base.adjoint(y, support_mode=support_mode)


def _grid9_candidate() -> dict:
    centers = []
    for z in [-6.0, 0.0, 6.0]:
        for x in [-6.0, 0.0, 6.0]:
            centers.append([x, z])
    return {
        "candidate_id": "grid9_p6_d1p25_5",
        "family": "grid3x3",
        "hole_centers_mm": centers,
        "hole_diameter_mm": 1.25,
        "min_distance_mm": 6.0,
        "total_open_area_mm2": float(9 * np.pi * (1.25 / 2.0) ** 2),
        "comments": "current grid baseline",
    }


def _load_candidate_pool(candidate_dir: Path) -> list[dict]:
    if not candidate_dir.exists():
        return []
    return [load_candidate_json(path) | {"json_path": str(path)} for path in sorted(candidate_dir.glob("*.json"))]


def _select_candidates(args: argparse.Namespace) -> list[dict]:
    selected = [_grid9_candidate()]
    seen = {selected[0]["candidate_id"]}
    top_path = Path(args.top_candidates)
    if top_path.exists():
        payload = json.loads(top_path.read_text(encoding="utf-8"))
        for row in payload.get("top_candidates", []):
            family = str(row.get("family", ""))
            if family in {"grid3x3", "single_center"}:
                continue
            json_path = row.get("json_path")
            candidate = load_candidate_json(json_path) | {"json_path": json_path} if json_path and Path(json_path).exists() else None
            if candidate and candidate["candidate_id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["candidate_id"])
            if args.quick and len(selected) >= 2:
                return selected
    for candidate in _load_candidate_pool(Path(args.candidate_dir)):
        if candidate["candidate_id"] in seen:
            continue
        if candidate["family"] in {"blue_noise", "sparse_random", "ring", "ring_two", "cross_plus_center"}:
            selected.append(candidate)
            seen.add(candidate["candidate_id"])
        if args.quick and len(selected) >= 2:
            break
        if len(selected) >= int(args.candidate_limit or 5):
            break
    return selected[: int(args.candidate_limit)] if args.candidate_limit is not None else selected


def _save_recon_panel(volume: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(5.6, 5.0))
    im = axis.imshow(volume[20], cmap="jet", origin="upper", vmin=0.0, vmax=3.0)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_residual_map(residual: np.ndarray, path: Path, title: str, angle_count: int) -> None:
    arr = residual.reshape(int(angle_count), 80, 160)
    fig, axes = plt.subplots(1, min(angle_count, 5), figsize=(3.6 * min(angle_count, 5), 3.2), squeeze=False)
    vmax = float(np.percentile(np.abs(arr), 99)) if arr.size else 1.0
    vmax = max(vmax, 1.0)
    for idx, axis in enumerate(axes.ravel()):
        im = axis.imshow(arr[idx], cmap="coolwarm", origin="upper", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_title(f"angle {idx}")
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        axis.axvline(39.5, color="k", linestyle="--", linewidth=0.7)
        axis.axvline(119.5, color="k", linestyle="--", linewidth=0.7)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_convergence(result, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(result.objective_history)
    axes[0].set_title("objective")
    axes[0].set_xlabel("iteration")
    axes[1].plot(result.deviance_history, label="deviance")
    axes[1].plot(result.relative_change, label="relative change")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("fit / convergence")
    axes[1].set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _roi_metrics(volume: np.ndarray) -> dict[str, float]:
    roi = roi_analysis(volume, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    polyf = np.asarray(roi["polyf"], dtype=float)
    return {
        "detection_limit_mgml": float(roi["DL"]),
        "roi_r_squared": float(roi["r_squared"]),
        "cnr_slope": float(polyf[0]),
        "cnr_intercept": float(polyf[1]),
        "roi_mean": float(np.mean(np.asarray(roi["V"], dtype=float))),
        "background_mean": float(np.asarray(roi["V"], dtype=float)[0]),
        "background_std": float(np.asarray(roi["S"], dtype=float)[0]),
    }


def run_candidate(candidate: dict, args: argparse.Namespace, seed: int, output_root: Path) -> dict:
    centers, diameter = candidate_to_mask_config(candidate)
    base_op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=centers,
            hole_diameter_mm=diameter,
            angle_indices=(0, 9, 18, 27, 36),
            aperture_mode="point" if args.quick else "finite",
            aperture_samples=1 if args.quick else int(args.aperture_samples),
            attenuation="none" if args.quick else args.attenuation,
        )
    )
    f_true = make_roi_detection_phantom(RECON_SHAPE).reshape(-1)
    lam0 = base_op.forward(f_true, support_mode="physical_padded")
    exposure_scale = float(args.target_counts) / max(float(np.sum(lam0)), EPS)
    op = ScaledOperator(base_op, exposure_scale)
    lam = op.forward(f_true, support_mode="physical_padded") + float(args.background)
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.maximum(lam, 0.0)).astype(np.float64)
    run_dir = output_root / candidate["candidate_id"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "raw_mask_projection.npy", y.reshape(5, 80, 160))
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
        roi_every=1,
    )
    runtime_s = time.time() - start
    volume = np.asarray(result.reconstruction, dtype=float)
    np.save(run_dir / "reconstruction.npy", volume)
    np.savez_compressed(
        run_dir / "poisson_tv_results.npz",
        objective_history=result.objective_history,
        deviance_history=result.deviance_history,
        relative_change=result.relative_change,
        lambda_hat=result.lambda_hat,
        residual=result.residual,
    )
    (run_dir / "roi_history.json").write_text(json.dumps(result.roi_history, indent=2), encoding="utf-8")
    _save_recon_panel(volume, run_dir / "reconstruction.png", candidate["candidate_id"])
    _save_residual_map(result.residual, run_dir / "residual_map.png", f"{candidate['candidate_id']} normalized residual", 5)
    _save_convergence(result, run_dir / "convergence_curve.png", candidate["candidate_id"])
    metrics = _roi_metrics(volume)
    metrics.update(
        {
            "candidate_id": candidate["candidate_id"],
            "family": candidate["family"],
            "seed": seed,
            "hole_count": len(candidate["hole_centers_mm"]),
            "hole_diameter_mm": float(candidate["hole_diameter_mm"]),
            "raw_total_counts": float(np.sum(y)),
            "expected_total_counts": float(np.sum(lam)),
            "exposure_scale": exposure_scale,
            "final_objective": float(result.objective_history[-1]),
            "final_deviance": float(result.deviance_history[-1]),
            "final_relative_change": float(result.relative_change[-1]),
            "residual_structure_score": residual_structure_score(result.residual),
            "runtime_s": runtime_s,
            "reconstruction_path": str(run_dir / "reconstruction.npy"),
            "residual_path": str(run_dir / "residual_map.png"),
            "comments": candidate.get("comments", ""),
        }
    )
    return metrics


def _write_summary(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    csv_path = output_root / "poisson_mbir_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Raw-Domain Poisson MBIR Mask Reconstruction",
        "",
        "All reconstructions use raw padded detector measurements with physical 80-column support; no fixed-shift decoding is used.",
        "",
        "| candidate | family | seed | counts | DL (mg/ml) | R2 | deviance | residual structure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {family} | {seed} | {raw_total_counts:.3e} | "
            "{detection_limit_mgml:.4f} | {roi_r_squared:.4f} | "
            "{final_deviance:.3e} | {residual_structure_score:.4f} |".format(**row)
        )
    (output_root / "poisson_mbir_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-domain Poisson-TV MBIR for mask-domain XFCT data.")
    parser.add_argument("--quick", action="store_true", help="Use point-pinhole geometry, two candidates, and few iterations.")
    parser.add_argument("--final", action="store_true", help="Use configured finite-aperture settings.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=2)
    parser.add_argument("--protocols", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--recon-methods", default="poisson_tv", help="Accepted for workflow compatibility.")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="matrix_free")
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--top-candidates", default=str(DEFAULT_TOP_CANDIDATES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
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
        args.num_iterations = min(int(args.num_iterations), 6)
        args.candidate_limit = min(int(args.candidate_limit), 2)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = _select_candidates(args)
    if not candidates:
        raise RuntimeError("No mask candidates available for raw Poisson MBIR.")
    rows = []
    for candidate in candidates:
        for seed_offset in range(int(args.num_seeds)):
            seed = int(args.seed) + seed_offset
            print(f"Running raw Poisson-TV MBIR: {candidate['candidate_id']} seed={seed}")
            rows.append(run_candidate(candidate, args, seed, output_root))
    _write_summary(rows, output_root)
    print(f"Summary: {output_root / 'poisson_mbir_summary.csv'}")


if __name__ == "__main__":
    main()
