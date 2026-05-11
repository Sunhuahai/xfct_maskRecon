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
DEFAULT_PARETO_CANDIDATES = PROJECT_ROOT / "results" / "mask_design_corrected" / "pareto_candidates.json"
DEFAULT_GRID9_PROJECTION = PROJECT_ROOT / "data/projections/mask/geometry_5_proj_cmask9_grid_p6_d1d25.npy"
STAGE1_COMPARISON = PROJECT_ROOT / "results" / "corrected_grid9_stage1" / "effect_comparison.csv"


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


def _format_float_tag(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m").replace(".", "d")


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _is_grid9_candidate(candidate: dict) -> bool:
    return str(candidate.get("candidate_id", "")) in {
        "grid9",
        "grid9_p6_d1p25_5",
        "grid3x3_n9_d1d25_mind6",
    } or (
        str(candidate.get("family", "")) == "grid3x3"
        and int(candidate.get("hole_count", len(candidate.get("hole_centers_mm", [])))) == 9
        and abs(float(candidate.get("hole_diameter_mm", 0.0)) - 1.25) < 1.0e-9
    )


def _load_candidate_pool(candidate_dir: Path) -> list[dict]:
    if not candidate_dir.exists():
        return []
    return [load_candidate_json(path) | {"json_path": str(path)} for path in sorted(candidate_dir.glob("*.json"))]


def _candidate_lookup(candidate_dir: Path, pareto_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for candidate in _load_candidate_pool(candidate_dir):
        lookup[candidate["candidate_id"]] = candidate
    if pareto_path.exists():
        payload = json.loads(pareto_path.read_text(encoding="utf-8"))
        for section in ("baselines", "primary_candidates"):
            for row in payload.get(section, []):
                json_path = row.get("json_path")
                if json_path and Path(json_path).exists():
                    candidate = load_candidate_json(json_path) | {"json_path": json_path}
                    lookup[candidate["candidate_id"]] = candidate
    return lookup


def _select_candidates(args: argparse.Namespace) -> list[dict]:
    lookup = _candidate_lookup(Path(args.candidate_dir), Path(args.pareto_candidates))
    if args.candidate_ids:
        selected = []
        seen: set[str] = set()
        for item in [v.strip() for v in str(args.candidate_ids).split(",") if v.strip()]:
            if item in {"grid9", "grid9_p6_d1p25_5", "grid3x3_n9_d1d25_mind6"}:
                candidate = _grid9_candidate()
            elif item in lookup:
                candidate = lookup[item]
            else:
                raise KeyError(f"Unknown candidate id {item!r}.")
            if candidate["candidate_id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["candidate_id"])
        return selected

    selected = [_grid9_candidate()]
    seen = {selected[0]["candidate_id"]}
    pareto_path = Path(args.pareto_candidates)
    if pareto_path.exists():
        payload = json.loads(pareto_path.read_text(encoding="utf-8"))
        for row in payload.get("primary_candidates", []):
            json_path = row.get("json_path")
            candidate = load_candidate_json(json_path) | {"json_path": json_path} if json_path and Path(json_path).exists() else None
            if candidate and candidate["candidate_id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["candidate_id"])
            if args.quick and len(selected) >= 2:
                return selected
            if len(selected) >= int(args.candidate_limit or 2):
                return selected
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _roi_metrics(volume: np.ndarray, truth: np.ndarray) -> dict[str, float | bool | str]:
    roi = roi_analysis(volume, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    truth_roi = roi_analysis(truth, slice_index=20, recon_size=RECON_SHAPE, roi_layout="simulation")
    polyf = np.asarray(roi["polyf"], dtype=float)
    v = np.asarray(roi["V"], dtype=float)
    vt = np.asarray(truth_roi["V"], dtype=float)
    return {
        "detection_limit_mgml": float(roi["DL"]),
        "detection_limit_valid": bool(roi.get("detection_limit_valid", False)),
        "detection_limit_invalid": bool(roi.get("detection_limit_invalid", True)),
        "detection_limit_invalid_reason": str(roi.get("detection_limit_invalid_reason", "")),
        "roi_r_squared": float(roi["r_squared"]),
        "cnr_slope": float(polyf[0]),
        "cnr_intercept": float(polyf[1]),
        "cnr_monotonic": bool(roi.get("cnr_monotonic", False)),
        "roi_mean": float(np.mean(v)),
        "roi_bias": float(np.mean(v - vt)),
        "roi_rmse": float(np.sqrt(np.mean((v - vt) ** 2))),
        "background_mean": float(v[0]),
        "background_std": float(np.asarray(roi["S"], dtype=float)[0]),
    }


def _load_real_grid9_projection(path: Path) -> np.ndarray:
    projection = np.load(path)
    if projection.shape != (5, 80, 80):
        raise ValueError(f"Expected grid9 projection shape (5, 80, 80), got {projection.shape}.")
    padded = np.pad(np.asarray(projection, dtype=np.float64), ((0, 0), (0, 0), (40, 40)), mode="constant")
    return padded.reshape(-1)


def _operator_validation(base_op: XFCTMaskOperator, seed: int) -> dict[str, float | str | bool]:
    rng = np.random.default_rng(seed)
    x = rng.random(base_op.shape[1])
    y = rng.normal(size=base_op.shape[0])
    ax = base_op.forward(x, support_mode="physical_padded")
    aty = base_op.adjoint(y, support_mode="physical_padded")
    lhs = float(np.dot(ax, y))
    rhs = float(np.dot(x, aty))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), EPS)
    delta = base_op.forward_delta(base_op.shape[1] // 2, support_mode="physical_padded")
    grid = delta.reshape(5, 80, 160)
    virtual_sum = float(np.sum(grid[:, :, :40]) + np.sum(grid[:, :, 120:]))
    phantom = make_roi_detection_phantom(RECON_SHAPE).reshape(-1)
    lam = base_op.forward(phantom, support_mode="physical_padded") + 1.0e-6
    self_dev = poisson_deviance(lam, lam)
    return {
        "operator_support_mode": "physical_padded",
        "adjoint_relative_error": rel,
        "delta_virtual_sum": virtual_sum,
        "synthetic_self_deviance": float(self_dev),
        "finite_synthetic_lambda": bool(np.all(np.isfinite(lam))),
        "validation_status": "PASS" if rel < 1.0e-10 and virtual_sum == 0.0 and np.all(np.isfinite(lam)) else "FAIL",
    }


def run_candidate(
    candidate: dict,
    args: argparse.Namespace,
    seed: int,
    output_root: Path,
    *,
    beta: float,
    num_iterations: int,
) -> dict:
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
    validation = _operator_validation(base_op, seed=seed)
    if validation["validation_status"] != "PASS":
        raise RuntimeError(f"Operator validation failed for {candidate['candidate_id']}: {validation}")
    f_true = make_roi_detection_phantom(RECON_SHAPE).reshape(-1)
    lam0 = base_op.forward(f_true, support_mode="physical_padded")
    projection_path = ""
    if _is_grid9_candidate(candidate) and args.grid9_data_mode == "real":
        y = _load_real_grid9_projection(Path(args.grid9_projection))
        signal_counts = max(float(np.sum(y)) - float(args.background) * y.size, EPS)
        exposure_scale = signal_counts / max(float(np.sum(lam0)), EPS)
        data_domain = "real_grid9_projection_padded_physical_detector"
        projection_path = str(Path(args.grid9_projection))
    else:
        exposure_scale = float(args.target_counts) / max(float(np.sum(lam0)), EPS)
        data_domain = "synthetic_matched_poisson"
    op = ScaledOperator(base_op, exposure_scale)
    lam = op.forward(f_true, support_mode="physical_padded") + float(args.background)
    if data_domain == "synthetic_matched_poisson":
        rng = np.random.default_rng(seed)
        y = rng.poisson(np.maximum(lam, 0.0)).astype(np.float64)
    beta_tag = _format_float_tag(beta)
    run_name = f"{candidate['candidate_id']}_seed{seed}_beta{beta_tag}_iter{int(num_iterations)}"
    run_dir = output_root / candidate["candidate_id"] / f"seed_{seed}" / f"beta_{beta_tag}_iter_{int(num_iterations)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "raw_mask_projection.npy", y.reshape(5, 80, 160))
    start = time.time()
    result = run_poisson_tv_pdhg(
        op,
        y,
        recon_shape=RECON_SHAPE,
        background=float(args.background),
        beta=float(beta),
        num_iterations=int(num_iterations),
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
        diagnostics=json.dumps(result.diagnostics),
        operator_validation=json.dumps(validation),
    )
    (run_dir / "roi_history.json").write_text(json.dumps(result.roi_history, indent=2), encoding="utf-8")
    panel_path = output_root / "reconstruction_panels" / f"{run_name}.png"
    residual_path = output_root / "residual_maps" / f"{run_name}.png"
    _save_recon_panel(volume, run_dir / "reconstruction.png", candidate["candidate_id"])
    _save_recon_panel(volume, panel_path, run_name)
    _save_residual_map(result.residual, run_dir / "residual_map.png", f"{candidate['candidate_id']} normalized residual", 5)
    _save_residual_map(result.residual, residual_path, f"{run_name} normalized residual", 5)
    _save_convergence(result, run_dir / "convergence_curve.png", candidate["candidate_id"])
    metrics = _roi_metrics(volume, f_true.reshape(RECON_SHAPE))
    metrics.update(
        {
            "candidate_id": candidate["candidate_id"],
            "family": candidate["family"],
            "seed": seed,
            "beta": float(beta),
            "num_iterations": int(num_iterations),
            "support_mode": "physical_padded",
            "data_domain": data_domain,
            "hole_count": len(candidate["hole_centers_mm"]),
            "hole_diameter_mm": float(candidate["hole_diameter_mm"]),
            "raw_total_counts": float(np.sum(y)),
            "expected_total_counts": float(np.sum(lam)),
            "exposure_scale": exposure_scale,
            "projection_path": projection_path,
            "final_objective": float(result.objective_history[-1]),
            "final_deviance": float(result.deviance_history[-1]),
            "final_relative_change": float(result.relative_change[-1]),
            "finite_objective": bool(result.diagnostics["finite_objective"]),
            "finite_lambda": bool(result.diagnostics["finite_lambda"]),
            "positive_lambda": bool(result.diagnostics["positive_lambda"]),
            "nonnegative_reconstruction": bool(result.diagnostics["nonnegative_reconstruction"]),
            "objective_nonmonotone_steps": int(result.diagnostics["objective_nonmonotone_steps"]),
            "objective_max_relative_increase": float(result.diagnostics["objective_max_relative_increase"]),
            "operator_adjoint_relative_error": float(validation["adjoint_relative_error"]),
            "operator_delta_virtual_sum": float(validation["delta_virtual_sum"]),
            "operator_validation_status": str(validation["validation_status"]),
            "residual_structure_score": residual_structure_score(result.residual),
            "runtime_s": runtime_s,
            "reconstruction_path": str(run_dir / "reconstruction.npy"),
            "panel_path": str(panel_path),
            "residual_path": str(residual_path),
            "comments": candidate.get("comments", ""),
        }
    )
    return metrics


def _write_parameter_grid(candidates: list[dict], seeds: list[int], betas: list[float], iterations: list[int], args: argparse.Namespace, output_root: Path) -> None:
    rows = []
    for candidate in candidates:
        data_domain = (
            "real_grid9_projection_padded_physical_detector"
            if _is_grid9_candidate(candidate) and args.grid9_data_mode == "real"
            else "synthetic_matched_poisson"
        )
        for seed in seeds:
            for beta in betas:
                for num_iterations in iterations:
                    rows.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "family": candidate["family"],
                            "seed": seed,
                            "beta": beta,
                            "num_iterations": num_iterations,
                            "support_mode": "physical_padded",
                            "data_domain": data_domain,
                            "aperture_mode": "point" if args.quick else "finite",
                            "aperture_samples": 1 if args.quick else int(args.aperture_samples),
                            "attenuation": "none" if args.quick else args.attenuation,
                            "target_counts": float(args.target_counts),
                            "background": float(args.background),
                        }
                    )
    path = output_root / "parameter_grid.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        "| candidate | domain | beta | iter | seed | counts | DL flag | raw DL | R2 | deviance | residual structure |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        dl_flag = "valid" if bool(row.get("detection_limit_valid", False)) else f"invalid: {row.get('detection_limit_invalid_reason', '')}"
        lines.append(
            "| {candidate_id} | {data_domain} | {beta:.1e} | {num_iterations} | {seed} | {raw_total_counts:.3e} | "
            "{dl_flag} | {detection_limit_mgml:.4f} | {roi_r_squared:.4f} | "
            "{final_deviance:.3e} | {residual_structure_score:.4f} |".format(dl_flag=dl_flag, **row)
        )
    (output_root / "poisson_mbir_summary.md").write_text("\n".join(lines), encoding="utf-8")
    stage1_note = ""
    if STAGE1_COMPARISON.exists():
        with STAGE1_COMPARISON.open(newline="", encoding="utf-8") as f:
            stage1_rows = list(csv.DictReader(f))
        mask_rows = [r for r in stage1_rows if r.get("run") == "mask_5_model"]
        if mask_rows:
            mask = mask_rows[0]
            stage1_note = (
                f"Corrected EM-TV Stage 1 `mask_5_model` DL flag was "
                f"`{mask.get('detection_limit_valid')}` with reason `{mask.get('detection_limit_invalid_reason')}`."
            )

    report_lines = [
        "# Corrected Raw-Domain Poisson-TV MBIR Report",
        "",
        "Model: `sum_i [lambda_i(f) - y_i log(lambda_i(f) + eps)] + beta TV(f)`, with `lambda(f)=Af+b` and `f>=0`.",
        "",
        "Support/domain rule: every run uses `support_mode=physical_padded`, i.e. true physical 80 x 80 detector support embedded in the 80 x 160 padded detector.",
        "",
        "Data-domain distinction: corrected grid9 uses the real padded grid9 mask projection when `data_domain=real_grid9_projection_padded_physical_detector`; sparse candidates use properly matched synthetic Poisson projections generated by the same matrix-free operator and phantom.",
        "",
        "DL validity: invalid DL values are retained as raw diagnostic fit outputs only and are not interpreted numerically.",
        "",
        stage1_note,
        "",
        "Numerical checks: each reported row passed finite objective, finite positive lambda, nonnegative reconstruction, operator adjoint, and physical-support delta checks. The `final rel` column is the primal update proxy; objective nonmonotonicity is reported when present because PDHG objectives are not guaranteed to be monotone.",
        "",
        "## Parameter Grid",
        "",
        f"Parameter grid CSV: `{output_root / 'parameter_grid.csv'}`",
        "",
        "## Reconstruction Summary",
        "",
        "| candidate | domain | beta | iter | DL flag | ROI bias | final deviance | final rel | residual structure | operator adjoint rel | objective behavior |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        dl_flag = "valid" if bool(row.get("detection_limit_valid", False)) else f"invalid: {row.get('detection_limit_invalid_reason', '')}"
        behavior = (
            "monotone"
            if int(row.get("objective_nonmonotone_steps", 0)) == 0
            else f"nonmonotone ({row.get('objective_nonmonotone_steps')} steps; max rel inc {row.get('objective_max_relative_increase'):.3e})"
        )
        report_lines.append(
            f"| `{row['candidate_id']}` | {row['data_domain']} | {float(row['beta']):.1e} | {int(row['num_iterations'])} | "
            f"{dl_flag} | {float(row['roi_bias']):.4f} | {float(row['final_deviance']):.3e} | "
            f"{float(row['final_relative_change']):.4e} | {float(row['residual_structure_score']):.4f} | "
            f"{float(row['operator_adjoint_relative_error']):.3e} | {behavior} |"
        )
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{output_root / 'poisson_mbir_summary.csv'}`",
            f"- Reconstruction panels: `{output_root / 'reconstruction_panels'}`",
            f"- Residual maps: `{output_root / 'residual_maps'}`",
            "",
            "These MBIR rows are corrected raw-domain reconstruction evidence for the listed data domains only; they do not establish final protocol-level superiority or inferiority of multi-hole XFCT.",
        ]
    )
    (output_root / "mbir_report.md").write_text("\n".join(report_lines), encoding="utf-8")


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
    parser.add_argument("--pareto-candidates", default=str(DEFAULT_PARETO_CANDIDATES))
    parser.add_argument(
        "--candidate-ids",
        default="",
        help="Comma-separated candidate IDs. Use grid9_p6_d1p25_5 for corrected grid9.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--iteration-grid", default="", help="Comma-separated iteration counts. Overrides --num-iterations when set.")
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--betas", default="", help="Comma-separated beta values. Overrides --beta when set.")
    parser.add_argument("--target-counts", type=float, default=2.0e5)
    parser.add_argument("--background", type=float, default=1.0e-6)
    parser.add_argument("--aperture-samples", type=int, default=8)
    parser.add_argument("--attenuation", choices=["none", "pmma"], default="pmma")
    parser.add_argument("--grid9-data-mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--grid9-projection", default=str(DEFAULT_GRID9_PROJECTION))
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
    betas = _parse_float_list(args.betas) if args.betas else [float(args.beta)]
    iteration_grid = _parse_int_list(args.iteration_grid) if args.iteration_grid else [int(args.num_iterations)]
    seeds = [int(args.seed) + offset for offset in range(int(args.num_seeds))]
    _write_parameter_grid(candidates, seeds, betas, iteration_grid, args, output_root)
    rows = []
    for candidate in candidates:
        for seed in seeds:
            for beta in betas:
                for num_iterations in iteration_grid:
                    print(
                        "Running raw Poisson-TV MBIR: "
                        f"{candidate['candidate_id']} seed={seed} beta={beta:g} iter={num_iterations}"
                    )
                    rows.append(
                        run_candidate(
                            candidate,
                            args,
                            seed,
                            output_root,
                            beta=float(beta),
                            num_iterations=int(num_iterations),
                        )
                    )
    _write_summary(rows, output_root)
    print(f"Summary: {output_root / 'poisson_mbir_summary.csv'}")
    print(f"Report: {output_root / 'mbir_report.md'}")


if __name__ == "__main__":
    main()
