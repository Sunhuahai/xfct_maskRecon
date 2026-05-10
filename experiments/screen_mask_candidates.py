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

from src.mask_xfct_model import (
    EPS,
    XFCTForwardConfig,
    XFCTMaskOperator,
    candidate_to_mask_config,
    load_candidate_json,
    make_lumpy_phantom,
    make_roi_detection_phantom,
    physical_support_columns,
    roi_task_vectors,
)


RECON_SHAPE = (40, 60, 60)
DET_Z = 80
DET_X = 160
PAD_X = 40
PHYSICAL_X = 80
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "data" / "masks" / "candidates"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mask_design"


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _angle_sets(quick: bool) -> dict[str, tuple[int, ...]]:
    sets = {"phase0_default": (0, 9, 18, 27, 36)}
    phase_count = 4 if quick else 9
    for phase in range(1, phase_count + 1):
        sets[f"phase{phase}"] = tuple((phase + 9 * k) % 45 for k in range(5))
    return sets


def _load_candidates(candidate_dir: Path, candidate_limit: int | None) -> list[dict]:
    paths = sorted(candidate_dir.glob("*.json"))
    if candidate_limit is not None:
        paths = paths[: int(candidate_limit)]
    return [load_candidate_json(path) | {"json_path": str(path)} for path in paths]


def _roi_union_indices() -> np.ndarray:
    tasks = roi_task_vectors(RECON_SHAPE, slice_index=20)
    union = np.zeros(np.prod(RECON_SHAPE), dtype=bool)
    for task in tasks.values():
        union |= np.asarray(task).reshape(-1) > 0.0
    return np.flatnonzero(union)


def _subsample_indices(rng: np.random.Generator, source: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(source, dtype=np.int64).reshape(-1)
    if source.size <= count:
        return source
    return np.sort(rng.choice(source, size=int(count), replace=False))


def _sensitivity_metrics(op: XFCTMaskOperator) -> tuple[np.ndarray, dict[str, float]]:
    sens = op.sensitivity("physical_padded")
    support = sens > 0.0
    positive = sens[support]
    if positive.size == 0:
        metrics = {
            "sensitivity_mean": 0.0,
            "sensitivity_cv": float("inf"),
            "sensitivity_min_over_mean": 0.0,
            "active_voxel_fraction": 0.0,
        }
    else:
        metrics = {
            "sensitivity_mean": float(np.mean(positive)),
            "sensitivity_cv": float(np.std(positive) / max(np.mean(positive), EPS)),
            "sensitivity_min_over_mean": float(np.min(positive) / max(np.mean(positive), EPS)),
            "active_voxel_fraction": float(np.mean(support)),
        }
    roi_indices = _roi_union_indices()
    roi_sens = sens[roi_indices]
    metrics["roi_sensitivity_mean"] = float(np.mean(roi_sens))
    metrics["roi_sensitivity_cv"] = float(np.std(roi_sens) / max(np.mean(roi_sens), EPS))
    metrics["roi_active_voxel_fraction"] = float(np.mean(roi_sens > 0.0))
    return sens, metrics


def _truncation_metrics(op: XFCTMaskOperator, rng: np.random.Generator, quick: bool) -> tuple[np.ndarray, dict[str, float]]:
    global_count = 800 if quick else 3000
    all_indices = np.arange(np.prod(RECON_SHAPE), dtype=np.int64)
    global_indices = _subsample_indices(rng, all_indices, global_count)
    roi_indices = _subsample_indices(rng, _roi_union_indices(), min(300 if quick else 1200, _roi_union_indices().size))
    phys_global = op.truncation_fraction(global_indices, physical=True)
    pad_global = op.truncation_fraction(global_indices, physical=False)
    phys_roi = op.truncation_fraction(roi_indices, physical=True)
    pad_roi = op.truncation_fraction(roi_indices, physical=False)
    trunc_map = np.zeros(np.prod(RECON_SHAPE), dtype=np.float64)
    trunc_map[global_indices] = phys_global
    return trunc_map, {
        "global_truncation_physical_mean": float(np.mean(phys_global)),
        "global_truncation_physical_p95": float(np.percentile(phys_global, 95)),
        "global_truncation_padded_mean": float(np.mean(pad_global)),
        "roi_truncation_physical_mean": float(np.mean(phys_roi)),
        "roi_truncation_physical_p95": float(np.percentile(phys_roi, 95)),
        "roi_truncation_padded_mean": float(np.mean(pad_roi)),
    }


def _overlap_metrics(op: XFCTMaskOperator, phantom_flat: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    n_holes = op.hole_centers_mm.shape[0]
    if n_holes <= 1:
        return np.eye(n_holes), {
            "overlap_mean": 0.0,
            "overlap_max": 0.0,
            "overlap_high_fraction": 0.0,
        }
    footprints = []
    for hole_idx in range(n_holes):
        fp = op.forward(phantom_flat, support_mode="physical_padded", hole_indices=[hole_idx])
        norm = np.linalg.norm(fp)
        footprints.append(fp / max(norm, EPS))
    matrix = np.abs(np.asarray(footprints) @ np.asarray(footprints).T)
    offdiag = matrix[~np.eye(n_holes, dtype=bool)]
    return matrix, {
        "overlap_mean": float(np.mean(offdiag)),
        "overlap_max": float(np.max(offdiag)),
        "overlap_high_fraction": float(np.mean(offdiag > 0.5)),
    }


def _column_matrix(op: XFCTMaskOperator, voxel_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    columns = []
    for voxel_idx in voxel_indices:
        col = op.forward_delta(int(voxel_idx), support_mode="physical_padded") * sqrt_w
        norm = np.linalg.norm(col)
        if norm > EPS:
            columns.append(col / norm)
    if not columns:
        return np.zeros((0, weights.size), dtype=np.float32)
    return np.asarray(columns, dtype=np.float32)


def _coherence_metrics(op: XFCTMaskOperator, lambda0: np.ndarray, rng: np.random.Generator, quick: bool) -> dict[str, float]:
    count_global = 40 if quick else 120
    count_roi = 24 if quick else 80
    all_indices = np.arange(np.prod(RECON_SHAPE), dtype=np.int64)
    global_indices = _subsample_indices(rng, all_indices, count_global)
    roi_indices = _subsample_indices(rng, _roi_union_indices(), count_roi)
    weights = 1.0 / np.maximum(lambda0, 1.0e-6)

    def corr_stats(indices: np.ndarray, prefix: str) -> dict[str, float]:
        cols = _column_matrix(op, indices, weights)
        if cols.shape[0] <= 1:
            return {f"{prefix}_coherence_max": 0.0, f"{prefix}_coherence_p95": 0.0}
        corr = np.abs(cols @ cols.T)
        corr[np.eye(corr.shape[0], dtype=bool)] = 0.0
        values = corr[~np.eye(corr.shape[0], dtype=bool)]
        return {
            f"{prefix}_coherence_max": float(np.max(values)),
            f"{prefix}_coherence_p95": float(np.percentile(values, 95)),
        }

    metrics = corr_stats(global_indices, "global")
    metrics.update(corr_stats(roi_indices, "roi"))
    return metrics


def _fisher_metrics(op: XFCTMaskOperator, lambda0: np.ndarray) -> dict[str, float]:
    weights = 1.0 / np.maximum(lambda0, 1.0e-6)
    tasks = roi_task_vectors(RECON_SHAPE, slice_index=20)
    d2_values = []
    crlb_values = []
    for name, task in tasks.items():
        if name == "roi_0":
            continue
        a_s = op.forward(task, support_mode="physical_padded")
        d2 = float(np.sum(a_s * a_s * weights))
        d2_values.append(d2)
        crlb_values.append(1.0 / max(d2, EPS))
    return {
        "task_fisher_d2_mean": float(np.mean(d2_values)),
        "task_fisher_d2_min": float(np.min(d2_values)),
        "task_crlb_mean": float(np.mean(crlb_values)),
        "task_crlb_max": float(np.max(crlb_values)),
    }


def _ranking_score(row: dict[str, float]) -> float:
    return float(
        np.log1p(max(row["task_fisher_d2_mean"], 0.0))
        + 0.15 * np.log1p(max(row["throughput_detection_phantom"], 0.0))
        - 1.8 * row["roi_truncation_physical_mean"]
        - 1.2 * row["global_truncation_physical_mean"]
        - 0.9 * row["overlap_max"]
        - 0.5 * row["roi_coherence_p95"]
        - 0.35 * row["sensitivity_cv"]
        - 0.35 * row["roi_sensitivity_cv"]
    )


def screen_one(
    candidate: dict,
    angle_name: str,
    angle_indices: tuple[int, ...],
    args: argparse.Namespace,
    phantoms: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    centers, diameter = candidate_to_mask_config(candidate)
    op = XFCTMaskOperator(
        XFCTForwardConfig(
            hole_centers_mm=centers,
            hole_diameter_mm=diameter,
            angle_indices=tuple(int(v) for v in angle_indices),
            aperture_mode="point" if args.quick else "finite",
            aperture_samples=1 if args.quick else int(args.aperture_samples),
            attenuation="none" if args.quick else args.attenuation,
        )
    )
    f_detection = phantoms["detection"].reshape(-1)
    f_lumpy = phantoms["lumpy"].reshape(-1)
    f_heldout = phantoms["heldout"].reshape(-1)
    lam_detection = op.forward(f_detection, support_mode="physical_padded") + float(args.background)
    lam_lumpy = op.forward(f_lumpy, support_mode="physical_padded") + float(args.background)
    lam_heldout = op.forward(f_heldout, support_mode="physical_padded") + float(args.background)
    sensitivity, sens_metrics = _sensitivity_metrics(op)
    trunc_map, trunc_metrics = _truncation_metrics(op, rng, quick=bool(args.quick))
    overlap_matrix, overlap_metrics = _overlap_metrics(op, f_detection)
    coherence_metrics = _coherence_metrics(op, lam_lumpy, rng, quick=bool(args.quick))
    fisher_metrics = _fisher_metrics(op, lam_lumpy)

    row: dict[str, float | str | int] = {
        "candidate_id": candidate["candidate_id"],
        "family": candidate["family"],
        "hole_count": len(candidate["hole_centers_mm"]),
        "hole_diameter_mm": float(candidate["hole_diameter_mm"]),
        "min_distance_mm": float(candidate["min_distance_mm"]),
        "total_open_area_mm2": float(candidate["total_open_area_mm2"]),
        "angle_set": angle_name,
        "angle_indices": ",".join(str(v) for v in angle_indices),
        "throughput_detection_phantom": float(np.sum(lam_detection)),
        "throughput_lumpy_phantom": float(np.sum(lam_lumpy)),
        "throughput_heldout_phantom": float(np.sum(lam_heldout)),
        "comments": candidate.get("comments", ""),
        "json_path": candidate.get("json_path", ""),
    }
    row.update(sens_metrics)
    row.update(trunc_metrics)
    row.update(overlap_metrics)
    row.update(coherence_metrics)
    row.update(fisher_metrics)
    row["ranking_score"] = _ranking_score(row)  # type: ignore[arg-type]
    artifacts = {
        "sensitivity": sensitivity.reshape(RECON_SHAPE),
        "truncation": trunc_map.reshape(RECON_SHAPE),
        "overlap": overlap_matrix,
        "lambda_detection": lam_detection.reshape(len(angle_indices), DET_Z, DET_X),
    }
    return row, artifacts


def _save_sensitivity_plot(sensitivity: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    central = sensitivity[20]
    mip = np.max(sensitivity, axis=0)
    for axis, image, label in zip(axes, [central, mip], ["central slice", "z max"]):
        im = axis.imshow(image, cmap="viridis", origin="upper")
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_truncation_plot(truncation: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(5.4, 4.6))
    im = axis.imshow(truncation[20], cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="physical truncation fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_overlap_plot(overlap: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(5.0, 4.5))
    im = axis.imshow(overlap, cmap="magma", origin="upper", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("hole")
    axis.set_ylabel("hole")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_ranking_plots(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    default_rows = [r for r in rows if r["angle_set"] == "phase0_default"] or rows
    labels = [str(r["candidate_id"]) for r in default_rows]
    scores = np.array([float(r["ranking_score"]) for r in default_rows])
    order = np.argsort(scores)[::-1][: min(20, len(scores))]
    fig, axis = plt.subplots(1, 1, figsize=(10.5, 5.2))
    axis.bar(np.arange(order.size), scores[order], color="tab:blue")
    axis.set_xticks(np.arange(order.size))
    axis.set_xticklabels([labels[i] for i in order], rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("combined ranking score")
    axis.set_title("Task Fisher / penalty ranking")
    fig.tight_layout()
    fig.savefig(output_root / "fisher_crlb_ranking.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(7.0, 5.0))
    x = [float(r["throughput_detection_phantom"]) for r in default_rows]
    y = [float(r["task_fisher_d2_mean"]) for r in default_rows]
    c = [float(r["global_truncation_physical_mean"]) for r in default_rows]
    scatter = axis.scatter(x, y, c=c, cmap="viridis", s=36, alpha=0.85)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("expected counts: detection phantom")
    axis.set_ylabel("mean task Fisher d2")
    axis.set_title("Throughput vs task detectability")
    fig.colorbar(scatter, ax=axis, label="mean physical truncation")
    fig.tight_layout()
    fig.savefig(output_root / "throughput_vs_task_detectability.png", dpi=180)
    plt.close(fig)


def _write_outputs(rows: list[dict], top_payload: dict, output_root: Path) -> None:
    csv_path = output_root / "candidate_screening.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output_root / "top_candidates.json").write_text(
        json.dumps(top_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _validation_warning() -> str:
    summary_path = PROJECT_ROOT / "results" / "forward_model_validation" / "validation_summary.json"
    if not summary_path.exists():
        return "Forward validation summary not found; screening rankings are provisional."
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Forward validation summary could not be parsed; screening rankings are provisional."
    if payload.get("overall_status") != "PASS":
        return "Forward validation did not pass; screening rankings are provisional and should not drive final mask selection."
    return "Forward validation passed."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen sparse multi-pinhole XFCT masks with task-based information metrics.")
    parser.add_argument("--quick", action="store_true", help="Use point-pinhole, reduced voxel subsampling, and fewer angle phases.")
    parser.add_argument("--final", action="store_true", help="Use finite aperture and full configured candidate set.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Accepted for workflow compatibility.")
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--protocols", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--recon-methods", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="matrix_free")
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--aperture-samples", type=int, default=8)
    parser.add_argument("--attenuation", choices=["none", "pmma"], default="pmma")
    parser.add_argument("--background", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20260509)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(Path(args.candidate_dir), args.candidate_limit)
    if not candidates:
        raise FileNotFoundError(
            f"No candidate JSON files found under {args.candidate_dir}. "
            "Run scripts/generate_mask_candidates.py first."
        )
    rng = np.random.default_rng(int(args.seed))
    phantoms = {
        "detection": make_roi_detection_phantom(RECON_SHAPE),
        "lumpy": make_lumpy_phantom(RECON_SHAPE, seed=int(args.seed), n_lumps=40 if args.quick else 100),
        "heldout": make_lumpy_phantom(RECON_SHAPE, seed=int(args.seed) + 99, n_lumps=45 if args.quick else 120),
    }
    rows: list[dict] = []
    artifacts_by_key: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    angle_sets = _angle_sets(bool(args.quick))
    warning = _validation_warning()
    print(warning)
    for cand_idx, candidate in enumerate(candidates, start=1):
        print(f"[{cand_idx}/{len(candidates)}] screening {candidate['candidate_id']}")
        for angle_name, angle_indices in angle_sets.items():
            row, artifacts = screen_one(candidate, angle_name, angle_indices, args, phantoms, rng)
            rows.append(row)
            if angle_name == "phase0_default":
                artifacts_by_key[(candidate["candidate_id"], angle_name)] = artifacts
    default_rows = [r for r in rows if r["angle_set"] == "phase0_default"]
    top_rows = sorted(default_rows, key=lambda r: float(r["ranking_score"]), reverse=True)
    top_payload = {
        "validation_warning": warning,
        "ranking_note": (
            "Ranking combines task Fisher information with penalties for physical-detector truncation, "
            "hole overlap, weighted coherence, and sensitivity nonuniformity. It is not a throughput-only score."
        ),
        "top_candidates": top_rows[:10],
        "top_by_family": {},
        "top_by_hole_count": {},
    }
    for row in top_rows:
        top_payload["top_by_family"].setdefault(row["family"], row)
        top_payload["top_by_hole_count"].setdefault(str(row["hole_count"]), row)
    _write_outputs(rows, top_payload, output_root)

    for rank, row in enumerate(top_rows[: min(5, len(top_rows))], start=1):
        key = (row["candidate_id"], "phase0_default")
        artifacts = artifacts_by_key.get(key)
        if artifacts is None:
            continue
        stem = f"rank{rank}_{row['candidate_id']}"
        _save_sensitivity_plot(
            artifacts["sensitivity"],
            output_root / f"sensitivity_{stem}.png",
            f"{row['candidate_id']} sensitivity",
        )
        _save_truncation_plot(
            artifacts["truncation"],
            output_root / f"truncation_{stem}.png",
            f"{row['candidate_id']} physical truncation",
        )
        _save_overlap_plot(
            artifacts["overlap"],
            output_root / f"overlap_{stem}.png",
            f"{row['candidate_id']} isolated-hole footprint overlap",
        )
    _save_ranking_plots(rows, output_root)
    print(f"Screened {len(candidates)} candidates across {len(angle_sets)} angle sets.")
    print(f"CSV: {output_root / 'candidate_screening.csv'}")
    print(f"Top candidates: {output_root / 'top_candidates.json'}")


if __name__ == "__main__":
    main()
