from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

from algorithm.em_tv import load_em_tv_shared_inputs, run_em_tv_reconstruction
from algorithm.rec_pseudo_mbir import run_pseudo_mbir
from algorithm.recon_common import CommonConfig
from src.reporting_figures import (
    _ensure_parent,
    apply_projection_median_filter,
    save_reconstruction_curve_figure,
    save_roi_detection_figure,
)
from src.reporting_roi import scale_reconstruction_to_roi_reference


def _roi_quality_save_fields(roi: dict[str, np.ndarray | float | object]) -> dict[str, object]:
    return {
        "detection_limit_valid": bool(roi.get("detection_limit_valid", False)),
        "detection_limit_invalid": bool(roi.get("detection_limit_invalid", True)),
        "detection_limit_quality": str(roi.get("detection_limit_quality", "invalid")),
        "detection_limit_invalid_reason": str(
            roi.get("detection_limit_invalid_reason", "missing quality check")
        ),
        "cnr_slope": float(roi.get("cnr_slope", np.nan)),
        "cnr_intercept": float(roi.get("cnr_intercept", np.nan)),
        "cnr_monotonic": bool(roi.get("cnr_monotonic", False)),
        "background_mean": float(roi.get("background_mean", np.nan)),
        "background_std": float(roi.get("background_std", np.nan)),
        "quality_min_r_squared": float(roi.get("quality_min_r_squared", np.nan)),
        "quality_background_std_min": float(
            roi.get("quality_background_std_min", np.nan)
        ),
    }


def run_reconstruction_and_save_figure(
    projection: np.ndarray,
    common: CommonConfig,
    image_output_path: str | Path,
    volume_output_path: str | Path | None = None,
    results_output_path: str | Path | None = None,
    projection_median_filter: bool = False,
    projection_median_filter_size: int = 3,
    use_tv: bool = True,
    tv_beta: float = 1.0,
    tv_delta: float = 1.0e-6,
    tv_steps: int = 5,
    tv_warmup: int = 10,
    tv_alpha_mode: str = "auto",
    tv_alpha_manual: float = 0.05,
    tv_alpha_decay: float = 0.0,
    slice_index: int | None = None,
    roi_layout: str = "experimental",
    roi_reference_index: int = 5,
    roi_reference_value_mgml: float = 3.0,
    post_recon_scaling: bool = True,
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    pseudo_mbir_cfg: dict | None = None,
) -> dict[str, np.ndarray | float]:
    projection = np.asarray(projection, dtype=float)
    if projection_median_filter:
        projection = apply_projection_median_filter(
            projection=projection,
            kernel_size=projection_median_filter_size,
        )

    cfg = {} if pseudo_mbir_cfg is None else dict(pseudo_mbir_cfg)
    result = run_pseudo_mbir(
        common=common,
        projection=projection,
        target_angle_count=cfg.get("target_angle_count"),
        target_cij_path=cfg.get("target_cij_path"),
        coarse_iterations=cfg.get("coarse_iterations", "auto"),
        mbir_iterations=cfg.get("mbir_iterations", "remaining"),
        completion_method=str(cfg.get("completion_method", "phase_shift")),
        blend_multiplicative=float(cfg.get("blend_multiplicative", 0.5)),
        uncertainty_floor=float(cfg.get("uncertainty_floor", 1.0)),
        pseudo_weight_scale=float(cfg.get("pseudo_weight_scale", 1.0)),
        pseudo_weight_percentile=float(cfg.get("pseudo_weight_percentile", 95.0)),
        depth_bin_count=int(cfg.get("depth_bin_count", 6)),
        completion_voxel_size=float(cfg.get("completion_voxel_size", 0.5)),
        completion_detector_pixel_size=float(
            cfg.get("completion_detector_pixel_size", 0.25)
        ),
        completion_detector_to_pinhole=float(
            cfg.get("completion_detector_to_pinhole", -30.0)
        ),
        completion_center_to_pinhole=float(
            cfg.get("completion_center_to_pinhole", 50.0)
        ),
        completion_detector_offset_x=float(
            cfg.get("completion_detector_offset_x", -0.5)
        ),
        residual_strength=float(cfg.get("residual_strength", 0.25)),
        pseudo_mu=float(cfg.get("pseudo_mu", 0.2)),
        pseudo_step=float(cfg.get("pseudo_step", 0.5)),
        lambda_l1=float(cfg.get("lambda_l1", 0.0)),
        lambda_flux=float(cfg.get("lambda_flux", 0.0)),
        flux_step=float(cfg.get("flux_step", 0.5)),
        flux_voxel_size=float(cfg.get("flux_voxel_size", 0.5)),
        flux_detector_pixel_size=float(cfg.get("flux_detector_pixel_size", 0.25)),
        flux_detector_to_pinhole=float(cfg.get("flux_detector_to_pinhole", -30.0)),
        flux_center_to_pinhole=float(cfg.get("flux_center_to_pinhole", 50.0)),
        flux_detector_offset_x=float(cfg.get("flux_detector_offset_x", -0.5)),
        flux_use_geometry_weight=bool(cfg.get("flux_use_geometry_weight", True)),
        shuffle_pseudo_projection=bool(cfg.get("shuffle_pseudo_projection", False)),
        pseudo_shuffle_seed=int(cfg.get("pseudo_shuffle_seed", 20260507)),
        wrong_pseudo_angle_shift=int(cfg.get("wrong_pseudo_angle_shift", 0)),
        use_tv=use_tv,
        tv_beta=tv_beta,
        tv_delta=tv_delta,
        tv_steps=tv_steps,
        tv_warmup=tv_warmup,
        tv_alpha_mode=tv_alpha_mode,
        tv_alpha_manual=tv_alpha_manual,
        tv_alpha_decay=tv_alpha_decay,
    )

    reconstruction = np.asarray(result["reconstruction"], dtype=float)
    nll_history = np.asarray(result["nll_history"], dtype=float)
    relative_change = np.asarray(result["relative_change"], dtype=float)
    slice_index = common.slice_index if slice_index is None else int(slice_index)
    scale_factor = 1.0

    if post_recon_scaling:
        reconstruction, scale_factor = scale_reconstruction_to_roi_reference(
            reconstruction=reconstruction,
            slice_index=slice_index,
            recon_size=common.recon_size,
            roi_layout=roi_layout,
            roi_reference_index=roi_reference_index,
            roi_reference_value_mgml=roi_reference_value_mgml,
            roi_xc=roi_xc,
            roi_yc=roi_yc,
        )

    image_output_path = _ensure_parent(image_output_path)
    fig, axis = plt.subplots(1, 1, figsize=(5.8, 5.0))
    slice_img = reconstruction[slice_index, :, :]
    vmax = max(float(roi_reference_value_mgml), 1e-12)
    im = axis.imshow(slice_img, cmap="jet", origin="upper", vmin=0.0, vmax=vmax)
    axis.set_title(
        f"Pseudo MBIR Reconstruction\nslice={slice_index}, iter={common.num_iterations}"
    )
    axis.set_xlabel("Y")
    axis.set_ylabel("X")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="mg/ml")
    fig.tight_layout()
    fig.savefig(image_output_path, dpi=200)
    plt.close(fig)

    curve_output_path = image_output_path.with_name(f"{image_output_path.stem}_curve.png")
    save_reconstruction_curve_figure(
        nll_history=nll_history,
        relative_change=relative_change,
        output_path=curve_output_path,
    )

    roi_output_path = image_output_path.with_name(f"{image_output_path.stem}_roi_dl.png")
    roi = save_roi_detection_figure(
        reconstruction=reconstruction,
        recon_shape=common.recon_size,
        slice_index=slice_index,
        roi_layout=roi_layout,
        roi_reference_value_mgml=roi_reference_value_mgml,
        output_path=roi_output_path,
        roi_xc=roi_xc,
        roi_yc=roi_yc,
    )

    if volume_output_path is not None:
        volume_output_path = _ensure_parent(volume_output_path)
        np.save(volume_output_path, reconstruction)

    if results_output_path is not None:
        results_output_path = _ensure_parent(results_output_path)
        np.savez_compressed(
            results_output_path,
            reconstruction=reconstruction,
            nll_history=nll_history,
            relative_change=relative_change,
            scale_factor=scale_factor,
            slice_index=slice_index,
            CNR=np.asarray(roi["CNR"], dtype=float),
            detection_limit=float(roi["DL"]),
            concentration=np.asarray(roi["concentration"], dtype=float),
            fit_concentration=np.asarray(roi["fit_concentration"], dtype=float),
            fit_cnr=np.asarray(roi["fit_cnr"], dtype=float),
            polyf=np.asarray(roi["polyf"], dtype=float),
            V=np.asarray(roi["V"], dtype=float),
            V2=np.asarray(roi["V2"], dtype=float),
            S=np.asarray(roi["S"], dtype=float),
            S2=np.asarray(roi["S2"], dtype=float),
            roi_xc=np.asarray(roi["xc"], dtype=float),
            roi_yc=np.asarray(roi["yc"], dtype=float),
            roi_radius=np.asarray(roi["radius"], dtype=float),
            roi_r_squared=float(roi["r_squared"]),
            **_roi_quality_save_fields(roi),
            solver_type="pseudo_mbir",
            projection_median_filter=bool(projection_median_filter),
            projection_median_filter_size=int(projection_median_filter_size),
            use_tv=bool(use_tv),
            tv_beta=float(tv_beta),
            tv_delta=float(tv_delta),
            tv_steps=int(tv_steps),
            tv_warmup=int(tv_warmup),
            tv_alpha_mode=str(tv_alpha_mode),
            tv_alpha_manual=float(tv_alpha_manual),
            tv_alpha_decay=float(tv_alpha_decay),
            pseudo_mbir_cfg=str(cfg),
        )

    result["reconstruction"] = reconstruction
    result["scale_factor"] = float(scale_factor)
    result["roi"] = roi
    return result


def run_em_tv_and_save_figure(
    projection: np.ndarray,
    common: CommonConfig,
    image_output_path: str | Path,
    volume_output_path: str | Path | None = None,
    results_output_path: str | Path | None = None,
    projection_median_filter: bool = False,
    projection_median_filter_size: int = 3,
    use_tv: bool = True,
    tv_beta: float = 1.0,
    tv_delta: float = 1.0e-6,
    tv_steps: int = 5,
    tv_warmup: int = 10,
    tv_alpha_mode: str = "auto",
    tv_alpha_manual: float = 0.05,
    tv_alpha_decay: float = 0.0,
    lambda_l1: float = 0.0,
    l1_step: float = 0.5,
    slice_index: int | None = None,
    roi_layout: str = "experimental",
    roi_reference_index: int = 5,
    roi_reference_value_mgml: float = 3.0,
    post_recon_scaling: bool = True,
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    name: str = "baseline_poisson_tv",
) -> dict[str, np.ndarray | float]:
    projection = np.asarray(projection, dtype=float)
    if projection_median_filter:
        projection = apply_projection_median_filter(
            projection=projection,
            kernel_size=projection_median_filter_size,
        )

    shared = load_em_tv_shared_inputs(common, projection=projection)
    result = run_em_tv_reconstruction(
        exp_cfg={
            "name": name,
            "use_tv": bool(use_tv),
            "tv_beta": float(tv_beta),
            "tv_delta": float(tv_delta),
            "tv_steps": int(tv_steps),
            "tv_warmup": int(tv_warmup),
            "tv_alpha_mode": str(tv_alpha_mode),
            "tv_alpha_manual": float(tv_alpha_manual),
            "tv_alpha_decay": float(tv_alpha_decay),
            "lambda_l1": float(lambda_l1),
            "l1_step": float(l1_step),
        },
        common=common,
        shared_inputs=shared,
    )

    reconstruction = np.asarray(result["reconstruction"], dtype=float)
    nll_history = np.asarray(result["nll_history"], dtype=float)
    relative_change = np.asarray(result["relative_change"], dtype=float)
    slice_index = common.slice_index if slice_index is None else int(slice_index)
    scale_factor = 1.0

    if post_recon_scaling:
        reconstruction, scale_factor = scale_reconstruction_to_roi_reference(
            reconstruction=reconstruction,
            slice_index=slice_index,
            recon_size=common.recon_size,
            roi_layout=roi_layout,
            roi_reference_index=roi_reference_index,
            roi_reference_value_mgml=roi_reference_value_mgml,
            roi_xc=roi_xc,
            roi_yc=roi_yc,
        )

    image_output_path = _ensure_parent(image_output_path)
    fig, axis = plt.subplots(1, 1, figsize=(5.8, 5.0))
    slice_img = reconstruction[slice_index, :, :]
    vmax = max(float(roi_reference_value_mgml), 1e-12)
    im = axis.imshow(slice_img, cmap="jet", origin="upper", vmin=0.0, vmax=vmax)
    axis.set_title(f"{name}\nslice={slice_index}, iter={common.num_iterations}")
    axis.set_xlabel("Y")
    axis.set_ylabel("X")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="mg/ml")
    fig.tight_layout()
    fig.savefig(image_output_path, dpi=200)
    plt.close(fig)

    curve_output_path = image_output_path.with_name(f"{image_output_path.stem}_curve.png")
    save_reconstruction_curve_figure(
        nll_history=nll_history,
        relative_change=relative_change,
        output_path=curve_output_path,
    )

    roi_output_path = image_output_path.with_name(f"{image_output_path.stem}_roi_dl.png")
    roi = save_roi_detection_figure(
        reconstruction=reconstruction,
        recon_shape=common.recon_size,
        slice_index=slice_index,
        roi_layout=roi_layout,
        roi_reference_value_mgml=roi_reference_value_mgml,
        output_path=roi_output_path,
        roi_xc=roi_xc,
        roi_yc=roi_yc,
    )

    if volume_output_path is not None:
        volume_output_path = _ensure_parent(volume_output_path)
        np.save(volume_output_path, reconstruction)

    if results_output_path is not None:
        results_output_path = _ensure_parent(results_output_path)
        np.savez_compressed(
            results_output_path,
            reconstruction=reconstruction,
            nll_history=nll_history,
            relative_change=relative_change,
            scale_factor=scale_factor,
            slice_index=slice_index,
            CNR=np.asarray(roi["CNR"], dtype=float),
            detection_limit=float(roi["DL"]),
            concentration=np.asarray(roi["concentration"], dtype=float),
            fit_concentration=np.asarray(roi["fit_concentration"], dtype=float),
            fit_cnr=np.asarray(roi["fit_cnr"], dtype=float),
            polyf=np.asarray(roi["polyf"], dtype=float),
            V=np.asarray(roi["V"], dtype=float),
            V2=np.asarray(roi["V2"], dtype=float),
            S=np.asarray(roi["S"], dtype=float),
            S2=np.asarray(roi["S2"], dtype=float),
            roi_xc=np.asarray(roi["xc"], dtype=float),
            roi_yc=np.asarray(roi["yc"], dtype=float),
            roi_radius=np.asarray(roi["radius"], dtype=float),
            solver_type=str(name),
            projection_median_filter=bool(projection_median_filter),
            projection_median_filter_size=int(projection_median_filter_size),
            use_tv=bool(use_tv),
            tv_beta=float(tv_beta),
            tv_delta=float(tv_delta),
            tv_steps=int(tv_steps),
            tv_warmup=int(tv_warmup),
            tv_alpha_mode=str(tv_alpha_mode),
            tv_alpha_manual=float(tv_alpha_manual),
            tv_alpha_decay=float(tv_alpha_decay),
            lambda_l1=float(lambda_l1),
            l1_step=float(l1_step),
            roi_r_squared=float(roi["r_squared"]),
            **_roi_quality_save_fields(roi),
        )

    result["reconstruction"] = reconstruction
    result["scale_factor"] = float(scale_factor)
    result["roi"] = roi
    return result
