from __future__ import annotations

from typing import Any

import numpy as np

from src.flux_projector import voxel_center_xyz
from src.lifted_projective_dynamics import EPS
from src.moment_closure import (
    compute_dimensionless_depth_metrics,
    compute_moments_mean_var,
)
from src.xfct_geometry import PinholeGeometry, detector_continuous_indices, uv


def _geometry_or_default(geometry: PinholeGeometry | None) -> PinholeGeometry:
    return PinholeGeometry() if geometry is None else geometry


def _projector_voxel_grid(x0: np.ndarray, projector: Any) -> np.ndarray:
    if hasattr(projector, "voxel_grid"):
        grid = np.asarray(projector.voxel_grid, dtype=np.float64)
        if grid.shape == tuple(x0.shape) + (3,):
            return grid
        if grid.ndim == 2 and grid.shape[1] == 3 and grid.shape[0] == x0.size:
            return grid.reshape(tuple(x0.shape) + (3,))
    voxel_size = float(getattr(projector, "voxel_size", 0.5))
    return voxel_center_xyz(tuple(x0.shape), voxel_size).reshape(tuple(x0.shape) + (3,))


def _detector_pixel_size(projector: Any, default: float = 0.25) -> float:
    return float(getattr(projector, "detector_pixel_size", default))


def _forward(projector: Any, volume: np.ndarray, theta: float) -> np.ndarray:
    if hasattr(projector, "forward"):
        projected = projector.forward(np.asarray(volume, dtype=np.float64), float(theta))
    elif callable(projector):
        projected = projector(np.asarray(volume, dtype=np.float64), float(theta))
    else:
        raise TypeError("projector must expose forward(volume, theta) or be callable.")
    return np.nan_to_num(
        np.asarray(projected, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _safe_nrmse(prediction: np.ndarray, truth: np.ndarray, eps: float = EPS) -> float:
    pred = np.nan_to_num(np.asarray(prediction, dtype=np.float64), nan=0.0)
    ref = np.nan_to_num(np.asarray(truth, dtype=np.float64), nan=0.0)
    return float(np.linalg.norm((pred - ref).ravel()) / (np.linalg.norm(ref.ravel()) + eps))


def _safe_ratio(numerator: float, denominator: float, eps: float = EPS) -> float:
    return float(numerator / (denominator + eps))


def _robust_variance(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    q05, q95 = np.percentile(finite, [5.0, 95.0])
    central = finite[(finite >= q05) & (finite <= q95)]
    if central.size == 0:
        return 0.0
    return float(np.var(central))


def _bilinear_sample_flat(
    image: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    sampled = np.zeros(row.shape, dtype=np.float64)
    if not np.any(valid):
        return sampled

    row_valid = np.clip(row[valid], 0.0, float(height - 1))
    col_valid = np.clip(col[valid], 0.0, float(width - 1))
    row0 = np.floor(row_valid).astype(np.int64)
    col0 = np.floor(col_valid).astype(np.int64)
    row1 = np.minimum(row0 + 1, height - 1)
    col1 = np.minimum(col0 + 1, width - 1)
    wr = row_valid - row0
    wc = col_valid - col0

    values = (
        (1.0 - wr) * (1.0 - wc) * image[row0, col0]
        + (1.0 - wr) * wc * image[row0, col1]
        + wr * (1.0 - wc) * image[row1, col0]
        + wr * wc * image[row1, col1]
    )
    sampled[valid] = values
    return sampled


def sample_detector_image_at_voxels(
    image: np.ndarray,
    theta: float,
    voxel_grid: np.ndarray,
    geometry: PinholeGeometry | None,
    mode: str = "bilinear",
    detector_pixel_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a detector image at each voxel's source-angle detector coordinate.

    ``voxel_grid`` must contain voxel centers with last dimension ``(x, y, z)``.
    The returned arrays have ``voxel_grid.shape[:-1]``.
    """
    if str(mode).lower() != "bilinear":
        raise ValueError("only bilinear detector sampling is currently supported.")
    img = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0)
    if img.ndim != 2:
        raise ValueError(f"image must be a 2-D detector image, got {img.shape}.")
    if detector_pixel_size <= 0.0:
        raise ValueError("detector_pixel_size must be positive.")

    grid = np.asarray(voxel_grid, dtype=np.float64)
    if grid.shape[-1] != 3:
        raise ValueError(f"voxel_grid last dimension must be 3, got {grid.shape}.")
    output_shape = grid.shape[:-1]
    xyz = grid.reshape(-1, 3)
    geom = _geometry_or_default(geometry)
    u_coord, v_coord = uv(float(theta), xyz, geom)
    row, col = detector_continuous_indices(
        u_coord,
        v_coord,
        detector_shape=tuple(img.shape),
        detector_pixel_size=float(detector_pixel_size),
    )
    height, width = img.shape
    valid = (
        np.isfinite(row)
        & np.isfinite(col)
        & (row >= 0.0)
        & (row <= float(height - 1))
        & (col >= 0.0)
        & (col <= float(width - 1))
    )
    sampled = _bilinear_sample_flat(img, row, col, valid)
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    return sampled.reshape(output_shape), valid.reshape(output_shape)


def compute_residual_ratio_map(
    y_meas: np.ndarray,
    p_pred: np.ndarray,
    eps: float = EPS,
    clip: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return multiplicative residual ratio ``(y+eps)/(p+eps)`` and diagnostics."""
    y = np.nan_to_num(np.asarray(y_meas, dtype=np.float64), nan=0.0)
    p = np.nan_to_num(np.asarray(p_pred, dtype=np.float64), nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_raw = (y + float(eps)) / (p + float(eps))
    ratio_raw = np.nan_to_num(ratio_raw, nan=1.0, posinf=1.0, neginf=1.0)

    clip_fraction = 0.0
    ratio = ratio_raw
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        if lo <= 0.0 or hi <= lo:
            raise ValueError("clip must satisfy 0 < clip_min < clip_max.")
        finite = np.isfinite(ratio_raw)
        clipped = finite & ((ratio_raw < lo) | (ratio_raw > hi))
        clip_fraction = float(np.mean(clipped)) if ratio_raw.size else 0.0
        ratio = np.clip(ratio_raw, lo, hi)

    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    percentiles = np.percentile(ratio, [1.0, 5.0, 50.0, 95.0, 99.0])
    diagnostics = {
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_median": float(percentiles[2]),
        "ratio_p01": float(percentiles[0]),
        "ratio_p05": float(percentiles[1]),
        "ratio_q05": float(percentiles[1]),
        "ratio_q95": float(percentiles[3]),
        "ratio_p95": float(percentiles[3]),
        "ratio_p99": float(percentiles[4]),
        "ratio_dynamic_range": float(percentiles[3] - percentiles[1]),
        "robust_ratio_variance": _robust_variance(ratio),
        "ratio_clip_fraction": clip_fraction,
    }
    return ratio, diagnostics


def compute_additive_residual_map(
    y_meas: np.ndarray,
    p_pred: np.ndarray,
) -> np.ndarray:
    """Return additive source residual ``y_meas - p_pred`` with finite values."""
    return np.nan_to_num(
        np.asarray(y_meas, dtype=np.float64) - np.asarray(p_pred, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _pullback_detector_image(
    image: np.ndarray,
    theta: float,
    volume: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    pullback_mode: str,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(pullback_mode).lower()
    voxel_grid = _projector_voxel_grid(volume, projector)
    detector_pixel_size = _detector_pixel_size(projector)
    if mode == "geometry":
        values, valid = sample_detector_image_at_voxels(
            image,
            float(theta),
            voxel_grid,
            geometry,
            detector_pixel_size=detector_pixel_size,
        )
        return values, valid, {
            "pullback_mode": "geometry",
            "valid_voxel_fraction": float(np.mean(valid)) if valid.size else 0.0,
        }
    if mode == "system_matrix":
        from src.system_matrix_residual_pullback import residual_pullback_system_matrix

        return residual_pullback_system_matrix(
            image,
            projector=projector,
            theta=float(theta),
            volume_shape=tuple(volume.shape),
            detector_shape=tuple(np.asarray(image).shape),
            eps=eps,
            allow_fallback=False,
            voxel_grid=voxel_grid,
            geometry=geometry,
            detector_pixel_size=detector_pixel_size,
        )
    raise ValueError("pullback_mode must be 'geometry' or 'system_matrix'.")


def voxel_lifted_multiplicative_transfer(
    x0: np.ndarray,
    theta_source: float,
    theta_target: float,
    y_source: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    eps: float = EPS,
    ratio_clip: tuple[float, float] | None = (0.2, 5.0),
    pullback_mode: str = "geometry",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer measured detector residual ratios through the voxel density."""
    volume = np.nan_to_num(np.asarray(x0, dtype=np.float64), nan=0.0)
    p_source = _forward(projector, volume, float(theta_source))
    ratio_source, ratio_diag = compute_residual_ratio_map(
        y_source,
        p_source,
        eps=eps,
        clip=ratio_clip,
    )
    ratio_voxels, valid_mask, pullback_diag = _pullback_detector_image(
        ratio_source,
        float(theta_source),
        volume,
        projector,
        geometry,
        pullback_mode=pullback_mode,
        eps=eps,
    )
    ratio_voxels = np.where(valid_mask, ratio_voxels, 1.0)
    ratio_voxels = np.nan_to_num(ratio_voxels, nan=1.0, posinf=1.0, neginf=1.0)
    x_mod = volume * ratio_voxels
    source_self = _forward(projector, x_mod, float(theta_source))
    y_hat_raw = _forward(projector, x_mod, float(theta_target))
    y_hat = np.maximum(np.nan_to_num(y_hat_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    x_mass = float(np.sum(volume))
    diagnostics: dict[str, Any] = {
        **ratio_diag,
        **pullback_diag,
        "source_pred_nrmse": _safe_nrmse(p_source, y_source, eps=eps),
        "source_self_fit_error": _safe_nrmse(source_self, y_source, eps=eps),
        "valid_voxel_fraction": float(np.mean(valid_mask)) if valid_mask.size else 0.0,
        "x_mod_total_mass_ratio": _safe_ratio(float(np.sum(x_mod)), x_mass, eps=eps),
        "p_source": p_source,
        "source_self_projection": source_self,
        "ratio_map": ratio_source,
        "ratio_voxels": ratio_voxels,
        "valid_voxel_mask": valid_mask,
    }
    return y_hat, diagnostics


def voxel_lifted_additive_transfer(
    x0: np.ndarray,
    theta_source: float,
    theta_target: float,
    y_source: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    eps: float = EPS,
    clip_nonnegative: bool = True,
    pullback_mode: str = "geometry",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer normalized additive detector residuals through the voxel density."""
    volume = np.nan_to_num(np.asarray(x0, dtype=np.float64), nan=0.0)
    p_source = _forward(projector, volume, float(theta_source))
    residual = compute_additive_residual_map(y_source, p_source)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = residual / (p_source + float(eps))
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    sampled, valid_mask, pullback_diag = _pullback_detector_image(
        normalized,
        float(theta_source),
        volume,
        projector,
        geometry,
        pullback_mode=pullback_mode,
        eps=eps,
    )
    sampled = np.where(valid_mask, sampled, 0.0)
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    x_corr = volume * sampled
    x_add = volume + x_corr
    source_self = _forward(projector, x_add, float(theta_source))
    y_base = _forward(projector, volume, float(theta_target))
    y_corr = _forward(projector, x_corr, float(theta_target))
    y_hat_raw = np.nan_to_num(y_base + y_corr, nan=0.0, posinf=0.0, neginf=0.0)
    negative_fraction = float(np.mean(y_hat_raw < 0.0)) if y_hat_raw.size else 0.0
    y_hat = np.maximum(y_hat_raw, 0.0) if clip_nonnegative else y_hat_raw

    x_mass = float(np.sum(volume))
    percentiles = np.percentile(normalized, [1.0, 5.0, 50.0, 95.0, 99.0])
    diagnostics: dict[str, Any] = {
        **pullback_diag,
        "source_pred_nrmse": _safe_nrmse(p_source, y_source, eps=eps),
        "source_self_fit_error": _safe_nrmse(source_self, y_source, eps=eps),
        "valid_voxel_fraction": float(np.mean(valid_mask)) if valid_mask.size else 0.0,
        "x_corr_total_mass_ratio": _safe_ratio(float(np.sum(x_corr)), x_mass, eps=eps),
        "negative_projection_fraction": negative_fraction,
        "normalized_residual_min": float(np.min(normalized)),
        "normalized_residual_max": float(np.max(normalized)),
        "normalized_residual_median": float(percentiles[2]),
        "normalized_residual_p01": float(percentiles[0]),
        "normalized_residual_p05": float(percentiles[1]),
        "normalized_residual_q05": float(percentiles[1]),
        "normalized_residual_q95": float(percentiles[3]),
        "normalized_residual_p95": float(percentiles[3]),
        "normalized_residual_p99": float(percentiles[4]),
        "normalized_residual_dynamic_range": float(percentiles[3] - percentiles[1]),
        "robust_ratio_variance": _robust_variance(normalized),
        "p_source": p_source,
        "source_self_projection": source_self,
        "additive_residual_map": residual,
        "normalized_residual_map": normalized,
        "sampled_normalized_residual": sampled,
        "valid_voxel_mask": valid_mask,
    }
    return y_hat, diagnostics


def _single_endpoint_transfer(
    x0: np.ndarray,
    theta_source: float,
    theta_target: float,
    y_source: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    mode: str,
    hybrid_alpha: float,
    eps: float,
    ratio_clip: tuple[float, float] | None,
    clip_nonnegative: bool,
    pullback_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_l = str(mode).lower()
    if mode_l == "multiplicative":
        return voxel_lifted_multiplicative_transfer(
            x0,
            theta_source,
            theta_target,
            y_source,
            projector,
            geometry,
            eps=eps,
            ratio_clip=ratio_clip,
            pullback_mode=pullback_mode,
        )
    if mode_l == "additive":
        return voxel_lifted_additive_transfer(
            x0,
            theta_source,
            theta_target,
            y_source,
            projector,
            geometry,
            eps=eps,
            clip_nonnegative=clip_nonnegative,
            pullback_mode=pullback_mode,
        )
    if mode_l == "hybrid":
        pred_mul, diag_mul = voxel_lifted_multiplicative_transfer(
            x0,
            theta_source,
            theta_target,
            y_source,
            projector,
            geometry,
            eps=eps,
            ratio_clip=ratio_clip,
            pullback_mode=pullback_mode,
        )
        pred_add, diag_add = voxel_lifted_additive_transfer(
            x0,
            theta_source,
            theta_target,
            y_source,
            projector,
            geometry,
            eps=eps,
            clip_nonnegative=clip_nonnegative,
            pullback_mode=pullback_mode,
        )
        alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))
        pred = alpha * pred_mul + (1.0 - alpha) * pred_add
        if clip_nonnegative:
            pred = np.maximum(pred, 0.0)
        diag: dict[str, Any] = {
            "hybrid_alpha": alpha,
            "multiplicative_diagnostics": diag_mul,
            "additive_diagnostics": diag_add,
            "source_pred_nrmse": float(diag_mul.get("source_pred_nrmse", np.nan)),
            "source_self_fit_error": float(
                alpha * float(diag_mul.get("source_self_fit_error", 0.0))
                + (1.0 - alpha) * float(diag_add.get("source_self_fit_error", 0.0))
            ),
            "valid_voxel_fraction": float(
                alpha * float(diag_mul.get("valid_voxel_fraction", 0.0))
                + (1.0 - alpha) * float(diag_add.get("valid_voxel_fraction", 0.0))
            ),
            "ratio_clip_fraction": float(diag_mul.get("ratio_clip_fraction", np.nan)),
            "ratio_min": float(diag_mul.get("ratio_min", np.nan)),
            "ratio_median": float(diag_mul.get("ratio_median", np.nan)),
            "ratio_max": float(diag_mul.get("ratio_max", np.nan)),
            "ratio_dynamic_range": float(diag_mul.get("ratio_dynamic_range", np.nan)),
            "robust_ratio_variance": float(
                diag_mul.get("robust_ratio_variance", np.nan)
            ),
            "pullback_mode": diag_mul.get("pullback_mode", pullback_mode),
            "x_mod_total_mass_ratio": float(
                diag_mul.get("x_mod_total_mass_ratio", np.nan)
            ),
            "x_corr_total_mass_ratio": float(
                diag_add.get("x_corr_total_mass_ratio", np.nan)
            ),
            "negative_projection_fraction": float(
                diag_add.get("negative_projection_fraction", np.nan)
            ),
            "ratio_map": diag_mul.get("ratio_map"),
            "normalized_residual_map": diag_add.get("normalized_residual_map"),
        }
        return pred, diag
    raise ValueError("mode must be 'multiplicative', 'additive', or 'hybrid'.")


def two_endpoint_voxel_lifted_transfer(
    x0: np.ndarray,
    theta_left: float,
    theta_right: float,
    theta_target: float,
    y_left: np.ndarray,
    y_right: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    mode: str = "multiplicative",
    hybrid_alpha: float = 0.5,
    eps: float = EPS,
    ratio_clip: tuple[float, float] | None = (0.2, 5.0),
    clip_nonnegative: bool = True,
    pullback_mode: str = "geometry",
    G0: np.ndarray | None = None,
    G1: np.ndarray | None = None,
    G2: np.ndarray | None = None,
    moment_u_grid: np.ndarray | None = None,
    moment_v_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend voxel-lifted residual transfer from two measured endpoints."""
    pred_left, diag_left = _single_endpoint_transfer(
        x0,
        theta_left,
        theta_target,
        y_left,
        projector,
        geometry,
        mode,
        hybrid_alpha,
        eps,
        ratio_clip,
        clip_nonnegative,
        pullback_mode,
    )
    pred_right, diag_right = _single_endpoint_transfer(
        x0,
        theta_right,
        theta_target,
        y_right,
        projector,
        geometry,
        mode,
        hybrid_alpha,
        eps,
        ratio_clip,
        clip_nonnegative,
        pullback_mode,
    )
    dist_left = abs(float(theta_target) - float(theta_left))
    dist_right = abs(float(theta_right) - float(theta_target))
    total = dist_left + dist_right
    if total <= float(eps):
        w_left, w_right = 1.0, 0.0
    else:
        w_left = dist_right / total
        w_right = dist_left / total
    blended = w_left * pred_left + w_right * pred_right
    if clip_nonnegative:
        blended = np.maximum(blended, 0.0)

    denom = np.linalg.norm((0.5 * (pred_left + pred_right)).ravel()) + float(eps)
    disagreement = float(np.linalg.norm((pred_left - pred_right).ravel()) / denom)
    uncertainty = np.abs(pred_left - pred_right)
    moment_uncertainty = None
    if G0 is not None and G1 is not None and G2 is not None:
        moment_uncertainty = moment_based_uncertainty_from_depth_moments(
            G0,
            G1,
            G2,
            eps=eps,
            u_grid=moment_u_grid,
            v_grid=moment_v_grid,
            geometry=geometry,
        )

    def _weighted_diag(key: str) -> float:
        left_value = float(diag_left.get(key, np.nan))
        right_value = float(diag_right.get(key, np.nan))
        left_finite = np.isfinite(left_value)
        right_finite = np.isfinite(right_value)
        if left_finite and right_finite:
            return float(w_left * left_value + w_right * right_value)
        if left_finite:
            return left_value
        if right_finite:
            return right_value
        return float("nan")

    def _min_diag(key: str) -> float:
        values = [
            float(value)
            for value in (diag_left.get(key, np.nan), diag_right.get(key, np.nan))
            if np.isfinite(float(value))
        ]
        return float(min(values)) if values else float("nan")

    def _max_diag(key: str) -> float:
        values = [
            float(value)
            for value in (diag_left.get(key, np.nan), diag_right.get(key, np.nan))
            if np.isfinite(float(value))
        ]
        return float(max(values)) if values else float("nan")

    diagnostics: dict[str, Any] = {
        "left_weight": float(w_left),
        "right_weight": float(w_right),
        "endpoint_disagreement_norm": disagreement,
        "uncertainty": uncertainty,
        "moment_uncertainty": moment_uncertainty,
        "left_prediction": pred_left,
        "right_prediction": pred_right,
        "left_diagnostics": diag_left,
        "right_diagnostics": diag_right,
        "source_pred_nrmse": _weighted_diag("source_pred_nrmse"),
        "source_self_fit_error": _weighted_diag("source_self_fit_error"),
        "valid_voxel_fraction": _weighted_diag("valid_voxel_fraction"),
        "ratio_clip_fraction": _weighted_diag("ratio_clip_fraction"),
        "ratio_min": _min_diag("ratio_min"),
        "ratio_median": _weighted_diag("ratio_median"),
        "ratio_max": _max_diag("ratio_max"),
        "ratio_dynamic_range": _weighted_diag("ratio_dynamic_range"),
        "robust_ratio_variance": _weighted_diag("robust_ratio_variance"),
        "pullback_mode": str(pullback_mode),
        "x_mod_total_mass_ratio": _weighted_diag("x_mod_total_mass_ratio"),
        "x_corr_total_mass_ratio": _weighted_diag("x_corr_total_mass_ratio"),
        "negative_projection_fraction": _weighted_diag("negative_projection_fraction"),
    }
    return np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0), diagnostics


def voxel_lifted_transfer(
    x0: np.ndarray,
    theta_source: float,
    theta_target: float,
    y_source: np.ndarray,
    projector: Any,
    geometry: PinholeGeometry | None,
    eps: float = EPS,
    transfer_mode: str = "multiplicative",
    ratio_clip: tuple[float, float] | None = (0.2, 5.0),
    hybrid_alpha: float = 0.5,
    clip_nonnegative: bool = True,
    pullback_mode: str = "geometry",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-endpoint voxel lifted transfer with selectable transfer/pullback mode."""
    return _single_endpoint_transfer(
        x0,
        theta_source,
        theta_target,
        y_source,
        projector,
        geometry,
        mode=transfer_mode,
        hybrid_alpha=hybrid_alpha,
        eps=eps,
        ratio_clip=ratio_clip,
        clip_nonnegative=clip_nonnegative,
        pullback_mode=pullback_mode,
    )


def moment_based_uncertainty_from_depth_moments(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
    u_grid: np.ndarray | None = None,
    v_grid: np.ndarray | None = None,
    geometry: PinholeGeometry | None = None,
) -> dict[str, np.ndarray]:
    """Return a normalized hidden-depth uncertainty map for diagnostics only."""
    g0 = np.maximum(np.nan_to_num(np.asarray(G0, dtype=np.float64), nan=0.0), 0.0)
    if u_grid is not None and v_grid is not None:
        metrics = compute_dimensionless_depth_metrics(
            g0,
            G1,
            G2,
            u_grid,
            v_grid,
            geometry=geometry,
            eps=eps,
        )
        m = metrics["m"]
        var = metrics["var_eta"]
        vdi = metrics["vdi"]
    else:
        m, var = compute_moments_mean_var(g0, G1, G2, eps=eps)
        vdi = np.zeros_like(g0)
    cv_eta = np.sqrt(np.maximum(var, 0.0)) / (np.maximum(m, 0.0) + float(eps))
    cv_eta = np.nan_to_num(cv_eta, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = g0 > float(eps)
    if np.any(nonzero):
        scale = float(np.percentile(cv_eta[nonzero], 95.0))
    else:
        scale = 0.0
    uncertainty = (
        np.clip(cv_eta / (scale + float(eps)), 0.0, 1.0)
        if scale > float(eps)
        else np.zeros_like(cv_eta)
    )
    return {
        "m": np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0),
        "var_eta": np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0),
        "cv_eta": cv_eta,
        "uncertainty": uncertainty,
        "vdi": np.nan_to_num(vdi, nan=0.0, posinf=0.0, neginf=0.0),
    }
