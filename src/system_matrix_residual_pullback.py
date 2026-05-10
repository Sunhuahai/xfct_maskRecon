from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.lifted_projective_dynamics import EPS
from src.voxel_lifted_residual_transfer import sample_detector_image_at_voxels
from src.xfct_geometry import PinholeGeometry

LOGGER = logging.getLogger(__name__)


def _finite_detector(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def residual_pullback_geometry(
    residual_image: np.ndarray,
    theta: float,
    voxel_grid: np.ndarray,
    geometry: PinholeGeometry | None,
    detector_pixel_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Pull detector residuals to voxels with detector-coordinate bilinear sampling."""
    values, valid = sample_detector_image_at_voxels(
        residual_image,
        float(theta),
        voxel_grid,
        geometry,
        detector_pixel_size=float(detector_pixel_size),
    )
    diagnostics: dict[str, float | str] = {
        "pullback_mode": "geometry",
        "valid_voxel_fraction": float(np.mean(valid)) if valid.size else 0.0,
    }
    return values, valid, diagnostics


def _projector_angle_matrix(projector: Any, theta: float) -> Any | None:
    for method_name in ("angle_matrix", "matrix_for_angle", "system_matrix_for_angle"):
        method = getattr(projector, method_name, None)
        if callable(method):
            matrix = method(float(theta))
            if matrix is not None:
                return matrix
    return None


def residual_pullback_system_matrix(
    residual_image: np.ndarray,
    *,
    angle_matrix: Any | None = None,
    projector: Any | None = None,
    theta: float | None = None,
    volume_shape: tuple[int, ...] | None = None,
    detector_shape: tuple[int, int] | None = None,
    eps: float = EPS,
    allow_fallback: bool = False,
    voxel_grid: np.ndarray | None = None,
    geometry: PinholeGeometry | None = None,
    detector_pixel_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Pull detector residuals to voxels by system-matrix column-weighted averaging."""
    image = _finite_detector(residual_image)
    if image.ndim != 2:
        raise ValueError(f"residual_image must be 2-D, got {image.shape}.")
    if detector_shape is None:
        detector_shape = tuple(int(v) for v in image.shape)
    if angle_matrix is None and projector is not None and theta is not None:
        angle_matrix = _projector_angle_matrix(projector, float(theta))

    if angle_matrix is None:
        if not allow_fallback:
            raise NotImplementedError(
                "system-matrix residual pullback requires an angle-specific "
                "detector-pixel-by-voxel matrix. Provide angle_matrix or a "
                "projector with angle_matrix(theta)."
            )
        if voxel_grid is None:
            raise NotImplementedError(
                "geometry fallback requested, but voxel_grid was not provided."
            )
        LOGGER.warning("system-matrix pullback unavailable; using geometry fallback")
        values, valid, diagnostics = residual_pullback_geometry(
            image,
            float(theta if theta is not None else 0.0),
            voxel_grid,
            geometry,
            detector_pixel_size=float(detector_pixel_size),
        )
        diagnostics["pullback_mode"] = "geometry_fallback"
        return values, valid, diagnostics

    matrix = angle_matrix.tocsr() if hasattr(angle_matrix, "tocsr") else angle_matrix
    pixel_count = int(np.prod(detector_shape))
    if int(matrix.shape[0]) != pixel_count:
        raise ValueError(
            "angle_matrix row count must match detector pixels: "
            f"{matrix.shape[0]} vs {pixel_count}."
        )
    flat = image.reshape(-1)
    numerator = np.asarray(matrix.T @ flat, dtype=np.float64).reshape(-1)
    denominator = np.asarray(
        matrix.T @ np.ones(pixel_count, dtype=np.float64),
        dtype=np.float64,
    ).reshape(-1)
    valid = np.isfinite(denominator) & (np.abs(denominator) > float(eps))
    pulled = np.zeros_like(numerator, dtype=np.float64)
    pulled[valid] = numerator[valid] / (denominator[valid] + float(eps))
    pulled = np.nan_to_num(pulled, nan=0.0, posinf=0.0, neginf=0.0)

    if volume_shape is not None:
        pulled = pulled.reshape(tuple(int(v) for v in volume_shape))
        valid = valid.reshape(tuple(int(v) for v in volume_shape))
    diagnostics = {
        "pullback_mode": "system_matrix",
        "valid_voxel_fraction": float(np.mean(valid)) if valid.size else 0.0,
        "system_matrix_denominator_min": float(np.min(denominator)) if denominator.size else 0.0,
        "system_matrix_denominator_median": (
            float(np.median(denominator)) if denominator.size else 0.0
        ),
        "system_matrix_denominator_max": float(np.max(denominator)) if denominator.size else 0.0,
    }
    return pulled, valid, diagnostics


def compare_geometry_vs_system_pullback(
    residual_image: np.ndarray,
    theta: float,
    voxel_grid: np.ndarray,
    geometry: PinholeGeometry | None,
    *,
    angle_matrix: Any | None = None,
    projector: Any | None = None,
    detector_pixel_size: float = 0.25,
    eps: float = EPS,
) -> dict[str, float]:
    """Compare geometry and system-matrix residual pullback maps."""
    geo, geo_valid, _ = residual_pullback_geometry(
        residual_image,
        float(theta),
        voxel_grid,
        geometry,
        detector_pixel_size=float(detector_pixel_size),
    )
    sys, sys_valid, _ = residual_pullback_system_matrix(
        residual_image,
        angle_matrix=angle_matrix,
        projector=projector,
        theta=float(theta),
        volume_shape=tuple(voxel_grid.shape[:-1]),
        detector_shape=tuple(np.asarray(residual_image).shape),
        eps=eps,
    )
    mask = np.asarray(geo_valid, dtype=bool) & np.asarray(sys_valid, dtype=bool)
    if not np.any(mask):
        return {
            "pullback_nrmse": float("nan"),
            "pullback_corr": float("nan"),
            "shared_valid_fraction": 0.0,
        }
    a = np.asarray(geo, dtype=np.float64)[mask].ravel()
    b = np.asarray(sys, dtype=np.float64)[mask].ravel()
    nrmse = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + float(eps)))
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    corr = float(np.dot(aa, bb) / denom) if denom > float(eps) else float("nan")
    return {
        "pullback_nrmse": nrmse,
        "pullback_corr": corr,
        "shared_valid_fraction": float(np.mean(mask)),
    }
