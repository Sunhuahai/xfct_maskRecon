from __future__ import annotations

from typing import Mapping

import numpy as np

from src.depth_moment_projector import project_depth_moments
from src.lifted_projective_dynamics import (
    EPS,
    compute_geometry_source_moment,
    compute_lifted_coefficients,
)
from src.xfct_geometry import (
    PinholeGeometry,
    detector_physical_coordinates,
)


def _spacing_value(
    detector_pixel_size: float | None,
    geometry: PinholeGeometry | None = None,
) -> float:
    spacing = (
        getattr(geometry, "detector_pixel_size", None)
        if detector_pixel_size is None
        else detector_pixel_size
    )
    spacing = 1.0 if spacing is None else float(spacing)
    if spacing <= 0.0:
        raise ValueError("detector pixel spacing must be positive.")
    return spacing


def gradient_u(values: np.ndarray, detector_pixel_size: float = 1.0) -> np.ndarray:
    """Return d(values) / du using central interior differences."""
    arr = np.asarray(values, dtype=np.float64)
    spacing = _spacing_value(detector_pixel_size)
    grad = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[-1] < 2:
        return grad

    grad[..., 1:-1] = (arr[..., 2:] - arr[..., :-2]) / (2.0 * spacing)
    grad[..., 0] = (arr[..., 1] - arr[..., 0]) / spacing
    grad[..., -1] = (arr[..., -1] - arr[..., -2]) / spacing
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def gradient_v(values: np.ndarray, detector_pixel_size: float = 1.0) -> np.ndarray:
    """Return d(values) / dv using central interior differences."""
    arr = np.asarray(values, dtype=np.float64)
    spacing = _spacing_value(detector_pixel_size)
    grad = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[0] < 2:
        return grad

    grad[1:-1, ...] = (arr[2:, ...] - arr[:-2, ...]) / (2.0 * spacing)
    grad[0, ...] = (arr[1, ...] - arr[0, ...]) / spacing
    grad[-1, ...] = (arr[-1, ...] - arr[-2, ...]) / spacing
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def divergence(
    Fu: np.ndarray,
    Fv: np.ndarray,
    detector_pixel_size: float = 1.0,
) -> np.ndarray:
    """Return div_s(F) = dFu/du + dFv/dv on the detector grid."""
    fu = np.asarray(Fu, dtype=np.float64)
    fv = np.asarray(Fv, dtype=np.float64)
    if fu.shape != fv.shape:
        raise ValueError(f"Fu and Fv shape mismatch: {fu.shape} vs {fv.shape}.")
    div = gradient_u(fu, detector_pixel_size) + gradient_v(
        fv,
        detector_pixel_size,
    )
    return np.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)


def detector_grid(
    detector_shape: tuple[int, int],
    detector_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.meshgrid(
        np.arange(detector_shape[0], dtype=np.float64),
        np.arange(detector_shape[1], dtype=np.float64),
        indexing="ij",
    )
    return detector_physical_coordinates(
        rows,
        cols,
        detector_shape=detector_shape,
        detector_pixel_size=detector_pixel_size,
    )


def compute_moment_flux(
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a G_n + b G_{n+1} detector flux components."""
    a_u, a_v, b_u, b_v, _, _ = compute_lifted_coefficients(
        u_grid,
        v_grid,
        geometry,
    )
    gn = np.asarray(Gn, dtype=np.float64)
    gnp1 = np.asarray(Gnp1, dtype=np.float64)
    Fu = a_u * gn + b_u * gnp1
    Fv = a_v * gn + b_v * gnp1
    return (
        np.nan_to_num(Fu, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(Fv, nan=0.0, posinf=0.0, neginf=0.0),
    )


def compute_moment_source_geo(
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    """Return S_n for the pure geometry distance weight."""
    return compute_geometry_source_moment(Gn, Gnp1, u_grid, v_grid, geometry)


def _validate_n(n: int) -> int:
    nn = int(n)
    if nn not in {0, 1}:
        raise ValueError("moment hierarchy diagnostics currently support n=0 or n=1.")
    return nn


def _require_keys(
    moments: Mapping[str, np.ndarray],
    required: tuple[str, ...],
    context: str,
    n: int,
) -> None:
    missing = [key for key in required if key not in moments]
    if missing:
        required_text = " and ".join(required)
        missing_text = ", ".join(missing)
        raise ValueError(
            f"moment hierarchy residual for n={n} requires {required_text}; "
            f"{context} is missing {missing_text}."
        )


def _source_term(
    source_mode: str | None,
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None,
) -> np.ndarray:
    mode = "geo" if source_mode is None else str(source_mode).lower()
    if mode in {"geo", "geometry", "pure_geo"}:
        return compute_moment_source_geo(Gn, Gnp1, u_grid, v_grid, geometry)
    if mode in {"none", "off", "zero", "no_source"}:
        return np.zeros_like(np.asarray(Gn, dtype=np.float64))
    raise ValueError(f"unsupported moment source_mode: {source_mode!r}")


def compute_moment_residual_from_moment_maps(
    moments_minus: Mapping[str, np.ndarray],
    moments_center: Mapping[str, np.ndarray],
    moments_plus: Mapping[str, np.ndarray],
    h: float,
    n: int,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
    source_mode: str | None = "geo",
    detector_pixel_size: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return moment hierarchy residual components from precomputed maps."""
    nn = _validate_n(n)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    geom = PinholeGeometry() if geometry is None else geometry

    key_n = f"G{nn}"
    key_np1 = f"G{nn + 1}"
    _require_keys(moments_minus, (key_n,), "moments_minus", nn)
    _require_keys(moments_plus, (key_n,), "moments_plus", nn)
    _require_keys(moments_center, (key_n, key_np1), "moments_center", nn)

    Gn_minus = np.asarray(moments_minus[key_n], dtype=np.float64)
    Gn_plus = np.asarray(moments_plus[key_n], dtype=np.float64)
    Gn = np.asarray(moments_center[key_n], dtype=np.float64)
    Gnp1 = np.asarray(moments_center[key_np1], dtype=np.float64)
    if Gn_minus.shape != Gn.shape or Gn_plus.shape != Gn.shape:
        raise ValueError("finite-difference moment maps must have matching shapes.")
    if Gnp1.shape != Gn.shape:
        raise ValueError(f"{key_n} and {key_np1} shape mismatch.")

    partial_theta = (Gn_plus - Gn_minus) / (2.0 * float(h))
    Fu, Fv = compute_moment_flux(Gn, Gnp1, u_grid, v_grid, geom)
    div_flux = divergence(Fu, Fv, detector_pixel_size=detector_pixel_size)
    _, _, _, _, alpha, _ = compute_lifted_coefficients(u_grid, v_grid, geom)
    delta = float(geom.detector_offset_x)
    source = _source_term(source_mode, Gn, Gnp1, u_grid, v_grid, geom)

    residual = (
        partial_theta
        + div_flux
        + float(nn) * alpha * Gn
        - float(nn) * delta * Gnp1
        - source
    )
    return {
        "residual": np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0),
        "partial_theta": np.nan_to_num(
            partial_theta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "div_flux": div_flux,
        "source": source,
        "Fu": Fu,
        "Fv": Fv,
    }


def _project_three_angles(
    volume: np.ndarray,
    theta: float,
    h: float,
    geometry: PinholeGeometry,
    weight_mode: str,
    detector_shape: tuple[int, int],
    voxel_size: float,
    detector_pixel_size: float,
    eps: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    kwargs = {
        "geometry": geometry,
        "detector_shape": detector_shape,
        "voxel_size": voxel_size,
        "detector_pixel_size": detector_pixel_size,
        "weight_mode": weight_mode,
        "eps": eps,
    }
    moments_minus = project_depth_moments(volume, float(theta) - float(h), **kwargs)
    moments_center = project_depth_moments(volume, float(theta), **kwargs)
    moments_plus = project_depth_moments(volume, float(theta) + float(h), **kwargs)
    return moments_minus, moments_center, moments_plus


def compute_moment_residual(
    volume: np.ndarray,
    theta: float,
    h: float,
    n: int,
    geometry: PinholeGeometry,
    source_mode: str = "geo",
    weight_mode: str = "geo_or_existing",
    detector_shape: tuple[int, int] = (80, 160),
    voxel_size: float = 0.5,
    detector_pixel_size: float | None = 0.25,
    eps: float = EPS,
) -> np.ndarray:
    """Project moments and return R_n for the diagnostic hierarchy."""
    nn = _validate_n(n)
    spacing = _spacing_value(detector_pixel_size, geometry)
    detector_shape = tuple(int(value) for value in detector_shape)
    moments_minus, moments_center, moments_plus = _project_three_angles(
        volume=volume,
        theta=float(theta),
        h=float(h),
        geometry=geometry,
        weight_mode=weight_mode,
        detector_shape=detector_shape,
        voxel_size=float(voxel_size),
        detector_pixel_size=spacing,
        eps=float(eps),
    )
    u_grid, v_grid = detector_grid(detector_shape, spacing)
    return compute_moment_residual_from_moment_maps(
        moments_minus=moments_minus,
        moments_center=moments_center,
        moments_plus=moments_plus,
        h=float(h),
        n=nn,
        u_grid=u_grid,
        v_grid=v_grid,
        geometry=geometry,
        source_mode=source_mode,
        detector_pixel_size=spacing,
    )["residual"]


def _residual_stats(
    residual: np.ndarray,
    partial_theta: np.ndarray,
    eps: float,
) -> dict[str, float]:
    residual_arr = np.nan_to_num(
        np.asarray(residual, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    partial_arr = np.nan_to_num(
        np.asarray(partial_theta, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    residual_norm = float(np.linalg.norm(residual_arr.ravel()))
    relative = residual_norm / (float(np.linalg.norm(partial_arr.ravel())) + eps)
    return {
        "residual_norm": residual_norm,
        "relative_residual_norm": float(relative),
        "mean_abs_residual": float(np.mean(np.abs(residual_arr))),
        "max_abs_residual": float(np.max(np.abs(residual_arr))),
    }


def _central_plane_residual(
    moments_minus: Mapping[str, np.ndarray],
    moments_center: Mapping[str, np.ndarray],
    moments_plus: Mapping[str, np.ndarray],
    h: float,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    detector_pixel_size: float,
) -> dict[str, np.ndarray]:
    _require_keys(moments_minus, ("G0",), "moments_minus", 0)
    _require_keys(moments_plus, ("G0",), "moments_plus", 0)
    _require_keys(moments_center, ("G0",), "moments_center", 0)
    G0 = np.asarray(moments_center["G0"], dtype=np.float64)
    partial_theta = (
        np.asarray(moments_plus["G0"], dtype=np.float64)
        - np.asarray(moments_minus["G0"], dtype=np.float64)
    ) / (2.0 * float(h))

    eta_central = 1.0 / float(geometry.center_to_pinhole)
    G1_central = eta_central * G0
    Fu, Fv = compute_moment_flux(G0, G1_central, u_grid, v_grid, geometry)
    div_flux = divergence(Fu, Fv, detector_pixel_size=detector_pixel_size)
    source = compute_moment_source_geo(G0, G1_central, u_grid, v_grid, geometry)
    residual = partial_theta + div_flux - source
    return {
        "residual": np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0),
        "partial_theta": np.nan_to_num(
            partial_theta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "div_flux": div_flux,
        "source": source,
        "Fu": Fu,
        "Fv": Fv,
    }


def compute_transport_residual_comparison(
    volume: np.ndarray,
    theta: float,
    h: float,
    geometry: PinholeGeometry,
    weight_mode: str = "geo_or_existing",
    detector_shape: tuple[int, int] = (80, 160),
    voxel_size: float = 0.5,
    detector_pixel_size: float | None = 0.25,
    eps: float = EPS,
) -> dict[str, dict[str, np.ndarray | float | int]]:
    """Compare central-plane and lifted moment hierarchy residuals."""
    if h <= 0.0:
        raise ValueError("h must be positive.")
    spacing = _spacing_value(detector_pixel_size, geometry)
    detector_shape = tuple(int(value) for value in detector_shape)
    u_grid, v_grid = detector_grid(detector_shape, spacing)
    moments_minus, moments_center, moments_plus = _project_three_angles(
        volume=volume,
        theta=float(theta),
        h=float(h),
        geometry=geometry,
        weight_mode=weight_mode,
        detector_shape=detector_shape,
        voxel_size=float(voxel_size),
        detector_pixel_size=spacing,
        eps=float(eps),
    )

    raw_cases: dict[str, tuple[int, dict[str, np.ndarray]]] = {
        "central_plane_transport": (
            0,
            _central_plane_residual(
                moments_minus,
                moments_center,
                moments_plus,
                float(h),
                u_grid,
                v_grid,
                geometry,
                spacing,
            ),
        ),
        "moment_no_source_n0": (
            0,
            compute_moment_residual_from_moment_maps(
                moments_minus,
                moments_center,
                moments_plus,
                float(h),
                0,
                u_grid,
                v_grid,
                geometry,
                source_mode="none",
                detector_pixel_size=spacing,
            ),
        ),
        "moment_geo_source_n0": (
            0,
            compute_moment_residual_from_moment_maps(
                moments_minus,
                moments_center,
                moments_plus,
                float(h),
                0,
                u_grid,
                v_grid,
                geometry,
                source_mode="geo",
                detector_pixel_size=spacing,
            ),
        ),
        "moment_geo_source_n1": (
            1,
            compute_moment_residual_from_moment_maps(
                moments_minus,
                moments_center,
                moments_plus,
                float(h),
                1,
                u_grid,
                v_grid,
                geometry,
                source_mode="geo",
                detector_pixel_size=spacing,
            ),
        ),
    }

    cases: dict[str, dict[str, np.ndarray | float | int]] = {}
    for case_name, (n_value, components) in raw_cases.items():
        stats = _residual_stats(
            components["residual"],
            components["partial_theta"],
            float(eps),
        )
        cases[case_name] = {
            **components,
            **stats,
            "n": int(n_value),
            "moments_center": moments_center,
        }
    return cases
