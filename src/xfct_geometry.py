from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class PinholeGeometry:
    detector_to_pinhole: float = 27.0
    center_to_pinhole: float = 50.0
    detector_offset_x: float = -0.5


def _xyz_components(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"xyz last dimension must be 3, got {pts.shape}.")
    return pts[..., 0], pts[..., 1], pts[..., 2]


def xprime_yprime(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    x, y, _ = _xyz_components(xyz)
    ct = np.cos(theta)
    st = np.sin(theta)
    xprime = x * ct - y * st + geom.detector_offset_x
    yprime = x * st + y * ct + geom.center_to_pinhole
    return xprime, yprime


def dot_xprime_yprime(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    xprime, yprime = xprime_yprime(theta, xyz, geom)
    return -(yprime - geom.center_to_pinhole), xprime - geom.detector_offset_x


def uv(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    xprime, yprime = xprime_yprime(theta, xyz, geom)
    _, _, z = _xyz_components(xyz)
    denom = np.maximum(np.abs(yprime), EPS) * np.sign(yprime + EPS)
    u = geom.detector_to_pinhole * xprime / denom
    v = geom.detector_to_pinhole * z / denom
    return u, v


def dot_uv(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    xprime, yprime = xprime_yprime(theta, xyz, geom)
    dxprime, dyprime = dot_xprime_yprime(theta, xyz, geom)
    _, _, z = _xyz_components(xyz)
    denom = np.maximum(np.abs(yprime), EPS) * np.sign(yprime + EPS)
    du = geom.detector_to_pinhole * (dxprime * denom - xprime * dyprime) / (denom**2)
    dv = -geom.detector_to_pinhole * z * dyprime / (denom**2)
    return du, dv


def dot_uv_from_detector_depth(
    u: np.ndarray,
    v: np.ndarray,
    depth_lambda: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    alpha = np.asarray(u, dtype=np.float64) / geom.detector_to_pinhole
    vv = np.asarray(v, dtype=np.float64)
    lam = np.asarray(depth_lambda, dtype=np.float64)
    du = geom.detector_to_pinhole * (
        -(1.0 + alpha**2)
        + (geom.center_to_pinhole + alpha * geom.detector_offset_x)
        / np.maximum(lam, EPS)
    )
    dv = -vv * (alpha - geom.detector_offset_x / np.maximum(lam, EPS))
    return du, dv


def finite_angle_map(
    u_i: np.ndarray,
    v_i: np.ndarray,
    depth_lambda: np.ndarray,
    delta_theta: float,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geom = PinholeGeometry() if geometry is None else geometry
    lam = np.asarray(depth_lambda, dtype=np.float64)
    alpha = np.asarray(u_i, dtype=np.float64) / geom.detector_to_pinhole
    beta = np.asarray(v_i, dtype=np.float64) / geom.detector_to_pinhole

    x_i = alpha * lam - geom.detector_offset_x
    y_i = lam - geom.center_to_pinhole
    ct = np.cos(delta_theta)
    st = np.sin(delta_theta)

    x_j = x_i * ct - y_i * st
    y_j = x_i * st + y_i * ct
    xprime_j = x_j + geom.detector_offset_x
    yprime_j = y_j + geom.center_to_pinhole
    z_j = beta * lam

    denom = np.maximum(np.abs(yprime_j), EPS) * np.sign(yprime_j + EPS)
    u_j = geom.detector_to_pinhole * xprime_j / denom
    v_j = geom.detector_to_pinhole * z_j / denom
    return u_j, v_j


def detector_continuous_indices(
    u: np.ndarray,
    v: np.ndarray,
    detector_shape: tuple[int, int],
    detector_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(detector_shape) != 2:
        raise ValueError(
            f"detector_shape must be (height, width), got {detector_shape}."
        )
    if detector_pixel_size <= 0.0:
        raise ValueError("detector_pixel_size must be positive.")

    height, width = detector_shape
    row = np.asarray(v, dtype=np.float64) / float(detector_pixel_size)
    col = np.asarray(u, dtype=np.float64) / float(detector_pixel_size)
    row = row + (float(height) - 1.0) / 2.0
    col = col + (float(width) - 1.0) / 2.0
    return row, col


def detector_physical_coordinates(
    row: np.ndarray,
    col: np.ndarray,
    detector_shape: tuple[int, int],
    detector_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(detector_shape) != 2:
        raise ValueError(
            f"detector_shape must be (height, width), got {detector_shape}."
        )
    if detector_pixel_size <= 0.0:
        raise ValueError("detector_pixel_size must be positive.")

    height, width = detector_shape
    v = np.asarray(row, dtype=np.float64) - (float(height) - 1.0) / 2.0
    u = np.asarray(col, dtype=np.float64) - (float(width) - 1.0) / 2.0
    return u * float(detector_pixel_size), v * float(detector_pixel_size)


def geometry_weight(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    xprime, yprime = xprime_yprime(theta, xyz, geometry)
    _, _, z = _xyz_components(xyz)
    r2 = xprime**2 + yprime**2 + z**2
    return yprime / np.maximum(r2, EPS) ** 1.5


def geometry_log_weight_dot(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    geom = PinholeGeometry() if geometry is None else geometry
    xprime, yprime = xprime_yprime(theta, xyz, geom)
    dxprime, dyprime = dot_xprime_yprime(theta, xyz, geom)
    _, _, z = _xyz_components(xyz)
    r2 = np.maximum(xprime**2 + yprime**2 + z**2, EPS)
    return dyprime / np.maximum(yprime, EPS) - 3.0 * (
        xprime * dxprime + yprime * dyprime
    ) / r2


def finite_angle_geometry_ratio(
    u_i: np.ndarray,
    v_i: np.ndarray,
    depth_lambda: np.ndarray,
    delta_theta: float,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    geom = PinholeGeometry() if geometry is None else geometry
    lam = np.asarray(depth_lambda, dtype=np.float64)
    alpha = np.asarray(u_i, dtype=np.float64) / geom.detector_to_pinhole
    beta = np.asarray(v_i, dtype=np.float64) / geom.detector_to_pinhole

    x_i = alpha * lam - geom.detector_offset_x
    y_i = lam - geom.center_to_pinhole
    ct = np.cos(delta_theta)
    st = np.sin(delta_theta)
    x_j = x_i * ct - y_i * st
    y_j = x_i * st + y_i * ct
    xprime_j = x_j + geom.detector_offset_x
    yprime_j = y_j + geom.center_to_pinhole
    z_j = beta * lam

    r2_i = lam**2 * (1.0 + alpha**2 + beta**2)
    r2_j = xprime_j**2 + yprime_j**2 + z_j**2
    w_i = lam / np.maximum(r2_i, EPS) ** 1.5
    w_j = yprime_j / np.maximum(r2_j, EPS) ** 1.5
    return w_j / np.maximum(w_i, EPS)
