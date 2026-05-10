from __future__ import annotations

import numpy as np

from src.xfct_geometry import PinholeGeometry, xprime_yprime

EPS = 1e-12


def _geometry_or_default(
    geometry: PinholeGeometry | None,
) -> PinholeGeometry:
    return PinholeGeometry() if geometry is None else geometry


def _safe_reciprocal_depth(yprime: np.ndarray) -> np.ndarray:
    depth = np.asarray(yprime, dtype=np.float64)
    sign = np.where(depth >= 0.0, 1.0, -1.0)
    denom = np.maximum(np.abs(depth), EPS) * sign
    return 1.0 / denom


def compute_eta(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    """Return the hidden reciprocal-depth coordinate eta = 1 / y'."""
    geom = _geometry_or_default(geometry)
    _, yprime = xprime_yprime(theta, xyz, geom)
    return _safe_reciprocal_depth(yprime)


def compute_dot_eta(
    theta: float | np.ndarray,
    xyz: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    """Return d eta / d theta for the center-projection dynamics."""
    geom = _geometry_or_default(geometry)
    xprime, yprime = xprime_yprime(theta, xyz, geom)
    eta = _safe_reciprocal_depth(yprime)
    alpha = xprime * eta
    return -alpha * eta + geom.detector_offset_x * eta**2


def compute_lifted_coefficients(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return detector-grid coefficients for dot_u and dot_v.

    The coefficients satisfy dot_u = a_u + b_u * eta and
    dot_v = a_v + b_v * eta.
    """
    geom = _geometry_or_default(geometry)
    detector_distance = float(geom.detector_to_pinhole)
    if abs(detector_distance) < EPS:
        raise ValueError("geometry.detector_to_pinhole must be nonzero.")

    u_arr, v_arr = np.broadcast_arrays(
        np.asarray(u_grid, dtype=np.float64),
        np.asarray(v_grid, dtype=np.float64),
    )
    alpha = u_arr / detector_distance
    beta = v_arr / detector_distance

    a_u = -detector_distance * (1.0 + alpha**2)
    b_u = detector_distance * (
        float(geom.center_to_pinhole) + alpha * float(geom.detector_offset_x)
    )
    a_v = -v_arr * alpha
    b_v = v_arr * float(geom.detector_offset_x)

    return a_u, a_v, b_u, b_v, alpha, beta


def compute_effective_velocity_from_moments(
    G0: np.ndarray,
    G1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return moment-averaged detector velocity components."""
    g0 = np.asarray(G0, dtype=np.float64)
    g1 = np.asarray(G1, dtype=np.float64)
    m1 = np.divide(g1, g0 + float(eps))
    a_u, a_v, b_u, b_v, _, _ = compute_lifted_coefficients(
        u_grid,
        v_grid,
        geometry,
    )
    v_u_eff = a_u + b_u * m1
    v_v_eff = a_v + b_v * m1
    return (
        np.nan_to_num(v_u_eff, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(v_v_eff, nan=0.0, posinf=0.0, neginf=0.0),
    )


def compute_depth_variance(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Return Var[eta] from lifted moments, with negative roundoff removed."""
    g0 = np.asarray(G0, dtype=np.float64)
    g1 = np.asarray(G1, dtype=np.float64)
    g2 = np.asarray(G2, dtype=np.float64)
    denom = g0 + float(eps)
    m1 = np.divide(g1, denom)
    m2 = np.divide(g2, denom)
    var_eta = m2 - m1**2
    var_eta = np.nan_to_num(var_eta, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(var_eta, 0.0)


def compute_velocity_covariance(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the detector-velocity covariance induced by Var[eta]."""
    var_eta = compute_depth_variance(G0, G1, G2, eps=eps)
    _, _, b_u, b_v, _, _ = compute_lifted_coefficients(u_grid, v_grid, geometry)
    sigma_uu = b_u**2 * var_eta
    sigma_uv = b_u * b_v * var_eta
    sigma_vv = b_v**2 * var_eta
    return (
        np.nan_to_num(sigma_uu, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(sigma_uv, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(sigma_vv, nan=0.0, posinf=0.0, neginf=0.0),
    )


def compute_geometry_source_moment(
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
) -> np.ndarray:
    """Return the pure-geometry source term for a lifted moment."""
    geom = _geometry_or_default(geometry)
    _, _, _, _, alpha, beta = compute_lifted_coefficients(u_grid, v_grid, geom)
    denominator = 1.0 + alpha**2 + beta**2
    coefficient = float(geom.detector_offset_x) + 3.0 * (
        alpha * float(geom.center_to_pinhole) - float(geom.detector_offset_x)
    ) / denominator
    source = alpha * np.asarray(Gn, dtype=np.float64) - coefficient * np.asarray(
        Gnp1,
        dtype=np.float64,
    )
    return np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
