from __future__ import annotations

import numpy as np

from src.lifted_projective_dynamics import EPS, compute_lifted_coefficients
from src.xfct_geometry import PinholeGeometry

MOMENT_CAP = 1.0e100


def _finite_nonnegative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=MOMENT_CAP, neginf=0.0)
    return np.clip(arr, 0.0, MOMENT_CAP)


def _finite_signed(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=MOMENT_CAP, neginf=-MOMENT_CAP)
    return np.clip(arr, -MOMENT_CAP, MOMENT_CAP)


def compute_moments_mean_var(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional eta mean and variance from G0/G1/G2 maps."""
    g0 = _finite_nonnegative(G0)
    g1 = _finite_nonnegative(G1)
    g2 = _finite_nonnegative(G2)
    denom = np.where(g0 > float(eps), g0, g0 + float(eps))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.maximum(g1 / denom, 0.0)
        var = g2 / denom - m**2
    m = np.where(g0 > float(eps), m, 0.0)
    var = np.where(g0 > float(eps), var, 0.0)
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    var = np.where(var < 0.0, np.maximum(var, -1.0e-12), var)
    var = np.maximum(var, 0.0)
    return (
        _finite_nonnegative(m),
        _finite_nonnegative(var),
    )


def gamma_closure_G3(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Return G3 using a positive Gamma closure for eta | (u, v)."""
    g0 = _finite_nonnegative(G0)
    m, var = compute_moments_mean_var(g0, G1, G2, eps=eps)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        third_raw = m**3 + 3.0 * m * var + 2.0 * var**2 / (m + float(eps))
        G3 = g0 * third_raw
    return _finite_nonnegative(G3)


def project_to_realizable_moment_cone(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically project G0/G1/G2 to a nonnegative realizable moment cone."""
    g0 = _finite_nonnegative(G0)
    g1 = _finite_nonnegative(G1)
    g2 = _finite_nonnegative(G2)

    zero_mass = g0 <= float(eps)
    g1 = np.where(zero_mass, 0.0, g1)
    g2 = np.where(zero_mass, 0.0, g2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower_bound = g1**2 / (g0 + float(eps))
    lower_bound = _finite_nonnegative(lower_bound)
    g2 = np.maximum(g2, lower_bound)
    with np.errstate(over="ignore", invalid="ignore"):
        max_g1 = np.sqrt(g2 * (g0 + float(eps)))
    g1 = np.minimum(g1, _finite_nonnegative(max_g1))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower_bound = g1**2 / (g0 + float(eps))
    lower_bound = _finite_nonnegative(lower_bound)
    g2 = np.maximum(g2, lower_bound)
    return (
        _finite_nonnegative(g0),
        _finite_nonnegative(g1),
        _finite_nonnegative(g2),
    )


def compute_dimensionless_depth_metrics(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry | None = None,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Return dimensionless hidden-depth and velocity-dispersion diagnostics."""
    g0, g1, g2 = project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    m, var_eta = compute_moments_mean_var(g0, g1, g2, eps=eps)
    cv_eta = np.sqrt(np.maximum(var_eta, 0.0)) / (m + float(eps))
    a_u, a_v, b_u, b_v, _, _ = compute_lifted_coefficients(
        u_grid,
        v_grid,
        geometry,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        v_u_eff = a_u + b_u * m
        v_v_eff = a_v + b_v * m
        sigma_uu = b_u**2 * var_eta
        sigma_uv = b_u * b_v * var_eta
        sigma_vv = b_v**2 * var_eta
        velocity_norm = np.sqrt(v_u_eff**2 + v_v_eff**2)
        vdi = np.sqrt(np.maximum(sigma_uu + sigma_vv, 0.0)) / (
            velocity_norm + float(eps)
        )
    return {
        "m": _finite_nonnegative(m),
        "var_eta": _finite_nonnegative(var_eta),
        "cv_eta": _finite_nonnegative(cv_eta),
        "v_u_eff": _finite_signed(v_u_eff),
        "v_v_eff": _finite_signed(v_v_eff),
        "sigma_uu": _finite_nonnegative(sigma_uu),
        "sigma_uv": _finite_signed(sigma_uv),
        "sigma_vv": _finite_nonnegative(sigma_vv),
        "vdi": _finite_nonnegative(vdi),
    }
