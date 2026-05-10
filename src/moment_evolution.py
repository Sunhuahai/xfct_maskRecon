from __future__ import annotations

import numpy as np

from src.lifted_projective_dynamics import EPS, compute_lifted_coefficients
from src.moment_closure import (
    compute_dimensionless_depth_metrics,
    compute_moments_mean_var,
    gamma_closure_G3,
    project_to_realizable_moment_cone,
)
from src.moment_hierarchy import (
    compute_moment_flux,
    compute_moment_source_geo,
    divergence,
)
from src.xfct_geometry import PinholeGeometry


def _source(
    source_mode: str,
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
) -> np.ndarray:
    mode = str(source_mode).strip().lower()
    if mode in {"geo", "geometry", "pure_geo"}:
        return compute_moment_source_geo(Gn, Gnp1, u_grid, v_grid, geometry)
    if mode in {"none", "off", "zero", "no_source"}:
        return np.zeros_like(np.asarray(Gn, dtype=np.float64))
    raise ValueError(f"unsupported source_mode: {source_mode!r}")


def _detector_spacing(u_grid: np.ndarray, v_grid: np.ndarray) -> float:
    u_arr = np.asarray(u_grid, dtype=np.float64)
    v_arr = np.asarray(v_grid, dtype=np.float64)
    spacings = []
    if u_arr.shape[-1] > 1:
        du = np.diff(u_arr, axis=-1)
        du = np.abs(du[np.isfinite(du)])
        du = du[du > 0.0]
        if du.size:
            spacings.append(float(np.median(du)))
    if v_arr.shape[0] > 1:
        dv = np.diff(v_arr, axis=0)
        dv = np.abs(dv[np.isfinite(dv)])
        dv = dv[dv > 0.0]
        if dv.size:
            spacings.append(float(np.median(dv)))
    spacings = [spacing for spacing in spacings if np.isfinite(spacing) and spacing > 0.0]
    return float(spacings[0]) if spacings else 1.0


def _rhs_for_n(
    n: int,
    Gn: np.ndarray,
    Gnp1: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    source_mode: str,
    detector_pixel_size: float,
) -> np.ndarray:
    Fu, Fv = compute_moment_flux(Gn, Gnp1, u_grid, v_grid, geometry)
    div_flux = divergence(Fu, Fv, detector_pixel_size=detector_pixel_size)
    _, _, _, _, alpha, _ = compute_lifted_coefficients(u_grid, v_grid, geometry)
    delta = float(geometry.detector_offset_x)
    source = _source(source_mode, Gn, Gnp1, u_grid, v_grid, geometry)
    rhs = -div_flux - float(n) * alpha * Gn + float(n) * delta * Gnp1 + source
    return np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)


def compute_rhs_gamma_closed(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    source_mode: str = "geo",
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Gamma-closed RHS maps for G0, G1, and G2."""
    g0, g1, g2 = project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    g3 = gamma_closure_G3(g0, g1, g2, eps=eps)
    spacing = _detector_spacing(u_grid, v_grid)
    rhs0 = _rhs_for_n(0, g0, g1, u_grid, v_grid, geometry, source_mode, spacing)
    rhs1 = _rhs_for_n(1, g1, g2, u_grid, v_grid, geometry, source_mode, spacing)
    rhs2 = _rhs_for_n(2, g2, g3, u_grid, v_grid, geometry, source_mode, spacing)
    return rhs0, rhs1, rhs2


def _add_scaled(
    moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    rhs: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return project_to_realizable_moment_cone(
        moments[0] + scale * rhs[0],
        moments[1] + scale * rhs[1],
        moments[2] + scale * rhs[2],
        eps=eps,
    )


def rk2_step(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    dtheta: float,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    source_mode: str = "geo",
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance moments by one midpoint RK2 step."""
    if float(dtheta) == 0.0:
        return project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    moments = project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    k1 = compute_rhs_gamma_closed(*moments, u_grid, v_grid, geometry, source_mode, eps)
    midpoint = _add_scaled(moments, k1, 0.5 * float(dtheta), eps)
    k2 = compute_rhs_gamma_closed(*midpoint, u_grid, v_grid, geometry, source_mode, eps)
    return _add_scaled(moments, k2, float(dtheta), eps)


def rk4_step(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    dtheta: float,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    source_mode: str = "geo",
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance moments by one RK4 step."""
    if float(dtheta) == 0.0:
        return project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    moments = project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    dt = float(dtheta)
    k1 = compute_rhs_gamma_closed(*moments, u_grid, v_grid, geometry, source_mode, eps)
    k2_state = _add_scaled(moments, k1, 0.5 * dt, eps)
    k2 = compute_rhs_gamma_closed(*k2_state, u_grid, v_grid, geometry, source_mode, eps)
    k3_state = _add_scaled(moments, k2, 0.5 * dt, eps)
    k3 = compute_rhs_gamma_closed(*k3_state, u_grid, v_grid, geometry, source_mode, eps)
    k4_state = _add_scaled(moments, k3, dt, eps)
    k4 = compute_rhs_gamma_closed(*k4_state, u_grid, v_grid, geometry, source_mode, eps)
    return project_to_realizable_moment_cone(
        moments[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        moments[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        moments[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        eps=eps,
    )


def _diagnostics(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    theta: float,
    eps: float,
) -> dict[str, float]:
    g0, g1, g2 = project_to_realizable_moment_cone(G0, G1, G2, eps=eps)
    _, var = compute_moments_mean_var(g0, g1, g2, eps=eps)
    metrics = compute_dimensionless_depth_metrics(g0, g1, g2, u_grid, v_grid, geometry, eps)
    return {
        "theta": float(theta),
        "min_G0": float(np.min(g0)),
        "min_G1": float(np.min(g1)),
        "min_var": float(np.min(var)),
        "max_cv_eta": float(np.max(metrics["cv_eta"])),
        "max_vdi": float(np.max(metrics["vdi"])),
        "total_mass_G0": float(np.sum(g0)),
    }


def _limit_total_mass(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    reference_mass: float,
    mass_cap_factor: float | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mass_cap_factor is None or float(mass_cap_factor) <= 0.0:
        return G0, G1, G2
    current_mass = float(np.sum(np.asarray(G0, dtype=np.float64)))
    max_mass = max(float(reference_mass) * float(mass_cap_factor), float(eps))
    if not np.isfinite(current_mass) or current_mass <= max_mass:
        return G0, G1, G2
    scale = max_mass / (current_mass + float(eps))
    return project_to_realizable_moment_cone(
        np.asarray(G0, dtype=np.float64) * scale,
        np.asarray(G1, dtype=np.float64) * scale,
        np.asarray(G2, dtype=np.float64) * scale,
        eps=eps,
    )


def evolve_moments_gamma_closed(
    G0_init: np.ndarray,
    G1_init: np.ndarray,
    G2_init: np.ndarray,
    theta0: float,
    theta_target: float,
    n_steps: int,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    geometry: PinholeGeometry,
    method: str = "rk2",
    source_mode: str = "geo",
    mass_cap_factor: float | None = None,
    eps: float = EPS,
) -> dict[str, np.ndarray | list[dict[str, float]]]:
    """Evolve G0/G1/G2 from theta0 to theta_target with Gamma closure."""
    steps = max(1, int(n_steps))
    total = float(theta_target) - float(theta0)
    if total == 0.0:
        g0, g1, g2 = project_to_realizable_moment_cone(G0_init, G1_init, G2_init, eps=eps)
        return {
            "G0": g0,
            "G1": g1,
            "G2": g2,
            "diagnostics": [_diagnostics(g0, g1, g2, u_grid, v_grid, geometry, theta0, eps)],
        }

    stepper_name = str(method).strip().lower()
    if stepper_name == "rk2":
        stepper = rk2_step
    elif stepper_name == "rk4":
        stepper = rk4_step
    else:
        raise ValueError("method must be 'rk2' or 'rk4'.")

    dtheta = total / float(steps)
    g0, g1, g2 = project_to_realizable_moment_cone(G0_init, G1_init, G2_init, eps=eps)
    reference_mass = float(np.sum(g0))
    diagnostics = [_diagnostics(g0, g1, g2, u_grid, v_grid, geometry, theta0, eps)]
    for idx in range(steps):
        g0, g1, g2 = stepper(
            g0,
            g1,
            g2,
            dtheta,
            u_grid,
            v_grid,
            geometry,
            source_mode=source_mode,
            eps=eps,
        )
        g0, g1, g2 = _limit_total_mass(
            g0,
            g1,
            g2,
            reference_mass,
            mass_cap_factor,
            eps,
        )
        diagnostics.append(
            _diagnostics(
                g0,
                g1,
                g2,
                u_grid,
                v_grid,
                geometry,
                float(theta0) + float(idx + 1) * dtheta,
                eps,
            )
        )
    return {"G0": g0, "G1": g1, "G2": g2, "diagnostics": diagnostics}
