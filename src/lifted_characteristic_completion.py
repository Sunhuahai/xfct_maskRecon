from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.lifted_projective_dynamics import EPS
from src.moment_closure import compute_moments_mean_var
from src.xfct_geometry import (
    PinholeGeometry,
    detector_continuous_indices,
    detector_physical_coordinates,
)


@dataclass(frozen=True)
class DetectorGeometry:
    detector_shape: tuple[int, int] = (80, 160)
    detector_pixel_size: float = 0.25
    values_are_density: bool = False


def _geometry_or_default(geometry: PinholeGeometry | None) -> PinholeGeometry:
    return PinholeGeometry() if geometry is None else geometry


def _detector_geometry_or_default(
    detector_geometry: DetectorGeometry | dict[str, Any] | None,
    fallback_shape: tuple[int, int] | None = None,
) -> DetectorGeometry:
    if detector_geometry is None:
        return DetectorGeometry(
            detector_shape=(80, 160) if fallback_shape is None else fallback_shape,
        )
    if isinstance(detector_geometry, DetectorGeometry):
        return detector_geometry
    shape = detector_geometry.get(
        "detector_shape",
        detector_geometry.get("shape", fallback_shape if fallback_shape else (80, 160)),
    )
    spacing = detector_geometry.get("detector_pixel_size", 0.25)
    values_are_density = detector_geometry.get("values_are_density", False)
    return DetectorGeometry(
        detector_shape=tuple(int(v) for v in shape),
        detector_pixel_size=float(spacing),
        values_are_density=bool(values_are_density),
    )


def _detector_grids(
    detector_geometry: DetectorGeometry | dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, DetectorGeometry]:
    det = _detector_geometry_or_default(detector_geometry)
    rows, cols = np.meshgrid(
        np.arange(det.detector_shape[0], dtype=np.float64),
        np.arange(det.detector_shape[1], dtype=np.float64),
        indexing="ij",
    )
    u_grid, v_grid = detector_physical_coordinates(
        rows,
        cols,
        detector_shape=det.detector_shape,
        detector_pixel_size=det.detector_pixel_size,
    )
    return u_grid, v_grid, det


def _safe_nonnegative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def finite_angle_lifted_map(
    alpha: np.ndarray,
    beta: np.ndarray,
    eta: np.ndarray,
    delta_theta: float,
    geometry: PinholeGeometry | None = None,
    eps: float = EPS,
    detector_geometry: DetectorGeometry | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map lifted detector coordinates by the exact finite-angle characteristic."""
    geom = _geometry_or_default(geometry)
    alpha_arr, beta_arr, eta_arr = np.broadcast_arrays(
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
    )
    c = np.cos(float(delta_theta))
    s = np.sin(float(delta_theta))
    delta = float(geom.detector_offset_x)
    L = float(geom.center_to_pinhole)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        D = alpha_arr * s + c + eta_arr * (L * (1.0 - c) - delta * s)
        eta_t = eta_arr / D
        alpha_t = (
            alpha_arr * c
            - s
            + eta_arr * (L * s + delta * (1.0 - c))
        ) / D
        beta_t = beta_arr / D

    valid = (
        np.isfinite(D)
        & np.isfinite(alpha_t)
        & np.isfinite(beta_t)
        & np.isfinite(eta_t)
        & (D > float(eps))
        & (eta_t > float(eps))
    )
    if detector_geometry is not None:
        det = _detector_geometry_or_default(detector_geometry)
        u_t = float(geom.detector_to_pinhole) * alpha_t
        v_t = float(geom.detector_to_pinhole) * beta_t
        row_t, col_t = detector_continuous_indices(
            u_t,
            v_t,
            detector_shape=det.detector_shape,
            detector_pixel_size=det.detector_pixel_size,
        )
        height, width = det.detector_shape
        valid &= (
            np.isfinite(row_t)
            & np.isfinite(col_t)
            & (row_t >= 0.0)
            & (row_t <= float(height - 1))
            & (col_t >= 0.0)
            & (col_t <= float(width - 1))
        )

    return (
        np.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(beta_t, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(eta_t, nan=0.0, posinf=0.0, neginf=0.0),
        valid,
    )


def geometry_weight_ratio(
    alpha: np.ndarray,
    beta: np.ndarray,
    eta: np.ndarray,
    alpha_t: np.ndarray,
    beta_t: np.ndarray,
    eta_t: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Return pure-geometry source weight ratio along a lifted characteristic."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w_i = np.asarray(eta, dtype=np.float64) ** 2
        w_i = w_i / (1.0 + np.asarray(alpha) ** 2 + np.asarray(beta) ** 2) ** 1.5
        w_t = np.asarray(eta_t, dtype=np.float64) ** 2
        w_t = w_t / (
            1.0 + np.asarray(alpha_t) ** 2 + np.asarray(beta_t) ** 2
        ) ** 1.5
        ratio = w_t / (w_i + float(eps))
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(ratio, 0.0)


def two_point_positive_quadrature(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive eta quadrature nodes and normalized pixel weights."""
    g0 = _safe_nonnegative(G0)
    m, var = compute_moments_mean_var(g0, G1, G2, eps=eps)
    nonzero = g0 > float(eps)
    m = np.where(nonzero, np.maximum(m, float(eps)), float(eps))
    var = np.where(nonzero, np.maximum(var, 0.0), 0.0)

    near_zero = (var <= max(1.0e-14, float(eps))) | (m <= float(eps))
    if np.all(near_zero):
        return m[None, ...], np.where(nonzero, 1.0, 0.0)[None, ...]

    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.minimum(np.sqrt(var), 0.9 * m)
        d = np.where(near_zero, 0.0, np.maximum(d, float(eps)))
        eta1 = np.maximum(m - d, float(eps))
        eta2 = np.maximum(m + var / (d + float(eps)), float(eps))
        p1 = var / (var + d**2 + float(eps))
        p2 = 1.0 - p1

    p1 = np.where(near_zero, 1.0, p1)
    p2 = np.where(near_zero, 0.0, p2)
    p1 = np.where(nonzero, np.maximum(p1, 0.0), 0.0)
    p2 = np.where(nonzero, np.maximum(p2, 0.0), 0.0)
    norm = p1 + p2
    p1_norm = np.zeros_like(p1)
    p2_norm = np.zeros_like(p2)
    np.divide(p1, norm, out=p1_norm, where=norm > float(eps))
    np.divide(p2, norm, out=p2_norm, where=norm > float(eps))
    p1, p2 = p1_norm, p2_norm
    eta1 = np.where(nonzero, eta1, float(eps))
    eta2 = np.where(nonzero, eta2, float(eps))

    nodes = np.stack(
        [
            np.nan_to_num(eta1, nan=float(eps), posinf=float(eps), neginf=float(eps)),
            np.nan_to_num(eta2, nan=float(eps), posinf=float(eps), neginf=float(eps)),
        ],
        axis=0,
    )
    weights = np.stack(
        [
            np.nan_to_num(p1, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(p2, nan=0.0, posinf=0.0, neginf=0.0),
        ],
        axis=0,
    )
    return np.maximum(nodes, float(eps)), np.maximum(weights, 0.0)


def measured_count_anchored_moments(
    y_measured: np.ndarray,
    G0_model: np.ndarray,
    G1_model: np.ndarray,
    G2_model: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use measured counts as G0 while borrowing model conditional depth moments."""
    y = _safe_nonnegative(y_measured)
    g0 = _safe_nonnegative(G0_model)
    m_model, var_model = compute_moments_mean_var(g0, G1_model, G2_model, eps=eps)
    valid_model = g0 > float(eps)

    if np.any(valid_model):
        fallback_m = float(np.median(m_model[valid_model]))
        fallback_var = float(np.median(var_model[valid_model]))
    else:
        fallback_m = 1.0
        fallback_var = 0.0
    fallback_m = max(fallback_m, float(eps))
    fallback_var = max(fallback_var, 0.0)

    m = np.where(valid_model, m_model, fallback_m)
    var = np.where(valid_model, var_model, fallback_var)
    g1_anchor = y * m
    g2_anchor = y * (m**2 + np.maximum(var, 0.0))
    return (
        np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(g1_anchor, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(g2_anchor, nan=0.0, posinf=0.0, neginf=0.0),
    )


def splat_mass_to_detector(
    u_target: np.ndarray,
    v_target: np.ndarray,
    mass: np.ndarray,
    detector_geometry: DetectorGeometry | dict[str, Any] | None,
    mode: str = "bilinear",
) -> np.ndarray:
    """Bilinearly splat detector mass to target detector pixels."""
    det = _detector_geometry_or_default(detector_geometry)
    if str(mode).lower() != "bilinear":
        raise ValueError("only bilinear splatting is currently supported.")

    row, col = detector_continuous_indices(
        u_target,
        v_target,
        detector_shape=det.detector_shape,
        detector_pixel_size=det.detector_pixel_size,
    )
    values = np.asarray(mass, dtype=np.float64)
    if det.values_are_density:
        values = values * float(det.detector_pixel_size) ** 2
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    output = np.zeros(det.detector_shape, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64).ravel()
    col = np.asarray(col, dtype=np.float64).ravel()
    values = values.ravel()
    valid = np.isfinite(row) & np.isfinite(col) & np.isfinite(values) & (values != 0.0)
    if not np.any(valid):
        return output

    row = row[valid]
    col = col[valid]
    values = values[valid]
    row_floor = np.floor(row).astype(np.int64)
    col_floor = np.floor(col).astype(np.int64)
    row_frac = row - row_floor
    col_frac = col - col_floor
    height, width = det.detector_shape

    for row_offset, row_weight in ((0, 1.0 - row_frac), (1, row_frac)):
        rr = row_floor + row_offset
        valid_row = (rr >= 0) & (rr < height)
        for col_offset, col_weight in ((0, 1.0 - col_frac), (1, col_frac)):
            cc = col_floor + col_offset
            valid_pixel = valid_row & (cc >= 0) & (cc < width)
            if np.any(valid_pixel):
                np.add.at(
                    output,
                    (rr[valid_pixel], cc[valid_pixel]),
                    values[valid_pixel]
                    * row_weight[valid_pixel]
                    * col_weight[valid_pixel],
                )
    if det.values_are_density:
        output = output / (float(det.detector_pixel_size) ** 2)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def _quadrature_nodes(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    quadrature: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(quadrature).lower()
    if mode == "two_point":
        return two_point_positive_quadrature(G0, G1, G2, eps=eps)
    if mode == "one_point":
        g0 = _safe_nonnegative(G0)
        m, _ = compute_moments_mean_var(g0, G1, G2, eps=eps)
        nodes = np.maximum(m, float(eps))[None, ...]
        weights = np.where(g0 > float(eps), 1.0, 0.0)[None, ...]
        return nodes, weights
    raise ValueError("quadrature must be 'one_point' or 'two_point'.")


def characteristic_pushforward_completion(
    G0: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    theta_source: float,
    theta_target: float,
    geometry: PinholeGeometry | None = None,
    source_mode: str = "geo",
    quadrature: str = "two_point",
    detector_geometry: DetectorGeometry | dict[str, Any] | None = None,
    eps: float = EPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Push G0/G1/G2 to a target angle with finite-angle lifted characteristics."""
    geom = _geometry_or_default(geometry)
    det = _detector_geometry_or_default(detector_geometry, fallback_shape=G0.shape)
    u_grid, v_grid, det = _detector_grids(det)
    yd = float(geom.detector_to_pinhole)
    if abs(yd) <= float(eps):
        raise ValueError("geometry.detector_to_pinhole must be nonzero.")
    alpha = u_grid / yd
    beta = v_grid / yd
    g0 = _safe_nonnegative(G0)
    nodes, weights = _quadrature_nodes(g0, G1, G2, quadrature, eps)
    delta_theta = float(theta_target) - float(theta_source)

    output = np.zeros(det.detector_shape, dtype=np.float64)
    input_mass = float(np.sum(g0))
    preclip_mass = 0.0
    valid_mass = 0.0
    ratio_weighted_sum = 0.0
    ratio_mass_sum = 0.0
    max_ratio = 0.0

    for node, node_weight in zip(nodes, weights):
        node_mass = g0 * np.asarray(node_weight, dtype=np.float64)
        if not np.any(node_mass > eps):
            continue
        alpha_t, beta_t, eta_t, valid = finite_angle_lifted_map(
            alpha,
            beta,
            node,
            delta_theta,
            geom,
            eps=eps,
            detector_geometry=det,
        )
        mode = str(source_mode).lower()
        if mode in {"none", "off", "zero", "no_source"}:
            ratio = np.ones_like(g0, dtype=np.float64)
        elif mode in {"geo", "geometry", "pure_geo"}:
            ratio = geometry_weight_ratio(
                alpha,
                beta,
                node,
                alpha_t,
                beta_t,
                eta_t,
                eps=eps,
            )
        else:
            raise ValueError(f"unsupported source_mode: {source_mode!r}")

        valid = valid & np.isfinite(ratio)
        weighted = np.where(valid, node_mass * ratio, 0.0)
        valid_mass += float(np.sum(np.where(valid, node_mass, 0.0)))
        preclip_mass += float(np.sum(weighted))
        ratio_weighted_sum += float(np.sum(np.where(valid, node_mass * ratio, 0.0)))
        ratio_mass_sum += float(np.sum(np.where(valid, node_mass, 0.0)))
        if np.any(valid):
            max_ratio = max(max_ratio, float(np.max(ratio[valid])))

        u_t = yd * alpha_t
        v_t = yd * beta_t
        output += splat_mass_to_detector(u_t, v_t, weighted, det)

    output_mass = float(np.sum(output))
    clipped_fraction = (
        max(0.0, 1.0 - output_mass / (preclip_mass + float(eps)))
        if preclip_mass > float(eps)
        else 0.0
    )
    diagnostics = {
        "input_mass": input_mass,
        "output_mass": output_mass,
        "clipped_mass_fraction": clipped_fraction,
        "mean_weight_ratio": (
            ratio_weighted_sum / (ratio_mass_sum + float(eps))
            if ratio_mass_sum > float(eps)
            else float("nan")
        ),
        "max_weight_ratio": max_ratio,
        "valid_fraction": (
            valid_mass / (input_mass + float(eps)) if input_mass > float(eps) else 0.0
        ),
    }
    return output, diagnostics


def two_endpoint_characteristic_completion(
    G0_left: np.ndarray,
    G1_left: np.ndarray,
    G2_left: np.ndarray,
    theta_left: float,
    G0_right: np.ndarray,
    G1_right: np.ndarray,
    G2_right: np.ndarray,
    theta_right: float,
    theta_target: float,
    geometry: PinholeGeometry | None = None,
    source_mode: str = "geo",
    quadrature: str = "two_point",
    detector_geometry: DetectorGeometry | dict[str, Any] | None = None,
    eps: float = EPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend characteristic predictions from two measured angular endpoints."""
    pred_left, diag_left = characteristic_pushforward_completion(
        G0_left,
        G1_left,
        G2_left,
        theta_left,
        theta_target,
        geometry=geometry,
        source_mode=source_mode,
        quadrature=quadrature,
        detector_geometry=detector_geometry,
        eps=eps,
    )
    pred_right, diag_right = characteristic_pushforward_completion(
        G0_right,
        G1_right,
        G2_right,
        theta_right,
        theta_target,
        geometry=geometry,
        source_mode=source_mode,
        quadrature=quadrature,
        detector_geometry=detector_geometry,
        eps=eps,
    )
    dist_left = abs(float(theta_target) - float(theta_left))
    dist_right = abs(float(theta_right) - float(theta_target))
    total = dist_left + dist_right
    if total <= float(eps):
        w_left, w_right = 1.0, 0.0
    else:
        w_left = dist_right / total
        w_right = dist_left / total
    pred = w_left * pred_left + w_right * pred_right
    keys = (
        "input_mass",
        "output_mass",
        "clipped_mass_fraction",
        "mean_weight_ratio",
        "max_weight_ratio",
        "valid_fraction",
    )
    diag = {
        key: (
            w_left * float(diag_left.get(key, 0.0))
            + w_right * float(diag_right.get(key, 0.0))
        )
        for key in keys
        if key not in {"max_weight_ratio"}
    }
    diag["max_weight_ratio"] = max(
        float(diag_left.get("max_weight_ratio", 0.0)),
        float(diag_right.get("max_weight_ratio", 0.0)),
    )
    diag["left_weight"] = float(w_left)
    diag["right_weight"] = float(w_right)
    return pred, diag
