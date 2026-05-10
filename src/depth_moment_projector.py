from __future__ import annotations

from typing import Iterable

import numpy as np

from src.flux_projector import voxel_center_xyz
from src.lifted_projective_dynamics import EPS, compute_depth_variance, compute_eta
from src.xfct_geometry import (
    PinholeGeometry,
    detector_continuous_indices,
    geometry_weight,
    uv,
)


def _splat_bilinear(
    values: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    output = np.zeros(output_shape, dtype=np.float64)
    if values.size == 0:
        return output

    row_floor = np.floor(row).astype(np.int64)
    col_floor = np.floor(col).astype(np.int64)
    row_frac = row - row_floor
    col_frac = col - col_floor
    height, width = output_shape

    for row_offset, row_weight in ((0, 1.0 - row_frac), (1, row_frac)):
        rr = row_floor + row_offset
        valid_row = (rr >= 0) & (rr < height)
        for col_offset, col_weight in ((0, 1.0 - col_frac), (1, col_frac)):
            cc = col_floor + col_offset
            valid = valid_row & (cc >= 0) & (cc < width)
            if np.any(valid):
                np.add.at(
                    output,
                    (rr[valid], cc[valid]),
                    values[valid] * row_weight[valid] * col_weight[valid],
                )
    return output


def _moment_powers(powers: Iterable[int]) -> tuple[int, ...]:
    requested = {0, 1, 2}
    requested.update(int(power) for power in powers)
    if any(power < 0 for power in requested):
        raise ValueError("moment powers must be nonnegative integers.")
    return tuple(sorted(requested))


def _projection_weight(
    theta: float,
    xyz: np.ndarray,
    geometry: PinholeGeometry,
    weight_mode: str | None,
) -> np.ndarray:
    mode = "existing_or_geo" if weight_mode is None else str(weight_mode).lower()
    if mode in {
        "existing_or_geo",
        "geo_or_existing",
        "existing",
        "geo",
        "geometry",
        "geometry_weight",
        "pure_geo",
    }:
        weight = geometry_weight(theta, xyz, geometry)
    elif mode in {"none", "unit", "unweighted"}:
        weight = np.ones(xyz.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"unsupported depth moment weight_mode: {weight_mode!r}")
    return np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)


def project_depth_moments(
    volume: np.ndarray,
    theta: float,
    geometry: PinholeGeometry,
    powers: Iterable[int] = (0, 1, 2),
    weight_mode: str | None = "existing_or_geo",
    detector_shape: tuple[int, int] = (80, 160),
    voxel_size: float = 0.5,
    detector_pixel_size: float = 0.25,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Project lifted reciprocal-depth moments with geometry-consistent splatting.

    This is a diagnostic geometry-only projector. It follows the same voxel-center,
    pinhole projection, and detector-index conventions as ``src.xfct_geometry`` and
    ``src.flux_projector`` without changing the default reconstruction path.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"volume must have shape [z, y, x], got {vol.shape}.")
    if len(detector_shape) != 2:
        raise ValueError(
            f"detector_shape must be (height, width), got {detector_shape}."
        )
    if detector_pixel_size <= 0.0:
        raise ValueError("detector_pixel_size must be positive.")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")

    moment_powers = _moment_powers(powers)
    detector_shape = tuple(int(v) for v in detector_shape)
    maps = {
        f"G{power}": np.zeros(detector_shape, dtype=np.float64)
        for power in moment_powers
    }

    f = vol.ravel()
    active = np.isfinite(f) & (f != 0.0)
    if not np.any(active):
        g0 = maps["G0"]
        maps["m1"] = np.zeros_like(g0)
        maps["var_eta"] = np.zeros_like(g0)
        return maps

    xyz = voxel_center_xyz(tuple(vol.shape), float(voxel_size))[active]
    f = f[active]
    u_coord, v_coord = uv(float(theta), xyz, geometry)
    row, col = detector_continuous_indices(
        u_coord,
        v_coord,
        detector_shape=detector_shape,
        detector_pixel_size=float(detector_pixel_size),
    )
    valid_projection = np.isfinite(row) & np.isfinite(col)
    if not np.any(valid_projection):
        g0 = maps["G0"]
        maps["m1"] = np.zeros_like(g0)
        maps["var_eta"] = np.zeros_like(g0)
        return maps

    xyz = xyz[valid_projection]
    f = f[valid_projection]
    row = row[valid_projection]
    col = col[valid_projection]
    eta = np.nan_to_num(
        compute_eta(float(theta), xyz, geometry),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    base = f * _projection_weight(float(theta), xyz, geometry, weight_mode)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    for power in moment_powers:
        values = base if power == 0 else base * eta**power
        maps[f"G{power}"] = _splat_bilinear(
            values=np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            row=row,
            col=col,
            output_shape=detector_shape,
        )

    G0 = maps["G0"]
    G1 = maps["G1"]
    G2 = maps["G2"]
    maps["m1"] = np.nan_to_num(
        np.divide(G1, G0 + float(eps)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    maps["var_eta"] = compute_depth_variance(G0, G1, G2, eps=eps)
    return maps
