from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.flux_projector import voxel_center_xyz
from src.xfct_geometry import (
    PinholeGeometry,
    detector_continuous_indices,
    detector_physical_coordinates,
    finite_angle_map,
    finite_angle_geometry_ratio,
    geometry_weight,
    uv,
    xprime_yprime,
)

EPS = 1e-12


def _phase_correlation_shift(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src = src - float(np.mean(src))
    dst = dst - float(np.mean(dst))

    cross_power = np.fft.fft2(dst) * np.conj(np.fft.fft2(src))
    cross_power /= np.maximum(np.abs(cross_power), EPS)
    correlation = np.fft.ifft2(cross_power)
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    shift = np.asarray(peak, dtype=np.float64)
    shape = np.asarray(src.shape, dtype=np.float64)
    midpoint = np.floor(shape / 2.0)
    shift[shift > midpoint] -= shape[shift > midpoint]
    return float(shift[0]), float(shift[1])


def _interpolate_pair(
    current: np.ndarray,
    next_projection: np.ndarray,
    alpha: float,
    method: str,
) -> np.ndarray:
    if method == "linear":
        return (1.0 - alpha) * current + alpha * next_projection

    if method == "phase_shift":
        shift_y, shift_x = _phase_correlation_shift(current, next_projection)
        current_warped = ndimage.shift(
            current,
            shift=(alpha * shift_y, alpha * shift_x),
            order=1,
            mode="nearest",
        )
        next_warped = ndimage.shift(
            next_projection,
            shift=(-(1.0 - alpha) * shift_y, -(1.0 - alpha) * shift_x),
            order=1,
            mode="nearest",
        )
        return (1.0 - alpha) * current_warped + alpha * next_warped

    raise ValueError(f"unsupported projection completion method: {method}")


def complete_projection_angles(
    sparse_projection: np.ndarray,
    upsample_factor: int = 3,
    method: str = "linear",
    clip_negative: bool = True,
) -> np.ndarray:
    projection = np.asarray(sparse_projection, dtype=np.float64)
    if projection.ndim != 3:
        raise ValueError(
            "sparse_projection must have shape [angle, height, width], "
            f"got {projection.shape}."
        )
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1.")
    if upsample_factor == 1:
        return projection.copy()

    method = str(method).lower()
    n_angles = projection.shape[0]
    completed = np.zeros(
        (n_angles * upsample_factor,) + projection.shape[1:],
        dtype=np.float64,
    )

    for angle_idx in range(n_angles):
        current = projection[angle_idx]
        next_projection = projection[(angle_idx + 1) % n_angles]
        base_idx = angle_idx * upsample_factor
        completed[base_idx] = current
        for sub_idx in range(1, upsample_factor):
            alpha = sub_idx / float(upsample_factor)
            completed[base_idx + sub_idx] = _interpolate_pair(
                current=current,
                next_projection=next_projection,
                alpha=alpha,
                method=method,
            )

    if clip_negative:
        completed = np.maximum(completed, 0.0)
    return completed


def completed_projection_weights(
    completed_shape: tuple[int, int, int],
    upsample_factor: int = 3,
    pseudo_weight: float = 0.1,
) -> np.ndarray:
    if len(completed_shape) != 3:
        raise ValueError(f"completed_shape must be 3D, got {completed_shape}.")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1.")
    weights = np.full(completed_shape, float(pseudo_weight), dtype=np.float64)
    weights[::upsample_factor] = 1.0
    return np.maximum(weights, 0.0)


def reconstruction_informed_completion(
    sparse_projection: np.ndarray,
    sparse_forward_projection: np.ndarray,
    target_forward_projection: np.ndarray,
    upsample_factor: int = 3,
    method: str = "linear",
    blend_multiplicative: float = 0.5,
    uncertainty_floor: float = 1.0,
    pseudo_weight_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    sparse = np.asarray(sparse_projection, dtype=np.float64)
    sparse_forward = np.asarray(sparse_forward_projection, dtype=np.float64)
    target_forward = np.asarray(target_forward_projection, dtype=np.float64)
    if sparse.shape != sparse_forward.shape:
        raise ValueError(
            "sparse_projection and sparse_forward_projection shape mismatch: "
            f"{sparse.shape} vs {sparse_forward.shape}."
        )
    expected_target_shape = (sparse.shape[0] * upsample_factor,) + sparse.shape[1:]
    if target_forward.shape != expected_target_shape:
        raise ValueError(
            "target_forward_projection shape mismatch: "
            f"expected {expected_target_shape}, got {target_forward.shape}."
        )

    alpha = float(np.clip(blend_multiplicative, 0.0, 1.0))
    additive_residual = sparse - sparse_forward
    residual_hat = complete_projection_angles(
        additive_residual,
        upsample_factor=upsample_factor,
        method=method,
        clip_negative=False,
    )

    ratio = (sparse + EPS) / (sparse_forward + EPS)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, 0.0, 10.0)
    ratio_hat = complete_projection_angles(
        ratio,
        upsample_factor=upsample_factor,
        method=method,
        clip_negative=False,
    )
    ratio_hat = np.clip(ratio_hat, 0.0, 10.0)

    additive_prediction = target_forward + residual_hat
    multiplicative_prediction = target_forward * ratio_hat
    completed = alpha * multiplicative_prediction + (1.0 - alpha) * additive_prediction
    completed = np.maximum(completed, 0.0)
    completed[::upsample_factor] = sparse

    residual_scale = complete_projection_angles(
        np.abs(additive_residual),
        upsample_factor=upsample_factor,
        method="linear",
        clip_negative=True,
    )
    disagreement = multiplicative_prediction - additive_prediction
    sigma2 = (
        np.maximum(completed, 1.0)
        + residual_scale**2
        + disagreement**2
        + float(uncertainty_floor) ** 2
    )
    weights = float(pseudo_weight_scale) / np.maximum(sigma2, EPS)
    weights[::upsample_factor] = 0.0

    return {
        "projection": completed,
        "weights": weights,
        "sigma": np.sqrt(np.maximum(sigma2, EPS)),
        "additive_prediction": np.maximum(additive_prediction, 0.0),
        "multiplicative_prediction": np.maximum(multiplicative_prediction, 0.0),
        "residual_hat": residual_hat,
        "ratio_hat": ratio_hat,
    }


def forward_projection_completion(
    sparse_projection: np.ndarray,
    target_forward_projection: np.ndarray,
    upsample_factor: int = 3,
    uncertainty_floor: float = 1.0,
    pseudo_weight_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    sparse = np.asarray(sparse_projection, dtype=np.float64)
    target_forward = np.asarray(target_forward_projection, dtype=np.float64)
    expected_target_shape = (sparse.shape[0] * upsample_factor,) + sparse.shape[1:]
    if target_forward.shape != expected_target_shape:
        raise ValueError(
            "target_forward_projection shape mismatch: "
            f"expected {expected_target_shape}, got {target_forward.shape}."
        )

    completed = np.maximum(target_forward.copy(), 0.0)
    completed[::upsample_factor] = sparse
    sigma2 = np.maximum(completed, 1.0) + float(uncertainty_floor) ** 2
    weights = float(pseudo_weight_scale) / np.maximum(sigma2, EPS)
    weights[::upsample_factor] = 0.0
    residual_hat = np.zeros_like(completed)
    ratio_hat = np.ones_like(completed)

    return {
        "projection": completed,
        "weights": weights,
        "sigma": np.sqrt(np.maximum(sigma2, EPS)),
        "additive_prediction": completed.copy(),
        "multiplicative_prediction": completed.copy(),
        "residual_hat": residual_hat,
        "ratio_hat": ratio_hat,
    }


def _splat_bilinear(
    values: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    output = np.zeros(output_shape, dtype=np.float64)
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


def _depth_bin_fraction_maps(
    coarse_reconstruction: np.ndarray,
    angles: np.ndarray,
    detector_shape: tuple[int, int],
    recon_shape: tuple[int, int, int],
    voxel_size: float,
    detector_pixel_size: float,
    geometry: PinholeGeometry,
    depth_bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    volume = np.maximum(np.asarray(coarse_reconstruction, dtype=np.float64), 0.0)
    xyz = voxel_center_xyz(recon_shape, voxel_size)
    f = volume.ravel()
    if np.max(f) > 0.0:
        active = f > (float(np.max(f)) * 1.0e-6)
    else:
        active = np.ones_like(f, dtype=bool)
    xyz = xyz[active]
    f = f[active]

    n_angles = len(angles)
    fractions = np.zeros(
        (n_angles, depth_bin_count) + detector_shape,
        dtype=np.float64,
    )
    depth_centers = np.zeros((n_angles, depth_bin_count), dtype=np.float64)

    for angle_idx, theta in enumerate(angles):
        _, depth_lambda = xprime_yprime(float(theta), xyz, geometry)
        weights = f * np.maximum(geometry_weight(float(theta), xyz, geometry), 0.0)
        if not np.any(weights > 0.0):
            depth_centers[angle_idx] = geometry.center_to_pinhole
            fractions[angle_idx, 0] = 1.0
            continue

        depth_min = float(np.min(depth_lambda))
        depth_max = float(np.max(depth_lambda))
        if depth_max <= depth_min + EPS:
            depth_max = depth_min + 1.0
        edges = np.linspace(depth_min, depth_max, depth_bin_count + 1)
        depth_centers[angle_idx] = 0.5 * (edges[:-1] + edges[1:])

        u_vox, v_vox = uv(float(theta), xyz, geometry)
        row, col = detector_continuous_indices(
            u_vox,
            v_vox,
            detector_shape=detector_shape,
            detector_pixel_size=detector_pixel_size,
        )
        total = np.zeros(detector_shape, dtype=np.float64)
        bin_maps = []
        bin_indices = np.clip(
            np.searchsorted(edges, depth_lambda, side="right") - 1,
            0,
            depth_bin_count - 1,
        )
        for bin_idx in range(depth_bin_count):
            mask = bin_indices == bin_idx
            if np.any(mask):
                bin_map = _splat_bilinear(
                    weights[mask],
                    row[mask],
                    col[mask],
                    detector_shape,
                )
            else:
                bin_map = np.zeros(detector_shape, dtype=np.float64)
            bin_maps.append(bin_map)
            total += bin_map

        total = np.maximum(total, EPS)
        for bin_idx, bin_map in enumerate(bin_maps):
            fractions[angle_idx, bin_idx] = bin_map / total

    return fractions, depth_centers


def _warp_depth_weighted_image(
    image: np.ndarray,
    fractions: np.ndarray,
    depth_centers: np.ndarray,
    delta_theta: float,
    geometry: PinholeGeometry,
    detector_pixel_size: float,
    fill_value: float,
    normalize_by_coverage: bool,
    apply_geometry_ratio: bool = False,
) -> np.ndarray:
    detector_shape = image.shape
    rows, cols = np.meshgrid(
        np.arange(detector_shape[0], dtype=np.float64),
        np.arange(detector_shape[1], dtype=np.float64),
        indexing="ij",
    )
    u_grid, v_grid = detector_physical_coordinates(
        rows,
        cols,
        detector_shape=detector_shape,
        detector_pixel_size=detector_pixel_size,
    )

    output = np.zeros(detector_shape, dtype=np.float64)
    coverage = np.zeros(detector_shape, dtype=np.float64)
    for bin_idx, depth in enumerate(depth_centers):
        frac = np.asarray(fractions[bin_idx], dtype=np.float64)
        u_warped, v_warped = finite_angle_map(
            u_grid,
            v_grid,
            np.full(detector_shape, float(depth), dtype=np.float64),
            float(delta_theta),
            geometry,
        )
        values = np.asarray(image, dtype=np.float64) * frac
        if apply_geometry_ratio:
            ratio = finite_angle_geometry_ratio(
                u_grid,
                v_grid,
                np.full(detector_shape, float(depth), dtype=np.float64),
                float(delta_theta),
                geometry,
            )
            values = values * np.maximum(np.nan_to_num(ratio), 0.0)
        row_warped, col_warped = detector_continuous_indices(
            u_warped,
            v_warped,
            detector_shape=detector_shape,
            detector_pixel_size=detector_pixel_size,
        )
        output += _splat_bilinear(
            values,
            row_warped,
            col_warped,
            detector_shape,
        )
        coverage += _splat_bilinear(frac, row_warped, col_warped, detector_shape)

    if normalize_by_coverage:
        return np.where(coverage > EPS, output / np.maximum(coverage, EPS), fill_value)
    return np.where(coverage > EPS, output, fill_value)


def depth_aware_reconstruction_completion(
    sparse_projection: np.ndarray,
    sparse_forward_projection: np.ndarray,
    target_forward_projection: np.ndarray,
    coarse_reconstruction: np.ndarray,
    recon_shape: tuple[int, int, int],
    upsample_factor: int = 3,
    blend_multiplicative: float = 0.5,
    uncertainty_floor: float = 1.0,
    pseudo_weight_scale: float = 1.0,
    depth_bin_count: int = 6,
    voxel_size: float = 0.5,
    detector_pixel_size: float = 0.25,
    geometry: PinholeGeometry | None = None,
    residual_strength: float = 0.25,
) -> dict[str, np.ndarray]:
    sparse = np.asarray(sparse_projection, dtype=np.float64)
    sparse_forward = np.asarray(sparse_forward_projection, dtype=np.float64)
    target_forward = np.asarray(target_forward_projection, dtype=np.float64)
    if sparse.shape != sparse_forward.shape:
        raise ValueError(
            "sparse_projection and sparse_forward_projection shape mismatch: "
            f"{sparse.shape} vs {sparse_forward.shape}."
        )
    expected_target_shape = (sparse.shape[0] * upsample_factor,) + sparse.shape[1:]
    if target_forward.shape != expected_target_shape:
        raise ValueError(
            "target_forward_projection shape mismatch: "
            f"expected {expected_target_shape}, got {target_forward.shape}."
        )

    geom = geometry or PinholeGeometry(
        detector_to_pinhole=-30.0,
        center_to_pinhole=50.0,
        detector_offset_x=-0.5,
    )
    sparse_angles = 2.0 * np.pi * np.arange(sparse.shape[0], dtype=np.float64)
    sparse_angles = sparse_angles / float(sparse.shape[0])
    sparse_step = 2.0 * np.pi / float(sparse.shape[0])
    fractions, depth_centers = _depth_bin_fraction_maps(
        coarse_reconstruction=coarse_reconstruction,
        angles=sparse_angles,
        detector_shape=tuple(sparse.shape[1:]),
        recon_shape=tuple(recon_shape),
        voxel_size=float(voxel_size),
        detector_pixel_size=float(detector_pixel_size),
        geometry=geom,
        depth_bin_count=max(1, int(depth_bin_count)),
    )

    additive_residual = sparse - sparse_forward
    ratio = (sparse + EPS) / (sparse_forward + EPS)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, 0.0, 10.0)

    residual_hat = np.zeros(expected_target_shape, dtype=np.float64)
    ratio_hat = np.ones(expected_target_shape, dtype=np.float64)
    alpha_mix = float(np.clip(blend_multiplicative, 0.0, 1.0))
    residual_gain = float(np.clip(residual_strength, 0.0, 1.0))

    for angle_idx in range(sparse.shape[0]):
        base_idx = angle_idx * upsample_factor
        next_idx = (angle_idx + 1) % sparse.shape[0]
        residual_hat[base_idx] = additive_residual[angle_idx]
        ratio_hat[base_idx] = ratio[angle_idx]
        for sub_idx in range(1, upsample_factor):
            alpha = sub_idx / float(upsample_factor)
            target_idx = base_idx + sub_idx
            current_residual = _warp_depth_weighted_image(
                additive_residual[angle_idx],
                fractions[angle_idx],
                depth_centers[angle_idx],
                alpha * sparse_step,
                geom,
                detector_pixel_size,
                fill_value=0.0,
                normalize_by_coverage=False,
                apply_geometry_ratio=True,
            )
            next_residual = _warp_depth_weighted_image(
                additive_residual[next_idx],
                fractions[next_idx],
                depth_centers[next_idx],
                -(1.0 - alpha) * sparse_step,
                geom,
                detector_pixel_size,
                fill_value=0.0,
                normalize_by_coverage=False,
                apply_geometry_ratio=True,
            )
            current_ratio = _warp_depth_weighted_image(
                ratio[angle_idx],
                fractions[angle_idx],
                depth_centers[angle_idx],
                alpha * sparse_step,
                geom,
                detector_pixel_size,
                fill_value=1.0,
                normalize_by_coverage=True,
                apply_geometry_ratio=False,
            )
            next_ratio = _warp_depth_weighted_image(
                ratio[next_idx],
                fractions[next_idx],
                depth_centers[next_idx],
                -(1.0 - alpha) * sparse_step,
                geom,
                detector_pixel_size,
                fill_value=1.0,
                normalize_by_coverage=True,
                apply_geometry_ratio=False,
            )
            residual_hat[target_idx] = (
                (1.0 - alpha) * current_residual + alpha * next_residual
            )
            ratio_hat[target_idx] = (1.0 - alpha) * current_ratio + alpha * next_ratio

    ratio_hat = np.clip(ratio_hat, 0.0, 10.0)
    additive_prediction = target_forward + residual_gain * residual_hat
    ratio_hat = 1.0 + residual_gain * (ratio_hat - 1.0)
    ratio_hat = np.clip(ratio_hat, 0.0, 10.0)
    multiplicative_prediction = target_forward * ratio_hat
    completed = (
        alpha_mix * multiplicative_prediction
        + (1.0 - alpha_mix) * additive_prediction
    )
    completed = np.maximum(completed, 0.0)
    completed[::upsample_factor] = sparse

    residual_scale = complete_projection_angles(
        np.abs(additive_residual),
        upsample_factor=upsample_factor,
        method="linear",
        clip_negative=True,
    )
    disagreement = multiplicative_prediction - additive_prediction
    sigma2 = (
        np.maximum(completed, 1.0)
        + residual_scale**2
        + disagreement**2
        + float(uncertainty_floor) ** 2
    )
    weights = float(pseudo_weight_scale) / np.maximum(sigma2, EPS)
    weights[::upsample_factor] = 0.0

    return {
        "projection": completed,
        "weights": weights,
        "sigma": np.sqrt(np.maximum(sigma2, EPS)),
        "additive_prediction": np.maximum(additive_prediction, 0.0),
        "multiplicative_prediction": np.maximum(multiplicative_prediction, 0.0),
        "residual_hat": residual_hat,
        "ratio_hat": ratio_hat,
        "depth_fractions": fractions,
        "depth_centers": depth_centers,
    }
