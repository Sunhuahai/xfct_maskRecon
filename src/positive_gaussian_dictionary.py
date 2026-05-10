from __future__ import annotations

from collections.abc import Iterable

import numpy as np

EPS = 1e-12


def _shape_zyx(volume_shape: tuple[int, int, int] | Iterable[int]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in volume_shape)
    if len(shape) != 3:
        raise ValueError(f"volume_shape must be (Z, Y, X), got {volume_shape}.")
    if any(value <= 0 for value in shape):
        raise ValueError(f"volume_shape entries must be positive, got {shape}.")
    return shape


def _spacing_zyx(
    voxel_spacing: float | tuple[float, float, float] | Iterable[float],
) -> tuple[float, float, float]:
    if np.isscalar(voxel_spacing):
        spacing = (float(voxel_spacing),) * 3
    else:
        spacing = tuple(float(value) for value in voxel_spacing)
    if len(spacing) != 3:
        raise ValueError(
            "voxel_spacing must be a scalar or (Z, Y, X), "
            f"got {voxel_spacing}."
        )
    if any(value <= 0.0 for value in spacing):
        raise ValueError(f"voxel_spacing entries must be positive, got {spacing}.")
    return spacing


def _stride_zyx(grid_stride: int | tuple[int, int, int] | Iterable[int]) -> tuple[int, int, int]:
    if np.isscalar(grid_stride):
        stride = (int(grid_stride),) * 3
    else:
        stride = tuple(int(value) for value in grid_stride)
    if len(stride) != 3:
        raise ValueError(
            f"grid_stride must be a scalar or (Z, Y, X), got {grid_stride}."
        )
    if any(value <= 0 for value in stride):
        raise ValueError(f"grid_stride entries must be positive, got {stride}.")
    return stride


def _sigma_xyz(
    sigma: float | tuple[float, float, float] | Iterable[float],
) -> tuple[float, float, float]:
    if np.isscalar(sigma):
        values = (float(sigma),) * 3
    else:
        values = tuple(float(value) for value in sigma)
        if len(values) == 1:
            values = (values[0],) * 3
    if len(values) != 3:
        raise ValueError(f"sigma must be scalar or (sigma_x, sigma_y, sigma_z), got {sigma}.")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"sigma entries must be positive, got {values}.")
    return values


def _sigmas_xyz(
    sigmas: float | Iterable[float] | Iterable[Iterable[float]],
) -> list[tuple[float, float, float]]:
    if np.isscalar(sigmas):
        return [_sigma_xyz(float(sigmas))]
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim == 0:
        return [_sigma_xyz(float(arr))]
    if arr.ndim == 1:
        return [_sigma_xyz(float(value)) for value in arr]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [_sigma_xyz(row) for row in arr]
    raise ValueError(
        "sigmas must be a scalar, a list of scalar sigmas, or an array of "
        f"(sigma_x, sigma_y, sigma_z) rows, got shape {arr.shape}."
    )


def volume_coordinate_grid(
    volume_shape: tuple[int, int, int],
    voxel_spacing: float | tuple[float, float, float],
) -> np.ndarray:
    """Return a centered coordinate grid with last axis ordered as (x, y, z)."""
    z_size, y_size, x_size = _shape_zyx(volume_shape)
    z_spacing, y_spacing, x_spacing = _spacing_zyx(voxel_spacing)
    z_axis = (np.arange(z_size, dtype=np.float64) - (z_size - 1.0) / 2.0) * z_spacing
    y_axis = (np.arange(y_size, dtype=np.float64) - (y_size - 1.0) / 2.0) * y_spacing
    x_axis = (np.arange(x_size, dtype=np.float64) - (x_size - 1.0) / 2.0) * x_spacing
    zz, yy, xx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    return np.stack((xx, yy, zz), axis=-1)


def _coerce_volume_grid(volume_grid: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]):
    if isinstance(volume_grid, tuple):
        if len(volume_grid) != 3:
            raise ValueError("volume_grid tuple must be (x_grid, y_grid, z_grid).")
        return tuple(np.asarray(component, dtype=np.float64) for component in volume_grid)
    grid = np.asarray(volume_grid, dtype=np.float64)
    if grid.ndim < 4 or grid.shape[-1] != 3:
        raise ValueError("volume_grid must have shape [..., 3] with (x, y, z).")
    return grid[..., 0], grid[..., 1], grid[..., 2]


def voxelize_gaussian_atom(
    center: tuple[float, float, float] | np.ndarray,
    sigma: float | tuple[float, float, float] | Iterable[float],
    volume_grid: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Voxelize one nonnegative Gaussian atom before dictionary normalization.

    ``center`` and anisotropic ``sigma`` use physical coordinate order
    ``(x, y, z)``. The returned atom is nonnegative and has the same ``(Z, Y, X)``
    shape as the supplied grid.
    """
    cx, cy, cz = np.asarray(center, dtype=np.float64).reshape(3)
    sx, sy, sz = _sigma_xyz(sigma)
    x_grid, y_grid, z_grid = _coerce_volume_grid(volume_grid)
    exponent = (
        ((x_grid - cx) / sx) ** 2
        + ((y_grid - cy) / sy) ** 2
        + ((z_grid - cz) / sz) ** 2
    )
    atom = np.exp(-0.5 * exponent)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool)
        if mask.shape != atom.shape:
            raise ValueError(
                f"support_mask shape {mask.shape} does not match atom shape {atom.shape}."
            )
        atom = np.where(mask, atom, 0.0)
    return np.maximum(np.nan_to_num(atom, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _candidate_indices(
    volume_shape: tuple[int, int, int],
    grid_stride: int | tuple[int, int, int],
    support_mask: np.ndarray | None,
) -> list[tuple[int, int, int]]:
    shape = _shape_zyx(volume_shape)
    stride = _stride_zyx(grid_stride)
    z_indices = np.arange(0, shape[0], stride[0], dtype=np.int64)
    y_indices = np.arange(0, shape[1], stride[1], dtype=np.int64)
    x_indices = np.arange(0, shape[2], stride[2], dtype=np.int64)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"support_mask shape {mask.shape} does not match {shape}.")
    else:
        mask = None

    indices = []
    for z_idx in z_indices:
        for y_idx in y_indices:
            for x_idx in x_indices:
                if mask is None or bool(mask[z_idx, y_idx, x_idx]):
                    indices.append((int(z_idx), int(y_idx), int(x_idx)))
    if not indices:
        raise ValueError("no Gaussian atom centers remain after support pruning.")
    return indices


def _normalize_atom(atom: np.ndarray, normalize: str) -> tuple[np.ndarray, float]:
    mode = str(normalize).strip().lower()
    if mode == "sum":
        norm = float(np.sum(atom))
        if norm > EPS:
            return atom / norm, norm
        return atom.copy(), 0.0
    if mode == "max":
        norm = float(np.max(atom))
        if norm > EPS:
            return atom / norm, norm
        return atom.copy(), 0.0
    if mode in {"none", "raw"}:
        return atom.copy(), 1.0
    raise ValueError("normalize must be one of 'sum', 'max', or 'none'.")


def create_gaussian_atoms(
    volume_shape: tuple[int, int, int],
    voxel_spacing: float | tuple[float, float, float],
    support_mask: np.ndarray | None = None,
    grid_stride: int | tuple[int, int, int] = 4,
    sigmas: float | Iterable[float] | Iterable[Iterable[float]] = (1.0,),
    normalize: str = "sum",
    dtype=np.float32,
    return_metadata: bool = False,
):
    """Create a positive Gaussian dictionary on a centered ``(Z, Y, X)`` grid.

    Normalization convention:
    - ``normalize="sum"`` makes each discrete atom satisfy ``sum_r phi_k(r)=1``.
    - ``normalize="max"`` makes each atom have unit peak value.
    - ``normalize="none"`` leaves raw sampled Gaussian values.
    """
    shape = _shape_zyx(volume_shape)
    grid = volume_coordinate_grid(shape, voxel_spacing)
    sigma_values = _sigmas_xyz(sigmas)
    indices = _candidate_indices(shape, grid_stride, support_mask)

    atoms = []
    metadata = []
    for z_idx, y_idx, x_idx in indices:
        center = grid[z_idx, y_idx, x_idx]
        for sigma_xyz in sigma_values:
            raw = voxelize_gaussian_atom(center, sigma_xyz, grid, support_mask)
            atom, normalization_value = _normalize_atom(raw, normalize)
            if np.max(atom) <= EPS:
                continue
            atoms.append(atom.astype(dtype, copy=False))
            metadata.append(
                {
                    "atom_index": len(metadata),
                    "center_z_index": int(z_idx),
                    "center_y_index": int(y_idx),
                    "center_x_index": int(x_idx),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "center_z": float(center[2]),
                    "sigma_x": float(sigma_xyz[0]),
                    "sigma_y": float(sigma_xyz[1]),
                    "sigma_z": float(sigma_xyz[2]),
                    "normalize": str(normalize),
                    "normalization_value": float(normalization_value),
                }
            )

    if not atoms:
        raise ValueError("all Gaussian atoms are empty after support masking.")
    atom_array = np.stack(atoms, axis=0)
    if return_metadata:
        return atom_array, metadata
    return atom_array


def reconstruct_from_amplitudes(
    atoms: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Return x(r) = sum_k a_k phi_k(r) for nonnegative amplitudes."""
    atom_array = np.asarray(atoms, dtype=np.float64)
    if atom_array.ndim != 4:
        raise ValueError(f"atoms must have shape [K, Z, Y, X], got {atom_array.shape}.")
    amps = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if amps.size != atom_array.shape[0]:
        raise ValueError(
            f"amplitude length {amps.size} does not match atom count {atom_array.shape[0]}."
        )
    if np.any(amps < -EPS):
        raise ValueError("reconstruct_from_amplitudes requires nonnegative amplitudes.")
    volume = np.tensordot(np.maximum(amps, 0.0), atom_array, axes=(0, 0))
    return np.maximum(np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def prune_atoms_by_support_or_projection_visibility(
    atoms: np.ndarray,
    support_mask: np.ndarray | None = None,
    projection_visibility: np.ndarray | None = None,
    min_visibility: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return atoms that overlap support and pass an optional visibility score."""
    atom_array = np.asarray(atoms)
    keep = np.ones(atom_array.shape[0], dtype=bool)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool)
        if mask.shape != atom_array.shape[1:]:
            raise ValueError(
                f"support_mask shape {mask.shape} does not match atom volume shape "
                f"{atom_array.shape[1:]}."
            )
        keep &= np.sum(atom_array * mask[None, ...], axis=(1, 2, 3)) > EPS
    if projection_visibility is not None:
        visibility = np.asarray(projection_visibility, dtype=np.float64).reshape(-1)
        if visibility.size != atom_array.shape[0]:
            raise ValueError("projection_visibility length must match atom count.")
        keep &= visibility >= float(min_visibility)
    return atom_array[keep], keep
