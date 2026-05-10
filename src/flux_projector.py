from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack

from src.xfct_geometry import (
    PinholeGeometry,
    detector_continuous_indices,
    dot_uv,
    geometry_log_weight_dot,
    geometry_weight,
    uv,
)

EPS = 1e-12


@dataclass(frozen=True)
class FluxProjectorConfig:
    angles: np.ndarray
    detector_shape: tuple[int, int]
    recon_shape: tuple[int, int, int]
    voxel_size: float = 0.5
    detector_pixel_size: float = 0.25
    geometry: PinholeGeometry = field(
        default_factory=lambda: PinholeGeometry(
            detector_to_pinhole=-30.0,
            center_to_pinhole=50.0,
            detector_offset_x=-0.5,
        )
    )
    use_geometry_weight: bool = True


def voxel_center_xyz(
    recon_shape: tuple[int, int, int],
    voxel_size: float,
) -> np.ndarray:
    z_axis = _centered_axis(recon_shape[0], voxel_size)
    y_axis = _centered_axis(recon_shape[1], voxel_size)
    x_axis = _centered_axis(recon_shape[2], voxel_size)
    zi, yi, xi = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    return np.column_stack((xi.ravel(), yi.ravel(), zi.ravel()))


def _centered_axis(size: int, spacing: float) -> np.ndarray:
    return (
        np.arange(size, dtype=np.float64) - (float(size) - 1.0) / 2.0
    ) * float(spacing)


def _uniform_delta_theta(angles: np.ndarray) -> float:
    if angles.size < 2:
        raise ValueError("at least two angles are required for M_forward.")
    wrapped = np.mod(np.asarray(angles, dtype=np.float64), 2.0 * np.pi)
    sorted_angles = np.sort(wrapped)
    deltas = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2.0 * np.pi])
    delta = float(np.mean(deltas))
    if not np.allclose(deltas, delta, rtol=1e-5, atol=1e-8):
        raise ValueError("M_forward requires uniformly spaced circular angles.")
    return delta


def _angle_derivative(projection: np.ndarray, delta_theta: float) -> np.ndarray:
    return (
        np.roll(projection, shift=-1, axis=0)
        - np.roll(projection, shift=1, axis=0)
    ) / (2.0 * float(delta_theta))


def _angle_derivative_adjoint(z: np.ndarray, delta_theta: float) -> np.ndarray:
    return (
        np.roll(z, shift=1, axis=0)
        - np.roll(z, shift=-1, axis=0)
    ) / (2.0 * float(delta_theta))


def _bilinear_entries(
    row: np.ndarray,
    col: np.ndarray,
    base_value: np.ndarray,
    detector_shape: tuple[int, int],
    detector_pixel_size: float,
    derivative_row_value: np.ndarray | None = None,
    derivative_col_value: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    height, width = detector_shape
    n_vox = row.size
    voxel_idx = np.arange(n_vox, dtype=np.int64)
    row_floor = np.floor(row).astype(np.int64)
    col_floor = np.floor(col).astype(np.int64)
    row_frac = row - row_floor
    col_frac = col - col_floor

    rows_all = []
    cols_all = []
    data_all = []
    drow_all = [] if derivative_row_value is not None else None
    dcol_all = [] if derivative_col_value is not None else None

    for row_offset, row_weight, row_deriv in (
        (0, 1.0 - row_frac, -np.ones_like(row_frac)),
        (1, row_frac, np.ones_like(row_frac)),
    ):
        rr = row_floor + row_offset
        valid_row = (rr >= 0) & (rr < height)
        for col_offset, col_weight, col_deriv in (
            (0, 1.0 - col_frac, -np.ones_like(col_frac)),
            (1, col_frac, np.ones_like(col_frac)),
        ):
            cc = col_floor + col_offset
            valid = valid_row & (cc >= 0) & (cc < width)
            if not np.any(valid):
                continue
            kernel = row_weight[valid] * col_weight[valid]
            rows_all.append(rr[valid] * width + cc[valid])
            cols_all.append(voxel_idx[valid])
            data_all.append(base_value[valid] * kernel)
            if drow_all is not None:
                d_kernel_dv = row_deriv[valid] * col_weight[valid]
                drow_all.append(
                    derivative_row_value[valid]
                    * d_kernel_dv
                    / float(detector_pixel_size)
                )
            if dcol_all is not None:
                d_kernel_du = row_weight[valid] * col_deriv[valid]
                dcol_all.append(
                    derivative_col_value[valid]
                    * d_kernel_du
                    / float(detector_pixel_size)
                )

    if not rows_all:
        empty = np.array([], dtype=np.float64)
        empty_idx = np.array([], dtype=np.int64)
        return empty_idx, empty_idx, empty, empty, empty

    rows = np.concatenate(rows_all).astype(np.int64, copy=False)
    cols = np.concatenate(cols_all).astype(np.int64, copy=False)
    data = np.concatenate(data_all).astype(np.float64, copy=False)
    drow = None if drow_all is None else np.concatenate(drow_all)
    dcol = None if dcol_all is None else np.concatenate(dcol_all)
    return rows, cols, data, drow, dcol


def _angle_sparse_matrices(
    theta: float,
    xyz: np.ndarray,
    config: FluxProjectorConfig,
    include_projection: bool,
    include_components: bool,
) -> dict[str, csr_matrix | None]:
    detector_pixels = int(np.prod(config.detector_shape))
    n_vox = xyz.shape[0]
    u_coord, v_coord = uv(theta, xyz, config.geometry)
    row, col = detector_continuous_indices(
        u_coord,
        v_coord,
        detector_shape=config.detector_shape,
        detector_pixel_size=config.detector_pixel_size,
    )

    if config.use_geometry_weight:
        weight = geometry_weight(theta, xyz, config.geometry)
    else:
        weight = np.ones(n_vox, dtype=np.float64)
    weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

    dot_u, dot_v = dot_uv(theta, xyz, config.geometry)
    source_value = weight * geometry_log_weight_dot(theta, xyz, config.geometry)
    flux_u_value = weight * dot_u
    flux_v_value = weight * dot_v

    rows, cols, projection_data, drow_data, dcol_data = _bilinear_entries(
        row=row,
        col=col,
        base_value=weight,
        detector_shape=config.detector_shape,
        detector_pixel_size=config.detector_pixel_size,
        derivative_row_value=flux_v_value,
        derivative_col_value=flux_u_value,
    )
    shape = (detector_pixels, n_vox)

    projection_matrix = None
    if include_projection:
        projection_matrix = coo_matrix((projection_data, (rows, cols)), shape=shape)
        projection_matrix = projection_matrix.tocsr()
        projection_matrix.eliminate_zeros()

    source_rows, source_cols, source_data, _, _ = _bilinear_entries(
        row=row,
        col=col,
        base_value=source_value,
        detector_shape=config.detector_shape,
        detector_pixel_size=config.detector_pixel_size,
    )
    transport_data = source_data + drow_data + dcol_data
    transport_rhs = coo_matrix(
        (transport_data, (source_rows, source_cols)),
        shape=shape,
    ).tocsr()
    transport_rhs.eliminate_zeros()

    flux_u = None
    flux_v = None
    source = None
    if include_components:
        comp_rows, comp_cols, flux_u_data, _, _ = _bilinear_entries(
            row=row,
            col=col,
            base_value=flux_u_value,
            detector_shape=config.detector_shape,
            detector_pixel_size=config.detector_pixel_size,
        )
        flux_u = coo_matrix((flux_u_data, (comp_rows, comp_cols)), shape=shape)
        flux_u = flux_u.tocsr()
        flux_u.eliminate_zeros()

        comp_rows, comp_cols, flux_v_data, _, _ = _bilinear_entries(
            row=row,
            col=col,
            base_value=flux_v_value,
            detector_shape=config.detector_shape,
            detector_pixel_size=config.detector_pixel_size,
        )
        flux_v = coo_matrix((flux_v_data, (comp_rows, comp_cols)), shape=shape)
        flux_v = flux_v.tocsr()
        flux_v.eliminate_zeros()

        source = coo_matrix((source_data, (source_rows, source_cols)), shape=shape)
        source = source.tocsr()
        source.eliminate_zeros()

    return {
        "projection": projection_matrix,
        "flux_u": flux_u,
        "flux_v": flux_v,
        "source": source,
        "transport_rhs": transport_rhs,
    }


class DepthAwareFluxProjector:
    def __init__(
        self,
        config: FluxProjectorConfig,
        system_matrix=None,
        include_projection: bool = False,
        include_components: bool = True,
    ):
        self.config = config
        self.angles = np.asarray(config.angles, dtype=np.float64).reshape(-1)
        if self.angles.size < 1:
            raise ValueError("angles must not be empty.")
        self.detector_shape = tuple(config.detector_shape)
        self.projection_shape = (self.angles.size,) + self.detector_shape
        self.n_det = int(np.prod(self.projection_shape))
        self.n_vox = int(np.prod(config.recon_shape))
        self.system_matrix = system_matrix
        self.delta_theta = (
            _uniform_delta_theta(self.angles) if self.angles.size > 1 else None
        )

        xyz = voxel_center_xyz(config.recon_shape, config.voxel_size)
        projection_parts = []
        flux_u_parts = []
        flux_v_parts = []
        source_parts = []
        transport_parts = []
        for theta in self.angles:
            matrices = _angle_sparse_matrices(
                theta=float(theta),
                xyz=xyz,
                config=config,
                include_projection=include_projection,
                include_components=include_components,
            )
            if include_projection:
                projection_parts.append(matrices["projection"])
            if include_components:
                flux_u_parts.append(matrices["flux_u"])
                flux_v_parts.append(matrices["flux_v"])
                source_parts.append(matrices["source"])
            transport_parts.append(matrices["transport_rhs"])

        self.projection_matrix = (
            vstack(projection_parts).tocsr() if projection_parts else None
        )
        self.flux_u_matrix = vstack(flux_u_parts).tocsr() if flux_u_parts else None
        self.flux_v_matrix = vstack(flux_v_parts).tocsr() if flux_v_parts else None
        self.source_matrix = vstack(source_parts).tocsr() if source_parts else None
        self.transport_rhs_matrix = vstack(transport_parts).tocsr()
        self.transport_rhs_matrix.eliminate_zeros()

        if self.system_matrix is not None:
            if self.system_matrix.shape != (self.n_det, self.n_vox):
                raise ValueError(
                    "system_matrix shape mismatch: "
                    f"expected {(self.n_det, self.n_vox)}, "
                    f"got {self.system_matrix.shape}."
                )

    def B_u(self, x: np.ndarray) -> np.ndarray:
        if self.flux_u_matrix is None:
            raise RuntimeError("B_u requires include_components=True.")
        return np.asarray(self.flux_u_matrix @ np.asarray(x).ravel()).reshape(
            self.projection_shape
        )

    def B_v(self, x: np.ndarray) -> np.ndarray:
        if self.flux_v_matrix is None:
            raise RuntimeError("B_v requires include_components=True.")
        return np.asarray(self.flux_v_matrix @ np.asarray(x).ravel()).reshape(
            self.projection_shape
        )

    def C(self, x: np.ndarray) -> np.ndarray:
        if self.source_matrix is None:
            raise RuntimeError("C requires include_components=True.")
        return np.asarray(self.source_matrix @ np.asarray(x).ravel()).reshape(
            self.projection_shape
        )

    def transport_rhs(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.transport_rhs_matrix @ np.asarray(x).ravel()
        ).reshape(self.projection_shape)

    def M_forward(self, x: np.ndarray) -> np.ndarray:
        if self.system_matrix is None:
            return self.transport_rhs(x)
        if self.delta_theta is None:
            raise ValueError("M_forward with system_matrix requires multiple angles.")

        f = np.asarray(x, dtype=np.float64).ravel()
        projection = np.asarray(self.system_matrix @ f).reshape(self.projection_shape)
        angle_term = _angle_derivative(projection, self.delta_theta)
        rhs = self.transport_rhs(f)
        return (angle_term - rhs).ravel()

    def M_adjoint(self, z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=np.float64).reshape(self.projection_shape)
        rhs_grad = np.asarray(self.transport_rhs_matrix.T @ zz.ravel()).ravel()
        if self.system_matrix is None:
            return rhs_grad
        if self.delta_theta is None:
            raise ValueError("M_adjoint with system_matrix requires multiple angles.")

        angle_grad = _angle_derivative_adjoint(zz, self.delta_theta)
        return np.asarray(self.system_matrix.T @ angle_grad.ravel()).ravel() - rhs_grad

    def m_forward(self, x: np.ndarray) -> np.ndarray:
        return self.M_forward(x)

    def m_adjoint(self, z: np.ndarray) -> np.ndarray:
        return self.M_adjoint(z)
