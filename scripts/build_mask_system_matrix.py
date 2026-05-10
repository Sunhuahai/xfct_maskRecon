from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.sparse import coo_matrix, csr_matrix, save_npz, vstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from projection.mask_geometry import MaskSpec, build_mask_spec


EPS = 1.0e-12


def _format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "d").replace("-", "m")


def format_image_grid_tag(voxel_size: float, image_xy: int, image_z: int) -> str:
    return f"lim{_format_float(voxel_size)}_xy{int(image_xy)}_z{int(image_z)}"


def build_image_grid(
    voxel_size: float,
    image_xy: int,
    image_z: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.arange(
        -voxel_size * (image_xy - 1) / 2,
        voxel_size * (image_xy - 1) / 2 + voxel_size / 2,
        voxel_size,
    )
    y_axis = np.arange(
        -voxel_size * (image_xy - 1) / 2,
        voxel_size * (image_xy - 1) / 2 + voxel_size / 2,
        voxel_size,
    )
    z_axis = np.arange(
        -voxel_size * (image_z - 1) / 2,
        voxel_size * (image_z - 1) / 2 + voxel_size / 2,
        voxel_size,
    )
    zi, yi, xi = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    return xi.ravel(), yi.ravel(), zi.ravel()


def rotate_object_to_lab(
    x: np.ndarray,
    y: np.ndarray,
    cos_t: float,
    sin_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def _transpose_matlab_array(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        return array
    return np.transpose(array, axes=tuple(range(array.ndim - 1, -1, -1)))


def load_mat_variable(path: str | Path, var_name: str | None = None) -> np.ndarray:
    path = Path(path)
    try:
        payload = loadmat(path)
        if var_name is not None:
            if var_name not in payload:
                raise KeyError(f"{var_name} not found in {path}")
            return np.asarray(payload[var_name], dtype=float)

        keys = [key for key in payload if not key.startswith("__")]
        if not keys:
            raise ValueError(f"No variables found in {path}")
        return np.asarray(payload[keys[0]], dtype=float)
    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower():
            raise
        with h5py.File(path, "r") as payload:
            if var_name is not None:
                if var_name not in payload:
                    raise KeyError(f"{var_name} not found in {path}")
                return np.asarray(_transpose_matlab_array(np.asarray(payload[var_name])), dtype=float)
            keys = [key for key in payload if not key.startswith("#")]
            if not keys:
                raise ValueError(f"No variables found in {path}")
            return np.asarray(_transpose_matlab_array(np.asarray(payload[keys[0]])), dtype=float)


def load_source_spectrum(
    spectrum_file: str | Path,
    spectrum_key: str,
    energy_key: str,
    min_kev: float | None,
    max_kev: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(load_mat_variable(spectrum_file, spectrum_key), dtype=float).ravel()
    energies = np.asarray(load_mat_variable(spectrum_file, energy_key), dtype=float).ravel()
    if counts.size != energies.size:
        raise ValueError(
            f"Spectrum count and energy sizes differ: {counts.size} vs {energies.size}."
        )
    if np.nanmax(energies) < 1.0:
        energies = energies * 1000.0

    valid = np.isfinite(energies) & np.isfinite(counts) & (counts > 0.0)
    if min_kev is not None:
        valid &= energies >= float(min_kev)
    if max_kev is not None:
        valid &= energies <= float(max_kev)
    if not np.any(valid):
        raise ValueError("No valid source spectrum bins after filtering.")

    energies = energies[valid]
    weights = counts[valid]
    weights = weights / np.sum(weights)
    return energies.astype(np.float64), weights.astype(np.float64)


def load_pmma_attenuation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Could not parse PMMA attenuation CSV: {path}")
    energy_kev = np.asarray(table[:, 0], dtype=np.float64) * 1000.0
    mu_mm_inv = np.asarray(table[:, 1], dtype=np.float64) / 10.0
    valid = (
        np.isfinite(energy_kev)
        & np.isfinite(mu_mm_inv)
        & (energy_kev > 0.0)
        & (mu_mm_inv > 0.0)
    )
    if not np.any(valid):
        raise ValueError(f"No valid PMMA attenuation rows in {path}")
    order = np.argsort(energy_kev[valid])
    return energy_kev[valid][order], mu_mm_inv[valid][order]


def interpolate_mu(
    energy_kev: np.ndarray | float,
    energy_table_kev: np.ndarray,
    mu_table_mm_inv: np.ndarray,
) -> np.ndarray:
    energy = np.asarray(energy_kev, dtype=float)
    clipped = np.clip(energy, energy_table_kev[0], energy_table_kev[-1])
    log_mu = np.interp(clipped, energy_table_kev, np.log(mu_table_mm_inv))
    return np.exp(log_mu)


def build_volume_bounds(
    voxel_size: float,
    image_xy: int,
    image_z: int,
) -> tuple[np.ndarray, np.ndarray]:
    half_x = image_xy * voxel_size / 2.0
    half_y = image_xy * voxel_size / 2.0
    half_z = image_z * voxel_size / 2.0
    return (
        np.array([-half_x, -half_y, -half_z], dtype=np.float64),
        np.array([half_x, half_y, half_z], dtype=np.float64),
    )


def _broadcast_points(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.atleast_2d(np.asarray(start, dtype=np.float64))
    end = np.atleast_2d(np.asarray(end, dtype=np.float64))
    if start.shape[1] != 3 or end.shape[1] != 3:
        raise ValueError("start/end points must be 3-D.")

    n = max(start.shape[0], end.shape[0])
    if start.shape[0] not in (1, n) or end.shape[0] not in (1, n):
        raise ValueError("start/end batch sizes are incompatible.")
    if start.shape[0] == 1 and n > 1:
        start = np.repeat(start, n, axis=0)
    if end.shape[0] == 1 and n > 1:
        end = np.repeat(end, n, axis=0)
    return start, end


def segment_box_inside_length(
    start: np.ndarray,
    end: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    start_pts, end_pts = _broadcast_points(start, end)
    direction = end_pts - start_pts
    segment_length = np.linalg.norm(direction, axis=1)

    t_min = np.zeros(segment_length.shape[0], dtype=np.float64)
    t_max = np.ones(segment_length.shape[0], dtype=np.float64)

    for axis in range(3):
        s = start_pts[:, axis]
        d = direction[:, axis]
        slab_min = bounds_min[axis]
        slab_max = bounds_max[axis]

        parallel = np.abs(d) < EPS
        inside = (s >= slab_min - EPS) & (s <= slab_max + EPS)
        miss_parallel = parallel & (~inside)
        t_min[miss_parallel] = 1.0
        t_max[miss_parallel] = 0.0

        non_parallel = ~parallel
        if not np.any(non_parallel):
            continue
        inv_d = 1.0 / d[non_parallel]
        t1 = (slab_min - s[non_parallel]) * inv_d
        t2 = (slab_max - s[non_parallel]) * inv_d
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        t_min[non_parallel] = np.maximum(t_min[non_parallel], near)
        t_max[non_parallel] = np.minimum(t_max[non_parallel], far)

    clipped = np.maximum(0.0, np.minimum(t_max, 1.0) - np.maximum(t_min, 0.0))
    return segment_length * clipped


def compute_incident_attenuation(
    incident_lengths_mm: np.ndarray,
    args: argparse.Namespace,
    energy_table_kev: np.ndarray,
    mu_table_mm_inv: np.ndarray,
    incident_energies_kev: np.ndarray | None,
    incident_weights: np.ndarray | None,
) -> np.ndarray:
    if args.incident_mode == "mono":
        mu_incident = float(
            interpolate_mu(args.incident_energy_kev, energy_table_kev, mu_table_mm_inv)
        )
        return np.exp(-mu_incident * incident_lengths_mm)

    if incident_energies_kev is None or incident_weights is None:
        raise ValueError("Spectrum incident mode requires source spectrum arrays.")

    mu_incident = interpolate_mu(
        incident_energies_kev,
        energy_table_kev,
        mu_table_mm_inv,
    )
    attenuation = np.zeros_like(incident_lengths_mm, dtype=np.float64)
    for weight, mu_value in zip(incident_weights, mu_incident):
        attenuation += weight * np.exp(-mu_value * incident_lengths_mm)
    return attenuation


def compute_incident_weights(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    z_lab: np.ndarray,
    args: argparse.Namespace,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    energy_table_kev: np.ndarray,
    mu_table_mm_inv: np.ndarray,
    incident_energies_kev: np.ndarray | None,
    incident_weights: np.ndarray | None,
) -> np.ndarray:
    voxels = np.column_stack((x_lab, y_lab, z_lab))
    source = np.array([-args.source_to_center, 0.0, 0.0], dtype=np.float64)
    incident_lengths = segment_box_inside_length(source, voxels, bounds_min, bounds_max)
    return compute_incident_attenuation(
        incident_lengths_mm=incident_lengths,
        args=args,
        energy_table_kev=energy_table_kev,
        mu_table_mm_inv=mu_table_mm_inv,
        incident_energies_kev=incident_energies_kev,
        incident_weights=incident_weights,
    )


def compute_exit_weights(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    z_lab: np.ndarray,
    hole_x_mm: float,
    hole_z_mm: float,
    args: argparse.Namespace,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    energy_table_kev: np.ndarray,
    mu_table_mm_inv: np.ndarray,
) -> np.ndarray:
    voxels = np.column_stack((x_lab, y_lab, z_lab))
    hole_lab = np.array(
        [
            float(hole_x_mm) - float(args.detector_offset_x),
            -float(args.center_to_pinhole),
            float(hole_z_mm),
        ],
        dtype=np.float64,
    )
    exit_lengths = segment_box_inside_length(voxels, hole_lab, bounds_min, bounds_max)
    mu_fluorescence = float(
        interpolate_mu(args.fluorescence_energy_kev, energy_table_kev, mu_table_mm_inv)
    )
    return np.exp(-mu_fluorescence * exit_lengths)


def parse_angle_indices(value: str | None) -> np.ndarray | None:
    if value is None or value.strip() == "":
        return None
    return np.array([int(item.strip()) for item in value.split(",") if item.strip()], dtype=int)


def default_output_path(
    args: argparse.Namespace,
    mask_spec: MaskSpec,
    n_angles: int,
) -> Path:
    attenuation_tag = "_att_pmma" if args.attenuation == "pmma" else ""
    detector_clip_tag = ""
    if getattr(args, "clip_to_physical_detector", False):
        detector_clip_tag = (
            f"_clipx{int(args.physical_detector_x)}pad{int(args.pad_x)}"
        )
    image_grid_tag = format_image_grid_tag(args.voxel_size, args.image_xy, args.image_z)
    filename = (
        f"cij_{int(n_angles)}_3d_mod{_format_float(args.detector_to_pinhole)}_"
        f"cmask_{mask_spec.tag}_{image_grid_tag}{attenuation_tag}{detector_clip_tag}.npz"
    )
    return Path(args.output_dir) / filename


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sparse XFCT system matrix for a multi-hole coded mask."
    )
    parser.add_argument("--theta-step", type=float, default=8.0, help="Angle step in degrees.")
    parser.add_argument(
        "--angle-indices",
        default=None,
        help="Optional comma-separated indices on the theta-step grid, e.g. 0,9,18,27,36.",
    )
    parser.add_argument("--detector-pixel-size", type=float, default=0.25)
    parser.add_argument("--detector-x", type=int, default=160)
    parser.add_argument("--detector-z", type=int, default=80)
    parser.add_argument(
        "--clip-to-physical-detector",
        action="store_true",
        help=(
            "Clip samples to a physical detector support before writing into the padded "
            "detector rows. This matches 80x80 projection generation followed by x padding."
        ),
    )
    parser.add_argument(
        "--physical-detector-x",
        type=int,
        default=80,
        help="Physical detector x pixels when --clip-to-physical-detector is used.",
    )
    parser.add_argument(
        "--pad-x",
        type=int,
        default=40,
        help="Left x padding used to embed the physical detector into detector-x rows.",
    )
    parser.add_argument("--detector-to-pinhole", type=float, default=30.0)
    parser.add_argument("--center-to-pinhole", type=float, default=50.0)
    parser.add_argument("--detector-offset-x", type=float, default=-0.5)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--image-xy", type=int, default=60)
    parser.add_argument("--image-z", type=int, default=40)
    parser.add_argument("--n-sample", type=int, default=500, help="Monte Carlo samples per voxel per hole.")
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument(
        "--flush-nnz",
        type=int,
        default=8000000,
        help="Flush accumulated COO triplets to CSR after this many sampled entries per angle.",
    )
    parser.add_argument("--seed", type=int, default=20260509)

    parser.add_argument(
        "--mask-layout",
        default="grid3x3",
        choices=["single", "grid3x3", "grid", "cross5", "random"],
    )
    parser.add_argument("--mask-rows", type=int, default=3)
    parser.add_argument("--mask-cols", type=int, default=3)
    parser.add_argument("--mask-pitch-mm", type=float, default=6.0)
    parser.add_argument("--mask-hole-diameter-mm", type=float, default=1.25)
    parser.add_argument("--mask-hole-count", type=int, default=9)
    parser.add_argument("--mask-random-extent-mm", type=float, default=18.0)
    parser.add_argument("--mask-min-separation-mm", type=float, default=None)
    parser.add_argument("--mask-file", default=None)

    parser.add_argument(
        "--attenuation",
        choices=["none", "pmma"],
        default="pmma",
        help="Optional attenuation model. Default matches existing *_att_pmma matrices.",
    )
    parser.add_argument("--source-to-center", type=float, default=300.0)
    parser.add_argument(
        "--attenuation-csv",
        default=str(PROJECT_ROOT / "data" / "attenuation_map" / "PMMA.csv"),
    )
    parser.add_argument(
        "--spectrum-file",
        default=str(PROJECT_ROOT / "data" / "projection_physics" / "spec_150kVp.mat"),
    )
    parser.add_argument("--spectrum-key", default="N")
    parser.add_argument("--spectrum-energy-key", default="E")
    parser.add_argument("--incident-mode", choices=["spectrum", "mono"], default="spectrum")
    parser.add_argument("--incident-energy-kev", type=float, default=80.0)
    parser.add_argument("--incident-min-kev", type=float, default=None)
    parser.add_argument("--incident-max-kev", type=float, default=None)
    parser.add_argument("--fluorescence-energy-kev", type=float, default=42.5)

    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "system_matrix"),
    )
    parser.add_argument("--output", default=None)
    return parser


def build_mask_system_matrix(args: argparse.Namespace) -> Path:
    mask_spec = build_mask_spec(
        layout=args.mask_layout,
        pitch_mm=args.mask_pitch_mm,
        hole_diameter_mm=args.mask_hole_diameter_mm,
        rows=args.mask_rows,
        cols=args.mask_cols,
        hole_count=args.mask_hole_count,
        random_extent_mm=args.mask_random_extent_mm,
        min_separation_mm=args.mask_min_separation_mm,
        seed=args.seed,
        mask_file=args.mask_file,
    )
    angle_indices = parse_angle_indices(args.angle_indices)
    if angle_indices is None:
        theta_deg = np.arange(0.0, 360.0, args.theta_step, dtype=np.float64)
    else:
        theta_deg = angle_indices.astype(np.float64) * float(args.theta_step)
    theta = np.deg2rad(theta_deg)
    n_angles = len(theta)

    n_det_per_angle = int(args.detector_z) * int(args.detector_x)
    yd_val = -float(args.detector_to_pinhole)
    xi, yi, zi = build_image_grid(args.voxel_size, args.image_xy, args.image_z)
    n_voxels = int(xi.size)
    rng = np.random.default_rng(args.seed)

    bounds_min = bounds_max = None
    energy_table_kev = mu_table_mm_inv = None
    incident_energies_kev = incident_weights = None
    if args.attenuation == "pmma":
        bounds_min, bounds_max = build_volume_bounds(args.voxel_size, args.image_xy, args.image_z)
        energy_table_kev, mu_table_mm_inv = load_pmma_attenuation(args.attenuation_csv)
        if args.incident_mode == "spectrum":
            incident_energies_kev, incident_weights = load_source_spectrum(
                spectrum_file=args.spectrum_file,
                spectrum_key=args.spectrum_key,
                energy_key=args.spectrum_energy_key,
                min_kev=args.incident_min_kev,
                max_kev=args.incident_max_kev,
            )

    print(f"Mask layout: {mask_spec.layout}")
    print(f"Mask holes : {mask_spec.hole_count}")
    print(f"Hole dia.  : {mask_spec.hole_diameter_mm:g} mm")
    print(f"Hole centers (x,z) mm:\n{mask_spec.hole_centers}")
    print(f"Angles     : {n_angles} ({theta_deg[0]:g}..{theta_deg[-1]:g} deg)")
    print(f"Detector   : {args.detector_z} x {args.detector_x} pixels")
    if args.clip_to_physical_detector:
        print(
            "Physical clip: "
            f"x=[{int(args.pad_x)}, {int(args.pad_x) + int(args.physical_detector_x) - 1}] "
            f"inside padded detector width {int(args.detector_x)}"
        )
    print(f"Voxels     : {n_voxels} ({args.image_z}, {args.image_xy}, {args.image_xy})")
    print(f"Samples    : {args.n_sample} per voxel per hole")
    print(f"Attenuation: {args.attenuation}")

    cij_list = []
    start_time = time.time()
    radius = float(mask_spec.hole_diameter_mm) / 2.0

    for angle_slot, angle_rad in enumerate(theta):
        angle_start = time.time()
        print(
            f"Angle {angle_slot + 1}/{n_angles} ({theta_deg[angle_slot]:.1f} deg)...",
            end="",
            flush=True,
        )
        cos_t = float(np.cos(angle_rad))
        sin_t = float(np.sin(angle_rad))

        x_lab, y_lab = rotate_object_to_lab(xi, yi, cos_t, sin_t)
        x_rot = x_lab + float(args.detector_offset_x)
        y_rot = y_lab + float(args.center_to_pinhole)
        z_rot = zi

        incident_all = None
        exit_by_hole: list[np.ndarray] | None = None
        if args.attenuation == "pmma":
            assert bounds_min is not None and bounds_max is not None
            assert energy_table_kev is not None and mu_table_mm_inv is not None
            incident_all = compute_incident_weights(
                x_lab=x_lab,
                y_lab=y_lab,
                z_lab=zi,
                args=args,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                energy_table_kev=energy_table_kev,
                mu_table_mm_inv=mu_table_mm_inv,
                incident_energies_kev=incident_energies_kev,
                incident_weights=incident_weights,
            )
            exit_by_hole = [
                compute_exit_weights(
                    x_lab=x_lab,
                    y_lab=y_lab,
                    z_lab=zi,
                    hole_x_mm=float(hole_center[0]),
                    hole_z_mm=float(hole_center[1]),
                    args=args,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    energy_table_kev=energy_table_kev,
                    mu_table_mm_inv=mu_table_mm_inv,
                )
                for hole_center in mask_spec.hole_centers
            ]

        row_indices_all = []
        col_indices_all = []
        data_values_all = []
        pending_nnz = 0
        cij_angle_accum = csr_matrix((n_det_per_angle, n_voxels))
        num_batches = int(np.ceil(n_voxels / int(args.batch_size)))

        def flush_pending() -> None:
            nonlocal row_indices_all, col_indices_all, data_values_all, pending_nnz, cij_angle_accum
            if pending_nnz <= 0:
                return
            rows = np.concatenate(row_indices_all)
            cols = np.concatenate(col_indices_all)
            data = np.concatenate(data_values_all)
            chunk = coo_matrix((data, (rows, cols)), shape=(n_det_per_angle, n_voxels)).tocsr()
            chunk.eliminate_zeros()
            cij_angle_accum = cij_angle_accum + chunk
            row_indices_all = []
            col_indices_all = []
            data_values_all = []
            pending_nnz = 0

        for batch_idx in range(num_batches):
            idx_start = batch_idx * int(args.batch_size)
            idx_end = min((batch_idx + 1) * int(args.batch_size), n_voxels)
            current_batch_size = idx_end - idx_start

            xb = x_rot[idx_start:idx_end]
            yb = y_rot[idx_start:idx_end]
            zb = z_rot[idx_start:idx_end]

            voxel_indices_local = np.arange(current_batch_size, dtype=np.int32)[:, np.newaxis]
            voxel_indices_expanded = np.repeat(voxel_indices_local, int(args.n_sample), axis=1)

            for hole_index, (hole_x, hole_z) in enumerate(mask_spec.hole_centers):
                scale = ((yb - yd_val) / np.maximum(yb, EPS))[:, np.newaxis]
                x_c = (xb / np.maximum(yb, EPS) * yd_val)[:, np.newaxis] + scale * float(hole_x)
                z_c = (zb / np.maximum(yb, EPS) * yd_val)[:, np.newaxis] + scale * float(hole_z)
                r_c = scale * radius

                a_rand = rng.random((current_batch_size, int(args.n_sample)))
                b_rand = rng.random((current_batch_size, int(args.n_sample)))
                r_sample = np.sqrt(a_rand) * r_c
                theta_sample = b_rand * 2.0 * np.pi
                xs = r_sample * np.cos(theta_sample) + x_c
                zs = r_sample * np.sin(theta_sample) + z_c

                idx_x = np.floor(
                    (xs + args.detector_pixel_size * (args.detector_x - 1) / 2)
                    / args.detector_pixel_size
                    + 0.5
                ).astype(np.int32)
                idx_z = np.floor(
                    (zs + args.detector_pixel_size * (args.detector_z - 1) / 2)
                    / args.detector_pixel_size
                    + 0.5
                ).astype(np.int32)
                valid_mask = (
                    (idx_x >= 0)
                    & (idx_x < args.detector_x)
                    & (idx_z >= 0)
                    & (idx_z < args.detector_z)
                )
                if args.clip_to_physical_detector:
                    physical_x0 = int(args.pad_x)
                    physical_x1 = physical_x0 + int(args.physical_detector_x)
                    if physical_x0 < 0 or physical_x1 > int(args.detector_x):
                        raise ValueError(
                            "Invalid physical detector embedding: "
                            f"pad_x={args.pad_x}, physical_detector_x={args.physical_detector_x}, "
                            f"detector_x={args.detector_x}."
                        )
                    valid_mask &= (idx_x >= physical_x0) & (idx_x < physical_x1)
                det_indices = idx_z * int(args.detector_x) + idx_x

                dist_sq = (xb - float(hole_x)) ** 2 + yb**2 + (zb - float(hole_z)) ** 2
                geom_weight = radius**2 * np.abs(yb) / (4.0 * dist_sq * np.sqrt(dist_sq))
                if args.attenuation == "pmma":
                    assert incident_all is not None and exit_by_hole is not None
                    attenuation = (
                        incident_all[idx_start:idx_end]
                        * exit_by_hole[hole_index][idx_start:idx_end]
                    )
                    geom_weight = geom_weight * attenuation

                sample_weight = geom_weight / float(args.n_sample)
                weights_expanded = np.repeat(sample_weight[:, np.newaxis], int(args.n_sample), axis=1)

                valid_det = det_indices[valid_mask]
                row_indices_all.append(valid_det)
                col_indices_all.append(voxel_indices_expanded[valid_mask] + idx_start)
                data_values_all.append(weights_expanded[valid_mask])
                pending_nnz += int(valid_det.size)
                if pending_nnz >= int(args.flush_nnz):
                    flush_pending()

        flush_pending()
        cij_angle_accum.eliminate_zeros()
        cij_list.append(cij_angle_accum)

        print(f" done. ({time.time() - angle_start:.2f}s)")

    print("Stacking angle matrices...")
    cij = vstack(cij_list).tocsr()
    output_path = Path(args.output) if args.output else default_output_path(args, mask_spec, n_angles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, cij)
    print(f"System matrix shape: {cij.shape}")
    print(f"nnz: {cij.nnz}")
    print(f"Elapsed: {time.time() - start_time:.1f}s")
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    args = build_argparser().parse_args()
    build_mask_system_matrix(args)


if __name__ == "__main__":
    main()
