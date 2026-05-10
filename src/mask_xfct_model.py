from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from scripts.build_mask_system_matrix import (
    build_image_grid,
    build_volume_bounds,
    compute_exit_weights,
    compute_incident_weights,
    load_pmma_attenuation,
    load_source_spectrum,
    rotate_object_to_lab,
)


EPS = 1.0e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_float_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "d").replace("-", "m")


def default_angle_indices(angle_count: int) -> list[int]:
    if int(angle_count) == 5:
        return [0, 9, 18, 27, 36]
    if int(angle_count) == 15:
        return list(range(0, 45, 3))
    if int(angle_count) == 45:
        return list(range(45))
    if int(angle_count) <= 0:
        raise ValueError("angle_count must be positive.")
    return np.linspace(0, 44, int(angle_count), dtype=int).tolist()


def parse_int_list(value: str | Iterable[int] | None, default: Iterable[int] | None = None) -> list[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        if value.strip() == "":
            return list(default or [])
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(v) for v in value]


def physical_support_columns(detector_x: int = 160, physical_detector_x: int = 80, pad_x: int = 40) -> np.ndarray:
    support = np.zeros(int(detector_x), dtype=bool)
    support[int(pad_x) : int(pad_x) + int(physical_detector_x)] = True
    return support


def load_candidate_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "hole_centers_mm" in payload:
        centers = payload["hole_centers_mm"]
    elif "hole_centers_xz_mm" in payload:
        centers = payload["hole_centers_xz_mm"]
    else:
        raise KeyError(f"{path}: missing hole_centers_mm/hole_centers_xz_mm")
    payload["hole_centers_mm"] = [[float(x), float(z)] for x, z in centers]
    payload["hole_diameter_mm"] = float(payload.get("hole_diameter_mm", 1.25))
    return payload


def candidate_to_mask_config(candidate: dict) -> tuple[np.ndarray, float]:
    return (
        np.asarray(candidate["hole_centers_mm"], dtype=np.float64).reshape(-1, 2),
        float(candidate["hole_diameter_mm"]),
    )


@dataclass
class XFCTForwardConfig:
    hole_centers_mm: np.ndarray
    hole_diameter_mm: float = 1.25
    angle_indices: tuple[int, ...] = (0, 9, 18, 27, 36)
    theta_step_deg: float = 8.0
    angle_offset_deg: float = 0.0
    detector_pixel_size_mm: float = 0.25
    detector_x: int = 160
    detector_z: int = 80
    physical_detector_x: int = 80
    pad_x: int = 40
    detector_to_pinhole_mm: float = 30.0
    center_to_pinhole_mm: float = 50.0
    detector_offset_x_mm: float = -0.5
    source_to_center_mm: float = 300.0
    voxel_size_mm: float = 0.5
    image_xy: int = 60
    image_z: int = 40
    aperture_samples: int = 1
    aperture_mode: str = "point"
    attenuation: str = "none"
    fluorescence_energy_kev: float = 42.5
    incident_mode: str = "spectrum"
    incident_energy_kev: float = 80.0
    seed: int = 20260509

    @property
    def recon_shape(self) -> tuple[int, int, int]:
        return (int(self.image_z), int(self.image_xy), int(self.image_xy))

    @property
    def n_voxels(self) -> int:
        return int(self.image_z) * int(self.image_xy) * int(self.image_xy)

    @property
    def n_angles(self) -> int:
        return len(self.angle_indices)

    @property
    def n_rows(self) -> int:
        return self.n_angles * int(self.detector_z) * int(self.detector_x)


def _golden_disk_samples(n_samples: int) -> np.ndarray:
    if int(n_samples) <= 1:
        return np.zeros((1, 2), dtype=np.float64)
    n = int(n_samples)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    offsets = np.zeros((n, 2), dtype=np.float64)
    for idx in range(n):
        radius = math.sqrt((idx + 0.5) / n)
        theta = idx * golden_angle
        offsets[idx, 0] = radius * math.cos(theta)
        offsets[idx, 1] = radius * math.sin(theta)
    return offsets


class XFCTMaskOperator:
    """Depth-dependent detector-side multi-pinhole XFCT forward model.

    The geometry follows scripts/build_mask_system_matrix.py.  The operator can
    either keep the full 80x160 padded detector support or clip to the physical
    80-column detector and embed that support into the padded array.
    """

    def __init__(self, config: XFCTForwardConfig):
        self.config = config
        self.hole_centers_mm = np.asarray(config.hole_centers_mm, dtype=np.float64).reshape(-1, 2)
        self.radius_mm = float(config.hole_diameter_mm) / 2.0
        self.xi, self.yi, self.zi = build_image_grid(
            float(config.voxel_size_mm),
            int(config.image_xy),
            int(config.image_z),
        )
        self._support_cols = physical_support_columns(
            detector_x=int(config.detector_x),
            physical_detector_x=int(config.physical_detector_x),
            pad_x=int(config.pad_x),
        )
        self._aperture_offsets = (
            np.zeros((1, 2), dtype=np.float64)
            if str(config.aperture_mode).lower() == "point"
            else _golden_disk_samples(int(config.aperture_samples))
        )
        self._attenuation_args = None
        self._pmma_energy_table = None
        self._pmma_mu_table = None
        self._incident_energies = None
        self._incident_weights = None
        if str(config.attenuation).lower() == "pmma":
            self._attenuation_args = SimpleNamespace(
                source_to_center=float(config.source_to_center_mm),
                center_to_pinhole=float(config.center_to_pinhole_mm),
                detector_offset_x=float(config.detector_offset_x_mm),
                fluorescence_energy_kev=float(config.fluorescence_energy_kev),
                incident_mode=str(config.incident_mode),
                incident_energy_kev=float(config.incident_energy_kev),
            )
            self._pmma_energy_table, self._pmma_mu_table = load_pmma_attenuation(
                PROJECT_ROOT / "data" / "attenuation_map" / "PMMA.csv"
            )
            if str(config.incident_mode).lower() == "spectrum":
                self._incident_energies, self._incident_weights = load_source_spectrum(
                    spectrum_file=PROJECT_ROOT / "data" / "projection_physics" / "spec_150kVp.mat",
                    spectrum_key="N",
                    energy_key="E",
                    min_kev=None,
                    max_kev=None,
                )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.config.n_rows, self.config.n_voxels)

    @property
    def recon_shape(self) -> tuple[int, int, int]:
        return self.config.recon_shape

    def _angle_coordinates(self, angle_slot: int, voxel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        angle_index = int(cfg.angle_indices[int(angle_slot)])
        angle_rad = math.radians(float(angle_index) * float(cfg.theta_step_deg) + float(cfg.angle_offset_deg))
        cos_t = math.cos(angle_rad)
        sin_t = math.sin(angle_rad)
        xi = self.xi[voxel_indices]
        yi = self.yi[voxel_indices]
        zi = self.zi[voxel_indices]
        x_lab, y_lab = rotate_object_to_lab(xi, yi, cos_t, sin_t)
        x_rot = x_lab + float(cfg.detector_offset_x_mm)
        y_rot = y_lab + float(cfg.center_to_pinhole_mm)
        return x_rot, y_rot, zi, x_lab, y_lab

    def _attenuation_weight(
        self,
        x_lab: np.ndarray,
        y_lab: np.ndarray,
        z_lab: np.ndarray,
        hole_x: float,
        hole_z: float,
    ) -> np.ndarray:
        if str(self.config.attenuation).lower() != "pmma":
            return np.ones_like(x_lab, dtype=np.float64)
        assert self._attenuation_args is not None
        assert self._pmma_energy_table is not None and self._pmma_mu_table is not None
        bounds_min, bounds_max = build_volume_bounds(
            float(self.config.voxel_size_mm),
            int(self.config.image_xy),
            int(self.config.image_z),
        )
        incident = compute_incident_weights(
            x_lab=x_lab,
            y_lab=y_lab,
            z_lab=z_lab,
            args=self._attenuation_args,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            energy_table_kev=self._pmma_energy_table,
            mu_table_mm_inv=self._pmma_mu_table,
            incident_energies_kev=self._incident_energies,
            incident_weights=self._incident_weights,
        )
        exit_w = compute_exit_weights(
            x_lab=x_lab,
            y_lab=y_lab,
            z_lab=z_lab,
            hole_x_mm=float(hole_x),
            hole_z_mm=float(hole_z),
            args=self._attenuation_args,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            energy_table_kev=self._pmma_energy_table,
            mu_table_mm_inv=self._pmma_mu_table,
        )
        return incident * exit_w

    def _valid_mask(self, idx_x: np.ndarray, idx_z: np.ndarray, support_mode: str) -> np.ndarray:
        cfg = self.config
        valid = (
            (idx_x >= 0)
            & (idx_x < int(cfg.detector_x))
            & (idx_z >= 0)
            & (idx_z < int(cfg.detector_z))
        )
        if support_mode == "physical_padded":
            valid &= self._support_cols[np.clip(idx_x, 0, int(cfg.detector_x) - 1)]
        return valid

    def forward(
        self,
        image: np.ndarray,
        *,
        support_mode: str = "padded",
        hole_indices: Iterable[int] | None = None,
    ) -> np.ndarray:
        cfg = self.config
        f = np.asarray(image, dtype=np.float64).reshape(-1)
        out = np.zeros(int(cfg.n_rows), dtype=np.float64)
        active = np.flatnonzero(np.abs(f) > 0.0)
        if active.size == 0:
            return out
        hole_list = list(range(self.hole_centers_mm.shape[0])) if hole_indices is None else [int(h) for h in hole_indices]
        yd_val = -float(cfg.detector_to_pinhole_mm)
        pixel = float(cfg.detector_pixel_size_mm)
        det_x = int(cfg.detector_x)
        det_z = int(cfg.detector_z)
        det_per_angle = det_x * det_z
        sample_count = int(self._aperture_offsets.shape[0])

        for angle_slot in range(int(cfg.n_angles)):
            x_rot, y_rot, z_rot, x_lab, y_lab = self._angle_coordinates(angle_slot, active)
            angle_base = angle_slot * det_per_angle
            for hole_index in hole_list:
                hole_x, hole_z = self.hole_centers_mm[hole_index]
                scale = (y_rot - yd_val) / np.maximum(y_rot, EPS)
                center_x = x_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_x)
                center_z = z_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_z)
                dist_sq = (x_rot - float(hole_x)) ** 2 + y_rot**2 + (z_rot - float(hole_z)) ** 2
                weight = self.radius_mm**2 * np.abs(y_rot) / (4.0 * dist_sq * np.sqrt(dist_sq))
                weight *= self._attenuation_weight(x_lab, y_lab, z_rot, float(hole_x), float(hole_z))
                weight = weight * f[active] / float(sample_count)

                for offset_x, offset_z in self._aperture_offsets:
                    xs = center_x + scale * self.radius_mm * float(offset_x)
                    zs = center_z + scale * self.radius_mm * float(offset_z)
                    idx_x = np.floor((xs + pixel * (det_x - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    idx_z = np.floor((zs + pixel * (det_z - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    valid = self._valid_mask(idx_x, idx_z, support_mode)
                    if np.any(valid):
                        rows = angle_base + idx_z[valid] * det_x + idx_x[valid]
                        np.add.at(out, rows, weight[valid])
        return out

    def adjoint(
        self,
        data: np.ndarray,
        *,
        support_mode: str = "padded",
        hole_indices: Iterable[int] | None = None,
    ) -> np.ndarray:
        cfg = self.config
        y = np.asarray(data, dtype=np.float64).reshape(-1)
        if y.size != int(cfg.n_rows):
            raise ValueError(f"data length {y.size} does not match operator rows {cfg.n_rows}.")
        out = np.zeros(int(cfg.n_voxels), dtype=np.float64)
        voxels = np.arange(int(cfg.n_voxels), dtype=np.int64)
        hole_list = list(range(self.hole_centers_mm.shape[0])) if hole_indices is None else [int(h) for h in hole_indices]
        yd_val = -float(cfg.detector_to_pinhole_mm)
        pixel = float(cfg.detector_pixel_size_mm)
        det_x = int(cfg.detector_x)
        det_z = int(cfg.detector_z)
        det_per_angle = det_x * det_z
        sample_count = int(self._aperture_offsets.shape[0])

        for angle_slot in range(int(cfg.n_angles)):
            x_rot, y_rot, z_rot, x_lab, y_lab = self._angle_coordinates(angle_slot, voxels)
            angle_base = angle_slot * det_per_angle
            for hole_index in hole_list:
                hole_x, hole_z = self.hole_centers_mm[hole_index]
                scale = (y_rot - yd_val) / np.maximum(y_rot, EPS)
                center_x = x_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_x)
                center_z = z_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_z)
                dist_sq = (x_rot - float(hole_x)) ** 2 + y_rot**2 + (z_rot - float(hole_z)) ** 2
                weight = self.radius_mm**2 * np.abs(y_rot) / (4.0 * dist_sq * np.sqrt(dist_sq))
                weight *= self._attenuation_weight(x_lab, y_lab, z_rot, float(hole_x), float(hole_z))
                weight = weight / float(sample_count)

                for offset_x, offset_z in self._aperture_offsets:
                    xs = center_x + scale * self.radius_mm * float(offset_x)
                    zs = center_z + scale * self.radius_mm * float(offset_z)
                    idx_x = np.floor((xs + pixel * (det_x - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    idx_z = np.floor((zs + pixel * (det_z - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    valid = self._valid_mask(idx_x, idx_z, support_mode)
                    if np.any(valid):
                        rows = angle_base + idx_z[valid] * det_x + idx_x[valid]
                        out[valid] += weight[valid] * y[rows]
        return out

    def sensitivity(self, support_mode: str = "padded") -> np.ndarray:
        return self.adjoint(np.ones(int(self.config.n_rows), dtype=np.float64), support_mode=support_mode)

    def forward_delta(
        self,
        voxel_index: int,
        *,
        support_mode: str = "padded",
        hole_indices: Iterable[int] | None = None,
    ) -> np.ndarray:
        f = np.zeros(int(self.config.n_voxels), dtype=np.float64)
        f[int(voxel_index)] = 1.0
        return self.forward(f, support_mode=support_mode, hole_indices=hole_indices)

    def truncation_fraction(
        self,
        voxel_indices: np.ndarray | None = None,
        *,
        physical: bool = True,
        hole_indices: Iterable[int] | None = None,
    ) -> np.ndarray:
        cfg = self.config
        voxels = (
            np.arange(int(cfg.n_voxels), dtype=np.int64)
            if voxel_indices is None
            else np.asarray(voxel_indices, dtype=np.int64).reshape(-1)
        )
        missed = np.zeros(voxels.size, dtype=np.float64)
        total = np.zeros(voxels.size, dtype=np.float64)
        hole_list = list(range(self.hole_centers_mm.shape[0])) if hole_indices is None else [int(h) for h in hole_indices]
        yd_val = -float(cfg.detector_to_pinhole_mm)
        pixel = float(cfg.detector_pixel_size_mm)
        det_x = int(cfg.detector_x)
        det_z = int(cfg.detector_z)
        support_mode = "physical_padded" if physical else "padded"
        for angle_slot in range(int(cfg.n_angles)):
            x_rot, y_rot, z_rot, _, _ = self._angle_coordinates(angle_slot, voxels)
            for hole_index in hole_list:
                hole_x, hole_z = self.hole_centers_mm[hole_index]
                scale = (y_rot - yd_val) / np.maximum(y_rot, EPS)
                center_x = x_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_x)
                center_z = z_rot / np.maximum(y_rot, EPS) * yd_val + scale * float(hole_z)
                for offset_x, offset_z in self._aperture_offsets:
                    xs = center_x + scale * self.radius_mm * float(offset_x)
                    zs = center_z + scale * self.radius_mm * float(offset_z)
                    idx_x = np.floor((xs + pixel * (det_x - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    idx_z = np.floor((zs + pixel * (det_z - 1) / 2.0) / pixel + 0.5).astype(np.int32)
                    valid = self._valid_mask(idx_x, idx_z, support_mode)
                    missed += (~valid).astype(np.float64)
                    total += 1.0
        return missed / np.maximum(total, 1.0)


def poisson_deviance(y: np.ndarray, lam: np.ndarray, eps: float = 1.0e-9) -> float:
    y = np.asarray(y, dtype=np.float64)
    lam = np.maximum(np.asarray(lam, dtype=np.float64), eps)
    positive = y > 0.0
    out = np.zeros_like(lam)
    out[positive] = y[positive] * np.log(np.maximum(y[positive], eps) / lam[positive]) - y[positive] + lam[positive]
    out[~positive] = lam[~positive]
    return float(2.0 * np.sum(out))


def residual_map(y: np.ndarray, lam: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    return (np.asarray(y, dtype=np.float64) - np.asarray(lam, dtype=np.float64)) / np.sqrt(
        np.maximum(lam, eps)
    )


def detector_moments(projection: np.ndarray, detector_x: int = 160, detector_z: int = 80) -> dict[str, float]:
    p = np.asarray(projection, dtype=np.float64).reshape(-1)
    total = float(np.sum(p))
    if total <= 0.0:
        return {
            "total": 0.0,
            "centroid_x_px": float("nan"),
            "centroid_z_px": float("nan"),
            "sigma_x_px": float("nan"),
            "sigma_z_px": float("nan"),
        }
    grid = p.reshape(-1, int(detector_z), int(detector_x))
    summed = np.sum(grid, axis=0)
    zz, xx = np.meshgrid(np.arange(detector_z), np.arange(detector_x), indexing="ij")
    cx = float(np.sum(summed * xx) / total)
    cz = float(np.sum(summed * zz) / total)
    sx = float(np.sqrt(np.sum(summed * (xx - cx) ** 2) / total))
    sz = float(np.sqrt(np.sum(summed * (zz - cz) ** 2) / total))
    return {
        "total": total,
        "centroid_x_px": cx,
        "centroid_z_px": cz,
        "sigma_x_px": sx,
        "sigma_z_px": sz,
    }


def scalar_fit_and_relative_error(reference: np.ndarray, estimate: np.ndarray, eps: float = 1.0e-12) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    denom = float(np.dot(est, est))
    scale = float(np.dot(ref, est) / denom) if denom > eps else 0.0
    rel = float(np.linalg.norm(ref - scale * est) / (np.linalg.norm(ref) + eps))
    return scale, rel


def make_roi_detection_phantom(
    recon_shape: tuple[int, int, int] = (40, 60, 60),
    *,
    slice_index: int = 20,
    thickness: int = 5,
    background: float = 0.05,
) -> np.ndarray:
    from src.reporting_roi import _simulation_roi_geometry

    z_dim, y_dim, x_dim = recon_shape
    volume = np.full(recon_shape, float(background), dtype=np.float64)
    xc, yc, radii = _simulation_roi_geometry()
    concentrations = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(y_dim, dtype=float), np.arange(x_dim, dtype=float), indexing="ij")
    z0 = max(0, int(slice_index) - int(thickness) // 2)
    z1 = min(z_dim, int(slice_index) + int(thickness) // 2 + 1)
    for idx, (cx, cy, radius) in enumerate(zip(xc, yc, radii)):
        mask = (yy - cx) ** 2 + (xx - cy) ** 2 <= radius**2
        volume[z0:z1, mask] = concentrations[idx]
    return volume


def make_lumpy_phantom(
    recon_shape: tuple[int, int, int] = (40, 60, 60),
    *,
    seed: int = 20260509,
    n_lumps: int = 80,
    amplitude: float = 0.25,
    sigma_vox: float = 3.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z_dim, y_dim, x_dim = recon_shape
    volume = np.zeros(recon_shape, dtype=np.float64)
    zz, yy, xx = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing="ij")
    for _ in range(int(n_lumps)):
        cz = rng.uniform(0, z_dim - 1)
        cy = rng.uniform(0, y_dim - 1)
        cx = rng.uniform(0, x_dim - 1)
        amp = rng.uniform(0.3, 1.0) * float(amplitude)
        volume += amp * np.exp(-((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_vox**2))
    volume /= max(float(np.max(volume)), EPS)
    return volume * float(amplitude)


def roi_task_vectors(recon_shape: tuple[int, int, int] = (40, 60, 60), slice_index: int = 20) -> dict[str, np.ndarray]:
    from src.reporting_roi import _simulation_roi_geometry

    z_dim, y_dim, x_dim = recon_shape
    xc, yc, radii = _simulation_roi_geometry()
    yy, xx = np.meshgrid(np.arange(y_dim, dtype=float), np.arange(x_dim, dtype=float), indexing="ij")
    tasks: dict[str, np.ndarray] = {}
    for idx, (cx, cy, radius) in enumerate(zip(xc, yc, radii)):
        volume = np.zeros(recon_shape, dtype=np.float64)
        mask = (yy - cx) ** 2 + (xx - cy) ** 2 <= radius**2
        volume[int(slice_index), mask] = 1.0 / max(float(np.sum(mask)), 1.0)
        tasks[f"roi_{idx}"] = volume.reshape(-1)
    return tasks


def residual_structure_score(residual: np.ndarray, detector_x: int = 160, detector_z: int = 80) -> float:
    r = np.asarray(residual, dtype=np.float64).reshape(-1, int(detector_z), int(detector_x))
    if r.size == 0:
        return 0.0
    # Ratio of low-frequency row/column structure to total residual energy.
    mean_x = np.mean(r, axis=1, keepdims=True)
    mean_z = np.mean(r, axis=2, keepdims=True)
    structured = mean_x + mean_z - np.mean(r, axis=(1, 2), keepdims=True)
    return float(np.linalg.norm(structured) / (np.linalg.norm(r) + EPS))


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start
        return False
