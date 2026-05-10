from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskDecodeConfig:
    detector_pixel_size_mm: float = 0.25
    detector_to_mask_mm: float = 30.0
    center_to_mask_mm: float = 50.0
    wiener_reg: float = 0.03
    normalize_kernel: bool = True
    clip_nonnegative: bool = True

    @property
    def nominal_magnification(self) -> float:
        return (self.center_to_mask_mm + self.detector_to_mask_mm) / self.center_to_mask_mm


def build_shift_kernel(
    detector_shape: tuple[int, int],
    hole_centers_xz_mm: np.ndarray,
    config: MaskDecodeConfig | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    cfg = MaskDecodeConfig() if config is None else config
    height, width = detector_shape
    centers = np.asarray(hole_centers_xz_mm, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"hole_centers_xz_mm must have shape [n, 2], got {centers.shape}.")

    if weights is None:
        weights_arr = np.ones(centers.shape[0], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.size != centers.shape[0]:
            raise ValueError("weights must have one entry per mask hole.")

    kernel = np.zeros((height, width), dtype=np.float64)
    pixel_shifts = cfg.nominal_magnification * centers / cfg.detector_pixel_size_mm
    for (dx_pix, dz_pix), weight in zip(pixel_shifts, weights_arr):
        row = int(np.round(dz_pix)) % height
        col = int(np.round(dx_pix)) % width
        kernel[row, col] += float(weight)

    if cfg.normalize_kernel:
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0.0:
            kernel = kernel / kernel_sum
    return kernel


def wiener_decode_projection(
    projection: np.ndarray,
    hole_centers_xz_mm: np.ndarray,
    config: MaskDecodeConfig | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = MaskDecodeConfig() if config is None else config
    proj = np.asarray(projection, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"projection must have shape [angle, z, x], got {proj.shape}.")

    kernel = build_shift_kernel(
        detector_shape=tuple(proj.shape[1:]),
        hole_centers_xz_mm=hole_centers_xz_mm,
        config=cfg,
        weights=weights,
    )
    h_fft = np.fft.fft2(kernel)
    denom = np.abs(h_fft) ** 2 + float(cfg.wiener_reg)

    decoded = np.empty_like(proj, dtype=np.float64)
    for angle_idx in range(proj.shape[0]):
        y_fft = np.fft.fft2(proj[angle_idx])
        decoded_angle = np.fft.ifft2(y_fft * np.conj(h_fft) / denom).real
        if cfg.clip_nonnegative:
            decoded_angle = np.maximum(decoded_angle, 0.0)
        decoded[angle_idx] = decoded_angle
    return decoded, kernel


def throughput_gain(
    hole_count: int,
    mask_hole_diameter_mm: float,
    reference_hole_diameter_mm: float,
) -> float:
    if reference_hole_diameter_mm <= 0.0:
        raise ValueError("reference_hole_diameter_mm must be positive.")
    return float(hole_count) * (float(mask_hole_diameter_mm) / reference_hole_diameter_mm) ** 2
