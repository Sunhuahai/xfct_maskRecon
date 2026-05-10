from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MaskSpec:
    layout: str
    hole_centers: np.ndarray
    hole_diameter_mm: float
    pitch_mm: float

    @property
    def hole_count(self) -> int:
        return int(self.hole_centers.shape[0])

    @property
    def tag(self) -> str:
        pitch = _format_float(self.pitch_mm)
        diameter = _format_float(self.hole_diameter_mm)
        return f"{self.layout}_n{self.hole_count}_p{pitch}_d{diameter}"


def _format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "d").replace("-", "m")


def centered_grid_mask(rows: int, cols: int, pitch_mm: float) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    x_coords = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0) * pitch_mm
    z_coords = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0) * pitch_mm
    xx, zz = np.meshgrid(x_coords, z_coords, indexing="xy")
    return np.column_stack([xx.ravel(), zz.ravel()])


def cross_mask(pitch_mm: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [-pitch_mm, 0.0],
            [pitch_mm, 0.0],
            [0.0, -pitch_mm],
            [0.0, pitch_mm],
        ],
        dtype=np.float64,
    )


def random_mask(
    hole_count: int,
    extent_mm: float,
    min_separation_mm: float,
    seed: int,
) -> np.ndarray:
    if hole_count <= 0:
        raise ValueError("hole_count must be positive.")
    if extent_mm <= 0:
        raise ValueError("extent_mm must be positive.")
    if min_separation_mm < 0:
        raise ValueError("min_separation_mm must be nonnegative.")

    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = [np.array([0.0, 0.0], dtype=np.float64)]
    attempts = 0
    max_attempts = 20000
    while len(centers) < hole_count and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(-extent_mm / 2.0, extent_mm / 2.0, size=2)
        distances = [float(np.linalg.norm(candidate - center)) for center in centers]
        if min(distances) >= min_separation_mm:
            centers.append(candidate)
    if len(centers) != hole_count:
        raise RuntimeError(
            "Could not place random mask holes. Reduce hole_count or min_separation_mm."
        )
    return np.asarray(centers, dtype=np.float64)


def load_mask_centers(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        centers = np.load(path)
    else:
        centers = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Mask center file must have shape [n, 2], got {centers.shape}.")
    return centers


def build_mask_spec(
    layout: str = "grid3x3",
    pitch_mm: float = 6.0,
    hole_diameter_mm: float = 1.25,
    *,
    rows: int = 3,
    cols: int = 3,
    hole_count: int = 9,
    random_extent_mm: float = 18.0,
    min_separation_mm: float | None = None,
    seed: int = 20260509,
    mask_file: str | Path | None = None,
) -> MaskSpec:
    layout_key = str(layout).strip().lower()
    if mask_file is not None:
        centers = load_mask_centers(mask_file)
        layout_key = "custom"
    elif layout_key in {"single", "pinhole"}:
        centers = np.array([[0.0, 0.0]], dtype=np.float64)
    elif layout_key in {"grid3x3", "grid"}:
        centers = centered_grid_mask(rows=rows, cols=cols, pitch_mm=pitch_mm)
        layout_key = f"grid{rows}x{cols}"
    elif layout_key == "cross5":
        centers = cross_mask(pitch_mm=pitch_mm)
    elif layout_key in {"random", "random9"}:
        min_sep = hole_diameter_mm if min_separation_mm is None else min_separation_mm
        centers = random_mask(
            hole_count=hole_count,
            extent_mm=random_extent_mm,
            min_separation_mm=min_sep,
            seed=seed,
        )
        layout_key = "random"
    else:
        raise ValueError(
            "layout must be one of single, grid3x3, grid, cross5, random, or use mask_file."
        )

    return MaskSpec(
        layout=layout_key,
        hole_centers=np.asarray(centers, dtype=np.float64),
        hole_diameter_mm=float(hole_diameter_mm),
        pitch_mm=float(pitch_mm),
    )


def save_mask_metadata(path: str | Path, spec: MaskSpec) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layout": spec.layout,
        "tag": spec.tag,
        "hole_count": spec.hole_count,
        "hole_diameter_mm": spec.hole_diameter_mm,
        "pitch_mm": spec.pitch_mm,
        "hole_centers_xz_mm": spec.hole_centers.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
