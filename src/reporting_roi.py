from __future__ import annotations

import numpy as np


def _normalize_roi_layout(roi_layout: str) -> str:
    layout = str(roi_layout).strip().lower()
    if layout in {"experimental", "experiment", "exp", "real"}:
        return "experimental"
    if layout in {"simulation", "sim", "simulated"}:
        return "simulation"
    raise ValueError(
        "roi_layout must be 'experimental' or 'simulation', "
        f"got {roi_layout!r}."
    )


def _experimental_roi_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc = np.array([46, 28, 13, 16, 33, 47], dtype=float)
    yc = np.array([42, 48, 37, 18, 12, 23], dtype=float)
    radii = np.full(6, 5.0, dtype=float)
    return xc, yc, radii


def _simulation_roi_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc = np.array([12.0, 20.0, 38.0, 48.0, 39.0, 21.0], dtype=float)
    yc = np.array([30.0, 14.0, 14.0, 29.0, 45.0, 45.5], dtype=float)
    radii = np.full(6, 5.0, dtype=float)
    return xc, yc, radii


def _resolve_roi_geometry(
    roi_layout: str,
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = _normalize_roi_layout(roi_layout)
    if layout == "experimental":
        default_xc, default_yc, radii = _experimental_roi_geometry()
    else:
        default_xc, default_yc, radii = _simulation_roi_geometry()

    if roi_xc is None and roi_yc is None:
        return default_xc, default_yc, radii
    if roi_xc is None or roi_yc is None:
        raise ValueError("roi_xc and roi_yc must be set together.")

    xc = np.asarray(roi_xc, dtype=float).reshape(-1)
    yc = np.asarray(roi_yc, dtype=float).reshape(-1)
    if xc.size != 6 or yc.size != 6:
        raise ValueError(
            f"roi_xc and roi_yc must each contain 6 values, got {xc.size} and {yc.size}."
        )
    return xc, yc, radii


def roi_analysis(
    volume: np.ndarray,
    slice_index: int,
    recon_size: tuple[int, int, int],
    roi_layout: str = "experimental",
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    xc, yc, radii = _resolve_roi_geometry(
        roi_layout=roi_layout,
        roi_xc=roi_xc,
        roi_yc=roi_yc,
    )

    ff = np.asarray(volume[slice_index, :, :], dtype=float)
    x_grid, y_grid = np.meshgrid(
        np.arange(recon_size[1], dtype=float),
        np.arange(recon_size[2], dtype=float),
        indexing="ij",
    )

    rois = [
        (x_grid - xc[idx]) ** 2 + (y_grid - yc[idx]) ** 2 < radii[idx] ** 2
        for idx in range(xc.size)
    ]

    v = np.array([np.mean(ff[roi]) for roi in rois], dtype=float)
    bg_mean = float(v[5]) if float(v[5]) > 1e-6 else 1.0
    v2 = v / bg_mean

    s = np.array([np.std(ff[roi]) for roi in rois], dtype=float)
    s2 = s / bg_mean
    s2_ref = float(s2[0]) if float(s2[0]) > 1e-12 else 1.0

    cnr = (v2 - v2[0]) / s2_ref
    concentration = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=float)
    fit_concentration = concentration[1:]
    fit_cnr = cnr[1:]
    polyf = np.polyfit(fit_concentration, fit_cnr, 1)
    slope = float(polyf[0])
    if abs(slope) <= 1e-12:
        dl = float("inf")
    else:
        dl = float((4.0 - polyf[1]) / slope)

    fit_predicted = polyf[0] * fit_concentration + polyf[1]
    ss_res = float(np.sum((fit_cnr - fit_predicted) ** 2))
    ss_tot = float(np.sum((fit_cnr - np.mean(fit_cnr)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "ff": ff,
        "CNR": cnr,
        "DL": dl,
        "r_squared": r_squared,
        "polyf": polyf,
        "concentration": concentration,
        "fit_concentration": fit_concentration,
        "fit_cnr": fit_cnr,
        "V": v,
        "V2": v2,
        "S": s,
        "S2": s2,
        "xc": xc,
        "yc": yc,
        "radius": radii,
    }


def scale_reconstruction_to_roi_reference(
    reconstruction: np.ndarray,
    slice_index: int,
    recon_size: tuple[int, int, int],
    roi_layout: str,
    roi_reference_index: int,
    roi_reference_value_mgml: float,
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, float]:
    roi = roi_analysis(
        reconstruction,
        slice_index,
        recon_size,
        roi_layout=roi_layout,
        roi_xc=roi_xc,
        roi_yc=roi_yc,
    )
    roi_values = np.asarray(roi["V"], dtype=float)
    reference_index = int(np.clip(roi_reference_index, 0, roi_values.size - 1))
    reference_mean = float(roi_values[reference_index])
    target_value = max(float(roi_reference_value_mgml), 0.0)
    if reference_mean <= 1e-12 or target_value <= 0.0:
        return np.asarray(reconstruction, dtype=float), 1.0

    scale = target_value / reference_mean
    return np.asarray(reconstruction, dtype=float) * scale, scale
