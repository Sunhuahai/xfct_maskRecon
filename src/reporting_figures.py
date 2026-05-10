from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from src.reporting_roi import roi_analysis


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def apply_projection_median_filter(
    projection: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 3:
        raise ValueError("projection must have shape [Angle, H, W].")

    kernel_size = max(int(kernel_size), 1)
    if kernel_size <= 1:
        return projection.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1

    return median_filter(
        projection,
        size=(1, kernel_size, kernel_size),
        mode="nearest",
    )


def save_reconstruction_curve_figure(
    nll_history: np.ndarray,
    relative_change: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = _ensure_parent(output_path)
    nll_history = np.asarray(nll_history, dtype=float).reshape(-1)
    relative_change = np.asarray(relative_change, dtype=float).reshape(-1)
    iterations = np.arange(1, nll_history.size + 1, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].plot(iterations, nll_history, linewidth=2.0, marker="o", markersize=3)
    axes[0].set_title("Poisson NLL")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("NLL")
    axes[0].grid(True, which="major", alpha=0.35)

    rel_safe = np.maximum(relative_change, 1e-12)
    axes[1].semilogy(iterations, rel_safe, linewidth=2.0, marker="o", markersize=3)
    axes[1].set_title("Relative Change")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative Change")
    axes[1].grid(True, which="major", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roi_detection_figure(
    reconstruction: np.ndarray,
    recon_shape: tuple[int, int, int],
    slice_index: int,
    roi_layout: str,
    roi_reference_value_mgml: float,
    output_path: str | Path,
    roi_xc: np.ndarray | list[float] | tuple[float, ...] | None = None,
    roi_yc: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    roi = roi_analysis(
        reconstruction,
        slice_index,
        recon_shape,
        roi_layout=roi_layout,
        roi_xc=roi_xc,
        roi_yc=roi_yc,
    )
    ff = np.asarray(roi["ff"], dtype=float)

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    vmax = max(float(roi_reference_value_mgml), 1e-12)
    im = axes[0].imshow(ff, cmap="jet", origin="upper", vmin=0.0, vmax=vmax)
    for idx, (xc, yc, radius) in enumerate(zip(roi["xc"], roi["yc"], roi["radius"]), start=1):
        circle = plt.Circle(
            (float(yc), float(xc)),
            float(radius),
            fill=False,
            color="white",
            linewidth=1.5,
        )
        axes[0].add_patch(circle)
        axes[0].text(
            float(yc),
            float(xc),
            f"R{idx}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
        )
    axes[0].set_title(f"ROI Slice\nslice={slice_index}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="mg/ml")

    concentration = np.asarray(roi["fit_concentration"], dtype=float)
    cnr = np.asarray(roi["fit_cnr"], dtype=float)
    polyf = np.asarray(roi["polyf"], dtype=float)
    dl = float(roi["DL"])
    full_concentration = np.asarray(roi["concentration"], dtype=float)
    fit_line_x = np.linspace(0.0, float(np.max(full_concentration)), 200)
    cnr_fit = np.polyval(polyf, concentration)
    fit_line_y = np.polyval(polyf, fit_line_x)
    ss_res = float(np.sum((cnr - cnr_fit) ** 2))
    ss_tot = float(np.sum((cnr - np.mean(cnr)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    axes[1].plot(
        concentration,
        cnr,
        "-o",
        color="#1f77b4",
        linewidth=2.0,
        markersize=5,
        label="Measured CNR",
    )
    axes[1].plot(
        fit_line_x,
        fit_line_y,
        "--",
        color="#d62728",
        linewidth=1.8,
        label="Linear fit",
    )
    axes[1].axhline(4.0, color="#2ca02c", linestyle="--", alpha=0.85, label="CNR=4")
    axes[1].set_title(f"Detection Limit\nDL={dl:.4f} mg/ml")
    axes[1].set_xlabel("Concentration (mg/ml)")
    axes[1].set_ylabel("CNR")
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, which="major", alpha=0.35)
    axes[1].legend(loc="best")
    axes[1].text(
        0.20,
        0.60,
        f"$R^2$={r2:.4f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return roi
