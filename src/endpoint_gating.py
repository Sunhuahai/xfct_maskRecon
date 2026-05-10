from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.lifted_projective_dynamics import EPS


@dataclass(frozen=True)
class EndpointUncertaintyCoefficients:
    c_angle: float = 1.0
    c_fit: float = 1.0
    c_clip: float = 1.0
    c_range: float = 0.1


def _finite_image(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def compute_oracle_global_weight(
    y_L: np.ndarray,
    y_R: np.ndarray,
    y_true: np.ndarray,
    eps: float = EPS,
) -> float:
    """Return least-squares scalar weight for y = omega*y_L + (1-omega)*y_R."""
    left = _finite_image(y_L)
    right = _finite_image(y_R)
    truth = _finite_image(y_true)
    direction = left - right
    numerator = float(np.dot((truth - right).ravel(), direction.ravel()))
    denominator = float(np.dot(direction.ravel(), direction.ravel()) + float(eps))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def compute_oracle_pixel_weight(
    y_L: np.ndarray,
    y_R: np.ndarray,
    y_true: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Return clipped per-pixel oracle blend weights."""
    left = _finite_image(y_L)
    right = _finite_image(y_R)
    truth = _finite_image(y_true)
    direction = left - right
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = (truth - right) * direction / (direction**2 + float(eps))
    return np.clip(np.nan_to_num(omega, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def angular_blend(
    y_L: np.ndarray,
    y_R: np.ndarray,
    theta_L: float,
    theta_R: float,
    theta_k: float,
    eps: float = EPS,
) -> tuple[np.ndarray, float]:
    """Blend endpoints by angular distance and return (prediction, left_weight)."""
    denom = float(theta_R) - float(theta_L)
    if abs(denom) <= float(eps):
        omega = 1.0
    else:
        omega = (float(theta_R) - float(theta_k)) / denom
    omega = float(np.clip(omega, 0.0, 1.0))
    pred = omega * _finite_image(y_L) + (1.0 - omega) * _finite_image(y_R)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), omega


def inverse_uncertainty_blend(
    y_L: np.ndarray,
    y_R: np.ndarray,
    sigma_L: float,
    sigma_R: float,
    eps: float = EPS,
) -> tuple[np.ndarray, float]:
    """Blend endpoints with weights proportional to inverse uncertainty."""
    left_unc = max(float(sigma_L), 0.0)
    right_unc = max(float(sigma_R), 0.0)
    w_left = 1.0 / (left_unc + float(eps))
    w_right = 1.0 / (right_unc + float(eps))
    denom = w_left + w_right + float(eps)
    omega = float(w_left / denom)
    pred = omega * _finite_image(y_L) + (1.0 - omega) * _finite_image(y_R)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), omega


def _diagnostic_float(diagnostics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(diagnostics.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value if np.isfinite(value) else default


def compute_endpoint_uncertainty(
    diagnostics: dict[str, Any],
    coefficients: EndpointUncertaintyCoefficients | dict[str, float] | None = None,
    eps: float = EPS,
) -> float:
    """Return sigma^2 for endpoint inverse-uncertainty gating."""
    if coefficients is None:
        coeff = EndpointUncertaintyCoefficients()
    elif isinstance(coefficients, EndpointUncertaintyCoefficients):
        coeff = coefficients
    else:
        coeff = EndpointUncertaintyCoefficients(
            c_angle=float(coefficients.get("c_angle", 1.0)),
            c_fit=float(coefficients.get("c_fit", 1.0)),
            c_clip=float(coefficients.get("c_clip", 1.0)),
            c_range=float(coefficients.get("c_range", 0.1)),
        )

    angular_distance = _diagnostic_float(diagnostics, "angular_distance", 0.0)
    source_fit = _diagnostic_float(diagnostics, "source_self_fit_error", 0.0)
    clip_fraction = _diagnostic_float(diagnostics, "ratio_clip_fraction", 0.0)
    robust_var = _diagnostic_float(diagnostics, "robust_ratio_variance", 0.0)

    sigma_sq = (
        float(coeff.c_angle) * angular_distance**2
        + float(coeff.c_fit) * source_fit**2
        + float(coeff.c_clip) * max(clip_fraction, 0.0)
        + float(coeff.c_range) * max(robust_var, 0.0)
        + float(eps)
    )
    return float(max(sigma_sq, float(eps)))
