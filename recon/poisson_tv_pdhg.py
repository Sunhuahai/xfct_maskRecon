from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from src.mask_xfct_model import EPS, poisson_deviance, residual_map
from src.reporting_roi import roi_analysis


def gradient3(volume: np.ndarray) -> np.ndarray:
    f = np.asarray(volume, dtype=np.float64)
    grad = np.zeros((3,) + f.shape, dtype=np.float64)
    grad[0, :-1, :, :] = f[1:, :, :] - f[:-1, :, :]
    grad[1, :, :-1, :] = f[:, 1:, :] - f[:, :-1, :]
    grad[2, :, :, :-1] = f[:, :, 1:] - f[:, :, :-1]
    return grad


def gradient_adjoint(dual: np.ndarray) -> np.ndarray:
    p = np.asarray(dual, dtype=np.float64)
    out = np.zeros(p.shape[1:], dtype=np.float64)
    out[:-1, :, :] -= p[0, :-1, :, :]
    out[1:, :, :] += p[0, :-1, :, :]
    out[:, :-1, :] -= p[1, :, :-1, :]
    out[:, 1:, :] += p[1, :, :-1, :]
    out[:, :, :-1] -= p[2, :, :, :-1]
    out[:, :, 1:] += p[2, :, :, :-1]
    return out


def tv_norm(volume: np.ndarray, eps: float = 1.0e-12) -> float:
    g = gradient3(volume)
    return float(np.sum(np.sqrt(np.sum(g * g, axis=0) + eps)))


def _forward(A: Any, x: np.ndarray, support_mode: str = "padded") -> np.ndarray:
    if sparse.issparse(A):
        return np.asarray(A @ x, dtype=np.float64).reshape(-1)
    if hasattr(A, "forward"):
        return np.asarray(A.forward(x, support_mode=support_mode), dtype=np.float64).reshape(-1)
    raise TypeError("A must be a scipy sparse matrix or expose forward(x, support_mode=...).")


def _adjoint(A: Any, y: np.ndarray, support_mode: str = "padded") -> np.ndarray:
    if sparse.issparse(A):
        return np.asarray(A.T @ y, dtype=np.float64).reshape(-1)
    if hasattr(A, "adjoint"):
        return np.asarray(A.adjoint(y, support_mode=support_mode), dtype=np.float64).reshape(-1)
    raise TypeError("A must be a scipy sparse matrix or expose adjoint(y, support_mode=...).")


def _operator_shape(A: Any) -> tuple[int, int]:
    if sparse.issparse(A):
        return A.shape
    if hasattr(A, "shape"):
        return tuple(A.shape)  # type: ignore[return-value]
    raise TypeError("A must expose shape.")


def _as_background(background: float | np.ndarray, n_rows: int) -> np.ndarray:
    b = np.asarray(background, dtype=np.float64)
    if b.ndim == 0:
        return np.full(n_rows, float(b), dtype=np.float64)
    b = b.reshape(-1)
    if b.size != int(n_rows):
        raise ValueError(f"background length {b.size} does not match row count {n_rows}.")
    return b


def estimate_operator_norm(
    A: Any,
    recon_shape: tuple[int, int, int],
    *,
    support_mode: str = "padded",
    power_iterations: int = 6,
    seed: int = 20260509,
) -> float:
    rng = np.random.default_rng(seed)
    _, n_cols = _operator_shape(A)
    x = rng.normal(size=n_cols)
    x /= max(np.linalg.norm(x), EPS)
    last_norm = 1.0
    for _ in range(max(int(power_iterations), 1)):
        Ax = _forward(A, x, support_mode=support_mode)
        gx = gradient3(x.reshape(recon_shape))
        y = _adjoint(A, Ax, support_mode=support_mode) + gradient_adjoint(gx).reshape(-1)
        last_norm = float(np.linalg.norm(y))
        x = y / max(last_norm, EPS)
    return float(np.sqrt(max(last_norm, EPS)))


@dataclass
class PoissonTVResult:
    reconstruction: np.ndarray
    objective_history: np.ndarray
    deviance_history: np.ndarray
    relative_change: np.ndarray
    roi_history: list[dict[str, float]]
    lambda_hat: np.ndarray
    residual: np.ndarray
    params: dict[str, Any]
    diagnostics: dict[str, Any]


def _roi_summary(volume: np.ndarray, recon_shape: tuple[int, int, int], slice_index: int, roi_layout: str) -> dict[str, Any]:
    roi = roi_analysis(volume, slice_index=slice_index, recon_size=recon_shape, roi_layout=roi_layout)
    polyf = np.asarray(roi["polyf"], dtype=float)
    return {
        "detection_limit_mgml": float(roi["DL"]),
        "detection_limit_valid": bool(roi.get("detection_limit_valid", False)),
        "detection_limit_invalid": bool(roi.get("detection_limit_invalid", True)),
        "detection_limit_invalid_reason": str(roi.get("detection_limit_invalid_reason", "")),
        "roi_r_squared": float(roi["r_squared"]),
        "cnr_slope": float(polyf[0]),
        "cnr_intercept": float(polyf[1]),
        "roi_mean_0": float(np.asarray(roi["V"], dtype=float)[0]),
        "roi_mean_5": float(np.asarray(roi["V"], dtype=float)[5]),
    }


def _history_diagnostics(
    *,
    reconstruction: np.ndarray,
    lambda_hat: np.ndarray,
    objective_history: np.ndarray,
    deviance_history: np.ndarray,
    relative_change: np.ndarray,
) -> dict[str, Any]:
    objective = np.asarray(objective_history, dtype=np.float64)
    deviance = np.asarray(deviance_history, dtype=np.float64)
    rel = np.asarray(relative_change, dtype=np.float64)
    finite_objective = bool(np.all(np.isfinite(objective)))
    finite_deviance = bool(np.all(np.isfinite(deviance)))
    finite_relative_change = bool(np.all(np.isfinite(rel)))
    finite_lambda = bool(np.all(np.isfinite(lambda_hat)))
    positive_lambda = bool(np.all(np.asarray(lambda_hat) > 0.0))
    nonnegative_reconstruction = bool(np.min(reconstruction) >= -1.0e-12)
    if objective.size >= 2 and finite_objective:
        deltas = np.diff(objective)
        increases = deltas > 1.0e-8 * np.maximum(np.abs(objective[:-1]), 1.0)
        max_rel_increase = float(
            np.max(deltas / np.maximum(np.abs(objective[:-1]), 1.0))
        )
        objective_nonmonotone_steps = int(np.sum(increases))
    else:
        max_rel_increase = 0.0
        objective_nonmonotone_steps = 0
    return {
        "finite_objective": finite_objective,
        "finite_deviance": finite_deviance,
        "finite_relative_change": finite_relative_change,
        "finite_lambda": finite_lambda,
        "positive_lambda": positive_lambda,
        "nonnegative_reconstruction": nonnegative_reconstruction,
        "objective_nonmonotone_steps": objective_nonmonotone_steps,
        "objective_max_relative_increase": max_rel_increase,
        "final_relative_change": float(rel[-1]) if rel.size else float("nan"),
        "objective_behavior": (
            "monotone"
            if objective_nonmonotone_steps == 0
            else "nonmonotone; PDHG objectives can increase, see recorded max relative increase"
        ),
    }


def run_poisson_tv_pdhg(
    A: Any,
    y: np.ndarray,
    *,
    recon_shape: tuple[int, int, int] = (40, 60, 60),
    background: float | np.ndarray = 0.0,
    beta: float = 1.0e-4,
    num_iterations: int = 50,
    init: np.ndarray | None = None,
    support_mode: str = "padded",
    tau: float | None = None,
    sigma: float | None = None,
    theta: float = 1.0,
    norm_power_iterations: int = 6,
    seed: int = 20260509,
    roi_layout: str = "simulation",
    slice_index: int = 20,
    roi_every: int = 1,
    eps: float = 1.0e-9,
) -> PoissonTVResult:
    n_rows, n_cols = _operator_shape(A)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size != int(n_rows):
        raise ValueError(f"y length {y.size} does not match operator rows {n_rows}.")
    b = _as_background(background, int(n_rows))
    rng = np.random.default_rng(seed)
    if init is None:
        # Backprojected constant initialization avoids a zero-gradient dead start.
        init_value = max(float(np.mean(np.maximum(y - b, 0.0))), 1.0e-3)
        f = np.full(int(n_cols), init_value, dtype=np.float64)
        sens = _adjoint(A, np.ones_like(y), support_mode=support_mode)
        if np.any(sens > 0.0):
            f *= float(np.mean(sens[sens > 0.0])) / np.maximum(sens, np.percentile(sens[sens > 0.0], 10))
            f = np.nan_to_num(f, nan=init_value, posinf=init_value, neginf=0.0)
    else:
        f = np.maximum(np.asarray(init, dtype=np.float64).reshape(-1), 0.0)
    if f.size != int(n_cols):
        raise ValueError(f"init size {f.size} does not match operator columns {n_cols}.")
    f_bar = f.copy()
    q = np.zeros(int(n_rows), dtype=np.float64)
    p = np.zeros((3,) + tuple(recon_shape), dtype=np.float64)

    if tau is None or sigma is None:
        op_norm = estimate_operator_norm(
            A,
            recon_shape,
            support_mode=support_mode,
            power_iterations=norm_power_iterations,
            seed=seed,
        )
        step = 0.98 / max(op_norm, EPS)
        tau = step if tau is None else float(tau)
        sigma = step if sigma is None else float(sigma)
    tau = float(tau)
    sigma = float(sigma)
    theta = float(theta)
    beta = float(beta)

    objective_history = np.zeros(int(num_iterations), dtype=np.float64)
    deviance_history = np.zeros(int(num_iterations), dtype=np.float64)
    relative_change = np.zeros(int(num_iterations), dtype=np.float64)
    roi_history: list[dict[str, float]] = []

    for iteration in range(int(num_iterations)):
        f_old = f.copy()

        lam_bar = _forward(A, f_bar, support_mode=support_mode) + b
        v = q + sigma * lam_bar
        # prox_{sigma F*}(v), where F(lambda)=lambda-y log(lambda)
        q = 0.5 * (v + 1.0 - np.sqrt((v - 1.0) ** 2 + 4.0 * sigma * np.maximum(y, 0.0)))

        if beta > 0.0:
            p = p + sigma * gradient3(f_bar.reshape(recon_shape))
            norm = np.sqrt(np.sum(p * p, axis=0))
            scale = np.maximum(1.0, norm / beta)
            p = p / scale
        else:
            p.fill(0.0)

        primal_grad = _adjoint(A, q, support_mode=support_mode) + gradient_adjoint(p).reshape(-1)
        f = np.maximum(f - tau * primal_grad, 0.0)
        f_bar = f + theta * (f - f_old)

        lam = _forward(A, f, support_mode=support_mode) + b
        safe_lam = np.maximum(lam, eps)
        data_obj = float(np.sum(safe_lam - y * np.log(safe_lam)))
        tv_obj = beta * tv_norm(f.reshape(recon_shape)) if beta > 0.0 else 0.0
        objective_history[iteration] = data_obj + tv_obj
        deviance_history[iteration] = poisson_deviance(y, safe_lam, eps=eps)
        relative_change[iteration] = float(np.linalg.norm(f - f_old) / (np.linalg.norm(f_old) + EPS))
        if not np.all(np.isfinite(lam)):
            raise FloatingPointError(f"Non-finite lambda at iteration {iteration + 1}.")
        if not np.isfinite(objective_history[iteration]):
            raise FloatingPointError(f"Non-finite objective at iteration {iteration + 1}.")
        if not np.isfinite(deviance_history[iteration]):
            raise FloatingPointError(f"Non-finite deviance at iteration {iteration + 1}.")
        if not np.isfinite(relative_change[iteration]):
            raise FloatingPointError(f"Non-finite relative change at iteration {iteration + 1}.")
        if np.min(f) < -1.0e-12:
            raise FloatingPointError(f"Negative reconstruction value at iteration {iteration + 1}.")
        if roi_every > 0 and (iteration % int(roi_every) == 0 or iteration == int(num_iterations) - 1):
            item = _roi_summary(f.reshape(recon_shape), recon_shape, slice_index, roi_layout)
            item["iteration"] = float(iteration + 1)
            roi_history.append(item)

    lambda_hat = _forward(A, f, support_mode=support_mode) + b
    if not np.all(np.isfinite(lambda_hat)):
        raise FloatingPointError("Non-finite final lambda.")
    if np.min(lambda_hat) <= 0.0:
        raise FloatingPointError("Final lambda must be strictly positive after background.")
    resid = residual_map(y, lambda_hat, eps=eps)
    diagnostics = _history_diagnostics(
        reconstruction=f.reshape(recon_shape),
        lambda_hat=lambda_hat,
        objective_history=objective_history,
        deviance_history=deviance_history,
        relative_change=relative_change,
    )
    return PoissonTVResult(
        reconstruction=f.reshape(recon_shape),
        objective_history=objective_history,
        deviance_history=deviance_history,
        relative_change=relative_change,
        roi_history=roi_history,
        lambda_hat=lambda_hat,
        residual=resid,
        params={
            "beta": beta,
            "num_iterations": int(num_iterations),
            "tau": tau,
            "sigma": sigma,
            "theta": theta,
            "support_mode": support_mode,
            "seed": int(seed),
            "roi_layout": roi_layout,
            "slice_index": int(slice_index),
        },
        diagnostics=diagnostics,
    )
