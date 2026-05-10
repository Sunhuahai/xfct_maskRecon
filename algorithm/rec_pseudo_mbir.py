from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.sparse import load_npz
from tqdm.auto import tqdm

from algorithm.em_tv import (
    load_em_tv_shared_inputs,
    run_em_tv_reconstruction,
    tv_inner_loop,
)
from algorithm.recon_common import CommonConfig
from src.flux_projector import DepthAwareFluxProjector, FluxProjectorConfig
from src.projection_completion import (
    depth_aware_reconstruction_completion,
    forward_projection_completion,
    reconstruction_informed_completion,
)
from src.xfct_geometry import PinholeGeometry

EPS = 1e-10


def infer_target_cij_path(cij_path: str | None, source_angle_count: int, target_angle_count: int) -> str:
    if cij_path is None:
        raise ValueError("target_cij_path is required when common.cij_path is not set.")
    path = str(cij_path)
    pattern = rf"cij_{int(source_angle_count)}_"
    replacement = f"cij_{int(target_angle_count)}_"
    if re.search(pattern, path):
        target = re.sub(pattern, replacement, path, count=1)
    else:
        target = path.replace(f"_{int(source_angle_count)}_", f"_{int(target_angle_count)}_", 1)
    if target == path:
        raise ValueError(
            "Could not infer target system matrix path from "
            f"{path!r}; pass target_cij_path explicitly."
        )
    return target


def _angle_mask(target_angle_count: int, upsample_factor: int) -> tuple[np.ndarray, np.ndarray]:
    measured_angles = np.zeros(target_angle_count, dtype=bool)
    measured_angles[::upsample_factor] = True
    inserted_angles = ~measured_angles
    return measured_angles, inserted_angles


def _flatten_angle_mask(angle_mask: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(
        angle_mask.reshape((angle_mask.size,) + (1,) * len(spatial_shape)),
        (angle_mask.size,) + spatial_shape,
    ).ravel()


def _resolve_coarse_iterations(value: Any, total_iterations: int) -> int:
    if value is None or str(value).lower() == "auto":
        return max(1, min(15, int(round(0.6 * max(1, int(total_iterations))))))
    return max(1, int(value))


def _resolve_mbir_iterations(
    value: Any,
    total_iterations: int,
    coarse_iterations: int,
) -> int:
    if value is None or str(value).lower() in {"auto", "remaining"}:
        return max(1, int(total_iterations) - int(coarse_iterations))
    return max(1, int(value))


def _pseudo_loss_and_grad(
    A_pseudo,
    f: np.ndarray,
    y_pseudo: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    residual = np.asarray(A_pseudo @ f).ravel() - y_pseudo
    weighted = weights * residual
    loss = 0.5 * float(np.dot(residual, weighted))
    grad = np.asarray(A_pseudo.T @ weighted).ravel()
    return loss, grad


def build_reconstruction_informed_pseudo_data(
    *,
    coarse_reconstruction: np.ndarray,
    sparse_projection: np.ndarray,
    sparse_system_matrix,
    target_system_matrix,
    upsample_factor: int,
    completion_method: str,
    blend_multiplicative: float,
    uncertainty_floor: float,
    pseudo_weight_scale: float,
    depth_bin_count: int = 6,
    completion_voxel_size: float = 0.5,
    completion_detector_pixel_size: float = 0.25,
    completion_geometry: PinholeGeometry | None = None,
    residual_strength: float = 0.25,
) -> dict[str, np.ndarray]:
    f0 = np.asarray(coarse_reconstruction, dtype=np.float64).ravel()
    sparse_forward = np.asarray(sparse_system_matrix @ f0).reshape(sparse_projection.shape)
    target_shape = (sparse_projection.shape[0] * upsample_factor,) + sparse_projection.shape[1:]
    target_forward = np.asarray(target_system_matrix @ f0).reshape(target_shape)
    method = str(completion_method).strip().lower()
    if method in {"forward_only", "x0_forward", "coarse_forward"}:
        return forward_projection_completion(
            sparse_projection=sparse_projection,
            target_forward_projection=target_forward,
            upsample_factor=upsample_factor,
            uncertainty_floor=uncertainty_floor,
            pseudo_weight_scale=pseudo_weight_scale,
        )
    if method in {"depth_warp", "depth-aware", "depth_aware", "finite_angle_warp"}:
        geometry = completion_geometry or PinholeGeometry(
            detector_to_pinhole=-30.0,
            center_to_pinhole=50.0,
            detector_offset_x=-0.5,
        )
        return depth_aware_reconstruction_completion(
            sparse_projection=sparse_projection,
            sparse_forward_projection=sparse_forward,
            target_forward_projection=target_forward,
            coarse_reconstruction=coarse_reconstruction,
            recon_shape=tuple(coarse_reconstruction.shape),
            upsample_factor=upsample_factor,
            blend_multiplicative=blend_multiplicative,
            uncertainty_floor=uncertainty_floor,
            pseudo_weight_scale=pseudo_weight_scale,
            depth_bin_count=depth_bin_count,
            voxel_size=completion_voxel_size,
            detector_pixel_size=completion_detector_pixel_size,
            geometry=geometry,
            residual_strength=residual_strength,
        )
    return reconstruction_informed_completion(
        sparse_projection=sparse_projection,
        sparse_forward_projection=sparse_forward,
        target_forward_projection=target_forward,
        upsample_factor=upsample_factor,
        method=completion_method,
        blend_multiplicative=blend_multiplicative,
        uncertainty_floor=uncertainty_floor,
        pseudo_weight_scale=pseudo_weight_scale,
    )


def _build_flux_projector(
    exp_cfg: dict[str, Any],
    common: CommonConfig,
    target_angle_count: int,
    target_proj_shape: tuple[int, ...],
    target_system_matrix,
) -> DepthAwareFluxProjector:
    angles = 2.0 * np.pi * np.arange(target_angle_count, dtype=np.float64)
    angles = angles / float(target_angle_count)
    geometry = PinholeGeometry(
        detector_to_pinhole=float(exp_cfg.get("flux_detector_to_pinhole", -30.0)),
        center_to_pinhole=float(exp_cfg.get("flux_center_to_pinhole", 50.0)),
        detector_offset_x=float(exp_cfg.get("flux_detector_offset_x", -0.5)),
    )
    config = FluxProjectorConfig(
        angles=angles,
        detector_shape=tuple(target_proj_shape[1:]),
        recon_shape=tuple(common.recon_size),
        voxel_size=float(exp_cfg.get("flux_voxel_size", 0.5)),
        detector_pixel_size=float(exp_cfg.get("flux_detector_pixel_size", 0.25)),
        geometry=geometry,
        use_geometry_weight=bool(exp_cfg.get("flux_use_geometry_weight", True)),
    )
    return DepthAwareFluxProjector(
        config=config,
        system_matrix=target_system_matrix,
        include_projection=False,
        include_components=False,
    )


def _apply_pseudo_negative_controls(
    pseudo_data: dict[str, np.ndarray],
    inserted_angles: np.ndarray,
    exp_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    projection = np.asarray(pseudo_data["projection"], dtype=np.float64).copy()
    weights = np.asarray(pseudo_data["weights"], dtype=np.float64).copy()
    sigma = np.asarray(pseudo_data["sigma"], dtype=np.float64).copy()

    wrong_shift = int(exp_cfg.get("wrong_pseudo_angle_shift", 0))
    if wrong_shift != 0:
        projection = np.roll(projection, shift=wrong_shift, axis=0)
        weights = np.roll(weights, shift=wrong_shift, axis=0)
        sigma = np.roll(sigma, shift=wrong_shift, axis=0)

    if bool(exp_cfg.get("shuffle_pseudo_projection", False)):
        inserted_idx = np.flatnonzero(inserted_angles)
        rng = np.random.default_rng(int(exp_cfg.get("pseudo_shuffle_seed", 20260507)))
        shuffled_idx = inserted_idx.copy()
        rng.shuffle(shuffled_idx)
        projection[inserted_idx] = projection[shuffled_idx]
        weights[inserted_idx] = weights[shuffled_idx]
        sigma[inserted_idx] = sigma[shuffled_idx]

    measured_angles = ~inserted_angles
    projection[measured_angles] = pseudo_data["projection"][measured_angles]
    weights[measured_angles] = pseudo_data["weights"][measured_angles]
    sigma[measured_angles] = pseudo_data["sigma"][measured_angles]

    controlled = dict(pseudo_data)
    controlled["projection"] = projection
    controlled["weights"] = weights
    controlled["sigma"] = sigma
    return controlled


def run_pseudo_mbir_experiment(
    exp_cfg: dict[str, Any],
    common: CommonConfig,
    shared_inputs: dict[str, Any],
    target_system_matrix,
) -> dict[str, Any]:
    name = str(exp_cfg.get("name", "pseudo_mbir"))
    upsample_factor = int(exp_cfg.get("upsample_factor", 3))
    target_angle_count = int(shared_inputs["n_angles"] * upsample_factor)
    target_proj_shape = (target_angle_count,) + tuple(shared_inputs["proj"].shape[1:])

    if int(target_system_matrix.shape[0]) != int(np.prod(target_proj_shape)):
        raise ValueError(
            "Target system matrix row count does not match target projection shape: "
            f"A_target.shape={target_system_matrix.shape}, target_proj_shape={target_proj_shape}."
        )
    if int(target_system_matrix.shape[1]) != int(shared_inputs["n_vox"]):
        raise ValueError(
            "Target system matrix voxel count mismatch: "
            f"A_target.shape={target_system_matrix.shape}, n_vox={shared_inputs['n_vox']}."
        )

    coarse_iterations = _resolve_coarse_iterations(
        exp_cfg.get("coarse_iterations", "auto"),
        common.num_iterations,
    )
    mbir_iterations = _resolve_mbir_iterations(
        exp_cfg.get("mbir_iterations", "remaining"),
        common.num_iterations,
        coarse_iterations,
    )
    coarse_common = CommonConfig(
        num_iterations=coarse_iterations,
        angle_count=common.angle_count,
        background_offset=common.background_offset,
        recon_size=common.recon_size,
        slice_index=common.slice_index,
        pad_x=common.pad_x,
        proj_path=common.proj_path,
        cij_path=common.cij_path,
        output_root=common.output_root,
        output_dir=common.output_dir,
    )
    coarse_result = run_em_tv_reconstruction(
        exp_cfg={
            "name": f"{name}_coarse_em_tv",
            "use_tv": bool(exp_cfg.get("coarse_use_tv", True)),
            "tv_beta": float(exp_cfg.get("coarse_tv_beta", exp_cfg.get("tv_beta", 1.0))),
            "tv_delta": float(exp_cfg.get("tv_delta", 1e-6)),
            "tv_steps": int(exp_cfg.get("tv_steps", 5)),
            "tv_warmup": int(exp_cfg.get("tv_warmup", 10)),
            "tv_alpha_mode": str(exp_cfg.get("tv_alpha_mode", "auto")),
            "tv_alpha_manual": float(exp_cfg.get("tv_alpha_manual", 0.05)),
            "tv_alpha_decay": float(exp_cfg.get("tv_alpha_decay", 0.0)),
        },
        common=coarse_common,
        shared_inputs=shared_inputs,
    )

    pseudo_data = build_reconstruction_informed_pseudo_data(
        coarse_reconstruction=coarse_result["reconstruction"],
        sparse_projection=shared_inputs["proj"],
        sparse_system_matrix=shared_inputs["A"],
        target_system_matrix=target_system_matrix,
        upsample_factor=upsample_factor,
        completion_method=str(exp_cfg.get("completion_method", "phase_shift")),
        blend_multiplicative=float(exp_cfg.get("blend_multiplicative", 0.5)),
        uncertainty_floor=float(exp_cfg.get("uncertainty_floor", 1.0)),
        pseudo_weight_scale=float(exp_cfg.get("pseudo_weight_scale", 1.0)),
        depth_bin_count=int(exp_cfg.get("depth_bin_count", 6)),
        completion_voxel_size=float(exp_cfg.get("completion_voxel_size", 0.5)),
        completion_detector_pixel_size=float(
            exp_cfg.get("completion_detector_pixel_size", 0.25)
        ),
        completion_geometry=PinholeGeometry(
            detector_to_pinhole=float(
                exp_cfg.get("completion_detector_to_pinhole", -30.0)
            ),
            center_to_pinhole=float(exp_cfg.get("completion_center_to_pinhole", 50.0)),
            detector_offset_x=float(exp_cfg.get("completion_detector_offset_x", -0.5)),
        ),
        residual_strength=float(exp_cfg.get("residual_strength", 0.25)),
    )

    _, inserted_angles = _angle_mask(target_angle_count, upsample_factor)
    pseudo_data = _apply_pseudo_negative_controls(
        pseudo_data=pseudo_data,
        inserted_angles=inserted_angles,
        exp_cfg=exp_cfg,
    )
    inserted_flat = _flatten_angle_mask(inserted_angles, target_proj_shape[1:])
    A_pseudo = target_system_matrix[inserted_flat]
    y_pseudo = np.asarray(pseudo_data["projection"], dtype=np.float64).ravel()[inserted_flat]
    weights = np.asarray(pseudo_data["weights"], dtype=np.float64).ravel()[inserted_flat]
    positive_weights = weights[weights > 0.0]
    if positive_weights.size:
        max_weight = float(np.percentile(positive_weights, float(exp_cfg.get("pseudo_weight_percentile", 95.0))))
        weights = np.minimum(weights, max_weight)

    A = shared_inputs["A"]
    y = shared_inputs["y"]
    sensitivity = shared_inputs["sensitivity"]
    pseudo_sensitivity = np.asarray(A_pseudo.T @ weights).ravel()
    pseudo_sensitivity = np.maximum(np.nan_to_num(pseudo_sensitivity), 0.0)

    pseudo_mu = float(exp_cfg.get("pseudo_mu", 0.2))
    pseudo_step = float(exp_cfg.get("pseudo_step", 0.5))
    lambda_l1 = float(exp_cfg.get("lambda_l1", 0.0))
    lambda_flux = float(exp_cfg.get("lambda_flux", 0.0))
    flux_step = float(exp_cfg.get("flux_step", pseudo_step))
    flux_projector = None
    if lambda_flux > 0.0 and flux_step > 0.0:
        flux_projector = _build_flux_projector(
            exp_cfg=exp_cfg,
            common=common,
            target_angle_count=target_angle_count,
            target_proj_shape=target_proj_shape,
            target_system_matrix=target_system_matrix,
        )

    use_tv = bool(exp_cfg.get("use_tv", True))
    tv_beta = float(exp_cfg.get("tv_beta", 1.0))
    tv_delta = float(exp_cfg.get("tv_delta", 1e-6))
    tv_steps = int(exp_cfg.get("tv_steps", 5))
    tv_warmup = int(exp_cfg.get("tv_warmup", 10))
    tv_alpha_mode = str(exp_cfg.get("tv_alpha_mode", "auto"))
    tv_alpha_manual = float(exp_cfg.get("tv_alpha_manual", 0.05))
    tv_alpha_decay = float(exp_cfg.get("tv_alpha_decay", 0.0))
    sens_mean = float(np.mean(sensitivity))

    f = np.maximum(np.asarray(coarse_result["reconstruction"], dtype=np.float64).ravel(), EPS)
    nll_hist = np.zeros(mbir_iterations, dtype=np.float64)
    relative_change = np.zeros(mbir_iterations, dtype=np.float64)
    pseudo_loss_hist = np.zeros(mbir_iterations, dtype=np.float64)
    flux_loss_hist = np.zeros(mbir_iterations, dtype=np.float64)

    for nk in tqdm(range(mbir_iterations), desc=name, leave=False):
        f_old = f.copy()

        f = np.maximum(f, EPS)
        Af = np.maximum(A @ f, EPS)
        f = f * np.asarray(A.T @ (y / Af)).ravel() / sensitivity
        f = np.maximum(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        precond = sensitivity + pseudo_mu * pseudo_sensitivity + EPS
        if pseudo_mu > 0.0 and pseudo_step > 0.0:
            pseudo_loss, pseudo_grad = _pseudo_loss_and_grad(A_pseudo, f, y_pseudo, weights)
            f = np.maximum(f - pseudo_step * pseudo_mu * pseudo_grad / precond, 0.0)
            pseudo_loss_hist[nk] = pseudo_mu * pseudo_loss

        if lambda_l1 > 0.0:
            f = np.maximum(f - pseudo_step * lambda_l1 / precond, 0.0)

        if flux_projector is not None:
            flux_residual = flux_projector.M_forward(f)
            flux_loss = 0.5 * float(np.dot(flux_residual, flux_residual))
            flux_grad = flux_projector.M_adjoint(flux_residual)
            f = np.maximum(f - flux_step * lambda_flux * flux_grad / precond, 0.0)
            flux_loss_hist[nk] = lambda_flux * flux_loss

        if use_tv and nk >= tv_warmup and tv_steps > 0:
            if tv_alpha_mode == "auto":
                alpha = tv_beta * (0.1 / (sens_mean + EPS))
            else:
                alpha = tv_beta * tv_alpha_manual
            if tv_alpha_decay > 0:
                alpha = alpha * ((1.0 - tv_alpha_decay) ** nk)
            f = tv_inner_loop(
                f_flat=f,
                recon_size=common.recon_size,
                delta=tv_delta,
                n_steps=tv_steps,
                alpha=alpha,
            )

        Af_chk = np.maximum(A @ np.maximum(f, 0.0), EPS)
        measured_nll = float(np.sum(Af_chk - y * np.log(Af_chk)))
        nll_hist[nk] = measured_nll + pseudo_loss_hist[nk] + flux_loss_hist[nk]
        relative_change[nk] = float(np.linalg.norm(f - f_old) / (np.linalg.norm(f_old) + EPS))

    reconstruction = f.reshape(common.recon_size)
    final_pseudo_loss, _ = _pseudo_loss_and_grad(A_pseudo, f, y_pseudo, weights)
    final_forward = np.maximum(A @ np.maximum(f, 0.0), EPS)
    final_measured_nll = float(np.sum(final_forward - y * np.log(final_forward)))
    summary = {
        "name": name,
        "coarse_iterations": coarse_iterations,
        "mbir_iterations": mbir_iterations,
        "total_iteration_budget": int(common.num_iterations),
        "target_angle_count": target_angle_count,
        "upsample_factor": upsample_factor,
        "completion_method": str(exp_cfg.get("completion_method", "phase_shift")),
        "blend_multiplicative": float(exp_cfg.get("blend_multiplicative", 0.5)),
        "residual_strength": float(exp_cfg.get("residual_strength", 0.25)),
        "pseudo_mu": pseudo_mu,
        "pseudo_step": pseudo_step,
        "lambda_l1": lambda_l1,
        "lambda_flux": lambda_flux,
        "flux_model": "depth_aware" if flux_projector is not None else "disabled",
        "use_tv": use_tv,
        "tv_beta": tv_beta,
        "tv_steps": tv_steps,
        "tv_warmup": tv_warmup,
        "final_nll": float(nll_hist[-1]),
        "final_measured_nll": final_measured_nll,
        "final_pseudo_loss": float(pseudo_mu * final_pseudo_loss),
        "final_flux_loss": float(flux_loss_hist[-1]),
        "final_rel": float(relative_change[-1]),
        "pseudo_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "pseudo_weight_max": float(np.max(weights)) if weights.size else 0.0,
        "shuffle_pseudo_projection": bool(exp_cfg.get("shuffle_pseudo_projection", False)),
        "wrong_pseudo_angle_shift": int(exp_cfg.get("wrong_pseudo_angle_shift", 0)),
    }

    return {
        "summary": summary,
        "name": name,
        "reconstruction": reconstruction,
        "nll_history": nll_hist,
        "relative_change": relative_change,
        "pseudo_loss_history": pseudo_loss_hist,
        "flux_loss_history": flux_loss_hist,
        "coarse_reconstruction": coarse_result["reconstruction"],
        "pseudo_projection": pseudo_data["projection"],
        "pseudo_weights": pseudo_data["weights"],
        "pseudo_sigma": pseudo_data["sigma"],
        "params": dict(summary),
    }


def run_pseudo_mbir(
    common: CommonConfig,
    target_angle_count: int | None = None,
    target_cij_path: str | None = None,
    target_system_matrix=None,
    projection: np.ndarray | None = None,
    system_matrix=None,
    **kwargs,
) -> dict[str, Any]:
    shared = load_em_tv_shared_inputs(
        common,
        projection=projection,
        system_matrix=system_matrix,
    )
    if target_angle_count is None:
        target_angle_count = int(common.angle_count) * int(kwargs.get("upsample_factor", 3))
    upsample_factor = int(target_angle_count) // int(common.angle_count)
    if int(common.angle_count) * upsample_factor != int(target_angle_count):
        raise ValueError(
            "target_angle_count must be an integer multiple of common.angle_count: "
            f"{target_angle_count} vs {common.angle_count}."
        )

    if target_system_matrix is None:
        if target_cij_path is None:
            target_cij_path = infer_target_cij_path(
                common.cij_path,
                common.angle_count,
                target_angle_count,
            )
        target_system_matrix = load_npz(target_cij_path)
    else:
        target_cij_path = "<provided>"

    exp_cfg = {
        "name": "pseudo_mbir",
        "upsample_factor": upsample_factor,
        **kwargs,
    }
    result = run_pseudo_mbir_experiment(
        exp_cfg=exp_cfg,
        common=common,
        shared_inputs=shared,
        target_system_matrix=target_system_matrix,
    )
    result["proj_path"] = shared["proj_path"]
    result["cij_path"] = shared["cij_path"]
    result["target_cij_path"] = str(target_cij_path)
    return result
