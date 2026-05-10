import numpy as np
from tqdm.auto import tqdm

from algorithm.recon_common import CommonConfig, load_recon_inputs

EPS = 1e-10


def compute_tv_derivative(img_3d, delta=1e-6):
    grad_x, grad_y, grad_z = np.gradient(img_3d)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + delta**2)

    norm_x = grad_x / grad_norm
    norm_y = grad_y / grad_norm
    norm_z = grad_z / grad_norm

    div_x = np.gradient(norm_x, axis=0)
    div_y = np.gradient(norm_y, axis=1)
    div_z = np.gradient(norm_z, axis=2)
    return -(div_x + div_y + div_z)


def tv_inner_loop(f_flat, recon_size, delta, n_steps, alpha):
    f = f_flat.reshape(recon_size)
    for _ in range(n_steps):
        f = f - alpha * compute_tv_derivative(f, delta=delta)
        f = np.maximum(f, 0.0)
    return f.ravel()


def load_em_tv_shared_inputs(
    common: CommonConfig,
    projection: np.ndarray | None = None,
    system_matrix=None,
):
    proj, y, A, proj_path, cij_path = load_recon_inputs(
        common.angle_count,
        common.background_offset,
        common.pad_x,
        proj_path=common.proj_path,
        cij_path=common.cij_path,
        projection=projection,
        cij=system_matrix,
    )
    n_angles = proj.shape[0]
    n_vox_cfg = int(np.prod(common.recon_size))
    n_vox_mat = int(A.shape[1])
    if n_vox_cfg != n_vox_mat:
        raise ValueError(
            "recon_size 与系统矩阵不匹配: "
            f"prod(recon_size)={n_vox_cfg}, A.shape[1]={n_vox_mat}. "
            "请在启动文件里修改 COMMON.recon_size。"
        )

    sensitivity = np.array(A.sum(axis=0)).ravel().astype(np.float64)
    sensitivity = np.maximum(sensitivity, EPS)

    return {
        "proj": proj,
        "y": y,
        "A": A,
        "proj_path": proj_path,
        "cij_path": cij_path,
        "n_angles": int(n_angles),
        "n_vox": n_vox_cfg,
        "sensitivity": sensitivity,
    }


def run_em_tv_reconstruction(
    exp_cfg: dict,
    common: CommonConfig,
    shared_inputs: dict,
):
    name = str(exp_cfg.get("name", "coarse_em_tv"))
    use_tv = bool(exp_cfg.get("use_tv", False))
    tv_beta = float(exp_cfg.get("tv_beta", 1.0))
    tv_delta = float(exp_cfg.get("tv_delta", 1e-6))
    tv_steps = int(exp_cfg.get("tv_steps", 5))
    tv_warmup = int(exp_cfg.get("tv_warmup", 3))
    tv_alpha_mode = str(exp_cfg.get("tv_alpha_mode", "auto"))
    tv_alpha_manual = float(exp_cfg.get("tv_alpha_manual", 0.05))
    tv_alpha_decay = float(exp_cfg.get("tv_alpha_decay", 0.0))
    lambda_l1 = float(exp_cfg.get("lambda_l1", 0.0))
    l1_step = float(exp_cfg.get("l1_step", 0.5))

    y = shared_inputs["y"]
    A = shared_inputs["A"]
    n_angles = shared_inputs["n_angles"]
    sensitivity = shared_inputs["sensitivity"]

    f = np.ones(shared_inputs["n_vox"], dtype=np.float64)
    sens_mean = float(np.mean(sensitivity))

    log_likelihood = np.zeros(common.num_iterations)
    relative_change = np.zeros(common.num_iterations)

    for nk in tqdm(range(common.num_iterations), desc=name, leave=False):
        f_old = f.copy()

        f = np.maximum(f, 1e-12)
        forward_proj = np.maximum(A @ f, EPS)
        correction_ratio = A.T @ (y / forward_proj)
        f = f * correction_ratio / sensitivity
        f = np.maximum(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        if lambda_l1 > 0.0 and l1_step > 0.0:
            f = np.maximum(f - l1_step * lambda_l1 / sensitivity, 0.0)

        if use_tv and nk >= tv_warmup and tv_steps > 0:
            if tv_alpha_mode == "auto":
                alpha = tv_beta * (0.1 / (sens_mean + 1e-12))
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

        forward_proj_check = np.maximum(A @ np.maximum(f, 0.0), EPS)
        log_likelihood[nk] = np.sum(forward_proj_check - y * np.log(forward_proj_check))
        relative_change[nk] = np.linalg.norm(f - f_old) / (np.linalg.norm(f_old) + 1e-12)

    f_out = f.reshape(common.recon_size)
    summary = {
        "name": name,
        "use_tv": use_tv,
        "tv_beta": tv_beta,
        "tv_delta": tv_delta,
        "tv_steps": tv_steps,
        "tv_warmup": tv_warmup,
        "tv_alpha_mode": tv_alpha_mode,
        "tv_alpha_manual": tv_alpha_manual,
        "tv_alpha_decay": tv_alpha_decay,
        "lambda_l1": lambda_l1,
        "l1_step": l1_step,
        "final_nll": float(log_likelihood[-1]),
        "final_rel": float(relative_change[-1]),
    }

    return {
        "summary": summary,
        "name": name,
        "reconstruction": f_out,
        "nll_history": log_likelihood,
        "relative_change": relative_change,
        "n_angles": int(n_angles),
        "params": {
            "angle_count": int(n_angles),
            "use_tv": use_tv,
            "tv_beta": tv_beta,
            "tv_delta": tv_delta,
            "tv_steps": tv_steps,
            "tv_warmup": tv_warmup,
            "tv_alpha_mode": tv_alpha_mode,
            "tv_alpha_manual": tv_alpha_manual,
            "tv_alpha_decay": tv_alpha_decay,
            "lambda_l1": lambda_l1,
            "l1_step": l1_step,
        },
    }
