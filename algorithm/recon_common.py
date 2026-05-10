import os
from dataclasses import dataclass

import numpy as np
from scipy.sparse import load_npz


@dataclass
class CommonConfig:
    num_iterations: int
    angle_count: int
    background_offset: float = 0.53
    recon_size: tuple = (60, 70, 70)
    slice_index: int = 27
    pad_x: int = 40
    proj_path: str | None = None
    cij_path: str | None = None
    output_root: str = "results"
    output_dir: str | None = None


def build_data_paths(
    angle_count: int,
    proj_path: str | None = None,
    cij_path: str | None = None,
):
    if proj_path is None:
        proj_path = f"data/fluorescence_projections/spec_{angle_count}_proj.npy"
    if cij_path is None:
        cij_path = f"data/system_matrix/cij_{angle_count}_3d_mod27_p1d25.npz"
    return proj_path, cij_path


def prepare_projection(
    projection: np.ndarray,
    angle_count: int,
    background_offset: float,
    pad_x: int = 40,
):
    proj = np.asarray(projection, dtype=np.float64)
    proj = proj - background_offset

    if proj.ndim == 3:
        if proj.shape[0] != angle_count:
            raise ValueError(
                "三维投影的 angle 维度与配置不匹配: "
                f"proj.shape={proj.shape}, angle_count={angle_count}."
            )
        proj = np.pad(proj, ((0, 0), (0, 0), (pad_x, pad_x)), mode="constant")
    elif proj.ndim == 4:
        if proj.shape[0] == angle_count:
            pass
        elif proj.shape[1] == angle_count:
            proj = np.transpose(proj, (1, 0, 2, 3))
        else:
            raise ValueError(
                "无法识别多探测器投影维度顺序: "
                f"proj.shape={proj.shape}, angle_count={angle_count}. "
                "期望为 (angle, detector, z, x) 或 (detector, angle, z, x)。"
            )
        proj = np.pad(proj, ((0, 0), (0, 0), (0, 0), (pad_x, pad_x)), mode="constant")
    else:
        raise ValueError(
            f"不支持的投影维度 proj.ndim={proj.ndim}, proj.shape={proj.shape}"
        )

    proj = np.maximum(proj, 0.0)
    y = proj.ravel().astype(np.float64)
    return proj, y


def load_recon_inputs(
    angle_count: int,
    background_offset: float,
    pad_x: int = 40,
    proj_path: str | None = None,
    cij_path: str | None = None,
    projection: np.ndarray | None = None,
    cij=None,
):
    proj_path, cij_path = build_data_paths(
        angle_count=angle_count,
        proj_path=proj_path,
        cij_path=cij_path,
    )
    if projection is None:
        projection = np.load(proj_path)
    if cij is None:
        cij = load_npz(cij_path)

    proj, y = prepare_projection(
        projection=projection,
        angle_count=angle_count,
        background_offset=background_offset,
        pad_x=pad_x,
    )

    if cij.shape[0] != y.size:
        raise ValueError(
            "投影数据长度与系统矩阵行数不匹配: "
            f"proj.shape={proj.shape}, y.size={y.size}, A.shape={cij.shape}. "
            "请确认角度数、探测器数和 pad_x 与系统矩阵一致。"
        )

    return proj, y, cij, proj_path, cij_path


def compose_output_dir(
    base_name: str,
    angle_count: int,
    suffix: str = "",
    output_root: str = "results",
    output_dir: str | None = None,
):
    if output_dir is not None:
        return output_dir

    if suffix:
        return os.path.join(output_root, base_name, f"{angle_count}angles_{suffix}")
    return os.path.join(output_root, base_name, f"{angle_count}angles")
