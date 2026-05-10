"""
fluorescence_cmask.py

功能: 处理GEANT4 XRF模拟数据，生成荧光断层图像

成像模式: X射线荧光 (XRF, coded mask 9孔九宫格版)

作者: wfang
日期: 原始代码 2024

==============================================================================
物理模型
==============================================================================
1. 立体角校正
   - 计算探测器对每个荧光点的立体角
   - coded mask版本中，对9个圆孔分别计算并求和

2. 物质衰减校正
   - 20种材料的线性衰减系数
   - 光子从荧光点到探测器的路径衰减
   - coded mask版本中，衰减路径使用实际随机选中的孔内采样点

3. 几何校正
   - 物体旋转校正
   - coded mask投影（不是简单拼接9个单孔结果）

==============================================================================
数据格式
==============================================================================
GEANT4模拟输出 (csct1):
  csct1[:, 0] = x  (mm, 原始坐标系)
  csct1[:, 1] = y  (mm)
  csct1[:, 2] = z  (mm)
  csct1[:, 3] = Es (keV, 荧光光子能量)

==============================================================================
依赖文件
==============================================================================
必须在工作目录中包含以下文件：
  - spec_150kVp.mat           % 150kVp X射线能谱
  - miu3.mat                  % 20种材料的线性衰减系数 (能量 1-150 keV)

==============================================================================
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from config import (
    DIS_DEC,
    E_BINS,
    ELECTRON_REST_ENERGY,
    L_CP,
    L_D,
    L_DP,
    N_ANGLES,
    N_DX,
    N_DZ,
    NUM_SEGMENTS,
    OUTPUT_DIR,
    PHANTOM_OFFSET_X,
    PHANTOM_OFFSET_Y,
    PHANTOM_OFFSET_Z,
    PHANTOM_SIZE,
    THETA_STEP,
    VOXEL_SIZE,
    XRAY_SPECTRUM_PHI_DEG,
    XRAY_SPECTRUM_THETA_DEG,
    X_SHIFT_MM,
    Y_SHIFT_MM,
    Y_SIGN,
)
from utils import (
    accumulate_projection,
    add_common_path_arguments,
    add_normalization_arguments,
    apply_poisson_noise,
    build_default_output_npy_name,
    calculate_attenuation,
    calculate_solid_angle,
    filter_boundaries,
    infer_dataset_name_from_input_dir,
    init_detector_grid,
    load_default_attenuation_data,
    load_xray_spectrum,
    print_normalization_summary,
    print_parameter_table,
    project_to_pinhole,
    resolve_angle_event_path,
    resolve_angle_indices,
    resolve_normalization_parameters,
    rotate_coordinates,
)
from mask_geometry import build_mask_spec, save_mask_metadata


def format_mask_tag(mask_pitch, mask_hole_diameter):
    pitch_str = f"{mask_pitch:g}".replace(".", "d")
    diameter_str = f"{mask_hole_diameter:g}".replace(".", "d")
    return f"_cmask9_grid_p{pitch_str}_d{diameter_str}"


def build_centered_3x3_mask(pitch):
    """
    构建以 sm_cal.py 单孔为中心的 3x3 九宫格 coded mask 孔中心坐标。

    布局:
      - 中心1孔，位于 (0, 0)
      - x / z 两个方向各增加 +/- pitch 的孔
      - 共 9 孔，形成九宫格

    返回:
        hole_centers: shape = (9, 2)，每行为 (x, z) 坐标，单位 mm
    """
    grid_coords = np.array(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return pitch * grid_coords


MASK_PITCH = 6.0
MASK_HOLE_DIAMETER = 1.25  # 与 xfct_fastRecon/src/sm_cal.py 单孔定义对齐
MASK_TAG = format_mask_tag(MASK_PITCH, MASK_HOLE_DIAMETER)
MASK_HOLE_CENTERS = build_centered_3x3_mask(MASK_PITCH)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "projections" / "mask")
DEFAULT_SPECTRUM_PATH = PROJECT_ROOT / "data" / "projection_physics" / "spec_150kVp.mat"


def sample_points_in_coded_mask(
    rand_u, rand_v, hole_centers, hole_diameter, hole_indices=None
):
    """
    在coded mask选中的圆孔内进行均匀采样。

    参数:
        rand_u, rand_v:   圆孔内极坐标采样随机数，范围 [0, 1)
        hole_centers:     孔中心坐标数组，shape = (n_holes, 2)
        hole_diameter:    单孔直径 (mm)
        hole_indices:     每个事件选中的孔索引；若为None则等概率随机选择

    返回:
        x_mask, z_mask:   mask板面上的实际穿过点坐标 (mm)
        hole_indices:     实际使用的孔索引
    """
    rand_u = np.asarray(rand_u, dtype=np.float64)
    rand_v = np.asarray(rand_v, dtype=np.float64)
    hole_centers = np.asarray(hole_centers, dtype=np.float64)

    if hole_indices is None:
        hole_indices = np.random.randint(0, len(hole_centers), size=len(rand_u))
    else:
        hole_indices = np.asarray(hole_indices, dtype=np.int64)

    selected_centers = hole_centers[hole_indices]
    radius = np.sqrt(rand_u) * hole_diameter / 2.0
    angle = rand_v * 2.0 * np.pi

    x_mask = selected_centers[:, 0] + radius * np.cos(angle)
    z_mask = selected_centers[:, 1] + radius * np.sin(angle)
    return x_mask, z_mask, hole_indices


def project_to_coded_mask(
    x, y, z, yd, l_d, hole_centers, hole_diameter, n_dx, n_dz, rng=None
):
    """
    计算光子通过coded mask后的探测器位置。

    与单针孔版本的区别:
      1) 每个事件先随机选择一个圆孔
      2) 再在该圆孔内部均匀采样一个板面穿过点
      3) 基于“散射点 -> 实际采样孔点 -> 探测器”的射线几何计算落点

    参数:
        x, y, z:          荧光点坐标 (mm)
        yd:               探测器y位置 (mm)
        l_d:              探测器像素间距 (mm)
        hole_centers:     孔中心坐标，shape = (n_holes, 2)
        hole_diameter:    单孔直径 (mm)
        n_dx, n_dz:       探测器像素尺寸
        rng:              numpy随机数生成器

    返回:
        xs, zs:           探测器平面上的落点 (mm)
        n_xs, n_zs:       探测器像素索引
        rand_u, rand_v:   圆孔内归一化随机位置
        hole_indices:     选中的孔索引
        x_pin0, y_pin0, z_pin0:
                          mask板面采样点在原始坐标系中的位置 (mm)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_events = len(x)
    rand_u = rng.random(num_events)
    rand_v = rng.random(num_events)
    hole_indices = rng.integers(0, len(hole_centers), size=num_events)

    # 板面位于 y = 0；先确定每个事件穿过的实际孔内点
    x_pin0, z_pin0, hole_indices = sample_points_in_coded_mask(
        rand_u, rand_v, hole_centers, hole_diameter, hole_indices=hole_indices
    )
    y_pin0 = np.zeros_like(x_pin0)

    # 对应原单针孔公式:
    #   x_det = x * yd / y + ((y - yd) / y) * x_pin0
    #   z_det = z * yd / y + ((y - yd) / y) * z_pin0
    scale = (y - yd) / y
    xs = x / y * yd + scale * x_pin0
    zs = z / y * yd + scale * z_pin0

    # 转换为探测器像素索引
    n_xs = np.round((xs + l_d * (n_dx - 1) / 2) / l_d + 1).astype(int)
    n_zs = np.round((zs + l_d * (n_dz - 1) / 2) / l_d + 1).astype(int)

    return xs, zs, n_xs, n_zs, rand_u, rand_v, hole_indices, x_pin0, y_pin0, z_pin0


def calculate_coded_mask_solid_angle(x, y, z, hole_centers, hole_diameter):
    """
    计算coded mask对每个荧光点的近似立体角。

    第一版近似:
      - 把9个圆孔视为9个独立小孔
      - 每个孔沿用原单孔立体角近似公式
      - 最后对所有孔的贡献求和

    参数:
        x, y, z:          荧光点坐标 (mm)
        hole_centers:     孔中心坐标数组，shape = (n_holes, 2)
        hole_diameter:    单孔直径 (mm)

    返回:
        solid_angle:      立体角 (sr)
    """
    hole_centers = np.asarray(hole_centers, dtype=np.float64)
    hole_area = np.pi * (hole_diameter / 2.0) ** 2

    dx = x[:, None] - hole_centers[None, :, 0]
    dy = y[:, None]
    dz = z[:, None] - hole_centers[None, :, 1]
    distance_sq = dx**2 + dy**2 + dz**2

    solid_angle_each = hole_area * np.abs(dy) / np.power(distance_sq, 1.5)
    solid_angle = np.sum(solid_angle_each, axis=1)
    return solid_angle


def build_default_output_npy_name_cmask(
    input_dir,
    metadata=None,
    mask_tag=None,
    angle_count=None,
):
    """
    生成coded mask版本默认输出文件名。
    规则: {dataset}_{angle_count}_proj_cmask9_grid_*.npy
    """
    base_name = build_default_output_npy_name(input_dir, metadata=metadata)
    if angle_count is not None:
        dataset_name = infer_dataset_name_from_input_dir(input_dir)
        base_name = f"{dataset_name}_{int(angle_count)}_proj.npy"
    stem, ext = os.path.splitext(base_name)
    tag = MASK_TAG if mask_tag is None else str(mask_tag)
    if not tag.startswith("_"):
        tag = f"_{tag}"
    if stem.endswith(tag):
        return base_name
    return f"{stem}{tag}{ext}"


def _parse_angle_indices(value):
    if value is None or str(value).strip() == "":
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="XRF projection generation for a configurable multi-hole coded mask."
    )
    add_common_path_arguments(parser, DEFAULT_OUTPUT_DIR)
    add_normalization_arguments(parser)
    parser.add_argument(
        "--mask-layout",
        default="grid3x3",
        choices=["single", "grid3x3", "grid", "cross5", "random"],
        help="Mask layout. Default: grid3x3.",
    )
    parser.add_argument("--mask-rows", type=int, default=3, help="Grid rows.")
    parser.add_argument("--mask-cols", type=int, default=3, help="Grid columns.")
    parser.add_argument(
        "--mask-pitch-mm",
        type=float,
        default=MASK_PITCH,
        help=f"Center-to-center hole pitch in mm. Default: {MASK_PITCH}.",
    )
    parser.add_argument(
        "--mask-hole-diameter-mm",
        type=float,
        default=MASK_HOLE_DIAMETER,
        help=f"Hole diameter in mm. Default: {MASK_HOLE_DIAMETER}.",
    )
    parser.add_argument(
        "--mask-hole-count",
        type=int,
        default=9,
        help="Number of holes for random masks.",
    )
    parser.add_argument(
        "--mask-random-extent-mm",
        type=float,
        default=18.0,
        help="Square placement extent for random masks.",
    )
    parser.add_argument(
        "--mask-min-separation-mm",
        type=float,
        default=None,
        help="Minimum random-mask center separation. Defaults to hole diameter.",
    )
    parser.add_argument(
        "--mask-file",
        default=None,
        help="Optional custom mask center file with two columns: x,z in mm.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=20260509,
        help="Seed for random mask layout and per-event hole sampling.",
    )
    parser.add_argument(
        "--angle-indices",
        default=None,
        help="Optional comma-separated source angle indices, e.g. 0,9,18,27,36.",
    )
    parser.add_argument(
        "--angle-count",
        type=int,
        default=None,
        help="Uniformly subsample this many angles from all available input angles.",
    )
    parser.add_argument(
        "--output-npy-name",
        default=None,
        help="Output .npy filename. Defaults to {dataset}_{angle_count}_proj_{mask_tag}.npy.",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def load_attenuation_data(phantom_map_path=None):
    """加载衰减系数和phantom数据。"""
    return load_default_attenuation_data(phantom_map_path=phantom_map_path)


def main():
    """XRF成像处理主函数。"""
    args = parse_args()
    mask_spec = build_mask_spec(
        layout=args.mask_layout,
        pitch_mm=args.mask_pitch_mm,
        hole_diameter_mm=args.mask_hole_diameter_mm,
        rows=args.mask_rows,
        cols=args.mask_cols,
        hole_count=args.mask_hole_count,
        random_extent_mm=args.mask_random_extent_mm,
        min_separation_mm=args.mask_min_separation_mm,
        seed=args.mask_seed,
        mask_file=args.mask_file,
    )
    mask_tag = f"_cmask_{mask_spec.tag}"
    hole_centers = mask_spec.hole_centers
    hole_diameter = mask_spec.hole_diameter_mm

    print("=" * 60)
    print("XRF成像处理 - GEANT4模拟数据 (coded mask)")
    print("=" * 60)
    print(f"Input dir   : {args.input_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Phantom map : {args.phantom_map if args.phantom_map else 'auto fallback'}")
    print(f"Mask tag    : {mask_tag}")
    print(f"Mask holes  : {mask_spec.hole_count}")
    print(f"Hole dia.   : {hole_diameter} mm")
    print(f"Mask pitch  : {mask_spec.pitch_mm} mm")

    normalization = resolve_normalization_parameters(args)
    print_normalization_summary(
        normalization,
        poisson_noise=args.poisson_noise,
        poisson_seed=args.poisson_seed,
    )
    angle_indices = resolve_angle_indices(
        args.input_dir,
        modality_subdir="fluorescence",
    )
    requested_angle_indices = _parse_angle_indices(args.angle_indices)
    if requested_angle_indices is not None:
        available = set(angle_indices)
        missing = [idx for idx in requested_angle_indices if idx not in available]
        if missing:
            raise ValueError(f"Requested angle indices not found in input: {missing}")
        angle_indices = requested_angle_indices
    elif args.angle_count is not None:
        if args.angle_count <= 0:
            raise ValueError("--angle-count must be positive.")
        if args.angle_count > len(angle_indices):
            raise ValueError(
                f"--angle-count={args.angle_count} exceeds available angles {len(angle_indices)}."
            )
        selected = np.linspace(0, len(angle_indices) - 1, args.angle_count, dtype=int)
        angle_indices = [angle_indices[int(i)] for i in selected]
    if not angle_indices:
        raise FileNotFoundError(
            f"No fluorescence angle files found under {args.input_dir}"
        )
    angle_count = len(angle_indices)

    # 加载衰减数据
    print("\n加载衰减数据...")
    miu_data, phantom_data = load_attenuation_data(args.phantom_map)

    # 初始化投影矩阵 (angle x z x x)
    # 维度: [angle, z, x]
    nxrf = np.zeros((angle_count, N_DZ, N_DX))

    # 初始化探测器网格
    X_d, Z_d, xd, zd = init_detector_grid(L_D, N_DX, N_DZ)

    # 探测器位置参数
    yd = -L_DP  # 探测器y位置 (mm)
    y_oc = L_CP  # 旋转中心y偏移 (mm)

    # 加载X射线能谱并计算归一化因子
    sig1 = load_xray_spectrum(
        str(DEFAULT_SPECTRUM_PATH),
        XRAY_SPECTRUM_THETA_DEG,
        XRAY_SPECTRUM_PHI_DEG,
        normalization["beam_current_ma"],
        normalization["sampling_time_s"],
        normalization["simulated_events"],
    )

    print_parameter_table(
        mode=f"XRF{mask_tag}",
        n_dx=N_DX,
        n_dz=N_DZ,
        l_d=L_D,
        d_pin=hole_diameter,
        dis_dec=DIS_DEC,
        l_dp=L_DP,
        l_cp=L_CP,
        theta_step=THETA_STEP,
        n_angles=angle_count,
    )
    print("Coded mask hole centers (x, z) [mm]:")
    print(hole_centers)

    rng = np.random.default_rng(args.mask_seed)

    # ========================================================================
    # 主循环 - 遍历各角度
    # ========================================================================
    for angle_slot, angle_value in enumerate(
        tqdm(angle_indices, desc=f"Processing angles {MASK_TAG}", unit="angle")
    ):
        # -------------------------------------------------------------------------
        # 加载模拟数据
        # -------------------------------------------------------------------------
        filename = resolve_angle_event_path(args.input_dir, "fluorescence", angle_value)
        try:
            csct1 = np.atleast_2d(np.loadtxt(filename))
        except (OSError, ValueError):
            continue

        # 过滤无效事件 (能量为0)
        valid_mask = csct1[:, 3] != 0
        csct1 = csct1[valid_mask]

        if len(csct1) == 0:
            continue

        # -------------------------------------------------------------------------
        # 坐标变换 (单位转换 + 旋转)
        # -------------------------------------------------------------------------
        theta = angle_value * THETA_STEP * np.pi / 180  # 旋转角度 (弧度)

        # 单位体系: mm
        # GEANT4 -> 重建坐标系配准:
        # 1) 列映射: [x_raw, y_raw, z_raw] -> [x0, y0, z0]
        # 2) x平移: 使用 X_SHIFT_MM 对齐几何中心
        # 3) y轴向与平移: 使用 Y_SIGN / Y_SHIFT_MM 处理方向差异与中心面对齐
        # 4) y_oc: 叠加旋转中心到mask板面的几何偏移
        x0 = csct1[:, 0] + X_SHIFT_MM
        y0 = csct1[:, 1] * Y_SIGN + Y_SHIFT_MM + y_oc
        z0 = csct1[:, 2]  # z: 保持mm
        photon_energy = csct1[:, 3]  # 荧光光子能量 (keV)

        # 物体旋转
        x, y, z = rotate_coordinates(x0, y0, z0, theta, y_oc)

        # -------------------------------------------------------------------------
        # coded mask投影
        # -------------------------------------------------------------------------
        xs, zs, n_xs, n_zs, rand_u, rand_v, hole_indices, x_pin0, y_pin0, z_pin0 = (
            project_to_coded_mask(
                x,
                y,
                z,
                yd,
                L_D,
                hole_centers,
                hole_diameter,
                N_DX,
                N_DZ,
                rng=rng,
            )
        )

        # -------------------------------------------------------------------------
        # 边界过滤
        # 注意: coded mask相关随机量和采样孔点必须与事件同步过滤
        # -------------------------------------------------------------------------
        result = filter_boundaries(
            N_DX,
            N_DZ,
            x,
            y,
            z,
            photon_energy,
            n_xs,
            n_zs,
            xs,
            zs,
            rand_u,
            rand_v,
            hole_indices,
            x_pin0,
            y_pin0,
            z_pin0,
            x0,
            y0,
            z0,
        )
        (
            x,
            y,
            z,
            photon_energy,
            n_xs,
            n_zs,
            xs,
            zs,
            rand_u,
            rand_v,
            hole_indices,
            x_pin0,
            y_pin0,
            z_pin0,
            x0,
            y0,
            z0,
        ) = result

        if len(x) == 0:
            continue

        # -------------------------------------------------------------------------
        # 计算立体角和权重
        # -------------------------------------------------------------------------
        # coded mask立体角近似 = 9个圆孔立体角求和
        solid_angle = calculate_coded_mask_solid_angle(
            x, y, z, hole_centers, hole_diameter
        )

        # 权重 = 立体角 / (4*pi)
        weight_no_att = solid_angle / (4 * np.pi)

        # -------------------------------------------------------------------------
        # 衰减校正
        # -------------------------------------------------------------------------
        # mask板面采样点先在原始坐标系 y=0 平面定义，再转换到旋转后的坐标系
        x_pin = x_pin0 * np.cos(-theta) - (y_pin0 - y_oc) * np.sin(-theta)
        y_pin = y_oc + (y_pin0 - y_oc) * np.cos(-theta) + x_pin0 * np.sin(-theta)
        z_pin = z_pin0

        # 计算衰减
        att = calculate_attenuation(
            x0,
            y0,
            z0,
            x_pin,
            y_pin,
            z_pin,
            photon_energy,
            NUM_SEGMENTS,
            y_oc,
            PHANTOM_SIZE,
            VOXEL_SIZE,
            PHANTOM_OFFSET_X,
            PHANTOM_OFFSET_Y,
            PHANTOM_OFFSET_Z,
            phantom_data,
            miu_data,
        )

        # 最终权重 = 立体角权重 * 衰减校正 * 归一化因子
        weight_final = weight_no_att * np.exp(-att) * sig1

        # -------------------------------------------------------------------------
        # 累加到投影矩阵
        # -------------------------------------------------------------------------
        angle_idx = angle_slot
        nxrf = accumulate_projection(
            nxrf, weight_final, photon_energy, n_xs, n_zs, angle_idx, 1
        )

    # ========================================================================
    # 第三部分: 保存结果与显示
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"保存结果 {mask_tag} ...")

    nxrf = apply_poisson_noise(nxrf, enabled=args.poisson_noise, seed=args.poisson_seed)

    # 保存投影数据
    os.makedirs(args.output_dir, exist_ok=True)
    output_npy_name = args.output_npy_name or build_default_output_npy_name_cmask(
        args.input_dir,
        normalization["metadata"],
        mask_tag=mask_tag,
        angle_count=angle_count,
    )
    output_file = os.path.join(args.output_dir, output_npy_name)
    np.save(output_file, nxrf)
    print(f"结果已保存到 {output_file}")
    save_mask_metadata(
        Path(output_file).with_suffix(".mask.json"),
        mask_spec,
    )

    # 保存结果图到输出目录；默认不弹出图窗。
    print(f"\n生成结果图 {mask_tag} ...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # View 1: ZX plane, summed over angles
    view_zx = np.sum(nxrf, axis=0)
    im1 = axes[0].imshow(view_zx, cmap="hot", aspect="auto")
    axes[0].set_title("ZX View (sum over angles)")
    axes[0].set_xlabel("Detector X Index")
    axes[0].set_ylabel("Detector Z Index")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # View 2: angle-x plane, summed over z
    view_ax = np.sum(nxrf, axis=1)
    im2 = axes[1].imshow(view_ax, cmap="hot", aspect="auto")
    axes[1].set_title("Angle-X View (sum over z)")
    axes[1].set_xlabel("Detector X Index")
    axes[1].set_ylabel("Angle Index")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # View 3: angle-z plane, summed over x
    view_az = np.sum(nxrf, axis=2)
    im3 = axes[2].imshow(view_az, cmap="hot", aspect="auto")
    axes[2].set_title("Angle-Z View (sum over x)")
    axes[2].set_xlabel("Detector Z Index")
    axes[2].set_ylabel("Angle Index")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    # Existing 1D summary plot
    axes[3].plot(nxrf[0, :, :].flatten())
    axes[3].set_title("First-Angle Projection")
    axes[3].set_xlabel("Detector Index (Flattened)")
    axes[3].set_ylabel("Intensity")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_file = os.path.join(args.output_dir, f"fluorescence_result{mask_tag}.png")
    plt.savefig(fig_file, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_file}")
    if args.show_figures:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
