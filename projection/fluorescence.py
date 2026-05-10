"""
fluorescence.py

用法示例:
  从仓库根目录运行:
    python3 read_make_proj/fluorescence.py \
      --input-dir <geant4_output_dir> \
      --phantom-map geant4_simulation/phantom/mouse/mousePhantom_100_100_125.txt \
      --output-dir read_make_proj/projections/fluorescence

  指定归一化参数并添加泊松噪声:
    python3 read_make_proj/fluorescence.py \
      --input-dir <geant4_output_dir> \
      --phantom-map geant4_simulation/phantom/mouse/mousePhantom_100_100_125.txt \
      --beam-current-ma 0.5 \
      --sampling-time-s 10 \
      --simulated-events 1000000 \
      --poisson-noise

功能: 处理GEANT4 XRF模拟数据，生成荧光断层图像

成像模式: X射线荧光 (XRF)

作者: wfang
日期: 原始代码 2024

==============================================================================
物理模型
==============================================================================
1. 立体角校正
   - 计算探测器对每个荧光点的立体角
   - 考虑针孔的几何限制

2. 物质衰减校正
   - 20种材料的线性衰减系数
   - 光子从荧光点到探测器的路径衰减

3. 几何校正
   - 物体旋转校正
   - 针孔投影

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

import numpy as np
import sys
import os
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from config import (L_D, N_DX, N_DZ, L_DP, L_CP, D_PIN, DIS_DEC,
                    XRAY_SPECTRUM_THETA_DEG, XRAY_SPECTRUM_PHI_DEG,
                    THETA_STEP, N_ANGLES, E_BINS, NUM_SEGMENTS,
                    ELECTRON_REST_ENERGY,
                    OUTPUT_DIR,
                    PHANTOM_SIZE, VOXEL_SIZE,
                    PHANTOM_OFFSET_X, PHANTOM_OFFSET_Y, PHANTOM_OFFSET_Z,
                    X_SHIFT_MM, Y_SIGN, Y_SHIFT_MM)
from utils import (init_detector_grid, load_xray_spectrum, rotate_coordinates,
                   project_to_pinhole, filter_boundaries, calculate_solid_angle,
                   calculate_attenuation, accumulate_projection,
                   load_default_attenuation_data, print_parameter_table,
                   add_common_path_arguments, add_normalization_arguments,
                   resolve_normalization_parameters, print_normalization_summary,
                   apply_poisson_noise, build_default_output_npy_name,
                   resolve_angle_indices, resolve_angle_event_path)

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'data' / 'projections' / 'single')
DEFAULT_SPECTRUM_PATH = PROJECT_ROOT / 'data' / 'projection_physics' / 'spec_150kVp.mat'


def parse_args():
    parser = argparse.ArgumentParser(description='XRF projection reconstruction.')
    add_common_path_arguments(parser, DEFAULT_OUTPUT_DIR)
    add_normalization_arguments(parser)
    parser.add_argument(
        '--output-npy-name',
        default=None,
        help='Output .npy filename. Defaults to {dataset}_{angle_count}_proj.npy.',
    )
    parser.add_argument(
        '--show-figures',
        action='store_true',
        help='Display figures interactively after saving them.',
    )
    return parser.parse_args()


def load_attenuation_data(phantom_map_path=None):
    """加载衰减系数和phantom数据。"""
    return load_default_attenuation_data(phantom_map_path=phantom_map_path)


def main():
    """XRF成像处理主函数。"""
    args = parse_args()

    print("=" * 60)
    print("XRF成像处理 - GEANT4模拟数据")
    print("=" * 60)
    print(f"Input dir   : {args.input_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Phantom map : {args.phantom_map if args.phantom_map else 'auto fallback'}")

    normalization = resolve_normalization_parameters(args)
    print_normalization_summary(
        normalization,
        poisson_noise=args.poisson_noise,
        poisson_seed=args.poisson_seed,
    )

    angle_indices = resolve_angle_indices(
        args.input_dir,
        modality_subdir='fluorescence',
    )
    if not angle_indices:
        raise FileNotFoundError(
            f"No fluorescence angle files found under {args.input_dir}"
        )
    angle_count = len(angle_indices)

    # 加载衰减数据
    print("\n加载衰减数据...")
    miu_data, phantom_data = load_attenuation_data(args.phantom_map)

    # 初始化投影矩阵 (n_angles x z x x)
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
        normalization['beam_current_ma'],
        normalization['sampling_time_s'],
        normalization['simulated_events'],
    )

    print_parameter_table(
        mode="XRF",
        n_dx=N_DX,
        n_dz=N_DZ,
        l_d=L_D,
        d_pin=D_PIN,
        dis_dec=DIS_DEC,
        l_dp=L_DP,
        l_cp=L_CP,
        theta_step=THETA_STEP,
        n_angles=angle_count,
    )

    # ========================================================================
    # 主循环 - 遍历各角度
    # ========================================================================
    for angle_slot, angle_value in enumerate(tqdm(angle_indices, desc="Processing angles", unit="angle")):

        # -------------------------------------------------------------------------
        # 加载模拟数据
        # -------------------------------------------------------------------------
        filename = resolve_angle_event_path(args.input_dir, 'fluorescence', angle_value)
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
        # 4) y_oc: 叠加旋转中心到针孔的几何偏移
        x0 = csct1[:, 0] + X_SHIFT_MM
        y0 = csct1[:, 1] * Y_SIGN + Y_SHIFT_MM + y_oc
        z0 = csct1[:, 2]  # z: 保持mm
        photon_energy = csct1[:, 3]  # 荧光光子能量 (keV)

        # 物体旋转
        x, y, z = rotate_coordinates(x0, y0, z0, theta, y_oc)

        # -------------------------------------------------------------------------
        # 针孔投影
        # -------------------------------------------------------------------------
        xs, zs, n_xs, n_zs, rand_u, rand_v, r_c = \
            project_to_pinhole(x, y, z, yd, L_D, D_PIN, N_DX, N_DZ)

        # -------------------------------------------------------------------------
        # 边界过滤
        # -------------------------------------------------------------------------
        result = filter_boundaries(N_DX, N_DZ,
                                   x, y, z, photon_energy, n_xs, n_zs,
                                   xs, zs, rand_u, rand_v, x0, y0, z0)
        x, y, z, photon_energy, n_xs, n_zs, xs, zs, rand_u, rand_v, x0, y0, z0 = result

        if len(x) == 0:
            continue

        # -------------------------------------------------------------------------
        # 计算立体角和权重
        # -------------------------------------------------------------------------
        # 立体角 (sr)
        solid_angle = calculate_solid_angle(x, y, z, D_PIN)

        # 权重 = 立体角 / (4*pi)
        weight_no_att = solid_angle / (4 * np.pi)

        # -------------------------------------------------------------------------
        # 衰减校正
        # -------------------------------------------------------------------------
        # 针孔入口坐标 (原始坐标系)
        x_pin0 = np.sqrt(rand_u) * D_PIN / 2 * np.cos(rand_v * 2 * np.pi)
        y_pin0 = np.zeros_like(x_pin0)
        z_pin0 = np.sqrt(rand_u) * D_PIN / 2 * np.sin(rand_v * 2 * np.pi)

        # 转换到旋转后的坐标系
        x_pin = x_pin0 * np.cos(-theta) - (y_pin0 - y_oc) * np.sin(-theta)
        y_pin = y_oc + (y_pin0 - y_oc) * np.cos(-theta) + x_pin0 * np.sin(-theta)
        z_pin = z_pin0

        # 计算衰减
        att = calculate_attenuation(x0, y0, z0, x_pin, y_pin, z_pin,
                                    photon_energy, NUM_SEGMENTS, y_oc,
                                    PHANTOM_SIZE, VOXEL_SIZE,
                                    PHANTOM_OFFSET_X, PHANTOM_OFFSET_Y, PHANTOM_OFFSET_Z,
                                    phantom_data, miu_data)

        # 最终权重 = 立体角权重 * 衰减校正 * 归一化因子
        weight_final = weight_no_att * np.exp(-att) * sig1

        # -------------------------------------------------------------------------
        # 累加到投影矩阵
        # -------------------------------------------------------------------------
        angle_idx = angle_slot
        nxrf = accumulate_projection(nxrf, weight_final, photon_energy,
                                     n_xs, n_zs, angle_idx, 1)

    # ========================================================================
    # 第三部分: 保存结果与显示
    # ========================================================================
    print("\n" + "=" * 60)
    print("保存结果...")

    nxrf = apply_poisson_noise(nxrf, enabled=args.poisson_noise, seed=args.poisson_seed)

    # 保存投影数据
    os.makedirs(args.output_dir, exist_ok=True)
    output_npy_name = args.output_npy_name or build_default_output_npy_name(
        args.input_dir,
        normalization['metadata'],
    )
    output_file = os.path.join(args.output_dir, output_npy_name)
    np.save(output_file, nxrf)
    print(f"结果已保存到 {output_file}")

    # 保存结果图到输出目录；默认不弹出图窗。
    print("\n生成结果图...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # View 1: ZX plane, summed over angles
    view_zx = np.sum(nxrf, axis=0)
    im1 = axes[0].imshow(view_zx, cmap='hot', aspect='auto')
    axes[0].set_title('ZX View (sum over angles)')
    axes[0].set_xlabel('Detector X Index')
    axes[0].set_ylabel('Detector Z Index')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # View 2: angle-x plane, summed over z
    view_ax = np.sum(nxrf, axis=1)
    im2 = axes[1].imshow(view_ax, cmap='hot', aspect='auto')
    axes[1].set_title('Angle-X View (sum over z)')
    axes[1].set_xlabel('Detector X Index')
    axes[1].set_ylabel('Angle Index')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # View 3: angle-z plane, summed over x
    view_az = np.sum(nxrf, axis=2)
    im3 = axes[2].imshow(view_az, cmap='hot', aspect='auto')
    axes[2].set_title('Angle-Z View (sum over x)')
    axes[2].set_xlabel('Detector Z Index')
    axes[2].set_ylabel('Angle Index')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    # Existing 1D summary plot
    axes[3].plot(nxrf[0, :, :].flatten())
    axes[3].set_title('First-Angle Projection')
    axes[3].set_xlabel('Detector Index (Flattened)')
    axes[3].set_ylabel('Intensity')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_file = os.path.join(args.output_dir, 'fluorescence_result.png')
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_file}")
    if args.show_figures:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
