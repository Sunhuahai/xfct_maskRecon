"""
get_data.py - 荧光信号提取 (多项式插值版)
功能: 从能谱数据中分离散射背景，提取纯荧光信号
算法: 多项式插值法
修改: 保存为 [Angle, H, W] 格式，无标签直接存储为 .npy
"""

import os
import time

import numpy as np
from scipy.io import loadmat

# ================== 1. 参数设置 ==================

# --- 文件路径 ---
spectrum_file = "data/raw_spectra/real/spec_1.mat"
calibration_file = "data/calibration_data/gdf.mat"

# --- 保存路径 ---
output_file = "data/fluorescence_projections/normal/real/spec_angle_45_proj.npy"

# --- 算法详细参数 ---
param = {}

# [通用参数: 荧光区间 (仅用于插值法)]
param["bin_l"] = 156  # Python uses 0-indexing (MATLAB: 157)
param["bin_h"] = 164  # (MATLAB: 165)

# [多项式参数]
param["poly_order"] = 2  # 多项式阶数

# --- 结果提取参数 ---
bins_half_width = 2  # 目标能道左右各取几个点求和

# ================== 2. 加载数据 ==================
print("Loading data...")

# 加载能谱 [n_bins, H, W, n_angles]
if not os.path.exists(spectrum_file):
    raise FileNotFoundError(f"File not found: {spectrum_file}")

S = loadmat(spectrum_file)
sn = [k for k in S.keys() if not k.startswith("__")]
spectrum = S[sn[0]].astype(float)

C = loadmat(calibration_file)
cn = [k for k in C.keys() if not k.startswith("__")]
calibration_map = C[cn[0]]

n_bins, H, W, n_angles = spectrum.shape
print(f"Data size: [{n_bins}, {H}, {W}, {n_angles}]")

# ================== 3. 背景扣除与荧光提取 ==================


def estimate_scatter(scatter_bins, scatter_values, all_bins, raw_curve, param):
    """估计散射背景"""
    # 使用多项式拟合估计背景
    p = np.polyfit(scatter_bins, scatter_values, param["poly_order"])
    scatter_est = np.polyval(p, all_bins)
    return scatter_est


# 准备辅助变量
bin_idx = np.arange(n_bins)
scatter_mask = np.ones(n_bins, dtype=bool)
scatter_mask[param["bin_l"] : param["bin_h"] + 1] = False
scatter_bins = bin_idx[scatter_mask]

# 预分配
fluorescence = np.zeros((n_bins, H, W, n_angles))

print(f"Processing {n_angles} angles...")
timer_start = time.time()

for ang in range(n_angles):
    for h in range(H):
        for w in range(W):
            # 1. 获取原始曲线
            raw_curve = spectrum[:, h, w, ang]

            # 2. 准备多项式插值需要的子集
            scatter_values = raw_curve[scatter_mask]

            # 3. 估计背景
            bg_est = estimate_scatter(
                scatter_bins, scatter_values, bin_idx, raw_curve, param
            )

            # 4. 扣除背景得到荧光
            fluorescence[:, h, w, ang] = raw_curve - bg_est

    # 进度条
    if (ang + 1) % 5 == 0 or (ang + 1) == n_angles:
        elapsed = time.time() - timer_start
        print(f"  Angle {ang + 1}/{n_angles} completed | Elapsed: {elapsed:.1f}s")

# 全局非负约束
fluorescence = np.maximum(fluorescence, 0)

# ================== 4. 目标能道投影提取 ==================
# 此时 proj 的形状是 (H, W, n_angles)
proj = np.zeros((H, W, n_angles))
print(f"Extracting target energy bin intensity (width +/-{bins_half_width})...")

for h in range(H):
    for w in range(W):
        center_bin = int(np.round(calibration_map[h, w]))

        # 边界保护
        b_start = max(0, center_bin - bins_half_width)
        b_end = min(n_bins - 1, center_bin + bins_half_width)

        # 求和
        if b_start <= b_end:
            proj[h, w, :] = np.sum(fluorescence[b_start : b_end + 1, h, w, :], axis=0)

# ================== 5. 保存 ==================
# 旋转方向: (H, W, Angle) -> (Angle, H, W)
print(f"Transposing projection data to [Angle, H, W]...")
proj_rotated = np.transpose(proj, (2, 0, 1))

# 保存为.npy格式 (无标签，直接保存数组)
save_dir = os.path.dirname(output_file)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(output_file, proj_rotated)
print(f"Processing complete. Data shape: {proj_rotated.shape}")
print(f"Results saved to: {output_file}")
