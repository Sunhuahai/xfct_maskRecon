"""
sm_cal.py - 系统矩阵计算 (Pythonic C-order Optimized Version)
使用蒙特卡洛方法计算XFCT系统矩阵
完全遵循 Python/NumPy 默认的行优先(C-order)内存布局:
- Image: (Z, Y, X)
- Detector: (Angle, Z, X)
"""

import os
import time

import numpy as np
from scipy.sparse import coo_matrix, save_npz, vstack


def format_pinhole_tag(d_pin):
    """
    生成与现有命名风格兼容的针孔标签。

    例如:
    - 0.5 -> p0d5
    - 1.25 -> p1d25
    """
    pin_str = f"{d_pin:g}".replace(".", "d")
    return f"p{pin_str}"


def format_image_grid_tag(l_im, n_imxy, n_imz):
    """
    生成图像体素与网格尺寸标签。

    例如:
    - l_im=0.5, n_imxy=120, n_imz=80 -> lim0d5_xy120_z80
    """
    voxel_str = f"{l_im:g}".replace(".", "d")
    return f"lim{voxel_str}_xy{n_imxy}_z{n_imz}"


# ================== 几何参数设置 (单位: mm) ==================

# 旋转角度
theta_deg = np.arange(0, 360, 8)
theta = theta_deg / 180 * np.pi
n_angles = len(theta)

# 探测器参数 (Pixel Size)
l_d = 0.25  # 探测器像素尺寸
n_dx = 160  # 探测器x方向像素数 (Width)
n_dz = 80  # 探测器z方向像素数 (Height)

# 针孔参数
d_pin = 1.0  # 针孔直径
L_dp = 27  # 探测器到针孔距离
L_cp = 50  # 针孔到旋转中心距离
delta_x = -2 * l_d  # x方向偏移

# 重建图像参数 (Voxel Size)
l_im = 0.5  # 图像体素尺寸(mm)
n_imxy = 60  # 图像xy方向体素数
n_imz = 40  # 图像z方向体素数

# 蒙特卡洛采样数
n_sample = 500

# 批处理大小s
batch_size = 20000

# ================== 1. 生成图像空间坐标 (Voxels) ==================
# Pythonic 顺序: Z (Axis 0), Y (Axis 1), X (Axis 2)
# 对应 shape (n_imz, n_imxy, n_imxy)

X_axis = np.arange(-l_im * (n_imxy - 1) / 2, l_im * (n_imxy - 1) / 2 + l_im / 2, l_im)
Y_axis = np.arange(-l_im * (n_imxy - 1) / 2, l_im * (n_imxy - 1) / 2 + l_im / 2, l_im)
Z_axis = np.arange(-l_im * (n_imz - 1) / 2, l_im * (n_imz - 1) / 2 + l_im / 2, l_im)

# indexing='ij' 确保 zi[0,0,0] 对应 Z_axis[0], Y_axis[0], X_axis[0]
zi, yi, xi = np.meshgrid(Z_axis, Y_axis, X_axis, indexing="ij")

# 扁平化 (C-order: X最快, Y次之, Z最慢)
xi = xi.ravel()
yi = yi.ravel()
zi = zi.ravel()
n_voxels = len(xi)

print(f"图像体素网格: {n_voxels} voxels")
print(f"网格形状 (Z, Y, X): ({n_imz}, {n_imxy}, {n_imxy})")

# ================== 2. 探测器参数预备 ==================
# 探测器总像素数
n_det_per_angle = n_dz * n_dx
yd_val = -L_dp

print(f"探测器像素: {n_det_per_angle} pixels/angle ({n_dz}x{n_dx})")

# ================== 3. 计算系统矩阵 ==================

cij_list = []
start_time = time.time()

for nt in range(n_angles):
    angle_start = time.time()
    print(f"计算角度 {nt + 1}/{n_angles} ({theta_deg[nt]:.1f}°)...", end="", flush=True)

    cos_t = np.cos(theta[nt])
    sin_t = np.sin(theta[nt])

    # 旋转坐标 (针对所有体素)
    # 逆时针旋转公式
    x_rot = xi * cos_t - yi * sin_t + delta_x
    y_rot = xi * sin_t + yi * cos_t + L_cp
    z_rot = zi

    # 准备构建稀疏矩阵的数据列表
    row_indices_all = []
    col_indices_all = []
    data_values_all = []

    # === 分批处理体素 (Vectorized Batch Processing) ===
    # 避免一次性生成 (n_voxels * n_sample) 的大数组导致内存溢出

    num_batches = int(np.ceil(n_voxels / batch_size))

    for b in range(num_batches):
        idx_start = b * batch_size
        idx_end = min((b + 1) * batch_size, n_voxels)
        current_batch_size = idx_end - idx_start

        # 获取当前批次的坐标
        xb = x_rot[idx_start:idx_end]
        yb = y_rot[idx_start:idx_end]
        zb = z_rot[idx_start:idx_end]

        # 1. 计算针孔投影几何 (针对当前批次)
        # 利用广播: (batch_size, 1)
        x_c = (xb / yb * yd_val)[:, np.newaxis]
        z_c = (zb / yb * yd_val)[:, np.newaxis]
        r_c = ((yb + L_dp) / yb * d_pin / 2)[:, np.newaxis]

        # 2. 蒙特卡洛采样 (batch_size, n_sample)
        a = np.random.rand(current_batch_size, n_sample)
        b_rand = np.random.rand(current_batch_size, n_sample)  # 重命名避免覆盖循环变量

        # 3. 计算探测器落点
        # r = sqrt(a) * R, theta = 2*pi*b
        r_sample = np.sqrt(a) * r_c
        theta_sample = b_rand * 2 * np.pi

        xs = r_sample * np.cos(theta_sample) + x_c
        zs = r_sample * np.sin(theta_sample) + z_c

        # 4. 映射到像素索引
        # x方向 (Cols): 0 to n_dx-1
        idx_x = np.floor((xs + l_d * (n_dx - 1) / 2) / l_d + 0.5).astype(np.int32)
        # z方向 (Rows): 0 to n_dz-1
        idx_z = np.floor((zs + l_d * (n_dz - 1) / 2) / l_d + 0.5).astype(np.int32)

        # 5. 过滤无效光子 (超出探测器范围)
        valid_mask = (idx_x >= 0) & (idx_x < n_dx) & (idx_z >= 0) & (idx_z < n_dz)

        # 6. 计算线性索引 (C-order: Row * Width + Col)
        # Row is Z, Col is X
        det_indices = idx_z * n_dx + idx_x

        # 7. 计算立体角权重 (Solid Angle Weight)
        # Weight只与体素位置有关，对同体素的所有样本相同
        dist_sq = xb**2 + yb**2 + zb**2
        # solid_angle = ((d_pin / 2)**2 * np.sqrt(zb**2 + yb**2) /
        #                (4 * dist_sq * np.sqrt(dist_sq)))
        solid_angle = (d_pin / 2) ** 2 * abs(yb) / (4 * dist_sq * np.sqrt(dist_sq))

        # 蒙特卡洛概率 = 1/N_sample * Solid_Angle
        # 如果一个体素有k个光子落入同一个像素，我们希望值是 k * (weight / N_sample)
        sample_weight = solid_angle / n_sample

        # 扩展权重以匹配样本形状
        weights_expanded = np.repeat(sample_weight[:, np.newaxis], n_sample, axis=1)

        # 8. 收集有效数据
        # 使用掩膜提取有效数据
        valid_det_idx = det_indices[valid_mask]
        valid_weights = weights_expanded[valid_mask]

        # 生成对应的体素索引 (列索引)
        # create voxel indices: [0, 0, ..., 1, 1, ...] + offset
        voxel_indices_local = np.arange(current_batch_size)[:, np.newaxis]
        voxel_indices_expanded = np.repeat(voxel_indices_local, n_sample, axis=1)
        valid_vox_idx = voxel_indices_expanded[valid_mask] + idx_start

        row_indices_all.append(valid_det_idx)
        col_indices_all.append(valid_vox_idx)
        data_values_all.append(valid_weights)

    # 合并当前角度的所有数据
    if len(data_values_all) > 0:
        rows = np.concatenate(row_indices_all)
        cols = np.concatenate(col_indices_all)
        data = np.concatenate(data_values_all)

        # 构建当前角度的稀疏矩阵
        cij_angle = coo_matrix((data, (rows, cols)), shape=(n_det_per_angle, n_voxels))
        cij_csr = cij_angle.tocsr()
        cij_csr.eliminate_zeros()

        cij_list.append(cij_csr)
    else:
        # 极端情况：该角度没有光子落入探测器
        from scipy.sparse import csr_matrix

        cij_list.append(csr_matrix((n_det_per_angle, n_voxels)))

    print(f" done. ({time.time() - angle_start:.2f}s)")

# 垂直拼接所有角度 (Angle 0, Angle 1, ...)
print("\n正在拼接系统矩阵...")
cij = vstack(cij_list)

print(f"系统矩阵计算完成!")
print(f"形状: {cij.shape}")
print(f"耗时: {time.time() - start_time:.1f}s")

# ================== 保存 ==================
output_dir = "data/system_matrix"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pinhole_tag = format_pinhole_tag(d_pin)
image_grid_tag = format_image_grid_tag(l_im, n_imxy, n_imz)
output_filename = (
    f"{output_dir}/cij_{n_angles}_3d_mod{L_dp}_{pinhole_tag}_{image_grid_tag}.npz"
)
save_npz(output_filename, cij)

print(f"系统矩阵已保存到: {output_filename}")
