# XFCT 多孔 Mask Reconstruction 调研 Brief

## 当前几何结构

本 baseline 沿用现有 XFCT 单针孔几何和重建网格。

- 探测器：`80 x 80` 像素，像素间距 `0.25 mm`。
- mask / pinhole 平面：定义在 `y = 0`。
- 探测器平面：位于 mask 后方 `30 mm`，即 `y = -30 mm`。
- 旋转中心到 mask 距离：`50 mm`。
- X 射线源到旋转中心距离：`300 mm`。
- 45 角度扫描步进：`8 deg`，角度索引 `0..44`。
- 5 角度默认从 45 角度均匀抽样：`[0, 9, 18, 27, 36]`，对应 `0, 72, 144, 216, 288 deg`。
- 15 角度默认抽样：`[0, 3, 6, ..., 42]`，对应每 `24 deg`。
- 重建体素网格：`40 x 60 x 60`。
- 系统矩阵输入行数使用 padded detector：原始投影 `80 x 80`，重建前在 x 方向两侧各 pad `40`，即每角度 `80 x 160`。

默认 mask 是 `3 x 3` 九孔阵列：

- 孔径：`1.25 mm`。
- 孔中心 pitch：`6 mm`。
- 孔中心位于 mask 平面的 `(x, z)` 坐标。
- 当前 copied mask 投影文件：`data/projections/mask/geometry_45_proj_cmask9_grid_p6_d1d25.npy`。

一个重要几何后果是，多孔投影不是简单的“9 个单孔投影相加后固定平移”。对发射点 `(x, y, z)` 和 mask 孔点 `(x_h, 0, z_h)`，探测器位置近似为：

```text
x_det = x / y * y_det + (y - y_det) / y * x_h
z_det = z / y * y_det + (y - y_det) / y * z_h
```

其中 `y_det = -30 mm`。因此每个孔带来的 detector shift 依赖发射点深度 `y`，是深度相关、视角相关、且会被衰减路径改变的混叠过程。

## Mask 的潜在优势

多孔 mask 的直接优势是进光量增加。若参考单孔直径同为 `1.25 mm`，九孔面积增益约为 `9x`；若参考当前部分投影代码中的 `0.5 mm` 单孔，则面积增益约为 `56.25x`。实际计数增益还会受 FOV 截断、衰减路径、立体角、孔间遮挡和 detector 边界影响。

可能收益：

- 同采集时间下显著提升光子计数，降低 Poisson 噪声。
- 同噪声水平下缩短采集时间，适合 few-view 或动态 XFCT。
- 若 mask 编码具备良好自相关性质，有机会把空间编码和角度稀疏性合并成压缩感知问题。
- 多孔布局可以作为硬件侧先验，为 5 角度极稀疏采样提供额外测量多样性。

## 主要难点

1. 混叠严重  
   `3 x 3`、`6 mm` pitch 在名义放大率 `(50 + 30) / 50 = 1.6` 下，对 detector 的孔间位移约为 `38.4` 像素。探测器宽度只有 `80` 像素，边缘孔会产生明显截断和强重叠。

2. forward model 不是空间不变卷积  
   固定 shift Wiener 解码只是失败 baseline。真实 shift 依赖深度 `y`、旋转角度、孔内采样点和衰减路径，所以当前 mask 基点改为显式多孔系统矩阵：

   ```text
   y ~ Poisson(A_mask f + background)
   A_mask = sum_h A_h
   ```

   其中 `A_h` 是每个孔对应的深度相关几何、立体角和衰减系统矩阵。这个矩阵已经生成并用于实验，但结果仍不理想，说明后续问题不只是“有没有多孔矩阵”，还包括几何/衰减一致性、矩阵条件数、mask 布局和重建正则化。

3. 系统矩阵规模和条件数  
   本次 5 角度、9 孔、500 samples/voxel/hole 的矩阵大小为 `2.0 GB`，shape 为 `(64000, 144000)`，`nnz=273,186,535`。单次 EM-TV 迭代约 `1.37 iter/s`。如果扩展到 15/45 角度或做 mask layout sweep，显式矩阵存储和 matvec 成本会很快成为瓶颈，需要研究 matrix-free projector、低秩/分块近似、或按孔分组的 GPU projector。

4. Poisson 统计与解码不匹配  
   先用线性 deconvolution 解码再 EM，会破坏原始 Poisson 噪声模型，并可能引入负值、相关噪声和边界振铃。更合理的 baseline 是在原始 mask measurement domain 中做 Poisson MBIR。

5. mask 优化目标不明确  
   需要同时考虑吞吐量、编码矩阵互相关、FOV 覆盖、深度可分辨性、制造约束、孔间最小间距、以及与 5 角度稀疏采样的联合条件数。

## 当前 Baseline 设计

入口脚本：

```bash
conda run -n xfct python experiments/run_effect_comparison.py
```

默认比较：

- `traditional_5`: 传统单孔 5 角度 EM-TV。
- `mask_5_naive`: 直接把 5 角度 mask 投影送入单孔系统矩阵。这个结果用于量化模型失配。
- `mask_5_wiener`: 用名义放大率构造 3x3 impulse kernel，做固定 shift Wiener 解码，再送入单孔系统矩阵。这个结果用于评估“简单解码 + 现有重建”是否有研究价值。
- `mask_5_model`: 用显式九孔 mask 系统矩阵直接重建 5 角度 mask 投影。这个结果是当前 mask 基点。
- `traditional_15`: 传统单孔 15 角度 EM-TV。
- `traditional_45`: 传统单孔 45 角度 EM-TV。

输出：

- `results/effect_comparison/effect_comparison.csv`
- `results/effect_comparison/effect_comparison.md`
- `results/effect_comparison/reconstruction_panel.png`
- 每个 run 的 reconstruction、curve、ROI detection-limit 图。

当前 35 iteration EM-TV baseline 结果已经生成，位置：

```text
results/effect_comparison/effect_comparison.csv
results/effect_comparison/reconstruction_panel.png
```

结果摘要：

| run | angles | DL (mg/ml) | R2 | projection counts |
| --- | ---: | ---: | ---: | ---: |
| traditional_5 | 5 | 0.2625 | 0.9600 | 2.1981e5 |
| mask_5_naive | 5 | 10.6846 | 0.4798 | 1.0913e7 |
| mask_5_wiener | 5 | -29.9987 | 0.5131 | 1.2523e7 |
| mask_5_model | 5 | 9.7767 | 0.4530 | 1.0913e7 |
| traditional_15 | 15 | 0.0633 | 0.9943 | 6.5731e5 |
| traditional_45 | 45 | 0.0039 | 0.9993 | 1.9684e6 |

解释：

- `mask_5_naive` 的总计数约为传统 5 角度的 `49.6x`，但 DL 明显恶化，说明不能把多孔数据直接送入单孔系统矩阵。
- `mask_5_wiener` 出现负 DL，表示 ROI CNR 线性拟合斜率/截距已经失真，不能解释为真实检测限；固定 shift 2D 去卷积不适合这个深度相关 mask forward model。
- `mask_5_model` 使用匹配的九孔系统矩阵后，DL 从 naive 的 `10.6846` 改到 `9.7767 mg/ml`，但仍远差于 `traditional_5` 的 `0.2625 mg/ml`。因此当前最重要结论是：多孔矩阵本身已经可以生成，但当前几何/数据/重建组合没有把高计数优势转化为重建优势。

## Mask 系统矩阵基点

之前的系统矩阵 Monte Carlo 生成程序本质上可以扩展到多孔 mask。单孔程序对每个体素采样针孔圆孔在 detector 上的 footprint，并按立体角权重写入 detector-pixel-by-voxel 稀疏矩阵。多孔版本对每个孔中心重复同样的采样，并把各孔贡献累加到同一 detector row。这个逻辑已经整理成：

```text
scripts/build_mask_system_matrix.py
```

默认 5 角度九孔矩阵生成命令：

```bash
conda run -n xfct python scripts/build_mask_system_matrix.py \
  --angle-indices 0,9,18,27,36 \
  --mask-layout grid3x3 \
  --mask-pitch-mm 6 \
  --mask-hole-diameter-mm 1.25
```

该脚本默认输出 PMMA 衰减修正矩阵，文件名为：

```text
data/system_matrix/cij_5_3d_mod30_cmask_grid3x3_n9_p6_d1d25_lim0d5_xy60_z40_att_pmma.npz
```

本次已生成的矩阵：

```text
shape: (64000, 144000)
nnz: 273,186,535
file size: 2.0 GB
generation time: 218 s
```

直接模型匹配 baseline 命令：

```bash
conda run -n xfct python experiments/run_effect_comparison.py \
  --runs traditional_5,mask_5_naive,mask_5_wiener,mask_5_model,traditional_15,traditional_45
```

可选参数：

- `--quick`: 3 iteration 冒烟测试。
- `--methods em_tv,pseudo_mbir`: 同时跑现有 pseudo-MBIR 路径。
- `--match-total-counts`: 将所有输入投影缩放到传统 5 角度总计数，用于隔离“编码混叠”与“计数增益”。
- `--poisson-seed <seed>`: 对输入投影做 Poisson sampling，用于低剂量/计数统计测试。