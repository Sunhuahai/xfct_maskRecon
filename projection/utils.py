"""
utils.py

公共函数库

功能: 封装XFCT/X射线散射成像处理中的通用函数

==============================================================================
函数列表
==============================================================================
  init_detector_grid()      - 初始化探测器网格
  load_xray_spectrum()      - 加载X射线能谱并计算归一化因子
  rotate_coordinates()      - 坐标旋转变换（物体旋转）
  project_to_pinhole()      - 针孔投影计算
  filter_boundaries()       - 边界过滤（移除探测器范围外的事件）
  calculate_solid_angle()   - 计算立体角
  calculate_kn_cross_section() - 计算Klein-Nishina散射截面
  calculate_scatter_angle() - 计算散射角
  calculate_attenuation()   - 计算物质衰减
  accumulate_projection()   - 累加到投影矩阵

==============================================================================
"""

import numpy as np
import json
import re
from pathlib import Path
import h5py
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BEAM_CURRENT_MA = 3.0
DEFAULT_SAMPLING_TIME_S = 60.0
DEFAULT_SIMULATED_EVENTS = 200000000

DEFAULT_MIU_CANDIDATES = [
    str(PROJECT_ROOT / 'data' / 'projection_physics' / 'miu.npz'),
    str(PROJECT_ROOT / 'data' / 'projection_physics' / 'miu3.mat'),
]
DEFAULT_GEANT4_LABEL_MAPPING_CANDIDATES = [
    str(PROJECT_ROOT / 'data' / 'projection_physics' / 'geant4' / 'PhantomLabelMapping.cc'),
]
GEANT4_LABEL_PATTERN = re.compile(
    r'AddLabelMapping\(\s*catalog,\s*name_to_index,\s*(\d+),\s*"([^"]+)"\s*\)',
    re.MULTILINE,
)

MODALITY_DIR_NAMES = {
    'fluorescence',
    'single_Compton_scattering',
    'multi_Compton_scattering',
}


def load_mat_variable(mat_path, var_name):
    """
    读取MAT文件变量，兼容v7/v7.3。
    """
    try:
        data = loadmat(mat_path)
        if var_name not in data:
            raise KeyError(f"{var_name} not found in {mat_path}")
        return np.array(data[var_name])
    except NotImplementedError:
        with h5py.File(mat_path, 'r') as f:
            if var_name not in f:
                raise KeyError(f"{var_name} not found in {mat_path}")
            arr = np.array(f[var_name])
            if arr.ndim > 1:
                arr = np.transpose(arr, axes=tuple(range(arr.ndim - 1, -1, -1)))
            return arr


def load_array_with_fallback(candidates, var_name=None, txt_dtype=np.float64):
    """
    按候选路径优先级读取数组。
    支持: .npy / .npz / .txt / .mat
    """
    checked = []
    for path_str in candidates:
        path = Path(path_str)
        checked.append(str(path))
        if not path.exists():
            continue

        suffix = path.suffix.lower()
        if suffix == '.npy':
            return np.array(np.load(path, allow_pickle=False))

        if suffix == '.npz':
            with np.load(path, allow_pickle=False) as data:
                if var_name and var_name in data:
                    return np.array(data[var_name])
                if len(data.files) == 1:
                    return np.array(data[data.files[0]])
                available = ", ".join(data.files)
                raise KeyError(
                    f"{path}: npz contains multiple arrays ({available}), "
                    f"but var_name='{var_name}' not found."
                )

        if suffix == '.txt':
            return np.array(np.loadtxt(path, dtype=txt_dtype))

        if suffix == '.mat':
            if not var_name:
                raise ValueError(f"{path}: var_name is required for .mat files.")
            return np.array(load_mat_variable(str(path), var_name))

    raise FileNotFoundError(
        "No candidate data file found. Checked: " + ", ".join(checked)
    )


def resolve_existing_candidate(candidates):
    checked = []
    for path_str in candidates:
        path = Path(path_str)
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(
        "No candidate data file found. Checked: " + ", ".join(checked)
    )


def _prepend_if_given(path, candidates):
    if path:
        path_str = str(path)
        return [path_str] + [c for c in candidates if c != path_str]
    return list(candidates)


def add_common_path_arguments(parser, default_output_dir):
    """
    为脚本添加统一的路径参数。
    """
    parser.add_argument(
        '--input-dir',
        required=True,
        help=(
            'Input directory of angle txt files. Prefers the current Geant4 layout '
            'angle_{i}/{mode}/{i}.txt, and also supports legacy flat/mode-first layouts.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=default_output_dir,
        help=f'Output directory (default: {default_output_dir}).',
    )
    parser.add_argument(
        '--phantom-map',
        required=True,
        help='Voxel label map or attenuation-id map file (.npy/.npz/.txt/.mat).',
    )
    return parser


def add_normalization_arguments(parser):
    """
    为脚本添加统一的采样时间/归一化参数。
    """
    parser.add_argument(
        '--metadata',
        default=None,
        help='Run metadata JSON path. If omitted, auto-detect from input dir or its parent directory.',
    )
    parser.add_argument(
        '--beam-current-ma',
        type=float,
        default=None,
        help='Beam current in mA. Overrides metadata/default value.',
    )
    parser.add_argument(
        '--sampling-time-s',
        type=float,
        default=None,
        help='Sampling time in seconds. Overrides metadata/default value.',
    )
    parser.add_argument(
        '--simulated-events',
        type=int,
        default=None,
        help='Primary event count used in Geant4. Overrides metadata/default value.',
    )
    parser.add_argument(
        '--poisson-noise',
        action='store_true',
        help='Apply Poisson sampling to the reconstructed expected counts.',
    )
    parser.add_argument(
        '--poisson-seed',
        type=int,
        default=1234,
        help='Random seed used when --poisson-noise is enabled (default: 1234).',
    )
    return parser


def _load_json_if_exists(path):
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    with open(path_obj, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_run_metadata(input_dir, explicit_metadata_path=None):
    """
    查找 GEANT4 运行 metadata。
    """
    candidates = []
    if explicit_metadata_path:
        candidates.append(Path(explicit_metadata_path))

    input_path = Path(input_dir)
    candidates.append(input_path / 'run_metadata.json')
    if input_path.parent != input_path:
        candidates.append(input_path.parent / 'run_metadata.json')

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        metadata = _load_json_if_exists(candidate)
        if metadata is not None:
            return str(candidate), metadata

    return None, {}


def resolve_normalization_parameters(args):
    """
    综合 CLI、metadata 和默认值，得到统一的归一化参数。
    """
    metadata_path, metadata = find_run_metadata(args.input_dir, getattr(args, 'metadata', None))

    beam_current_ma = getattr(args, 'beam_current_ma', None)
    if beam_current_ma is None:
        beam_current_ma = metadata.get('beam_current_ma', DEFAULT_BEAM_CURRENT_MA)

    sampling_time_s = getattr(args, 'sampling_time_s', None)
    if sampling_time_s is None:
        sampling_time_s = metadata.get('sampling_time_s', DEFAULT_SAMPLING_TIME_S)

    simulated_events = getattr(args, 'simulated_events', None)
    if simulated_events is None:
        simulated_events = metadata.get('events', DEFAULT_SIMULATED_EVENTS)

    beam_current_ma = float(beam_current_ma)
    sampling_time_s = float(sampling_time_s)
    simulated_events = int(simulated_events)

    if beam_current_ma <= 0:
        raise ValueError(f"beam_current_ma must be > 0, got {beam_current_ma}")
    if sampling_time_s <= 0:
        raise ValueError(f"sampling_time_s must be > 0, got {sampling_time_s}")
    if simulated_events <= 0:
        raise ValueError(f"simulated_events must be > 0, got {simulated_events}")

    return {
        'beam_current_ma': beam_current_ma,
        'sampling_time_s': sampling_time_s,
        'simulated_events': simulated_events,
        'metadata_path': metadata_path,
        'metadata': metadata,
    }


def print_normalization_summary(normalization, poisson_noise=False, poisson_seed=1234):
    """
    打印采样时间相关归一化参数。
    """
    print("Normalization:")
    print(f"  beam current   : {normalization['beam_current_ma']} mA")
    print(f"  sampling time  : {normalization['sampling_time_s']} s")
    print(f"  simulated events: {normalization['simulated_events']}")
    print(f"  metadata path  : {normalization['metadata_path'] or 'not found (using CLI/defaults)'}")
    print(f"  poisson noise  : {'enabled' if poisson_noise else 'disabled'}")
    if poisson_noise:
        print(f"  poisson seed   : {poisson_seed}")


def apply_poisson_noise(projection, enabled=False, seed=1234):
    """
    对期望计数应用泊松采样。
    """
    if not enabled:
        return projection

    rng = np.random.default_rng(seed)
    lam = np.clip(projection, 0.0, None)
    return rng.poisson(lam).astype(np.float64, copy=False)


def format_sampling_time_label(sampling_time_s):
    """
    把采样时间格式化为适合文件名的标签。
    """
    value = float(sampling_time_s)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    text = f"{value:g}"
    return text.replace('.', 'p')


def infer_dataset_name_from_input_dir(input_dir):
    """
    从输入目录推导数据集名称。
    如果输入目录是具体模态子目录，则取其父目录名。
    """
    input_path = Path(input_dir)
    base_name = input_path.name
    if base_name in MODALITY_DIR_NAMES and input_path.parent != input_path:
        return input_path.parent.name
    return base_name


def infer_angle_count(input_dir, metadata=None):
    """
    从输入目录实际存在的角度文件/目录推导角度数。
    不读取 metadata 里的 angle_indices。
    """
    return len(resolve_angle_indices(input_dir))


def _normalize_angle_index_list(values):
    result = []
    for value in values:
        result.append(int(value))
    return result


def _collect_numeric_txt_indices(directory):
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []

    indices = []
    for child in path.iterdir():
        if not child.is_file() or child.suffix.lower() != '.txt':
            continue
        if re.fullmatch(r'\d+', child.stem):
            indices.append(int(child.stem))

    return sorted(set(indices))


def resolve_angle_indices(input_dir, modality_subdir=None, metadata=None):
    """
    解析当前输入目录下可用的角度索引。

    优先级：
    1. 当前 Geant4 输出目录：angle_{i}/{mode}/{i}.txt
    2. mode-first 目录：{mode}/{i}.txt
    3. 平铺目录：{i}.txt

    不读取 metadata 里的 angle_indices。
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return []

    angle_dir_indices = []
    for child in input_path.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r'angle_(\d+)', child.name)
        if match:
            angle_dir_indices.append(int(match.group(1)))
    if angle_dir_indices:
        return sorted(set(angle_dir_indices))

    if modality_subdir:
        mode_indices = _collect_numeric_txt_indices(input_path / modality_subdir)
        if mode_indices:
            return mode_indices
    else:
        mode_indices = []
        for mode_name in MODALITY_DIR_NAMES:
            mode_indices.extend(_collect_numeric_txt_indices(input_path / mode_name))
        if mode_indices:
            return sorted(set(mode_indices))

    return _collect_numeric_txt_indices(input_path)


def resolve_angle_event_path(input_root, modality_subdir, angle_idx):
    """
    解析单个角度事件文件路径。

    优先匹配当前 Geant4 输出目录：
      angle_{i}/{mode}/{i}.txt
    同时兼容旧布局：
      {mode}/{i}.txt
      {i}.txt
      angle_{i}/{i}.txt
    """
    angle_dir = Path(input_root) / f'angle_{angle_idx}'
    candidates = [
        angle_dir / modality_subdir / f'{angle_idx}.txt',
        Path(input_root) / modality_subdir / f'{angle_idx}.txt',
        Path(input_root) / f'{angle_idx}.txt',
        angle_dir / f'{angle_idx}.txt',
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def build_default_output_npy_name(input_dir, metadata=None):
    """
    生成默认输出文件名：{dataset}_{angle_count}_proj.npy
    例如：geometry_5_proj.npy
    """
    dataset_name = infer_dataset_name_from_input_dir(input_dir)
    angle_count = infer_angle_count(input_dir, metadata=metadata)
    return f"{dataset_name}_{angle_count}_proj.npy"


def parse_geant4_label_mapping(source_path):
    text = Path(source_path).read_text(encoding='utf-8')
    records = GEANT4_LABEL_PATTERN.findall(text)
    if not records:
        raise ValueError(f"No label mappings found in {source_path}")

    label_to_material_name = {}
    for label_text, material_name in records:
        label = int(label_text)
        existing = label_to_material_name.get(label)
        if existing is not None and existing != material_name:
            raise ValueError(
                f"Conflicting material name for label {label}: {existing} vs {material_name}"
            )
        label_to_material_name[label] = material_name
    return label_to_material_name


def load_named_miu_registry(miu_source_path):
    suffix = miu_source_path.suffix.lower()
    if suffix == '.npz':
        with np.load(miu_source_path, allow_pickle=False) as data:
            if 'material_names' not in data:
                return None
            material_names = np.array(data['material_names']).astype(str).flatten()
    elif suffix == '.mat':
        return None
    else:
        return None

    material_to_row_index = {}
    for row_index, material_name in enumerate(material_names):
        if not material_name:
            continue
        material_to_row_index[str(material_name)] = int(row_index)
    return material_to_row_index


def map_phantom_labels_to_attenuation_rows(phantom_data, label_to_material_name, material_to_row_index):
    phantom_data = np.asarray(phantom_data, dtype=np.int32).flatten()
    if np.any(phantom_data < 0):
        raise ValueError("phantom_map contains negative voxel labels.")

    if phantom_data.size == 0:
        return phantom_data

    unique_labels = np.unique(phantom_data)
    missing_labels = [label for label in unique_labels if int(label) not in label_to_material_name]
    if missing_labels:
        bad_text = ", ".join(str(int(v)) for v in missing_labels[:16])
        if len(missing_labels) > 16:
            bad_text += ", ..."
        raise ValueError(f"phantom_map contains labels without Geant4 material mapping: {bad_text}")

    missing_materials = []
    label_to_row_index = {}
    for label in unique_labels:
        material_name = label_to_material_name[int(label)]
        if material_name not in material_to_row_index:
            missing_materials.append((int(label), material_name))
            continue
        label_to_row_index[int(label)] = int(material_to_row_index[material_name])

    if missing_materials:
        bad_text = ", ".join(f"{label}:{material_name}" for label, material_name in missing_materials[:16])
        if len(missing_materials) > 16:
            bad_text += ", ..."
        raise ValueError(
            "phantom_map contains labels whose materials are missing from attenuation registry: "
            f"{bad_text}"
        )

    remapped = np.empty_like(phantom_data)
    for label, row_index in label_to_row_index.items():
        remapped[phantom_data == label] = row_index
    return remapped


def load_default_attenuation_data(phantom_map_path=None, miu_path=None):
    """
    加载衰减系数和 phantom label map，并按 Geant4 label 映射转换到材料谱行号。
    """
    if not phantom_map_path:
        raise ValueError("phantom_map_path is required; pass --phantom-map explicitly.")

    miu_candidates = _prepend_if_given(miu_path, DEFAULT_MIU_CANDIDATES)
    phantom_candidates = [str(phantom_map_path)]
    miu_source_path = resolve_existing_candidate(miu_candidates)

    miu_data = load_array_with_fallback([str(miu_source_path)], var_name='miu')
    phantom = load_array_with_fallback(
        phantom_candidates, var_name='phantom', txt_dtype=np.int32
    )
    phantom_data = np.array(phantom).flatten(order='F')

    material_to_row_index = load_named_miu_registry(miu_source_path)
    if material_to_row_index is None:
        raise ValueError(
            f"{miu_source_path}: material_names are required. "
            "Use read_make_proj/build_named_attenuation_npz.py to generate miu_named.npz."
        )
    label_mapping_path = resolve_existing_candidate(DEFAULT_GEANT4_LABEL_MAPPING_CANDIDATES)
    label_to_material_name = parse_geant4_label_mapping(label_mapping_path)
    phantom_data = map_phantom_labels_to_attenuation_rows(
        phantom_data,
        label_to_material_name,
        material_to_row_index,
    )
    return miu_data, phantom_data


def print_parameter_table(mode, n_dx, n_dz, l_d, d_pin, dis_dec, l_dp, l_cp,
                          theta_step, n_angles, e_bins=None):
    """
    以ASCII表格输出运行参数。
    """
    rows = [
        ("Mode", mode),
        ("Detector size", f"{n_dx} x {n_dz} pixels"),
        ("Pixel pitch", f"{l_d} mm"),
        ("Pinhole diameter", f"{d_pin} mm"),
        ("Source-to-object distance", f"{dis_dec} mm"),
        ("L_dp (Detector to pinhole)", f"{l_dp} mm"),
        ("L_cp (Center to pinhole)", f"{l_cp} mm"),
        ("Rotation", f"{theta_step} deg step, {n_angles} angles"),
    ]
    if e_bins is not None:
        rows.append(("Energy bins", str(e_bins)))

    key_w = max(len(k) for k, _ in rows)
    val_w = max(len(v) for _, v in rows)
    border = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print(border)
    for k, v in rows:
        print(f"| {k.ljust(key_w)} | {v.ljust(val_w)} |")
    print(border)


def init_detector_grid(l_d, n_dx, n_dz):
    """
    初始化探测器网格坐标

    参数:
        l_d:     探测器像素间距 (mm)
        n_dx:    x方向像素数
        n_dz:    z方向像素数

    返回:
        X_d:     探测器x坐标 (n_dx x 1)
        Z_d:     探测器z坐标 (n_dz x 1)
        xd:      展平后的x网格坐标 (n_dx*n_dz x 1)
        zd:      展平后的z网格坐标 (n_dx*n_dz x 1)
    """
    # x方向探测器中心坐标
    X_d = np.linspace(-l_d * (n_dx - 1) / 2, l_d * (n_dx - 1) / 2, n_dx).reshape(-1, 1)
    # z方向探测器中心坐标
    Z_d = np.linspace(-l_d * (n_dz - 1) / 2, l_d * (n_dz - 1) / 2, n_dz).reshape(-1, 1)

    # 生成网格并展平
    xd_grid, zd_grid = np.meshgrid(X_d.flatten(), Z_d.flatten())
    xd = xd_grid.flatten()
    zd = zd_grid.flatten()

    return X_d, Z_d, xd, zd


def load_xray_spectrum(spec_file, theta11, phi11, current, t, simulated_events=DEFAULT_SIMULATED_EVENTS):
    """
    加载X射线能谱并计算归一化因子

    参数:
        spec_file: 能谱文件名 (默认: 'spec_150kVp.mat')
        theta11:   角度参数 (度)
        phi11:     角度参数 (度)
        current:   束流 (mA)
        t:         曝光时间 (s)

    返回:
        sig1:      归一化因子
    """
    # 加载能谱数据
    N = load_mat_variable(spec_file, 'N').flatten()

    # 计算有效面积 (cm^2)
    area = np.tan(np.deg2rad(theta11)) * 2 * np.tan(np.deg2rad(phi11)) * 2 * 100**2

    # 计算入射光子总数
    N1 = N * area * current * t

    # 归一化因子 = 实验入射光子数 / Geant4 模拟初级历史数
    sig1 = np.sum(N1) / simulated_events

    return sig1


def rotate_coordinates(x0, y0, z0, theta, y_oc):
    """
    坐标旋转变换（物体绕原点旋转）

    参数:
        x0, y0, z0: 原始坐标 (mm)
        theta:      旋转角度 (弧度)
        y_oc:       旋转中心y偏移 (mm)

    返回:
        x, y, z:    旋转后坐标 (mm)

    说明:
        绕z轴旋转（实际是物体旋转，坐标系固定）
        旋转公式:
            x' = x*cos(theta) - (y-y_oc)*sin(theta)
            y' = y_oc + (y-y_oc)*cos(theta) + x*sin(theta)
            z' = z
    """
    x = x0 * np.cos(theta) - (y0 - y_oc) * np.sin(theta)
    y = y_oc + (y0 - y_oc) * np.cos(theta) + x0 * np.sin(theta)
    z = z0

    return x, y, z


def project_to_pinhole(x, y, z, yd, l_d, d_pin, n_dx, n_dz):
    """
    计算光子通过针孔后的探测器位置

    参数:
        x, y, z:   散射点坐标 (mm)
        yd:        探测器y位置 (mm)
        l_d:       探测器像素间距 (mm)
        d_pin:     针孔直径 (mm)
        n_dx:      探测器x尺寸
        n_dz:      探测器z尺寸

    返回:
        xs, zs:    探测器平面上的落点 (mm)
        n_xs, n_zs: 探测器像素索引
        rand_u, rand_v: 针孔内归一化随机位置
        r_c:       针孔内最大半径
    """
    # 计算散射点到针孔的投影
    # 针孔位于 (x_c, yd, z_c)，其中:
    #   x_c = x/y * yd
    #   z_c = z/y * yd
    x_c = x / y * yd
    z_c = z / y * yd

    # 针孔内可接收区域半径（随距离变化）
    r_c = (y - yd) / y * d_pin / 2

    # 生成针孔内的随机位置（均匀分布）
    rand_u = np.random.rand(len(x_c))  # r^2 归一化因子
    rand_v = np.random.rand(len(x_c))  # 角度因子

    # 计算针孔内实际落点（极坐标采样）
    xs = np.sqrt(rand_u) * r_c * np.cos(rand_v * 2 * np.pi) + x_c
    zs = np.sqrt(rand_u) * r_c * np.sin(rand_v * 2 * np.pi) + z_c

    # 转换为探测器像素索引
    n_xs = np.round((xs + l_d * (n_dx - 1) / 2) / l_d + 1).astype(int)
    n_zs = np.round((zs + l_d * (n_dz - 1) / 2) / l_d + 1).astype(int)

    return xs, zs, n_xs, n_zs, rand_u, rand_v, r_c


def filter_boundaries(n_dx, n_dz, *arrays):
    """
    边界过滤（移除探测器范围外的事件）

    参数:
        n_dx, n_dz: 探测器尺寸
        *arrays:    所有要过滤的数组（第5和第6个应该是n_xs, n_zs）

    返回:
        过滤后的数组
    """
    if len(arrays) < 6:
        return arrays

    n_xs = arrays[4]
    n_zs = arrays[5]

    # 边界条件: 1 <= index <= n_dx/n_dz
    is_valid = (n_xs > 0) & (n_xs < n_dx + 1) & (n_zs > 0) & (n_zs < n_dz + 1)

    # 处理is_valid是布尔数组的情况
    filtered_arrays = []
    for arr in arrays:
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and len(arr) == len(is_valid):
            filtered_arrays.append(arr[is_valid])
        else:
            filtered_arrays.append(arr)

    return tuple(filtered_arrays)


def calculate_solid_angle(x, y, z, d_pin):
    """
    计算探测器对每个散射点的立体角

    参数:
        x, y, z:   散射点坐标 (mm)
        d_pin:     针孔直径 (mm)

    返回:
        solid_angle: 立体角 (sr)
    """
    # 立体角 = 针孔面积 / 距离^2 * 投影因子
    # 投影因子考虑散射点到针孔的入射角
    distance_sq = x**2 + y**2 + z**2
    solid_angle = 1 / distance_sq * np.pi * (d_pin / 2)**2 * np.abs(y) / np.sqrt(distance_sq)

    return solid_angle


def calculate_kn_cross_section(photon_energy, scatter_angle, solid_angle):
    """
    计算Klein-Nishina散射截面

    参数:
        photon_energy:   入射光子能量 (keV)
        scatter_angle:   散射角 (弧度)
        solid_angle:     立体角 (sr)

    返回:
        d_KN:            微分散射截面 (相对于立体角)
        KN:              总散射截面
        Esca:            散射光子能量 (keV)
    """
    # 物理常数
    m0c2 = 511  # 电子静止能量 (keV)

    # 无量纲参数 alpha = E / (m0*c^2)
    alpha = photon_energy / m0c2

    # 散射光子能量 (康普顿公式)
    # Es = Ei / (1 + alpha*(1-cos(theta)))
    Esca = photon_energy / (1 + alpha * (1 - np.cos(scatter_angle)))

    # 散射后的 alpha
    alpha_sca = Esca / m0c2

    # 微分散射截面 (Klein-Nishina公式)
    # dK/dOmega = r_e^2 * (Es/Ei)^2 * ...
    #             (1 + cos^2(theta)) / (1 + alpha*(1-cos(theta)))^2 * ...
    #             [1 + alpha^2*(1-cos(theta))^2 / ((1+cos^2(theta))*(1+alpha*(1-cos(theta))))]
    cos_theta = np.cos(scatter_angle)
    one_minus_cos = 1 - cos_theta
    cos_sq = cos_theta**2

    d_KN = solid_angle * 1 / (1 + alpha * one_minus_cos)**2 * ((1 + cos_sq) / 2) \
           * (1 + alpha**2 * one_minus_cos**2 / ((1 + cos_sq) * (1 + alpha * one_minus_cos)))

    # 总截面 (Klein-Nishina积分结果)
    # KN = 2*pi*r_e^2 * [ ... ]
    two_alpha = 2 * alpha
    KN = 2 * np.pi * ((alpha**3 + 9*alpha**2 + 8*alpha + 2) / (alpha**2 * (1 + two_alpha)**2) +
                      np.log(1 + two_alpha) * (alpha**2 - 2 - 2*alpha) / (2 * alpha**3))

    return d_KN, KN, Esca


def calculate_scatter_angle(x1, y1, z1, x2, y2, z2):
    """
    计算散射角

    参数:
        x1, y1, z1: 入射方向向量 (散射点 - 源点)
        x2, y2, z2: 出射方向向量 (探测器 - 散射点)

    返回:
        theta:       散射角 (弧度)
    """
    # 方向向量归一化
    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)

    # 余弦值 = (v1 . v2) / (|v1| * |v2|)
    cos_theta = (x1*x2 + y1*y2 + z1*z2) / (r1 * r2)

    # 散射角 (限制在 [0, pi] 范围)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    return theta


def calculate_attenuation(x0, y0, z0, x_pin, y_pin, z_pin,
                          photon_energy, num_segments, y_oc,
                          phantom_size, voxel_size,
                          phantom_offset_x, phantom_offset_y, phantom_offset_z,
                          phantom_data, miu_data):
    """
    计算光子从散射点到探测器的物质衰减

    参数:
        x0, y0, z0:       散射点坐标 (mm, 原始坐标系)
        x_pin, y_pin, z_pin: 针孔入口坐标 (mm)
        photon_energy:    光子能量 (keV)
        num_segments:     路径分段数
        y_oc:             旋转中心y偏移 (mm)
        phantom_size:     Phantom体素尺寸 [x, y, z]
        voxel_size:       体素大小 (mm)
        phantom_offset_x, phantom_offset_y, phantom_offset_z: Phantom中心偏移 (mm)
        phantom_data:     Phantom材料映射数组
        miu_data:         线性衰减系数 (20种材料 x 150个能量)

    返回:
        att_total:        总衰减因子 (无单位)
    """
    # 路径段长度（mm）
    dx = x_pin - x0
    dy = y_pin - y0
    dz = z_pin - z0
    segment_length_mm = np.sqrt(dx**2 + dy**2 + dz**2) / num_segments
    # miu单位为1/cm，长度需从mm转换为cm
    segment_length = segment_length_mm / 10.0

    # 初始化衰减累加器
    att_total = np.zeros(len(x0))

    energy_range = np.arange(1, 150.01, 0.01)
    if miu_data.ndim != 2:
        raise ValueError(f"Unexpected miu_data rank: {miu_data.ndim}")
    if miu_data.shape[0] != 20 and miu_data.shape[1] == 20:
        miu_data = miu_data.T
    if miu_data.shape[0] <= 0:
        raise ValueError(f"Unexpected miu_data shape: {miu_data.shape}")
    if np.any(phantom_data < 0):
        raise ValueError("phantom_data contains negative material indices.")
    if phantom_data.size > 0:
        max_material_index = int(np.max(phantom_data))
        if max_material_index >= miu_data.shape[0]:
            raise ValueError(
                "phantom_data contains material indices outside the attenuation table range: "
                f"max index {max_material_index}, attenuation rows {miu_data.shape[0]}. "
                "Pass a remapped phantom attenuation map, or use --disable-attenuation."
            )

    referenced_materials = np.unique(phantom_data).astype(int)
    miu_interp = {}
    for material_idx in referenced_materials:
        row = np.asarray(miu_data[material_idx], dtype=np.float64)
        if row.ndim != 1:
            row = row.flatten()
        if row.size != energy_range.size:
            raise ValueError(
                f"Material attenuation row {material_idx} has unexpected length {row.size}; "
                f"expected {energy_range.size}."
            )
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            raise ValueError(
                f"Material attenuation row {material_idx} is missing or invalid. "
                "Update the attenuation table or disable attenuation."
            )
        log_row = np.log(row)
        miu_interp[material_idx] = np.exp(
            np.interp(photon_energy, energy_range, log_row, left=log_row[0], right=log_row[-1])
        )

    for m in range(num_segments):
        # 计算路径上各点的坐标
        t = (m + 0.5) / num_segments  # 中点位置
        x_path = x0 + t * dx
        y_path = y0 + t * dy
        z_path = z0 + t * dz

        # 转换为phantom体素索引 (1-based)
        x_idx = np.ceil((x_path + phantom_offset_x) / voxel_size + 1).astype(int)
        y_idx = np.ceil((y_path + phantom_offset_y - y_oc) / voxel_size + 1).astype(int)
        z_idx = np.ceil((z_path + phantom_offset_z) / voxel_size + 1).astype(int)

        # 检查边界（是否在模体内）
        in_bounds = ((x_idx >= 1) & (x_idx <= phantom_size[0]) &
                     (y_idx >= 1) & (y_idx <= phantom_size[1]) &
                     (z_idx >= 1) & (z_idx <= phantom_size[2]))

        # 计算线性索引
        linear_idx = (y_idx - 1) + (x_idx - 1) * phantom_size[1] + \
                     (z_idx - 1) * phantom_size[0] * phantom_size[1]
        linear_idx[~in_bounds] = 0  # 越界时使用第一个体素

        # 获取材料索引
        materials = phantom_data[linear_idx]

        # 对每种材料计算衰减并累加
        att_segment = np.zeros(len(x0))
        for i, interp_values in miu_interp.items():
            # 只累加对应材料的贡献
            att_segment += interp_values * segment_length * (materials == i)

        # 只累加模体内的衰减
        att_segment[~in_bounds] = 0
        att_total += att_segment

    return att_total


def accumulate_projection(projection, weight, energy, det_idx_x, det_idx_z,
                          angle_idx, e_bins):
    """
    将权重累加到投影矩阵

    参数:
        projection:   投影矩阵
        weight:       光子权重
        energy:       光子能量 (keV, 用于确定能量bin)
        det_idx_x:    探测器x索引
        det_idx_z:    探测器z索引
        angle_idx:    角度索引
        e_bins:       能量分箱数
    返回:
        projection:   更新后的投影矩阵
    """
    # 创建副本以避免修改原始数据
    projection = projection.copy()
    det_idx_x = det_idx_x.copy() if isinstance(det_idx_x, np.ndarray) else det_idx_x
    det_idx_z = det_idx_z.copy() if isinstance(det_idx_z, np.ndarray) else det_idx_z
    weight = weight.copy() if isinstance(weight, np.ndarray) else weight
    energy = energy.copy() if isinstance(energy, np.ndarray) else energy

    if e_bins == 1:
        # 对于XRF模式，直接累加
        for i in range(len(weight)):
            projection[angle_idx, det_idx_z[i] - 1, det_idx_x[i] - 1] += weight[i]
    else:
        # 对于散射模式，按能量分箱累加
        for i in range(len(weight)):
            e_bin = int(np.ceil(energy[i]))
            e_bin = max(1, min(e_bin, e_bins))
            projection[e_bin - 1, det_idx_z[i] - 1, det_idx_x[i] - 1, angle_idx] += weight[i]

    return projection
