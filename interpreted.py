import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from copy import deepcopy
import json
import math
import os
import warnings
from tqdm import tqdm
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
# 导入自定义模块
from Nets import PredictModel, prepare_device
from RRP import CausalExplainer, normalize_causal_scores, compute_lag
# 读取配置文件（支持通过环境变量切换）
CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", "./model_config_SIF.json")
ls_id =0
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
print(f"加载配置文件: {CONFIG_PATH}")

DEFAULT_LAG_SELECTION_CONFIG = {
    "smooth_window": 3,
    "plateau_rel_tol": 0,
    "boundary_penalty":0.01,
    "seasonal_lag_penalty":0,
    "seasonal_period": 12,
    "edge_guard": 1,
    "min_raw_peak_fraction": 0.2,
    "use_vegetation_lag_prior": True,
    "vegetation_max_lag": 24,
}

LAG_SELECTION_CONFIG = {
    **DEFAULT_LAG_SELECTION_CONFIG,
    **config.get("analyze", {}).get("lag_selection", {})
}

LAG_DIAGNOSTIC_FIELDS = [
    "raw_lag",
    "lag_confidence",
    "lag_peak_raw",
    "lag_peak_adjusted",
    "max_effective_lag",
    "boundary_warning",
    "raw_boundary_warning",
    "seasonal_lag_warning",
    "lag_selection_method",
]


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return int(default)
    return int(value)


def configure_torch_runtime():
    """
    Tune PyTorch CPU threading for long spatial interpretation runs.

    Environment variables:
    - DTLN_NUM_THREADS: intra-op CPU threads, default min(32, os.cpu_count()).
    - DTLN_INTEROP_THREADS: inter-op threads, default 2.
    """
    if not _env_flag("DTLN_CONFIGURE_TORCH_THREADS", True):
        return

    cpu_count = os.cpu_count() or 1
    num_threads = _env_int("DTLN_NUM_THREADS", min(32, cpu_count))
    interop_threads = _env_int("DTLN_INTEROP_THREADS", 2)

    try:
        torch.set_num_threads(max(1, num_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except Exception:
        pass

    print(
        "PyTorch CPU threads: "
        f"intra={torch.get_num_threads()}, "
        f"interop={interop_threads} (requested)"
    )


_CAUSAL_WORKER_STATE = {}


def _init_causal_point_worker(model_path, worker_threads):
    state = _CAUSAL_WORKER_STATE
    torch.set_num_threads(max(1, int(worker_threads)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = torch.device("cpu")
    state["device"] = device

    model = PredictModel(
        state["model_cfg"],
        d_model=state["base_config"]["model"]["d_model"],
        n_head=state["base_config"]["model"]["n_head"],
        n_layers=state["base_config"]["model"]["n_layers"],
        ffn_hidden=state["base_config"]["model"]["hidden_layers"],
        drop_prob=state["base_config"]["model"]["drop_prob"],
        tau=state["base_config"]["model"]["tau"],
        use_geo_encoding=state["use_geo_encoding"],
        aux_series_num=state["aux_series_num"],
        static_dim=state["static_dim"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state["model"] = model


def _causal_point_worker(point_idx):
    state = _CAUSAL_WORKER_STATE
    coords = state["coords"]
    data = state["data"]
    aux_data = state.get("aux_data")
    static_data = state.get("static_data")
    seq_len = state["seq_len"]
    time_step = state["time_step"]
    var_names = state["var_names"]
    series_num = state["series_num"]
    m = state["m"]
    n = state["n"]
    model = state["model"]
    device = state["device"]
    use_geo_encoding = state["use_geo_encoding"]

    coord = coords[point_idx]
    point_data = data[point_idx]
    samples = []
    for t in range(seq_len, point_data.shape[0] + 1):
        samples.append(point_data[t - seq_len:t].reshape(seq_len, series_num, 1))

    if len(samples) == 0:
        return {
            "point_result": {
                "point_id": point_idx,
                "lat": coord["lat"],
                "lon": coord["lon"],
                "lat_idx": coord["lat_idx"],
                "lon_idx": coord["lon_idx"],
                "causal_edges": [],
                "num_edges": 0,
            },
            "valid": False,
            "relA": None,
            "relK": None,
        }

    input_tensor = torch.FloatTensor(np.array(samples)).to(device)
    input_tensor.requires_grad = True

    aux_tensor = None
    if aux_data is not None:
        point_aux = aux_data[point_idx]
        aux_samples = []
        for t in range(seq_len, point_aux.shape[0] + 1):
            aux_samples.append(point_aux[t - seq_len:t].reshape(seq_len, point_aux.shape[1], 1))
        if aux_samples:
            aux_tensor = torch.FloatTensor(np.array(aux_samples)).to(device)

    static_tensor = None
    if static_data is not None:
        batch_n = input_tensor.shape[0]
        static_point = np.asarray(static_data[point_idx], dtype=float)
        static_tensor = torch.FloatTensor(np.repeat(static_point[None, :], batch_n, axis=0)).to(device)

    if use_geo_encoding:
        batch_n = input_tensor.shape[0]
        lat_tensor = torch.full((batch_n,), coord["lat"], dtype=torch.float32).to(device)
        lon_tensor = torch.full((batch_n,), coord["lon"], dtype=torch.float32).to(device)
    else:
        lat_tensor, lon_tensor = None, None

    point_relA = []
    point_relK = []
    try:
        for interpreted_series in range(series_num):
            relA, relK_aligned = generate_RRP_scores_all(
                model,
                input_tensor,
                interpreted_series,
                device,
                debug=False,
                lat=lat_tensor,
                lon=lon_tensor,
                aux_data=aux_tensor,
                static_vars=static_tensor,
            )
            relA_i = relA[interpreted_series].detach().cpu().numpy()
            relK_i = relK_aligned.detach().cpu().numpy() if torch.is_tensor(relK_aligned) else relK_aligned
            point_relA.append(relA_i)
            point_relK.append(relK_i)

        causal_edges = analyze_all_causes(point_relA, point_relK, m, n, time_step, var_names)
        valid = True
    except Exception as e:
        causal_edges = []
        valid = False

    return {
        "point_result": {
            "point_id": point_idx,
            "lat": coord["lat"],
            "lon": coord["lon"],
            "lat_idx": coord["lat_idx"],
            "lon_idx": coord["lon_idx"],
            "causal_edges": causal_edges,
            "num_edges": len(causal_edges),
        },
        "valid": valid,
        "relA": point_relA if valid else None,
        "relK": point_relK if valid else None,
    }
    
def load_nc_data(file_path):
    print(f"正在读取netCDF文件: {file_path}")
    if not os.path.exists(file_path): raise FileNotFoundError(f"文件不存在: {file_path}")
    ds = nc.Dataset(file_path, 'r')
    lat = ds.variables['lat'][:] if 'lat' in ds.variables else ds.variables['latitude'][:]
    lon = ds.variables['lon'][:] if 'lon' in ds.variables else ds.variables['longitude'][:]
    time = ds.variables['time'][:] if 'time' in ds.variables else None
    return ds, lat, lon, time

def preprocess_geo_data(ds, predictors, target_var, lat, lon, time, lulc_ids, 
                         return_coords=False, stride=1, aux_predictors=None, static_vars=None):
    """
    预处理地理数据
    
    Args:
        ds: netCDF4 数据集
        predictors: 主要预测变量列表 (参与因果发现)
        target_var: 目标变量
        lat: 纬度数组
        lon: 经度数组
        time: 时间数组
        lulc_ids: 土地利用类型ID列表
        return_coords: 是否返回坐标信息
        stride: 空间扫描步长 (默认1，即每个格点都取)
        aux_predictors: 辅助时间序列变量列表 (不参与因果发现)
        static_vars: 静态变量列表 (不随时间变化)
    
    Returns:
        data: 主时序数据 (N, T, V)
        aux_data: 辅助时序数据 (N, T, V_aux) 或 None
        static_data: 静态数据 (N, V_static) 或 None
        coords (可选): 坐标列表 [{'lat': float, 'lon': float, ...}, ...]
    """
    print("预处理数据...")
    aux_predictors = aux_predictors or []
    static_vars = static_vars or []
    
    # 缓存文件路径 (使用 analyze 配置)
    data_cache_path = config["analyze"]["PROCEED_DATA"]
    coords_cache_path = data_cache_path.replace('.npy', '_coords.json')
    aux_cache_path = data_cache_path.replace('.npy', '_aux.npy') if aux_predictors else None
    static_cache_path = data_cache_path.replace('.npy', '_static.npy') if static_vars else None
    
    # 尝试加载缓存数据
    if os.path.exists(data_cache_path):
        print(f"  加载缓存数据: {data_cache_path}")
        data = np.load(data_cache_path)
        
        # 加载辅助数据缓存
        aux_data = None
        if aux_cache_path and os.path.exists(aux_cache_path):
            aux_data = np.load(aux_cache_path)
            print(f"  加载辅助变量缓存: {aux_cache_path}")
        elif aux_predictors:
            print(f"  警告: 未找到辅助变量缓存，将重新处理...")
            return _process_from_netcdf_with_aux(
                ds, predictors, target_var, lat, lon, time, lulc_ids, stride,
                data_cache_path, coords_cache_path, aux_predictors, static_vars,
                aux_cache_path, static_cache_path, return_coords
            )
        
        # 加载静态数据缓存
        static_data = None
        if static_cache_path and os.path.exists(static_cache_path):
            static_data = np.load(static_cache_path)
            print(f"  加载静态变量缓存: {static_cache_path}")
        elif static_vars:
            print(f"  警告: 未找到静态变量缓存，将重新处理...")
            return _process_from_netcdf_with_aux(
                ds, predictors, target_var, lat, lon, time, lulc_ids, stride,
                data_cache_path, coords_cache_path, aux_predictors, static_vars,
                aux_cache_path, static_cache_path, return_coords
            )
        
        # 加载坐标缓存
        coords = None
        if return_coords:
            if os.path.exists(coords_cache_path):
                with open(coords_cache_path, 'r') as f:
                    coords = json.load(f)
                print(f"  加载坐标缓存: {len(coords)} 个地理点")
            else:
                print(f"  警告: 未找到坐标缓存，将重新处理...")
                return _process_from_netcdf_with_aux(
                    ds, predictors, target_var, lat, lon, time, lulc_ids, stride,
                    data_cache_path, coords_cache_path, aux_predictors, static_vars,
                    aux_cache_path, static_cache_path, return_coords
                )
        
        if return_coords:
            return data, aux_data, static_data, coords
        return data, aux_data, static_data
    
    # 如果没有缓存，从 netCDF 处理数据
    print(f"  未找到缓存，从 netCDF 文件处理数据...")
    return _process_from_netcdf_with_aux(
        ds, predictors, target_var, lat, lon, time, lulc_ids, stride,
        data_cache_path, coords_cache_path, aux_predictors, static_vars,
        aux_cache_path, static_cache_path, return_coords
    )


def _process_from_netcdf(ds, predictors, target_var, lat, lon, time, lulc_ids, stride, 
                          data_cache_path, coords_cache_path):
    """
    从 netCDF 文件处理数据并保存缓存 (旧版兼容)
    """
    return _process_from_netcdf_with_aux(
        ds, predictors, target_var, lat, lon, time, lulc_ids, stride,
        data_cache_path, coords_cache_path, [], [], None, None, True
    )


def _process_from_netcdf_with_aux(ds, predictors, target_var, lat, lon, time, lulc_ids, stride, 
                                    data_cache_path, coords_cache_path, aux_predictors, static_vars,
                                    aux_cache_path, static_cache_path, return_coords):
    """
    从 netCDF 文件处理数据并保存缓存 (支持辅助变量和静态变量)
    
    Returns:
        data: 主时序数据 (N, T, V)
        aux_data: 辅助时序数据 (N, T, V_aux) 或 None
        static_data: 静态数据 (N, V_static) 或 None
        coords: 坐标列表 [{'lat': float, 'lon': float, ...}, ...]
    """
    all_data = []
    all_aux_data = [] if aux_predictors else None
    all_static_data = [] if static_vars else None
    coords = []
    time_len = len(time)
    
    # 构建变量字典
    all_vars = predictors + [target_var, 'CLCD'] + aux_predictors + static_vars
    var_dict = {v: ds.variables[v] for v in all_vars if v in ds.variables}
    
    # 检查辅助变量和静态变量是否存在
    missing_aux = [v for v in aux_predictors if v not in var_dict]
    missing_static = [v for v in static_vars if v not in var_dict]
    if missing_aux:
        print(f"  警告: 辅助变量不存在: {missing_aux}")
    if missing_static:
        print(f"  警告: 静态变量不存在: {missing_static}")
    
    for i in tqdm(range(0, len(lat), stride), desc="扫描网格"):
        for j in range(0, len(lon), stride):
            # LULC 筛选
            if 'CLCD' in var_dict:
                c = var_dict['CLCD']
                if c.ndim == 3:
                    lulc_ts = c[:, i, j]
                    if isinstance(lulc_ts, np.ma.MaskedArray):
                        lulc_ts = lulc_ts.filled(np.nan)
                    valid_lulc = lulc_ts[~np.isnan(lulc_ts)]
                    if len(valid_lulc) == 0:
                        continue
                    from scipy import stats
                    mode_result = stats.mode(valid_lulc.astype(int))
                    val = mode_result.mode if hasattr(mode_result, 'mode') else mode_result[0]
                    if isinstance(val, np.ndarray) and val.ndim > 0:
                        val = val.item()
                else:
                    val = c[i, j]
                
                if isinstance(val, np.ma.core.MaskedConstant) or np.isnan(val):
                    continue
                if int(val) not in lulc_ids:
                    continue
            
            # 提取主变量的时序数据
            pt_data, valid = {}, True
            for v_name in predictors + [target_var]:
                if v_name not in var_dict: 
                    pt_data[v_name] = np.zeros(time_len)
                    continue
                v = var_dict[v_name]
                ts = v[:, i, j] if v.ndim == 3 else np.full(time_len, v[i, j])
                if isinstance(ts, np.ma.MaskedArray): 
                    ts = ts.filled(np.nan)
                if np.all(np.isnan(ts)): 
                    valid = False
                    break
                pt_data[v_name] = ts
            
            if not valid: 
                continue
            
            # 提取辅助时序变量
            aux_pt_data = {}
            if aux_predictors:
                for v_name in aux_predictors:
                    if v_name not in var_dict:
                        aux_pt_data[v_name] = np.zeros(time_len)
                        continue
                    v = var_dict[v_name]
                    ts = v[:, i, j] if v.ndim == 3 else np.full(time_len, v[i, j])
                    if isinstance(ts, np.ma.MaskedArray):
                        ts = ts.filled(np.nan)
                    aux_pt_data[v_name] = ts
            
            # 提取静态变量
            static_pt_data = {}
            if static_vars:
                for v_name in static_vars:
                    if v_name not in var_dict:
                        static_pt_data[v_name] = 0.0
                        continue
                    v = var_dict[v_name]
                    if v.ndim == 2:
                        val = v[i, j]
                    elif v.ndim == 3:
                        val = v[0, i, j]
                    else:
                        val = v[i, j] if hasattr(v, '__getitem__') else 0.0
                    
                    if isinstance(val, np.ma.core.MaskedConstant):
                        val = 0.0
                    elif isinstance(val, np.ma.MaskedArray):
                        val = val.filled(0.0)
                    static_pt_data[v_name] = float(val)
            
            # 插值填充主变量缺失值
            df = pd.DataFrame(pt_data).interpolate().dropna()
            if len(df) != time_len:
                continue
            
            all_data.append(df.values)
            
            # 处理辅助时序变量
            if aux_predictors:
                aux_df = pd.DataFrame(aux_pt_data).interpolate().fillna(0)
                all_aux_data.append(aux_df.values)
            
            # 处理静态变量
            if static_vars:
                static_values = [static_pt_data.get(v, 0.0) for v in static_vars]
                all_static_data.append(static_values)
            
            # 保存坐标信息
            coords.append({
                'lat': float(lat[i]),
                'lon': float(lon[j]),
                'lat_idx': int(i),
                'lon_idx': int(j)
            })
    
    if not all_data:
        raise ValueError("无有效数据")
    
    data = np.array(all_data)
    print(f"  主变量处理完成: {data.shape[0]} 个有效地理点, 时间步长 {data.shape[1]}, 变量数 {data.shape[2]}")
    
    # 保存主数据缓存
    np.save(data_cache_path, data)
    print(f"  已保存主数据缓存: {data_cache_path}")
    
    # 处理辅助数据
    aux_data = None
    if aux_predictors and all_aux_data:
        aux_data = np.array(all_aux_data)
        np.save(aux_cache_path, aux_data)
        print(f"  辅助变量处理完成: {aux_data.shape}, 已保存到 {aux_cache_path}")
    
    # 处理静态数据
    static_data = None
    if static_vars and all_static_data:
        static_data = np.array(all_static_data)
        np.save(static_cache_path, static_data)
        print(f"  静态变量处理完成: {static_data.shape}, 已保存到 {static_cache_path}")
    
    # 保存坐标缓存
    with open(coords_cache_path, 'w') as f:
        json.dump(coords, f, indent=2)
    print(f"  已保存坐标缓存: {coords_cache_path} ({len(coords)} 个地理点)")
    
    if return_coords:
        return data, aux_data, static_data, coords
    return data, aux_data, static_data


class GeoDataset(Dataset):
    """
    地理时序数据加载器，与 TimeseriesDataLoader 处理方式一致
    输入: (time_step, series_num, feature_dim)
    目标: (output_window, series_num, output_dim)
    
    新增功能：支持地理坐标 (lat, lon) 信息
    """
    def __init__(self, data, seq_len, output_window=1, coords=None):
        """
        Args:
            data: 时序数据 (N_points, T, V)
            seq_len: 输入窗口长度
            output_window: 输出窗口长度
            coords: 坐标列表 [{'lat': float, 'lon': float, ...}, ...]。
                    如果提供，则每个样本将包含对应的经纬度。
        """
        self.samples, self.targets = [], []
        self.lats, self.lons = [], []  # 新增: 存储坐标
        self.has_coords = coords is not None and len(coords) > 0
        n_p, n_t, n_v = data.shape
        
        assert seq_len < n_t + 1, "输入窗口长度必须小于数据总长度"
        assert output_window < seq_len, "输出窗口长度必须小于输入窗口长度"
        
        for p in range(n_p):
            ts = data[p]  # (n_t, n_v)
            # 获取该点的坐标
            if self.has_coords and p < len(coords):
                pt_lat = coords[p]['lat']
                pt_lon = coords[p]['lon']
            else:
                pt_lat, pt_lon = 0.0, 0.0  # 默认值
            
            for t in range(seq_len, n_t + 1):
                # 输入: 前 seq_len 步的所有变量 (time_step, series_num, feature_dim)
                sample = ts[t-seq_len:t].reshape(seq_len, n_v, 1)
                # 目标: 后 output_window 步的所有变量 (output_window, series_num, output_dim)
                target = ts[t-output_window:t].reshape(output_window, n_v, 1)
                self.samples.append(sample)
                self.targets.append(target)
                # 每个样本都保存对应的坐标
                self.lats.append(pt_lat)
                self.lons.append(pt_lon)
        
        self.samples = torch.FloatTensor(np.array(self.samples))
        self.targets = torch.FloatTensor(np.array(self.targets))
        self.lats = torch.FloatTensor(self.lats)
        self.lons = torch.FloatTensor(self.lons)
        print(f"数据集大小 - 样本: {self.samples.shape}, 目标: {self.targets.shape}")
        if self.has_coords:
            print(f"  包含地理坐标: lat 范围 [{self.lats.min():.2f}, {self.lats.max():.2f}], "
                  f"lon 范围 [{self.lons.min():.2f}, {self.lons.max():.2f}]")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        if self.has_coords:
            return self.samples[idx], self.targets[idx], self.lats[idx], self.lons[idx]
        return self.samples[idx], self.targets[idx]


def load_model_and_explain(model_path, config2, input_data, target_idx, var_names, output_dir):
    """
    加载训练好的模型并生成因果解释
    
    Args:
        model_path: 模型检查点路径
        config: 模型配置字典
        input_data: 输入数据
        target_idx: 目标变量索引
        var_names: 变量名称列表
        output_dir: 输出目录
    
    Returns:
        causal_results: 因果分析结果
    """
    # 初始化模型
    model = PredictModel(config2, d_model=config["model"]["d_model"], n_head=config["model"]["n_head"], n_layers=config["model"]["n_layers"], 
                         ffn_hidden=config["model"]["hidden_layers"], drop_prob=config["model"]["drop_prob"], tau=config["model"]["tau"])
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(model.device)
    model.eval()
    
    print(f"模型已加载: {model_path}")
    if 'rmse' in checkpoint:
        print(f"  RMSE: {checkpoint['rmse']:.6f}")
    if 'iteration' in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    
    # 生成因果解释
    return generate_causal_explanation(model, input_data, target_idx, var_names, output_dir)

def generate_causal_explanation(
    model,
    input_data,
    target_series_idx,
    var_names,
    output_dir,
    batch_size=config["analyze"]["BATCH_SIZE"],
    save_plots=True,
    lat=None,
    lon=None,
    aux_data=None,
    static_vars=None,
):
    """
    独立的因果解释函数，可用于新数据点的推理
    
    Args:
        model: 训练好的模型
        input_data: 输入数据 [N, time_step, series_num, feature_dim] 或 DataLoader
        target_series_idx: 目标时间序列的索引
        var_names: 变量名称列表
        output_dir: 输出目录
        batch_size: 批处理大小
        save_plots: 是否保存图表
    
    Returns:
        causal_results: 因果分析结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化解释器
    explainer = CausalExplainer(model)
    
    # 如果输入是DataLoader，转换为张量
    if isinstance(input_data, DataLoader):
        all_data = []
        for batch_x, _ in input_data:
            all_data.append(batch_x)
        input_data = torch.cat(all_data, dim=0)
    
    input_data = input_data.to(model.device)
    input_data.requires_grad = True
    lat = lat.to(model.device) if torch.is_tensor(lat) else lat
    lon = lon.to(model.device) if torch.is_tensor(lon) else lon
    aux_data = aux_data.to(model.device) if torch.is_tensor(aux_data) else aux_data
    static_vars = static_vars.to(model.device) if torch.is_tensor(static_vars) else static_vars
    
    print(f"\n{'='*60}")
    print(f"正在生成因果解释...")
    print(f"输入数据形状: {input_data.shape}")
    print(f"目标变量: {var_names[target_series_idx]}")
    print(f"{'='*60}")
    
    # 生成因果分数
    relA, relK = explainer.generate_causal_scores(
        input_data,
        target_series_idx,
        batch_size,
        lat=lat,
        lon=lon,
        aux_data=aux_data,
        static_vars=static_vars,
    )
    
    # 解释因果图
    causal_results = explainer.interpret_causal_graph(
        input_data,
        target_series_idx,
        var_names,
        batch_size,
        threshold=0.05,
        lat=lat,
        lon=lon,
        aux_data=aux_data,
        static_vars=static_vars,
    )
    
    # 打印因果关系
    print(f"\n=== 发现的因果关系 (影响 {var_names[target_series_idx]}) ===")
    for rel in causal_results['causal_relations'][:20]:  # 只显示前20个
        print(f"  {rel['cause']} -> {rel['effect']}: {rel['score']:.4f}")
    
    if save_plots:
        seq_len = input_data.shape[1]
        
        # 1. 绘制注意力矩阵因果分数热力图
        plt.figure(figsize=(12, 10))
        relA_np = causal_results['relA']
        sns.heatmap(relA_np, xticklabels=var_names, yticklabels=var_names, 
                    cmap='RdBu_r', center=0, annot=False)
        plt.title(f'Attention Causal Scores (Target: {var_names[target_series_idx]})')
        plt.xlabel('Effect Variables')
        plt.ylabel('Cause Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'causal_attention_heatmap.png'), dpi=500)
        plt.close()
        
        # 2. 绘制综合因果分数热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(causal_results['combined_scores'], 
                    xticklabels=var_names, yticklabels=var_names,
                    cmap='RdBu_r', center=0, annot=False)
        plt.title(f'Combined Causal Scores (Target: {var_names[target_series_idx]})')
        plt.xlabel('Effect Variables')
        plt.ylabel('Cause Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'causal_combined_heatmap.png'), dpi=500)
        plt.close()
        
        # 3. 绘制对目标变量的因果影响条形图
        plt.figure(figsize=(14, 6))
        target_scores = causal_results['combined_scores'][:, target_series_idx]
        sorted_idx = np.argsort(target_scores)[::-1]
        sorted_names = [var_names[i] for i in sorted_idx]
        sorted_scores = target_scores[sorted_idx]
        
        colors = ['red' if s > 0 else 'blue' for s in sorted_scores]
        plt.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Causal Score')
        plt.ylabel('Variables')
        plt.title(f'Causal Influence on {var_names[target_series_idx]}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'causal_influence_bar.png'), dpi=500)
        plt.close()
        
        # 4. 绘制时空因果关系热力图
        try:
            relK_np = causal_results['relK']
            print(f"  relK_np shape for temporal plot: {relK_np.shape}")
            
            if relK_np.ndim >= 3:
                # 根据维度处理时空因果
                if relK_np.ndim == 5:
                    # (n_head, series_num, series_num, time_step, time_step)
                    # 取对目标变量的影响，对头和最后一个时间维度取平均
                    temporal_causal = relK_np.mean(axis=0)[:, target_series_idx, :, :].mean(axis=-1)
                elif relK_np.ndim == 4:
                    # (series_num, series_num, time_step, time_step)
                    temporal_causal = relK_np[:, target_series_idx, :, :].mean(axis=-1)
                elif relK_np.ndim == 3:
                    # (series_num, series_num, time_step)
                    temporal_causal = relK_np[:, target_series_idx, :]
                else:
                    temporal_causal = None
                
                if temporal_causal is not None and temporal_causal.ndim == 2:
                    plt.figure(figsize=(14, 8))
                    # 确保 x 轴标签与数据匹配
                    n_vars, n_time = temporal_causal.shape
                    sns.heatmap(temporal_causal, 
                               xticklabels=range(1, n_time+1), 
                               yticklabels=var_names[:n_vars] if n_vars <= len(var_names) else [f'Var_{i}' for i in range(n_vars)], 
                               cmap='RdBu_r', center=0)
                    plt.title(f'Temporal Causal Influence on {var_names[target_series_idx]}')
                    plt.xlabel('Lag Time')
                    plt.ylabel('Variables')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'temporal_causal_heatmap.png'), dpi=500)
                    plt.close()
                else:
                    print(f"  跳过时空因果图: temporal_causal shape 不合适")
            else:
                print(f"  跳过时空因果图: relK 维度不足 ({relK_np.ndim})")
        except Exception as e:
            print(f"时空因果图绘制失败: {e}")
        
        print(f"\n因果分析图表已保存到: {output_dir}")
    
    # 保存结果到文件
    import json
    results_to_save = {
        'target_variable': var_names[target_series_idx],
        'causal_relations': causal_results['causal_relations']
    }
    with open(os.path.join(output_dir, 'causal_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    return causal_results


def run_causal_analysis(model_path, nc_file, output_dir, predictors, target_var, lulc_ids,
                         seq_len=config["model"]["SEQ_LEN"], output_window=config["model"].get("output_window", 1), 
                         batch_size=config["analyze"]["BATCH_SIZE"], m=config["analyze"]["KMeans_M"], n=config["analyze"]["KMeans_N"]):
    """
    独立的因果分析函数，只分析 PREDICTORS → TARGET 的因果关系
    
    参考 code/CausalFormer/interpret.py 中的实现，生成 (cause, effect=TARGET, lag) 格式的因果图
    
    Args:
        model_path: 训练好的模型路径 (.pth 文件)
        nc_file: netCDF 数据文件路径
        output_dir: 输出目录
        predictors: 预测变量列表
        target_var: 目标变量名称 (只分析对此变量的因果)
        lulc_ids: 土地利用类型 ID 列表
        seq_len: 序列长度
        output_window: 输出窗口大小
        batch_size: 批处理大小
        m: 选择top m个聚类簇 (参考interpret.py)
        n: KMeans聚类数 (参考interpret.py)
    
    Returns:
        all_results: 因果分析结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"因果分析模式 - 只分析对 {target_var} 的因果影响")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"数据文件: {nc_file}")
    print(f"目标变量: {target_var}")
    print(f"KMeans参数: m={m}, n={n}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据（并获取坐标信息）
    ds, lat, lon, time = load_nc_data(nc_file)
    
    # 获取辅助变量配置 (如果存在)
    aux_predictors = config["analyze"].get("AUX_PREDICTORS", [])
    static_vars_list = config["analyze"].get("STATIC_VARS", [])
    
    data, aux_data, static_data, coords = preprocess_geo_data(
        ds, predictors, target_var, lat, lon, time, lulc_ids, 
        return_coords=True,
        aux_predictors=aux_predictors,
        static_vars=static_vars_list
    )
    
    # 确定变量名称列表
    var_names = predictors.copy()
    t_idx = predictors.index(target_var) if target_var in predictors else -1
    if t_idx == -1:
        var_names.append(target_var)
        t_idx = len(var_names) - 1
    
    print(f"变量列表: {var_names}")
    print(f"目标变量 '{target_var}' 索引: {t_idx}")
    print(f"地理点数量: {len(coords)}")
    
    # 标准化主数据
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    N, T, V = data.shape
    data = scaler.fit_transform(data.reshape(-1, V)).reshape(N, T, V)
    
    # 标准化辅助数据
    if aux_data is not None:
        aux_scaler = MinMaxScaler(feature_range=(0.1, 1))
        V_aux = aux_data.shape[2]
        aux_data = aux_scaler.fit_transform(aux_data.reshape(-1, V_aux)).reshape(N, T, V_aux)
    
    # 标准化静态数据
    if static_data is not None:
        static_scaler = MinMaxScaler(feature_range=(0.1, 1))
        static_data = static_scaler.fit_transform(static_data)
    
    # 2. 加载模型
    cfg = {
        'n_gpu': ls_id, 
        'data_loader': {
            'args': {
                'time_step': seq_len, 
                'output_window': output_window, 
                'series_num': len(var_names), 
                'feature_dim': 1, 
                'output_dim': 1
            }
        }
    }
    
    device = prepare_device(ls_id)
    
    # 检测是否使用地理编码 (从配置文件读取)
    use_geo_encoding = config["train"].get("USE_GEO_ENCODING", False)
    
    model = PredictModel(cfg, d_model=config["model"]["d_model"], n_head=config["model"]["n_head"], n_layers=config["model"]["n_layers"], 
                         ffn_hidden=config["model"]["hidden_layers"], drop_prob=config["model"]["drop_prob"], tau=config["model"]["tau"],
                         use_geo_encoding=use_geo_encoding,
                         aux_series_num=len(aux_predictors) if aux_predictors else 0,
                         static_dim=len(static_vars_list) if static_vars_list else 0).to(device)
    
    if use_geo_encoding:
        print(f"  模型已启用地理位置编码 (GeoPositionalEncoding)")
    if aux_predictors:
        print(f"  辅助时序变量: {aux_predictors}")
    if static_vars_list:
        print(f"  静态变量: {static_vars_list}")
    
    # 加载权重
    print(f"\n正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印模型信息
    if 'iteration' in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    if 'rmse' in checkpoint:
        print(f"  RMSE: {checkpoint['rmse']:.6f}")
    
    # 3. 参数设置
    series_num = len(var_names)
    time_step = seq_len
    
    # 确保 m < n，且 n 不超过 series_num
    n = min(n, series_num)
    m = min(m, n - 1) if n > 1 else 1
    print(f"调整后的 KMeans 参数: m={m}, n={n}")
    
    # 4. 为每个地理点生成因果图 (只分析对 TARGET 的因果)
    print(f"\n正在为 {len(coords)} 个地理点分析 PREDICTORS → {target_var} 的因果关系...")
    
    all_point_data = []  # 存储所有点的详细信息
    
    # 累积全局 relA (只针对 TARGET)
    global_relA = np.zeros(series_num)  # 所有变量对 TARGET 的因果分数
    global_relK = np.zeros((series_num, time_step))  # 时间滞后
    valid_points = 0
    
    for point_idx in tqdm(range(len(coords)), desc="因果分析"):
        coord = coords[point_idx]
        point_data = data[point_idx]  # (T, V)
        
        # 生成输入样本
        samples = []
        for t in range(seq_len, T + 1):
            sample = point_data[t-seq_len:t].reshape(seq_len, V, 1)
            samples.append(sample)
        
        if len(samples) == 0:
            continue
        
        input_tensor = torch.FloatTensor(np.array(samples)).to(device)
        # print(input_tensor.shape)
        input_tensor.requires_grad = True

        aux_tensor = None
        if aux_data is not None:
            point_aux = aux_data[point_idx]
            aux_samples = []
            for t in range(seq_len, point_aux.shape[0] + 1):
                aux_samples.append(point_aux[t - seq_len:t].reshape(seq_len, point_aux.shape[1], 1))
            if aux_samples:
                aux_tensor = torch.FloatTensor(np.array(aux_samples)).to(device)

        static_tensor = None
        if static_data is not None:
            batch_size_pt = input_tensor.shape[0]
            static_point = np.asarray(static_data[point_idx], dtype=float)
            static_tensor = torch.FloatTensor(np.repeat(static_point[None, :], batch_size_pt, axis=0)).to(device)
        
        # 准备坐标张量 (如果启用地理编码)
        if use_geo_encoding:
            batch_size_pt = input_tensor.shape[0]
            lat_tensor = torch.full((batch_size_pt,), coord['lat'], dtype=torch.float32).to(device)
            lon_tensor = torch.full((batch_size_pt,), coord['lon'], dtype=torch.float32).to(device)
        else:
            lat_tensor, lon_tensor = None, None
        
        try:
            # 只对 TARGET 变量进行 RRP 分析
            # interpreted_series = t_idx (TARGET 的索引)
            relA, relK_aligned = generate_RRP_scores(
                model,
                input_tensor,
                t_idx,
                device,
                lat=lat_tensor,
                lon=lon_tensor,
                aux_data=aux_tensor,
                static_vars=static_tensor,
            )
            
            # relA 形状: (series_num, series_num)
            # 提取对 TARGET 的因果分数: relA[t_idx] = 所有变量对 TARGET 的影响
            relA_target = relA[t_idx].detach().cpu().numpy()  # (series_num,)
            
            # relK_aligned 形状: (series_num, time_step)
            # relK_aligned[j] = 变量 j 对 TARGET 的时间滞后分数
            if torch.is_tensor(relK_aligned):
                relK_target = relK_aligned.detach().cpu().numpy()  # (series_num, time_step)
            else:
                relK_target = relK_aligned
            
            # 调试信息（仅第一个点）
            if point_idx == 0:
                print(f"\n调试 - relK_target 形状: {relK_target.shape}")
                print(f"调试 - relA_target: min={relA_target.min():.6f}, max={relA_target.max():.6f}")
                # 打印每个变量的 relK 分布
                for var_idx in range(min(5, series_num)):
                    rk = relK_target[var_idx]
                    if np.sum(rk) > 0:
                        max_idx = np.argmax(rk)
                        lag = time_step - 1 - max_idx
                        print(f"  Var {var_idx} ({var_names[var_idx]}): argmax={max_idx}, lag={lag}, max_val={rk.max():.6f}")
                    else:
                        print(f"  Var {var_idx} ({var_names[var_idx]}): all zeros")
            
            # 累加到全局
            global_relA += relA_target
            global_relK += relK_target
            valid_points += 1
            
            # 为该点生成因果边 (只针对 TARGET)
            causal_edges = analyze_target_causes(
                relA_target, relK_target, m, n, time_step, var_names, target_var, t_idx
            )
            
        except Exception as e:
            print(f"  点 {point_idx} 处理失败: {e}")
            causal_edges = []
        
        # 记录该点的因果图
        point_result = {
            'point_id': point_idx,
            'lat': coord['lat'],
            'lon': coord['lon'],
            'lat_idx': coord['lat_idx'],
            'lon_idx': coord['lon_idx'],
            'causal_edges': causal_edges,
            'num_edges': len(causal_edges)
        }
        all_point_data.append(point_result)
    
    # 5. 计算全局因果图 (所有点的平均)
    if valid_points > 0:
        global_relA = global_relA / valid_points
        global_relK = global_relK / valid_points
    
    global_causal_edges = analyze_target_causes(
        global_relA, global_relK, m, n, time_step, var_names, target_var, t_idx
    )
    
    print(f"\n调试信息:")
    print(f"  有效点数: {valid_points}")
    print(f"  全局 relA (对 {target_var}): {global_relA}")
    print(f"  全局因果边数: {len(global_causal_edges)}")
    
    # 6. 保存结果
    # 6.1 保存为 JSON
    results_json = {
        'metadata': {
            'model_path': model_path,
            'nc_file': nc_file,
            'target_var': target_var,
            'predictors': predictors,
            'var_names': var_names,
            'seq_len': seq_len,
            'output_window': output_window,
            'n_points': valid_points,
            'kmeans_m': m,
            'kmeans_n': n,
            'lag_selection': _lag_selection_cfg(),
            'analysis_type': 'PREDICTORS → TARGET only'
        },
        'global_causal_graph': {
            'description': f'全局因果图: 所有变量 → {target_var}',
            'edges': global_causal_edges
        },
        'point_causal_graphs': all_point_data
    }
    
    json_path = os.path.join(output_dir, 'causal_graph.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"\n因果图已保存: {json_path}")
    
    # 6.2 保存全局因果图为 CSV (格式: cause, effect=TARGET, lag)
    global_csv_path = os.path.join(output_dir, 'global_causal_graph.csv')
    if len(global_causal_edges) > 0:
        df_global = pd.DataFrame(global_causal_edges)
    else:
        df_global = pd.DataFrame(columns=['cause_idx', 'cause_name', 'effect_idx', 'effect_name', 'lag', 'score'] + LAG_DIAGNOSTIC_FIELDS)
    df_global.to_csv(global_csv_path, index=False)
    print(f"全局因果图 CSV: {global_csv_path}")
    
    # 6.3 保存每个点的因果图为 CSV (包含经纬度)
    all_edges_data = []
    for pt in all_point_data:
        for edge in pt['causal_edges']:
            edge_record = {
                'point_id': pt['point_id'],
                'lat': pt['lat'],
                'lon': pt['lon'],
                'cause_idx': edge['cause_idx'],
                'cause_name': edge['cause_name'],
                'effect_idx': edge['effect_idx'],
                'effect_name': edge['effect_name'],
                'lag': edge['lag'],
                'score': edge.get('score', 0)
            }
            for field in LAG_DIAGNOSTIC_FIELDS:
                edge_record[field] = edge.get(field)
            all_edges_data.append(edge_record)
    
    points_csv_path = os.path.join(output_dir, 'point_causal_graphs.csv')
    if len(all_edges_data) > 0:
        df_points = pd.DataFrame(all_edges_data)
    else:
        df_points = pd.DataFrame(columns=['point_id', 'lat', 'lon', 'cause_idx', 'cause_name', 'effect_idx', 'effect_name', 'lag', 'score'] + LAG_DIAGNOSTIC_FIELDS)
    df_points.to_csv(points_csv_path, index=False)
    print(f"点位因果图 CSV: {points_csv_path}")
    save_lag_diagnostics(all_edges_data, output_dir, time_step, suffix="target")
    
    # 6.4 保存地理点统计信息
    points_summary = [{
        'point_id': pt['point_id'],
        'lat': pt['lat'],
        'lon': pt['lon'],
        'lat_idx': pt['lat_idx'],
        'lon_idx': pt['lon_idx'],
        'num_causal_edges': pt['num_edges']
    } for pt in all_point_data]
    
    summary_csv_path = os.path.join(output_dir, 'points_summary.csv')
    pd.DataFrame(points_summary).to_csv(summary_csv_path, index=False)
    print(f"点位摘要 CSV: {summary_csv_path}")
    
    # 7. 可视化因果图
    if len(global_causal_edges) > 0:
        plot_target_causal_graph(global_causal_edges, var_names, target_var, output_dir)
    else:
        print("\n警告: 没有发现因果关系，跳过可视化")
    
    # 8. 为每个点生成单独的因果图
    point_graphs_dir = os.path.join(output_dir, 'point_graphs')
    os.makedirs(point_graphs_dir, exist_ok=True)
    print(f"\n正在为每个点生成因果图...")
    
    for pt in tqdm(all_point_data[:50], desc="生成点位因果图"):  # 限制前50个点
        if len(pt['causal_edges']) > 0:
            plot_point_causal_graph(
                pt['causal_edges'], 
                var_names, 
                target_var, 
                pt['point_id'],
                pt['lat'],
                pt['lon'],
                point_graphs_dir
            )
    
    print(f"点位因果图已保存到: {point_graphs_dir}")
    
    print(f"\n{'='*60}")
    print(f"因果分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"发现 {len(global_causal_edges)} 个变量对 {target_var} 有因果影响")
    print(f"{'='*60}")
    
    return {
        'global_causal_graph': global_causal_edges,
        'point_causal_graphs': all_point_data,
        'var_names': var_names,
        'target_var': target_var
    }


def run_causal_analysis_all(model_path, nc_file, output_dir, predictors, target_var, lulc_ids,
                             seq_len=config["model"]["SEQ_LEN"], output_window=config["model"].get("output_window", 1), 
                             batch_size=config["analyze"]["BATCH_SIZE"], m=config["analyze"]["KMeans_M"], n=config["analyze"]["KMeans_N"]):
    """
    全变量因果分析函数，分析所有变量之间的因果关系
    
    参考 code/CausalFormer/interpret.py 中的实现，生成 (cause, effect, lag) 格式的因果图
    
    Args:
        model_path: 训练好的模型路径 (.pth 文件)
        nc_file: netCDF 数据文件路径
        output_dir: 输出目录
        predictors: 预测变量列表
        target_var: 目标变量名称
        lulc_ids: 土地利用类型 ID 列表
        seq_len: 序列长度
        output_window: 输出窗口大小
        batch_size: 批处理大小
        m: 选择top m个聚类簇
        n: KMeans聚类数
    
    Returns:
        all_results: 因果分析结果字典
    """
    configure_torch_runtime()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"全变量因果分析模式 - 分析所有变量之间的因果关系")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"数据文件: {nc_file}")
    print(f"KMeans参数: m={m}, n={n}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据
    ds, lat, lon, time = load_nc_data(nc_file)
    
    # 获取辅助变量配置 (如果存在)
    aux_predictors = config["analyze"].get("AUX_PREDICTORS", [])
    static_vars_list = config["analyze"].get("STATIC_VARS", [])
    
    data, aux_data, static_data, coords = preprocess_geo_data(
        ds, predictors, target_var, lat, lon, time, lulc_ids, 
        return_coords=True,
        aux_predictors=aux_predictors,
        static_vars=static_vars_list
    )
    
    # 确定变量名称列表
    var_names = predictors.copy()
    if target_var not in predictors:
        var_names.append(target_var)
    
    print(f"变量列表: {var_names}")
    print(f"地理点数量: {len(coords)}")
    
    # 标准化主数据
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    N, T, V = data.shape
    data = scaler.fit_transform(data.reshape(-1, V)).reshape(N, T, V)
    
    # 标准化辅助数据
    if aux_data is not None:
        aux_scaler = MinMaxScaler(feature_range=(0.1, 1))
        V_aux = aux_data.shape[2]
        aux_data = aux_scaler.fit_transform(aux_data.reshape(-1, V_aux)).reshape(N, T, V_aux)
    
    # 标准化静态数据
    if static_data is not None:
        static_scaler = MinMaxScaler(feature_range=(0.1, 1))
        static_data = static_scaler.fit_transform(static_data)
    
    # 2. 加载模型
    cfg = {
        'n_gpu': ls_id, 
        'data_loader': {
            'args': {
                'time_step': seq_len, 
                'output_window': output_window, 
                'series_num': len(var_names), 
                'feature_dim': 1, 
                'output_dim': 1
            }
        }
    }
    
    device = prepare_device(ls_id)
    
    # 检测是否使用地理编码 (从配置文件读取)
    use_geo_encoding = config["train"].get("USE_GEO_ENCODING", False)
    
    model = PredictModel(cfg, d_model=config["model"]["d_model"], n_head=config["model"]["n_head"], n_layers=config["model"]["n_layers"], 
                         ffn_hidden=config["model"]["hidden_layers"], drop_prob=config["model"]["drop_prob"], tau=config["model"]["tau"],
                         use_geo_encoding=use_geo_encoding,
                         aux_series_num=len(aux_predictors) if aux_predictors else 0,
                         static_dim=len(static_vars_list) if static_vars_list else 0).to(device)
    
    if use_geo_encoding:
        print(f"  模型已启用地理位置编码 (GeoPositionalEncoding)")
    if aux_predictors:
        print(f"  辅助时序变量: {aux_predictors}")
    if static_vars_list:
        print(f"  静态变量: {static_vars_list}")
    
    print(f"\n正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'rmse' in checkpoint:
        print(f"  RMSE: {checkpoint['rmse']:.6f}")
    
    # 3. 参数设置
    series_num = len(var_names)
    time_step = seq_len
    
    n = min(n, series_num)
    m = min(m, n - 1) if n > 1 else 1
    print(f"调整后的 KMeans 参数: m={m}, n={n}")
    
    # 4. 为每个地理点生成全变量因果图
    print(f"\n正在为 {len(coords)} 个地理点分析所有变量间的因果关系...")
    
    all_point_data = []
    
    # 累积全局 relA 和 relK
    global_relA = [np.zeros(series_num) for _ in range(series_num)]
    global_relK = [np.zeros((series_num, time_step)) for _ in range(series_num)]
    valid_points = 0
    
    num_workers = _env_int("DTLN_NUM_WORKERS", config["analyze"].get("NUM_WORKERS", 1))
    enable_experimental_mp = _env_flag("DTLN_ENABLE_EXPERIMENTAL_MULTIPROCESS", False)
    if num_workers > 1 and not enable_experimental_mp:
        print(
            "DTLN_NUM_WORKERS 已设置，但多进程点位并行仍为实验功能；"
            "当前改用稳定单进程。若要测试: DTLN_ENABLE_EXPERIMENTAL_MULTIPROCESS=1"
        )
    if num_workers > 1 and enable_experimental_mp and device.type == "cpu":
        worker_threads = _env_int("DTLN_WORKER_THREADS", max(1, (os.cpu_count() or 1) // num_workers))
        chunk_size = _env_int("DTLN_WORKER_CHUNKSIZE", 8)
        print(
            f"启用CPU多进程点位并行: workers={num_workers}, "
            f"worker_threads={worker_threads}, chunksize={chunk_size}"
        )

        global _CAUSAL_WORKER_STATE
        _CAUSAL_WORKER_STATE = {
            "base_config": config,
            "model_cfg": cfg,
            "data": data,
            "aux_data": aux_data,
            "static_data": static_data,
            "coords": coords,
            "seq_len": seq_len,
            "time_step": time_step,
            "var_names": var_names,
            "series_num": series_num,
            "m": m,
            "n": n,
            "use_geo_encoding": use_geo_encoding,
            "aux_series_num": len(aux_predictors) if aux_predictors else 0,
            "static_dim": len(static_vars_list) if static_vars_list else 0,
        }
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_causal_point_worker,
            initargs=(model_path, worker_threads),
        ) as executor:
            iterator = executor.map(_causal_point_worker, range(len(coords)), chunksize=chunk_size)
            for result in tqdm(iterator, total=len(coords), desc="全变量因果分析"):
                all_point_data.append(result["point_result"])
                if result["valid"]:
                    valid_points += 1
                    for interpreted_series in range(series_num):
                        global_relA[interpreted_series] += result["relA"][interpreted_series]
                        global_relK[interpreted_series] += result["relK"][interpreted_series]
    else:
        if num_workers > 1 and device.type != "cpu":
            print("检测到非CPU设备，DTLN_NUM_WORKERS 多进程点位并行已禁用。")

        for point_idx in tqdm(range(len(coords)), desc="全变量因果分析"):
            coord = coords[point_idx]
            point_data = data[point_idx]  # (T, V)
            
            # 生成输入样本
            samples = []
            for t in range(seq_len, T + 1):
                sample = point_data[t-seq_len:t].reshape(seq_len, V, 1)
                samples.append(sample)
            
            if len(samples) == 0:
                continue
            
            input_tensor = torch.FloatTensor(np.array(samples)).to(device)
            # print(input_tensor.shape)
            input_tensor.requires_grad = True

            aux_tensor = None
            if aux_data is not None:
                point_aux = aux_data[point_idx]
                aux_samples = []
                for t in range(seq_len, point_aux.shape[0] + 1):
                    aux_sample = point_aux[t - seq_len:t].reshape(seq_len, point_aux.shape[1], 1)
                    aux_samples.append(aux_sample)
                if aux_samples:
                    aux_tensor = torch.FloatTensor(np.array(aux_samples)).to(device)

            static_tensor = None
            if static_data is not None:
                batch_size = input_tensor.shape[0]
                static_point = np.asarray(static_data[point_idx], dtype=float)
                static_tensor = torch.FloatTensor(np.repeat(static_point[None, :], batch_size, axis=0)).to(device)
            
            # 准备坐标张量 (如果启用地理编码)
            if use_geo_encoding:
                batch_size = input_tensor.shape[0]
                lat_tensor = torch.full((batch_size,), coord['lat'], dtype=torch.float32).to(device)
                lon_tensor = torch.full((batch_size,), coord['lon'], dtype=torch.float32).to(device)
            else:
                lat_tensor, lon_tensor = None, None
            
            point_relA = []
            point_relK = []
            
            try:
                # 为每个目标变量生成因果分数
                for interpreted_series in range(series_num):
                    enable_debug = (
                        _env_flag("DTLN_DEBUG_FIRST_POINT", False)
                        and point_idx == 0
                        and interpreted_series == 0
                    )
                    relA, relK_aligned = generate_RRP_scores_all(
                        model,
                        input_tensor,
                        interpreted_series,
                        device,
                        debug=enable_debug,
                        lat=lat_tensor,
                        lon=lon_tensor,
                        aux_data=aux_tensor,
                        static_vars=static_tensor,
                    )
                    
                    # relA[interpreted_series] = 所有变量对该变量的影响
                    relA_i = relA[interpreted_series].detach().cpu().numpy()  # (series_num,)
                    
                    if torch.is_tensor(relK_aligned):
                        relK_i = relK_aligned.detach().cpu().numpy()  # (series_num, time_step)
                    else:
                        relK_i = relK_aligned
                    
                    point_relA.append(relA_i)
                    point_relK.append(relK_i)
                    
                    # 累加到全局
                    global_relA[interpreted_series] += relA_i
                    global_relK[interpreted_series] += relK_i
                
                valid_points += 1
                
                # 生成该点的全变量因果图
                causal_edges = analyze_all_causes(point_relA, point_relK, m, n, time_step, var_names)
                
            except Exception as e:
                print(f"  点 {point_idx} 处理失败: {e}")
                causal_edges = []
            
            point_result = {
                'point_id': point_idx,
                'lat': coord['lat'],
                'lon': coord['lon'],
                'lat_idx': coord['lat_idx'],
                'lon_idx': coord['lon_idx'],
                'causal_edges': causal_edges,
                'num_edges': len(causal_edges)
            }
            all_point_data.append(point_result)
    
    # 5. 计算全局因果图
    if valid_points > 0:
        global_relA = [ra / valid_points for ra in global_relA]
        global_relK = [rk / valid_points for rk in global_relK]
    
    global_causal_edges = analyze_all_causes(global_relA, global_relK, m, n, time_step, var_names)
    
    print(f"\n调试信息:")
    print(f"  有效点数: {valid_points}")
    print(f"  全局因果边数: {len(global_causal_edges)}")
    
    # 6. 保存结果
    # 6.1 保存为 JSON
    results_json = {
        'metadata': {
            'model_path': model_path,
            'nc_file': nc_file,
            'var_names': var_names,
            'seq_len': seq_len,
            'output_window': output_window,
            'n_points': valid_points,
            'kmeans_m': m,
            'kmeans_n': n,
            'lag_selection': _lag_selection_cfg(),
            'analysis_type': 'ALL variables'
        },
        'global_causal_graph': {
            'description': '全局因果图: 所有变量之间',
            'edges': global_causal_edges
        },
        'point_causal_graphs': all_point_data
    }
    
    json_path = os.path.join(output_dir, 'causal_graph_all.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"\n因果图已保存: {json_path}")
    
    # 6.2 保存全局因果图为 CSV
    global_csv_path = os.path.join(output_dir, 'global_causal_graph_all.csv')
    if len(global_causal_edges) > 0:
        df_global = pd.DataFrame(global_causal_edges)
    else:
        df_global = pd.DataFrame(columns=['cause_idx', 'cause_name', 'effect_idx', 'effect_name', 'lag', 'score'] + LAG_DIAGNOSTIC_FIELDS)
    df_global.to_csv(global_csv_path, index=False)
    print(f"全局因果图 CSV: {global_csv_path}")
    
    # 6.3 保存每个点的因果图
    all_edges_data = []
    for pt in all_point_data:
        for edge in pt['causal_edges']:
            edge_record = {
                'point_id': pt['point_id'],
                'lat': pt['lat'],
                'lon': pt['lon'],
                'cause_idx': edge['cause_idx'],
                'cause_name': edge['cause_name'],
                'effect_idx': edge['effect_idx'],
                'effect_name': edge['effect_name'],
                'lag': edge['lag'],
                'score': edge.get('score', 0)
            }
            for field in LAG_DIAGNOSTIC_FIELDS:
                edge_record[field] = edge.get(field)
            all_edges_data.append(edge_record)
    
    points_csv_path = os.path.join(output_dir, 'point_causal_graphs_all.csv')
    if len(all_edges_data) > 0:
        df_points = pd.DataFrame(all_edges_data)
    else:
        df_points = pd.DataFrame(columns=['point_id', 'lat', 'lon', 'cause_idx', 'cause_name', 'effect_idx', 'effect_name', 'lag', 'score'] + LAG_DIAGNOSTIC_FIELDS)
    df_points.to_csv(points_csv_path, index=False)
    print(f"点位因果图 CSV: {points_csv_path}")
    save_lag_diagnostics(all_edges_data, output_dir, time_step, suffix="all")
    
    save_global_plots = _env_flag("DTLN_SAVE_GLOBAL_PLOTS", True)
    save_point_graphs = _env_flag("DTLN_SAVE_POINT_GRAPHS", False)
    max_point_graphs = _env_int("DTLN_MAX_POINT_GRAPHS", 50)
    save_point_jsons = _env_flag("DTLN_SAVE_POINT_JSONS", True)

    # 7. 可视化全变量因果图
    if save_global_plots and len(global_causal_edges) > 0:
        plot_all_causal_graph(global_causal_edges, var_names, output_dir)
    
    # 8. 为每个点生成因果图和 JSON
    point_graphs_dir = os.path.join(output_dir, 'point_graphs')
    point_json_dir = os.path.join(output_dir, 'point_jsons')
    if save_point_graphs:
        os.makedirs(point_graphs_dir, exist_ok=True)
    if save_point_jsons:
        os.makedirs(point_json_dir, exist_ok=True)
    print(f"\n正在生成点位输出: JSON={save_point_jsons}, PNG={save_point_graphs}")
    
    graph_count = 0
    iterator_desc = "生成点位JSON/图" if save_point_graphs else "生成点位JSON"
    for pt in tqdm(all_point_data, desc=iterator_desc):
        lat = pt['lat']
        lon = pt['lon']
        point_id = pt['point_id']
        causal_edges = pt['causal_edges']
        
        # 以经纬度为名称 (lat_lon 格式)
        coord_name = f"lat{lat:.4f}_lon{lon:.4f}"
        
        if len(causal_edges) > 0:
            # 绘制因果图
            if save_point_graphs and graph_count < max_point_graphs:
                plot_point_all_causal_graph(
                    causal_edges,
                    var_names,
                    point_id,
                    lat,
                    lon,
                    point_graphs_dir
                )
                graph_count += 1
            
            # 保存 JSON (以经纬度命名)
            if save_point_jsons:
                point_json = {
                    'point_id': point_id,
                    'lat': lat,
                    'lon': lon,
                    'lat_idx': pt['lat_idx'],
                    'lon_idx': pt['lon_idx'],
                    'coord_name': coord_name,
                    'num_edges': len(causal_edges),
                    'causal_edges': causal_edges
                }
                json_filename = f"{coord_name}.json"
                with open(os.path.join(point_json_dir, json_filename), 'w', encoding='utf-8') as f:
                    json.dump(point_json, f, ensure_ascii=False, indent=2)
    
    if save_point_graphs:
        print(f"点位因果图已保存到: {point_graphs_dir} (最多 {max_point_graphs} 张)")
    else:
        print("已跳过点位 PNG 绘图。如需开启: DTLN_SAVE_POINT_GRAPHS=1")
    if save_point_jsons:
        print(f"点位 JSON 已保存到: {point_json_dir}")
    else:
        print("已跳过点位 JSON。如需开启: DTLN_SAVE_POINT_JSONS=1")
    
    print(f"\n{'='*60}")
    print(f"全变量因果分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"发现 {len(global_causal_edges)} 个因果关系")
    print(f"{'='*60}")
    
    return {
        'global_causal_graph': global_causal_edges,
        'point_causal_graphs': all_point_data,
        'var_names': var_names
    }


def generate_RRP_scores_all(model, input_data, interpreted_series, device, debug=False, 
                              lat=None, lon=None, aux_data=None, static_vars=None):
    """
    为指定目标变量生成 RRP 因果分数
    
    完全参考 code/CausalFormer/explainer/explainer.py
    论文: https://arxiv.org/html/2406.16708v1
    
    重要说明:
    - RRP 只分析主要变量 (PREDICTORS) 之间的因果关系
    - 辅助变量 (AUX_PREDICTORS, STATIC_VARS) 通过 FiLM 机制在模型内部处理
    - 辅助变量不参与 RRP 因果分数的计算
    
    Args:
        model: 模型
        input_data: 主输入数据 (PREDICTORS + TARGET)
        interpreted_series: 目标序列索引
        device: 设备
        debug: 是否输出调试信息
        lat: 纬度张量 (batch,) - 当模型启用地理编码时需要
        lon: 经度张量 (batch,) - 当模型启用地理编码时需要
        aux_data: 辅助时序数据 - 当模型使用辅助变量时需要
        static_vars: 静态变量数据 - 当模型使用静态变量时需要
    
    Returns:
        relA: 注意力因果分数 (series_num, series_num)
        relK_aligned: 卷积核因果分数 (series_num, time_step)
    """
    model.eval()
    
    # 前向传播 (包含辅助变量)
    output = model(input_data, lat, lon, aux_data, static_vars)
    
    # 创建 one-hot 张量 (参考 explainer.py 第35-36行)
    one_hot = torch.zeros_like(output, dtype=torch.float).to(device)
    one_hot[:, :, interpreted_series, :] = 1
    
    one_hot_vector = one_hot.clone()
    one_hot.requires_grad_(True)
    
    # 计算目标输出
    one_hot_sum = torch.sum(one_hot * output)
    
    # 反向传播
    model.zero_grad()
    one_hot_sum.backward(retain_graph=True)
    
    # 应用 RRP
    model.relprop(one_hot_vector)
    
    relAs = []
    relKs = []  # 修正: 使用 relK 而非 kernel weights
    
    # 从每个 encoder 层收集因果分数 (参考 explainer.py 第50-62行)
    for layer_idx, layer in enumerate(model.encoder.layers):
        # 梯度调制 - relA (参考 explainer.py 第52行)
        relA_raw = layer.attention.attention.get_rel()
        relA_grad = layer.attention.attention.get_grad()
        
        if relA_raw is None or relA_grad is None:
            if debug:
                print(f"  警告: Layer {layer_idx} relA_raw={relA_raw is not None}, relA_grad={relA_grad is not None}")
            continue
            
        relA = relA_raw * torch.abs(relA_grad)
        # 不再使用 clamp(0)，保留负因果（抑制作用）
        relA = relA.clamp(min=0)
        
        # 对每层进行标准化，消除层间数值差异
        relA_mean_head = relA.mean((0, 1))  # mean for sample and head
        # Z-score 标准化
        ra_mean = relA_mean_head.mean()
        ra_std = relA_mean_head.std()
        if ra_std > 0:
            relA_normalized = (relA_mean_head - ra_mean) / ra_std
        else:
            relA_normalized = relA_mean_head
        relAs.append(relA_normalized)
        
        # 梯度调制 - relK (参考 explainer.py 第53行)
        # 关键修正: 使用 get_rel() * abs(get_grad()) 而非 get_wgt()
        relK_raw = layer.attention.Wv.get_rel()
        relK_grad = layer.attention.Wv.get_grad()
        
        if relK_raw is None or relK_grad is None:
            if debug:
                print(f"  警告: Layer {layer_idx} relK_raw={relK_raw is not None}, relK_grad={relK_grad is not None}")
            continue
        
        if debug and layer_idx == 0:
            print(f"  relK_raw shape: {relK_raw.shape}")
            print(f"  relK_grad shape: {relK_grad.shape}")
            # 检查原始值分布
            print(f"  relK_raw sum: {relK_raw.sum().item():.6f}, max: {relK_raw.max().item():.6f}")
            print(f"  relK_grad sum: {relK_grad.abs().sum().item():.6f}, max: {relK_grad.abs().max().item():.6f}")
            
        relK = relK_raw * torch.abs(relK_grad)
        # 不再使用 clamp(0)，保留负因果（抑制作用）
        relK = relK.clamp(min=0)
        
        # 对每层进行标准化，消除层间数值差异
        relK_mean_head = relK.mean(0)  # mean for head
        # Z-score 标准化：(x - mean) / std
        rk_mean = relK_mean_head.mean()
        rk_std = relK_mean_head.std()
        if rk_std > 0:
            relK_normalized = (relK_mean_head - rk_mean) / rk_std
        else:
            relK_normalized = relK_mean_head
        relKs.append(relK_normalized)
        
        if debug:
            print(f"  Layer {layer_idx}: relK sum={relK.sum().item():.4f}, normalized mean={relK_normalized.mean().item():.4f}, std={relK_normalized.std().item():.4f}")
    
    if len(relAs) == 0 or len(relKs) == 0:
        # 如果没有收集到任何数据，返回零矩阵
        series_num = model.encoder.layers[0].attention.Wv.series_num
        time_step = model.encoder.layers[0].attention.Wv.input_window
        return torch.zeros((series_num, series_num)), torch.zeros((series_num, time_step))
    
    # 沿编码器层维度相乘 (参考 explainer.py 第63-64行)
    if debug:
        print(f"  Number of encoder layers: {len(relKs)}")
        for i, rk in enumerate(relKs):
            print(f"  Before mean - Layer {i}: mean={rk.mean().item():.4f}, std={rk.std().item():.4f}")
    
    relA = torch.stack(relAs).mean(0)  # 改用 mean 聚合 (series_num, series_num)
    # 使用 mean(0) 聚合多层，捕捉平均因果效应
    relK = torch.stack(relKs).mean(0)  # (series_num, series_num, time_step, time_step)
    
    if debug:
        print(f"  After mean: relK mean={relK.mean().item():.4f}, std={relK.std().item():.4f}")
    
    series_num = relA.shape[0]
    time_step = model.encoder.layers[0].attention.Wv.input_window
    
    if debug:
        print(f"  relK final shape: {relK.shape}")
        print(f"  relK sum: {relK.sum().item():.6f}, max: {relK.max().item():.6f}")
        # 检查特定位置的值
        print(f"  relK[:, {interpreted_series}, -1, :] sum: {relK[:, interpreted_series, -1, :].sum().item():.6f}")
        print(f"  relK[:, {interpreted_series}, -2, :] sum: {relK[:, interpreted_series, -2, :].sum().item():.6f}")
        print(f"  relK[:, {interpreted_series}, 0, :] sum: {relK[:, interpreted_series, 0, :].sum().item():.6f}")
    
    # 提取 relK_aligned (参考 interpret.py 第112-115行)
    relK_np = relK.detach().cpu().numpy()
    
    # relK_aligned = relK[:, interpreted_series, -1, :]
    # 形状: (series_num, series_num, time_step, time_step)
    # [:, interpreted_series, -1, :] -> (series_num, time_step)
    relK_aligned = deepcopy(relK_np[:, interpreted_series, -1, :])  # (series_num, time_step)
    
    # 修正对角线: 变量自身对自身用倒数第二步 (参考 interpret.py 第114行)
    # The relK[i][i][-1] is zero vector due to the time_step th data can not be used to predict itself
    relK_aligned[interpreted_series, :] = relK_np[interpreted_series, interpreted_series, -2, :]
    
    if debug:
        print(f"  relK_aligned shape: {relK_aligned.shape}")
        # 检查 relK_aligned 的分布
        for j in range(min(3, series_num)):
            rk = relK_aligned[j]
            if np.sum(np.abs(rk)) > 0:
                max_idx = np.argmax(np.abs(rk))
                lag = time_step - 1 - max_idx
                print(f"    Var {j}: argmax={max_idx}, lag={lag}, sum={np.sum(np.abs(rk)):.6f}")
            else:
                print(f"    Var {j}: all zeros")
    
    return relA, torch.tensor(relK_aligned)


def analyze_all_causes(relA_list, relK_list, m, n, time_step, var_names):
    """
    分析所有变量之间的因果关系
    
    Args:
        relA_list: 每个目标变量的因果分数列表
        relK_list: 每个目标变量的时间滞后列表
        m: 选择 top m 个聚类簇
        n: KMeans 聚类数
        time_step: 时间步数
        var_names: 变量名称列表
    
    Returns:
        causal_edges: 因果边列表
    """
    causal_edges = []
    series_num = len(relA_list)
    
    if series_num == 0:
        return causal_edges
    
    n_actual = min(n, series_num)
    m_actual = min(m, n_actual - 1) if n_actual > 1 else 1
    
    # 为每个目标变量查找原因
    for i, relA_i in enumerate(relA_list):
        if not isinstance(relA_i, np.ndarray):
            relA_i = np.array(relA_i)
        
        if len(relA_i) == 0 or relA_i.sum() == 0.0:
            continue
        
        relK_i = relK_list[i] if i < len(relK_list) else np.zeros((series_num, time_step))
        
        if n_actual < 2:
            threshold = np.mean(relA_i) + np.std(relA_i)
            for j in range(len(relA_i)):
                if relA_i[j] > threshold and i != j:  # 排除自己对自己
                    lag_info = compute_lag_details(
                        relK_i, j, time_step,
                        cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                        effect_name=var_names[i] if i < len(var_names) else f'Var_{i}'
                    )
                    causal_edges.append({
                        'cause_idx': int(j),
                        'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                        'effect_idx': int(i),
                        'effect_name': var_names[i] if i < len(var_names) else f'Var_{i}',
                        'lag': int(lag_info['lag']),
                        'score': float(relA_i[j]),
                        **lag_info
                    })
        else:
            try:
                data = relA_i.reshape(-1, 1)
                estimator = KMeans(n_clusters=n_actual, random_state=42, n_init=10)
                estimator.fit(data)
                cluster_labels = estimator.labels_
                cluster_centers = estimator.cluster_centers_.reshape(-1)
                
                largest_m_clusters = np.argsort(cluster_centers)[-m_actual:]
                
                for j in range(len(relA_i)):
                    if cluster_labels[j] in largest_m_clusters and i != j:  # 排除自己对自己
                        lag_info = compute_lag_details(
                            relK_i, j, time_step,
                            cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                            effect_name=var_names[i] if i < len(var_names) else f'Var_{i}'
                        )
                        causal_edges.append({
                            'cause_idx': int(j),
                            'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                            'effect_idx': int(i),
                            'effect_name': var_names[i] if i < len(var_names) else f'Var_{i}',
                            'lag': int(lag_info['lag']),
                            'score': float(relA_i[j]),
                            **lag_info
                        })
            except Exception as e:
                pass
    
    causal_edges.sort(key=lambda x: x['score'], reverse=True)
    
    # 归一化分数到 0-1 范围
    # causal_edges = normalize_causal_scores(causal_edges)
    
    return causal_edges


def normalize_causal_scores(causal_edges):
    """
    将因果分数归一化到 0-1 范围
    
    Args:
        causal_edges: 因果边列表
    
    Returns:
        causal_edges: 归一化后的因果边列表
    """
    if len(causal_edges) == 0:
        return causal_edges
    
    scores = [edge['score'] for edge in causal_edges]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # 所有分数相同，设为 1.0
        for edge in causal_edges:
            edge['score'] = 1.0
            edge['score_normalized'] = 1.0
    else:
        for edge in causal_edges:
            normalized = (edge['score'] - min_score) / (max_score - min_score)
            edge['score_original'] = edge['score']  # 保留原始分数
            edge['score'] = round(normalized, 4)  # 归一化分数
    
    return causal_edges


def plot_all_causal_graph(causal_edges, var_names, output_dir):
    """
    绘制全变量因果图 - SCI 科研绘图标准
    
    Args:
        causal_edges: 因果边列表
        var_names: 变量名称列表
        output_dir: 输出目录
    """
    import networkx as nx
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.linewidth'] = 1.5
    
    G = nx.DiGraph()
    
    for name in var_names:
        G.add_node(name)
    
    edge_labels = {}
    for edge in causal_edges:
        cause = edge['cause_name']
        effect = edge['effect_name']
        lag = edge['lag']
        score = edge.get('score', 0)
        G.add_edge(cause, effect, weight=score)
        edge_labels[(cause, effect)] = f'{lag}'
    
    if len(G.edges()) == 0:
        return
    
    # 绘图 - SCI 标准
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # 节点颜色根据入度/出度
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    node_colors = []
    for node in G.nodes():
        if out_degrees[node] > in_degrees[node]:
            node_colors.append('#E74C3C')  # 红色 - 主要是原因
        elif in_degrees[node] > out_degrees[node]:
            node_colors.append('#27AE60')  # 绿色 - 主要是结果
        else:
            node_colors.append('#F39C12')  # 橙色 - 平衡
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=2500, alpha=0.9, edgecolors='black', linewidths=1.5)
    
    # 边宽度根据分数
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if len(edge_weights) > 0:
        max_w = max(edge_weights) if max(edge_weights) > 0 else 1
        edge_widths = [1.5 + 2.5 * w / max_w for w in edge_weights]
    else:
        edge_widths = [1.5]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#2C3E50', arrows=True, 
                           arrowsize=20, width=edge_widths, 
                           connectionstyle='arc3,rad=0.1', alpha=0.8)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold', font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9, font_family='Arial')
    
    ax.set_title('Causal Graph: All Variables\n(Red: Cause, Green: Effect, Orange: Both)', 
                 fontsize=14, fontweight='bold', fontfamily='Arial')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_causal_graph.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"全变量因果图已保存: {os.path.join(output_dir, 'all_causal_graph.png')}")
    
    # 绘制因果矩阵热力图 - SCI 标准
    series_num = len(var_names)
    causal_matrix = np.zeros((series_num, series_num))
    lag_matrix = np.zeros((series_num, series_num))
    
    for edge in causal_edges:
        i = edge['cause_idx']
        j = edge['effect_idx']
        if i < series_num and j < series_num:
            causal_matrix[i, j] = edge['score']
            lag_matrix[i, j] = edge['lag']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    
    # 因果分数热力图
    sns.heatmap(causal_matrix, ax=ax1, xticklabels=var_names, yticklabels=var_names,
                cmap='RdBu_r', annot=True, fmt='.2f', annot_kws={'size': 10, 'fontweight': 'bold'},
                cbar_kws={'label': 'Causal Score (Normalized)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax1.set_title('Causal Score Matrix (Row → Column)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Effect Variable', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cause Variable', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)
    
    # 时间滞后热力图
    sns.heatmap(lag_matrix, ax=ax2, xticklabels=var_names, yticklabels=var_names,
                cmap='YlOrRd', annot=True, fmt='.0f', annot_kws={'size': 10, 'fontweight': 'bold'},
                cbar_kws={'label': 'Time Lag (steps)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax2.set_title('Time Lag Matrix (Row → Column)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Effect Variable', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cause Variable', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', rotation=0, labelsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_causal_heatmaps.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"全变量因果热力图已保存: {os.path.join(output_dir, 'all_causal_heatmaps.png')}")


def plot_point_all_causal_graph(causal_edges, var_names, point_id, lat, lon, output_dir):
    """
    为单个点绘制全变量因果图 - SCI 科研绘图标准
    
    Args:
        causal_edges: 因果边列表
        var_names: 变量名称列表
        point_id: 点ID
        lat: 纬度
        lon: 经度
        output_dir: 输出目录
    """
    import networkx as nx
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.linewidth'] = 1.2
    
    if len(causal_edges) == 0:
        return
    
    # 创建有向图
    G = nx.DiGraph()
    
    for name in var_names:
        G.add_node(name)
    
    edge_labels = {}
    for edge in causal_edges:
        cause = edge['cause_name']
        effect = edge['effect_name']
        lag = edge['lag']
        score = edge.get('score', 0)
        G.add_edge(cause, effect, weight=score)
        edge_labels[(cause, effect)] = f'{lag}'
    
    if len(G.edges()) == 0:
        return
    
    # 绘图 - SCI 标准
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    
    # 左图: 因果网络图
    pos = nx.spring_layout(G, k=2.5, iterations=30, seed=42)
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    node_colors = []
    for node in G.nodes():
        if out_degrees[node] > in_degrees[node]:
            node_colors.append('#E74C3C')  # 红色 - 主要是原因
        elif in_degrees[node] > out_degrees[node]:
            node_colors.append('#27AE60')  # 绿色 - 主要是结果
        else:
            node_colors.append('#F39C12')  # 橙色 - 平衡
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=1800, 
                           alpha=0.9, edgecolors='black', linewidths=1.2)
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if len(edge_weights) > 0:
        max_w = max(edge_weights) if max(edge_weights) > 0 else 1
        edge_widths = [1.2 + 2 * w / max_w for w in edge_weights]
    else:
        edge_widths = [1.2]
    
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#2C3E50', arrows=True, arrowsize=15,
                           width=edge_widths, connectionstyle='arc3,rad=0.1', alpha=0.8)
    
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold', font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=7, font_family='Arial')
    
    ax1.set_title(f'Point {point_id}\nLat: {lat:.4f}, Lon: {lon:.4f}', 
                  fontsize=12, fontweight='bold', fontfamily='Arial')
    ax1.axis('off')
    
    # 右图: 因果矩阵热力图
    series_num = len(var_names)
    causal_matrix = np.zeros((series_num, series_num))
    
    for edge in causal_edges:
        i = edge['cause_idx']
        j = edge['effect_idx']
        if i < series_num and j < series_num:
            causal_matrix[i, j] = edge['score']
    
    sns.heatmap(causal_matrix, ax=ax2, xticklabels=var_names, yticklabels=var_names,
                cmap='RdBu_r', annot=True, fmt='.2f', 
                annot_kws={'size': 8, 'fontweight': 'bold'},
                cbar_kws={'label': 'Causal Score (Normalized)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax2.set_title('Causal Score Matrix (Row → Column)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Effect Variable', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Cause Variable', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    
    # 以经纬度命名
    coord_name = f"lat{lat:.4f}_lon{lon:.4f}"
    plt.savefig(os.path.join(output_dir, f'{coord_name}.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def analyze_target_causes(relA, relK, m, n, time_step, var_names, target_var, target_idx):
    """
    分析对 TARGET 变量的因果关系
    
    Args:
        relA: (series_num,) 数组，所有变量对 TARGET 的因果分数
        relK: (series_num, time_step) 数组，时间滞后信息
        m: 选择 top m 个聚类簇
        n: KMeans 聚类数
        time_step: 时间步数
        var_names: 变量名称列表
        target_var: 目标变量名称
        target_idx: 目标变量索引
    
    Returns:
        causal_edges: 因果边列表 [{cause_idx, cause_name, effect_idx, effect_name, lag, score}, ...]
    """
    causal_edges = []
    
    if not isinstance(relA, np.ndarray):
        relA = np.array(relA)
    
    series_num = len(relA)
    
    if series_num == 0 or relA.sum() == 0.0:
        return causal_edges
    
    # 使用 KMeans 聚类选择重要的因果关系 (参考 interpret.py)
    n_actual = min(n, series_num)
    m_actual = min(m, n_actual - 1) if n_actual > 1 else 1
    
    if n_actual < 2:
        # 如果聚类数太小，使用阈值方法
        threshold = np.mean(relA) + np.std(relA)
        for j in range(series_num):
            if relA[j] > threshold:
                lag_info = compute_lag_details(
                    relK, j, time_step,
                    cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                    effect_name=target_var
                )
                causal_edges.append({
                    'cause_idx': int(j),
                    'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                    'effect_idx': int(target_idx),
                    'effect_name': target_var,
                    'lag': int(lag_info['lag']),
                    'score': float(relA[j]),
                    **lag_info
                })
        return causal_edges
    
    try:
        # KMeans 聚类
        data = relA.reshape(-1, 1)
        estimator = KMeans(n_clusters=n_actual, random_state=42, n_init=10)
        estimator.fit(data)
        cluster_labels = estimator.labels_
        cluster_centers = estimator.cluster_centers_.reshape(-1)
        
        # 找到最大的 m 个聚类
        largest_m_clusters = np.argsort(cluster_centers)[-m_actual:]
        
        for j in range(series_num):
            if cluster_labels[j] in largest_m_clusters:
                lag_info = compute_lag_details(
                    relK, j, time_step,
                    cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                    effect_name=target_var
                )
                causal_edges.append({
                    'cause_idx': int(j),
                    'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                    'effect_idx': int(target_idx),
                    'effect_name': target_var,
                    'lag': int(lag_info['lag']),
                    'score': float(relA[j]),
                    **lag_info
                })
    except Exception as e:
        print(f"  聚类失败: {e}")
    
    # 按分数排序
    causal_edges.sort(key=lambda x: x['score'], reverse=True)
    
    # 归一化分数到 0-1 范围
    causal_edges = normalize_causal_scores(causal_edges)
    
    return causal_edges


def _lag_selection_cfg():
    return {
        **DEFAULT_LAG_SELECTION_CONFIG,
        **config.get("analyze", {}).get("lag_selection", {}),
    }


def _is_vegetation_response(effect_name):
    if not effect_name:
        return False
    name = str(effect_name).lower()
    return any(token in name for token in ("gpp", "gosif", "sif", "evi", "ndvi", "fpar", "lai"))


def _effective_max_lag(time_step, cause_name=None, effect_name=None):
    cfg = _lag_selection_cfg()
    max_lag = int(time_step) - 1
    configured = cfg.get("max_effective_lag", None)
    if configured is not None:
        max_lag = min(max_lag, int(configured))
    elif cfg.get("use_vegetation_lag_prior", True) and _is_vegetation_response(effect_name):
        max_lag = min(max_lag, int(cfg.get("vegetation_max_lag", 12)))
    return max(0, max_lag)


def _smooth_lag_spectrum(values, window):
    values = np.asarray(values, dtype=float)
    window = int(window)
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def compute_lag_details(relK, j, time_step, cause_name=None, effect_name=None):
    """
    Compute a robust lag from the relevance spectrum.

    The old implementation used a single argmax over |relK[j]|. That is
    fragile for seasonal/autocorrelated ecological series because small
    differences can push the peak to the maximum lag. This version smooths
    the spectrum, applies a weak temporal-priority penalty, and picks the
    shortest lag among near-optimal candidates.

    Args:
        relK: (series_num, time_step) 数组，或 torch.Tensor
        j: 原因变量索引
        time_step: 时间步数
        cause_name: 原因变量名，用于输出诊断
        effect_name: 结果变量名，用于生态滞后先验
    
    Returns:
        dict: lag 及其诊断信息
    """
    fallback = {
        "lag": 0,
        "raw_lag": 0,
        "lag_confidence": 0.0,
        "lag_peak_raw": 0.0,
        "lag_peak_adjusted": 0.0,
        "max_effective_lag": max(0, int(time_step) - 1),
        "boundary_warning": False,
        "raw_boundary_warning": False,
        "seasonal_lag_warning": False,
        "lag_selection_method": "fallback_zero",
    }

    try:
        if torch.is_tensor(relK):
            relK = relK.detach().cpu().numpy()
        elif isinstance(relK, list):
            relK = np.array(relK)

        if isinstance(relK, np.ndarray):
            if relK.ndim == 2 and j < relK.shape[0]:
                relK_j = np.asarray(relK[j], dtype=float)
            elif relK.ndim == 1:
                relK_j = np.asarray(relK, dtype=float)
            else:
                return fallback
            relK_j = np.nan_to_num(relK_j, nan=0.0, posinf=0.0, neginf=0.0)
            if len(relK_j) > 0 and np.sum(np.abs(relK_j)) > 0:
                cfg = _lag_selection_cfg()
                strength_by_lag = np.abs(relK_j)[::-1]
                max_effective_lag = _effective_max_lag(time_step, cause_name, effect_name)
                max_effective_lag = min(max_effective_lag, len(strength_by_lag) - 1)

                valid = strength_by_lag[:max_effective_lag + 1]
                if valid.size == 0 or np.max(valid) <= 0:
                    return fallback

                smoothed = _smooth_lag_spectrum(valid, cfg.get("smooth_window", 3))
                scale = float(np.max(smoothed))
                normalized = smoothed / scale if scale > 0 else smoothed
                raw_scale = float(np.max(valid))
                raw_normalized = valid / raw_scale if raw_scale > 0 else valid

                lags = np.arange(valid.size, dtype=float)
                denom = max(1.0, float(max_effective_lag))
                adjusted = normalized.copy()
                adjusted -= float(cfg.get("boundary_penalty", 0.15)) * (lags / denom) ** 2

                seasonal_period = int(cfg.get("seasonal_period", 12))
                if seasonal_period > 0:
                    seasonal_mask = (lags > 0) & ((lags % seasonal_period) == 0)
                    adjusted[seasonal_mask] -= float(cfg.get("seasonal_lag_penalty", 0.08))

                raw_lag = int(np.argmax(strength_by_lag))
                raw_lag = min(raw_lag, int(time_step) - 1)
                best_adjusted = float(np.max(adjusted))
                plateau_tol = float(cfg.get("plateau_rel_tol", 0.05))
                min_raw_fraction = float(cfg.get("min_raw_peak_fraction", 0.2))
                candidates = np.where(
                    (adjusted >= best_adjusted - plateau_tol)
                    & (raw_normalized >= min_raw_fraction)
                )[0]
                lag = int(candidates[0]) if len(candidates) else int(np.argmax(adjusted))
                selected_idx = min(lag, len(valid) - 1)

                edge_guard = int(cfg.get("edge_guard", 1))
                raw_boundary = raw_lag >= int(time_step) - 1 - edge_guard
                boundary = lag >= max_effective_lag - edge_guard
                seasonal_warning = seasonal_period > 0 and lag > 0 and lag % seasonal_period == 0
                confidence = max(0.0, min(1.0, float(normalized[selected_idx])))

                return {
                    "lag": max(0, int(lag)),
                    "raw_lag": max(0, int(raw_lag)),
                    "lag_confidence": confidence,
                    "lag_peak_raw": float(valid[selected_idx]),
                    "lag_peak_adjusted": float(adjusted[selected_idx]),
                    "max_effective_lag": int(max_effective_lag),
                    "boundary_warning": bool(boundary),
                    "raw_boundary_warning": bool(raw_boundary),
                    "seasonal_lag_warning": bool(seasonal_warning),
                    "lag_selection_method": "smoothed_shortest_near_peak",
                }
    except Exception as e:
        pass
    return fallback


def compute_lag(relK, j, time_step, cause_name=None, effect_name=None):
    """
    Backward-compatible wrapper returning only the selected lag.
    """
    return compute_lag_details(relK, j, time_step, cause_name, effect_name)["lag"]


def summarize_lag_diagnostics(edges, time_step):
    """
    Aggregate lag warnings by cause/effect pair.
    """
    summary = {}
    for edge in edges:
        key = (edge.get("cause_name", "unknown"), edge.get("effect_name", "unknown"))
        if key not in summary:
            summary[key] = {
                "cause_name": key[0],
                "effect_name": key[1],
                "n_edges": 0,
                "selected_boundary_count": 0,
                "raw_boundary_count": 0,
                "seasonal_lag_count": 0,
                "lag_sum": 0.0,
                "raw_lag_sum": 0.0,
                "lag_counts": {},
                "raw_lag_counts": {},
            }
        item = summary[key]
        lag = int(edge.get("lag", 0))
        raw_lag = int(edge.get("raw_lag", lag))
        item["n_edges"] += 1
        item["selected_boundary_count"] += int(bool(edge.get("boundary_warning", False)))
        item["raw_boundary_count"] += int(bool(edge.get("raw_boundary_warning", False)))
        item["seasonal_lag_count"] += int(bool(edge.get("seasonal_lag_warning", False)))
        item["lag_sum"] += lag
        item["raw_lag_sum"] += raw_lag
        item["lag_counts"][str(lag)] = item["lag_counts"].get(str(lag), 0) + 1
        item["raw_lag_counts"][str(raw_lag)] = item["raw_lag_counts"].get(str(raw_lag), 0) + 1

    rows = []
    for item in summary.values():
        n_edges = max(1, item["n_edges"])
        rows.append({
            "cause_name": item["cause_name"],
            "effect_name": item["effect_name"],
            "n_edges": item["n_edges"],
            "mean_lag": item["lag_sum"] / n_edges,
            "mean_raw_lag": item["raw_lag_sum"] / n_edges,
            "selected_boundary_ratio": item["selected_boundary_count"] / n_edges,
            "raw_boundary_ratio": item["raw_boundary_count"] / n_edges,
            "seasonal_lag_ratio": item["seasonal_lag_count"] / n_edges,
            "lag_counts": item["lag_counts"],
            "raw_lag_counts": item["raw_lag_counts"],
        })
    rows.sort(key=lambda x: (x["effect_name"], x["cause_name"]))
    return rows


def save_lag_diagnostics(edges, output_dir, time_step, suffix=""):
    """
    Save boundary/seasonal lag diagnostics used to detect unstable lag maps.
    """
    summary = summarize_lag_diagnostics(edges, time_step)
    suffix_part = f"_{suffix}" if suffix else ""
    json_path = os.path.join(output_dir, f"lag_diagnostics{suffix_part}.json")
    csv_path = os.path.join(output_dir, f"lag_diagnostics{suffix_part}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "lag_selection_config": _lag_selection_cfg(),
            "time_step": int(time_step),
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)

    csv_rows = []
    for row in summary:
        csv_rows.append({k: v for k, v in row.items() if not isinstance(v, dict)})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Lag诊断 JSON: {json_path}")
    print(f"Lag诊断 CSV: {csv_path}")


def plot_target_causal_graph(causal_edges, var_names, target_var, output_dir):
    """
    绘制对 TARGET 的因果图 - SCI 科研绘图标准
    
    Args:
        causal_edges: 因果边列表
        var_names: 变量名称列表
        target_var: 目标变量名称
        output_dir: 输出目录
    """
    import networkx as nx
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.linewidth'] = 1.5
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加 TARGET 节点
    G.add_node(target_var)
    
    # 添加边
    edge_labels = {}
    scores = []
    for edge in causal_edges:
        cause = edge['cause_name']
        lag = edge['lag']
        score = edge.get('score', 0)
        G.add_node(cause)
        G.add_edge(cause, target_var)
        edge_labels[(cause, target_var)] = f'lag={lag}'
        scores.append(score)
    
    if len(G.nodes()) == 0:
        return
    
    # 绘图 - 星形布局 (TARGET 在中心) - SCI 标准
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 使用自定义布局：TARGET 在中心
    pos = {target_var: (0, 0)}
    n_causes = len(causal_edges)
    if n_causes > 0:
        for i, edge in enumerate(causal_edges):
            angle = 2 * np.pi * i / n_causes
            radius = 1.5
            pos[edge['cause_name']] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # 节点颜色
    node_colors = ['#E74C3C' if node == target_var else '#3498DB' for node in G.nodes()]
    node_sizes = [3500 if node == target_var else 2500 for node in G.nodes()]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.9,
                           edgecolors='black', linewidths=1.5)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#2C3E50', 
                           arrows=True, arrowsize=25,
                           connectionstyle='arc3,rad=0.1', alpha=0.8, width=2)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold', font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=10, font_family='Arial')
    
    ax.set_title(f'Causal Graph: Variables → {target_var}', fontsize=16, fontweight='bold', fontfamily='Arial')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_causal_graph.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"因果图已保存: {os.path.join(output_dir, 'target_causal_graph.png')}")
    
    # 绘制因果影响条形图 - SCI 标准
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    cause_names = [edge['cause_name'] for edge in causal_edges]
    cause_scores = [edge['score'] for edge in causal_edges]
    cause_lags = [edge['lag'] for edge in causal_edges]
    
    # 按分数排序
    sorted_idx = np.argsort(cause_scores)[::-1]
    sorted_names = [cause_names[i] for i in sorted_idx]
    sorted_scores = [cause_scores[i] for i in sorted_idx]
    sorted_lags = [cause_lags[i] for i in sorted_idx]
    
    # 颜色根据分数
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_scores)))
    
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels([f"{name} (lag={sorted_lags[i]})" for i, name in enumerate(sorted_names)], 
                       fontsize=11, fontfamily='Arial')
    ax.set_xlabel('Causal Score (Normalized)', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_ylabel('Cause Variables', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_title(f'Causal Influence on {target_var}', fontsize=14, fontweight='bold', fontfamily='Arial')
    # ax.set_xlim(0, 1.05)  # 归一化后范围 0-1
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_influence_bar.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"因果影响图已保存: {os.path.join(output_dir, 'causal_influence_bar.png')}")
    
    # 绘制时间滞后热力图 - SCI 标准
    lag_data = np.array([sorted_lags]).T
    score_data = np.array([sorted_scores]).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
    
    # 因果分数热力图
    sns.heatmap(score_data, ax=ax1, yticklabels=sorted_names, xticklabels=[target_var],
                cmap='RdBu_r', annot=True, fmt='.3f', 
                annot_kws={'size': 11, 'fontweight': 'bold'},
                cbar_kws={'label': 'Causal Score (Normalized)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax1.set_title(f'Causal Scores → {target_var}', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Effect Variable', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cause Variable', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    
    # 时间滞后热力图
    sns.heatmap(lag_data, ax=ax2, yticklabels=sorted_names, xticklabels=[target_var],
                cmap='YlOrRd', annot=True, fmt='d', 
                annot_kws={'size': 11, 'fontweight': 'bold'},
                cbar_kws={'label': 'Time Lag (steps)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax2.set_title(f'Time Lags → {target_var}', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Effect Variable', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cause Variable', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_heatmaps.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"因果热力图已保存: {os.path.join(output_dir, 'causal_heatmaps.png')}")


def plot_point_causal_graph(causal_edges, var_names, target_var, point_id, lat, lon, output_dir):
    """
    为单个点绘制因果图 - SCI 科研绘图标准
    
    Args:
        causal_edges: 因果边列表
        var_names: 变量名称列表
        target_var: 目标变量名称
        point_id: 点ID
        lat: 纬度
        lon: 经度
        output_dir: 输出目录
    """
    import networkx as nx
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.linewidth'] = 1.2
    
    if len(causal_edges) == 0:
        return
    
    # 创建有向图
    G = nx.DiGraph()
    G.add_node(target_var)
    
    edge_labels = {}
    for edge in causal_edges:
        cause = edge['cause_name']
        lag = edge['lag']
        score = edge.get('score', 0)
        G.add_node(cause)
        G.add_edge(cause, target_var, weight=score)
        edge_labels[(cause, target_var)] = f'lag={lag}'
    
    # 绘图 - SCI 标准
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=200)
    
    # 左图: 因果网络图
    pos = {target_var: (0, 0)}
    n_causes = len(causal_edges)
    for i, edge in enumerate(causal_edges):
        angle = 2 * np.pi * i / max(n_causes, 1)
        pos[edge['cause_name']] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
    
    node_colors = ['#E74C3C' if node == target_var else '#3498DB' for node in G.nodes()]
    node_sizes = [2500 if node == target_var else 1800 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.9, edgecolors='black', linewidths=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#2C3E50', arrows=True, 
                           arrowsize=18, width=1.5, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight='bold', font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8, font_family='Arial')
    
    ax1.set_title(f'Point {point_id}\nLat: {lat:.4f}, Lon: {lon:.4f}', 
                  fontsize=12, fontweight='bold', fontfamily='Arial')
    ax1.axis('off')
    
    # 右图: 因果分数条形图
    cause_names = [edge['cause_name'] for edge in causal_edges]
    cause_scores = [edge['score'] for edge in causal_edges]
    cause_lags = [edge['lag'] for edge in causal_edges]
    
    sorted_idx = np.argsort(cause_scores)[::-1]
    sorted_names = [cause_names[i] for i in sorted_idx]
    sorted_scores = [cause_scores[i] for i in sorted_idx]
    sorted_lags = [cause_lags[i] for i in sorted_idx]
    
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_scores)))
    ax2.barh(range(len(sorted_names)), sorted_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels([f"{name} (lag={sorted_lags[i]})" for i, name in enumerate(sorted_names)], 
                        fontsize=9, fontfamily='Arial')
    ax2.set_xlabel('Causal Score (Normalized)', fontsize=11, fontweight='bold', fontfamily='Arial')
    ax2.set_title(f'Causal Influence on {target_var}', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax2.set_xlim(0, 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    coord_name = f"lat{lat:.4f}_lon{lon:.4f}"
    plt.savefig(os.path.join(output_dir, f'{coord_name}.png'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_RRP_scores(
    model,
    input_data,
    interpreted_series,
    device,
    lat=None,
    lon=None,
    aux_data=None,
    static_vars=None,
    debug=False,
):
    """
    生成 RRP (Regression Relevance Propagation) 因果分数
    
    完全参考 code/CausalFormer/explainer/explainer.py 和 interpret.py
    论文: https://arxiv.org/html/2406.16708v1
    
    Args:
        model: 模型
        input_data: 输入数据 [batch, time_step, series_num, feature_dim]
        interpreted_series: 目标序列索引 (被解释的系列)
        device: 设备
        lat: 纬度张量 (batch,) - 当模型启用地理编码时需要
        lon: 经度张量 (batch,) - 当模型启用地理编码时需要
        aux_data: 辅助时序数据 - 当模型使用 FiLM 条件时需要
        static_vars: 静态变量数据 - 当模型使用 FiLM 条件时需要
    
    Returns:
        relA: 注意力因果分数 (series_num, series_num)
        relK_aligned: 卷积核因果分数 (series_num, time_step)
               relK_aligned[j] = 变量 j 对 interpreted_series 的时间滞后分数
    """
    model.eval()
    
    # 前向传播
    output = model(input_data, lat, lon, aux_data, static_vars)
    
    # 创建 one-hot 张量 (参考 explainer.py 第35-36行)
    one_hot = torch.zeros_like(output, dtype=torch.float).to(device)
    one_hot[:, :, interpreted_series, :] = 1
    
    one_hot_vector = one_hot.clone()
    one_hot.requires_grad_(True)
    
    # 计算目标输出 (参考 explainer.py 第41行)
    one_hot_sum = torch.sum(one_hot * output)
    
    # 反向传播 (参考 explainer.py 第43-44行)
    model.zero_grad()
    one_hot_sum.backward(retain_graph=True)
    
    # 应用 RRP (参考 explainer.py 第46行)
    model.relprop(one_hot_vector)
    
    relAs = []
    relKs = []  # 收集 relK 用于时间滞后计算
    
    # 从每个 encoder 层收集因果分数 (参考 explainer.py 第50-62行)
    for layer_idx, layer in enumerate(model.encoder.layers):
        # 梯度调制 - relA (参考 explainer.py 第52行)
        # relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
        relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
        # 不再使用 clamp(0)，保留负因果（抑制作用）
        relA = relA.clamp(min=0)
        
        # 对每层进行标准化，消除层间数值差异
        relA_mean_head = relA.mean((0, 1))  # mean for sample and head (参考 explainer.py 第61行)
        # Z-score 标准化
        ra_mean = relA_mean_head.mean()
        ra_std = relA_mean_head.std()
        if ra_std > 0:
            relA_normalized = (relA_mean_head - ra_mean) / ra_std
        else:
            relA_normalized = relA_mean_head
        relAs.append(relA_normalized)
        
        # 梯度调制 - relK (参考 explainer.py 第53行)
        # 关键修正: 使用 get_rel() * abs(get_grad()) 而非 get_wgt()
        relK_layer = layer.attention.Wv.get_rel() * torch.abs(layer.attention.Wv.get_grad())
        # 不再使用 clamp(0)，保留负因果（抑制作用）
        relK_layer = relK_layer.clamp(min=0)
        
        # 对每层进行标准化，消除层间数值差异
        relK_mean_head = relK_layer.mean(0)  # mean for head
        # Z-score 标准化：(x - mean) / std
        rk_mean = relK_mean_head.mean()
        rk_std = relK_mean_head.std()
        if rk_std > 0:
            relK_normalized = (relK_mean_head - rk_mean) / rk_std
        else:
            relK_normalized = relK_mean_head
        relKs.append(relK_normalized)
        
        if debug:
            print(f"  Layer {layer_idx}: relK sum={relK_layer.sum().item():.4f}, normalized mean={relK_normalized.mean().item():.4f}, std={relK_normalized.std().item():.4f}")
    
    # 沿编码器层维度聚合
    if debug:
        for i, rk in enumerate(relKs):
            print(f"  Before mean - Layer {i}: mean={rk.mean().item():.6f}, std={rk.std().item():.6f}")
    relA = torch.stack(relAs).mean(0)  # 改用 mean 聚合 (series_num, series_num)
    # 使用 mean(0) 聚合多层，捕捉平均因果效应
    relK = torch.stack(relKs).mean(0)  # (series_num, series_num, time_step, time_step)
    
    if debug:
        print(f"  After mean: relK mean={relK.mean().item():.6f}, std={relK.std().item():.6f}")
    
    series_num = relA.shape[0]
    time_step = model.encoder.layers[0].attention.Wv.input_window
    
    # 提取 relK_aligned (参考 interpret.py 第112-115行)
    # relK 形状: (series_num, series_num, time_step, time_step)
    # relK[j, i, out_t, in_t] = 变量 j 在输入时间 in_t 对变量 i 在输出时间 out_t 的影响
    
    # 转为 numpy
    relK_np = relK.detach().cpu().numpy()
    
    # relK_aligned = relK[:, interpreted_series, -1, :] (参考 interpret.py 第112行)
    # 所有变量对 interpreted_series 在最后输出时间的影响
    relK_aligned = deepcopy(relK_np[:, interpreted_series, -1, :])  # (series_num, time_step)
    
    # 修正对角线: 变量自身对自身用倒数第二步 (参考 interpret.py 第114行)
    # The relK[i][i][-1] is zero vector due to the time_step th data can not be used to predict the time_step th future itself.
    relK_aligned[interpreted_series, :] = relK_np[interpreted_series, interpreted_series, -2, :]
    
    return relA, torch.tensor(relK_aligned)


def analyze_causal_graph(relA, relK, m, n, time_step, var_names):
    """
    分析因果图，生成 (i, j, t) 格式的因果关系
    
    完全参考 code/CausalFormer/interpret.py 中的 analyze 函数
    
    Args:
        relA (List[np.array]): 每个目标序列的因果分数，relA[i] 是 (series_num,) 向量
        relK (List[np.array]): 每个目标序列的时间因果分数，relK[i] 是 (series_num, time_step) 矩阵
        m (int): 选择 top m 个聚类簇
        n (int): KMeans 聚类数
        time_step (int): 时间步数
        var_names (List[str]): 变量名称列表
    
    Returns:
        causal_edges (List[dict]): 因果边列表 [{cause_idx, cause_name, effect_idx, effect_name, lag}, ...]
    """
    causal_edges = []
    series_num = len(relA)
    
    if series_num == 0:
        return causal_edges
    
    # 确保 n 不超过 series_num
    n_actual = min(n, series_num)
    m_actual = min(m, n_actual - 1) if n_actual > 1 else 1
    
    if n_actual < 2:
        # 如果聚类数太小，直接使用阈值方法
        for i, relA_i in enumerate(relA):
            if not isinstance(relA_i, np.ndarray):
                relA_i = np.array(relA_i)
            if relA_i.sum() == 0:
                continue
            threshold = np.mean(relA_i) + np.std(relA_i)
            for j in range(len(relA_i)):
                if relA_i[j] > threshold:
                    lag_info = compute_lag_details(
                        relK[i] if i < len(relK) else np.zeros((series_num, time_step)),
                        j,
                        time_step,
                        cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                        effect_name=var_names[i] if i < len(var_names) else f'Var_{i}'
                    )
                    causal_edges.append({
                        'cause_idx': int(j),
                        'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                        'effect_idx': int(i),
                        'effect_name': var_names[i] if i < len(var_names) else f'Var_{i}',
                        'lag': int(lag_info['lag']),
                        **lag_info
                    })
        return causal_edges
    
    # 为每个目标序列 i 查找原因
    for i, relA_i in enumerate(relA):
        # 转换为 numpy 数组
        if not isinstance(relA_i, np.ndarray):
            relA_i = np.array(relA_i)
        
        if len(relA_i) == 0 or relA_i.sum() == 0.0:
            continue
        
        # 使用 KMeans 聚类 (参考 interpret.py 第64-75行)
        try:
            data = relA_i.reshape(-1, 1)
            n_clusters = min(n_actual, len(data))
            if n_clusters < 2:
                continue
            
            estimator = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            estimator.fit(data)
            cluster_labels = estimator.labels_
            cluster_centers = estimator.cluster_centers_.reshape(-1)
            
            # 找到最大的 m 个聚类 (参考 interpret.py 第75行)
            m_use = min(m_actual, len(cluster_centers))
            largest_m_clusters = np.argsort(cluster_centers)[-m_use:]
            
            for j in range(len(relA_i)):
                if cluster_labels[j] in largest_m_clusters:
                    # 从 relK 中获取时间滞后 (参考 interpret.py 第78-80行)
                    relK_for_lag = np.zeros((series_num, time_step))
                    if i < len(relK):
                        relK_i = relK[i]
                        if isinstance(relK_i, list):
                            relK_for_lag = np.array(relK_i)
                        elif isinstance(relK_i, np.ndarray):
                            relK_for_lag = relK_i
                    if isinstance(relK_for_lag, np.ndarray) and relK_for_lag.ndim == 1:
                        relK_for_lag = np.tile(relK_for_lag.reshape(1, -1), (series_num, 1))
                    lag_info = compute_lag_details(
                        relK_for_lag,
                        j,
                        time_step,
                        cause_name=var_names[j] if j < len(var_names) else f'Var_{j}',
                        effect_name=var_names[i] if i < len(var_names) else f'Var_{i}'
                    )
                    
                    # 添加因果边: (cause=j, effect=i, lag=t)
                    causal_edges.append({
                        'cause_idx': int(j),
                        'cause_name': var_names[j] if j < len(var_names) else f'Var_{j}',
                        'effect_idx': int(i),
                        'effect_name': var_names[i] if i < len(var_names) else f'Var_{i}',
                        'lag': int(lag_info['lag']),
                        **lag_info
                    })
        except Exception as e:
            print(f"  聚类失败 (i={i}): {e}")
            continue
    
    return causal_edges


def plot_global_causal_graph(causal_edges, var_names, output_dir):
    """
    绘制全局因果图
    
    Args:
        causal_edges: 因果边列表
        var_names: 变量名称列表
        output_dir: 输出目录
    """
    import networkx as nx
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for name in var_names:
        G.add_node(name)
    
    # 添加边
    edge_labels = {}
    for edge in causal_edges:
        cause = edge['cause_name']
        effect = edge['effect_name']
        lag = edge['lag']
        G.add_edge(cause, effect)
        edge_labels[(cause, effect)] = f't={lag}'
    
    # 绘图
    plt.figure(figsize=(16, 12))
    
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=2000, alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, 
                           connectionstyle='arc3,rad=0.1')
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    plt.title('Global Causal Graph', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_causal_graph.png'), dpi=500, bbox_inches='tight')
    plt.close()
    print(f"因果图已保存: {os.path.join(output_dir, 'global_causal_graph.png')}")
    
    # 绘制因果矩阵热力图
    n_vars = len(var_names)
    causal_matrix = np.zeros((n_vars, n_vars))
    lag_matrix = np.zeros((n_vars, n_vars))
    
    for edge in causal_edges:
        i = edge['cause_idx']
        j = edge['effect_idx']
        causal_matrix[i, j] = 1
        lag_matrix[i, j] = edge['lag']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 因果矩阵
    sns.heatmap(causal_matrix, ax=axes[0], xticklabels=var_names, yticklabels=var_names,
                cmap='Blues', annot=True, fmt='.0f', cbar_kws={'label': 'Causal Connection'})
    axes[0].set_title('Causal Adjacency Matrix')
    axes[0].set_xlabel('Effect')
    axes[0].set_ylabel('Cause')
    
    # 时间滞后矩阵
    sns.heatmap(lag_matrix, ax=axes[1], xticklabels=var_names, yticklabels=var_names,
                cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Time Lag'})
    axes[1].set_title('Time Lag Matrix')
    axes[1].set_xlabel('Effect')
    axes[1].set_ylabel('Cause')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_matrices.png'), dpi=500, bbox_inches='tight')
    plt.close()
    print(f"因果矩阵已保存: {os.path.join(output_dir, 'causal_matrices.png')}")

if __name__ == "__main__":

    mode='analyze'
    model_path=config["analyze"]["model_path"]
    NC_FILE = config["analyze"]["NC_FILE"]
    OUT_DIR = config["analyze"]["OUT_DIR"]
    PREDICTORS = config["analyze"]["PREDICTORS"]
    TARGET = config["analyze"]["TARGET"]
    LULC_IDS = [1,2,4,9]
    if mode == 'analyze':
        # 因果分析模式
        print("=" * 60)
        print("运行模式: 因果分析")
        print("=" * 60)
        
        if model_path is None:
            # 自动查找 best model
            best_models = [f for f in os.listdir(OUT_DIR) if f.startswith('best_model_rmse_')]
            if best_models:
                model_path = os.path.join(OUT_DIR, best_models[0])
                print(f"自动检测到最佳模型: {model_path}")
            else:
                # 尝试使用 checkpoint_latest
                ckpt_path = os.path.join(OUT_DIR, "checkpoint_latest.pth")
                if os.path.exists(ckpt_path):
                    model_path = ckpt_path
                    print(f"使用最新检查点: {model_path}")
                else:
                    raise ValueError("未找到模型文件，请指定 --model_path")
        
        # causal_output_dir = os.path.join(OUT_DIR, "causal_analysis")
        
        # # 针对 TARGET 的因果分析
        # run_causal_analysis(
        #     model_path=model_path,
        #     nc_file=NC_FILE,
        #     output_dir=causal_output_dir,
        #     predictors=PREDICTORS,
        #     target_var=TARGET,
        #     lulc_ids=LULC_IDS
        # )
        
        # 全变量因果分析
        causal_all_output_dir = os.path.join(OUT_DIR, "causal_analysis_all")
        run_causal_analysis_all(
            model_path=model_path,
            nc_file=NC_FILE,
            output_dir=causal_all_output_dir,
            predictors=PREDICTORS,
            target_var=TARGET,
            lulc_ids=LULC_IDS
        )
