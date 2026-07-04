import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
from tqdm import tqdm
import json
# 导入自定义模块
from Nets import PredictModel, prepare_device
from Optimizer import Muon

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", "./model_config_NDVI.json")
# CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", "semi_synthetic_benchmark/configs/model_config_GPP_syn.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
print(f"加载配置文件: {CONFIG_PATH}")

# =============================================================================
# 全局设置
# =============================================================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def load_training_metrics_history(output_dir):
    history_path = os.path.join(output_dir, "training_metrics_history.json")
    empty_history = {
        "iteration": [],
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "val_mae": [],
        "val_r2": []
    }
    if not os.path.exists(history_path):
        return empty_history

    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        for key, value in empty_history.items():
            history.setdefault(key, value.copy())
        return history
    except Exception as exc:
        print(f"警告: 读取训练曲线历史失败，将重新记录: {exc}")
        return empty_history


def save_and_plot_training_metrics(history, output_dir):
    history_path = os.path.join(output_dir, "training_metrics_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if not history["iteration"]:
        return

    iterations = history["iteration"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_rmse = history["train_rmse"]
    val_rmse = history["val_rmse"]

    def _set_metric_ylim(ax, series_list, robust=False):
        values = np.concatenate([np.asarray(series, dtype=float) for series in series_list])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        if robust and values.size >= 8:
            y_min = float(np.percentile(values, 5))
            y_max = float(np.percentile(values, 95))
            clipped_count = int(np.sum(values > y_max))
            if clipped_count > 0:
                ax.text(
                    0.99, 0.97,
                    f"{clipped_count} spike(s) above view",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="crimson"
                )
        else:
            y_min, y_max = float(values.min()), float(values.max())

        if np.isclose(y_min, y_max):
            padding = max(abs(y_max) * 0.1, 1e-8)
        else:
            padding = (y_max - y_min) * 0.12
        ax.set_ylim(max(0.0, y_min - padding), y_max + padding)

    def _plot_curves(x_values, y1, y2, label1, label2, ylabel, title, output_name, robust_ylim=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_values, y1, marker="o", markersize=4, linewidth=2, label=label1)
        ax.plot(x_values, y2, marker="s", markersize=4, linewidth=2, label=label2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        _set_metric_ylim(ax, [y1, y2], robust=robust_ylim)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
        plt.close(fig)

    _plot_curves(
        iterations, train_loss, val_loss,
        "Train Loss", "Validation Loss", "MSE Loss",
        "Training vs Validation Loss",
        "training_validation_loss.png"
    )
    _plot_curves(
        iterations, train_rmse, val_rmse,
        "Train RMSE", "Validation RMSE", "RMSE",
        "Training vs Validation RMSE",
        "training_validation_rmse.png"
    )

    if len(iterations) > 2:
        _plot_curves(
            iterations[1:], train_loss[1:], val_loss[1:],
            "Train Loss", "Validation Loss", "MSE Loss",
            "Training vs Validation Loss (Zoom)",
            "training_validation_loss_zoom.png",
            robust_ylim=True
        )
        _plot_curves(
            iterations[1:], train_rmse[1:], val_rmse[1:],
            "Train RMSE", "Validation RMSE", "RMSE",
            "Training vs Validation RMSE (Zoom)",
            "training_validation_rmse_zoom.png",
            robust_ylim=True
        )


def apply_train_subset_ratio(train_idx, algin=False, al=1.0, random_seed=42):
    """
    按比例抽样训练样本索引

    Args:
        train_idx: 原始训练索引列表
        algin: 是否启用比例抽样
        al: 训练样本保留比例，范围 (0, 1]
        random_seed: 随机种子
    """
    if not algin:
        return train_idx

    al = float(al)
    if not (0 < al <= 1):
        raise ValueError(f"参数 al 必须在 (0, 1]，当前值: {al}")
    if al >= 1.0:
        return train_idx

    keep_num = max(1, int(len(train_idx) * al))
    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(train_idx, size=keep_num, replace=False)
    sampled_idx = sorted(sampled_idx.tolist())
    print(f"启用 algin: al={al:.3f}，训练样本从 {len(train_idx)} 下采样到 {len(sampled_idx)}")
    return sampled_idx

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
    
    # 缓存文件路径
    data_cache_path = config["train"]["PROCEED_DATA"]
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
            # LULC 筛选: 检查所有时间步的众数
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
                    # 辅助变量允许部分缺失，用插值填充
                    aux_pt_data[v_name] = ts
            
            # 提取静态变量 (不随时间变化)
            static_pt_data = {}
            if static_vars:
                for v_name in static_vars:
                    if v_name not in var_dict:
                        static_pt_data[v_name] = 0.0
                        continue
                    v = var_dict[v_name]
                    # 静态变量: 取空间位置的值
                    if v.ndim == 2:
                        val = v[i, j]
                    elif v.ndim == 3:
                        # 如果有时间维，取平均或第一个时间步
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
    
    支持功能：
    - 地理坐标 (lat, lon) 信息
    - 辅助时间序列变量 (AUX_PREDICTORS)
    - 静态变量 (STATIC_VARS)
    """
    def __init__(self, data, seq_len, output_window=1, coords=None, 
                 aux_data=None, static_data=None):
        """
        Args:
            data: 主时序数据 (N_points, T, V)
            seq_len: 输入窗口长度
            output_window: 输出窗口长度
            coords: 坐标列表 [{'lat': float, 'lon': float, ...}, ...]。
            aux_data: 辅助时序数据 (N_points, T, V_aux) - 可选
            static_data: 静态数据 (N_points, V_static) - 可选
        """
        self.samples, self.targets = [], []
        self.lats, self.lons = [], []
        self.aux_samples = [] if aux_data is not None else None
        self.static_samples = [] if static_data is not None else None
        
        self.has_coords = coords is not None and len(coords) > 0
        self.has_aux = aux_data is not None
        self.has_static = static_data is not None
        
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
                pt_lat, pt_lon = 0.0, 0.0
            
            # 获取该点的辅助时序数据
            aux_ts = aux_data[p] if self.has_aux else None  # (n_t, n_v_aux)
            
            # 获取该点的静态数据
            static_pt = static_data[p] if self.has_static else None  # (n_v_static,)
            
            for t in range(seq_len, n_t + 1):
                # 主输入: 前 seq_len 步的所有变量 (time_step, series_num, feature_dim)
                sample = ts[t-seq_len:t].reshape(seq_len, n_v, 1)
                # 目标: 后 output_window 步的所有变量 (output_window, series_num, output_dim)
                target = ts[t-output_window:t].reshape(output_window, n_v, 1)
                self.samples.append(sample)
                self.targets.append(target)
                
                # 坐标
                self.lats.append(pt_lat)
                self.lons.append(pt_lon)
                
                # 辅助时序数据
                if self.has_aux:
                    aux_sample = aux_ts[t-seq_len:t].reshape(seq_len, aux_ts.shape[1], 1)
                    self.aux_samples.append(aux_sample)
                
                # 静态数据 (每个样本都相同)
                if self.has_static:
                    self.static_samples.append(static_pt)
        
        self.samples = torch.FloatTensor(np.array(self.samples))
        self.targets = torch.FloatTensor(np.array(self.targets))
        self.lats = torch.FloatTensor(self.lats)
        self.lons = torch.FloatTensor(self.lons)
        
        if self.has_aux:
            self.aux_samples = torch.FloatTensor(np.array(self.aux_samples))
        if self.has_static:
            self.static_samples = torch.FloatTensor(np.array(self.static_samples))
        
        print(f"数据集大小 - 主样本: {self.samples.shape}, 目标: {self.targets.shape}")
        if self.has_coords:
            print(f"  包含地理坐标: lat 范围 [{self.lats.min():.2f}, {self.lats.max():.2f}], "
                  f"lon 范围 [{self.lons.min():.2f}, {self.lons.max():.2f}]")
        if self.has_aux:
            print(f"  包含辅助时序变量: {self.aux_samples.shape}")
        if self.has_static:
            print(f"  包含静态变量: {self.static_samples.shape}")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        result = [self.samples[idx], self.targets[idx]]
        
        # 返回坐标
        result.extend([self.lats[idx], self.lons[idx]])
        
        # 返回辅助数据
        if self.has_aux:
            result.append(self.aux_samples[idx])
        else:
            result.append(torch.tensor([]))  # 占位符
        
        # 返回静态数据
        if self.has_static:
            result.append(self.static_samples[idx])
        else:
            result.append(torch.tensor([]))  # 占位符
        
        return tuple(result)



def train_model(nc_file, output_dir, predictors, target_var, lulc_ids,
                use_geo_encoding=False, aux_predictors=None, static_vars=None,
                algin=False, al=1.0):
    """
    训练 CausalFormer 模型
    
    Args:
        nc_file: netCDF 数据文件路径
        output_dir: 输出目录
        predictors: 主要预测变量列表 (参与因果发现)
        target_var: 目标变量
        lulc_ids: 土地利用类型 ID 列表
        use_geo_encoding: 是否使用地理位置编码 (默认 False)
        aux_predictors: 辅助时间序列变量列表 (不参与因果发现)
                        如: ["temperature_2m", "LST", "total_evaporation_sum"]
        static_vars: 静态变量列表 (不随时间变化)
                    如: ["DEM"]
    """
    os.makedirs(output_dir, exist_ok=True)
    aux_predictors = aux_predictors or []
    static_vars = static_vars or []
    
    # 1. Prepare Data
    ds, lat, lon, time = load_nc_data(nc_file)
    
    # 加载数据 (包括辅助变量和静态变量)
    if use_geo_encoding or aux_predictors or static_vars:
        data, aux_data, static_data, coords = preprocess_geo_data(
            ds, predictors, target_var, lat, lon, time, lulc_ids, 
            return_coords=True,
            aux_predictors=aux_predictors,
            static_vars=static_vars
        )
        print(f"\n已加载 {len(coords)} 个地理点的数据")
        if aux_data is not None:
            print(f"  辅助时序变量: {aux_predictors}")
        if static_data is not None:
            print(f"  静态变量: {static_vars}")
    else:
        result = preprocess_geo_data(
            ds, predictors, target_var, lat, lon, time, lulc_ids,
            aux_predictors=aux_predictors,
            static_vars=static_vars
        )
        data, aux_data, static_data = result[0], result[1], result[2]
        coords = None
    
    # 标准化主数据
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    N, T, V = data.shape
    data = scaler.fit_transform(data.reshape(-1, V)).reshape(N, T, V)
    
    # 标准化辅助数据
    aux_scaler = None
    if aux_data is not None:
        aux_scaler = MinMaxScaler(feature_range=(0.1, 1))
        V_aux = aux_data.shape[2]
        aux_data = aux_scaler.fit_transform(aux_data.reshape(-1, V_aux)).reshape(N, T, V_aux)
    
    # 标准化静态数据
    static_scaler = None
    if static_data is not None:
        static_scaler = MinMaxScaler(feature_range=(0.1, 1))
        static_data = static_scaler.fit_transform(static_data)

    # 保存 scaler，确保验证/推理使用同一尺度
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    if aux_scaler is not None:
        joblib.dump(aux_scaler, os.path.join(output_dir, "aux_scaler.pkl"))
    if static_scaler is not None:
        joblib.dump(static_scaler, os.path.join(output_dir, "static_scaler.pkl"))
    
    SEQ_LEN = config["model"]["SEQ_LEN"]
    OUTPUT_WINDOW = config["model"]["output_window"]
    t_idx = predictors.index(target_var) if target_var in predictors else -1
    if t_idx == -1: 
        predictors.append(target_var)
        t_idx = len(predictors) - 1
    
    full_ds = GeoDataset(data, SEQ_LEN, OUTPUT_WINDOW, coords=coords,
                          aux_data=aux_data, static_data=static_data)
    train_idx, test_idx = train_test_split(list(range(len(full_ds))), test_size=0.3, random_state=42)
    raw_train_count = len(train_idx)
    train_idx = apply_train_subset_ratio(train_idx, algin=algin, al=al, random_seed=42)
    used_train_count = len(train_idx)

    # 将训练/验证划分索引保存，便于验证阶段复用同一划分
    split_path = os.path.join(output_dir, "split_indices.json")
    with open(split_path, "w") as f:
        json.dump({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "test_size": 0.3,
            "random_seed": 42,
            "total_samples": len(full_ds),
            "algin": bool(algin),
            "al": float(al),
            "raw_train_samples": raw_train_count,
            "used_train_samples": used_train_count
        }, f)
    print(f"已保存划分索引: {split_path}")

    train_ds, test_ds = torch.utils.data.Subset(full_ds, train_idx), torch.utils.data.Subset(full_ds, test_idx)
    
    BATCH_SIZE = config["train"]["BATCH_SIZE"]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Setup Model & Optimizers
    cfg = {
        'n_gpu': 0, 
        'data_loader': {
            'args': {
                'time_step': SEQ_LEN, 
                'output_window': OUTPUT_WINDOW, 
                'series_num': len(predictors), 
                'feature_dim': 1, 
                'output_dim': 1
            }
        }
    }
    
    model = PredictModel(
        cfg, 
        d_model=config["model"]["d_model"], 
        n_head=config["model"]["n_head"], 
        n_layers=config["model"]["n_layers"], 
        ffn_hidden=config["model"]["hidden_layers"], 
        drop_prob=config["model"]["drop_prob"], 
        tau=config["model"]["tau"],
        use_geo_encoding=use_geo_encoding,
        aux_series_num=len(aux_predictors) if aux_predictors else 0,
        static_dim=len(static_vars) if static_vars else 0
    ).to(prepare_device(0))
    
    print(f"\n模型配置:")
    print(f"  地理位置编码: {'已启用' if use_geo_encoding else '未启用'}")
    print(f"  辅助时序变量数量: {len(aux_predictors) if aux_predictors else 0}")
    print(f"  静态变量数量: {len(static_vars) if static_vars else 0}")
    
    optimizers = [Muon(model.parameters(), lr=0.0001), optim.AdamW(model.parameters(), lr=0.0001)]
    criterion = nn.MSELoss()
    
    # 3. Iteration Settings
    TOTAL_ITERS = config["train"]["TOTAL_ITERS"]
    SAVE_FREQ = config["train"]["SAVE_FREQ"]
    EVAL_FREQ = config["train"]["EVAL_FREQ"]
    
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-10, last_epoch=-1, verbose=True) for opt in optimizers]
    
    # 4. Resume Logic (断点续训)
    start_iter = 0
    best_rmse_resume = float('inf')
    metrics_history = load_training_metrics_history(output_dir)
    ckpt_path = os.path.join(output_dir, "checkpoint_latest.pth")
    if os.path.exists(ckpt_path):
        print(f"检测到断点文件 {ckpt_path}，正在恢复...")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        for i, opt in enumerate(optimizers): 
            opt.load_state_dict(checkpoint['optimizer_state_dicts'][i])
        for i, sch in enumerate(schedulers): 
            sch.load_state_dict(checkpoint['scheduler_state_dicts'][i])
        start_iter = checkpoint['iteration']
        best_rmse_resume = checkpoint.get('best_rmse', float('inf'))
        metrics_history = checkpoint.get('metrics_history', metrics_history)
        print(f"成功恢复至 Iteration {start_iter}, Best RMSE: {best_rmse_resume:.6f}")
    
    # 5. Training Loop
    print(f"开始训练: Total Iters={TOTAL_ITERS}, Save Freq={SAVE_FREQ}, Eval Freq={EVAL_FREQ}")
    model.train()
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(start_iter, TOTAL_ITERS), initial=start_iter, total=TOTAL_ITERS)
    total_loss = 0.0
    loss_count = 0
    eval_train_loss = 0.0
    eval_train_count = 0
    best_rmse = best_rmse_resume
    
    for current_iter in pbar:
        # Fetch data
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)
        
        # 解析 batch 数据: (x, y, lat, lon, aux, static)
        x, y, lat_batch, lon_batch, aux_batch, static_batch = batch_data
        
        x = x.to(model.device)
        y = y.to(model.device)
        lat_batch = lat_batch.to(model.device) if use_geo_encoding else None
        lon_batch = lon_batch.to(model.device) if use_geo_encoding else None
        
        # 处理辅助数据
        if aux_batch.numel() > 0:
            aux_batch = aux_batch.to(model.device)
        else:
            aux_batch = None
        
        # 处理静态数据
        if static_batch.numel() > 0:
            static_batch = static_batch.to(model.device)
        else:
            static_batch = None
        
        # Train Step
        for opt in optimizers: 
            opt.zero_grad()
        pred = model(x, lat_batch, lon_batch, aux_batch, static_batch)
        loss = criterion(pred, y)
        loss.backward()
        for opt in optimizers: 
            opt.step()
        for sch in schedulers: 
            sch.step()
        
        total_loss += loss.item()
        loss_count += 1
        eval_train_loss += loss.item()
        eval_train_count += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg_loss': f"{total_loss/loss_count:.4f}"})
        
        # Evaluation & Breakpoint Save (Every EVAL_FREQ iters)
        if (current_iter + 1) % EVAL_FREQ == 0:
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0
            
            with torch.no_grad():
                for batch_data in test_loader:
                    vx, vy, vlat, vlon, vaux, vstatic = batch_data
                    vx = vx.to(model.device)
                    vy = vy.to(model.device)
                    vlat = vlat.to(model.device) if use_geo_encoding else None
                    vlon = vlon.to(model.device) if use_geo_encoding else None
                    vaux = vaux.to(model.device) if vaux.numel() > 0 else None
                    vstatic = vstatic.to(model.device) if vstatic.numel() > 0 else None
                    
                    vpred = model(vx, vlat, vlon, vaux, vstatic)
                    val_loss += criterion(vpred, vy).item()
                    all_preds.append(vpred.cpu().numpy())
                    all_targets.append(vy.cpu().numpy())
            
            # 计算评估指标
            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_targets = np.concatenate(all_targets, axis=0).flatten()
            
            avg_val_loss = val_loss / len(test_loader)
            avg_train_loss = eval_train_loss / eval_train_count if eval_train_count > 0 else total_loss / loss_count
            
            rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
            train_rmse = np.sqrt(max(avg_train_loss, 0.0))
            mae = mean_absolute_error(all_targets, all_preds)
            r2 = r2_score(all_targets, all_preds)

            metrics_history["iteration"].append(int(current_iter + 1))
            metrics_history["train_loss"].append(float(avg_train_loss))
            metrics_history["val_loss"].append(float(avg_val_loss))
            metrics_history["train_rmse"].append(float(train_rmse))
            metrics_history["val_rmse"].append(float(rmse))
            metrics_history["val_mae"].append(float(mae))
            metrics_history["val_r2"].append(float(r2))
            save_and_plot_training_metrics(metrics_history, output_dir)
            
            print(f"\n{'='*60}")
            print(f"Iter {current_iter+1}:")
            print(f"  Train Loss (avg): {avg_train_loss:.6f}")
            print(f"  Train RMSE: {train_rmse:.6f}")
            print(f"  Test Loss:  {avg_val_loss:.6f}")
            print(f"  RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f}")
            
            # Best Model 保存机制 (基于 RMSE)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_path = os.path.join(output_dir, f"best_model_rmse_{rmse:.6f}.pth")
                # 删除旧的 best model
                for f in os.listdir(output_dir):
                    if f.startswith("best_model_rmse_"):
                        os.remove(os.path.join(output_dir, f))
                torch.save({
                    'iteration': current_iter + 1,
                    'model_state_dict': model.state_dict(),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'aux_predictors': aux_predictors,
                    'static_vars': static_vars,
                    'use_geo_encoding': use_geo_encoding
                }, best_model_path)
                print(f"  *** New Best Model Saved! RMSE: {rmse:.6f} ***")
            print(f"  Best RMSE so far: {best_rmse:.6f}")
            print(f"{'='*60}")
            
            # Save Breakpoint (Overwrites latest)
            checkpoint = {
                'iteration': current_iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                'scheduler_state_dicts': [sch.state_dict() for sch in schedulers],
                'best_rmse': best_rmse,
                'use_geo_encoding': use_geo_encoding,
                'metrics_history': metrics_history
            }
            torch.save(checkpoint, ckpt_path)
            eval_train_loss = 0.0
            eval_train_count = 0
            model.train()
            
        # Archive Save (Every SAVE_FREQ iters)
        if (current_iter + 1) % SAVE_FREQ == 0:
            archive_path = os.path.join(output_dir, f"checkpoint_iter_{current_iter+1}.pth")
            torch.save({
                'iteration': current_iter + 1,
                'model_state_dict': model.state_dict()
            }, archive_path)
            print(f"已归档模型: {archive_path}")

    # 6. Final Explanation
    print("\n训练结束！")
    print(f"模型已保存到: {output_dir}")
    print(f"\n如需运行因果分析，请使用:")
    print(f"  run_causal_analysis(model_path, nc_file, output_dir, predictors, target_var, lulc_ids)")
    print(f"\n注意: 因果分析只分析 PREDICTORS → TARGET 的关系")
    print(f"       辅助变量 (AUX_PREDICTORS, STATIC_VARS) 不参与因果发现")
    
    return model, best_rmse

if __name__ == "__main__":
    
    mode='train'
    # mode='analyze'

    # model_path="./Result/CausalResult_IterTrain_GPP/best_model_rmse_0.146009.pth"
   
    NC_FILE = config["train"]["NC_FILE"]
    OUT_DIR = config["train"]["OUT_DIR"]
    PREDICTORS = config["train"]["PREDICTORS"]
    TARGET = config["train"]["TARGET"]
    LULC_IDS = [1,2,3,4,9]
    USE_GEO_ENCODING = config["train"].get("USE_GEO_ENCODING", False)
    ALGIN = config["train"].get("algin", True)
    AL = config["train"].get("al", 1.0)
    print(f"训练比例{AL}")
    
    # 新增: 辅助变量配置
    AUX_PREDICTORS = config["train"].get("AUX_PREDICTORS", [])  # 辅助时间序列变量
    STATIC_VARS = config["train"].get("STATIC_VARS", [])  # 静态变量
    
    if mode == 'train':
        # 训练模式
        print("=" * 60)
        print("运行模式: 训练")
        print(f"主要预测变量 (PREDICTORS): {PREDICTORS}")
        print(f"目标变量 (TARGET): {TARGET}")
        if AUX_PREDICTORS:
            print(f"辅助时序变量 (AUX_PREDICTORS): {AUX_PREDICTORS}")
        if STATIC_VARS:
            print(f"静态变量 (STATIC_VARS): {STATIC_VARS}")
        if USE_GEO_ENCODING:
            print("地理位置编码: 已启用")
        print(f"训练比例参数: algin={ALGIN}, al={AL}")
        print("=" * 60)
        print("\n注意: 因果分析只分析 PREDICTORS → TARGET 的关系")
        print("       AUX_PREDICTORS 和 STATIC_VARS 不参与因果发现，仅作为辅助信息\n")
        
        train_model(
            NC_FILE, OUT_DIR, PREDICTORS, TARGET, LULC_IDS, 
            use_geo_encoding=USE_GEO_ENCODING,
            aux_predictors=AUX_PREDICTORS if AUX_PREDICTORS else None,
            static_vars=STATIC_VARS if STATIC_VARS else None,
            algin=True,
            al=AL
        )
        
    
