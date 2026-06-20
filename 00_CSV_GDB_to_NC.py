"""
CSV_to_NC完全版.py
将GDB/CSV点数据转换为NetCDF格式，不进行插值，直接按经纬度对应值

处理变量：
- GOSIF: Scale factor=0.0001, Fill values=32767(水体)/32766(冰雪)
- GOGPP: Scale factor=0.001(8天)/0.01(月度)/0.1(年度), Fill values=65535(水体)/65534(冰雪)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import re
import os
import gc
import warnings
import multiprocessing
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# 变量元数据配置
# =============================================================================
# 统一的输出填充值
OUTPUT_FILL_VALUE = -9999  # NetCDF填充值（输出时统一使用）

VARIABLE_CONFIG = {
    'GOSIF': {
        'scale_factor': 0.0001,
        'input_fill_values': [32767, 32766],  # 原始数据中的填充值: 32767=水体, 32766=冰雪
        'output_fill_value': OUTPUT_FILL_VALUE,  # 输出统一使用-9999
        'units': 'W m-2 um-1 sr-1',
        'long_name': 'GOSIF Solar-Induced Fluorescence',
        'valid_range': (-1, 1)
    },
    'GOSIFGPP': {
        'scale_factor': 0.001,  # 8天GPP使用0.001
        'input_fill_values': [65535, 65534],  # 原始数据中的填充值: 65535=水体, 65534=冰雪
        'output_fill_value': OUTPUT_FILL_VALUE,
        'units': 'g C m-2 d-1',
        'long_name': 'GOSIF Gross Primary Production (8-day)',
        'valid_range': (0, 50)
    },
    'GOSIFGPP_monthly': {
        'scale_factor': 0.01,
        'input_fill_values': [65535, 65534],
        'output_fill_value': OUTPUT_FILL_VALUE,
        'units': 'g C m-2 mo-1',
        'long_name': 'GOSIF Gross Primary Production (monthly)',
        'valid_range': (0, 500)
    },
    'GOSIFGPP_annual': {
        'scale_factor': 0.1,
        'input_fill_values': [65535, 65534],
        'output_fill_value': OUTPUT_FILL_VALUE,
        'units': 'g C m-2 yr-1',
        'long_name': 'GOSIF Gross Primary Production (annual)',
        'valid_range': (0, 5000)
    },
    # 静态/分类变量：CLCD 按动态变量处理，但不做去季节/去趋势
    'CLCD': {
        'scale_factor': 1.0,
        'input_fill_values': [-9999],
        'output_fill_value': OUTPUT_FILL_VALUE,
        'units': 'category',
        'long_name': 'Land Cover Classification',
        'valid_range': None
    },
    # GOGPP 元数据与 GOSIFGPP 一致，按时间分辨率选择缩放
    'GOGPP': {
        'scale_factor': 0.001,  # 默认按8天
        'input_fill_values': [65535, 65534],
        'output_fill_value': OUTPUT_FILL_VALUE,
        'units': 'g C m-2 d-1',
        'long_name': 'GOSIF Gross Primary Production (8-day)',
        'valid_range': (0, 50)
    }
}

# 默认配置（用于其他变量）
DEFAULT_CONFIG = {
    'scale_factor': 1.0,
    'input_fill_values': [-9999],  # 原始数据中的填充值
    'output_fill_value': OUTPUT_FILL_VALUE,  # 输出统一使用-9999
    'units': 'unknown',
    'long_name': 'Unknown variable',
    'valid_range': None
}

# 完整变量列表（参考nc去季节趋势差值并行.py）
ALL_VARIABLES = [
    'SMs', 'E', 'Eb', 'Ec', 'Ei', 'Ep', 'Ep_aero', 'Ep_rad', 'Es', 'Et', 'Ew', 'H', 'S', 'SMrz', 'TWS', 'GA', 
    'evaporation_from_the_top_of_canopy_sum', 'potential_evaporation_sum', 'runoff_sum', 'surface_latent_heat_flux_sum', 
    'surface_net_thermal_radiation_sum', 'temperature_2m', 'total_evaporation_sum', 'total_precipitation_sum', 
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 
    'volumetric_soil_water_layer_4', 'Precipitation', 'NDWI', 'EVI', 'FPAR', 'GPP', 'LAI', 'LST', 'NDSI', 'NDVI', 
    'GOSIFGPP', 'GOSIF', 'TWS_tavg', 'GWS_tavg', 'CLCD',"DEM"
]


def get_variable_config(var_name: str) -> dict:
    """
    获取变量的元数据配置
    
    Parameters:
    -----------
    var_name : str
        变量名
        
    Returns:
    --------
    dict
        变量配置字典
    """
    # 检查变量名是否在配置中
    if var_name in VARIABLE_CONFIG:
        return VARIABLE_CONFIG[var_name]
    
    # 统一大写方便匹配
    var_upper = var_name.upper()

    # GOSIF (不含GPP)
    if 'GOSIF' in var_upper and 'GPP' not in var_upper:
        return VARIABLE_CONFIG['GOSIF']

    # GPP 相关（GOSIFGPP 或 GOGPP），根据时间分辨率关键词选择缩放
    if ('GOSIFGPP' in var_upper) or ('GOGPP' in var_upper) or ('GPP' in var_upper and 'SIF' in var_upper):
        # 月尺度
        if 'MONTH' in var_upper or 'MON' in var_upper or 'MONTHLY' in var_upper:
            return VARIABLE_CONFIG.get('GOSIFGPP_monthly', VARIABLE_CONFIG['GOSIFGPP'])
        # 年尺度
        if 'ANNUAL' in var_upper or 'YEAR' in var_upper or 'YR' in var_upper:
            return VARIABLE_CONFIG.get('GOSIFGPP_annual', VARIABLE_CONFIG['GOSIFGPP'])
        # 默认按8天
        if 'GOGPP' in var_upper:
            return VARIABLE_CONFIG.get('GOGPP', VARIABLE_CONFIG['GOSIFGPP'])
        return VARIABLE_CONFIG['GOSIFGPP']

    return DEFAULT_CONFIG.copy()


# NOTE: 保留供兼容的函数（未直接使用），但确保只识别原始填充值，不对缺测进行插值。
def apply_scale_and_mask_fill_values(
    data: np.ndarray,
    var_name: str,
    apply_scale: bool = True
) -> np.ndarray:
    """仅识别原始填充值并缩放有效值，不做插值。"""
    config = get_variable_config(var_name)

    result = data.astype(np.float32)

    # 识别原始填充值
    fill_mask = np.isin(result, config['input_fill_values'])

    # 缩放有效值
    if apply_scale and config['scale_factor'] != 1.0:
        valid_mask = (~fill_mask) & np.isfinite(result)
        result[valid_mask] = result[valid_mask] * config['scale_factor']

    # 仅将原始填充值改写为输出填充值，不触碰其它缺测
    result[fill_mask] = config['output_fill_value']

    return result


def read_and_tidy_gdb(gdb_path: str, layer_name: str) -> gpd.GeoDataFrame:
    """
    读取GDB点图层并将其从宽格式转换为整洁的长格式
    
    Parameters:
    -----------
    gdb_path : str
        GDB文件路径
    layer_name : str
        图层名称
        
    Returns:
    --------
    gpd.GeoDataFrame
        长格式的GeoDataFrame
    """
    print(f"正在从 '{gdb_path}' 读取图层 '{layer_name}'...")
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    print(f"读取到 {len(gdf)} 个点，共 {len(gdf.columns)} 列")
    
    # 识别静态列和动态列（格式为: 变量名_年_月）
    static_cols = [col for col in gdf.columns 
                   if not re.match(r'.*_\d{4}_\d{2}', col) and col != 'geometry']
    dynamic_cols = [col for col in gdf.columns if re.match(r'.*_\d{4}_\d{2}', col)]
    
    print(f"静态列: {len(static_cols)}, 动态列: {len(dynamic_cols)}")
    
    if not dynamic_cols:
        raise ValueError("在属性表中未找到格式为 '变量_年_月' 的动态列。")
    
    print("正在将数据从宽格式重塑为长格式...")
    gdf_long = pd.melt(gdf, id_vars=static_cols, value_vars=dynamic_cols,
                       var_name='variable_raw', value_name='value')
    
    # 解析变量名、年份和月份
    parsed_cols = gdf_long['variable_raw'].str.rsplit('_', n=2, expand=True)
    gdf_long['variable'] = parsed_cols[0]
    gdf_long['year'] = pd.to_numeric(parsed_cols[1])
    gdf_long['month'] = pd.to_numeric(parsed_cols[2])
    
    # 创建datetime类型的时间列
    gdf_long['time'] = pd.to_datetime(gdf_long[['year', 'month']].assign(day=1))
    
    # 将原始的geometry信息合并回来
    gdf_geom = gdf[static_cols + ['geometry']].drop_duplicates(subset=static_cols)
    gdf_long = pd.merge(gdf_long, gdf_geom, on=static_cols, how='left')
    
    # 转换回GeoDataFrame
    gdf_long = gpd.GeoDataFrame(gdf_long, geometry='geometry')
    
    # 从geometry中提取经纬度
    if 'lon' not in gdf_long.columns:
        gdf_long['lon'] = gdf_long.geometry.x
    if 'lat' not in gdf_long.columns:
        gdf_long['lat'] = gdf_long.geometry.y
    
    # 清理不再需要的列
    gdf_long = gdf_long.drop(columns=['variable_raw', 'year', 'month'])
    
    print(f"转换完成，共 {len(gdf_long)} 条记录")
    print(f"变量列表: {gdf_long['variable'].unique().tolist()}")
    
    return gdf_long


def read_csv_data(csv_path: str) -> pd.DataFrame:
    """
    读取CSV格式的点数据
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径
        
    Returns:
    --------
    pd.DataFrame
        长格式的DataFrame
    """
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"读取到 {len(df)} 条记录，共 {len(df.columns)} 列")
    
    # 检查是否已经是长格式（包含 lon, lat, time, variable, value 列）
    required_long_cols = {'lon', 'lat', 'time', 'variable', 'value'}
    if required_long_cols.issubset(set(df.columns)):
        print("数据已经是长格式")
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    # 否则尝试将宽格式转换为长格式
    # 假设有 lon, lat 列，其他列为 变量_年_月 格式
    static_cols = [col for col in df.columns 
                   if not re.match(r'.*_\d{4}_\d{2}', col)]
    dynamic_cols = [col for col in df.columns if re.match(r'.*_\d{4}_\d{2}', col)]
    
    if not dynamic_cols:
        raise ValueError("CSV文件格式无法识别，请确保列名格式为 '变量_年_月'")
    
    print(f"静态列: {static_cols}")
    print(f"动态列数量: {len(dynamic_cols)}")
    
    df_long = pd.melt(df, id_vars=static_cols, value_vars=dynamic_cols,
                      var_name='variable_raw', value_name='value')
    
    # 解析变量名、年份和月份
    parsed_cols = df_long['variable_raw'].str.rsplit('_', n=2, expand=True)
    df_long['variable'] = parsed_cols[0]
    df_long['year'] = pd.to_numeric(parsed_cols[1])
    df_long['month'] = pd.to_numeric(parsed_cols[2])
    df_long['time'] = pd.to_datetime(df_long[['year', 'month']].assign(day=1))
    
    df_long = df_long.drop(columns=['variable_raw', 'year', 'month'])
    
    return df_long


def process_single_variable(
    var: str,
    var_df_values: np.ndarray,
    var_df_times: np.ndarray,
    var_df_lat_idx: np.ndarray,
    var_df_lon_idx: np.ndarray,
    time_grid: np.ndarray,
    n_time: int,
    n_lat: int,
    n_lon: int,
    config: dict,
    apply_scale: bool
) -> Tuple[str, np.ndarray, dict, int]:
    """
    处理单个变量的数据填充（用于并行处理）
    使用numpy数组作为输入，避免pickle问题
    
    Returns:
    --------
    tuple: (var_name, var_data, config, valid_count)
    """
    # 初始化为输出填充值，确保未观测区域不出现“-nan”
    output_fill = config['output_fill_value']
    var_data = np.full((n_time, n_lat, n_lon), output_fill, dtype=np.float32)
    
    if len(var_df_values) == 0:
        return var, var_data, config, 0
    
    # 复制值以避免修改原始数据
    values = var_df_values.astype(np.float32).copy()

    # 识别输入填充值
    fill_mask = np.isin(values, config['input_fill_values'])

    # 应用缩放因子（仅对有效值）
    if apply_scale and config['scale_factor'] != 1.0:
        valid_mask = (~fill_mask) & np.isfinite(values)
        values[valid_mask] = values[valid_mask] * config['scale_factor']
    
    # 创建时间索引映射（time_grid已为np.datetime64，需转为Timestamp做映射）
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(time_grid)}
    
    valid_count = 0
    # 填充数据
    for i in range(len(values)):
        t = pd.Timestamp(var_df_times[i])
        if t in time_to_idx:
            t_idx = time_to_idx[t]
            lat_idx = int(var_df_lat_idx[i])
            lon_idx = int(var_df_lon_idx[i])
            
            if not fill_mask[i] and not np.isnan(values[i]):
                var_data[t_idx, lat_idx, lon_idx] = values[i]
                valid_count += 1
            else:
                # 原始填充值或无效值统一写为输出填充值
                var_data[t_idx, lat_idx, lon_idx] = output_fill
    
    return var, var_data, config, valid_count


def create_nc_without_interpolation(
    df_long: pd.DataFrame,
    variables: List[str],
    output_path: str,
    resolution: float = 0.05,
    time_range: Tuple[str, str] = ('2000-01-01', '2024-12-31'),
    fill_value: float = np.nan,
    apply_scale: bool = True,
    add_deseasonalized: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 1000,
    static_vars: Optional[set] = None
) -> xr.Dataset:
    """
    将点数据直接转换为NetCDF，不进行插值
    每个点直接对应到最近的网格单元
    
    Parameters:
    -----------
    df_long : pd.DataFrame
        长格式的数据框，需包含 lon, lat, time, variable, value 列
    variables : List[str]
        要处理的变量列表
    output_path : str
        输出NC文件路径
    resolution : float
        网格分辨率（度）
    time_range : Tuple[str, str]
        时间范围
    fill_value : float
        填充值
    apply_scale : bool
        是否应用缩放因子
    add_deseasonalized : bool
        是否添加去季节化变量
    n_jobs : int
        并行任务数
    chunk_size : int
        去季节化处理的块大小
        
    Returns:
    --------
    xr.Dataset
        创建的数据集
    """
    print(f"\n{'='*60}")
    print("创建NetCDF数据集（无插值模式）")
    print(f"{'='*60}")
    
    # 静态变量（不随时间变化，例如 DEM）；CLCD 属于动态
    static_vars = static_vars or {'DEM'}

    # 如果静态变量以独立列存在但未进入 variable 列，追加到长格式
    static_cols_in_df = [col for col in static_vars if col in df_long.columns]
    missing_static_in_varcol = [col for col in static_cols_in_df if col not in df_long['variable'].unique()]
    if missing_static_in_varcol:
        anchor_time = df_long['time'].min() if 'time' in df_long.columns and not df_long['time'].isnull().all() else pd.to_datetime(time_range[0])
        static_rows = []
        for col in missing_static_in_varcol:
            tmp = df_long[['lon', 'lat', col]].drop_duplicates()
            tmp = tmp.rename(columns={col: 'value'})
            tmp['variable'] = col
            tmp['time'] = anchor_time
            static_rows.append(tmp[['lon', 'lat', 'time', 'variable', 'value']])
        df_static = pd.concat(static_rows, ignore_index=True)
        df_long = pd.concat([df_long, df_static], ignore_index=True)
        print(f"静态列已追加到长格式: {missing_static_in_varcol}")
    # 过滤数据框，只保留需要的变量
    available_vars = df_long['variable'].unique()
    vars_to_process = [v for v in variables if v in available_vars and v not in static_vars]
    static_vars_present = [v for v in variables if v in available_vars and v in static_vars]
    
    if not vars_to_process:
        raise ValueError(f"在数据中未找到指定的变量。可用变量: {available_vars.tolist()}")
    
    print(f"将处理以下动态变量: {vars_to_process}")
    if static_vars_present:
        print(f"检测到静态变量: {static_vars_present}")
    
    df_filtered = df_long[df_long['variable'].isin(vars_to_process + static_vars_present)].copy()
    
    # 根据输入数据实际时间戳构建时间坐标，避免频率不匹配导致全NaN
    df_filtered = df_filtered[(df_filtered['time'] >= pd.to_datetime(time_range[0])) &
                              (df_filtered['time'] <= pd.to_datetime(time_range[1]))]

    if df_filtered.empty:
        raise ValueError("筛选后无数据，请检查时间范围或变量选择是否正确")

    time_grid = np.sort(df_filtered['time'].unique())
    print(f"时间范围: {pd.to_datetime(time_grid.min())} 到 {pd.to_datetime(time_grid.max())}, 共 {len(time_grid)} 个时间步")
    
    # 获取数据的空间范围
    lon_min, lon_max = df_filtered['lon'].min(), df_filtered['lon'].max()
    lat_min, lat_max = df_filtered['lat'].min(), df_filtered['lat'].max()
    
    print(f"空间范围: 经度 [{lon_min:.4f}, {lon_max:.4f}], 纬度 [{lat_min:.4f}, {lat_max:.4f}]")
    
    # 对齐到网格分辨率
    lon_min_aligned = np.floor(lon_min / resolution) * resolution
    lon_max_aligned = np.ceil(lon_max / resolution) * resolution
    lat_min_aligned = np.floor(lat_min / resolution) * resolution
    lat_max_aligned = np.ceil(lat_max / resolution) * resolution
    
    # 创建网格坐标
    lon_grid = np.arange(lon_min_aligned, lon_max_aligned + resolution/2, resolution)
    lat_grid = np.arange(lat_min_aligned, lat_max_aligned + resolution/2, resolution)
    
    print(f"网格大小: {len(lon_grid)} x {len(lat_grid)} (经度 x 纬度)")
    
    # 为每个点计算最近的网格索引
    df_filtered['lon_idx'] = np.round((df_filtered['lon'] - lon_min_aligned) / resolution).astype(int)
    df_filtered['lat_idx'] = np.round((df_filtered['lat'] - lat_min_aligned) / resolution).astype(int)
    
    # 确保索引在有效范围内
    df_filtered['lon_idx'] = df_filtered['lon_idx'].clip(0, len(lon_grid) - 1)
    df_filtered['lat_idx'] = df_filtered['lat_idx'].clip(0, len(lat_grid) - 1)
    
    # 创建数据变量字典
    data_vars = {}
    
    # 计算并行任务数
    if n_jobs == -1:
        cpu_count = multiprocessing.cpu_count()
        if os.name == 'nt':  # Windows
            actual_jobs = max(1, min(cpu_count - 1, 60))
        else:
            actual_jobs = max(1, min(cpu_count, 64))
    else:
        actual_jobs = n_jobs
    
    print(f"\n使用 {actual_jobs} 个并行任务处理 {len(vars_to_process)} 个变量...")
    
    # 预处理数据：将DataFrame转换为numpy数组，避免pickle问题（动态变量）
    var_data_dict = {}
    for var in vars_to_process:
        var_df = df_filtered[df_filtered['variable'] == var]
        var_data_dict[var] = {
            'values': var_df['value'].values,
            'times': var_df['time'].values,
            'lat_idx': var_df['lat_idx'].values,
            'lon_idx': var_df['lon_idx'].values,
            'config': get_variable_config(var)
        }
    
    # 转换time_grid为numpy数组
    time_grid_np = time_grid  # time_grid已是numpy datetime64数组
    n_time = len(time_grid)
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    
    # 并行处理所有变量
    results = Parallel(n_jobs=actual_jobs, verbose=10)(
        delayed(process_single_variable)(
            var,
            var_data_dict[var]['values'],
            var_data_dict[var]['times'],
            var_data_dict[var]['lat_idx'],
            var_data_dict[var]['lon_idx'],
            time_grid_np,
            n_time, n_lat, n_lon,
            var_data_dict[var]['config'],
            apply_scale
        )
        for var in vars_to_process
    )
    
    # 收集结果
    for var, var_data, config, valid_count in results:
        if valid_count == 0:
            print(f"  警告: 变量 {var} 没有有效数据")
            continue
        
        print(f"  {var}: {valid_count} 个有效点, scale={config['scale_factor']}")
        
        # 创建DataArray
        data_vars[var] = xr.DataArray(
            data=var_data,
            dims=('time', 'latitude', 'longitude'),
            coords={
                'time': time_grid,
                'latitude': lat_grid,
                'longitude': lon_grid
            },
            attrs={
                'units': config['units'],
                'long_name': config['long_name'],
                'scale_factor_applied': config['scale_factor'] if apply_scale else 'not applied',
                'original_fill_values': str(config['input_fill_values']),
                'output_fill_value': config['output_fill_value']
            }
        )
    
    # 处理静态变量（如 DEM）：直接最近邻映射到网格，不随时间变化
    for svar in static_vars_present:
        svar_df = df_filtered[df_filtered['variable'] == svar]
        if svar_df.empty:
            continue
        config = get_variable_config(svar)
        output_fill = config['output_fill_value']

        # 最近邻映射到最近的网格（与动态一致的索引）
        static_grid = np.full((len(lat_grid), len(lon_grid)), output_fill, dtype=np.float32)
        values = svar_df['value'].values.astype(np.float32)
        fill_mask = np.isin(values, config['input_fill_values'])
        for idx in range(len(values)):
            if fill_mask[idx] or np.isnan(values[idx]):
                continue
            i = int(svar_df['lat_idx'].iloc[idx])
            j = int(svar_df['lon_idx'].iloc[idx])
            static_grid[i, j] = values[idx]

        data_vars[svar] = xr.DataArray(
            data=static_grid,
            dims=('latitude', 'longitude'),
            coords={'latitude': lat_grid, 'longitude': lon_grid},
            attrs={
                'units': config['units'],
                'long_name': config['long_name'],
                'scale_factor_applied': config['scale_factor'] if apply_scale else 'not applied',
                'original_fill_values': str(config['input_fill_values']),
                'output_fill_value': config['output_fill_value']
            }
        )

    # 创建Dataset
    ds = xr.Dataset(data_vars)

    # 添加去季节化变量（跳过静态变量和 CLCD 分类变量）
    seasonal_components = {}
    if add_deseasonalized:
        skip_set = set(static_vars_present) | {'CLCD'}
        ds, seasonal_components = add_deseasonalized_variables(
            ds, vars_to_process + static_vars_present, n_jobs=n_jobs, chunk_size=chunk_size, skip_deseasonalize=skip_set
        )
    
    # 添加全局属性
    ds.attrs['title'] = 'Point data converted to NetCDF without interpolation'
    ds.attrs['Conventions'] = 'CF-1.6'
    ds.attrs['institution'] = 'Generated by CSV_to_NC完全版.py'
    ds.attrs['history'] = f'Created on {pd.Timestamp.now()}'
    ds.attrs['source'] = 'Point observations'
    ds.attrs['spatial_resolution'] = f'{resolution} degree'
    ds.attrs['interpolation'] = 'None - direct point to grid mapping'
    ds.attrs['crs'] = 'EPSG:4326'
    
    # 设置坐标属性
    ds['latitude'].attrs = {
        'units': 'degrees_north',
        'long_name': 'latitude',
        'standard_name': 'latitude'
    }
    ds['longitude'].attrs = {
        'units': 'degrees_east',
        'long_name': 'longitude',
        'standard_name': 'longitude'
    }
    ds['time'].attrs = {
        'long_name': 'time',
        'standard_name': 'time'
    }
    
    # 保存到NetCDF
    print(f"\n保存数据到: {output_path}")
    
    # 设置编码 - 使用-9999作为填充值
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': OUTPUT_FILL_VALUE  # 使用-9999
        }
    
    ds.to_netcdf(output_path, encoding=encoding)
    print("保存完成！")
    
    # 如果有季节分量，保存到单独的文件
    if seasonal_components:
        seasonal_path = output_path.replace('.nc', '_seasonal.nc')
        print(f"\n保存季节分量到: {seasonal_path}")
        
        seasonal_ds = xr.Dataset()
        for var, component in seasonal_components.items():
            seasonal_ds[f"{var}_seasonal"] = component
        
        seasonal_encoding = {}
        for var in seasonal_ds.data_vars:
            seasonal_encoding[var] = {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': OUTPUT_FILL_VALUE  # 使用-9999
            }
        
        seasonal_ds.to_netcdf(seasonal_path, encoding=seasonal_encoding)
        print("季节分量保存完成！")
    
    return ds


def process_gdb_to_nc(
    gdb_path: str,
    layer_name: str,
    output_path: str,
    variables: List[str] = None,
    resolution: float = 0.05,
    time_range: Tuple[str, str] = ('2000-01-01', '2024-12-31'),
    apply_scale: bool = True,
    add_deseasonalized: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 1000
) -> xr.Dataset:
    """
    从GDB读取数据并转换为NetCDF
    
    Parameters:
    -----------
    gdb_path : str
        GDB文件路径
    layer_name : str
        图层名称
    output_path : str
        输出NC文件路径
    variables : List[str]
        要处理的变量列表，默认为GOSIF和GOSIFGPP
    resolution : float
        网格分辨率（度）
    time_range : Tuple[str, str]
        时间范围
    apply_scale : bool
        是否应用缩放因子
    add_deseasonalized : bool
        是否添加去季节化变量
    n_jobs : int
        并行任务数
    chunk_size : int
        去季节化处理的块大小
        
    Returns:
    --------
    xr.Dataset
        创建的数据集
    """
    if variables is None:
        variables = ['GOSIF', 'GOSIFGPP']
    
    # 读取GDB数据
    gdf_long = read_and_tidy_gdb(gdb_path, layer_name)
    
    # 转换为DataFrame
    df_long = pd.DataFrame(gdf_long.drop(columns='geometry'))
    
    # 创建NC文件
    ds = create_nc_without_interpolation(
        df_long=df_long,
        variables=variables,
        output_path=output_path,
        resolution=resolution,
        time_range=time_range,
        apply_scale=apply_scale,
        add_deseasonalized=add_deseasonalized,
        n_jobs=n_jobs,
        chunk_size=chunk_size
    )
    
    return ds


def process_csv_to_nc(
    csv_path: str,
    output_path: str,
    variables: List[str] = None,
    resolution: float = 0.05,
    time_range: Tuple[str, str] = ('2000-01-01', '2024-12-31'),
    apply_scale: bool = True,
    add_deseasonalized: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 1000
) -> xr.Dataset:
    """
    从CSV读取数据并转换为NetCDF
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径
    output_path : str
        输出NC文件路径
    variables : List[str]
        要处理的变量列表，默认为GOSIF和GOSIFGPP
    resolution : float
        网格分辨率（度）
    time_range : Tuple[str, str]
        时间范围
    apply_scale : bool
        是否应用缩放因子
    add_deseasonalized : bool
        是否添加去季节化变量
    n_jobs : int
        并行任务数
    chunk_size : int
        去季节化处理的块大小
        
    Returns:
    --------
    xr.Dataset
        创建的数据集
    """
    if variables is None:
        variables = ['GOSIF', 'GOSIFGPP']
    
    # 读取CSV数据
    df_long = read_csv_data(csv_path)
    
    # 创建NC文件
    ds = create_nc_without_interpolation(
        df_long=df_long,
        variables=variables,
        output_path=output_path,
        resolution=resolution,
        time_range=time_range,
        apply_scale=apply_scale,
        add_deseasonalized=add_deseasonalized,
        n_jobs=n_jobs,
        chunk_size=chunk_size
    )
    
    return ds


# =============================================================================
# 去季节化处理函数
# =============================================================================

def stl_decomposition(ts_filled: np.ndarray, period: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行STL分解，带后备方法
    
    Parameters:
    -----------
    ts_filled : np.ndarray
        时间序列数据
    period : int
        季节周期（默认12表示月度数据）
        
    Returns:
    --------
    tuple
        (去季节化序列, 季节分量)
    """
    try:
        ts_clean = np.array(ts_filled, dtype=float)
        
        # 用前向/后向填充处理NaN
        if np.isnan(ts_clean).any():
            ts_series = pd.Series(ts_clean)
            ts_clean = ts_series.fillna(method='ffill').fillna(method='bfill').values
        
        # 如果全是NaN或长度小于周期，返回NaN
        if np.isnan(ts_clean).all() or len(ts_clean) < period:
            return np.full_like(ts_clean, np.nan), np.full_like(ts_clean, np.nan)
        
        # 如果时间序列长度至少有2个周期，尝试STL分解
        if len(ts_clean) >= 2 * period:
            # try:
                stl_result = STL(ts_clean, period=period, seasonal=7, trend=15,
                                 low_pass=15, robust=True, seasonal_deg=0, trend_deg=1).fit()
                
                seasonal = stl_result.seasonal
                trend = stl_result.trend
                residual = stl_result.resid
                
                deseasonalized = trend + residual
                
                if np.isnan(deseasonalized).any() or np.isnan(seasonal).any():
                    raise ValueError("NaN values in STL components")
                
                return deseasonalized, seasonal
        #     except Exception:
        #         pass
        
        # # 后备方法：基于月度气候均值的简单去季节方法
        # monthly_means = np.zeros(period)
        # for month in range(period):
        #     month_data = ts_clean[month::period]
        #     if len(month_data) > 0 and not np.isnan(month_data).all():
        #         monthly_means[month] = np.nanmean(month_data)
        #     else:
        #         monthly_means[month] = 0
        
        # n_years = len(ts_clean) // period + 1
        # seasonal = np.tile(monthly_means, n_years)[:len(ts_clean)]
        # deseasonalized = ts_clean - seasonal
        
        # return deseasonalized, seasonal
    
    except Exception:
        return np.full_like(ts_filled, np.nan), np.full_like(ts_filled, np.nan)


def process_spatial_point(args: tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    处理单个空间点的去季节化
    
    Parameters:
    -----------
    args : tuple
        (i, j, ts, period, flat_idx)
        
    Returns:
    --------
    tuple
        (flat_idx, 去季节化序列, 季节分量)
    """
    i, j, ts, period, flat_idx = args
    
    try:
        if np.isnan(ts).all():
            return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)
        
        ts_series = pd.Series(ts)
        ts_filled = ts_series.fillna(method='ffill').fillna(method='bfill').values
        
        if np.isnan(ts_filled).all():
            return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)
        
        deseasonalized, seasonal = stl_decomposition(ts_filled, period)
        
        return flat_idx, deseasonalized, seasonal
    
    except Exception:
        return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)


def deseasonalize_variable(
    var_data: np.ndarray,
    var_name: str,
    n_jobs: int = -1,
    chunk_size: int = 1000,
    period: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个变量进行去季节化处理
    
    Parameters:
    -----------
    var_data : np.ndarray
        变量数据，形状为 (time, lat, lon)
    var_name : str
        变量名
    n_jobs : int
        并行任务数
    chunk_size : int
        每个块处理的空间点数
    period : int
        季节周期
        
    Returns:
    --------
    tuple
        (去季节化数据, 季节分量)
    """
    # 计算并行任务数
    if n_jobs == -1:
        cpu_count = multiprocessing.cpu_count()
        if os.name == 'nt':  # Windows
            n_jobs = max(1, min(cpu_count - 1, 60))
        else:
            n_jobs = max(1, min(cpu_count, 64))
    
    n_time, n_lat, n_lon = var_data.shape
    total_points = n_lat * n_lon
    
    print(f"  去季节化 {var_name}...使用 {n_jobs} 个并行任务")
    
    # 初始化结果数组
    deseasonalized_data = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    seasonal_component = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    
    # 按块处理
    n_chunks = (total_points + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_points)
        
        if n_chunks > 1:
            print(f"    处理块 {chunk_idx + 1}/{n_chunks}")
        
        # 并行处理块内的空间点
        chunk_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_spatial_point)(
                (flat_idx // n_lon, flat_idx % n_lon, 
                 var_data[:, flat_idx // n_lon, flat_idx % n_lon], period, flat_idx)
            )
            for flat_idx in range(start_idx, end_idx)
        )
        
        # 将结果写回数组
        for flat_idx, deseasonalized, seasonal in chunk_results:
            i = flat_idx // n_lon
            j = flat_idx % n_lon
            deseasonalized_data[:, i, j] = deseasonalized
            seasonal_component[:, i, j] = seasonal
    
    return deseasonalized_data, seasonal_component


def add_deseasonalized_variables(
    ds: xr.Dataset,
    variables: List[str],
    n_jobs: int = -1,
    chunk_size: int = 1000,
    skip_deseasonalize: Optional[set] = None
) -> Tuple[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    为数据集添加去季节化变量
    
    Parameters:
    -----------
    ds : xr.Dataset
        输入数据集
    variables : List[str]
        要处理的变量列表
    n_jobs : int
        并行任务数
    chunk_size : int
        块大小
        
    Returns:
    --------
    tuple
        (处理后的数据集, 季节分量字典)
    """
    print("\n" + "="*60)
    print("添加去季节化变量")
    print("="*60)
    
    seasonal_components = {}
    skip_deseasonalize = skip_deseasonalize or set()
    
    # 筛选实际存在且需要做去季节化的变量
    vars_to_process = [v for v in variables if v in ds.data_vars and v not in skip_deseasonalize]
    skipped = [v for v in variables if v in ds.data_vars and v in skip_deseasonalize]
    
    print(f"将处理 {len(vars_to_process)} 个变量的去季节化")
    if skipped:
        print(f"跳过去季节化: {skipped}")
    
    for var in vars_to_process:
        print(f"\n处理变量: {var}")
        
        var_data = ds[var].values
        # 将输出填充值改为NaN后再做去季节化，避免-9999被当作有效值
        output_fill = ds[var].attrs.get('output_fill_value', OUTPUT_FILL_VALUE)
        var_data = np.where(var_data == output_fill, np.nan, var_data)
        
        # 检查数据有效性
        valid_count = np.sum(~np.isnan(var_data))
        if valid_count == 0:
            print(f"  跳过 {var}：没有有效数据")
            continue
        
        # 执行去季节化
        deseasonalized_data, seasonal_data = deseasonalize_variable(
            var_data, var, n_jobs=n_jobs, chunk_size=chunk_size
        )
        
        # 添加去季节化变量
        ds[f"{var}_deseasonalized"] = xr.DataArray(
            data=deseasonalized_data,
            coords=ds[var].coords,
            dims=ds[var].dims,
            attrs={
                'units': ds[var].attrs.get('units', 'unknown'),
                'long_name': f'Deseasonalized {var}',
                'description': f'Deseasonalized {var} using STL decomposition'
            }
        )
        
        # 保存季节分量
        seasonal_components[var] = xr.DataArray(
            data=seasonal_data,
            coords=ds[var].coords,
            dims=ds[var].dims,
            attrs={'description': f'Seasonal component of {var}'}
        )
        
        print(f"  完成: 添加了 {var}_deseasonalized")
        
        # 释放内存
        del deseasonalized_data, seasonal_data
        gc.collect()
    
    return ds, seasonal_components


def verify_nc_file(nc_path: str):
    """
    验证生成的NetCDF文件
    
    Parameters:
    -----------
    nc_path : str
        NetCDF文件路径
    """
    print(f"\n{'='*60}")
    print(f"验证NetCDF文件: {nc_path}")
    print(f"{'='*60}")
    
    ds = xr.open_dataset(nc_path)
    
    print(f"\n维度: {dict(ds.dims)}")
    print(f"\n坐标:")
    for coord in ds.coords:
        print(f"  - {coord}: {ds[coord].shape}")
    
    print(f"\n数据变量:")
    for var in ds.data_vars:
        data = ds[var]
        valid_count = np.sum(~np.isnan(data.values))
        total_count = data.size
        valid_ratio = valid_count / total_count * 100
        
        print(f"\n  {var}:")
        print(f"    - 形状: {data.shape}")
        print(f"    - 有效值比例: {valid_ratio:.2f}% ({valid_count}/{total_count})")
        
        valid_data = data.values[~np.isnan(data.values)]
        if len(valid_data) > 0:
            print(f"    - 最小值: {np.min(valid_data):.6f}")
            print(f"    - 最大值: {np.max(valid_data):.6f}")
            print(f"    - 均值: {np.mean(valid_data):.6f}")
        
        print(f"    - 属性: {dict(data.attrs)}")
    
    print(f"\n全局属性:")
    for attr, value in ds.attrs.items():
        print(f"  - {attr}: {value}")
    
    ds.close()


def main():
    """主函数"""
    
    # =========================================================================
    # 用户配置区域 - 根据需要修改以下参数
    # =========================================================================
    
    # 输入数据路径（支持GDB或CSV）
    INPUT_PATH = "./test.gdb"  # GDB路径
    LAYER_NAME = "NLH01点"  # GDB图层名（如果是CSV则忽略此参数）
    
    # 或者使用CSV
    # INPUT_PATH = "./your_data.csv"
    
    # 输出路径
    OUTPUT_PATH = "./NLH01_dataset_small.nc"
    
    # 要处理的变量列表
    # 处理全部变量，GOSIF和GOSIFGPP会自动应用特殊的scale factor和fill values
    VARIABLES = ALL_VARIABLES
    
    # 网格分辨率（度）
    RESOLUTION = 0.1
    
    # 时间范围
    TIME_RANGE = ('2003-01-01', '2024-12-31')
    
    # 是否应用缩放因子
    APPLY_SCALE = True
    
    # 是否添加去季节化变量 {var}_deseasonalized
    ADD_DESEASONALIZED = True
    
    # 并行任务数（-1表示自动选择）
    N_JOBS = -1
    
    # 去季节化处理的块大小
    CHUNK_SIZE = 1000*5
    
    # =========================================================================
    # 执行处理
    # =========================================================================
    
    print("="*60)
    print("CSV/GDB to NetCDF 转换工具（无插值版 + 去季节化）")
    print("="*60)
    print(f"\n输入路径: {INPUT_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"变量: {VARIABLES}")
    print(f"分辨率: {RESOLUTION}度")
    print(f"时间范围: {TIME_RANGE}")
    print(f"应用缩放: {APPLY_SCALE}")
    print(f"添加去季节化: {ADD_DESEASONALIZED}")
    
    # 判断输入类型并处理
    if INPUT_PATH.endswith('.gdb'):
        ds = process_gdb_to_nc(
            gdb_path=INPUT_PATH,
            layer_name=LAYER_NAME,
            output_path=OUTPUT_PATH,
            variables=VARIABLES,
            resolution=RESOLUTION,
            time_range=TIME_RANGE,
            apply_scale=APPLY_SCALE,
            add_deseasonalized=ADD_DESEASONALIZED,
            n_jobs=N_JOBS,
            chunk_size=CHUNK_SIZE
        )
    elif INPUT_PATH.endswith('.csv'):
        ds = process_csv_to_nc(
            csv_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            variables=VARIABLES,
            resolution=RESOLUTION,
            time_range=TIME_RANGE,
            apply_scale=APPLY_SCALE,
            add_deseasonalized=ADD_DESEASONALIZED,
            n_jobs=N_JOBS,
            chunk_size=CHUNK_SIZE
        )
    else:
        raise ValueError(f"不支持的文件格式: {INPUT_PATH}")
    
    # 验证输出文件
    verify_nc_file(OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == '__main__':
    main()
