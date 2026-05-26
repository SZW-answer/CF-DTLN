import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import os
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 本脚本：并行化的时空数据插值与去季节化处理
# 说明：对原始代码逐段添加中文注释，解释每个函数、参数、返回值和关键实现逻辑。
# 请注意：注释仅用于说明，不改变原有代码行为。
# -----------------------------------------------------------------------------

def spatial_interpolate_slice(slice_data):
    """
    Perform spatial interpolation on a single time slice
    
    Parameters:
    -----------
    slice_data : numpy.ndarray
        2D array with NaN values to be filled
    
    Returns:
    --------
    numpy.ndarray
        Filled 2D array
    """
    # 如果整张切片全部为 NaN 或者没有 NaN，直接返回原切片（无需插值）
    if np.isnan(slice_data).all() or not np.isnan(slice_data).any():
        return slice_data
    
    # 复制切片并构造 NaN 掩码，避免修改输入的原始数组
    filled_slice = slice_data.copy()
    nan_mask = np.isnan(slice_data)
    
    # 找到所有非 NaN 点的索引以及对应的有效值（用于最近邻填充）
    y_indices, x_indices = np.where(~nan_mask)
    valid_values = slice_data[~nan_mask]
    
    # 找到所有 NaN 的像元索引
    nan_y_indices, nan_x_indices = np.where(nan_mask)
    
    # 对每个 NaN 像元，使用暴力最近邻方法找到最近的有效像元并填值
    # 注意：这种方法在大尺度栅格上计算开销大，但实现简单且无需额外依赖
    for i in range(len(nan_y_indices)):
        y, x = nan_y_indices[i], nan_x_indices[i]
        # 计算到所有有效点的距离（向量化实现）
        dy = y - y_indices
        dx = x - x_indices
        distances = np.sqrt(dy*dy + dx*dx)
        # 选取最近点索引并用其值填充
        min_idx = np.argmin(distances)
        filled_slice[y, x] = valid_values[min_idx]
    
    return filled_slice

def stl_decomposition(ts_filled, period=12):
    """
    Perform STL decomposition with fallback methods
    
    Parameters:
    -----------
    ts_filled : numpy.ndarray
        Time series data
    period : int
        Seasonal period (default: 12 for monthly data)
    
    Returns:
    --------
    tuple
        (deseasonalized, seasonal) components
    """
    try:
        # 将输入转换为浮点数组，便于数值计算
        ts_clean = np.array(ts_filled, dtype=float)

        # 如果还有 NaN，先用前向/后向填充填补
        if np.isnan(ts_clean).any():
            ts_series = pd.Series(ts_clean)
            ts_clean = ts_series.fillna(method='ffill').fillna(method='bfill').values

        # 如果全部是 NaN 或长度小于季节长度，返回全 NaN（无法做分解）
        if np.isnan(ts_clean).all() or len(ts_clean) < period:
            return np.full_like(ts_clean, np.nan), np.full_like(ts_clean, np.nan)

        # 若时间序列长度至少有 2 个周期，优先尝试 STL 分解
        if len(ts_clean) >= 2 * period:
            try:
                # 调用 statsmodels 的 STL，使用一组经验参数（可根据数据调优）
                stl_result = STL(ts_clean, period=period, seasonal=7, trend=15, 
                                 low_pass=15, robust=True, seasonal_deg=0, trend_deg=1).fit()

                seasonal = stl_result.seasonal
                trend = stl_result.trend
                residual = stl_result.resid

                # 去季节化取趋势 + 残差
                deseasonalized = trend + residual

                # 若结果包含 NaN 则抛出异常，回退到气候平均法
                if np.isnan(deseasonalized).any() or np.isnan(seasonal).any():
                    raise ValueError("NaN values in STL components")

                return deseasonalized, seasonal

            except Exception:
                # 若 STL 失败，则继续使用后备方法
                pass

        # 后备：基于月度气候均值的简单去季节方法（适用于短序列或 STL 失败）
        monthly_means = np.zeros(period)

        # 计算每个月份的均值（忽略 NaN）
        for month in range(period):
            month_data = ts_clean[month::period]
            if len(month_data) > 0 and not np.isnan(month_data).all():
                monthly_means[month] = np.nanmean(month_data)
            else:
                monthly_means[month] = 0

        # 将月度均值重复以匹配时间序列长度并计算去季节化序列
        n_years = len(ts_clean) // period + 1
        seasonal = np.tile(monthly_means, n_years)[:len(ts_clean)]
        deseasonalized = ts_clean - seasonal

        return deseasonalized, seasonal

    except Exception:
        # 最终兜底：在任何不可预见错误时返回 NaN 数组
        return np.full_like(ts_filled, np.nan), np.full_like(ts_filled, np.nan)

def process_spatial_point(args):
    """
    Process a single spatial point for deseasonalization
    
    Parameters:
    -----------
    args : tuple
        (i, j, ts, period, flat_idx)
    
    Returns:
    --------
    tuple
        (flat_idx, deseasonalized, seasonal)
    """
    i, j, ts, period, flat_idx = args

    try:
        # 如果时间序列全为 NaN，直接返回全 NaN（跳过）
        if np.isnan(ts).all():
            return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)

        # 用前向/后向填充处理残余 NaN
        ts_series = pd.Series(ts)
        ts_filled = ts_series.fillna(method='ffill').fillna(method='bfill').values

        # 如果填充后仍全为 NaN，返回全 NaN
        if np.isnan(ts_filled).all():
            return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)

        # 对该空间点执行 STL 或气候均值去季节分解
        deseasonalized, seasonal = stl_decomposition(ts_filled, period)

        return flat_idx, deseasonalized, seasonal

    except Exception:
        # 发生异常时返回 NaN 以保证并行流程不中断
        return flat_idx, np.full_like(ts, np.nan), np.full_like(ts, np.nan)

def process_spatial_chunk(args):
    """
    Process a chunk of spatial points
    
    Parameters:
    -----------
    args : tuple
        (var_name, var_data, start_idx, end_idx, period, n_lat, n_lon)
    
    Returns:
    --------
    list
        List of tuples: (flat_idx, deseasonalized, seasonal)
    """
    var_name, var_data, start_idx, end_idx, period, n_lat, n_lon = args

    results = []
    # 遍历扁平化的空间索引区间（按列优先或行优先取决于计算方式），并处理每个点
    for flat_idx in range(start_idx, end_idx):
        i = flat_idx // n_lon
        j = flat_idx % n_lon

        ts = var_data[:, i, j]
        result = process_spatial_point((i, j, ts, period, flat_idx))
        results.append(result)

    return results

def interpolate_and_deseasonalize_parallel(ds, variables=None, n_jobs=-1, chunk_size=1000):
    """
    Perform interpolation and deseasonalization on spatiotemporal data with parallel processing
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset
    variables : list
        List of variables to process (default: all time-dependent data variables)
    n_jobs : int
        Number of parallel jobs (-1 means all available cores)
    chunk_size : int
        Number of spatial points to process in each chunk
        
    Returns:
    --------
    xarray.Dataset
        Processed dataset with interpolated and deseasonalized variables
    dict
        Dictionary containing seasonal components
    """
    # 计算要使用的并行任务数：如果 n_jobs==-1，使用机器可用核心数并设置安全上限
    # 在 Windows 下，loky/backend 会为每个 worker 创建若干句柄，
    # Windows 的 WaitForMultipleObjects 对象句柄数量上限为 63（含管理句柄），
    # 因此需要确保并发 worker 数量不要导致超过该限制。常用经验：限制为 cpu_count()-1 或不超过 60。
    if n_jobs == -1:
        cpu_count = multiprocessing.cpu_count()
        # 在 Windows 上更加保守：保留一个主进程句柄，且不要超过 60
        if os.name == 'nt':
            n_jobs = max(1, min(cpu_count - 1, 60))
        else:
            # 类 Unix 系统通常可以使用全部核数，但仍设置一个合理上限
            n_jobs = max(1, min(cpu_count, 64))

    print(f"Using {n_jobs} parallel jobs for processing")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    # If no variables specified, use all time-dependent variables
    if variables is None:
        variables = [var for var in ds.data_vars 
                    if 'time' in ds[var].dims and len(ds[var].dims) >= 3]
    
    print(f"Processing {len(variables)} variables: {', '.join(variables)}")
    
    # 复制数据集（注意：复制可能会增加内存占用，若数据很大建议使用 dask-chunks）
    processed_ds = ds.copy()
    seasonal_components = {}
    
    # 第一步：时间插值（沿 time 维度进行线性插值）
    print("\nPerforming temporal interpolation...")
    # 这里按变量顺序进行时间插值（xarray 的 interpolate_na 内部已向量化）
    for var in variables:
        print(f"  Temporal interpolating {var}...")
        # 如果数据是 dask-backed（有 chunks），需要在 time 维度上合并为单个 chunk
        # 否则 xarray 在内部使用 apply_ufunc with dask='parallelized' 时会报错。
        da = processed_ds[var]
        try:
            # 仅对存在 chunk 的数组进行 rechunk，减小内存和计算不确定性
            if hasattr(da.data, 'chunks') and da.data.chunks is not None:
                da = da.chunk({'time': -1})
        except Exception:
            # 若 rechunk 失败则继续使用原始数据（可能会触发错误）
            pass

        # 执行时间插值
        processed_ds[var] = da.interpolate_na(dim='time', method='linear')
    
    # 第二步：空间插值（对每个时间切片进行最近邻填充）
    print("\nPerforming spatial interpolation...")

    for var in variables:
        print(f"  Spatial interpolating {var}...")
        var_data = processed_ds[var].values
        n_time, n_lat, n_lon = var_data.shape

        # 用 joblib 并行处理每个时间切片以加速空间填充
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(spatial_interpolate_slice)(var_data[t])
            for t in range(n_time)
        )

        # 将填充后的切片写回数据集中
        for t, filled_slice in enumerate(results):
            processed_ds[var].values[t] = filled_slice
    
    # 第三步：去季节化。由于每个空间点的时间序列可独立处理，采用分块 + 并行的方式减少内存峰值
    print("\nPerforming deseasonalization...")

    for var in variables:
        print(f"  Deseasonalizing {var}...")

        # 获取变量数据并计算空间尺寸
        var_data = processed_ds[var].values
        n_time, n_lat, n_lon = var_data.shape
        total_points = n_lat * n_lon

        # 为去季节化结果创建占位数组（初始化为 NaN）
        deseasonalized_data = np.full((n_time, n_lat, n_lon), np.nan)
        seasonal_component = np.full((n_time, n_lat, n_lon), np.nan)

        # 按块处理以控制内存使用
        n_chunks = (total_points + chunk_size - 1) // chunk_size

        print(f"    Processing {total_points} spatial points in {n_chunks} chunks...")

        # 对每个块：为块内的每个空间点并行执行 stl 分解
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_points)
            print(f"    Processing chunk {chunk_idx + 1}/{n_chunks} (points {start_idx} to {end_idx})")

            # 对块内的每个扁平索引并行调用 process_spatial_point
            chunk_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(process_spatial_point)(
                    (flat_idx // n_lon, flat_idx % n_lon, var_data[:, flat_idx // n_lon, flat_idx % n_lon], 12, flat_idx)
                )
                for flat_idx in range(start_idx, end_idx)
            )

            # 将块结果写回到主数组中
            for flat_idx, deseasonalized, seasonal in chunk_results:
                i = flat_idx // n_lon
                j = flat_idx % n_lon
                deseasonalized_data[:, i, j] = deseasonalized
                seasonal_component[:, i, j] = seasonal

        # 将去季节化结果添加为新的 DataArray 并保留原有坐标/维度信息
        processed_ds[f"这俩"] = xr.DataArray(
            deseasonalized_data,
            coords=processed_ds[var].coords,
            dims=processed_ds[var].dims,
            attrs=processed_ds[var].attrs
        )

        # 将季节分量存入字典，随后统一写入 NetCDF
        seasonal_components[var] = xr.DataArray(
            seasonal_component,
            coords=processed_ds[var].coords,
            dims=processed_ds[var].dims,
            attrs={'description': f'Seasonal component of {var}'}
        )

        # 释放临时变量并进行 GC，减小内存占用峰值
        del deseasonalized_data, seasonal_component, var_data
        gc.collect()
    
    return processed_ds, seasonal_components

# Define variable list
ls = [
    'SMs', 'E', 'Eb', 'Ec', 'Ei', 'Ep', 'Ep_aero', 'Ep_rad', 'Es', 'Et', 'Ew', 'H', 'S', 'SMrz', 'TWS', 'GA', 
    'evaporation_from_the_top_of_canopy_sum', 'potential_evaporation_sum', 'runoff_sum', 'surface_latent_heat_flux_sum', 
    'surface_net_thermal_radiation_sum', 'temperature_2m', 'total_evaporation_sum', 'total_precipitation_sum', 
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 
    'volumetric_soil_water_layer_4', 'Precipitation', 'NDWI', 'EVI', 'FPAR', 'GPP', 'LAI', 'LST', 'NDSI', 'NDVI', 
    'GOSIFGPP', 'GOSIF', 'TWS_tavg', 'GWS_tavg'
]

# Function to create visualization directory if it doesn't exist
def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Main execution block
def main():
    try:
        # Load the NC file
        data_path = './NLH_010_dataset_small.nc'  # Update with your file path
        print("Loading dataset...")
        
        # Use chunks to avoid memory issues
        data = xr.open_dataset(data_path, chunks={'time': 100})
        
        print(f"Dataset dimensions: {dict(data.dims)}")
        print(f"Available variables: {list(data.data_vars)}")
        
        # Create output directory for visualizations
        output_dir = 'preprocessing_results_010'
        ensure_dir(output_dir)
        
        # Run interpolation and deseasonalization with parallel processing
        print("\nStarting parallel preprocessing...")
        processed_data, seasonal_components = interpolate_and_deseasonalize_parallel(
            data, variables=ls, n_jobs=-1, chunk_size=2000  # More conservative chunk size
        )
        
        # Preserve DEM if it exists
        if 'DEM' in data.variables:
            print("Preserving DEM variable...")
            processed_data['DEM'] = data['DEM']
        
        # Save processed dataset to new NetCDF file
        print("\nSaving processed data...")
        processed_data.to_netcdf(os.path.join(output_dir, 'processed_data.nc'))
        print("Processed data saved successfully")
        
        # Create a separate dataset for seasonal components
        seasonal_ds = xr.Dataset()
        for var, component in seasonal_components.items():
            seasonal_ds[f"{var}_seasonal"] = component
        
        if 'DEM' in data.variables:
            seasonal_ds['DEM'] = data['DEM']
        
        print("Saving seasonal components...")
        seasonal_ds.to_netcdf(os.path.join(output_dir, 'seasonal_components.nc'))
        print("Seasonal components saved successfully")
        
        # Visualize example results for key variables
        print("\nCreating visualizations...")
        plot_vars = ['NDVI', 'Precipitation', 'GA']
        
        for var in plot_vars:
            if var in processed_data.data_vars:
                try:
                    print(f"Creating visualization for {var}...")
                    
                    # Select a sample point (middle of the grid)
                    lat_idx = processed_data.dims['latitude'] // 2
                    lon_idx = processed_data.dims['longitude'] // 2
                    
                    # Get coordinates for this point
                    lat = processed_data.latitude.values[lat_idx]
                    lon = processed_data.longitude.values[lon_idx]
                    
                    # Create figure
                    plt.figure(figsize=(15, 8))
                    
                    # Plot original time series
                    plt.subplot(3, 1, 1)
                    processed_data[var].isel(latitude=lat_idx, longitude=lon_idx).plot()
                    plt.title(f'Original {var} at ({lat:.2f}, {lon:.2f})')
                    plt.xlabel('Time')
                    plt.ylabel(var)
                    
                    # Plot seasonal component
                    plt.subplot(3, 1, 2)
                    seasonal_components[var].isel(latitude=lat_idx, longitude=lon_idx).plot()
                    plt.title(f'Seasonal Component of {var}')
                    plt.xlabel('Time')
                    plt.ylabel('Seasonal Component')
                    
                    # Plot deseasonalized time series
                    plt.subplot(3, 1, 3)
                    processed_data[f"{var}_deseasonalized"].isel(latitude=lat_idx, longitude=lon_idx).plot()
                    plt.title(f'Deseasonalized {var}')
                    plt.xlabel('Time')
                    plt.ylabel('Deseasonalized Value')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{var}_preprocessing_example.png'))
                    plt.close()
                    
                    # Create spatial maps for a specific time point
                    time_idx = processed_data.dims['time'] // 2  # Middle of the time range
                    
                    plt.figure(figsize=(15, 5))
                    
                    # Plot original data
                    plt.subplot(1, 3, 1)
                    processed_data[var].isel(time=time_idx).plot(cmap='viridis')
                    plt.title(f'Original {var}')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    
                    # Plot seasonal component
                    plt.subplot(1, 3, 2)
                    seasonal_components[var].isel(time=time_idx).plot(cmap='viridis')
                    plt.title(f'Seasonal Component')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    
                    # Plot deseasonalized data
                    plt.subplot(1, 3, 3)
                    processed_data[f"{var}_deseasonalized"].isel(time=time_idx).plot(cmap='viridis')
                    plt.title(f'Deseasonalized {var}')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{var}_spatial_maps.png'))
                    plt.close()
                    
                except Exception as e:
                    print(f"Error creating visualization for {var}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Variable {var} not found in processed data. Skipping visualization.")
        
        # If CLCD exists in the data, create a land cover map
        if 'CLCD' in data.variables:
            try:
                print("Creating land cover map...")
                plt.figure(figsize=(10, 8))
                
                # If CLCD is time-varying, use the most recent time point
                if 'time' in data['CLCD'].dims:
                    data['CLCD'].isel(time=-1).plot(cmap='tab20')
                else:
                    data['CLCD'].plot(cmap='tab20')
                
                plt.title('Land Cover Classification')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.savefig(os.path.join(output_dir, 'land_cover_map.png'))
                plt.close()
            except Exception as e:
                print(f"Error creating land cover map: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\nPreprocessing complete!")
        print(f"Results saved to {output_dir}/")
        print(f"Processed data saved to {output_dir}/processed_data.nc")
        print(f"Seasonal components saved to {output_dir}/seasonal_components.nc")
        
    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()