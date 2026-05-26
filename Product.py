import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.netcdf as netcdf
import matplotlib.colors as mcolors

# ================= 配置区域 =================
# 网格分辨率 (度)
GRID_RES = 0.1 # 可以设置为 0.01, 0.05, 0.1 等

# 尝试读取配置文件，如果失败则使用默认或抛出错误
try:
    with open('./model_config.json', 'r') as f:
        config = json.load(f)
    # 根据您的描述，动态获取路径
    
    # /mnt/c/Users/szw/Desktop/NLH_ZH - 副本/Result/CausalResult_IterTrain_GPP_desasonalized/causal_analysis_all_5year/2005_2009
    INPUT_DIR = f"{config['analyze']['OUT_DIR']}/causal_analysis_all_5year/2005_2009/point_jsons"
    OUTPUT_DIR = f"{config['analyze']['OUT_DIR']}/causal_analysis_all_5year/2005_2009/spatial_analysis_results_fig2"
    TARGET_VAR = config['analyze']['TARGET']
except Exception as e:
    print(f"配置文件读取失败或路径错误，使用默认回退配置: {e}")
    # 回退配置（方便单独测试）
    INPUT_DIR = 'point_jsons'
    OUTPUT_DIR = 'spatial_analysis_results_fig_standard'
    TARGET_VAR = 'EVI_deseasonalized' 

# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_name(name):
    """标准化变量名，去除后缀"""
    return name.replace('_deseasonalized', '').replace('_sum', '').replace('total_', '')

# 获取清洗后的目标变量名
TARGET_LABEL = clean_name(TARGET_VAR)
print(f"当前分析目标变量: {TARGET_LABEL} (原始: {TARGET_VAR})")

# 1. 数据加载与处理
print(f"正在读取 {INPUT_DIR} 中的文件...")
if not os.path.exists(INPUT_DIR):
    print(f"错误: 输入目录 {INPUT_DIR} 不存在!")
    exit()

files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
data_records = []

for file in files:
    try:
        with open(os.path.join(INPUT_DIR, file), 'r') as f:
            content = json.load(f)
            
        lat, lon = content['lat'], content['lon']
        edges = content['causal_edges']
        
        target_drivers = [] # Others -> TARGET
        target_driven = []  # TARGET -> Others
        pair_info = {}
        
        for edge in edges:
            u = clean_name(edge['cause_name'])
            v = clean_name(edge['effect_name'])
            score = edge['score']
            lag = edge['lag']
            
            # 记录每对关系 (Map 2 & 3 用)
            pair_info[f"{u}_to_{v}_score"] = score
            pair_info[f"{u}_to_{v}_lag"] = lag
            
            # 筛选主导因子 (Map 1 & 4 用)
            if edge['cause_name'] == TARGET_VAR:
                # 物理约束: 排除目标变量驱动降水
                if 'precipitation' not in v.lower(): 
                    target_driven.append({'var': v, 'score': score, 'lag': lag})
            
            if edge['effect_name'] == TARGET_VAR:
                target_drivers.append({'var': u, 'score': score, 'lag': lag})
                
        # Map 1: TARGET -> Dominant Target
        if target_driven:
            best = max(target_driven, key=lambda x: x['score'])
            rec1 = {'dom_driven_type': best['var'], 'dom_driven_score': best['score'], 'dom_driven_lag': best['lag']}
        else:
            rec1 = {'dom_driven_type': 'None', 'dom_driven_score': 0, 'dom_driven_lag': 0}
            
        # Map 4: Dominant Driver -> TARGET
        if target_drivers:
            best = max(target_drivers, key=lambda x: x['score'])
            rec4 = {'dom_driver_type': best['var'], 'dom_driver_score': best['score'], 'dom_driver_lag': best['lag']}
        else:
            rec4 = {'dom_driver_type': 'None', 'dom_driver_score': 0, 'dom_driver_lag': 0}
            
        record = {'lat': lat, 'lon': lon, **rec1, **rec4, **pair_info}
        data_records.append(record)
        
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")

df = pd.DataFrame(data_records)
print(f"数据加载完成: {df.shape}")

# 2. 核心功能：保存标准网格 NetCDF (ArcGIS 兼容)
def save_combined_nc(df, lat_col, lon_col, data_dict, filename, grid_res=GRID_RES):
    """
    保存为标准网格 NC 文件，使用等间距网格以兼容 ArcGIS
    
    Args:
        grid_res: 网格分辨率 (度)，默认使用全局 GRID_RES
    """
    df = df.copy()
    
    # 根据分辨率计算小数位数
    decimals = max(int(-np.log10(grid_res)), 0) + 2  # 增加精度以避免浮点误差
    
    # 四舍五入到最近的网格点
    df['lat_idx'] = (df[lat_col] / grid_res).round() * grid_res
    df['lon_idx'] = (df[lon_col] / grid_res).round() * grid_res
    
    # 确保精度
    df['lat_idx'] = df['lat_idx'].round(decimals)
    df['lon_idx'] = df['lon_idx'].round(decimals)
    
    # 获取数据边界
    lat_min = df['lat_idx'].min()
    lat_max = df['lat_idx'].max()
    lon_min = df['lon_idx'].min()
    lon_max = df['lon_idx'].max()
    
    # 创建完整的等间距网格 (关键修复: ArcGIS 需要等间距坐标)
    n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
    n_lon = int(round((lon_max - lon_min) / grid_res)) + 1
    
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    
    # 确保精确的等间距 (重要!)
    lats = np.round(lats, decimals).astype(np.float64)
    lons = np.round(lons, decimals).astype(np.float64)
    
    # 验证等间距
    lat_diffs = np.diff(lats)
    lon_diffs = np.diff(lons)
    print(f"[DEBUG] 网格分辨率: {grid_res}度")
    print(f"[DEBUG] 网格大小: lat={len(lats)}, lon={len(lons)}")
    print(f"[DEBUG] lat 范围: [{lats.min():.6f}, {lats.max():.6f}]")
    print(f"[DEBUG] lon 范围: [{lons.min():.6f}, {lons.max():.6f}]")
    print(f"[DEBUG] lat 间距: min={lat_diffs.min():.6f}, max={lat_diffs.max():.6f}, std={lat_diffs.std():.2e}")
    print(f"[DEBUG] lon 间距: min={lon_diffs.min():.6f}, max={lon_diffs.max():.6f}, std={lon_diffs.std():.2e}")
    
    fill_value = -9999.0
    
    with netcdf.netcdf_file(filename, 'w') as f:
        f.createDimension('lat', len(lats))
        f.createDimension('lon', len(lons))
        
        lat_v = f.createVariable('lat', 'd', ('lat',))  # 使用 double 精度
        lat_v[:] = lats
        lat_v.units = 'degrees_north'
        lat_v.standard_name = 'latitude'
        lat_v.axis = 'Y'
        
        lon_v = f.createVariable('lon', 'd', ('lon',))  # 使用 double 精度
        lon_v[:] = lons
        lon_v.units = 'degrees_east'
        lon_v.standard_name = 'longitude'
        lon_v.axis = 'X'
        
        for nc_name, df_col in data_dict.items():
            # 使用 pivot_table 将数据映射到网格
            grid = df.pivot_table(index='lat_idx', columns='lon_idx', values=df_col, aggfunc='mean')
            # reindex 到完整的等间距网格
            grid = grid.reindex(index=lats, columns=lons)
            
            print(f"[DEBUG] {nc_name}: 有效值={grid.notna().sum().sum()}, NaN={grid.isna().sum().sum()}")
            
            v = f.createVariable(nc_name, 'f', ('lat', 'lon'))
            v.missing_value = np.float32(fill_value)
            v._FillValue = np.float32(fill_value)
            
            data = grid.fillna(fill_value).values.astype(np.float32)
            v[:] = data
            
            print(f"[DEBUG] {nc_name}: 写入范围=[{data[data != fill_value].min() if (data != fill_value).any() else 'N/A'}, {data[data != fill_value].max() if (data != fill_value).any() else 'N/A'}]")
            
    print(f"已保存 NC: {filename}")

# 3. 核心功能：绘制组合图
def plot_combined_map(df, type_col, score_col, lag_col, title_main, filename_base):
    # 准备数据：分类变量数字化
    cats = df[type_col].dropna().unique()
    cat_map = {c: i for i, c in enumerate(cats)}
    df_plot = df.copy()
    df_plot['type_code'] = df_plot[type_col].map(cat_map)
    
    # 保存组合 NC 文件 (包含三个变量)
    nc_path = os.path.join(OUTPUT_DIR, f"{filename_base}.nc")
    save_combined_nc(df_plot, 'lat', 'lon', 
                     {'var_type_code': 'type_code', 'score': score_col, 'lag': lag_col}, 
                     nc_path)
    
    # 绘图 (PNG 仅作预览，主要使用 NC)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle(title_main, fontsize=16, fontweight='bold')
    
    # 子图1: 类型
    ax = axes[0]
    cmap_cat = plt.cm.get_cmap('tab10', len(cats)) if len(cats) > 0 else 'gray'
    sc1 = ax.scatter(df['lon'], df['lat'], c=df_plot['type_code'], cmap=cmap_cat, s=40, marker='s')
    ax.set_title('(a) Dominant Variable Type', fontsize=12)
    if len(cats) > 0:
        cbar1 = plt.colorbar(sc1, ax=ax, ticks=range(len(cats)), shrink=0.6)
        cbar1.ax.set_yticklabels(cats)
    
    # 子图2: 强度
    ax = axes[1]
    sc2 = ax.scatter(df['lon'], df['lat'], c=df[score_col], cmap='viridis', s=40, marker='s')
    ax.set_title('(b) Causal Strength (Score)', fontsize=12)
    plt.colorbar(sc2, ax=ax, shrink=0.6)
    
    # 子图3: 滞后
    ax = axes[2]
    sc3 = ax.scatter(df['lon'], df['lat'], c=df[lag_col], cmap='plasma', s=40, marker='s')
    ax.set_title('(c) Time Lag (Steps)', fontsize=12)
    plt.colorbar(sc3, ax=ax, shrink=0.6)
    
    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), dpi=300)
    plt.close()
    print(f"已保存组合图预览: {filename_base}.png")

# 4. 单图绘制函数
def plot_single_map(df, val_col, title, filename, cmap='viridis'):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['lon'], df['lat'], c=df[val_col], cmap=cmap, s=40, marker='s')
    plt.colorbar(sc, label='Value')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.close()
    
    # 保存单个 NC
    save_combined_nc(df, 'lat', 'lon', {'value': val_col}, os.path.join(OUTPUT_DIR, f"{filename}.nc"))

# ================= 执行任务 =================

if not df.empty:
    # 任务 1: Map 1 (组合图 + 组合NC)
    plot_combined_map(
        df, 
        type_col='dom_driven_type', 
        score_col='dom_driven_score', 
        lag_col='dom_driven_lag',
        title_main=f'Map 1: Dominant Factors Driven by {TARGET_LABEL}',
        filename_base=f'Map1_{TARGET_LABEL}_Driven_Dominant_Combined'
    )

    # 任务 4: Map 4 (组合图 + 组合NC)
    plot_combined_map(
        df, 
        type_col='dom_driver_type', 
        score_col='dom_driver_score', 
        lag_col='dom_driver_lag',
        title_main=f'Map 4: Dominant Drivers of {TARGET_LABEL}',
        filename_base=f'Map4_{TARGET_LABEL}_Driver_Dominant_Combined'
    )

    # 任务 2 & 3: 两两敏感性 (单图)
    cols = [c for c in df.columns if '_to_' in c and '_score' in c]

    for col in cols:
        parts = col.split('_to_')
        src = parts[0]
        dst = parts[1].replace('_score', '')
        
        # Map 2: TARGET -> Others
        if src == TARGET_LABEL:
            if 'precipitation' in dst.lower(): continue 
            
            plot_single_map(df, col, f'Map 2: Strength ({TARGET_LABEL} -> {dst})', f'Map2_Strength_{TARGET_LABEL}_to_{dst}')
            lag_col = col.replace('_score', '_lag')
            if lag_col in df.columns:
                plot_single_map(df, lag_col, f'Map 2: Time Lag ({TARGET_LABEL} -> {dst})', f'Map2_Lag_{TARGET_LABEL}_to_{dst}', cmap='plasma')
                
        # Map 3: Others -> TARGET
        elif dst == TARGET_LABEL:
            plot_single_map(df, col, f'Map 3: Strength ({src} -> {TARGET_LABEL})', f'Map3_Strength_{src}_to_{TARGET_LABEL}')
            lag_col = col.replace('_score', '_lag')
            if lag_col in df.columns:
                plot_single_map(df, lag_col, f'Map 3: Time Lag ({src} -> {TARGET_LABEL})', f'Map3_Lag_{src}_to_{TARGET_LABEL}', cmap='plasma')

    print("所有处理完成。")
else:
    print("没有数据被处理，请检查输入目录或 JSON 文件格式。")