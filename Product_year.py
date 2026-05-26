import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.netcdf as netcdf
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings('ignore')

# ================= Configuration Area =================
# Grid Resolution (Degrees)
GRID_RES = 0.1 # Can be set to 0.01, 0.05, 0.1 etc.

# Define Time Periods (Consistent with interpreted_5year.py)
PERIODS = [
    "2003_2004",
    "2005_2009",
    "2010_2014",
    "2015_2019",
    "2020_2024"
]

# Attempt to read configuration file
try:
    # Try model_config_SIF.json first (based on train.py)
    config_path = './model_config.json'
  
        
    with open(config_path, 'rb') as f:
        config = json.load(f)
    
    BASE_OUT_DIR = config['analyze']['OUT_DIR']
    TARGET_VAR = config['analyze']['TARGET']
    print(f"Configuration file loaded successfully: {config_path}")
    print(f"Output Root Directory: {BASE_OUT_DIR}")
    print(f"Target Variable: {TARGET_VAR}")
    
except Exception as e:
    print(f"Failed to read configuration file or path error, using default fallback config: {e}")
    # Fallback configuration (for standalone testing)
    BASE_OUT_DIR = './Result' 
    TARGET_VAR = 'EVI_deseasonalized'

# ================= Helper Functions =================

def clean_name(name):
    """Standardize variable names, remove suffixes"""
    if not isinstance(name, str): return str(name)
    return name.replace('_deseasonalized', '').replace('_sum', '').replace('total_', '')

def save_combined_nc(df, lat_col, lon_col, data_dict, filename, grid_res=GRID_RES):
    """
    Save as standard grid NC file, using equally spaced grid to be compatible with ArcGIS
    """
    try:
        df = df.copy()
        
        # Calculate decimal places based on resolution
        decimals = max(int(-np.log10(grid_res)), 0) + 2  # Increase precision to avoid floating point errors
        
        # Round to the nearest grid point
        df['lat_idx'] = (df[lat_col] / grid_res).round() * grid_res
        df['lon_idx'] = (df[lon_col] / grid_res).round() * grid_res
        
        # Ensure precision
        df['lat_idx'] = df['lat_idx'].round(decimals)
        df['lon_idx'] = df['lon_idx'].round(decimals)
        
        # Get data boundaries
        lat_min = df['lat_idx'].min()
        lat_max = df['lat_idx'].max()
        lon_min = df['lon_idx'].min()
        lon_max = df['lon_idx'].max()
        
        if pd.isna(lat_min) or pd.isna(lon_min):
            print(f"[Warning] Data is empty or all NaN, skipping save {filename}")
            return

        # Create full equally spaced grid
        n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
        n_lon = int(round((lon_max - lon_min) / grid_res)) + 1
        
        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)
        
        # Ensure exact equal spacing
        lats = np.round(lats, decimals).astype(np.float64)
        lons = np.round(lons, decimals).astype(np.float64)
        
        fill_value = -9999.0
        
        with netcdf.netcdf_file(filename, 'w') as f:
            f.createDimension('lat', len(lats))
            f.createDimension('lon', len(lons))
            
            lat_v = f.createVariable('lat', 'd', ('lat',))
            lat_v[:] = lats
            lat_v.units = 'degrees_north'
            lat_v.standard_name = 'latitude'
            lat_v.axis = 'Y'
            
            lon_v = f.createVariable('lon', 'd', ('lon',))
            lon_v[:] = lons
            lon_v.units = 'degrees_east'
            lon_v.standard_name = 'longitude'
            lon_v.axis = 'X'
            
            for nc_name, df_col in data_dict.items():
                if df_col not in df.columns:
                    continue
                    
                # Use pivot_table to map data to grid
                grid = df.pivot_table(index='lat_idx', columns='lon_idx', values=df_col, aggfunc='mean')
                # reindex to full equally spaced grid
                grid = grid.reindex(index=lats, columns=lons)
                
                v = f.createVariable(nc_name, 'f', ('lat', 'lon'))
                v.missing_value = np.float32(fill_value)
                v._FillValue = np.float32(fill_value)
                
                data = grid.fillna(fill_value).values.astype(np.float32)
                v[:] = data
                
        print(f"Saved NC: {filename}")
    except Exception as e:
        print(f"Failed to save NC file {filename}: {e}")

def _to_grid(df, value_col, grid_res=GRID_RES, agg='mean'):
    """将散点映射到等间距栅格（ArcGIS风格像元）"""
    if value_col not in df.columns or df.empty:
        return None, None, None

    d = df[['lat', 'lon', value_col]].copy()
    d = d.dropna(subset=['lat', 'lon'])
    if d.empty:
        return None, None, None

    decimals = max(int(-np.log10(grid_res)), 0) + 2
    d['lat_idx'] = (d['lat'] / grid_res).round() * grid_res
    d['lon_idx'] = (d['lon'] / grid_res).round() * grid_res
    d['lat_idx'] = d['lat_idx'].round(decimals)
    d['lon_idx'] = d['lon_idx'].round(decimals)

    lat_min, lat_max = d['lat_idx'].min(), d['lat_idx'].max()
    lon_min, lon_max = d['lon_idx'].min(), d['lon_idx'].max()

    n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
    n_lon = int(round((lon_max - lon_min) / grid_res)) + 1

    lats = np.round(np.linspace(lat_min, lat_max, n_lat), decimals)
    lons = np.round(np.linspace(lon_min, lon_max, n_lon), decimals)

    grid = d.pivot_table(index='lat_idx', columns='lon_idx', values=value_col, aggfunc=agg)
    grid = grid.reindex(index=lats, columns=lons)

    X, Y = np.meshgrid(lons, lats)
    Z = grid.values.astype(float)
    return X, Y, Z


def _style_geo_axis(ax, X, Y):
    ax.set_xlabel('Longitude (°)', fontsize=10)
    ax.set_ylabel('Latitude (°)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal', adjustable='box')
    if X is not None and Y is not None:
        ax.set_xlim(np.nanmin(X), np.nanmax(X))
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))


def plot_combined_map(df, type_col, score_col, lag_col, title_main, filename_base, output_dir):
    # Prepare data: Convert categorical variables to numbers
    if type_col not in df.columns:
        return

    cats = sorted([c for c in df[type_col].dropna().unique() if c != 'None'])
    cat_map = {c: i for i, c in enumerate(cats)}
    df_plot = df.copy()
    df_plot['type_code'] = df_plot[type_col].map(cat_map)

    # Save combined NC file
    nc_path = os.path.join(output_dir, f"{filename_base}.nc")
    save_combined_nc(df_plot, 'lat', 'lon',
                     {'var_type_code': 'type_code', 'score': score_col, 'lag': lag_col},
                     nc_path)

    # 高质量论文版式参数
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'figure.titlesize': 13
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
    fig.suptitle(title_main, fontweight='bold')

    # Subplot 1: Type（离散配色）
    ax = axes[0]
    X1, Y1, Z1 = _to_grid(df_plot, 'type_code', GRID_RES, agg='first')
    if X1 is not None and len(cats) > 0:
        cmap_cat = mcolors.ListedColormap(plt.cm.Set3(np.linspace(0.05, 0.95, max(len(cats), 3))))
        bounds = np.arange(-0.5, len(cats) + 0.5, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap_cat.N)
        im1 = ax.pcolormesh(X1, Y1, Z1, cmap=cmap_cat, norm=norm, shading='nearest')
        cbar1 = plt.colorbar(im1, ax=ax, ticks=np.arange(len(cats)), fraction=0.046, pad=0.03)
        cbar1.ax.set_yticklabels(cats)
        cbar1.ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title('(a) Dominant Variable Type')
    _style_geo_axis(ax, X1, Y1)

    # Subplot 2: Strength（连续配色，期刊常用viridis）
    ax = axes[1]
    X2, Y2, Z2 = _to_grid(df, score_col, GRID_RES, agg='mean')
    if X2 is not None:
        vmin, vmax = np.nanpercentile(Z2, [2, 98]) if np.isfinite(Z2).any() else (0, 1)
        im2 = ax.pcolormesh(X2, Y2, Z2, cmap='viridis', shading='nearest', vmin=vmin, vmax=vmax)
        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
        cbar2.set_label('Score', fontsize=9)
        cbar2.ax.tick_params(labelsize=8)
    ax.set_title('(b) Causal Strength (Score)')
    _style_geo_axis(ax, X2, Y2)

    # Subplot 3: Lag（连续配色，科学绘图常用cividis）
    ax = axes[2]
    X3, Y3, Z3 = _to_grid(df, lag_col, GRID_RES, agg='mean')
    if X3 is not None:
        vmin3, vmax3 = np.nanpercentile(Z3, [2, 98]) if np.isfinite(Z3).any() else (0, 1)
        im3 = ax.pcolormesh(X3, Y3, Z3, cmap='cividis', shading='nearest', vmin=vmin3, vmax=vmax3)
        cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.03)
        cbar3.set_label('Lag (steps)', fontsize=9)
        cbar3.ax.tick_params(labelsize=8)
    ax.set_title('(c) Time Lag (Steps)')
    _style_geo_axis(ax, X3, Y3)

    plt.savefig(os.path.join(output_dir, f"{filename_base}.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Combined Map Preview: {filename_base}.png")

def plot_single_map(df, val_col, title, filename, output_dir, cmap='viridis'):
    if val_col not in df.columns:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2), constrained_layout=True)
    X, Y, Z = _to_grid(df, val_col, GRID_RES, agg='mean')

    if X is not None:
        if np.isfinite(Z).any():
            vmin, vmax = np.nanpercentile(Z, [2, 98])
        else:
            vmin, vmax = (0, 1)
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='nearest', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Value', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=11)
    _style_geo_axis(ax, X, Y)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # Save single NC
    save_combined_nc(df, 'lat', 'lon', {'value': val_col}, os.path.join(output_dir, f"{filename}.nc"))

# ================= Core Processing Logic =================

def process_period(period_name):
    print(f"\n{'='*60}")
    print(f"Processing Period: {period_name}")
    print(f"{'='*60}")
    
    # Construct input and output paths dynamically
    # Input: .../causal_analysis_all_5year/2005_2009/point_jsons
    input_dir = os.path.join(BASE_OUT_DIR, "causal_analysis_all_5year", period_name, "point_jsons")
    output_dir = os.path.join(BASE_OUT_DIR, "causal_analysis_all_5year", period_name, "spatial_analysis_results_fig2")
    
    if not os.path.exists(input_dir):
        print(f"  [Skip] Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    target_label = clean_name(TARGET_VAR)
    print(f"  Current Target Variable: {target_label} (Original: {TARGET_VAR})")
    print(f"  Reading from: {input_dir} ...")
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not files:
        print(f"  [Skip] No JSON files in directory")
        return
        
    data_records = []
    
    for file in files:
        try:
            with open(os.path.join(input_dir, file), 'r') as f:
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
                
                # Record each pair (For Map 2 & 3)
                pair_info[f"{u}_to_{v}_score"] = score
                pair_info[f"{u}_to_{v}_lag"] = lag
                
                # Filter dominant factors (For Map 1 & 4)
                if edge['cause_name'] == TARGET_VAR:
                    # Physical Constraint: Exclude Target driving precipitation
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
            print(f"  Error processing file {file}: {e}")
            
    df = pd.DataFrame(data_records)
    print(f"  Data Loaded: {df.shape}")
    
    if df.empty:
        print("  No valid data, skipping plotting")
        return

    # Execute Plotting Tasks
    
    # Task 1: Map 1 (Combined Plot + Combined NC)
    plot_combined_map(
        df, 
        type_col='dom_driven_type', 
        score_col='dom_driven_score', 
        lag_col='dom_driven_lag',
        title_main=f'Map 1: Dominant Factors Driven by {target_label}',
        filename_base=f'Map1_{target_label}_Driven_Dominant_Combined',
        output_dir=output_dir
    )

    # Task 4: Map 4 (Combined Plot + Combined NC)
    plot_combined_map(
        df, 
        type_col='dom_driver_type', 
        score_col='dom_driver_score', 
        lag_col='dom_driver_lag',
        title_main=f'Map 4: Dominant Drivers of {target_label}',
        filename_base=f'Map4_{target_label}_Driver_Dominant_Combined',
        output_dir=output_dir
    )

    # Task 2 & 3: Pairwise Sensitivity (Single Plot)
    cols = [c for c in df.columns if '_to_' in c and '_score' in c]

    for col in cols:
        parts = col.split('_to_')
        src = parts[0]
        dst = parts[1].replace('_score', '')
        
        # Map 2: TARGET -> Others
        if src == target_label:
            if 'precipitation' in dst.lower(): continue 
            
            plot_single_map(df, col, f'Map 2: Strength ({target_label} -> {dst})', 
                            f'Map2_Strength_{target_label}_to_{dst}', output_dir)
            
            lag_col = col.replace('_score', '_lag')
            if lag_col in df.columns:
                plot_single_map(df, lag_col, f'Map 2: Time Lag ({target_label} -> {dst})', 
                                f'Map2_Lag_{target_label}_to_{dst}', output_dir, cmap='plasma')
                
        # Map 3: Others -> TARGET
        elif dst == target_label:
            plot_single_map(df, col, f'Map 3: Strength ({src} -> {target_label})', 
                            f'Map3_Strength_{src}_to_{target_label}', output_dir)
            
            lag_col = col.replace('_score', '_lag')
            if lag_col in df.columns:
                plot_single_map(df, lag_col, f'Map 3: Time Lag ({src} -> {target_label})', 
                                f'Map3_Lag_{src}_to_{target_label}', output_dir, cmap='plasma')

    print(f"  [{period_name}] Processing completed. Results saved to: {output_dir}")

# ================= Main Execution =================

if __name__ == "__main__":
    print(f"Starting batch plotting for 5-year windows...")
    
    for period in PERIODS:
        try:
            process_period(period)
        except Exception as e:
            print(f"Uncaught exception processing period {period}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nAll periods processed.")