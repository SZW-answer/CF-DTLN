import numpy as np
import os
import json
import shutil
import torch
import warnings
from copy import deepcopy

# Import functionality from interpreted.py
import interpreted as interp

# =============================================================================
# Configuration & Constants
# =============================================================================
warnings.filterwarnings('ignore')

# Defined time windows as per requirements
PERIODS = [
    ("2003_2004", 2003, 2004), # 24 months
    ("2005_2009", 2005, 2009), # 60 months
    ("2010_2014", 2010, 2014), # 60 months
    ("2015_2019", 2015, 2019), # 60 months
    ("2020_2024", 2020, 2024)  # 60 months
]

# Assumed Global Start Year of the dataset (for index slicing)
GLOBAL_START_YEAR = 2003

def slice_time_series(data, start_year, end_year, global_start=GLOBAL_START_YEAR):
    """
    Slice the time series data (N, T, V) based on years.
    Assumes monthly data (12 steps per year).
    """
    if data is None:
        return None
    
    n_points, n_time, n_vars = data.shape
    
    # Calculate indices
    start_idx = (start_year - global_start) * 12
    end_idx = (end_year - global_start + 1) * 12
    
    # Boundary checks
    if start_idx >= n_time:
        return None # Range is completely outside data
    
    # Clamp end index
    actual_end_idx = min(end_idx, n_time)
    
    if start_idx >= actual_end_idx:
        return None
        
    sliced_data = data[:, start_idx:actual_end_idx, :]
    print(f"    Slicing {start_year}-{end_year}: Indices [{start_idx}:{actual_end_idx}], Result Shape {sliced_data.shape}")
    
    return sliced_data

def run_5year_analysis_loop():
    # 1. Load Configuration
    base_config = deepcopy(interp.config)
    
    # Paths from config
    NC_FILE = base_config["analyze"]["NC_FILE"]
    BASE_OUT_DIR = base_config["analyze"]["OUT_DIR"]
    PREDICTORS = base_config["analyze"]["PREDICTORS"]
    TARGET = base_config["analyze"]["TARGET"]
    LULC_IDS = [1, 2, 4, 9] 
    
    # Params
    SEQ_LEN = base_config["model"]["SEQ_LEN"]
    
    print("=" * 60)
    print("Starting 5-Year Window Causal Analysis")
    print(f"Global Start Year Assumption: {GLOBAL_START_YEAR}")
    print("=" * 60)

    # 2. Load Full Dataset ONCE
    print("\n[Step 1] Loading Full Dataset...")
    ds, lat, lon, time = interp.load_nc_data(NC_FILE)
    
    aux_predictors = base_config["analyze"].get("AUX_PREDICTORS", [])
    static_vars_list = base_config["analyze"].get("STATIC_VARS", [])
    
    full_data, full_aux_data, full_static_data, coords = interp.preprocess_geo_data(
        ds, PREDICTORS, TARGET, lat, lon, time, LULC_IDS, 
        return_coords=True,
        aux_predictors=aux_predictors,
        static_vars=static_vars_list
    )
    
    # 3. Iterate over defined periods
    for period_name, start_year, end_year in PERIODS:
        print(f"\n{'='*40}")
        print(f"Processing Window: {period_name} ({start_year}-{end_year})")
        print(f"{'='*40}")
        
        # 3.1 Slice Data
        sliced_data = slice_time_series(full_data, start_year, end_year)
        
        if sliced_data is None:
            print(f"  [Skipping] Data not available for {period_name}.")
            continue
            
        # [修改点]：这里将 <= 改为 < 
        # 只要 sliced_data 长度 (24) >= SEQ_LEN (24)，就能至少生成 1 个样本
        if sliced_data.shape[1] < SEQ_LEN:
             print(f"  [Skipping] Time dimension ({sliced_data.shape[1]}) < SEQ_LEN ({SEQ_LEN}). Cannot predict.")
             continue
        elif sliced_data.shape[1] == SEQ_LEN:
             print(f"  [Warning] Time dimension ({sliced_data.shape[1]}) equals SEQ_LEN. Will generate exactly 1 sample per point.")

        # Slice Aux Data (Time dependent)
        sliced_aux = slice_time_series(full_aux_data, start_year, end_year) if full_aux_data is not None else None
        
        # Static Data (Time independent) - No slicing needed, just pass through
        sliced_static = full_static_data 
        
        # 3.2 Prepare Cache for this Slice
        period_out_dir = os.path.join(BASE_OUT_DIR, "causal_analysis_all_5year", period_name)
        os.makedirs(period_out_dir, exist_ok=True)
        
        # Define cache paths
        temp_cache_dir = os.path.join(BASE_OUT_DIR, "temp_cache", period_name)
        os.makedirs(temp_cache_dir, exist_ok=True)
        
        slice_data_path = os.path.join(temp_cache_dir, f"data_{period_name}.npy")
        slice_coords_path = slice_data_path.replace('.npy', '_coords.json')
        slice_aux_path = slice_data_path.replace('.npy', '_aux.npy')
        slice_static_path = slice_data_path.replace('.npy', '_static.npy')
        
        # Save slices to disk
        np.save(slice_data_path, sliced_data)
        with open(slice_coords_path, 'w') as f:
            json.dump(coords, f)
            
        if sliced_aux is not None:
            np.save(slice_aux_path, sliced_aux)
        if sliced_static is not None:
            np.save(slice_static_path, sliced_static)
            
        print(f"  Cached slice data to: {temp_cache_dir}")
        
        # 3.3 Inject Configuration
        original_proceed_path = interp.config["analyze"]["PROCEED_DATA"]
        interp.config["analyze"]["PROCEED_DATA"] = slice_data_path
        
        # 3.4 Run Analysis
        try:
            model_path = base_config["analyze"]["model_path"]
            
            if model_path is None:
                 best_models = [f for f in os.listdir(BASE_OUT_DIR) if f.startswith('best_model_rmse_')]
                 if best_models:
                     model_path = os.path.join(BASE_OUT_DIR, best_models[0])
            
            print(f"  Running analysis using model: {os.path.basename(model_path)}")
            
            interp.run_causal_analysis_all(
                model_path=model_path,
                nc_file=NC_FILE, 
                output_dir=period_out_dir,
                predictors=PREDICTORS,
                target_var=TARGET,
                lulc_ids=LULC_IDS,
                seq_len=SEQ_LEN,
                output_window=base_config["model"].get("output_window", 1),
                batch_size=base_config["analyze"]["BATCH_SIZE"],
                m=base_config["analyze"]["KMeans_M"],
                n=base_config["analyze"]["KMeans_N"]
            )
            
            print(f"  [Success] Finished analysis for {period_name}")
            
        except Exception as e:
            print(f"  [Error] Failed analysis for {period_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Restore config
        interp.config["analyze"]["PROCEED_DATA"] = original_proceed_path
        
    print("\n" + "="*60)
    print("All 5-year window analyses completed.")
    print(f"Results stored in: {os.path.join(BASE_OUT_DIR, 'causal_analysis_all_5year')}")
    print("="*60)

if __name__ == "__main__":
    run_5year_analysis_loop()