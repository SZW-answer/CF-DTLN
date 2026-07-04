import json
import multiprocessing as mp
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


warnings.filterwarnings("ignore")


# =============================================================================
# User Settings
# =============================================================================
# 所有路径/命名/GPU 分配都集中在这里改。
RUN_SETTINGS = {
    "configs": ["./model_config_EVI.json", "./model_config_SIF.json", "./model_config.json"],
    "config_devices": {
        "./model_config_EVI.json": "cuda:1",
        "./model_config_SIF.json": "cuda:1",
        "./model_config.json": "cuda:0",
    },
    "parallel_configs": True,

    "analysis_output_subdir": "causal_analysis_all_5year",
    "cache_subdir": "temp_cache",
    "slice_data_prefix": "data",

    "lulc_ids": [1, 2, 4, 9],
    "periods": [
        ("2003_2004", 2003, 2005),  # 与原脚本保持一致
        ("2005_2009", 2005, 2010),
        ("2010_2014", 2010, 2015),
        ("2015_2019", 2015, 2020),
        ("2020_2024", 2020, 2024),
    ],
    "global_start_year": 2003,
}


def safe_filename(value):
    text = str(value).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed"


def normalize_device_label(value):
    device = str(value).strip().lower()
    if device in {"cpu", "cuda"}:
        return device
    if re.fullmatch(r"cuda:\d+", device):
        return device
    raise ValueError("device must be 'cpu', 'cuda', or an explicit CUDA device like 'cuda:0'")


def resolve_device_label(config_path, settings):
    mapping = dict(settings.get("config_devices", {}))
    key = str(config_path)
    value = (
        mapping.get(key)
        or mapping.get(str(Path(key)))
        or mapping.get(Path(key).name)
        or settings.get("device", "cuda")
    )
    return normalize_device_label(value)


def device_to_ls_id(device_label):
    device_label = normalize_device_label(device_label)
    if device_label == "cpu":
        return -1
    if device_label == "cuda":
        return 0
    return int(device_label.split(":", 1)[1])


def prepare_process_device(device_label):
    device_label = normalize_device_label(device_label)
    if device_label == "cpu":
        os.environ["DTLN_FORCE_CPU"] = "1"
        return torch.device("cpu")

    os.environ.pop("DTLN_FORCE_CPU", None)
    if not torch.cuda.is_available():
        print(f"  [Device fallback] CUDA unavailable, using CPU instead of {device_label}")
        os.environ["DTLN_FORCE_CPU"] = "1"
        return torch.device("cpu")

    if device_label == "cuda":
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    index = int(device_label.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        print(f"  [Device fallback] {device_label} unavailable, using cuda:0")
        index = 0
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def load_interpreted_for_config(config_path, device_label):
    os.environ["MODEL_CONFIG_PATH"] = str(config_path)
    device = prepare_process_device(device_label)

    if "interpreted" in sys.modules:
        del sys.modules["interpreted"]
    import interpreted as interp

    interp.ls_id = device_to_ls_id(str(device)) if device.type == "cuda" else -1
    return interp, device


def resolve_model_path(config):
    model_path = config["analyze"].get("model_path")
    out_dir = Path(config["analyze"]["OUT_DIR"])
    if model_path:
        return str(model_path)

    best_models = sorted(out_dir.glob("best_model_rmse_*.pth"))
    if best_models:
        return str(best_models[0])

    latest = out_dir / "checkpoint_latest.pth"
    if latest.exists():
        return str(latest)

    raise FileNotFoundError(f"No model checkpoint found in {out_dir}")


def slice_time_series(data, start_year, end_year, global_start_year):
    """
    Slice monthly time series data (N, T, V) based on years.

    注意：这里保持原 interpreted_5year.py 的结束年包含式逻辑，避免改变既有结果目录含义。
    """
    if data is None:
        return None

    _, n_time, _ = data.shape
    start_idx = (int(start_year) - int(global_start_year)) * 12
    end_idx = (int(end_year) - int(global_start_year) + 1) * 12
    start_idx = max(0, start_idx)
    actual_end_idx = min(end_idx, n_time)

    if start_idx >= actual_end_idx:
        return None

    sliced_data = data[:, start_idx:actual_end_idx, :]
    print(
        f"    Slicing {start_year}-{end_year}: "
        f"Indices [{start_idx}:{actual_end_idx}], Result Shape {sliced_data.shape}"
    )
    return sliced_data


def write_period_cache(cache_dir, period_name, data, aux_data, static_data, coords, prefix):
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{prefix}_{period_name}.npy"
    coords_path = data_path.with_name(f"{prefix}_{period_name}_coords.json")
    aux_path = data_path.with_name(f"{prefix}_{period_name}_aux.npy")
    static_path = data_path.with_name(f"{prefix}_{period_name}_static.npy")

    np.save(data_path, data)
    with coords_path.open("w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=2)
    if aux_data is not None:
        np.save(aux_path, aux_data)
    if static_data is not None:
        np.save(static_path, static_data)
    return str(data_path)


def run_one_config(config_path, device_label, settings):
    config_path = str(config_path)
    print("\n" + "=" * 78)
    print(f"[Config start] {config_path} on {device_label}")
    print("=" * 78)

    interp, device = load_interpreted_for_config(config_path, device_label)
    base_config = deepcopy(interp.config)

    nc_file = base_config["analyze"]["NC_FILE"]
    base_out_dir = Path(base_config["analyze"]["OUT_DIR"])
    predictors = base_config["analyze"]["PREDICTORS"]
    target = base_config["analyze"]["TARGET"]
    lulc_ids = settings["lulc_ids"]
    seq_len = int(base_config["model"]["SEQ_LEN"])
    output_window = int(base_config["model"].get("output_window", 1))
    model_path = resolve_model_path(base_config)

    print(f"  Effective device: {device}")
    print(f"  NC file: {nc_file}")
    print(f"  Output dir: {base_out_dir / settings['analysis_output_subdir']}")
    print(f"  Cache dir: {base_out_dir / settings['cache_subdir']}")
    print(f"  Model: {model_path}")

    print("\n[Step 1] Loading full dataset once for this config...")
    ds, lat, lon, time = interp.load_nc_data(nc_file)
    aux_predictors = base_config["analyze"].get("AUX_PREDICTORS", [])
    static_vars = base_config["analyze"].get("STATIC_VARS", [])

    full_data, full_aux_data, full_static_data, coords = interp.preprocess_geo_data(
        ds,
        predictors,
        target,
        lat,
        lon,
        time,
        lulc_ids,
        return_coords=True,
        aux_predictors=aux_predictors,
        static_vars=static_vars,
    )

    original_proceed_path = interp.config["analyze"].get("PROCEED_DATA")
    completed_periods = []
    try:
        for period_name, start_year, end_year in settings["periods"]:
            print("\n" + "-" * 70)
            print(f"  Period: {period_name} ({start_year}-{end_year})")
            print("-" * 70)

            sliced_data = slice_time_series(
                full_data,
                start_year,
                end_year,
                settings["global_start_year"],
            )
            if sliced_data is None:
                print(f"  [Skip] No data for {period_name}")
                continue
            if sliced_data.shape[1] < seq_len:
                print(f"  [Skip] T={sliced_data.shape[1]} < SEQ_LEN={seq_len}")
                continue
            if sliced_data.shape[1] == seq_len:
                print("  [Warning] T equals SEQ_LEN; each point yields one sample.")

            sliced_aux = (
                slice_time_series(full_aux_data, start_year, end_year, settings["global_start_year"])
                if full_aux_data is not None
                else None
            )
            sliced_static = full_static_data

            period_out_dir = base_out_dir / settings["analysis_output_subdir"] / period_name
            period_out_dir.mkdir(parents=True, exist_ok=True)

            cache_dir = base_out_dir / settings["cache_subdir"] / period_name
            proceed_data = write_period_cache(
                cache_dir,
                period_name,
                sliced_data,
                sliced_aux,
                sliced_static,
                coords,
                settings["slice_data_prefix"],
            )
            interp.config["analyze"]["PROCEED_DATA"] = proceed_data

            print(f"  Running all-variable analysis using model: {Path(model_path).name}")
            interp.run_causal_analysis_all(
                model_path=model_path,
                nc_file=nc_file,
                output_dir=str(period_out_dir),
                predictors=predictors,
                target_var=target,
                lulc_ids=lulc_ids,
                seq_len=seq_len,
                output_window=output_window,
                batch_size=base_config["analyze"]["BATCH_SIZE"],
                m=base_config["analyze"]["KMeans_M"],
                n=base_config["analyze"]["KMeans_N"],
            )
            completed_periods.append(period_name)
            print(f"  [Done] {period_name}: {period_out_dir}")
    finally:
        interp.config["analyze"]["PROCEED_DATA"] = original_proceed_path

    print(f"[Config done] {config_path}: {len(completed_periods)} periods")
    return {
        "config": config_path,
        "device": str(device),
        "target": target,
        "completed_periods": completed_periods,
        "output_dir": str(base_out_dir / settings["analysis_output_subdir"]),
    }


def normalized_settings():
    settings = dict(RUN_SETTINGS)
    settings["configs"] = [str(path) for path in settings["configs"]]
    settings["config_devices"] = {
        str(path): normalize_device_label(device)
        for path, device in dict(settings.get("config_devices", {})).items()
    }
    settings["parallel_configs"] = bool(settings.get("parallel_configs", True))
    settings["analysis_output_subdir"] = str(settings["analysis_output_subdir"])
    settings["cache_subdir"] = str(settings["cache_subdir"])
    settings["slice_data_prefix"] = str(settings["slice_data_prefix"])
    settings["lulc_ids"] = [int(value) for value in settings.get("lulc_ids", [1, 2, 4, 9])]
    settings["global_start_year"] = int(settings.get("global_start_year", 2003))
    return settings


def main():
    settings = normalized_settings()
    config_jobs = [
        (config_path, resolve_device_label(config_path, settings))
        for config_path in settings["configs"]
    ]

    print("=" * 78)
    print("5-Year Full-Region All-Variable Interpretation")
    print(f"Configs: {config_jobs}")
    print(f"Output subdir: {settings['analysis_output_subdir']}")
    print(f"Cache subdir: {settings['cache_subdir']}")
    print("=" * 78)

    results = []
    if settings["parallel_configs"] and len(config_jobs) > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(config_jobs), mp_context=ctx) as executor:
            futures = {
                executor.submit(run_one_config, config_path, device_label, settings): config_path
                for config_path, device_label in config_jobs
            }
            for future in as_completed(futures):
                config_path = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"[Config failed] {config_path}: {exc}")
                    import traceback

                    traceback.print_exc()
    else:
        for config_path, device_label in config_jobs:
            try:
                results.append(run_one_config(config_path, device_label, settings))
            except Exception as exc:
                print(f"[Config failed] {config_path}: {exc}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 78)
    print(f"[Done] Completed {len(results)}/{len(config_jobs)} config jobs.")
    for item in results:
        print(
            f"  {item['config']} | target={item['target']} | device={item['device']} | "
            f"periods={item['completed_periods']} | {item['output_dir']}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
