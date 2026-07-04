import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch


RUN_SETTINGS = {
    "configs": ["./model_config_EVI.json", "./model_config_SIF.json", "./model_config.json"],
    "lulc_ids": [1, 2, 4, 9],
    "max_points": 1000,
    "max_windows": 512,
    "device": "cuda:0",
    "high_corr_threshold": 0.70,
    "moderate_corr_threshold": 0.50,
    "film_strong_threshold": 0.30,
}


def load_interpreted(config_path):
    os.environ["MODEL_CONFIG_PATH"] = str(config_path)
    if "interpreted" in sys.modules:
        del sys.modules["interpreted"]
    import interpreted as interp
    return interp


def resolve_model_path(config):
    model_path = config["analyze"].get("model_path")
    if model_path:
        return model_path
    out_dir = Path(config["analyze"]["OUT_DIR"])
    models = sorted(out_dir.glob("best_model_rmse_*.pth"))
    if models:
        return str(models[0])
    latest = out_dir / "checkpoint_latest.pth"
    if latest.exists():
        return str(latest)
    raise FileNotFoundError(f"No model checkpoint found under {out_dir}")


def clean_name(name):
    text = str(name)
    for token in ["_deseasonalized", "_sum", "total_"]:
        text = text.replace(token, "")
    low = text.lower()
    if "precip" in low:
        return "Precipitation"
    if "runoff" in low:
        return "Runoff"
    if low == "ga" or "groundwater" in low:
        return "GWS"
    if "gpp" in low:
        return "GPP"
    if "gosif" in low or low == "sif":
        return "SIF"
    if low in {"evi", "ndvi"}:
        return low.upper()
    return text


def finite_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def sample_points(n_points, max_points):
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def scaler_path(model_path, name):
    return Path(model_path).parent / name


def load_scaler(model_path, name):
    path = scaler_path(model_path, name)
    if not path.exists():
        return None, path
    return joblib.load(path), path


def scaler_rows(raw_2d, var_names, scaler, scaler_file):
    rows = []
    raw = np.asarray(raw_2d, dtype=float)
    if scaler is None:
        for idx, name in enumerate(var_names):
            rows.append({
                "variable": name,
                "scaler_file": str(scaler_file),
                "scaler_found": False,
                "raw_min": float(np.nanmin(raw[:, idx])),
                "raw_max": float(np.nanmax(raw[:, idx])),
                "train_min": np.nan,
                "train_max": np.nan,
                "outside_train_range_percent": np.nan,
                "scaled_min": np.nan,
                "scaled_max": np.nan,
            })
        return rows

    scaled = scaler.transform(raw)
    lo = np.asarray(scaler.data_min_, dtype=float)
    hi = np.asarray(scaler.data_max_, dtype=float)
    outside = (raw < lo) | (raw > hi)
    for idx, name in enumerate(var_names):
        rows.append({
            "variable": name,
            "scaler_file": str(scaler_file),
            "scaler_found": True,
            "raw_min": float(np.nanmin(raw[:, idx])),
            "raw_max": float(np.nanmax(raw[:, idx])),
            "train_min": float(lo[idx]),
            "train_max": float(hi[idx]),
            "outside_train_range_percent": float(np.nanmean(outside[:, idx]) * 100.0),
            "scaled_min": float(np.nanmin(scaled[:, idx])),
            "scaled_max": float(np.nanmax(scaled[:, idx])),
        })
    return rows


def leakage_rows(data, aux_data, static_data, var_names, aux_names, static_names, target_idx, target_label, settings):
    target = data[:, :, target_idx].reshape(-1)
    rows = []

    vegetation_targets = {"EVI", "SIF", "GPP", "NDVI"}
    target_is_veg = target_label in vegetation_targets

    if aux_data is not None:
        for idx, name in enumerate(aux_names):
            arr = aux_data[:, :, idx].reshape(-1)
            corr0 = finite_corr(arr, target)
            lag_corrs = []
            for lag in range(1, 7):
                lag_corrs.append(finite_corr(aux_data[:, :-lag, idx].reshape(-1), data[:, lag:, target_idx].reshape(-1)))
            max_lag_corr = np.nanmax(np.abs(lag_corrs)) if lag_corrs else np.nan
            clean = clean_name(name)
            name_flag = "none"
            if (clean in vegetation_targets or "fpar" in str(name).lower()) and target_is_veg:
                name_flag = "vegetation_proxy"
            if clean == target_label:
                name_flag = "target_duplicate"
            rows.append({
                "kind": "aux",
                "variable": name,
                "clean_name": clean,
                "corr_same_time_with_target": corr0,
                "max_abs_corr_lag_1_6": float(max_lag_corr) if np.isfinite(max_lag_corr) else np.nan,
                "name_leakage_flag": name_flag,
                "risk_level": classify_leakage(corr0, max_lag_corr, name_flag, settings),
            })

    if static_data is not None:
        target_point_mean = np.nanmean(data[:, :, target_idx], axis=1)
        for idx, name in enumerate(static_names):
            arr = static_data[:, idx]
            corr0 = finite_corr(arr, target_point_mean)
            rows.append({
                "kind": "static",
                "variable": name,
                "clean_name": clean_name(name),
                "corr_same_time_with_target": corr0,
                "max_abs_corr_lag_1_6": np.nan,
                "name_leakage_flag": "none",
                "risk_level": classify_leakage(corr0, np.nan, "none", settings),
            })

    return rows


def classify_leakage(corr0, max_lag_corr, name_flag, settings):
    corr_abs = np.nanmax(np.abs([corr0, max_lag_corr]))
    if name_flag in {"target_duplicate", "vegetation_proxy"}:
        return "high_name_proxy"
    if np.isfinite(corr_abs) and corr_abs >= settings["high_corr_threshold"]:
        return "high_corr"
    if np.isfinite(corr_abs) and corr_abs >= settings["moderate_corr_threshold"]:
        return "moderate_corr"
    return "low"


def build_model(interp, config, var_names, device):
    cfg = {
        "n_gpu": int(device.index or 0) if device.type == "cuda" else -1,
        "data_loader": {
            "args": {
                "time_step": int(config["model"]["SEQ_LEN"]),
                "output_window": int(config["model"].get("output_window", 1)),
                "series_num": len(var_names),
                "feature_dim": 1,
                "output_dim": 1,
            }
        },
    }
    model = interp.PredictModel(
        cfg,
        d_model=config["model"]["d_model"],
        n_head=config["model"]["n_head"],
        n_layers=config["model"]["n_layers"],
        ffn_hidden=config["model"]["hidden_layers"],
        drop_prob=config["model"]["drop_prob"],
        tau=config["model"]["tau"],
        use_geo_encoding=config["train"].get("USE_GEO_ENCODING", False),
        aux_series_num=len(config["analyze"].get("AUX_PREDICTORS", [])),
        static_dim=len(config["analyze"].get("STATIC_VARS", [])),
    ).to(device)
    return model


def build_windows(data, aux_data, static_data, seq_len, max_windows):
    samples, aux_samples, static_samples = [], [], []
    n_points, n_time, n_vars = data.shape
    for point_idx in range(n_points):
        for t in range(seq_len, n_time + 1):
            samples.append(data[point_idx, t - seq_len:t].reshape(seq_len, n_vars, 1))
            if aux_data is not None:
                aux_samples.append(aux_data[point_idx, t - seq_len:t].reshape(seq_len, aux_data.shape[2], 1))
            if static_data is not None:
                static_samples.append(static_data[point_idx])
            if len(samples) >= max_windows:
                break
        if len(samples) >= max_windows:
            break
    x = torch.FloatTensor(np.asarray(samples))
    aux = torch.FloatTensor(np.asarray(aux_samples)) if aux_samples else None
    static = torch.FloatTensor(np.asarray(static_samples)) if static_samples else None
    return x, aux, static


def film_sensitivity(interp, config, model_path, data, aux_data, static_data, var_names, settings):
    device_label = settings["device"]
    if device_label.startswith("cuda") and torch.cuda.is_available():
        idx = int(device_label.split(":")[1]) if ":" in device_label else 0
        if idx >= torch.cuda.device_count():
            idx = 0
        device = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(idx)
    else:
        device = torch.device("cpu")

    model = build_model(interp, config, var_names, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seq_len = int(config["model"]["SEQ_LEN"])
    x, aux, static = build_windows(data, aux_data, static_data, seq_len, settings["max_windows"])
    x = x.to(device)
    aux = aux.to(device) if aux is not None else None
    static = static.to(device) if static is not None else None

    with torch.no_grad():
        real = model(x, None, None, aux, static)
        no_cond = model(x, None, None, None, None)
        if aux is not None and aux.shape[0] > 1:
            perm = torch.randperm(aux.shape[0], device=device)
            aux_shuf = aux[perm]
            static_shuf = static[perm] if static is not None and static.shape[0] == aux.shape[0] else static
            shuffled = model(x, None, None, aux_shuf, static_shuf)
        else:
            shuffled = no_cond

    denom = torch.mean(torch.abs(real)).item() + 1e-8
    return {
        "n_windows": int(x.shape[0]),
        "device": str(device),
        "mean_abs_output": float(torch.mean(torch.abs(real)).item()),
        "mean_abs_diff_real_vs_no_condition": float(torch.mean(torch.abs(real - no_cond)).item()),
        "relative_diff_real_vs_no_condition": float(torch.mean(torch.abs(real - no_cond)).item() / denom),
        "mean_abs_diff_real_vs_shuffled_condition": float(torch.mean(torch.abs(real - shuffled)).item()),
        "relative_diff_real_vs_shuffled_condition": float(torch.mean(torch.abs(real - shuffled)).item() / denom),
    }


def run_config(config_path, settings):
    interp = load_interpreted(config_path)
    config = interp.config
    model_path = resolve_model_path(config)
    out_dir = Path(config["analyze"]["OUT_DIR"]) / "aux_static_diagnostics_08"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictors = list(config["analyze"]["PREDICTORS"])
    target = config["analyze"]["TARGET"]
    var_names = predictors.copy()
    if target not in var_names:
        var_names.append(target)
    target_idx = var_names.index(target)
    target_label = clean_name(target)
    aux_names = config["analyze"].get("AUX_PREDICTORS", [])
    static_names = config["analyze"].get("STATIC_VARS", [])

    print("\n" + "=" * 78)
    print(f"Config: {config_path}")
    print(f"Target: {target} ({target_label})")
    print(f"Model: {model_path}")

    ds, lat, lon, time = interp.load_nc_data(config["analyze"]["NC_FILE"])
    data, aux_data, static_data, coords = interp.preprocess_geo_data(
        ds,
        predictors,
        target,
        lat,
        lon,
        time,
        settings["lulc_ids"],
        return_coords=True,
        aux_predictors=aux_names,
        static_vars=static_names,
    )

    idx = sample_points(data.shape[0], settings["max_points"])
    data_raw = data[idx]
    aux_raw = aux_data[idx] if aux_data is not None else None
    static_raw = static_data[idx] if static_data is not None else None

    # Leakage/proxy check on raw data.
    leak = leakage_rows(data_raw, aux_raw, static_raw, var_names, aux_names, static_names, target_idx, target_label, settings)
    leak_df = pd.DataFrame(leak)
    leak_path = out_dir / f"{clean_name(target)}_aux_static_leakage.csv"
    leak_df.to_csv(leak_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {leak_path}")

    # Standardization consistency check.
    scaler_main, main_path = load_scaler(model_path, "scaler.pkl")
    scaler_aux, aux_path = load_scaler(model_path, "aux_scaler.pkl")
    scaler_static, static_path = load_scaler(model_path, "static_scaler.pkl")
    scale_rows = []
    scale_rows += scaler_rows(data_raw.reshape(-1, data_raw.shape[2]), var_names, scaler_main, main_path)
    if aux_raw is not None:
        scale_rows += scaler_rows(aux_raw.reshape(-1, aux_raw.shape[2]), aux_names, scaler_aux, aux_path)
    if static_raw is not None:
        scale_rows += scaler_rows(static_raw, static_names, scaler_static, static_path)
    scale_df = pd.DataFrame(scale_rows)
    scale_path = out_dir / f"{clean_name(target)}_scaler_consistency.csv"
    scale_df.to_csv(scale_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {scale_path}")

    # Use fixed training scaler for FiLM sensitivity.
    data_scaled, aux_scaled, static_scaled = interp.standardize_inference_data(data_raw, aux_raw, static_raw, model_path)
    film = film_sensitivity(interp, config, model_path, data_scaled, aux_scaled, static_scaled, var_names, settings)
    film["film_risk_level"] = (
        "strong" if film["relative_diff_real_vs_no_condition"] >= settings["film_strong_threshold"] else "moderate_or_low"
    )
    film_path = out_dir / f"{clean_name(target)}_film_sensitivity.json"
    with film_path.open("w", encoding="utf-8") as f:
        json.dump(film, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {film_path}")
    print(json.dumps(film, ensure_ascii=False, indent=2))

    summary = {
        "config": config_path,
        "target": target,
        "target_label": target_label,
        "high_or_proxy_aux": leak_df[leak_df["risk_level"].isin(["high_name_proxy", "high_corr"])][
            ["kind", "variable", "risk_level", "corr_same_time_with_target", "max_abs_corr_lag_1_6"]
        ].to_dict("records") if not leak_df.empty else [],
        "max_outside_train_range_percent": float(scale_df["outside_train_range_percent"].max(skipna=True)),
        "film_relative_diff_real_vs_no_condition": film["relative_diff_real_vs_no_condition"],
    }
    summary_path = out_dir / f"{clean_name(target)}_diagnostic_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {summary_path}")
    return summary


def main():
    summaries = []
    for config_path in RUN_SETTINGS["configs"]:
        try:
            summaries.append(run_config(config_path, RUN_SETTINGS))
        except Exception as exc:
            print(f"[Failed] {config_path}: {exc}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("Diagnostic summaries:")
    for item in summaries:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
