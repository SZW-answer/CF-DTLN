import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from tqdm import tqdm


BASE_SCRIPT = Path(__file__).with_name("05_basin_rrp_lag_spectrum.py")
spec = importlib.util.spec_from_file_location("basin_rrp_lag_spectrum_05", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


DEFAULT_CONFIG_PATHS = [
    "./model_config_EVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]
DEFAULT_SHP = "Dataset/NLH/NLH.shp"
PERIODS = base.PERIODS
DRIVERS = base.DRIVERS
PERIOD_COLORS = base.PERIOD_COLORS
OUTPUT_SUBDIR = "basin_voted_rrp_lag_spectrum_06"
CACHE_VERSION = "v2"


# =============================================================================
# Run Settings
# =============================================================================
# 这个脚本和 05 一样会重新提取每个点的完整 relK 曲线。
# 区别是：05 对流域内点曲线取平均；06 先按每个点的峰值 lag 投票，
# 再取“票数最多 lag”对应的代表性点曲线，不做流域平均曲线。
RUN_SETTINGS = {
    # 三个 config 同时跑。
    "configs": ["./model_config_EVI.json", "./model_config_SIF.json", "./model_config.json"],

    "shp": DEFAULT_SHP,
    "name_field": "name",

    # None 表示从 config 自动识别；也可以写 "SIF" / "EVI" / "GPP"。
    "target": None,

    # ["forward"]   = driver -> target
    # ["reverse"]   = target -> driver
    # ["combined"]  = 双向平均曲线后再投票
    # ["forward", "reverse", "combined"] = 三套结果都输出
    "directions": ["forward", "reverse", "combined"],

    # 默认设备；如果下面 config_devices 配了，会按每个 config 覆盖。
    # 支持 "cpu" / "cuda" / "cuda:0" / "cuda:1"。
    "device": "cuda",

    # 每个 config 指定 GPU。你的两块 GPU：EVI 放 cuda:0，SIF/GPP 放 cuda:1。
    "config_devices": {
        "./model_config_EVI.json": "cuda:0",
        "./model_config_SIF.json": "cuda:1",
        "./model_config.json": "cuda:1",
    },

    # True = 多进程并行跑多个 config；False = 按顺序跑。
    "parallel_configs": True,

    # None 表示读取 config analyze.lag_selection.smooth_window，若没有则使用 3。
    "smooth_window": None,

    # 投票 lag 的来源：
    # selected_argmax = 使用 interpreted.py 的 compute_lag_details 稳健 lag 选择逻辑
    # smoothed_argmax = 使用滑动平均后的 relK 曲线峰值
    # raw_argmax      = 使用原始 relK 曲线峰值
    "vote_lag_source": "selected_argmax",

    # None 表示全部流域；也可指定 ["塔里木河干流"]。
    "only_basin": None,

    # None 表示 5 个时期全跑；也可指定 ["2003_2004"]。
    "periods": None,

    # 调试用，每个时期最多处理多少点。None 表示全点。
    "max_points_per_period": None,

    # True 时只输出 CSV，不画图。
    "no_figures": False,

    # 缓存每个点的完整 relK 曲线。第一次仍要推理；之后改图/统计可直接复用。
    # 如果改了 only_basin 或 max_points_per_period，建议把 reuse 设 False 或删除 point_spectra 目录。
    "reuse_point_spectra_cache": True,
    "write_point_spectra_cache": True,
}


def configure_matplotlib():
    for font_path in [
        "/mnt/c/Windows/Fonts/times.ttf",
        "/mnt/c/Windows/Fonts/timesbd.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]:
        if Path(font_path).exists():
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "SimSun", "DejaVu Serif"],
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.linewidth": 0.8,
        }
    )


def safe_filename(value):
    text = str(value).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed_basin"


def normalize_device_label(value):
    device = str(value).strip().lower()
    if device == "cpu" or device == "cuda":
        return device
    if re.fullmatch(r"cuda:\d+", device):
        return device
    raise ValueError("device must be 'cpu', 'cuda', or an explicit CUDA device like 'cuda:0'")


def resolve_torch_device(device_label):
    device_label = normalize_device_label(device_label)
    if device_label == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print(f"  [Device fallback] CUDA is not available, using CPU instead of {device_label}")
        return torch.device("cpu")
    if ":" in device_label:
        idx = int(device_label.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            print(f"  [Device fallback] {device_label} is not available, using cuda:0")
            torch.cuda.set_device(0)
            return torch.device("cuda:0")
        torch.cuda.set_device(idx)
        return torch.device(device_label)
    torch.cuda.set_device(0)
    return torch.device(device_label)


def device_to_n_gpu(device):
    if device.type != "cuda":
        return -1
    return int(device.index or 0)


def config_device(config_path, args):
    key = str(config_path)
    mapping = args.config_devices or {}
    device = mapping.get(key) or mapping.get(str(Path(key))) or mapping.get(Path(key).name) or args.device
    return normalize_device_label(device)


def build_model_for_device(config, var_names, seq_len, output_window, device):
    cfg = {
        "n_gpu": device_to_n_gpu(device),
        "data_loader": {
            "args": {
                "time_step": seq_len,
                "output_window": output_window,
                "series_num": len(var_names),
                "feature_dim": 1,
                "output_dim": 1,
            }
        },
    }
    aux_predictors = config["analyze"].get("AUX_PREDICTORS", [])
    static_vars = config["analyze"].get("STATIC_VARS", [])
    model = base.interp.PredictModel(
        cfg,
        d_model=config["model"]["d_model"],
        n_head=config["model"]["n_head"],
        n_layers=config["model"]["n_layers"],
        ffn_hidden=config["model"]["hidden_layers"],
        drop_prob=config["model"]["drop_prob"],
        tau=config["model"]["tau"],
        use_geo_encoding=config["train"].get("USE_GEO_ENCODING", False),
        aux_series_num=len(aux_predictors) if aux_predictors else 0,
        static_dim=len(static_vars) if static_vars else 0,
    ).to(device)
    model.device = device
    return model


def load_model_for_device(config, var_names, device):
    seq_len = int(config["model"]["SEQ_LEN"])
    output_window = int(config["model"].get("output_window", 1))
    model = build_model_for_device(config, var_names, seq_len, output_window, device)
    model_path = base.resolve_model_path(config)
    print(f"  Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_path


def short_hash(text):
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8]


def cache_tag(args):
    directions = "-".join(args.directions)
    basins = "all" if not args.only_basin else short_hash("|".join(sorted(args.only_basin)))
    periods = "all" if not args.periods else short_hash("|".join(sorted(args.periods)))
    max_points = "allpoints" if not args.max_points_per_period else f"max{args.max_points_per_period}"
    return f"{CACHE_VERSION}_vote-{args.vote_lag_source}_dirs-{directions}_basins-{basins}_periods-{periods}_{max_points}"


def load_run_settings():
    valid_directions = {"forward", "reverse", "combined"}
    valid_vote_sources = {"selected_argmax", "smoothed_argmax", "raw_argmax"}
    settings = dict(RUN_SETTINGS)

    directions = settings.get("directions")
    if directions is None:
        directions = [settings.get("direction", "combined")]
    elif isinstance(directions, str):
        directions = [directions]
    directions = [str(direction).strip().lower() for direction in directions]
    bad_directions = [direction for direction in directions if direction not in valid_directions]
    if bad_directions:
        raise ValueError(f"Invalid directions: {bad_directions}; choose from {sorted(valid_directions)}")

    device = normalize_device_label(settings["device"])
    config_devices = {
        str(path): normalize_device_label(device_value)
        for path, device_value in dict(settings.get("config_devices", {})).items()
    }

    vote_lag_source = str(settings["vote_lag_source"]).strip().lower()
    if vote_lag_source not in valid_vote_sources:
        raise ValueError(f"RUN_SETTINGS['vote_lag_source'] must be one of {sorted(valid_vote_sources)}")

    return SimpleNamespace(
        configs=settings["configs"],
        shp=settings["shp"],
        name_field=settings["name_field"],
        target=settings["target"],
        directions=directions,
        device=device,
        config_devices=config_devices,
        parallel_configs=bool(settings.get("parallel_configs", False)),
        smooth_window=settings["smooth_window"],
        vote_lag_source=vote_lag_source,
        only_basin=settings["only_basin"],
        periods=settings["periods"],
        max_points_per_period=settings["max_points_per_period"],
        no_figures=bool(settings["no_figures"]),
        reuse_point_spectra_cache=bool(settings["reuse_point_spectra_cache"]),
        write_point_spectra_cache=bool(settings["write_point_spectra_cache"]),
    )


def select_vote_lag(raw_abs, raw_signed, time_step, smooth_window, vote_lag_source, cause_name, effect_name):
    raw_abs = np.asarray(raw_abs, dtype=float)
    raw_signed = np.asarray(raw_signed, dtype=float)
    if raw_abs.size == 0 or not np.any(np.isfinite(raw_abs)):
        return 0, 0, 0, 0.0

    raw_argmax = int(np.nanargmax(np.nan_to_num(raw_abs, nan=0.0)))
    smoothed = base.smooth_spectrum(raw_abs, smooth_window)
    smoothed_argmax = int(np.nanargmax(np.nan_to_num(smoothed, nan=0.0)))

    if vote_lag_source == "raw_argmax":
        return raw_argmax, raw_argmax, smoothed_argmax, float(raw_abs[raw_argmax])
    if vote_lag_source == "smoothed_argmax":
        return smoothed_argmax, raw_argmax, smoothed_argmax, float(smoothed[smoothed_argmax])

    relk_for_lag = np.zeros((1, time_step), dtype=float)
    relk_for_lag[0, :] = raw_signed[::-1]
    lag_info = base.interp.compute_lag_details(
        relk_for_lag,
        0,
        time_step,
        cause_name=cause_name,
        effect_name=effect_name,
    )
    selected_lag = int(np.clip(lag_info["lag"], 0, time_step - 1))
    return selected_lag, raw_argmax, smoothed_argmax, float(lag_info.get("lag_confidence", 0.0))


def point_spectrum_cache_path(output_root, period_key, args):
    return Path(output_root) / "point_spectra" / cache_tag(args) / f"{period_key}_point_rrp_spectra.csv"


def lag_fieldnames(time_step):
    return [f"lag_abs_{lag:02d}" for lag in range(time_step)]


def load_point_spectrum_cache(output_root, period_key, time_step, args):
    path = point_spectrum_cache_path(output_root, period_key, args)
    if not path.exists():
        return None

    records = []
    lag_fields = lag_fieldnames(time_step)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_abs = np.array([float(row[field]) for field in lag_fields], dtype=float)
            row["point_id"] = int(row["point_id"])
            row["lat"] = float(row["lat"])
            row["lon"] = float(row["lon"])
            row["vote_lag"] = int(row["vote_lag"])
            row["raw_argmax_lag"] = int(row["raw_argmax_lag"])
            row["smoothed_argmax_lag"] = int(row["smoothed_argmax_lag"])
            row["vote_strength"] = float(row["vote_strength"])
            row["relA_score"] = float(row["relA_score"])
            row["raw_abs"] = raw_abs
            records.append(row)
    print(f"  Loaded point spectrum cache: {path} ({len(records)} rows)")
    return records


def write_point_spectrum_cache(output_root, period_key, records, time_step, args):
    path = point_spectrum_cache_path(output_root, period_key, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    lag_fields = lag_fieldnames(time_step)
    fieldnames = [
        "point_id",
        "basin",
        "period",
        "period_label",
        "driver",
        "direction",
        "lat",
        "lon",
        "vote_lag",
        "raw_argmax_lag",
        "smoothed_argmax_lag",
        "vote_strength",
        "relA_score",
    ] + lag_fields

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {key: rec[key] for key in fieldnames if key in rec}
            raw_abs = np.asarray(rec["raw_abs"], dtype=float)
            for lag, field in enumerate(lag_fields):
                row[field] = float(raw_abs[lag])
            writer.writerow(row)
    print(f"  [Saved point spectra] {path}")


def append_direction_record(
    records,
    point_idx,
    basin_name,
    period_key,
    period_label,
    driver,
    direction,
    lat,
    lon,
    item,
    time_step,
    smooth_window,
    vote_lag_source,
    cause_name,
    effect_name,
):
    if item is None:
        return

    raw_abs = np.asarray(item["raw_abs"], dtype=float)
    raw_signed = np.asarray(item["raw_signed"], dtype=float)
    vote_lag, raw_argmax_lag, smoothed_argmax_lag, vote_strength = select_vote_lag(
        raw_abs,
        raw_signed,
        time_step,
        smooth_window,
        vote_lag_source,
        cause_name,
        effect_name,
    )
    records.append(
        {
            "point_id": int(point_idx),
            "basin": basin_name,
            "period": period_key,
            "period_label": period_label,
            "driver": driver,
            "direction": direction,
            "lat": float(lat),
            "lon": float(lon),
            "vote_lag": int(vote_lag),
            "raw_argmax_lag": int(raw_argmax_lag),
            "smoothed_argmax_lag": int(smoothed_argmax_lag),
            "vote_strength": float(vote_strength),
            "relA_score": float(item["relA_score"]),
            "raw_abs": raw_abs,
        }
    )


def run_period(config, basin_list, args, target_label, period_key, period_label, output_root, smooth_window):
    seq_len = int(config["model"]["SEQ_LEN"])
    if args.reuse_point_spectra_cache:
        cached = load_point_spectrum_cache(output_root, period_key, seq_len, args)
        if cached is not None:
            return cached

    out_dir = Path(config["analyze"]["OUT_DIR"])
    predictors = list(config["analyze"]["PREDICTORS"])
    target_var = config["analyze"]["TARGET"]
    var_names = predictors.copy()
    if target_var not in var_names:
        var_names.append(target_var)

    target_idx = base.find_var_index(var_names, target_label)
    if target_idx is None:
        raise ValueError(f"Target {target_label} not found in var_names={var_names}")

    driver_indices = {}
    for driver in DRIVERS:
        idx = base.find_var_index(var_names, driver)
        if idx is not None:
            driver_indices[driver] = idx
    if not driver_indices:
        raise ValueError(f"No configured drivers found in var_names={var_names}")

    aux_predictors = config["analyze"].get("AUX_PREDICTORS", [])
    static_vars = config["analyze"].get("STATIC_VARS", [])
    data, aux_data, static_data, coords = base.load_period_cache(out_dir, period_key, aux_predictors, static_vars)
    data, aux_data, static_data = base.scale_period_arrays(data, aux_data, static_data)

    output_window = int(config["model"].get("output_window", 1))
    total_time = data.shape[1]
    n_vars = data.shape[2]
    if total_time < seq_len:
        print(f"  [Skip] {period_label}: T={total_time} < SEQ_LEN={seq_len}")
        return []

    device = resolve_torch_device(args.device)
    print(f"  Device: {device}")
    model, _ = load_model_for_device(config, var_names, device)
    use_geo_encoding = config["train"].get("USE_GEO_ENCODING", False)

    directions = set(args.directions)
    cause_indices = dict(driver_indices)
    cause_indices[target_label] = target_idx
    basin_filter = set(args.only_basin) if args.only_basin else None
    max_points = args.max_points_per_period if args.max_points_per_period and args.max_points_per_period > 0 else data.shape[0]

    records = []
    point_cache = {}
    iterator = range(min(data.shape[0], max_points))
    print(f"  Period {period_label}: points={min(data.shape[0], max_points)}, drivers={list(driver_indices)}")

    for point_idx in tqdm(iterator, desc=f"{target_label} {period_label} voted spectra"):
        coord = coords[point_idx]
        point_key = (round(float(coord["lon"]), 7), round(float(coord["lat"]), 7))
        if point_key not in point_cache:
            point_cache[point_key] = base.assign_basin(coord["lon"], coord["lat"], basin_list)
        basin_name = point_cache[point_key]
        if basin_name is None:
            continue
        if basin_filter and basin_name not in basin_filter:
            continue

        point_data = data[point_idx]
        samples = [
            point_data[t - seq_len : t].reshape(seq_len, n_vars, 1)
            for t in range(seq_len, total_time + 1)
        ]
        if not samples:
            continue

        input_tensor = torch.FloatTensor(np.asarray(samples)).to(device)
        input_tensor.requires_grad = True

        aux_tensor = None
        if aux_data is not None:
            point_aux = aux_data[point_idx]
            aux_samples = [
                point_aux[t - seq_len : t].reshape(seq_len, point_aux.shape[1], 1)
                for t in range(seq_len, total_time + 1)
            ]
            aux_tensor = torch.FloatTensor(np.asarray(aux_samples)).to(device)

        static_tensor = None
        if static_data is not None:
            batch_size = input_tensor.shape[0]
            static_point = np.asarray(static_data[point_idx], dtype=float)
            static_tensor = torch.FloatTensor(np.repeat(static_point[None, :], batch_size, axis=0)).to(device)

        interpreted_indices = set()
        if directions & {"forward", "combined"}:
            interpreted_indices.add(target_idx)
        if directions & {"reverse", "combined"}:
            interpreted_indices.update(driver_indices.values())

        try:
            spectra = base.extract_point_spectra(
                model,
                input_tensor,
                interpreted_indices,
                cause_indices,
                device,
                coord,
                use_geo_encoding,
                seq_len,
                aux_tensor=aux_tensor,
                static_tensor=static_tensor,
            )
        except Exception as exc:
            print(f"    [Point skip] {point_idx}: {exc}")
            continue

        for driver, driver_idx in driver_indices.items():
            forward_item = spectra.get((driver_idx, target_idx))
            reverse_item = spectra.get((target_idx, driver_idx))

            if "forward" in directions:
                append_direction_record(
                    records,
                    point_idx,
                    basin_name,
                    period_key,
                    period_label,
                    driver,
                    "forward",
                    coord["lat"],
                    coord["lon"],
                    forward_item,
                    seq_len,
                    smooth_window,
                    args.vote_lag_source,
                    cause_name=driver,
                    effect_name=target_label,
                )

            if "reverse" in directions:
                append_direction_record(
                    records,
                    point_idx,
                    basin_name,
                    period_key,
                    period_label,
                    driver,
                    "reverse",
                    coord["lat"],
                    coord["lon"],
                    reverse_item,
                    seq_len,
                    smooth_window,
                    args.vote_lag_source,
                    cause_name=target_label,
                    effect_name=driver,
                )

            if "combined" in directions:
                pieces = [item for item in (forward_item, reverse_item) if item is not None]
                if pieces:
                    combined = {
                        "raw_abs": np.mean([piece["raw_abs"] for piece in pieces], axis=0),
                        "raw_signed": np.mean([piece["raw_signed"] for piece in pieces], axis=0),
                        "relA_score": float(np.mean([piece["relA_score"] for piece in pieces])),
                    }
                    append_direction_record(
                        records,
                        point_idx,
                        basin_name,
                        period_key,
                        period_label,
                        driver,
                        "combined",
                        coord["lat"],
                        coord["lon"],
                        combined,
                        seq_len,
                        smooth_window,
                        args.vote_lag_source,
                        cause_name=driver,
                        effect_name=target_label,
                    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.write_point_spectra_cache:
        write_point_spectrum_cache(output_root, period_key, records, seq_len, args)
    return records


def choose_representative_curve(items, time_step):
    if not items:
        return None
    vote_lags = np.array([item["vote_lag"] for item in items], dtype=int)
    counts = np.bincount(np.clip(vote_lags, 0, time_step - 1), minlength=time_step)
    modal_lag = int(np.argmax(counts))
    modal_items = [item for item in items if int(item["vote_lag"]) == modal_lag]
    if not modal_items:
        modal_items = items

    # 不做平均：取投给主导 lag 的点中，该 lag relK 强度最大的完整曲线作为代表。
    rep = max(modal_items, key=lambda item: float(np.asarray(item["raw_abs"])[modal_lag]))
    return rep, modal_lag, counts


def aggregate_voted_records(records, time_step, smooth_window):
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["basin"], rec["driver"], rec["period"], rec["direction"])].append(rec)

    summary_rows = []
    curve_rows = []
    for key, items in sorted(grouped.items()):
        basin, driver, period_key, direction = key
        period_label = dict(PERIODS).get(period_key, period_key)
        rep_pack = choose_representative_curve(items, time_step)
        if rep_pack is None:
            continue
        rep, modal_lag, counts = rep_pack
        vote_freq = counts / max(float(np.sum(counts)), 1.0) * 100.0
        raw_abs = np.asarray(rep["raw_abs"], dtype=float)
        smoothed = base.smooth_spectrum(raw_abs, smooth_window)
        vote_lags = np.array([item["vote_lag"] for item in items], dtype=int)
        raw_argmax_lags = np.array([item["raw_argmax_lag"] for item in items], dtype=int)
        smoothed_argmax_lags = np.array([item["smoothed_argmax_lag"] for item in items], dtype=int)

        summary_rows.append(
            {
                "basin": basin,
                "driver": driver,
                "direction": direction,
                "period": period_key,
                "period_label": period_label,
                "n_points": len(items),
                "modal_vote_lag": modal_lag,
                "modal_vote_percent": float(vote_freq[modal_lag]),
                "mean_vote_lag": float(np.nanmean(vote_lags)),
                "median_vote_lag": float(np.nanmedian(vote_lags)),
                "mean_raw_argmax_lag": float(np.nanmean(raw_argmax_lags)),
                "mean_smoothed_argmax_lag": float(np.nanmean(smoothed_argmax_lags)),
                "representative_point_id": int(rep["point_id"]),
                "representative_lat": float(rep["lat"]),
                "representative_lon": float(rep["lon"]),
                "representative_relA_score": float(rep["relA_score"]),
            }
        )

        for lag in range(time_step):
            curve_rows.append(
                {
                    "basin": basin,
                    "driver": driver,
                    "direction": direction,
                    "period": period_key,
                    "period_label": period_label,
                    "lag": lag,
                    "vote_frequency_percent": float(vote_freq[lag]),
                    "representative_raw_abs_strength": float(raw_abs[lag]),
                    "representative_smoothed_abs_strength": float(smoothed[lag]),
                    "modal_vote_lag": modal_lag,
                    "modal_vote_percent": float(vote_freq[modal_lag]),
                    "n_points": len(items),
                    "representative_point_id": int(rep["point_id"]),
                    "representative_lat": float(rep["lat"]),
                    "representative_lon": float(rep["lon"]),
                }
            )
    return summary_rows, curve_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def write_outputs(output_root, summary_rows, curve_rows):
    table_dir = Path(output_root) / "tables"
    write_csv(
        table_dir / "basin_voted_rrp_lag_spectrum_summary.csv",
        summary_rows,
        [
            "basin",
            "driver",
            "direction",
            "period",
            "period_label",
            "n_points",
            "modal_vote_lag",
            "modal_vote_percent",
            "mean_vote_lag",
            "median_vote_lag",
            "mean_raw_argmax_lag",
            "mean_smoothed_argmax_lag",
            "representative_point_id",
            "representative_lat",
            "representative_lon",
            "representative_relA_score",
        ],
    )
    write_csv(
        table_dir / "basin_voted_rrp_lag_spectrum_curves.csv",
        curve_rows,
        [
            "basin",
            "driver",
            "direction",
            "period",
            "period_label",
            "lag",
            "vote_frequency_percent",
            "representative_raw_abs_strength",
            "representative_smoothed_abs_strength",
            "modal_vote_lag",
            "modal_vote_percent",
            "n_points",
            "representative_point_id",
            "representative_lat",
            "representative_lon",
        ],
    )


def finite_max(values, default=1.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    return max(float(np.nanmax(vals)), default)


def plot_basin_direction(basin, direction, curve_rows, output_root, target, time_step):
    rows = [
        row for row in curve_rows
        if row["basin"] == basin and row["direction"] == direction
    ]
    if not rows:
        return

    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row["driver"], row["period"])][int(row["lag"])] = row

    lags = np.arange(time_step)
    fig, axes = plt.subplots(3, len(DRIVERS), figsize=(13.4, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.12, wspace=0.22, hspace=0.32)

    for col, driver in enumerate(DRIVERS):
        axes[0, col].set_title(f"{target}-{driver}", fontsize=12, fontweight="bold")
        freq_values = []
        raw_values = []
        smooth_values = []

        for period_key, period_label in PERIODS:
            per_lag = lookup.get((driver, period_key), {})
            if not per_lag:
                continue
            vote_freq = np.array([per_lag.get(lag, {}).get("vote_frequency_percent", np.nan) for lag in lags])
            raw = np.array([per_lag.get(lag, {}).get("representative_raw_abs_strength", np.nan) for lag in lags])
            smooth = np.array([per_lag.get(lag, {}).get("representative_smoothed_abs_strength", np.nan) for lag in lags])
            modal_lag = per_lag.get(0, {}).get("modal_vote_lag", np.nan)
            color = PERIOD_COLORS[period_key]
            label = f"{period_label} (mode={modal_lag:g})" if np.isfinite(modal_lag) else period_label

            axes[0, col].plot(lags, vote_freq, color=color, linewidth=1.8, label=label)
            axes[1, col].plot(lags, raw, color=color, linewidth=1.6, alpha=0.92)
            axes[2, col].plot(lags, smooth, color=color, linewidth=2.0)
            freq_values.extend(vote_freq)
            raw_values.extend(raw)
            smooth_values.extend(smooth)

        row_settings = [
            (0, "Vote frequency (%)", freq_values, "Dominant-lag voting distribution"),
            (1, "Representative raw |relK|", raw_values, "Actual curve from modal-lag representative point"),
            (2, "Representative smoothed |relK|", smooth_values, "Smoothed representative curve"),
        ]
        for row_idx, ylabel, values, label in row_settings:
            ax = axes[row_idx, col]
            ax.set_xlim(0, time_step - 1)
            ax.set_xticks(np.arange(0, time_step, 3))
            ax.set_ylim(0, finite_max(values, default=0.1) * 1.12)
            ax.grid(True, color="0.88", linewidth=0.6)
            ax.tick_params(labelsize=8)
            ax.text(
                0.02,
                0.95,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_idx == 2:
                ax.set_xlabel("Lag (months)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(PERIODS), frameon=False, fontsize=8)
    fig.suptitle(f"{basin} | {target} voted RRP lag spectrum ({direction})", fontsize=14, fontweight="bold")

    out_dir = Path(output_root) / direction / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(basin)
    png = out_dir / f"Basin_{safe}_{target}_{direction}_voted_rrp_lag_spectrum.png"
    tif = out_dir / f"Basin_{safe}_{target}_{direction}_voted_rrp_lag_spectrum.tif"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tif, dpi=450, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {tif}")


def plot_figures(curve_rows, output_root, target, time_step, only_basin=None):
    only = set(only_basin or [])
    basins = sorted({row["basin"] for row in curve_rows})
    directions = sorted({row["direction"] for row in curve_rows})
    for basin in basins:
        if only and basin not in only:
            continue
        for direction in directions:
            plot_basin_direction(basin, direction, curve_rows, output_root, target, time_step)


def load_config_job(config_path):
    config = base.load_config(config_path)
    target = base.clean_name(config["analyze"]["TARGET"])
    result_root = Path(config["analyze"]["OUT_DIR"])
    return config, target, result_root


def run_config(config_path, args, basins):
    config, inferred_target, result_root = load_config_job(config_path)
    base.interp.config = config
    target = base.clean_name(args.target) if args.target else inferred_target
    output_root = result_root / "causal_analysis_all_5year" / OUTPUT_SUBDIR
    output_root.mkdir(parents=True, exist_ok=True)
    smooth_window = int(args.smooth_window or config.get("analyze", {}).get("lag_selection", {}).get("smooth_window", 3))
    time_step = int(config["model"]["SEQ_LEN"])

    print(f"\n{'=' * 78}")
    print(f"Config: {config_path}")
    print(f"Target: {target}")
    print(f"Output root: {output_root}")
    print(f"Directions: {', '.join(args.directions)}")
    print(f"Device: {args.device}")
    print(f"Cache tag: {cache_tag(args)}")
    print(f"Vote lag source: {args.vote_lag_source}")
    print(f"Smooth window: {smooth_window}")
    print(f"{'=' * 78}")

    all_records = []
    for period_key, period_label in PERIODS:
        if args.periods and period_key not in set(args.periods):
            continue
        period_records = run_period(
            config,
            basins,
            args,
            target,
            period_key,
            period_label,
            output_root,
            smooth_window,
        )
        all_records.extend(period_records)

    if not all_records:
        print("[Stop] No point spectra generated.")
        return False

    summary_rows, curve_rows = aggregate_voted_records(all_records, time_step, smooth_window)
    if not curve_rows:
        print("[Stop] No voted curves generated.")
        return False
    write_outputs(output_root, summary_rows, curve_rows)
    if not args.no_figures:
        plot_figures(curve_rows, output_root, target, time_step, only_basin=args.only_basin)
    return True


def make_job_args(args, config_path):
    job_args = SimpleNamespace(**vars(args))
    job_args.device = config_device(config_path, args)
    return job_args


def run_config_worker(config_path, args_dict):
    configure_matplotlib()
    args = SimpleNamespace(**args_dict)
    args.device = config_device(config_path, args)
    basins = base.read_basins(args.shp, args.name_field)
    return run_config(config_path, args, basins)


def main():
    configure_matplotlib()
    args = load_run_settings()

    completed = 0
    if args.parallel_configs and len(args.configs) > 1:
        print(f"[Parallel] Launching {len(args.configs)} config jobs.")
        for config_path in args.configs:
            print(f"  {config_path} -> {config_device(config_path, args)}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(args.configs)) as pool:
            results = [
                pool.apply_async(run_config_worker, (config_path, vars(args)))
                for config_path in args.configs
            ]
            for config_path, result in zip(args.configs, results):
                try:
                    completed += int(bool(result.get()))
                except Exception as exc:
                    print(f"[Skip Config] {config_path}: {exc}")
                    import traceback

                    traceback.print_exc()
        print(f"\n[Done] Completed {completed}/{len(args.configs)} config jobs.")
        return

    basins = base.read_basins(args.shp, args.name_field)
    for config_path in args.configs:
        job_args = make_job_args(args, config_path)
        try:
            if run_config(config_path, job_args, basins):
                completed += 1
        except Exception as exc:
            print(f"[Skip Config] {config_path}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n[Done] Completed {completed}/{len(args.configs)} config jobs.")


if __name__ == "__main__":
    main()
