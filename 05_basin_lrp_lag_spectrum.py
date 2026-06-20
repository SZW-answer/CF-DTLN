import csv
import importlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import torch
from matplotlib import font_manager
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point, shape as shapely_shape
from tqdm import tqdm

import interpreted as interp


DEFAULT_CONFIG_PATHS = [
    "./model_config_EVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]
DEFAULT_SHP = "Dataset/NLH/NLH.shp"

# =============================================================================
# Run Settings
# =============================================================================
# 常规运行只需要改这里，不需要在命令行写 parser 参数。
# 本脚本不再读取命令行参数；运行前直接修改这里。
RUN_SETTINGS = {
    # 批量运行的配置文件。备选:
    # ["./model_config_EVI.json"]
    # ["./model_config_SIF.json"]
    # ["./model_config.json"]
    # ["./model_config_EVI.json", "./model_config_SIF.json", "./model_config.json"]
    "configs":["./model_config_EVI.json", "./model_config_SIF.json", "./model_config.json"],

    # NLH 流域面矢量，以及流域名称字段。
    "shp": DEFAULT_SHP,
    "name_field": "name",

    # target=None 表示从每个 config 的 analyze.TARGET 自动识别 EVI/SIF/GPP。
    # 也可手动写 "GPP" / "EVI" / "SIF"。
    "target": None,

    # 可一次运行多个方向。备选:
    # ["forward"]   = driver -> target
    # ["reverse"]   = target -> driver
    # ["combined"]  = 双向平均
    # ["forward", "reverse", "combined"] = 三套结果全部输出
    "directions": ["forward", "reverse", "combined"],

    # cpu 更稳；cuda 更快但需要显存和驱动正常。
    "device": "cuda",

    # 原始 relK 谱的滑动平均窗口。None 表示读取 config analyze.lag_selection.smooth_window，
    # 若 config 没有则使用 3。
    "smooth_window": None,

    # None 表示全流域；也可以指定若干流域:
    # ["塔里木河干流"]
    # ["塔里木河干流", "和田河", "疏勒河"]
    "only_basin": None,

    # None 表示 5 个时期全跑；也可以指定:
    # ["2003_2004"]
    # ["2003_2004", "2005_2009"]
    "periods": None,

    # 调试用，每个时期最多处理多少点。None 表示全点。
    "max_points_per_period": None,

    # True 只输出 CSV，不画曲线/热力图。
    "no_figures": False,
}

PERIODS = [
    ("2003_2004", "2003-2005"),
    ("2005_2009", "2005-2010"),
    ("2010_2014", "2010-2015"),
    ("2015_2019", "2015-2020"),
    ("2020_2024", "2020-2024"),
]
DRIVERS = ["Precipitation", "Runoff", "GWS"]
DRIVER_COLORS = {
    "Precipitation": "#2A9D8F",
    "Runoff": "#E76F51",
    "GWS": "#355C9A",
}
PERIOD_COLORS = {
    "2003_2004": "#313695",
    "2005_2009": "#4575B4",
    "2010_2014": "#1A9850",
    "2015_2019": "#FDAE61",
    "2020_2024": "#D73027",
}


@dataclass
class Basin:
    name: str
    safe_name: str
    geometry: object
    bbox: tuple


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
        }
    )


def safe_filename(value):
    text = str(value).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed_basin"


def clean_name(name):
    low = (
        str(name)
        .replace("_deseasonalized", "")
        .replace("_sum", "")
        .replace("total_", "")
        .strip()
        .lower()
    )
    if "precip" in low:
        return "Precipitation"
    if "runoff" in low or "streamflow" in low:
        return "Runoff"
    if low in {"ga", "gws"} or "groundwater" in low:
        return "GWS"
    if low in {"gosifgpp", "gpp"} or "gpp" in low:
        return "GPP"
    if low in {"gosif", "go_sif"}:
        return "SIF"
    if low in {"evi", "ndvi", "sif"}:
        return low.upper()
    return str(name).replace("_deseasonalized", "").replace("_sum", "")


def load_config(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_basins(shp_path, name_field="name"):
    reader = shapefile.Reader(str(shp_path))
    field_names = [field[0] for field in reader.fields[1:]]
    match = None
    for field in field_names:
        if field.lower() == name_field.lower():
            match = field
            break
    if match is None:
        raise ValueError(f"Cannot find basin name field '{name_field}'. Available fields: {field_names}")

    basins = []
    for idx, shape_record in enumerate(reader.iterShapeRecords()):
        record = shape_record.record.as_dict()
        name = str(record.get(match) or f"basin_{idx + 1}")
        geom = shapely_shape(shape_record.shape.__geo_interface__)
        if geom.is_empty:
            continue
        basins.append(Basin(name=name, safe_name=safe_filename(name), geometry=geom, bbox=geom.bounds))
    return basins


def assign_basin(lon, lat, basins):
    point = Point(float(lon), float(lat))
    for basin in basins:
        xmin, ymin, xmax, ymax = basin.bbox
        if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
            continue
        if basin.geometry.covers(point):
            return basin.name
    return None


def load_period_cache(out_dir, period_key, aux_predictors=None, static_vars=None):
    cache_dir = Path(out_dir) / "temp_cache" / period_key
    data_path = cache_dir / f"data_{period_key}.npy"
    coords_path = cache_dir / f"data_{period_key}_coords.json"
    aux_path = cache_dir / f"data_{period_key}_aux.npy"
    static_path = cache_dir / f"data_{period_key}_static.npy"

    if not data_path.exists() or not coords_path.exists():
        raise FileNotFoundError(
            f"Missing 5-year cache for {period_key}: {data_path} / {coords_path}. "
            "Please run interpreted_5year.py first."
        )

    data = np.load(data_path)
    with coords_path.open("r", encoding="utf-8") as f:
        coords = json.load(f)

    aux_data = np.load(aux_path) if aux_predictors and aux_path.exists() else None
    static_data = np.load(static_path) if static_vars and static_path.exists() else None
    return data, aux_data, static_data, coords


def scale_period_arrays(data, aux_data=None, static_data=None):
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    n, t, v = data.shape
    data = scaler.fit_transform(data.reshape(-1, v)).reshape(n, t, v)

    if aux_data is not None:
        aux_scaler = MinMaxScaler(feature_range=(0.1, 1))
        va = aux_data.shape[2]
        aux_data = aux_scaler.fit_transform(aux_data.reshape(-1, va)).reshape(aux_data.shape)

    if static_data is not None:
        static_scaler = MinMaxScaler(feature_range=(0.1, 1))
        static_data = static_scaler.fit_transform(static_data)

    return data, aux_data, static_data


def resolve_model_path(config):
    out_dir = Path(config["analyze"]["OUT_DIR"])
    model_path = config["analyze"].get("model_path")
    if model_path:
        return Path(model_path)
    best = sorted(out_dir.glob("best_model_rmse_*.pth"))
    if best:
        return best[0]
    checkpoints = sorted(out_dir.glob("checkpoint*.pth"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No model checkpoint found under {out_dir}")


def build_model(config, var_names, seq_len, output_window, device):
    cfg = {
        "n_gpu": 0 if device.type == "cuda" else -1,
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
    model = interp.PredictModel(
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
    return model


def load_model(config, var_names, device):
    seq_len = int(config["model"]["SEQ_LEN"])
    output_window = int(config["model"].get("output_window", 1))
    model = build_model(config, var_names, seq_len, output_window, device)
    model_path = resolve_model_path(config)
    print(f"  Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_path


def find_var_index(var_names, label):
    for idx, name in enumerate(var_names):
        if clean_name(name) == label:
            return idx
    return None


def relk_to_lag_spectrum(relk_aligned, cause_idx, time_step):
    arr = np.asarray(relk_aligned[cause_idx], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    signed = arr[::-1]
    strength = np.abs(arr)[::-1]
    if strength.size < time_step:
        strength = np.pad(strength, (0, time_step - strength.size), constant_values=0.0)
        signed = np.pad(signed, (0, time_step - signed.size), constant_values=0.0)
    return strength[:time_step], signed[:time_step]


def smooth_spectrum(values, window):
    return interp._smooth_lag_spectrum(np.asarray(values, dtype=float), int(window))


def update_aggregate(agg, key, raw_strength, signed_strength, relA_score):
    item = agg[key]
    if item["count"] == 0:
        item["raw_abs_sum"] = np.zeros_like(raw_strength, dtype=float)
        item["raw_signed_sum"] = np.zeros_like(signed_strength, dtype=float)
    item["raw_abs_sum"] += raw_strength
    item["raw_signed_sum"] += signed_strength
    item["score_sum"] += float(relA_score)
    item["count"] += 1


def empty_aggregate():
    return {
        "count": 0,
        "score_sum": 0.0,
        "raw_abs_sum": None,
        "raw_signed_sum": None,
    }


def extract_point_spectra(
    model,
    input_tensor,
    interpreted_indices,
    cause_indices,
    device,
    coord,
    use_geo_encoding,
    time_step,
    aux_tensor=None,
    static_tensor=None,
):
    batch_size = input_tensor.shape[0]
    if use_geo_encoding:
        lat_tensor = torch.full((batch_size,), coord["lat"], dtype=torch.float32).to(device)
        lon_tensor = torch.full((batch_size,), coord["lon"], dtype=torch.float32).to(device)
    else:
        lat_tensor, lon_tensor = None, None

    result = {}
    for interpreted_idx in interpreted_indices:
        relA, relK = interp.generate_RRP_scores_all(
            model,
            input_tensor,
            interpreted_idx,
            device,
            debug=False,
            lat=lat_tensor,
            lon=lon_tensor,
            aux_data=aux_tensor,
            static_vars=static_tensor,
        )
        relA_np = relA.detach().cpu().numpy()
        relK_np = relK.detach().cpu().numpy() if torch.is_tensor(relK) else np.asarray(relK)
        for _, cause_idx in cause_indices.items():
            strength, signed = relk_to_lag_spectrum(relK_np, cause_idx, time_step)
            result[(cause_idx, interpreted_idx)] = {
                "raw_abs": strength,
                "raw_signed": signed,
                "relA_score": float(relA_np[interpreted_idx, cause_idx]),
            }
    return result


def run_period(config, config_path, basin_list, args, target_label, period_key, period_label, agg_by_direction):
    out_dir = Path(config["analyze"]["OUT_DIR"])
    predictors = list(config["analyze"]["PREDICTORS"])
    target_var = config["analyze"]["TARGET"]
    var_names = predictors.copy()
    if target_var not in var_names:
        var_names.append(target_var)

    target_idx = find_var_index(var_names, target_label)
    if target_idx is None:
        raise ValueError(f"Target {target_label} not found in var_names={var_names}")

    driver_indices = {}
    for driver in DRIVERS:
        idx = find_var_index(var_names, driver)
        if idx is not None:
            driver_indices[driver] = idx
    if not driver_indices:
        raise ValueError(f"No configured drivers found in var_names={var_names}")

    aux_predictors = config["analyze"].get("AUX_PREDICTORS", [])
    static_vars = config["analyze"].get("STATIC_VARS", [])
    data, aux_data, static_data, coords = load_period_cache(out_dir, period_key, aux_predictors, static_vars)
    data, aux_data, static_data = scale_period_arrays(data, aux_data, static_data)

    seq_len = int(config["model"]["SEQ_LEN"])
    time_step = seq_len
    n_points, total_time, n_vars = data.shape
    if total_time < seq_len:
        print(f"  [Skip] {period_label}: T={total_time} < SEQ_LEN={seq_len}")
        return

    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_model(config, var_names, device)
    use_geo_encoding = config["train"].get("USE_GEO_ENCODING", False)
    directions = set(args.directions)
    cause_indices = dict(driver_indices)
    cause_indices[target_label] = target_idx
    basin_filter = set(args.only_basin) if args.only_basin else None

    point_cache = {}
    max_points = args.max_points_per_period if args.max_points_per_period and args.max_points_per_period > 0 else n_points
    iterator = range(min(n_points, max_points))
    print(f"  Period {period_label}: points={min(n_points, max_points)}, drivers={list(driver_indices)}")

    for point_idx in tqdm(iterator, desc=f"{clean_name(target_var)} {period_label} spectra"):
        coord = coords[point_idx]
        point_key = (round(float(coord["lon"]), 7), round(float(coord["lat"]), 7))
        if point_key not in point_cache:
            point_cache[point_key] = assign_basin(coord["lon"], coord["lat"], basin_list)
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
            spectra = extract_point_spectra(
                model,
                input_tensor,
                interpreted_indices,
                cause_indices,
                device,
                coord,
                use_geo_encoding,
                time_step,
                aux_tensor=aux_tensor,
                static_tensor=static_tensor,
            )
        except Exception as exc:
            print(f"    [Point skip] {point_idx}: {exc}")
            continue

        for driver, driver_idx in driver_indices.items():
            forward_item = spectra.get((driver_idx, target_idx))
            reverse_item = spectra.get((target_idx, driver_idx))
            for direction in directions:
                pieces = []
                scores = []
                if direction == "forward" and forward_item is not None:
                    pieces.append(forward_item)
                    scores.append(forward_item["relA_score"])
                elif direction == "reverse" and reverse_item is not None:
                    pieces.append(reverse_item)
                    scores.append(reverse_item["relA_score"])
                elif direction == "combined":
                    if forward_item is not None:
                        pieces.append(forward_item)
                        scores.append(forward_item["relA_score"])
                    if reverse_item is not None:
                        pieces.append(reverse_item)
                        scores.append(reverse_item["relA_score"])
                if not pieces:
                    continue

                raw_abs = np.mean([p["raw_abs"] for p in pieces], axis=0)
                raw_signed = np.mean([p["raw_signed"] for p in pieces], axis=0)
                relA_score = float(np.mean(scores)) if scores else 0.0
                key = (basin_name, driver, period_key)
                update_aggregate(agg_by_direction[direction], key, raw_abs, raw_signed, relA_score)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_to_rows(agg, smooth_window, time_step, direction):
    rows = []
    for (basin, driver, period_key), item in sorted(agg.items()):
        period_label = dict(PERIODS).get(period_key, period_key)
        count = item["count"]
        if count <= 0:
            continue
        raw_abs = item["raw_abs_sum"] / count
        raw_signed = item["raw_signed_sum"] / count
        smooth_abs = smooth_spectrum(raw_abs, smooth_window)
        smooth_signed = smooth_spectrum(raw_signed, smooth_window)
        relA_score = item["score_sum"] / count
        for lag in range(time_step):
            rows.append(
                {
                    "basin": basin,
                    "driver": driver,
                    "direction": direction,
                    "period": period_key,
                    "period_label": period_label,
                    "lag": lag,
                    "raw_abs_strength": float(raw_abs[lag]),
                    "raw_signed_strength": float(raw_signed[lag]),
                    "smoothed_abs_strength": float(smooth_abs[lag]),
                    "smoothed_signed_strength": float(smooth_signed[lag]),
                    "mean_relA_score": float(relA_score),
                    "n_points": int(count),
                    "smooth_window": int(smooth_window),
                }
            )
    return rows


def write_rows(rows, output_dir):
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    out = table_dir / "basin_rrp_lag_spectrum_mean.csv"
    fieldnames = [
        "basin",
        "driver",
        "direction",
        "period",
        "period_label",
        "lag",
        "raw_abs_strength",
        "raw_signed_strength",
        "smoothed_abs_strength",
        "smoothed_signed_strength",
        "mean_relA_score",
        "n_points",
        "smooth_window",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {out}")
    return out


def rows_to_lookup(rows):
    lookup = defaultdict(dict)
    for row in rows:
        key = (row["basin"], row["driver"], row["period"])
        lookup[key][int(row["lag"])] = row
    return lookup


def finite_max(values, default=1.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return max(float(np.nanmax(arr)), default)


def plot_basin_curves(basin, rows, target, direction, output_dir, time_step):
    lookup = rows_to_lookup(rows)
    lags = np.arange(time_step)
    fig, axes = plt.subplots(2, len(DRIVERS), figsize=(13.2, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.16, wspace=0.22, hspace=0.32)

    for col, driver in enumerate(DRIVERS):
        raw_values = []
        smoothed_values = []
        axes[0, col].set_title(f"{target}-{driver}", fontsize=12, fontweight="bold")
        for period_key, period_label in PERIODS:
            per_lag = lookup.get((basin, driver, period_key), {})
            if not per_lag:
                continue
            raw = np.array([per_lag.get(lag, {}).get("raw_abs_strength", np.nan) for lag in lags])
            smooth = np.array([per_lag.get(lag, {}).get("smoothed_abs_strength", np.nan) for lag in lags])
            color = PERIOD_COLORS[period_key]
            axes[0, col].plot(lags, raw, color=color, linewidth=1.45, alpha=0.85, label=period_label)
            axes[1, col].plot(lags, smooth, color=color, linewidth=1.8, label=period_label)
            raw_values.extend(raw)
            smoothed_values.extend(smooth)

        for row_idx, ylabel in [(0, "Raw |relK| strength"), (1, "Moving-average |relK| strength")]:
            ax = axes[row_idx, col]
            ax.set_xlim(0, time_step - 1)
            ax.set_xticks(np.arange(0, time_step, 3))
            ax.set_ylim(0, finite_max(raw_values if row_idx == 0 else smoothed_values, default=0.1) * 1.12)
            ax.grid(True, color="0.88", linewidth=0.6)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Lag (months)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(PERIODS), frameon=False, fontsize=9)
    fig.suptitle(f"{basin} | {target} true RRP lag-strength spectrum ({direction})", fontsize=14, fontweight="bold")

    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(basin)
    png = out_dir / f"Basin_{safe}_{target}_{direction}_rrp_lag_spectrum_curves.png"
    tif = out_dir / f"Basin_{safe}_{target}_{direction}_rrp_lag_spectrum_curves.tif"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tif, dpi=450, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {tif}")


def plot_basin_heatmaps(basin, rows, target, direction, output_dir, time_step):
    lookup = rows_to_lookup(rows)
    lags = np.arange(time_step)
    fig, axes = plt.subplots(2, len(DRIVERS), figsize=(13.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.93, top=0.84, bottom=0.12, wspace=0.16, hspace=0.28)

    all_values = []
    grids = {}
    for driver in DRIVERS:
        raw_grid = []
        smooth_grid = []
        for period_key, _ in PERIODS:
            per_lag = lookup.get((basin, driver, period_key), {})
            raw_grid.append([per_lag.get(lag, {}).get("raw_abs_strength", np.nan) for lag in lags])
            smooth_grid.append([per_lag.get(lag, {}).get("smoothed_abs_strength", np.nan) for lag in lags])
        raw_grid = np.asarray(raw_grid, dtype=float)
        smooth_grid = np.asarray(smooth_grid, dtype=float)
        grids[driver] = (raw_grid, smooth_grid)
        all_values.extend(raw_grid[np.isfinite(raw_grid)])
        all_values.extend(smooth_grid[np.isfinite(smooth_grid)])

    vmax = finite_max(all_values, default=0.1)
    last_im = None
    ylabels = [label for _, label in PERIODS]
    for col, driver in enumerate(DRIVERS):
        axes[0, col].set_title(f"{target}-{driver}", fontsize=12, fontweight="bold")
        for row_idx, title in [(0, "Raw |relK|"), (1, "Moving average")]:
            grid = grids[driver][row_idx]
            ax = axes[row_idx, col]
            last_im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks(np.arange(0, time_step, 3))
            ax.set_xticklabels(np.arange(0, time_step, 3))
            ax.set_yticks(np.arange(len(PERIODS)))
            ax.set_yticklabels(ylabels if col == 0 else [])
            if col == 0:
                ax.set_ylabel(title, fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Lag (months)", fontsize=10)
            ax.tick_params(labelsize=8)

    if last_im is not None:
        cax = fig.add_axes([0.945, 0.18, 0.015, 0.58])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("|relK| strength", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"{basin} | {target} true RRP lag-strength heatmap ({direction})", fontsize=14, fontweight="bold")
    out_dir = Path(output_dir) / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(basin)
    png = out_dir / f"Basin_{safe}_{target}_{direction}_rrp_lag_spectrum_heatmap.png"
    tif = out_dir / f"Basin_{safe}_{target}_{direction}_rrp_lag_spectrum_heatmap.tif"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tif, dpi=450, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {tif}")


def plot_figures(rows, output_dir, target, direction, time_step, only_basin=None):
    basins = sorted({row["basin"] for row in rows})
    if only_basin:
        allow = set(only_basin)
        basins = [basin for basin in basins if basin in allow]
    basin_rows = defaultdict(list)
    for row in rows:
        basin_rows[row["basin"]].append(row)
    for basin in basins:
        plot_basin_curves(basin, basin_rows[basin], target, direction, output_dir, time_step)
        plot_basin_heatmaps(basin, basin_rows[basin], target, direction, output_dir, time_step)


def run_config(config_path, args, basins):
    config = load_config(config_path)
    interp.config = config
    target_label = clean_name(args.target) if args.target else clean_name(config["analyze"]["TARGET"])
    base_out_dir = (
        Path(config["analyze"]["OUT_DIR"])
        / "causal_analysis_all_5year"
        / "basin_rrp_lag_spectrum"
    )
    base_out_dir.mkdir(parents=True, exist_ok=True)
    smooth_window = int(args.smooth_window or config.get("analyze", {}).get("lag_selection", {}).get("smooth_window", 3))
    seq_len = int(config["model"]["SEQ_LEN"])

    print(f"\n{'=' * 78}")
    print(f"Config: {config_path}")
    print(f"Target: {target_label}")
    print(f"Output root: {base_out_dir}")
    print(f"Directions: {', '.join(args.directions)}")
    print(f"Smooth window: {smooth_window}")
    print(f"{'=' * 78}")

    agg_by_direction = {direction: defaultdict(empty_aggregate) for direction in args.directions}
    for period_key, period_label in PERIODS:
        if args.periods and period_key not in set(args.periods):
            continue
        run_period(config, config_path, basins, args, target_label, period_key, period_label, agg_by_direction)

    any_output = False
    for direction in args.directions:
        rows = aggregate_to_rows(agg_by_direction[direction], smooth_window, seq_len, direction)
        if not rows:
            print(f"[Stop] No spectra rows generated for {direction}.")
            continue
        out_dir = base_out_dir / direction
        out_dir.mkdir(parents=True, exist_ok=True)
        write_rows(rows, out_dir)
        if not args.no_figures:
            plot_figures(rows, out_dir, target_label, direction, seq_len, only_basin=args.only_basin)
        any_output = True
    return any_output


def load_run_settings():
    valid_directions = {"forward", "reverse", "combined"}
    valid_devices = {"cuda"}
    settings = dict(RUN_SETTINGS)

    directions = settings.get("directions")
    if directions is None:
        directions = [settings.get("direction", "combined")]
    elif isinstance(directions, str):
        directions = [directions]
    directions = [str(direction).strip().lower() for direction in directions]
    bad_directions = [direction for direction in directions if direction not in valid_directions]
    if bad_directions:
        raise ValueError(
            f"RUN_SETTINGS['directions'] has invalid value(s) {bad_directions}; "
            f"must be chosen from {sorted(valid_directions)}"
        )
    if settings["device"] not in valid_devices:
        raise ValueError(f"RUN_SETTINGS['device'] must be one of {sorted(valid_devices)}")

    return SimpleNamespace(
        configs=settings["configs"],
        shp=settings["shp"],
        name_field=settings["name_field"],
        target=settings["target"],
        directions=directions,
        device=settings["device"],
        smooth_window=settings["smooth_window"],
        only_basin=settings["only_basin"],
        periods=settings["periods"],
        max_points_per_period=settings["max_points_per_period"],
        no_figures=bool(settings["no_figures"]),
    )


def main():
    configure_matplotlib()
    args = load_run_settings()
    basins = read_basins(args.shp, args.name_field)
    completed = 0
    for config_path in args.configs:
        try:
            if run_config(config_path, args, basins):
                completed += 1
        except Exception as exc:
            print(f"[Skip Config] {config_path}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n[Done] Completed {completed}/{len(args.configs)} config jobs.")


if __name__ == "__main__":
    main()
