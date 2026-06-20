import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import shapefile
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Point, shape as shapely_shape


PERIODS = [
    ("2003_2004", "2003-2005"),
    ("2005_2009", "2005-2010"),
    ("2010_2014", "2010-2015"),
    ("2015_2019", "2015-2020"),
    ("2020_2024", "2020-2024"),
]
DRIVERS = ["Precipitation", "Runoff", "GWS"]
LAGS = np.arange(0, 25, dtype=int)
CONV_KERNEL_SIGMA = 1.5
DEFAULT_SHP = "Dataset/NLH/NLH.shp"
DEFAULT_RESULT_ROOT = "Result/CausalResult_IterTrain_GPP_desasonalized"
DEFAULT_CONFIG_PATHS = [
    "./model_config_EVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]
PANEL_LETTERS = ["(a)", "(b)", "(c)"]

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

LAG_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "academic_lag_0_24",
    [
        "#24135f",
        "#2446b8",
        "#1689e5",
        "#13c4c6",
        "#32d26b",
        "#b6e441",
        "#f2c43a",
        "#f06a24",
        "#b40426",
    ],
    N=512,
)


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
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
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


def infer_target_from_result_root(result_root):
    name = Path(result_root).name.upper()
    for target in ["GPP", "EVI", "NDVI", "SIF"]:
        if target in name:
            return target
    return "GPP"


def load_job_from_config(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)
    analyze = config.get("analyze", {})
    result_root = analyze.get("OUT_DIR")
    target = analyze.get("TARGET")
    if not result_root or not target:
        raise ValueError(f"{config_path} must contain analyze.OUT_DIR and analyze.TARGET")
    return {
        "config": str(path),
        "result_root": Path(result_root),
        "target": clean_name(target),
    }


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


def point_pair_metrics(edges, target, direction):
    target = clean_name(target)
    pair_data = {driver: {} for driver in DRIVERS}

    for edge in edges:
        cause = clean_name(edge.get("cause_name"))
        effect = clean_name(edge.get("effect_name"))
        score = edge.get("score")
        lag = edge.get("lag")
        if score is None or lag is None:
            continue
        try:
            score = float(score)
            lag = float(lag)
        except (TypeError, ValueError):
            continue

        for driver in DRIVERS:
            if cause == driver and effect == target:
                pair_data[driver]["forward"] = (lag, score)
            elif cause == target and effect == driver:
                pair_data[driver]["reverse"] = (lag, score)

    metrics = {}
    for driver, values in pair_data.items():
        if direction == "forward":
            item = values.get("forward")
        elif direction == "reverse":
            item = values.get("reverse")
        else:
            candidates = [values[key] for key in ("forward", "reverse") if key in values]
            if candidates:
                lag = float(np.nanmean([x[0] for x in candidates]))
                score = float(np.nanmean([x[1] for x in candidates]))
                item = (lag, score)
            else:
                item = None

        if item is None:
            continue
        lag = int(np.clip(round(item[0]), LAGS.min(), LAGS.max()))
        metrics[driver] = {"lag": lag, "score": float(item[1])}
    return metrics


def iter_point_jsons(result_root, period_key):
    folder = Path(result_root) / "causal_analysis_all_5year" / period_key / "point_jsons"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def collect_records(result_root, shp_path, target, direction, name_field):
    basins = read_basins(shp_path, name_field=name_field)
    basin_names = {basin.name for basin in basins}
    point_cache = {}
    records = []

    for period_key, period_label in PERIODS:
        paths = iter_point_jsons(result_root, period_key)
        print(f"[{period_label}] reading {len(paths)} point JSON files")
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"  [skip] failed JSON {path}: {exc}")
                continue

            lat = data.get("lat")
            lon = data.get("lon")
            if lat is None or lon is None:
                continue
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue

            point_key = (round(lon, 7), round(lat, 7))
            if point_key not in point_cache:
                point_cache[point_key] = assign_basin(lon, lat, basins)
            basin_name = point_cache[point_key]
            if basin_name not in basin_names:
                continue

            metrics = point_pair_metrics(data.get("causal_edges", []), target, direction)
            for driver, values in metrics.items():
                records.append(
                    {
                        "basin": basin_name,
                        "period": period_key,
                        "period_label": period_label,
                        "driver": driver,
                        "lag": values["lag"],
                        "score": values["score"],
                        "lat": lat,
                        "lon": lon,
                    }
                )
    return basins, records


def empty_driver_period_stats():
    return {
        "count": 0,
        "lags": np.array([], dtype=int),
        "scores": np.array([], dtype=float),
        "lag_freq": np.full(len(LAGS), np.nan, dtype=float),
        "strength_by_lag": np.full(len(LAGS), np.nan, dtype=float),
        "conv_strength_by_lag": np.full(len(LAGS), np.nan, dtype=float),
        "mean_lag": np.nan,
        "median_lag": np.nan,
        "mean_score": np.nan,
        "boundary_ratio": np.nan,
    }


def gaussian_kernel1d(sigma=CONV_KERNEL_SIGMA, radius=None):
    if radius is None:
        radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def convolved_strength_curve(lags, scores, sigma=CONV_KERNEL_SIGMA):
    if len(lags) == 0:
        return np.full(len(LAGS), np.nan, dtype=float)

    strength_sum = np.zeros(len(LAGS), dtype=float)
    weight_count = np.zeros(len(LAGS), dtype=float)
    valid = np.isfinite(scores)
    for lag, score in zip(np.asarray(lags)[valid], np.asarray(scores)[valid]):
        lag = int(np.clip(round(lag), LAGS.min(), LAGS.max()))
        strength_sum[lag] += float(score)
        weight_count[lag] += 1.0

    if weight_count.sum() == 0:
        return np.full(len(LAGS), np.nan, dtype=float)

    kernel = gaussian_kernel1d(sigma=sigma)
    smooth_sum = np.convolve(strength_sum, kernel, mode="same")
    smooth_weight = np.convolve(weight_count, kernel, mode="same")
    curve = np.divide(
        smooth_sum,
        smooth_weight,
        out=np.full(len(LAGS), np.nan, dtype=float),
        where=smooth_weight > 1e-12,
    )
    return curve


def build_stats(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["basin"], rec["driver"], rec["period"])].append(rec)

    stats = defaultdict(lambda: defaultdict(dict))
    rows = []

    basins = sorted({rec["basin"] for rec in records})
    for basin in basins:
        for driver in DRIVERS:
            for period_key, period_label in PERIODS:
                items = grouped.get((basin, driver, period_key), [])
                if not items:
                    stats[basin][driver][period_key] = empty_driver_period_stats()
                    rows.append(summary_row(basin, driver, period_key, period_label, empty_driver_period_stats()))
                    continue

                lags = np.array([item["lag"] for item in items], dtype=int)
                scores = np.array([item["score"] for item in items], dtype=float)
                counts = np.bincount(np.clip(lags, 0, 24), minlength=25).astype(float)
                lag_freq = counts / max(counts.sum(), 1.0) * 100.0
                strength_by_lag = np.full(len(LAGS), np.nan, dtype=float)
                for lag in LAGS:
                    mask = lags == lag
                    if np.any(mask):
                        strength_by_lag[lag] = float(np.nanmean(scores[mask]))
                conv_strength_by_lag = convolved_strength_curve(lags, scores)

                item_stats = {
                    "count": int(len(items)),
                    "lags": lags,
                    "scores": scores,
                    "lag_freq": lag_freq,
                    "strength_by_lag": strength_by_lag,
                    "conv_strength_by_lag": conv_strength_by_lag,
                    "mean_lag": float(np.nanmean(lags)),
                    "median_lag": float(np.nanmedian(lags)),
                    "mean_score": float(np.nanmean(scores)),
                    "boundary_ratio": float(np.mean(lags >= 24) * 100.0),
                }
                stats[basin][driver][period_key] = item_stats
                rows.append(summary_row(basin, driver, period_key, period_label, item_stats))
    return stats, rows


def summary_row(basin, driver, period_key, period_label, stats):
    return {
        "basin": basin,
        "driver": driver,
        "period": period_key,
        "period_label": period_label,
        "n_points": stats["count"],
        "mean_lag": stats["mean_lag"],
        "median_lag": stats["median_lag"],
        "mean_score": stats["mean_score"],
        "lag24_ratio_percent": stats["boundary_ratio"],
    }


def write_summary_csv(rows, output_dir):
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    out = table_dir / "basin_lag_response_summary.csv"
    fieldnames = [
        "basin",
        "driver",
        "period",
        "period_label",
        "n_points",
        "mean_lag",
        "median_lag",
        "mean_score",
        "lag24_ratio_percent",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {out}")


def write_convolved_curve_csv(stats, output_dir):
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    out = table_dir / "basin_convolved_lag_strength_curves.csv"
    fieldnames = ["basin", "driver", "period", "period_label", "lag", "convolved_strength"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for basin in sorted(stats.keys()):
            for driver in DRIVERS:
                for period_key, period_label in PERIODS:
                    item = stats[basin][driver][period_key]
                    for lag, value in zip(LAGS, item["conv_strength_by_lag"]):
                        writer.writerow(
                            {
                                "basin": basin,
                                "driver": driver,
                                "period": period_key,
                                "period_label": period_label,
                                "lag": int(lag),
                                "convolved_strength": value,
                            }
                        )
    print(f"[Saved] {out}")


def style_axis(ax):
    ax.grid(True, color="0.88", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    ax.tick_params(labelsize=8, width=0.7, length=3)


def finite_max(values, default=1.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    return max(float(np.nanmax(vals)), default)


def finite_limits(values, default=(0.0, 1.0), pad_fraction=0.10, lower_bound=None, upper_bound=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        lo, hi = default
    else:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if math.isclose(lo, hi):
            delta = max(abs(lo) * 0.1, 0.1)
            lo -= delta
            hi += delta
        else:
            pad = (hi - lo) * pad_fraction
            lo -= pad
            hi += pad
    if lower_bound is not None:
        lo = max(lo, lower_bound)
    if upper_bound is not None:
        hi = min(hi, upper_bound)
    if lo >= hi:
        hi = lo + 1e-6
    return lo, hi


def robust_max(values, percentile=98.0, default=1.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    hi = float(np.nanpercentile(vals, percentile))
    return max(hi, default)


def plot_geometry_boundary(ax, geom, **kwargs):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, **kwargs)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            plot_geometry_boundary(ax, part, **kwargs)
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            plot_geometry_boundary(ax, part, **kwargs)


def basin_records_by_key(records, basin_name):
    grouped = defaultdict(list)
    for rec in records:
        if rec["basin"] == basin_name:
            grouped[(rec["driver"], rec["period"])].append(rec)
    return grouped


def set_basin_extent(ax, basin):
    xmin, ymin, xmax, ymax = basin.bbox
    dx = max((xmax - xmin) * 0.06, 0.05)
    dy = max((ymax - ymin) * 0.06, 0.05)
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)


def plot_basin_spatial_panel(basin, records, target, direction, output_dir, metric="lag", spatial_aspect="auto"):
    metric_label = "Lag (months)" if metric == "lag" else "Causal strength"
    values = [rec[metric] for rec in records if rec["basin"] == basin.name and np.isfinite(rec[metric])]
    if not values:
        return

    if metric == "lag":
        cmap = LAG_CMAP
        norm = mcolors.Normalize(vmin=0, vmax=24)
        ticks = np.arange(0, 25, 3)
    else:
        cmap = plt.get_cmap("plasma")
        norm = mcolors.Normalize(vmin=0, vmax=robust_max(values, percentile=98.0, default=0.1))
        ticks = None

    grouped = basin_records_by_key(records, basin.name)
    fig, axes = plt.subplots(len(PERIODS), len(DRIVERS), figsize=(11.8, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.91, bottom=0.105, wspace=0.08, hspace=0.12)

    for col, driver in enumerate(DRIVERS):
        axes[0, col].set_title(f"{PANEL_LETTERS[col]} {target}-{driver}", fontsize=13, fontweight="bold", pad=8)

    last_scatter = None
    for row, (period_key, period_label) in enumerate(PERIODS):
        for col, driver in enumerate(DRIVERS):
            ax = axes[row, col]
            items = grouped.get((driver, period_key), [])
            plot_geometry_boundary(ax, basin.geometry, color="0.15", linewidth=0.8, zorder=3)
            set_basin_extent(ax, basin)
            ax.set_aspect("equal" if spatial_aspect == "equal" else "auto", adjustable="box")

            if items:
                lon = [item["lon"] for item in items]
                lat = [item["lat"] for item in items]
                val = [item[metric] for item in items]
                last_scatter = ax.scatter(
                    lon,
                    lat,
                    c=val,
                    s=7,
                    marker="s",
                    cmap=cmap,
                    norm=norm,
                    linewidths=0,
                    alpha=0.95,
                    zorder=2,
                )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)

            ax.text(
                0.018,
                0.955,
                period_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
                zorder=5,
            )
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("0.35")
            ax.tick_params(labelsize=7, length=2.5, width=0.6)
            if row < len(PERIODS) - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
            if row == len(PERIODS) - 1:
                ax.set_xlabel("Longitude", fontsize=9)

    if last_scatter is not None:
        cbar = fig.colorbar(
            last_scatter,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.032,
            pad=0.035,
            ticks=ticks,
        )
        cbar.set_label(metric_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    direction_label = {
        "forward": "Drivers -> target",
        "reverse": "Target -> drivers",
        "combined": "Bidirectional combined",
    }[direction]
    fig.suptitle(f"{basin.name} | {target} spatial {metric} panel ({direction_label})", fontsize=15, fontweight="bold")

    out_dir = Path(output_dir) / "spatial_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(basin.name)
    png = out_dir / f"Basin_{safe}_{target}_{direction}_spatial_{metric}_panel.png"
    tif = out_dir / f"Basin_{safe}_{target}_{direction}_spatial_{metric}_panel.tif"
    fig.savefig(png, dpi=450)
    fig.savefig(tif, dpi=450, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {tif}")


def plot_basin_panel(basin_name, stats, target, direction, output_dir):
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 11.8), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.925, bottom=0.095, wspace=0.34, hspace=0.34)

    for col, driver in enumerate(DRIVERS):
        axes[0, col].set_title(
            f"{PANEL_LETTERS[col]} {target}-{driver}",
            fontsize=13,
            fontweight="bold",
            pad=7,
        )

        freq_values = []
        strength_values = []
        conv_strength_values = []
        for period_key, period_label in PERIODS:
            item = stats[driver][period_key]
            color = PERIOD_COLORS[period_key]
            if item["count"] > 0:
                axes[0, col].plot(
                    LAGS,
                    item["lag_freq"],
                    color=color,
                    linewidth=1.7,
                    label=period_label,
                )
                axes[1, col].plot(
                    LAGS,
                    item["strength_by_lag"],
                    color=color,
                    linewidth=1.7,
                    marker="o",
                    markersize=2.8,
                    label=period_label,
                )
                axes[2, col].plot(
                    LAGS,
                    item["conv_strength_by_lag"],
                    color=color,
                    linewidth=2.0,
                    label=period_label,
                )
                freq_values.extend(item["lag_freq"])
                strength_values.extend(item["strength_by_lag"])
                conv_strength_values.extend(item["conv_strength_by_lag"])

        axes[0, col].set_xlim(0, 24)
        axes[0, col].set_xticks(np.arange(0, 25, 3))
        axes[0, col].set_ylim(0, finite_max(freq_values, default=10.0) * 1.12)
        axes[0, col].set_ylabel("Lag frequency (%)" if col == 0 else "")
        axes[0, col].text(
            0.02,
            0.95,
            "Optimal-lag distribution",
            transform=axes[0, col].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        style_axis(axes[0, col])

        axes[1, col].set_xlim(0, 24)
        axes[1, col].set_xticks(np.arange(0, 25, 3))
        axes[1, col].set_ylim(*finite_limits(strength_values, default=(0.0, 0.1)))
        axes[1, col].set_ylabel("Mean causal strength" if col == 0 else "")
        axes[1, col].text(
            0.02,
            0.95,
            "Response strength by lag",
            transform=axes[1, col].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        style_axis(axes[1, col])

        axes[2, col].set_xlim(0, 24)
        axes[2, col].set_xticks(np.arange(0, 25, 3))
        axes[2, col].set_ylim(*finite_limits(conv_strength_values, default=(0.0, 0.1)))
        axes[2, col].set_ylabel("Convolved strength" if col == 0 else "")
        axes[2, col].text(
            0.02,
            0.95,
            "Convolved 24-month lag-strength",
            transform=axes[2, col].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        style_axis(axes[2, col])

        period_labels = [label for _, label in PERIODS]
        x = np.arange(len(PERIODS))
        mean_lags = [stats[driver][period_key]["mean_lag"] for period_key, _ in PERIODS]
        mean_scores = [stats[driver][period_key]["mean_score"] for period_key, _ in PERIODS]
        color = DRIVER_COLORS[driver]
        ax_lag = axes[3, col]
        ax_score = ax_lag.twinx()
        lag_line, = ax_lag.plot(x, mean_lags, color=color, marker="s", linewidth=1.9, label="Mean lag (solid)")
        score_line, = ax_score.plot(
            x,
            mean_scores,
            color="0.25",
            marker="o",
            linewidth=1.5,
            linestyle="--",
            label="Mean strength (dashed)",
        )
        ax_lag.set_xticks(x)
        ax_lag.set_xticklabels(period_labels, rotation=28, ha="right")
        ax_lag.set_ylim(*finite_limits(mean_lags, default=(0.0, 24.0), lower_bound=0.0, upper_bound=24.0))
        ax_lag.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_score.set_ylim(*finite_limits(mean_scores, default=(0.0, 0.1)))
        ax_lag.set_ylabel("Mean lag (months)", color=color)
        ax_score.set_ylabel("Mean strength", color="0.25")
        ax_lag.tick_params(axis="y", colors=color)
        ax_score.tick_params(axis="y", colors="0.25")
        ax_lag.text(
            0.02,
            0.95,
            "Temporal change",
            transform=ax_lag.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        style_axis(ax_lag)
        ax_score.tick_params(labelsize=8, width=0.7, length=3)
        for spine in ax_score.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.25")
        ax_lag.legend(
            handles=[lag_line, score_line],
            loc="lower left",
            fontsize=7,
            frameon=False,
            handlelength=2.6,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(PERIODS),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.018),
        )

    direction_label = {
        "forward": "Drivers -> target",
        "reverse": "Target -> drivers",
        "combined": "Bidirectional combined",
    }[direction]
    fig.suptitle(f"{basin_name} | {target} causal lag-response curves ({direction_label})", fontsize=15, fontweight="bold")

    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(basin_name)
    png = out_dir / f"Basin_{safe}_{target}_{direction}_lag_response_panel.png"
    tif = out_dir / f"Basin_{safe}_{target}_{direction}_lag_response_panel.tif"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(tif, dpi=450, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {tif}")


def plot_all_basins(
    basins,
    records,
    stats,
    rows,
    target,
    direction,
    output_dir,
    min_points,
    only_basin=None,
    summary_only=False,
    make_spatial=True,
    spatial_aspect="auto",
):
    write_summary_csv(rows, output_dir)
    write_convolved_curve_csv(stats, output_dir)
    if summary_only:
        print("[Done] Summary-only mode. No basin panels were generated.")
        return

    only_basin = set(only_basin or [])
    basin_lookup = {basin.name: basin for basin in basins}
    plotted = 0
    for basin_name in sorted(stats.keys(), key=lambda x: basin_lookup.get(x, Basin(x, x, None, (0, 0, 0, 0))).safe_name):
        if only_basin and basin_name not in only_basin:
            continue
        total_points = sum(
            stats[basin_name][driver][period_key]["count"]
            for driver in DRIVERS
            for period_key, _ in PERIODS
        )
        if total_points < min_points:
            print(f"[Skip] {basin_name}: only {total_points} driver-period records")
            continue
        plot_basin_panel(basin_name, stats[basin_name], target, direction, output_dir)
        if make_spatial:
            basin = basin_lookup.get(basin_name)
            if basin is not None:
                plot_basin_spatial_panel(
                    basin,
                    records,
                    target,
                    direction,
                    output_dir,
                    metric="lag",
                    spatial_aspect=spatial_aspect,
                )
                plot_basin_spatial_panel(
                    basin,
                    records,
                    target,
                    direction,
                    output_dir,
                    metric="score",
                    spatial_aspect=spatial_aspect,
                )
        plotted += 1
    print(f"[Done] Generated panels for {plotted} basins.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build basin-level 0-24 month lag-response panels from DTLN point JSON outputs and a basin shapefile."
    )
    parser.add_argument("--shp", default=DEFAULT_SHP, help=f"Basin polygon shapefile. Default: {DEFAULT_SHP}")
    parser.add_argument("--name-field", default="name", help="Basin name field in the shapefile. Default: name")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIG_PATHS,
        help="Config JSON paths to run when --result-root is not provided.",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="Run a single DTLN result root instead of reading --configs.",
    )
    parser.add_argument("--target", default=None, help="Target label, e.g. GPP, EVI, SIF. Default: infer from result-root")
    parser.add_argument(
        "--direction",
        choices=["forward", "reverse", "combined"],
        default="combined",
        help="Which causal direction to summarize. Default: combined means bidirectional average.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: <result-root>/causal_analysis_all_5year/basin_lag_response_panels",
    )
    parser.add_argument("--min-points", type=int, default=10, help="Skip basin panels with fewer records. Default: 10")
    parser.add_argument(
        "--only-basin",
        nargs="*",
        default=None,
        help="Optional basin name(s) to plot, matching the shapefile NAME field exactly.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Write the summary CSV only; do not generate figures.")
    parser.add_argument("--no-spatial", action="store_true", help="Only generate curve panels; skip basin spatial map panels.")
    parser.add_argument(
        "--spatial-aspect",
        choices=["auto", "equal"],
        default="auto",
        help="Spatial subplot aspect. auto is compact; equal preserves geographic aspect. Default: auto.",
    )
    return parser.parse_args()


def run_one_job(args, result_root, target, config_path=None):
    output_dir = args.output_dir or result_root / "causal_analysis_all_5year" / "basin_lag_response_panels"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}")
    if config_path:
        print(f"Config: {config_path}")
    print(f"Shapefile: {args.shp}")
    print(f"Result root: {result_root}")
    print(f"Target: {target}")
    print(f"Direction: {args.direction}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 78}")

    basins, records = collect_records(result_root, args.shp, target, args.direction, args.name_field)
    print(f"Loaded basins: {len(basins)}")
    print(f"Matched driver-period point records: {len(records)}")
    if not records:
        print("[Stop] No matched records. Check target/direction/result-root and shapefile coverage.")
        return False

    stats, rows = build_stats(records)
    plot_all_basins(
        basins,
        records,
        stats,
        rows,
        target,
        args.direction,
        output_dir,
        args.min_points,
        only_basin=args.only_basin,
        summary_only=args.summary_only,
        make_spatial=not args.no_spatial,
        spatial_aspect=args.spatial_aspect,
    )
    return True


def main():
    configure_matplotlib()
    args = parse_args()

    if args.result_root:
        result_root = Path(args.result_root)
        target = clean_name(args.target) if args.target else infer_target_from_result_root(result_root)
        run_one_job(args, result_root, target)
        return

    configs = args.configs or DEFAULT_CONFIG_PATHS
    completed = 0
    for config_path in configs:
        try:
            job = load_job_from_config(config_path)
        except Exception as exc:
            print(f"[Skip Config] {config_path}: {exc}")
            continue
        target = clean_name(args.target) if args.target else job["target"]
        if run_one_job(args, job["result_root"], target, config_path=job["config"]):
            completed += 1
    print(f"\n[Done] Completed {completed}/{len(configs)} config jobs.")


if __name__ == "__main__":
    main()
