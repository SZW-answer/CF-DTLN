import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator

try:
    import scipy.io.netcdf as netcdf

    HAS_NETCDF = True
except Exception:
    HAS_NETCDF = False

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# =============================================================================
# User Configuration
# =============================================================================
CONFIG_PATHS = [
    "./model_config_EVI.json",
    # "./model_config_NDVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]

PERIODS = [
    "2003_2004",
    "2005_2009",
    "2010_2014",
    "2015_2019",
    "2020_2024",
]

PERIOD_LABELS = {
    "2003_2004": "2003-2005",
    "2005_2009": "2005-2010",
    "2010_2014": "2010-2015",
    "2015_2019": "2015-2020",
    "2020_2024": "2020-2024",
}

DRIVER_ORDER = ["Precipitation", "Runoff", "GWS"]
GRID_RES = 0.1
USE_CARTOPY = True
FIG_DPI = int(os.environ.get("DTLN_FIG_DPI", "1200"))
SAVE_TIF = os.environ.get("DTLN_SAVE_TIF", "1").strip().lower() not in {"0", "false", "no"}
SAVE_NC = os.environ.get("DTLN_SAVE_NC", "1").strip().lower() not in {"0", "false", "no"}

# Lag display modes:
#   adaptive       = per panel/driver adaptive vmax with power stretch, default.
#   adaptive_linear= per panel/driver adaptive vmax with linear stretch.
#   power          = fixed 0-24 with power stretch.
#   linear         = fixed 0-24 linear.
LAG_DISPLAY_MODE = os.environ.get("DTLN_LAG_DISPLAY_MODE", "adaptive").strip().lower()
LAG_GAMMA = float(os.environ.get("DTLN_LAG_GAMMA", "0.55"))
LAG_VMIN = 0.0
LAG_VMAX = 24.0
LAG_TICKS = np.arange(0, 25, 3, dtype=float)
LAG_ADAPTIVE_PERCENTILE = float(os.environ.get("DTLN_LAG_ADAPTIVE_PERCENTILE", "98"))
LAG_ADAPTIVE_MIN_VMAX = float(os.environ.get("DTLN_LAG_ADAPTIVE_MIN_VMAX", "3"))
LAG_ADAPTIVE_ROUND_TO = float(os.environ.get("DTLN_LAG_ADAPTIVE_ROUND_TO", "1"))

NEW_SUMMARY_DIR = "temporal_summary_new"
RESULT_DIR_CANDIDATES = [
    # "causal_analysis_all_5year_all0624",
    "causal_analysis_all_5year",
    # "causal_analysis_all_5year_noFilm",
    # "causal_analysis_all_5year_ls",
    # "causal_analysis_target_5year",
]
POINT_CSV_CANDIDATES = ["point_causal_graphs_all.csv", "point_causal_graphs.csv"]
DIRECTIONS = ["forward", "reverse", "combined"]
PANEL_BOTTOM = 0.19
PANEL_HSPACE = 0.22
CBAR_FRACTION = 0.026
CBAR_PAD = 0.095
CBAR_HEIGHT = 0.026
CBAR_PAD_FIG = 0.075
PERIOD_LABEL_FONTSIZE = 12
PERIOD_LABEL_X = 0.018
PERIOD_LABEL_Y = 0.965

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# =============================================================================
# Name Helpers
# =============================================================================
VAR_ALIAS = {
    "ga": "GWS",
    "gws": "GWS",
    "groundwater": "GWS",
    "groundwater_storage": "GWS",
    "groundwater_storage_anomaly": "GWS",
    "groundwater_anomaly": "GWS",
    "precipitation": "Precipitation",
    "precip": "Precipitation",
    "pr": "Precipitation",
    "runoff": "Runoff",
    "qs": "Runoff",
    "streamflow": "Runoff",
}

DRIVER_COLORS = {
    "Precipitation": "#2A9D8F",
    "Runoff": "#E76F51",
    "GWS": "#355C9A",
    "None": "#D9D9D9",
}


def strip_name_suffix(name):
    return (
        str(name)
        .replace("_deseasonalized", "")
        .replace("_sum", "")
        .replace("total_", "")
    )


def canonical_name(name):
    return strip_name_suffix(name).strip().lower().replace(" ", "_")


def clean_name(name):
    base = strip_name_suffix(name)
    low = base.lower()
    if low in VAR_ALIAS:
        return VAR_ALIAS[low]
    if "groundwater" in low or low == "ga":
        return "GWS"
    if "precip" in low:
        return "Precipitation"
    if "runoff" in low or "streamflow" in low:
        return "Runoff"
    if low in {"gosif", "go_sif", "sif"}:
        return "SIF"
    if low in {"gosifgpp", "gpp"}:
        return "GPP"
    return base


def clean_target_name(name, config_path=None):
    stem = Path(config_path).stem.upper() if config_path else ""
    low = canonical_name(name)
    if "EVI" in stem:
        return "EVI"
    if "NDVI" in stem:
        return "NDVI"
    if "SIF" in stem:
        return "SIF"
    if "GPP" in stem or low in {"gosifgpp", "gpp", "gosifgpp_deseasonalized"}:
        return "GPP"
    return clean_name(name)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    analyze = cfg["analyze"]
    out_dir = Path(analyze["OUT_DIR"])
    target = analyze["TARGET"]
    target_label = clean_target_name(target, config_path)
    return cfg, out_dir, target, target_label


# =============================================================================
# Plot Helpers
# =============================================================================
def get_lag_cmap():
    colors = [
        "#24135f",
        "#2446b8",
        "#1689e5",
        "#13c4c6",
        "#32d26b",
        "#b6e441",
        "#f2c43a",
        "#f06a24",
        "#b40426",
    ]
    cmap = LinearSegmentedColormap.from_list(
        "academic_lag_0_24", list(zip(np.linspace(0, 1, len(colors)), colors)), N=512
    )
    cmap.set_bad("white")
    return cmap


def make_discrete_lag_cmap(vmax):
    vmax_i = int(np.ceil(float(vmax)))
    vmax_i = max(1, min(vmax_i, int(LAG_VMAX)))
    base = get_lag_cmap()
    colors = base(np.linspace(0.04, 0.96, vmax_i))
    cmap = ListedColormap(colors, name=f"discrete_lag_0_{vmax_i}")
    cmap.set_bad("white")
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    boundaries = np.arange(0, vmax_i + 1, 1, dtype=float)
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm, boundaries


def lag_norm():
    if LAG_DISPLAY_MODE in {"power", "gamma"}:
        return PowerNorm(gamma=LAG_GAMMA, vmin=LAG_VMIN, vmax=LAG_VMAX, clip=True)
    return mcolors.Normalize(vmin=LAG_VMIN, vmax=LAG_VMAX, clip=True)


def _lag_ticks_for_vmax(vmax):
    if vmax <= 6:
        step = 1
    elif vmax <= 12:
        step = 2
    else:
        step = 3
    ticks = np.arange(0, vmax + 1e-9, step, dtype=float)
    if ticks.size == 0 or ticks[0] != 0:
        ticks = np.insert(ticks, 0, 0.0)
    # 不强行追加末尾 vmax，避免 18 和 19 这类标签挤在一起。
    return ticks


def lag_style(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    clipped = np.clip(values, LAG_VMIN, LAG_VMAX)
    mode = LAG_DISPLAY_MODE

    if mode in {"adaptive", "column_adaptive", "adaptive_power", "adaptive_linear"} and clipped.size:
        hi = np.nanpercentile(clipped, LAG_ADAPTIVE_PERCENTILE)
        vmax = max(float(hi), LAG_ADAPTIVE_MIN_VMAX)
        round_to = max(float(LAG_ADAPTIVE_ROUND_TO), 1e-9)
        vmax = np.ceil(vmax / round_to) * round_to
        vmax = float(np.clip(vmax, LAG_ADAPTIVE_MIN_VMAX, LAG_VMAX))
        ticks = _lag_ticks_for_vmax(vmax)
        extend = "max" if np.nanmax(clipped) > vmax else "neither"
        cmap, norm, boundaries = make_discrete_lag_cmap(vmax)
        return cmap, norm, ticks, extend, boundaries

    cmap, norm, boundaries = make_discrete_lag_cmap(LAG_VMAX)
    return cmap, norm, LAG_TICKS, "neither", boundaries


def get_strength_cmap():
    colors = [
        "#3b4cc0",
        "#4b9fc5",
        "#7ec9a5",
        "#c7e89a",
        "#f7fcb9",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#7a0177",
    ]
    cmap = LinearSegmentedColormap.from_list("academic_strength_blue_yellow_red_purple", colors, N=512)
    cmap.set_bad("white")
    return cmap


def set_academic_spines(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.25")


def robust_limits(values, pct=(2, 98), positive=True):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(arr, pct)
    if positive:
        lo = 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = 0.0 if positive else float(np.nanmin(arr))
        hi = max(float(np.nanmax(arr)), lo + 1e-9)
    return float(lo), float(hi)


def to_grid(df, value_col, grid_res=GRID_RES, agg="mean"):
    if df.empty or value_col not in df.columns:
        return None, None, None
    d = df[["lat", "lon", value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None, None, None

    decimals = max(int(-np.log10(grid_res)), 0) + 2
    d["lat_idx"] = ((d["lat"] / grid_res).round() * grid_res).round(decimals)
    d["lon_idx"] = ((d["lon"] / grid_res).round() * grid_res).round(decimals)

    lat_min, lat_max = d["lat_idx"].min(), d["lat_idx"].max()
    lon_min, lon_max = d["lon_idx"].min(), d["lon_idx"].max()
    n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
    n_lon = int(round((lon_max - lon_min) / grid_res)) + 1

    lats = np.round(np.linspace(lat_min, lat_max, n_lat), decimals)
    lons = np.round(np.linspace(lon_min, lon_max, n_lon), decimals)
    grid = d.pivot_table(index="lat_idx", columns="lon_idx", values=value_col, aggfunc=agg)
    grid = grid.reindex(index=lats, columns=lons)
    X, Y = np.meshgrid(lons, lats)
    Z = grid.values.astype(float)
    return X, Y, Z


def compute_common_extent(period_results):
    frames = [r["df"][["lat", "lon"]] for r in period_results if not r["df"].empty]
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None
    decimals = max(int(-np.log10(GRID_RES)), 0) + 2
    lat_idx = ((d["lat"] / GRID_RES).round() * GRID_RES).round(decimals)
    lon_idx = ((d["lon"] / GRID_RES).round() * GRID_RES).round(decimals)
    return float(lon_idx.min()), float(lon_idx.max()), float(lat_idx.min()), float(lat_idx.max())


def _axis_extent(X, Y, extent=None):
    if extent is not None:
        return extent
    if X is None or Y is None:
        return None
    return float(np.nanmin(X)), float(np.nanmax(X)), float(np.nanmin(Y)), float(np.nanmax(Y))


def setup_geo_axis(
    ax,
    X,
    Y,
    use_geo,
    show_ylabel=True,
    show_xticklabels=True,
    show_xlabel=False,
    extent=None,
):
    axis_extent = _axis_extent(X, Y, extent)
    if axis_extent is None:
        set_academic_spines(ax)
        return

    xmin, xmax, ymin, ymax = axis_extent

    if use_geo:
        ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
        xticks = np.linspace(xmin, xmax, 4)
        yticks = np.linspace(ymin, ymax, 4)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".1f"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".1f"))
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.set_xlabel("Longitude" if show_xlabel else "", labelpad=4)
    if not show_xticklabels:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("Latitude")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.grid(False)
    set_academic_spines(ax)


def draw_map(ax, X, Y, Z, cmap, norm, use_geo):
    if use_geo:
        return ax.pcolormesh(
            X, Y, Z, cmap=cmap, norm=norm, shading="nearest", transform=ccrs.PlateCarree()
        )
    return ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="nearest")


def add_aligned_horizontal_cbar(fig, axes, mappable, ticks=None, extend="neither"):
    axes = np.atleast_1d(axes).ravel()
    positions = [ax.get_position() for ax in axes]
    x0 = min(pos.x0 for pos in positions)
    x1 = max(pos.x1 for pos in positions)
    y0 = min(pos.y0 for pos in positions)
    cbar_y = max(0.045, y0 - CBAR_PAD_FIG)
    cax = fig.add_axes([x0, cbar_y, x1 - x0, CBAR_HEIGHT])
    return fig.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        ticks=ticks,
        extend=extend,
        spacing="proportional",
    )


def annotate_period(ax, period):
    ax.text(
        PERIOD_LABEL_X,
        PERIOD_LABEL_Y,
        PERIOD_LABELS.get(period, period.replace("_", "-")),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PERIOD_LABEL_FONTSIZE,
        color="black",
        zorder=20,
    )


def save_figure(fig, path_base):
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    png = path_base.with_suffix(".png")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[Saved] {png}")
    if SAVE_TIF:
        tif = path_base.with_suffix(".tif")
        fig.savefig(tif, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[Saved] {tif}")
    plt.close(fig)


def save_panel_nc(period_results, value_col, output_path, variable_prefix="value", agg="mean"):
    if not SAVE_NC:
        return
    output_path = Path(output_path)
    if not HAS_NETCDF:
        print(f"[Warning] scipy.io.netcdf is unavailable, skipping NC: {output_path}")
        return

    frames = []
    for res in period_results:
        df = res["df"]
        if value_col not in df.columns:
            continue
        d = df[["lat", "lon", value_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            continue
        d = d.copy()
        d["period"] = res["period"]
        frames.append(d)

    if not frames:
        print(f"[Skip] No NC data for {value_col}: {output_path}")
        return

    all_df = pd.concat(frames, ignore_index=True)
    decimals = max(int(-np.log10(GRID_RES)), 0) + 2
    all_df["lat_idx"] = ((all_df["lat"] / GRID_RES).round() * GRID_RES).round(decimals)
    all_df["lon_idx"] = ((all_df["lon"] / GRID_RES).round() * GRID_RES).round(decimals)

    lat_min, lat_max = all_df["lat_idx"].min(), all_df["lat_idx"].max()
    lon_min, lon_max = all_df["lon_idx"].min(), all_df["lon_idx"].max()
    if pd.isna(lat_min) or pd.isna(lon_min):
        print(f"[Skip] Empty NC extent for {value_col}: {output_path}")
        return

    n_lat = int(round((lat_max - lat_min) / GRID_RES)) + 1
    n_lon = int(round((lon_max - lon_min) / GRID_RES)) + 1
    lats = np.round(np.linspace(lat_min, lat_max, n_lat), decimals).astype(np.float64)
    lons = np.round(np.linspace(lon_min, lon_max, n_lon), decimals).astype(np.float64)
    fill_value = np.float32(-9999.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with netcdf.netcdf_file(str(output_path), "w") as f:
            f.createDimension("lat", len(lats))
            f.createDimension("lon", len(lons))
            f.grid_resolution = float(GRID_RES)
            f.source_column = str(value_col)

            lat_v = f.createVariable("lat", "d", ("lat",))
            lat_v[:] = lats
            lat_v.units = "degrees_north"
            lat_v.standard_name = "latitude"
            lat_v.axis = "Y"

            lon_v = f.createVariable("lon", "d", ("lon",))
            lon_v[:] = lons
            lon_v.units = "degrees_east"
            lon_v.standard_name = "longitude"
            lon_v.axis = "X"

            for res in period_results:
                period = res["period"]
                df = res["df"]
                if value_col not in df.columns:
                    continue
                d = df[["lat", "lon", value_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if d.empty:
                    continue
                d = d.copy()
                d["lat_idx"] = ((d["lat"] / GRID_RES).round() * GRID_RES).round(decimals)
                d["lon_idx"] = ((d["lon"] / GRID_RES).round() * GRID_RES).round(decimals)

                grid = d.pivot_table(index="lat_idx", columns="lon_idx", values=value_col, aggfunc=agg)
                grid = grid.reindex(index=lats, columns=lons)

                nc_name = f"{variable_prefix}_{period}".replace("-", "_").replace(" ", "_")
                v = f.createVariable(nc_name, "f", ("lat", "lon"))
                v.missing_value = fill_value
                v._FillValue = fill_value
                v.period = str(period)
                v.source_column = str(value_col)
                v[:] = grid.fillna(float(fill_value)).values.astype(np.float32)
        print(f"[Saved] {output_path}")
    except Exception as exc:
        print(f"[Warning] Failed to save NC {output_path}: {exc}")


# =============================================================================
# Data Processing
# =============================================================================
def resolve_result_root(out_dir):
    override = os.environ.get("DTLN_CAUSAL_INPUT_DIR", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = out_dir / p
        if p.exists():
            return p
        print(f"[Warning] DTLN_CAUSAL_INPUT_DIR does not exist: {p}")

    for name in RESULT_DIR_CANDIDATES:
        root = out_dir / name
        if not root.exists():
            continue
        for period in PERIODS:
            if period_csv_path(root, period) is not None:
                return root
    return out_dir / "causal_analysis_target_5year"


def period_csv_path(result_root, period):
    period_dir = result_root / period
    for filename in POINT_CSV_CANDIDATES:
        path = period_dir / filename
        if path.exists():
            return path
    return None


def _best_edge_by_point(df, cause, effect):
    sub = df[(df["cause_clean"] == cause) & (df["effect_clean"] == effect)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["score", "lag"])
    sub["_abs_score"] = sub["score"].abs()
    sub = sub.sort_values(["point_id", "_abs_score"], ascending=[True, False])
    return sub.drop_duplicates("point_id", keep="first").set_index("point_id")[["score", "lag"]]


def _dominant_from_columns(out, candidates, prefix):
    score_cols = [score_col for _, score_col, _ in candidates]
    if not score_cols:
        out[f"dom_{prefix}_type"] = "None"
        out[f"dom_{prefix}_score"] = np.nan
        out[f"dom_{prefix}_lag"] = np.nan
        return

    score_abs = out[score_cols].abs()
    has_any = score_abs.notna().any(axis=1)
    dom_idx = score_abs.idxmax(axis=1)
    col_to_driver = {score_col: driver for driver, score_col, _ in candidates}
    dom_type = dom_idx.map(col_to_driver).where(has_any, "None")

    out[f"dom_{prefix}_type"] = dom_type
    out[f"dom_{prefix}_score"] = np.nan
    out[f"dom_{prefix}_lag"] = np.nan
    for driver, score_col, lag_col in candidates:
        mask = out[f"dom_{prefix}_type"] == driver
        out.loc[mask, f"dom_{prefix}_score"] = out.loc[mask, score_col]
        out.loc[mask, f"dom_{prefix}_lag"] = out.loc[mask, lag_col]


def _add_combined_columns(out, target_label):
    for driver in DRIVER_ORDER:
        f_score = f"{driver}_to_{target_label}_score"
        f_lag = f"{driver}_to_{target_label}_lag"
        r_score = f"{target_label}_to_{driver}_score"
        r_lag = f"{target_label}_to_{driver}_lag"
        c_score = f"{driver}_{target_label}_combined_score"
        c_lag = f"{driver}_{target_label}_combined_lag"

        existing_scores = [c for c in [f_score, r_score] if c in out.columns]
        existing_lags = [c for c in [f_lag, r_lag] if c in out.columns]
        if existing_scores:
            out[c_score] = out[existing_scores].mean(axis=1)
        if existing_lags:
            out[c_lag] = out[existing_lags].mean(axis=1)

    forward_score = out.get("dom_driver_score", pd.Series(np.nan, index=out.index)).abs()
    reverse_score = out.get("dom_driven_score", pd.Series(np.nan, index=out.index)).abs()
    use_forward = forward_score.fillna(-np.inf) >= reverse_score.fillna(-np.inf)

    out["dom_combined_type"] = out.get("dom_driver_type", pd.Series("None", index=out.index)).where(
        use_forward, out.get("dom_driven_type", pd.Series("None", index=out.index))
    )
    out["dom_combined_score"] = out.get("dom_driver_score", pd.Series(np.nan, index=out.index)).where(
        use_forward, out.get("dom_driven_score", pd.Series(np.nan, index=out.index))
    )
    out["dom_combined_lag"] = out.get("dom_driver_lag", pd.Series(np.nan, index=out.index)).where(
        use_forward, out.get("dom_driven_lag", pd.Series(np.nan, index=out.index))
    )


def read_period_table(result_root, period, target_label):
    csv_path = period_csv_path(result_root, period)
    if csv_path is None:
        print(f"[Skip] Missing CSV under: {result_root / period}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[Skip] Empty CSV: {csv_path}")
        return None

    df["cause_clean"] = df["cause_name"].map(clean_name)
    df["effect_clean"] = df["effect_name"].map(clean_name)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["lag"] = pd.to_numeric(df["lag"], errors="coerce")

    related = df[
        ((df["effect_clean"] == target_label) & (df["cause_clean"].isin(DRIVER_ORDER)))
        | ((df["cause_clean"] == target_label) & (df["effect_clean"].isin(DRIVER_ORDER)))
    ].copy()
    if related.empty:
        print(f"[Skip] No target-driver edges in {csv_path}")
        return None

    base = related[["point_id", "lat", "lon"]].drop_duplicates("point_id").set_index("point_id")
    out = base.copy()

    forward_candidates = []
    reverse_candidates = []
    for driver in DRIVER_ORDER:
        f = _best_edge_by_point(df, driver, target_label)
        if not f.empty:
            score_col = f"{driver}_to_{target_label}_score"
            lag_col = f"{driver}_to_{target_label}_lag"
            out[score_col] = f["score"]
            out[lag_col] = f["lag"]
            forward_candidates.append((driver, score_col, lag_col))

        r = _best_edge_by_point(df, target_label, driver)
        if not r.empty:
            score_col = f"{target_label}_to_{driver}_score"
            lag_col = f"{target_label}_to_{driver}_lag"
            out[score_col] = r["score"]
            out[lag_col] = r["lag"]
            reverse_candidates.append((driver, score_col, lag_col))

    _dominant_from_columns(out, forward_candidates, "driver")
    _dominant_from_columns(out, reverse_candidates, "driven")
    _add_combined_columns(out, target_label)

    out = out.reset_index()
    return out


def build_period_results(result_root, target_label, table_dir):
    period_results = []
    for period in PERIODS:
        table = read_period_table(result_root, period, target_label)
        if table is None:
            continue
        table["period"] = period
        table_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_dir / f"{period}_point_driver_table.csv", index=False, encoding="utf-8-sig")
        period_results.append({"period": period, "df": table})
        print(f"[Loaded] {target_label} {period}: {len(table)} points")
    if period_results:
        all_df = pd.concat([r["df"] for r in period_results], ignore_index=True)
        all_df.to_csv(table_dir / "all_periods_point_driver_table.csv", index=False, encoding="utf-8-sig")
    return period_results


# =============================================================================
# Panel Plotting
# =============================================================================
def direction_label(direction):
    labels = {
        "forward": "Forward",
        "reverse": "Reverse",
        "combined": "BidirectionalCombined",
    }
    return labels.get(direction, direction)


def direction_title(direction):
    labels = {
        "forward": "driver -> target",
        "reverse": "target -> driver",
        "combined": "bidirectional combined",
    }
    return labels.get(direction, direction)


def dominant_columns(direction):
    if direction == "forward":
        return "dom_driver_type", "dom_driver_score", "dom_driver_lag"
    if direction == "reverse":
        return "dom_driven_type", "dom_driven_score", "dom_driven_lag"
    return "dom_combined_type", "dom_combined_score", "dom_combined_lag"


def metric_columns(target_label, driver, metric, direction):
    if direction == "forward":
        return f"{driver}_to_{target_label}_{metric}"
    if direction == "reverse":
        return f"{target_label}_to_{driver}_{metric}"
    return f"{driver}_{target_label}_combined_{metric}"


def plot_dominant_panel(period_results, target_label, output_dir, direction="forward"):
    type_col, score_col, _ = dominant_columns(direction)
    usable = [r for r in period_results if type_col in r["df"].columns and r["df"][type_col].notna().any()]
    if not usable:
        print(f"[Skip] No dominant data for {target_label} {direction}")
        return None

    cats = [d for d in DRIVER_ORDER if any((r["df"][type_col] == d).any() for r in usable)]
    if not cats:
        print(f"[Skip] No dominant driver categories for {target_label} {direction}")
        return None
    cat_map = {c: i for i, c in enumerate(cats)}
    plot_results = []
    for res in usable:
        df = res["df"].copy()
        df["type_code"] = df[type_col].map(cat_map)
        plot_results.append({"period": res["period"], "df": df})

    cmap = mcolors.ListedColormap([DRIVER_COLORS[c] for c in cats])
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, len(cats) + 0.5, 1), cmap.N)
    common_extent = compute_common_extent(plot_results)

    use_geo = USE_CARTOPY and HAS_CARTOPY
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_geo else {}
    n = len(plot_results)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 2.25 * n + 0.75), subplot_kw=subplot_kw)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=PANEL_BOTTOM, hspace=PANEL_HSPACE)
    fig.suptitle(f"{target_label} dominant driver ({direction_title(direction)})", fontweight="bold", y=0.987)

    ims = []
    for i, res in enumerate(plot_results):
        ax = axes[i]
        df = res["df"]
        X, Y, Z = to_grid(df, "type_code", agg="first")
        if X is not None:
            ims.append(draw_map(ax, X, Y, Z, cmap, norm, use_geo))
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        annotate_period(ax, res["period"])
        setup_geo_axis(
            ax,
            X,
            Y,
            use_geo,
            show_ylabel=True,
            show_xticklabels=(i == n - 1),
            show_xlabel=(i == n - 1),
            extent=common_extent,
        )

    if ims:
        cbar = add_aligned_horizontal_cbar(
            fig,
            axes,
            ims[0],
            ticks=np.arange(len(cats)),
            extend="neither",
        )
        cbar.ax.set_xticklabels(cats)
        cbar.set_label(f"Dominant driver ({direction_title(direction)})", labelpad=6)

    out_base = Path(output_dir) / f"Panel_Dominant_Drivers_{direction_label(direction)}_{n}Periods"
    save_panel_nc(plot_results, "type_code", out_base.with_suffix(".nc"), variable_prefix="dominant_driver_code", agg="first")
    save_figure(fig, out_base)
    return out_base.with_suffix(".png")


def plot_metric_panel(period_results, target_label, driver, metric, output_dir, direction="forward"):
    col = metric_columns(target_label, driver, metric, direction)
    usable = [r for r in period_results if col in r["df"].columns]
    if not usable:
        print(f"[Skip] No {metric} data for {target_label}-{driver} {direction}")
        return None
    common_extent = compute_common_extent(usable)

    grids = []
    values = []
    for res in usable:
        X, Y, Z = to_grid(res["df"], col, agg="mean")
        grids.append((X, Y, Z))
        if Z is not None:
            vals = Z[np.isfinite(Z)]
            if vals.size:
                values.append(vals)

    if metric == "lag":
        merged = np.concatenate(values) if values else np.array([])
        cmap, norm, ticks, extend, _ = lag_style(merged)
        label = "Lag (months)"
    else:
        merged = np.concatenate(values) if values else np.array([0.0, 1.0])
        vmin, vmax = robust_limits(merged, positive=True)
        cmap = get_strength_cmap()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ticks = None
        label = "Causal strength"
        extend = "max"

    use_geo = USE_CARTOPY and HAS_CARTOPY
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_geo else {}
    n = len(usable)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 2.25 * n + 0.75), subplot_kw=subplot_kw)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=PANEL_BOTTOM, hspace=PANEL_HSPACE)

    metric_title = "lag" if metric == "lag" else "causal strength"
    fig.suptitle(
        f"{target_label}-{driver} {metric_title} ({direction_title(direction)})",
        fontweight="bold",
        y=0.987,
    )

    ims = []
    for i, res in enumerate(usable):
        ax = axes[i]
        X, Y, Z = grids[i]
        if X is not None:
            ims.append(draw_map(ax, X, Y, Z, cmap, norm, use_geo))
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        annotate_period(ax, res["period"])
        setup_geo_axis(
            ax,
            X,
            Y,
            use_geo,
            show_ylabel=True,
            show_xticklabels=(i == n - 1),
            show_xlabel=(i == n - 1),
            extent=common_extent,
        )

    if ims:
        cbar = add_aligned_horizontal_cbar(
            fig,
            axes,
            ims[0],
            ticks=ticks,
            extend=extend,
        )
        cbar.set_label(label, labelpad=6)
        cbar.ax.tick_params(labelsize=9)

    metric_name = "Lag" if metric == "lag" else "Strength"
    if direction == "forward":
        panel_tag = f"{driver}_to_{target_label}_Forward"
    elif direction == "reverse":
        panel_tag = f"{target_label}_to_{driver}_Reverse"
    else:
        panel_tag = f"{driver}_{target_label}_BidirectionalCombined"
    out_base = Path(output_dir) / f"Panel_{metric_name}_{panel_tag}_{n}Periods"
    save_panel_nc(usable, col, out_base.with_suffix(".nc"), variable_prefix=metric, agg="mean")
    save_figure(fig, out_base)
    return out_base.with_suffix(".png")


def plot_timeseries(period_results, target_label, output_dir, direction="forward"):
    if not period_results:
        return
    if not any(
        metric_columns(target_label, driver, "score", direction) in res["df"].columns
        for res in period_results
        for driver in DRIVER_ORDER
    ):
        print(f"[Skip] No timeseries data for {target_label} {direction}")
        return
    x = np.arange(len(period_results))
    labels = [PERIOD_LABELS.get(r["period"], r["period"].replace("_", "-")) for r in period_results]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for driver in DRIVER_ORDER:
        score = []
        lag = []
        for res in period_results:
            df = res["df"]
            score.append(float(np.nanmean(df.get(metric_columns(target_label, driver, "score", direction), np.nan))))
            lag.append(float(np.nanmean(df.get(metric_columns(target_label, driver, "lag", direction), np.nan))))
        color = DRIVER_COLORS.get(driver)
        axes[0].plot(x, score, marker="o", linewidth=1.8, color=color, label=driver)
        axes[1].plot(x, lag, marker="s", linewidth=1.8, color=color, label=driver)

    axes[0].set_title(f"{target_label}: mean causal strength ({direction_title(direction)})")
    axes[0].set_ylabel("Causal strength")
    axes[1].set_title(f"{target_label}: mean lag ({direction_title(direction)})")
    axes[1].set_ylabel("Lag (months)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("Period")
        ax.grid(False)
        set_academic_spines(ax)
    axes[0].legend(frameon=False)

    out_base = Path(output_dir) / f"TimeSeries_{target_label}_Drivers_{direction_label(direction)}"
    save_figure(fig, out_base)


def run_one_config(config_path):
    cfg, out_dir, target_name, target_label = load_config(config_path)
    root = resolve_result_root(out_dir)
    if not root.exists():
        print(f"[Skip] Result root does not exist: {root}")
        return False

    summary_dir = root / NEW_SUMMARY_DIR
    table_dir = summary_dir / "tables"
    panel_dir = summary_dir / "panels"
    dominant_dir = panel_dir / "dominant"
    lag_dir = panel_dir / "lag"
    strength_dir = panel_dir / "strength"
    ts_dir = summary_dir / "timeseries"
    for p in [table_dir, dominant_dir, lag_dir, strength_dir, ts_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print(f"Config: {config_path}")
    print(f"Target: {target_label} ({target_name})")
    print(f"Input:  {root}")
    print(f"Output: {summary_dir}")
    print("#" * 78)

    period_results = build_period_results(root, target_label, table_dir)
    if not period_results:
        print(f"[Skip] No valid period tables for {target_label}")
        return False

    for direction in DIRECTIONS:
        plot_dominant_panel(period_results, target_label, dominant_dir, direction=direction)
        for driver in DRIVER_ORDER:
            plot_metric_panel(period_results, target_label, driver, "score", strength_dir, direction=direction)
            plot_metric_panel(period_results, target_label, driver, "lag", lag_dir, direction=direction)
        plot_timeseries(period_results, target_label, ts_dir, direction=direction)
    return True


def main():
    completed = 0
    for config_path in CONFIG_PATHS:
        try:
            if run_one_config(config_path):
                completed += 1
        except Exception as exc:
            print(f"[Error] {config_path}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n[Done] Completed {completed}/{len(CONFIG_PATHS)} configs.")


if __name__ == "__main__":
    main()
