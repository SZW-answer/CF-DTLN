import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io.netcdf as netcdf
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, TwoSlopeNorm, PowerNorm
from matplotlib.ticker import MaxNLocator, FixedLocator

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    from skimage import exposure
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

# ================= Configuration Area =================
GRID_RES = 0.1
PERIODS = [
    "2003_2004",
    "2005_2009",
    "2010_2014",
    "2015_2019",
    "2020_2024"
]
PERIOD_DISPLAY_LABELS = {
    "2003_2004": "2003-2005",
    "2005_2009": "2005-2010",
    "2010_2014": "2010-2015",
    "2015_2019": "2015-2020",
    "2020_2024": "2020-2024",
}

# ---- display / style switches ----
USE_CARTOPY = True          # if cartopy is installed, draw lon/lat ticks and labels
USE_HIST_EQ_DISPLAY = False # display only; keep False for publication cross-panel comparability
STRENGTH_HIST_EQ_DISPLAY = True
LAG_HIST_EQ_DISPLAY = False
LAG_DISPLAY_MODE = os.environ.get("DTLN_LAG_DISPLAY_MODE", "adaptive").strip().lower()
STRETCH_PERCENTILES = (5, 95)
STRENGTH_PERCENTILES = (2, 98)
STRENGTH_GAMMA = 0.50
LAG_GAMMA = float(os.environ.get("DTLN_LAG_GAMMA", "0.55"))
LAG_DISCRETE_LEVELS = 24
LAG_VMIN = 0.0
LAG_VMAX = 24.0
LAG_TICKS = np.arange(0, 25, 3, dtype=float)
LAG_ADAPTIVE_PERCENTILE = float(os.environ.get("DTLN_LAG_ADAPTIVE_PERCENTILE", "98"))
LAG_ADAPTIVE_MIN_VMAX = float(os.environ.get("DTLN_LAG_ADAPTIVE_MIN_VMAX", "6"))
FIG_DPI = 700
COMPARE_PERIODS =  [
    "2003_2004",
    "2005_2009",
    "2010_2014",
    "2015_2019",
    "2020_2024"
]

CONFIG_PATHS = ["./model_config_EVI.json", "./model_config_NDVI.json","./model_config_SIF.json", "./model_config.json"]
MAP_TITLE_SIZE = 14
MAP_LABEL_SIZE = 12
MAP_TICK_SIZE = 11
CBAR_LABEL_SIZE = 12
CBAR_TICK_SIZE = 10
PANEL_HSPACE = 0.07
ONLY_TEMPORAL_SUMMARY = True

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})

BASE_OUT_DIR = './Result'
TARGET_VAR = 'EVI_deseasonalized'


def load_runtime_config(config_path):
    global BASE_OUT_DIR, TARGET_VAR
    try:
        with open(config_path, 'rb') as f:
            config = json.load(f)
        BASE_OUT_DIR = config['analyze']['OUT_DIR']
        TARGET_VAR = config['analyze']['TARGET']
        print(f"Configuration file loaded successfully: {config_path}")
        print(f"Output Root Directory: {BASE_OUT_DIR}")
        print(f"Target Variable: {TARGET_VAR}")
        return True
    except Exception as e:
        print(f"[Skip Config] Failed to read configuration file {config_path}: {e}")
        return False


# Keep backward-compatible default behavior for direct function calls/import use.
if os.path.exists(CONFIG_PATHS[0]):
    load_runtime_config(CONFIG_PATHS[0])

# ================= Name and Style Helpers =================
VAR_ALIAS = {
    'ga': 'GWS',
    'gws': 'GWS',
    'groundwater': 'GWS',
    'groundwater_storage': 'GWS',
    'groundwater_storage_anomaly': 'GWS',
    'groundwater_anomaly': 'GWS',
    'precipitation': 'Precipitation',
    'precip': 'Precipitation',
    'pr': 'Precipitation',
    'runoff': 'Runoff',
    'qs': 'Runoff',
    'streamflow': 'Runoff',
}

DRIVER_ORDER = ['GWS', 'Precipitation', 'Runoff']
CORE_TARGETS = ['EVI', 'SIF', 'GPP']
DRIVER_COLORS = {
    'GWS': '#355C9A',
    'Precipitation': '#2A9D8F',
    'Runoff': '#E76F51',
    'Other': '#8D99AE'
}


def _strip_name_suffix(name):
    if not isinstance(name, str):
        return str(name)
    return name.replace('_deseasonalized', '').replace('_sum', '').replace('total_', '')


def canonical_name(name):
    base = _strip_name_suffix(name).strip().lower()
    return base.replace(' ', '_')


def clean_name(name):
    base = _strip_name_suffix(name)
    low = base.lower()
    if low in {'gosif', 'go_sif', 'sif'}:
        return 'GPP'
    if low in VAR_ALIAS:
        return VAR_ALIAS[low]
    if 'groundwater' in low or low == 'ga':
        return 'GWS'
    if 'precip' in low:
        return 'Precipitation'
    if 'runoff' in low or 'streamflow' in low:
        return 'Runoff'
    return base


def clean_target_name(name):
    low = canonical_name(name)
    if low in {'gosif', 'go_sif', 'sif'}:
        return 'SIF'
    return clean_name(name)


def set_academic_spines(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color('0.2')


def get_strength_cmap():
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('white')
    return cmap


def get_lag_cmap():
    # High-contrast sequential palette for a strictly linear 0-24 month scale.
    # Anchor colors correspond to ticks 0, 3, 6, ..., 24.
    colors = [
        '#24135f',  # 0
        '#2446b8',  # 3
        '#1689e5',  # 6
        '#13c4c6',  # 9
        '#32d26b',  # 12
        '#b6e441',  # 15
        '#f2c43a',  # 18
        '#f06a24',  # 21
        '#b40426',  # 24
    ]
    cmap = LinearSegmentedColormap.from_list(
        'academic_lag_linear_high_contrast_0_24',
        list(zip(np.linspace(0, 1, len(colors)), colors)),
        N=512
    )
    cmap.set_bad('white')
    return cmap


def _lag_tick_labels(ticks=LAG_TICKS):
    return [f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in ticks]


def _adaptive_lag_vmax(reference_values):
    vals = np.asarray(reference_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return LAG_VMAX
    vals = np.clip(vals, LAG_VMIN, LAG_VMAX)
    hi = np.nanpercentile(vals, LAG_ADAPTIVE_PERCENTILE)
    # Round to a readable 3-month boundary while preserving short-lag contrast.
    vmax = np.ceil(max(hi, LAG_ADAPTIVE_MIN_VMAX) / 3.0) * 3.0
    return float(np.clip(vmax, LAG_ADAPTIVE_MIN_VMAX, LAG_VMAX))


def _adaptive_lag_ticks(vmax):
    step = 1 if vmax <= 9 else 3
    ticks = np.arange(LAG_VMIN, vmax + 0.5 * step, step, dtype=float)
    if ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)
    return ticks


def _lag_ecdf_transform(Z, reference_values):
    Z_arr = np.asarray(Z, dtype=float)
    out = np.full_like(Z_arr, np.nan, dtype=float)
    vals = np.asarray(reference_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Z_arr
    vals = np.sort(np.clip(vals, LAG_VMIN, LAG_VMAX))
    mask = np.isfinite(Z_arr)
    clipped = np.clip(Z_arr[mask], LAG_VMIN, LAG_VMAX)
    out[mask] = np.searchsorted(vals, clipped, side='right') / max(vals.size, 1)
    return out


def get_lag_display_style(reference_values=None):
    cmap = get_lag_cmap()
    mode = LAG_DISPLAY_MODE
    if mode in {'hist', 'histeq', 'hist_eq', 'equalize'} and reference_values is not None:
        ticks = _lag_ecdf_transform(LAG_TICKS, reference_values)
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        return cmap, norm, ticks, _lag_tick_labels(), 'hist'

    if mode in {'adaptive', 'auto', 'robust'} and reference_values is not None:
        vmax = _adaptive_lag_vmax(reference_values)
        ticks = _adaptive_lag_ticks(vmax)
        norm = mcolors.Normalize(vmin=LAG_VMIN, vmax=vmax, clip=False)
        return cmap, norm, ticks, _lag_tick_labels(ticks), 'adaptive'

    if mode in {'power', 'gamma'}:
        norm = PowerNorm(gamma=LAG_GAMMA, vmin=LAG_VMIN, vmax=LAG_VMAX, clip=True)
        return cmap, norm, LAG_TICKS, _lag_tick_labels(), 'power'

    if mode in {'linear', 'discrete'}:
        norm = mcolors.Normalize(vmin=LAG_VMIN, vmax=LAG_VMAX, clip=True)
        return cmap, norm, LAG_TICKS, _lag_tick_labels(), 'linear'

    norm = mcolors.Normalize(vmin=LAG_VMIN, vmax=LAG_VMAX, clip=True)
    return cmap, norm, LAG_TICKS, _lag_tick_labels(), 'linear'


def get_lag_discrete_style():
    cmap, norm, ticks, ticklabels, _ = get_lag_display_style()
    return cmap, norm, ticks


def prepare_lag_for_display(Z, reference_values=None):
    mode = LAG_DISPLAY_MODE
    if mode in {'hist', 'histeq', 'hist_eq', 'equalize'} and reference_values is not None:
        return _lag_ecdf_transform(Z, reference_values)
    return Z


def get_driver_discrete_cmap(categories):
    colors = []
    for cat in categories:
        colors.append(DRIVER_COLORS.get(cat, DRIVER_COLORS['Other']))
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad('white')
    bounds = np.arange(-0.5, len(categories) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def robust_limits(Z, pct=(2, 98), symmetric=False, positive_only=False):
    vals = Z[np.isfinite(Z)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.nanpercentile(vals, pct)
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-9)
        return -m, m
    if positive_only:
        lo = max(0.0, lo)
        hi = max(lo + 1e-9, hi)
        return lo, hi
    if lo == hi:
        hi = lo + 1e-9
    return lo, hi


def format_period_label(period):
    period = str(period)
    return PERIOD_DISPLAY_LABELS.get(period, period.replace("_", "-"))


def annotate_period_label(ax, period):
    ax.text(
        0.018,
        0.965,
        format_period_label(period),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=MAP_TITLE_SIZE,
        color='black',
        bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.2),
        zorder=20,
    )


class MidpointPowerNorm(mcolors.Normalize):
    """Piecewise power-law stretch around a center value, keeping the center at 0.5.
    gamma < 1 enhances low-to-mid contrast while preserving the original data range.
    """
    def __init__(self, vcenter=0.0, gamma=0.85, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.vcenter = vcenter
        self.gamma = gamma

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = result.data.astype(float)
        self.autoscale_None(data)

        vmin, vmax, vcenter = self.vmin, self.vmax, self.vcenter
        out = np.ma.masked_array(np.empty(data.shape, dtype=float), mask=np.ma.getmask(result))

        neg = data < vcenter
        pos = ~neg

        if vcenter > vmin:
            t = np.clip((data[neg] - vcenter) / (vmin - vcenter), 0, 1)
            out[neg] = 0.5 - 0.5 * np.power(t, self.gamma)
        else:
            out[neg] = 0.0

        if vmax > vcenter:
            t = np.clip((data[pos] - vcenter) / (vmax - vcenter), 0, 1)
            out[pos] = 0.5 + 0.5 * np.power(t, self.gamma)
        else:
            out[pos] = 1.0

        if is_scalar:
            out = out[0]
        return out


def maybe_hist_equalize_for_display(Z, vmin, vmax, enabled=None):
    if enabled is None:
        enabled = USE_HIST_EQ_DISPLAY
    if (not enabled) or (not HAS_SKIMAGE):
        return Z, None

    Z2 = np.full_like(Z, np.nan, dtype=float)
    mask = np.isfinite(Z)
    if not np.any(mask):
        return Z, None

    clipped = np.clip(Z[mask], vmin, vmax)
    scaled = (clipped - vmin) / max(vmax - vmin, 1e-12)
    Z2[mask] = exposure.equalize_hist(scaled)
    return Z2, mcolors.Normalize(vmin=0.0, vmax=1.0)


# ================= Data Helpers =================
def save_combined_nc(df, lat_col, lon_col, data_dict, filename, grid_res=GRID_RES):
    try:
        df = df.copy()
        decimals = max(int(-np.log10(grid_res)), 0) + 2
        df['lat_idx'] = (df[lat_col] / grid_res).round() * grid_res
        df['lon_idx'] = (df[lon_col] / grid_res).round() * grid_res
        df['lat_idx'] = df['lat_idx'].round(decimals)
        df['lon_idx'] = df['lon_idx'].round(decimals)

        lat_min = df['lat_idx'].min()
        lat_max = df['lat_idx'].max()
        lon_min = df['lon_idx'].min()
        lon_max = df['lon_idx'].max()

        if pd.isna(lat_min) or pd.isna(lon_min):
            print(f"[Warning] Data is empty or all NaN, skipping save {filename}")
            return

        n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
        n_lon = int(round((lon_max - lon_min) / grid_res)) + 1

        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)
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
                grid = df.pivot_table(index='lat_idx', columns='lon_idx', values=df_col, aggfunc='mean')
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


def _get_axes(ncols, figsize):
    use_geo = USE_CARTOPY and HAS_CARTOPY
    subplot_kw = {'projection': ccrs.PlateCarree()} if use_geo else {}
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True, subplot_kw=subplot_kw)
    if ncols == 1:
        axes = [axes]
    return fig, axes, use_geo


def _build_ticks(vmin, vmax, n=4):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return []
    if np.isclose(vmin, vmax):
        return [float(vmin)]
    return list(np.linspace(vmin, vmax, n))


def _style_geo_axis(ax, X, Y, use_geo, show_ylabel=True):
    if X is None or Y is None:
        return

    xmin, xmax = np.nanmin(X), np.nanmax(X)
    ymin, ymax = np.nanmin(Y), np.nanmax(Y)

    if use_geo:
        ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
        xticks = _build_ticks(xmin, xmax, n=4)
        yticks = _build_ticks(ymin, ymax, n=4)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f'))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f'))
        if not show_ylabel:
            ax.set_yticklabels([])
        ax.grid(False)
        ax.set_xlabel('Longitude', fontsize=MAP_LABEL_SIZE)
        ax.set_ylabel('Latitude' if show_ylabel else '', fontsize=MAP_LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=MAP_TICK_SIZE)
        set_academic_spines(ax)
    else:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Longitude (deg E)', fontsize=MAP_LABEL_SIZE)
        if show_ylabel:
            ax.set_ylabel('Latitude (deg N)', fontsize=MAP_LABEL_SIZE)
        else:
            ax.set_ylabel('')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis='both', labelsize=MAP_TICK_SIZE)
        ax.grid(False)
        set_academic_spines(ax)


def _draw_continuous_map(ax, X, Y, Z, cmap, norm, use_geo):
    if use_geo:
        return ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest', transform=ccrs.PlateCarree())
    return ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')


# ================= Plotting =================
def plot_combined_map(df, type_col, score_col, lag_col, title_main, filename_base, output_dir):
    if type_col not in df.columns:
        return

    cats_present = [c for c in df[type_col].dropna().unique() if c != 'None']
    cats = [c for c in DRIVER_ORDER if c in cats_present] + [c for c in cats_present if c not in DRIVER_ORDER]
    cat_map = {c: i for i, c in enumerate(cats)}

    df_plot = df.copy()
    df_plot['type_code'] = df_plot[type_col].map(cat_map)

    nc_path = os.path.join(output_dir, f"{filename_base}.nc")
    save_combined_nc(
        df_plot, 'lat', 'lon',
        {'var_type_code': 'type_code', 'score': score_col, 'lag': lag_col},
        nc_path
    )

    fig, axes, use_geo = _get_axes(ncols=3, figsize=(14.8, 4.9))
    fig.suptitle(title_main, fontweight='bold')

    # (a) Driver type - discrete
    ax = axes[0]
    X1, Y1, Z1 = _to_grid(df_plot, 'type_code', GRID_RES, agg='first')
    if X1 is not None and len(cats) > 0:
        cmap_cat, norm_cat = get_driver_discrete_cmap(cats)
        im1 = _draw_continuous_map(ax, X1, Y1, Z1, cmap_cat, norm_cat, use_geo)
        cbar1 = fig.colorbar(im1, ax=ax, ticks=np.arange(len(cats)), fraction=0.046, pad=0.02)
        cbar1.ax.set_yticklabels(cats)
        cbar1.set_label('Dominant driver', fontsize=9)
        cbar1.ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('(a) Dominant driver class')
    _style_geo_axis(ax, X1, Y1, use_geo, show_ylabel=True)

    # (b) Strength - diverging, centered at zero
    ax = axes[1]
    X2, Y2, Z2 = _to_grid(df, score_col, GRID_RES, agg='mean')
    if X2 is not None:
        vmin2, vmax2 = robust_limits(Z2, pct=STRENGTH_PERCENTILES, symmetric=True)
        Z2_draw, norm2_hist = maybe_hist_equalize_for_display(Z2, vmin2, vmax2, enabled=STRENGTH_HIST_EQ_DISPLAY)
        if norm2_hist is None:
            norm2 = MidpointPowerNorm(vcenter=0.0, gamma=STRENGTH_GAMMA, vmin=vmin2, vmax=vmax2)
        else:
            norm2 = norm2_hist
        im2 = _draw_continuous_map(ax, X2, Y2, Z2_draw, get_strength_cmap(), norm2, use_geo)
        cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02, extend='both')
        cbar2.set_label('Causal strength', fontsize=9)
        cbar2.ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('(b) Causal strength')
    _style_geo_axis(ax, X2, Y2, use_geo, show_ylabel=False)

    # (c) Lag - sequential, high contrast
    ax = axes[2]
    X3, Y3, Z3 = _to_grid(df, lag_col, GRID_RES, agg='mean')
    if X3 is not None:
        ref3 = Z3[np.isfinite(Z3)]
        cmap3, norm3, ticks3, ticklabels3, lag_mode3 = get_lag_display_style(ref3)
        Z3_draw = prepare_lag_for_display(Z3, ref3)
        im3 = _draw_continuous_map(ax, X3, Y3, Z3_draw, cmap3, norm3, use_geo)
        extend3 = 'max' if lag_mode3 == 'adaptive' else 'neither'
        cbar3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.02, ticks=ticks3, extend=extend3)
        if lag_mode3 == 'hist':
            cbar3.ax.set_yticklabels(ticklabels3)
        cbar3.set_label('Lag (months)', fontsize=9)
        cbar3.ax.tick_params(labelsize=7, pad=1)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('(c) Dominant lag')
    _style_geo_axis(ax, X3, Y3, use_geo, show_ylabel=False)

    plt.savefig(os.path.join(output_dir, f"{filename_base}.png"), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Combined Map Preview: {filename_base}.png")


def _guess_single_cmap(val_col):
    low = val_col.lower()
    if 'lag' in low:
        return get_lag_cmap(), 'lag'
    return get_strength_cmap(), 'strength'


def plot_single_map(df, val_col, title, filename, output_dir):
    if val_col not in df.columns:
        return

    fig, axes, use_geo = _get_axes(ncols=1, figsize=(6.3, 5.2))
    ax = axes[0]
    X, Y, Z = _to_grid(df, val_col, GRID_RES, agg='mean')

    if X is not None:
        cmap, mode = _guess_single_cmap(val_col)
        if mode == 'lag':
            ref = Z[np.isfinite(Z)]
            cmap, norm, lag_ticks, lag_ticklabels, lag_mode = get_lag_display_style(ref)
            Z_draw = prepare_lag_for_display(Z, ref)
            label = 'Lag (months)'
            extend = 'max' if lag_mode == 'adaptive' else 'neither'
        else:
            vmin, vmax = robust_limits(Z, pct=STRENGTH_PERCENTILES, symmetric=True)
            Z_draw, hist_norm = maybe_hist_equalize_for_display(Z, vmin, vmax, enabled=STRENGTH_HIST_EQ_DISPLAY)
            norm = hist_norm if hist_norm is not None else MidpointPowerNorm(vcenter=0.0, gamma=STRENGTH_GAMMA, vmin=vmin, vmax=vmax)
            label = 'Causal strength'
            extend = 'both'

        im = _draw_continuous_map(ax, X, Y, Z_draw, cmap, norm, use_geo)
        if mode == 'lag':
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend=extend, ticks=lag_ticks)
            if lag_mode == 'hist':
                cbar.ax.set_yticklabels(lag_ticklabels)
            cbar.ax.tick_params(labelsize=7, pad=1)
        else:
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend=extend)
            cbar.ax.tick_params(labelsize=8)
        cbar.set_label(label, fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=11)
    _style_geo_axis(ax, X, Y, use_geo, show_ylabel=True)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    save_combined_nc(df, 'lat', 'lon', {'value': val_col}, os.path.join(output_dir, f"{filename}.nc"))


def _iter_period_results(period_results, periods):
    result_map = {r['period']: r for r in period_results}
    return [result_map[p] for p in periods if p in result_map]


def plot_driver_timeseries(period_results, target_label, output_dir):
    if not period_results:
        return
    ordered = _iter_period_results(period_results, PERIODS)
    if not ordered:
        return

    x_labels = [r['period'] for r in ordered]
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.2), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    for driver in DRIVER_ORDER:
        forward_scores = []
        reverse_scores = []
        forward_lags = []
        reverse_lags = []
        combined_scores = []
        combined_lags = []
        for res in ordered:
            df = res['df']
            f_s_col = f"{driver}_to_{target_label}_score"
            r_s_col = f"{target_label}_to_{driver}_score"
            f_l_col = f"{driver}_to_{target_label}_lag"
            r_l_col = f"{target_label}_to_{driver}_lag"
            fs = float(np.nanmean(df[f_s_col])) if f_s_col in df.columns else np.nan
            rs = float(np.nanmean(df[r_s_col])) if r_s_col in df.columns else np.nan
            fl = float(np.nanmean(df[f_l_col])) if f_l_col in df.columns else np.nan
            rl = float(np.nanmean(df[r_l_col])) if r_l_col in df.columns else np.nan
            forward_scores.append(fs)
            reverse_scores.append(rs)
            forward_lags.append(fl)
            reverse_lags.append(rl)
            combined_scores.append(float(np.nanmean([fs, rs])))
            combined_lags.append(float(np.nanmean([fl, rl])))

        color = DRIVER_COLORS.get(driver, DRIVER_COLORS['Other'])
        ax1.plot(x, forward_scores, marker='o', linewidth=1.8, color=color, label=driver)
        ax2.plot(x, reverse_scores, marker='o', linewidth=1.8, color=color, label=driver)
        ax3.plot(x, forward_lags, marker='s', linewidth=1.8, color=color, label=driver)
        ax4.plot(x, reverse_lags, marker='s', linewidth=1.8, color=color, label=driver)
        ax5.plot(x, combined_scores, marker='^', linewidth=1.8, color=color, label=driver)
        ax6.plot(x, combined_lags, marker='^', linewidth=1.8, color=color, label=driver)

    ax1.set_title(f'(a) Driver -> {target_label} strength')
    ax1.set_ylabel('Strength')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(False)
    set_academic_spines(ax1)
    ax1.legend(loc='best', frameon=False)

    ax2.set_title(f'(b) {target_label} -> driver strength')
    ax2.set_ylabel('Strength')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.grid(False)
    set_academic_spines(ax2)

    ax3.set_title(f'(c) Driver -> {target_label} lag')
    ax3.set_ylabel('Lag (months)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_xlabel('Period')
    ax3.grid(False)
    set_academic_spines(ax3)

    ax4.set_title(f'(d) {target_label} -> driver lag')
    ax4.set_ylabel('Lag (months)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels)
    ax4.set_xlabel('Period')
    ax4.grid(False)
    set_academic_spines(ax4)

    ax5.set_title(f'(e) Combined (both directions) strength')
    ax5.set_ylabel('Strength')
    ax5.set_xticks(x)
    ax5.set_xticklabels(x_labels)
    ax5.set_xlabel('Period')
    ax5.grid(False)
    set_academic_spines(ax5)

    ax6.set_title(f'(f) Combined (both directions) lag')
    ax6.set_ylabel('Lag (months)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(x_labels)
    ax6.set_xlabel('Period')
    ax6.grid(False)
    set_academic_spines(ax6)

    out = os.path.join(output_dir, f'TimeSeries_Bidirectional_Strength_Lag_{target_label}_and_Drivers.png')
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Time Series: {out}")


def plot_weighted_interaction_timeseries(period_results, target_label, output_dir):
    if not period_results:
        return
    ordered = _iter_period_results(period_results, PERIODS)
    if not ordered:
        return

    x_labels = [r['period'] for r in ordered]
    x = np.arange(len(x_labels))
    pos_series = []
    neg_series = []
    net_series = []
    coupling_series = []

    for res in ordered:
        df = res['df']
        vals = []
        for driver in DRIVER_ORDER:
            c1 = f"{driver}_to_{target_label}_score"
            c2 = f"{target_label}_to_{driver}_score"
            if c1 in df.columns:
                vals.extend(df[c1].replace([np.inf, -np.inf], np.nan).dropna().tolist())
            if c2 in df.columns:
                vals.extend(df[c2].replace([np.inf, -np.inf], np.nan).dropna().tolist())

        if len(vals) == 0:
            pos_series.append(np.nan)
            neg_series.append(np.nan)
            net_series.append(np.nan)
            coupling_series.append(np.nan)
            continue

        arr = np.asarray(vals, dtype=float)
        pos = arr[arr > 0]
        neg = np.abs(arr[arr < 0])

        pos_w = float(np.mean(pos)) if pos.size else 0.0
        neg_w = float(np.mean(neg)) if neg.size else 0.0
        net = pos_w - neg_w
        coupling = 0.5 * (pos_w + neg_w)

        pos_series.append(pos_w)
        neg_series.append(neg_w)
        net_series.append(net)
        coupling_series.append(coupling)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), constrained_layout=True)
    ax1, ax2 = axes
    ax1.plot(x, pos_series, marker='o', linewidth=2.0, color='#C53334', label='Positive weighted')
    ax1.plot(x, neg_series, marker='o', linewidth=2.0, color='#2C6BA0', label='Negative weighted (abs)')
    ax1.set_title(f'(a) Positive/negative weighted interaction ({target_label} <-> drivers)')
    ax1.set_ylabel('Weighted strength')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(False)
    set_academic_spines(ax1)
    ax1.legend(loc='best', frameon=False)

    ax2.plot(x, net_series, marker='s', linewidth=2.0, color='#5E548E', label='Signed net (pos - neg)')
    ax2.plot(x, coupling_series, marker='s', linewidth=2.0, color='#2A9D8F', label='Coupling index 0.5*(pos+neg)')
    ax2.set_title(f'(b) Bidirectional interaction indices ({target_label} <-> drivers)')
    ax2.set_ylabel('Index')
    ax2.set_xlabel('Period')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.grid(False)
    set_academic_spines(ax2)
    ax2.legend(loc='best', frameon=False)

    out = os.path.join(output_dir, f'TimeSeries_Weighted_PosNeg_Interaction_{target_label}_Drivers.png')
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Weighted Interaction Time Series: {out}")


def _get_available_nodes(period_results):
    nodes = set()
    for res in period_results:
        for c in res['df'].columns:
            if '_to_' in c and c.endswith('_score'):
                src, dst = c[:-6].split('_to_', 1)
                nodes.add(src)
                nodes.add(dst)
    return nodes


def plot_multitarget_bidirectional_strength_timeseries(period_results, target_label, output_dir):
    if not period_results:
        return
    ordered = _iter_period_results(period_results, PERIODS)
    if not ordered:
        return

    nodes = _get_available_nodes(ordered)
    candidates = []
    for t in [target_label] + CORE_TARGETS:
        if t in nodes and t not in candidates:
            candidates.append(t)
    if not candidates:
        return

    x_labels = [r['period'] for r in ordered]
    x = np.arange(len(x_labels))
    n = len(candidates)
    fig, axes = plt.subplots(n, 2, figsize=(13.2, 3.2 * n), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, t in enumerate(candidates):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        for driver in DRIVER_ORDER:
            forward = []
            reverse = []
            for res in ordered:
                df = res['df']
                c1 = f"{t}_to_{driver}_score"
                c2 = f"{driver}_to_{t}_score"
                forward.append(float(np.nanmean(df[c1])) if c1 in df.columns else np.nan)
                reverse.append(float(np.nanmean(df[c2])) if c2 in df.columns else np.nan)

            color = DRIVER_COLORS.get(driver, DRIVER_COLORS['Other'])
            ax_l.plot(x, forward, marker='o', linewidth=1.6, color=color, label=driver)
            ax_r.plot(x, reverse, marker='o', linewidth=1.6, color=color, label=driver)

        ax_l.set_title(f'{t} -> Drivers (Strength)')
        ax_l.set_ylabel('Strength')
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(x_labels)
        ax_l.grid(False)
        set_academic_spines(ax_l)

        ax_r.set_title(f'Drivers -> {t} (Strength)')
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(x_labels)
        ax_r.grid(False)
        set_academic_spines(ax_r)
        if i == 0:
            ax_r.legend(loc='best', frameon=False)

    axes[-1, 0].set_xlabel('Period')
    axes[-1, 1].set_xlabel('Period')

    out = os.path.join(output_dir, 'TimeSeries_MultiTarget_Bidirectional_Strength.png')
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Multi-Target Bidirectional Time Series: {out}")


def plot_driver_type_period_panel(period_results, output_dir, direction='forward'):
    if not period_results:
        return
    if direction not in {'forward', 'reverse', 'combined'}:
        return

    cats_present = set()
    for res in period_results:
        d = res['df']
        if direction == 'forward':
            col = 'dom_driver_type'
            if col in d.columns:
                cats_present.update([c for c in d[col].dropna().unique() if c != 'None'])
        elif direction == 'reverse':
            col = 'dom_driven_type'
            if col in d.columns:
                cats_present.update([c for c in d[col].dropna().unique() if c != 'None'])
        else:
            t1 = d['dom_driver_type'] if 'dom_driver_type' in d.columns else pd.Series('None', index=d.index)
            t2 = d['dom_driven_type'] if 'dom_driven_type' in d.columns else pd.Series('None', index=d.index)
            s1 = d['dom_driver_score'].abs() if 'dom_driver_score' in d.columns else pd.Series(np.nan, index=d.index)
            s2 = d['dom_driven_score'].abs() if 'dom_driven_score' in d.columns else pd.Series(np.nan, index=d.index)
            use_forward = s1.fillna(-np.inf) >= s2.fillna(-np.inf)
            t_combo = t1.where(use_forward, t2)
            t_combo = t_combo.fillna(t1).fillna(t2)
            cats_present.update([c for c in t_combo.dropna().unique() if c != 'None'])
    cats = [c for c in DRIVER_ORDER if c in cats_present] + [c for c in sorted(cats_present) if c not in DRIVER_ORDER]
    if not cats:
        return

    cat_map = {c: i for i, c in enumerate(cats)}
    n = len(period_results)
    ncols = 1 if n >= 5 else 2
    nrows = int(np.ceil(n / ncols))
    use_geo = USE_CARTOPY and HAS_CARTOPY
    subplot_kw = {'projection': ccrs.PlateCarree()} if use_geo else {}
    fig_h = max(3.05 * nrows + 1.1, 6.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, fig_h), constrained_layout=False, subplot_kw=subplot_kw)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.15, hspace=PANEL_HSPACE)
    axes = np.atleast_1d(axes).ravel()

    cmap_cat, norm_cat = get_driver_discrete_cmap(cats)
    ims = []
    active_axes = []

    for idx, res in enumerate(period_results):
        ax = axes[idx]
        active_axes.append(ax)
        df = res['df'].copy()
        if direction == 'forward':
            df['type_code'] = df['dom_driver_type'].map(cat_map) if 'dom_driver_type' in df.columns else np.nan
        elif direction == 'reverse':
            df['type_code'] = df['dom_driven_type'].map(cat_map) if 'dom_driven_type' in df.columns else np.nan
        else:
            t1 = df['dom_driver_type'] if 'dom_driver_type' in df.columns else pd.Series('None', index=df.index)
            t2 = df['dom_driven_type'] if 'dom_driven_type' in df.columns else pd.Series('None', index=df.index)
            s1 = df['dom_driver_score'].abs() if 'dom_driver_score' in df.columns else pd.Series(np.nan, index=df.index)
            s2 = df['dom_driven_score'].abs() if 'dom_driven_score' in df.columns else pd.Series(np.nan, index=df.index)
            use_forward = s1.fillna(-np.inf) >= s2.fillna(-np.inf)
            t_combo = t1.where(use_forward, t2)
            t_combo = t_combo.fillna(t1).fillna(t2)
            df['type_code'] = t_combo.map(cat_map)
        X, Y, Z = _to_grid(df, 'type_code', GRID_RES, agg='first')
        if X is not None:
            im = _draw_continuous_map(ax, X, Y, Z, cmap_cat, norm_cat, use_geo)
            ims.append(im)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        _style_geo_axis(ax, X, Y, use_geo, show_ylabel=True)
        annotate_period_label(ax, res['period'])
        if idx < (n - 1):
            ax.set_xlabel('')
            ax.set_xticklabels([])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if ims:
        cbar = fig.colorbar(
            ims[0], ax=active_axes, ticks=np.arange(len(cats)),
            orientation='horizontal', fraction=0.032, pad=0.02
        )
        cbar.ax.set_xticklabels(cats)
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
        if direction == 'forward':
            cbar.set_label('Dominant driver (driver -> target)', fontsize=CBAR_LABEL_SIZE)
        elif direction == 'reverse':
            cbar.set_label('Dominant driven factor (target -> driver)', fontsize=CBAR_LABEL_SIZE)
        else:
            cbar.set_label('Dominant factor (bidirectional combined)', fontsize=CBAR_LABEL_SIZE)

    if direction == 'forward':
        tag = 'Drivers_Forward'
    elif direction == 'reverse':
        tag = 'Drivers_Reverse'
    else:
        tag = 'Drivers_BidirectionalCombined'
    out = os.path.join(output_dir, f'Panel_Dominant_{tag}_{n}Periods.png')
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Period Driver Panel: {out}")


def plot_driver_metric_period_panel(period_results, target_label, driver, metric, output_dir, direction='forward'):
    if direction not in {'forward', 'reverse', 'combined'}:
        return

    if direction == 'forward':
        col = f"{driver}_to_{target_label}_{metric}"
        usable = [r for r in period_results if col in r['df'].columns]
    elif direction == 'reverse':
        col = f"{target_label}_to_{driver}_{metric}"
        usable = [r for r in period_results if col in r['df'].columns]
    else:
        f_col = f"{driver}_to_{target_label}_{metric}"
        r_col = f"{target_label}_to_{driver}_{metric}"
        usable = [r for r in period_results if (f_col in r['df'].columns or r_col in r['df'].columns)]

    if not usable:
        return

    n = len(usable)
    ncols = 1 if n >= 5 else 2
    nrows = int(np.ceil(n / ncols))
    use_geo = USE_CARTOPY and HAS_CARTOPY
    subplot_kw = {'projection': ccrs.PlateCarree()} if use_geo else {}
    fig_h = max(3.05 * nrows + 1.1, 6.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, fig_h), constrained_layout=False, subplot_kw=subplot_kw)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.15, hspace=PANEL_HSPACE)
    axes = np.atleast_1d(axes).ravel()

    grids = []
    vals = []
    for res in usable:
        if direction == 'combined':
            d = res['df'].copy()
            combo_col = f'__combo_{driver}_{metric}'
            d[combo_col] = np.nan
            has_f = f_col in d.columns
            has_r = r_col in d.columns
            if has_f and has_r:
                d[combo_col] = d[[f_col, r_col]].mean(axis=1)
            elif has_f:
                d[combo_col] = d[f_col]
            elif has_r:
                d[combo_col] = d[r_col]
            X, Y, Z = _to_grid(d, combo_col, GRID_RES, agg='mean')
        else:
            X, Y, Z = _to_grid(res['df'], col, GRID_RES, agg='mean')
        grids.append((X, Y, Z))
        if Z is not None:
            v = Z[np.isfinite(Z)]
            if v.size:
                vals.append(v)

    vmin = None
    vmax = None
    lag_ticks = None
    lag_ticklabels = None
    lag_mode = None
    if vals:
        merged = np.concatenate(vals)
        if metric == 'lag':
            cmap, norm, lag_ticks, lag_ticklabels, lag_mode = get_lag_display_style(merged)
            cbar_label = 'Lag (months)'
            extend = 'max' if lag_mode == 'adaptive' else 'neither'
        else:
            vmin, vmax = robust_limits(merged, pct=STRENGTH_PERCENTILES, symmetric=True)
            cmap = get_strength_cmap()
            norm = MidpointPowerNorm(vcenter=0.0, gamma=STRENGTH_GAMMA, vmin=vmin, vmax=vmax)
            cbar_label = 'Causal strength'
            extend = 'both'
    else:
        if metric == 'lag':
            cmap, norm, lag_ticks, lag_ticklabels, lag_mode = get_lag_display_style()
            cbar_label = 'Lag (months)'
            extend = 'neither'
        else:
            cmap = get_strength_cmap()
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            cbar_label = 'Causal strength'
            extend = 'neither'

    ims = []
    active_axes = []
    for idx, res in enumerate(usable):
        ax = axes[idx]
        active_axes.append(ax)
        X, Y, Z = grids[idx]
        if X is not None:
            Z_draw = Z
            norm_draw = norm
            if metric == 'lag' and vals:
                Z_draw = prepare_lag_for_display(Z, merged)
            elif metric != 'lag' and vmin is not None and vmax is not None:
                Z_draw, norm_hist = maybe_hist_equalize_for_display(
                    Z, vmin, vmax, enabled=STRENGTH_HIST_EQ_DISPLAY
                )
                if norm_hist is not None:
                    norm_draw = norm_hist
            im = _draw_continuous_map(ax, X, Y, Z_draw, cmap, norm_draw, use_geo)
            ims.append(im)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        _style_geo_axis(ax, X, Y, use_geo, show_ylabel=True)
        annotate_period_label(ax, res['period'])
        if idx < (n - 1):
            ax.set_xlabel('')
            ax.set_xticklabels([])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if metric != 'lag' and STRENGTH_HIST_EQ_DISPLAY and HAS_SKIMAGE:
        extend = 'neither'

    if ims:
        if metric == 'lag':
            cbar = fig.colorbar(
                ims[0], ax=active_axes, orientation='horizontal',
                fraction=0.032, pad=0.02, extend=extend, ticks=lag_ticks
            )
            if lag_mode == 'hist' and lag_ticklabels is not None:
                cbar.ax.set_xticklabels(lag_ticklabels)
            cbar.ax.tick_params(labelsize=max(7, CBAR_TICK_SIZE - 2), pad=1)
        else:
            cbar = fig.colorbar(
                ims[0], ax=active_axes, orientation='horizontal',
                fraction=0.032, pad=0.02, extend=extend
            )
        cbar.set_label(cbar_label, fontsize=CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    metric_name = 'Strength' if metric == 'score' else 'Lag'
    if direction == 'forward':
        panel_tag = f'{driver}_to_{target_label}'
    elif direction == 'reverse':
        panel_tag = f'{target_label}_to_{driver}'
    else:
        panel_tag = f'{driver}_{target_label}_BidirectionalCombined'
    out = os.path.join(output_dir, f'Panel_{metric_name}_{panel_tag}_{n}Periods.png')
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Period Metric Panel: {out}")


def plot_period_summaries(period_results, target_label):
    if not period_results:
        return
    summary_dir = os.path.join(BASE_OUT_DIR, 'causal_analysis_all_5year', 'temporal_summary')
    os.makedirs(summary_dir, exist_ok=True)

    # Organized output folders for cleaner project structure
    ts_dir = os.path.join(summary_dir, 'timeseries')
    panel_dir = os.path.join(summary_dir, 'panels')
    panel_dominant_dir = os.path.join(panel_dir, 'dominant')
    panel_strength_dir = os.path.join(panel_dir, 'strength')
    panel_lag_dir = os.path.join(panel_dir, 'lag')
    for d in [ts_dir, panel_dir, panel_dominant_dir, panel_strength_dir, panel_lag_dir]:
        os.makedirs(d, exist_ok=True)

    plot_driver_timeseries(period_results, target_label, ts_dir)
    plot_multitarget_bidirectional_strength_timeseries(period_results, target_label, ts_dir)
    plot_weighted_interaction_timeseries(period_results, target_label, ts_dir)

    compare = _iter_period_results(period_results, COMPARE_PERIODS)
    plot_driver_type_period_panel(compare, panel_dominant_dir, direction='forward')
    plot_driver_type_period_panel(compare, panel_dominant_dir, direction='reverse')
    plot_driver_type_period_panel(compare, panel_dominant_dir, direction='combined')
    for driver in DRIVER_ORDER:
        for metric in ['score', 'lag']:
            metric_dir = panel_strength_dir if metric == 'score' else panel_lag_dir
            plot_driver_metric_period_panel(compare, target_label, driver, metric, metric_dir, direction='forward')
            plot_driver_metric_period_panel(compare, target_label, driver, metric, metric_dir, direction='reverse')
            plot_driver_metric_period_panel(compare, target_label, driver, metric, metric_dir, direction='combined')


# ================= Core Processing Logic =================
def process_period(period_name):
    print(f"\n{'='*60}")
    print(f"Processing Period: {period_name}")
    print(f"{'='*60}")

    input_dir = os.path.join(BASE_OUT_DIR, 'causal_analysis_all_5year', period_name, 'point_jsons')
    output_dir = os.path.join(BASE_OUT_DIR, 'causal_analysis_all_5year', period_name, 'spatial_analysis_results_fig2')

    if not os.path.exists(input_dir):
        print(f"  [Skip] Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    target_label = clean_target_name(TARGET_VAR)
    target_key = canonical_name(TARGET_VAR)
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

            target_drivers = []
            target_driven = []
            pair_info = {}

            for edge in edges:
                cause_key = canonical_name(edge['cause_name'])
                effect_key = canonical_name(edge['effect_name'])

                cause_is_target = cause_key == target_key
                effect_is_target = effect_key == target_key

                u = target_label if cause_is_target else clean_name(edge['cause_name'])
                v = target_label if effect_is_target else clean_name(edge['effect_name'])
                score = edge['score']
                lag = edge['lag']

                pair_info[f"{u}_to_{v}_score"] = score
                pair_info[f"{u}_to_{v}_lag"] = lag

                if cause_is_target:
                    if 'precipitation' not in v.lower():
                        target_driven.append({'var': v, 'score': score, 'lag': lag})

                if effect_is_target:
                    target_drivers.append({'var': u, 'score': score, 'lag': lag})

            if target_driven:
                best = max(target_driven, key=lambda x: x['score'])
                rec1 = {'dom_driven_type': best['var'], 'dom_driven_score': best['score'], 'dom_driven_lag': best['lag']}
            else:
                rec1 = {'dom_driven_type': 'None', 'dom_driven_score': 0, 'dom_driven_lag': 0}

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
        print('  No valid data, skipping plotting')
        return

    if not ONLY_TEMPORAL_SUMMARY:
        plot_combined_map(
            df,
            type_col='dom_driven_type',
            score_col='dom_driven_score',
            lag_col='dom_driven_lag',
            title_main=f'Map 1: Dominant factors driven by {target_label}',
            filename_base=f'Map1_{target_label}_Driven_Dominant_Combined',
            output_dir=output_dir
        )

        plot_combined_map(
            df,
            type_col='dom_driver_type',
            score_col='dom_driver_score',
            lag_col='dom_driver_lag',
            title_main=f'Map 4: Dominant drivers of {target_label}',
            filename_base=f'Map4_{target_label}_Driver_Dominant_Combined',
            output_dir=output_dir
        )

        cols = [c for c in df.columns if '_to_' in c and '_score' in c]

        for col in cols:
            parts = col.split('_to_')
            src = parts[0]
            dst = parts[1].replace('_score', '')

            if src == target_label:
                if 'precipitation' in dst.lower():
                    continue
                plot_single_map(df, col, f'Map 2: Strength ({target_label} -> {dst})',
                                f'Map2_Strength_{target_label}_to_{dst}', output_dir)
                lag_col = col.replace('_score', '_lag')
                if lag_col in df.columns:
                    plot_single_map(df, lag_col, f'Map 2: Lag ({target_label} -> {dst})',
                                    f'Map2_Lag_{target_label}_to_{dst}', output_dir)

            elif dst == target_label:
                plot_single_map(df, col, f'Map 3: Strength ({src} -> {target_label})',
                                f'Map3_Strength_{src}_to_{target_label}', output_dir)
                lag_col = col.replace('_score', '_lag')
                if lag_col in df.columns:
                    plot_single_map(df, lag_col, f'Map 3: Lag ({src} -> {target_label})',
                                    f'Map3_Lag_{src}_to_{target_label}', output_dir)

    print(f"  [{period_name}] Processing completed. Results saved to: {output_dir}")
    return {
        'period': period_name,
        'target_label': target_label,
        'output_dir': output_dir,
        'df': df
    }


if __name__ == '__main__':
    print('Starting batch plotting for multiple configs and 5-year windows...')
    for cfg in CONFIG_PATHS:
        print(f"\n{'#'*72}")
        print(f"Running config: {cfg}")
        print(f"{'#'*72}")
        if not load_runtime_config(cfg):
            continue

        period_results = []
        for period in PERIODS:
            try:
                result = process_period(period)
                if result is not None:
                    period_results.append(result)
            except Exception as e:
                print(f"Uncaught exception processing period {period} under {cfg}: {e}")
                import traceback
                traceback.print_exc()
        if period_results:
            plot_period_summaries(period_results, period_results[0]['target_label'])
        else:
            print(f"[Skip Summary] No valid period results under {cfg}")
    print('\nAll configs processed.')
