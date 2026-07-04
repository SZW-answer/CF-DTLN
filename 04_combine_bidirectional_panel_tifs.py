import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Run 02new_json绘图_dtln_spatial_map_revised.py first.
CONFIG_PATHS = [
    "./model_config_EVI.json",
    # "./model_config_NDVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]

DRIVER_ORDER = ["Precipitation", "Runoff", "GWS"]
SUMMARY_DIR_NAME = "temporal_summary_new"
RESULT_DIR_CANDIDATES = [
    # "causal_analysis_all_5year_all0624",
    "causal_analysis_all_5year",
    # "causal_analysis_all_5year_noFilm",
    # "causal_analysis_all_5year_ls",
    # "causal_analysis_target_5year",
]
DIRECTIONS = ["forward", "reverse", "combined"]
OUTPUT_DIR_NAME = "combined_tif_panels_new_0625"
OUTPUT_DPI = 1200
GUTTER_RATIO = 0.010
DRAW_COLUMN_TITLES = False
COLUMN_TITLE_RIGHT_ANCHOR_RATIO = 0.90
COLUMN_TITLE_TOP_RATIO = 0.022
COLUMN_TITLE_FONT_RATIO = 0.016

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class TargetPanels:
    config_path: Path
    out_dir: Path
    file_label: str
    display_label: str
    panel_root: Path


def strip_name_suffix(name):
    return (
        str(name)
        .replace("_deseasonalized", "")
        .replace("_sum", "")
        .replace("total_", "")
    )


def canonical_name(name):
    return strip_name_suffix(name).strip().lower().replace(" ", "_")


def clean_target_label(config_path, target_name):
    stem = config_path.stem.upper()
    low = canonical_name(target_name)
    if "EVI" in stem:
        return "EVI"
    if "NDVI" in stem:
        return "NDVI"
    if "SIF" in stem:
        return "SIF"
    if "GPP" in stem or low in {"gosifgpp", "gpp"}:
        return "GPP"
    if low in {"gosif", "go_sif", "sif"}:
        return "SIF"
    return strip_name_suffix(target_name)


def direction_label(direction):
    labels = {
        "forward": "Forward",
        "reverse": "Reverse",
        "combined": "BidirectionalCombined",
    }
    return labels.get(direction, direction)


def resolve_result_root(out_dir):
    override = os.environ.get("DTLN_CAUSAL_INPUT_DIR", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = out_dir / p
        if (p / SUMMARY_DIR_NAME / "panels").exists():
            return p
        print(f"[Warning] DTLN_CAUSAL_INPUT_DIR panel root does not exist: {p / SUMMARY_DIR_NAME / 'panels'}")

    for name in RESULT_DIR_CANDIDATES:
        root = out_dir / name
        panel_root = root / SUMMARY_DIR_NAME / "panels"
        if panel_root.exists():
            return root
    return out_dir / "causal_analysis_target_5year"


def load_target_panels(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    analyze = config["analyze"]
    out_dir = Path(analyze["OUT_DIR"])
    target_name = analyze["TARGET"]
    display_label = clean_target_label(path, target_name)
    result_root = resolve_result_root(out_dir)
    panel_root = result_root / SUMMARY_DIR_NAME / "panels"
    return TargetPanels(path, out_dir, display_label, display_label, panel_root)


def find_times_font(bold=False):
    candidates = [
        Path("/mnt/c/Windows/Fonts/timesbd.ttf" if bold else "/mnt/c/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path(
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        ),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_font(size, bold=False):
    font_path = find_times_font(bold=bold)
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


def open_rgb(path):
    return Image.open(path).convert("RGB")


def resize_to_height(image, target_height):
    if image.height == target_height:
        return image
    width = round(image.width * target_height / image.height)
    return image.resize((width, target_height), Image.Resampling.LANCZOS)


def draw_column_title(draw, box, text, font):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    col_w = right - left
    col_h = bottom - top
    title_right = left + col_w * COLUMN_TITLE_RIGHT_ANCHOR_RATIO
    title_top = top + col_h * COLUMN_TITLE_TOP_RATIO
    x = title_right - text_w
    y = title_top - bbox[1]
    draw.text((x, y), text, fill="black", font=font)


def stitch_columns(items, output_path, dpi=OUTPUT_DPI):
    existing = [(title, path) for title, path in items if path.exists()]
    missing = [(title, path) for title, path in items if not path.exists()]
    if missing:
        print("[Missing]")
        for title, path in missing:
            print(f"  {title}: {path}")
    if not existing:
        print(f"[Skip] No existing panels for {output_path.name}")
        return False

    images = [(title, open_rgb(path)) for title, path in existing]
    target_height = max(image.height for _, image in images)
    images = [(title, resize_to_height(image, target_height)) for title, image in images]

    gutter = max(38, round(target_height * GUTTER_RATIO))
    width = sum(image.width for _, image in images) + gutter * (len(images) - 1)
    height = target_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = make_font(max(64, round(target_height * COLUMN_TITLE_FONT_RATIO)), bold=True)

    x = 0
    for title, image in images:
        canvas.paste(image, (x, 0))
        if DRAW_COLUMN_TITLES:
            draw_column_title(draw, (x, 0, x + image.width, image.height), title, title_font)
        x += image.width + gutter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"[Saved] {output_path}")
    return True


def dominant_panel_path(target, direction):
    return target.panel_root / "dominant" / f"Panel_Dominant_Drivers_{direction_label(direction)}_5Periods.png"


def metric_panel_path(target, metric, driver, direction):
    metric_name = "Lag" if metric == "lag" else "Strength"
    if direction == "forward":
        panel_tag = f"{driver}_to_{target.display_label}_Forward"
    elif direction == "reverse":
        panel_tag = f"{target.display_label}_to_{driver}_Reverse"
    else:
        panel_tag = f"{driver}_{target.display_label}_BidirectionalCombined"
    return target.panel_root / metric / f"Panel_{metric_name}_{panel_tag}_5Periods.png"


def build_outputs(config_paths, output_dir):
    targets = [load_target_panels(path) for path in config_paths]

    print("Targets:")
    for target in targets:
        print(f"  {target.display_label}: {target.panel_root}")

    jobs = []
    for direction in DIRECTIONS:
        tag = direction_label(direction)
        jobs.append(
            (
                f"dominant-{tag}",
                [(target.display_label, dominant_panel_path(target, direction)) for target in targets],
                output_dir / f"Panel_AllTargets_Dominant_Drivers_{tag}_5Periods.tif",
            )
        )

    for target in targets:
        for direction in DIRECTIONS:
            tag = direction_label(direction)
            lag_items = [
                (f"{target.display_label}-{driver}", metric_panel_path(target, "lag", driver, direction))
                for driver in DRIVER_ORDER
            ]
            jobs.append(
                (
                    f"lag-{target.display_label}-{tag}",
                    lag_items,
                    output_dir / f"Panel_Lag_{target.display_label}_Drivers_{tag}_5Periods.tif",
                )
            )

            strength_items = [
                (f"{target.display_label}-{driver}", metric_panel_path(target, "strength", driver, direction))
                for driver in DRIVER_ORDER
            ]
            jobs.append(
                (
                    f"strength-{target.display_label}-{tag}",
                    strength_items,
                    output_dir / f"Panel_Strength_{target.display_label}_Drivers_{tag}_5Periods.tif",
                )
            )

    for _, items, out_path in jobs:
        stitch_columns(items, out_path)


if __name__ == "__main__":
    build_outputs(CONFIG_PATHS, Path(OUTPUT_DIR_NAME))
