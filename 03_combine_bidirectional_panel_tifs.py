import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Run this script after dtln_spatial_map_revised.py has finished all configs.
CONFIG_PATHS = [
    "./model_config_EVI.json",
    # "./model_config_NDVI.json",
    "./model_config_SIF.json",
    "./model_config.json",
]

DRIVER_ORDER = ["Precipitation", "Runoff", "GWS"]
OUTPUT_DPI = 1200
GUTTER_RATIO = 0.010
COLUMN_TITLE_RIGHT_ANCHOR_RATIO = 0.90
COLUMN_TITLE_TOP_RATIO = 0.022
COLUMN_TITLE_FONT_RATIO = 0.016
OUTPUT_DIR_NAME = "combined_tif_panels"


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


def clean_target_file_label(target_name):
    low = canonical_name(target_name)
    if low in {"gosif", "go_sif", "sif"}:
        return "SIF"
    return strip_name_suffix(target_name)


def infer_display_label(config_path, target_name, file_label):
    stem = config_path.stem.upper()
    if "EVI" in stem:
        return "EVI"
    if "NDVI" in stem:
        return "NDVI"
    if "SIF" in stem:
        return "SIF"
    if "GPP" in stem or "gosifgpp" in canonical_name(target_name):
        return "GPP"
    return file_label


def load_target_panels(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    analyze = config["analyze"]
    out_dir = Path(analyze["OUT_DIR"])
    target_name = analyze["TARGET"]
    file_label = clean_target_file_label(target_name)
    display_label = infer_display_label(path, target_name, file_label)
    panel_root = out_dir / "causal_analysis_all_5year" / "temporal_summary" / "panels"
    return TargetPanels(path, out_dir, file_label, display_label, panel_root)


def resolve_panel_path(target, subdir, exact_name):
    path = target.panel_root / subdir / exact_name
    if path.exists():
        return path

    pattern = exact_name.replace(target.file_label, "*")
    matches = sorted((target.panel_root / subdir).glob(pattern))
    if matches:
        return matches[0]
    return path


def dominant_panel_path(target):
    return resolve_panel_path(
        target,
        "dominant",
        "Panel_Dominant_Drivers_BidirectionalCombined_5Periods.png",
    )


def metric_panel_path(target, metric, driver):
    metric_name = "Lag" if metric == "lag" else "Strength"
    return resolve_panel_path(
        target,
        metric,
        f"Panel_{metric_name}_{driver}_{target.file_label}_BidirectionalCombined_5Periods.png",
    )


def find_times_font(bold=False):
    candidates = [
        Path("/mnt/c/Windows/Fonts/timesbd.ttf" if bold else "/mnt/c/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
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
    for _, (title, image) in enumerate(images):
        canvas.paste(image, (x, 0))
        draw_column_title(draw, (x, 0, x + image.width, image.height), title, title_font)
        x += image.width + gutter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"[Saved] {output_path}")
    return True


def build_outputs(config_paths, output_dir, dry_run=False):
    targets = [load_target_panels(path) for path in config_paths]

    print("Targets:")
    for target in targets:
        print(f"  {target.display_label}: {target.panel_root}")

    jobs = []
    jobs.append(
        (
            "dominant",
            [
                (target.display_label, dominant_panel_path(target))
                for target in targets
            ],
            output_dir / "Panel_AllTargets_Dominant_Drivers_BidirectionalCombined_5Periods.tif",
        )
    )

    for target in targets:
        lag_items = [
            (f"{target.display_label}-{driver}", metric_panel_path(target, "lag", driver))
            for driver in DRIVER_ORDER
        ]
        jobs.append(
            (
                f"lag-{target.display_label}",
                lag_items,
                output_dir / f"Panel_Lag_{target.display_label}_Drivers_BidirectionalCombined_5Periods.tif",
            )
        )

        strength_items = [
            (f"{target.display_label}-{driver}", metric_panel_path(target, "strength", driver))
            for driver in DRIVER_ORDER
        ]
        jobs.append(
            (
                f"strength-{target.display_label}",
                strength_items,
                output_dir / f"Panel_Strength_{target.display_label}_Drivers_BidirectionalCombined_5Periods.tif",
            )
        )

    if dry_run:
        print("\nDry run. Planned outputs:")
        for job_name, items, out_path in jobs:
            print(f"  [{job_name}] {out_path}")
            for title, path in items:
                status = "OK" if path.exists() else "MISSING"
                print(f"    {status}: {title} <- {path}")
        return

    for _, items, out_path in jobs:
        stitch_columns(items, out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine bidirectional 5-period panel PNGs into publication TIFF figures."
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_NAME,
        help=f"Directory for combined TIFFs. Default: {OUTPUT_DIR_NAME}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print input/output paths without writing TIFF files.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        default=CONFIG_PATHS,
        help="Optional config JSON paths. Default: the four CONFIG_PATHS in this script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_outputs(args.configs, Path(args.output_dir), dry_run=args.dry_run)
