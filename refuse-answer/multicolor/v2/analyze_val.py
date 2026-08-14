#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from detector import detect_region
from image_io import load_rgb_image
from mask_io import DEFAULT_ANNOTATIONS_ROOT, DEFAULT_DATA_ROOT, load_upper_mask_for_image


DEFAULT_IMAGE_DIR = HERE.parent / "val_multicolor" / "upper_multicolor"
DEFAULT_OUTPUT = HERE / "upper_multicolor_results.json"


def run(image_dir=DEFAULT_IMAGE_DIR, output=DEFAULT_OUTPUT, annotations_root=DEFAULT_ANNOTATIONS_ROOT, data_root=DEFAULT_DATA_ROOT, limit=None):
    image_dir = Path(image_dir)
    output = Path(output)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    results = []
    stats = {
        "total_images": len(image_paths),
        "processed": 0,
        "errors": 0,
        "detected_upper_pure": 0,
        "detected_upper_multicolor": 0,
    }

    start = time.time()
    for image_path in image_paths:
        try:
            image_rgb, width, height = load_rgb_image(image_path)
            mask_info = load_upper_mask_for_image(
                image_path,
                annotations_root=annotations_root,
                data_root=data_root,
            )
            if mask_info["width"] != width or mask_info["height"] != height:
                mask_info["mask"] = resize_mask_nearest(mask_info["mask"], width, height)
                mask_info["width"] = width
                mask_info["height"] = height
            detection = detect_region(image_rgb, mask_info["mask"], 1, "upper")
            stats["processed"] += 1
            if detection["is_pure"]:
                stats["detected_upper_pure"] += 1
            else:
                stats["detected_upper_multicolor"] += 1
            results.append({
                "id": image_path.name,
                "image_path": str(image_path),
                "annotation_path": mask_info["annotation_path"],
                "label_path": mask_info["label_path"],
                "search_key": mask_info["search_key"],
                "lip_ids": mask_info["lip_ids"],
                "detection": detection,
            })
        except Exception as exc:
            stats["errors"] += 1
            results.append({
                "id": image_path.name,
                "image_path": str(image_path),
                "error": str(exc),
            })

    stats["elapsed_seconds"] = round(time.time() - start, 3)
    report = {
        "config": _config_dict(),
        "stats": stats,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _config_dict():
    return {
        "min_roi_pixels": config.MIN_ROI_PIXELS,
        "ab_bin_width": config.AB_BIN_WIDTH,
        "ab_peak_threshold": config.AB_PEAK_THRESHOLD,
        "ab_merge_dist": config.AB_MERGE_DIST,
        "effective_cluster_min_ratio": config.EFFECTIVE_CLUSTER_MIN_RATIO,
        "main_ratio_pure_min": config.MAIN_RATIO_PURE_MIN,
        "second_ratio_pure_max": config.SECOND_RATIO_PURE_MAX,
        "minor_total_pure_max": config.MINOR_TOTAL_PURE_MAX,
        "multi_small_colors_min_count": config.MULTI_SMALL_COLORS_MIN_COUNT,
        "multi_small_colors_minor_total_min": config.MULTI_SMALL_COLORS_MINOR_TOTAL_MIN,
        "neutral_chroma_max": config.NEUTRAL_CHROMA_MAX,
        "neutral_region_ratio_min": config.NEUTRAL_REGION_RATIO_MIN,
        "l_peak_min_ratio": config.L_PEAK_MIN_RATIO,
        "l_peak_min_delta": config.L_PEAK_MIN_DELTA,
        "l_minor_total_min": config.L_MINOR_TOTAL_MIN,
    }


def resize_mask_nearest(mask, target_width, target_height):
    src_height = len(mask)
    src_width = len(mask[0]) if src_height else 0
    if src_width == 0 or src_height == 0:
        return [[0 for _ in range(target_width)] for _ in range(target_height)]

    resized = []
    for y in range(target_height):
        src_y = min(src_height - 1, int(y * src_height / target_height))
        row = []
        for x in range(target_width):
            src_x = min(src_width - 1, int(x * src_width / target_width))
            row.append(mask[src_y][src_x])
        resized.append(row)
    return resized


def main():
    parser = argparse.ArgumentParser(description="Run V2 pure-color gate on upper_multicolor images using upper masks only.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--annotations-root", default=str(DEFAULT_ANNOTATIONS_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run(
        image_dir=args.image_dir,
        output=args.output,
        annotations_root=args.annotations_root,
        data_root=args.data_root,
        limit=args.limit,
    )
    stats = report["stats"]
    print(
        f"processed={stats['processed']}/{stats['total_images']} "
        f"errors={stats['errors']} pure={stats['detected_upper_pure']} "
        f"multicolor={stats['detected_upper_multicolor']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
