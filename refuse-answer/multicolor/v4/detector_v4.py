import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent / "v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from color_space import rgb_to_lab
from adaptive_lab import cluster_lab_pixels
from config_v4 import MAX_CLUSTER_PIXELS, MIN_ROI_PIXELS, SAMPLE_SEED
from purity_rules_v4 import decide_purity
from shadow_merge import merge_shadow_clusters


def detect_region(image_rgb, mask, region_id, region_name="region"):
    pixels_rgb = _extract_roi_pixels(image_rgb, mask, region_id)
    n_pixels = len(pixels_rgb)
    if n_pixels < MIN_ROI_PIXELS:
        decision = {
            "is_pure": False,
            "is_multicolor": True,
            "decision_reason": "too_few_pixels",
        }
        return _format_result(region_name, n_pixels, decision, None)

    sampled_pixels_rgb = _sample_pixels(pixels_rgb, MAX_CLUSTER_PIXELS)
    pixels_lab = rgb_to_lab(sampled_pixels_rgb)
    raw_stats = cluster_lab_pixels(pixels_lab)
    stats = merge_shadow_clusters(raw_stats)
    decision = decide_purity(stats)
    return _format_result(region_name, n_pixels, decision, stats, len(sampled_pixels_rgb), raw_stats)


def _extract_roi_pixels(image_rgb, mask, region_id):
    pixels = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value == region_id:
                pixels.append(image_rgb[y][x])
    return pixels


def _sample_pixels(pixels_rgb, max_pixels):
    if len(pixels_rgb) <= max_pixels:
        return pixels_rgb
    rng = Random(SAMPLE_SEED)
    indices = rng.sample(range(len(pixels_rgb)), max_pixels)
    return [pixels_rgb[idx] for idx in indices]


def _format_result(region_name, n_pixels, decision, stats, sampled_pixels=0, raw_stats=None):
    stats = stats or {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
        "shadow_merges": 0,
    }
    raw_stats = raw_stats or {}
    return {
        "region": region_name,
        "n_pixels": n_pixels,
        "sampled_pixels": sampled_pixels,
        "is_pure": decision["is_pure"],
        "is_multicolor": decision["is_multicolor"],
        "decision_reason": decision["decision_reason"],
        "main_ratio": stats["main_ratio"],
        "second_ratio": stats["second_ratio"],
        "minor_total_ratio": stats["minor_total_ratio"],
        "n_effective_colors": stats["n_effective_colors"],
        "shadow_merges": stats.get("shadow_merges", 0),
        "raw_main_ratio": raw_stats.get("main_ratio"),
        "raw_second_ratio": raw_stats.get("second_ratio"),
        "raw_minor_total_ratio": raw_stats.get("minor_total_ratio"),
        "raw_n_effective_colors": raw_stats.get("n_effective_colors"),
        "clusters": stats["clusters"],
        "raw_clusters": raw_stats.get("clusters", []),
    }
