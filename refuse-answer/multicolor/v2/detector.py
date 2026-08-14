from color_clusters import compute_cluster_stats_from_centers, find_ab_centers
from color_space import rgb_to_lab
from config import MIN_ROI_PIXELS
from neutral_l import analyze_neutral_l
from purity_rules import decide_purity


def detect_region(image_rgb, mask, region_id, region_name="region"):
    pixels_rgb = _extract_roi_pixels(image_rgb, mask, region_id)
    n_pixels = len(pixels_rgb)
    if n_pixels < MIN_ROI_PIXELS:
        decision = {
            "is_pure": False,
            "is_multicolor": True,
            "decision_reason": "too_few_pixels",
        }
        return _format_result(region_name, n_pixels, decision, None, None)

    pixels_lab = rgb_to_lab(pixels_rgb)
    centers_ab = find_ab_centers(pixels_lab)
    stats = compute_cluster_stats_from_centers(pixels_lab, centers_ab)
    neutral = analyze_neutral_l(pixels_lab)
    decision = decide_purity(
        main_ratio=stats["main_ratio"],
        second_ratio=stats["second_ratio"],
        minor_total_ratio=stats["minor_total_ratio"],
        n_effective_colors=stats["n_effective_colors"],
        neutral_l_multipeak=neutral["neutral_l_multipeak"],
    )
    return _format_result(region_name, n_pixels, decision, stats, neutral)


def detect_upper_lower(image_rgb, mask):
    return {
        "upper": detect_region(image_rgb, mask, 1, "upper"),
        "lower": detect_region(image_rgb, mask, 2, "lower"),
    }


def _extract_roi_pixels(image_rgb, mask, region_id):
    pixels = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value == region_id:
                pixels.append(image_rgb[y][x])
    return pixels


def _format_result(region_name, n_pixels, decision, stats, neutral):
    stats = stats or {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
    }
    neutral = neutral or {"neutral_l_multipeak": False}
    return {
        "region": region_name,
        "n_pixels": n_pixels,
        "is_pure": decision["is_pure"],
        "is_multicolor": decision["is_multicolor"],
        "decision_reason": decision["decision_reason"],
        "main_ratio": stats["main_ratio"],
        "second_ratio": stats["second_ratio"],
        "minor_total_ratio": stats["minor_total_ratio"],
        "n_effective_colors": stats["n_effective_colors"],
        "clusters": stats["clusters"],
        "neutral_l": neutral,
    }
