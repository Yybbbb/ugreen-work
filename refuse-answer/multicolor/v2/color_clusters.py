import math
from collections import defaultdict

from config import AB_BIN_WIDTH, AB_MERGE_DIST, AB_PEAK_THRESHOLD, EFFECTIVE_CLUSTER_MIN_RATIO


def find_ab_centers(pixels_lab):
    if not pixels_lab:
        return []

    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for lab in pixels_lab:
        a = max(-128.0, min(127.0, lab[1]))
        b = max(-128.0, min(127.0, lab[2]))
        key = (int((a + 128.0) // AB_BIN_WIDTH), int((b + 128.0) // AB_BIN_WIDTH))
        bins[key][0] += 1
        bins[key][1] += a
        bins[key][2] += b

    candidates = []
    for count, sum_a, sum_b in bins.values():
        candidates.append([sum_a / count, sum_b / count, count])
    candidates.sort(key=lambda item: item[2], reverse=True)

    if not candidates:
        return []

    min_count = max(1.0, candidates[0][2] * AB_PEAK_THRESHOLD)
    candidates = [candidate for candidate in candidates if candidate[2] >= min_count]

    merged = []
    for a, b, count in candidates:
        matched = None
        for kept in merged:
            if _dist_ab((a, b), (kept[0], kept[1])) < AB_MERGE_DIST:
                matched = kept
                break
        if matched is None:
            merged.append([a, b, count])
        else:
            total = matched[2] + count
            matched[0] = (matched[0] * matched[2] + a * count) / total
            matched[1] = (matched[1] * matched[2] + b * count) / total
            matched[2] = total

    merged.sort(key=lambda item: item[2], reverse=True)
    return [[item[0], item[1]] for item in merged]


def compute_cluster_stats_from_centers(pixels_lab, centers_ab):
    if not pixels_lab or not centers_ab:
        return _empty_stats()

    counts = [0 for _ in centers_ab]
    labels = []
    for lab in pixels_lab:
        label = _nearest_center(lab, centers_ab)
        labels.append(label)
        counts[label] += 1

    total = float(len(pixels_lab))
    ratios = [count / total for count in counts]
    order = sorted(range(len(ratios)), key=lambda idx: ratios[idx], reverse=True)

    main_ratio = ratios[order[0]] if order else 0.0
    second_ratio = ratios[order[1]] if len(order) > 1 else 0.0
    minor_total = max(0.0, 1.0 - main_ratio)
    n_effective = sum(1 for ratio in ratios if ratio >= EFFECTIVE_CLUSTER_MIN_RATIO)

    clusters = [
        {
            "ab": [round(centers_ab[idx][0], 4), round(centers_ab[idx][1], 4)],
            "ratio": round(ratios[idx], 6),
        }
        for idx in order
    ]

    return {
        "main_ratio": round(main_ratio, 6),
        "second_ratio": round(second_ratio, 6),
        "minor_total_ratio": round(minor_total, 6),
        "n_effective_colors": n_effective,
        "clusters": clusters,
        "labels": labels,
    }


def _nearest_center(lab, centers_ab):
    best_idx = 0
    best_dist = None
    for idx, center in enumerate(centers_ab):
        dist = (lab[1] - center[0]) ** 2 + (lab[2] - center[1]) ** 2
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def _dist_ab(left, right):
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _empty_stats():
    return {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
        "labels": [],
    }
