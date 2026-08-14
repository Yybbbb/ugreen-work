import math

from config_v3 import (
    CHROMATIC_AB_WEIGHT,
    CHROMATIC_L_WEIGHT,
    CLUSTER_RADIUS,
    EFFECTIVE_CLUSTER_MIN_RATIO,
    NEUTRAL_AB_WEIGHT,
    NEUTRAL_CHROMA_HIGH,
    NEUTRAL_CHROMA_LOW,
    NEUTRAL_L_WEIGHT,
)


def adaptive_lab_distance(left, right):
    dl = left[0] - right[0]
    da = left[1] - right[1]
    db = left[2] - right[2]
    chroma = (_chroma(left) + _chroma(right)) / 2.0
    neutral_weight = _neutral_weight(chroma)
    w_l = CHROMATIC_L_WEIGHT + neutral_weight * (NEUTRAL_L_WEIGHT - CHROMATIC_L_WEIGHT)
    w_ab = CHROMATIC_AB_WEIGHT + neutral_weight * (NEUTRAL_AB_WEIGHT - CHROMATIC_AB_WEIGHT)
    return math.sqrt(w_l * dl * dl + w_ab * (da * da + db * db))


def cluster_lab_pixels(pixels_lab, radius=CLUSTER_RADIUS):
    if not pixels_lab:
        return _empty_stats()

    centers = []
    counts = []
    for lab in pixels_lab:
        idx = _nearest_center(lab, centers)
        if idx is None or adaptive_lab_distance(lab, centers[idx]) > radius:
            centers.append([lab[0], lab[1], lab[2]])
            counts.append(1)
        else:
            count = counts[idx]
            new_count = count + 1
            centers[idx] = [
                (centers[idx][0] * count + lab[0]) / new_count,
                (centers[idx][1] * count + lab[1]) / new_count,
                (centers[idx][2] * count + lab[2]) / new_count,
            ]
            counts[idx] = new_count

    centers, counts = _merge_close_centers(centers, counts, radius)
    labels, pixel_counts = _assign_pixels(pixels_lab, centers)
    return _stats_from_counts(centers, pixel_counts, labels, len(pixels_lab))


def _merge_close_centers(centers, counts, radius):
    changed = True
    while changed:
        changed = False
        for i in range(len(centers)):
            if changed:
                break
            for j in range(i + 1, len(centers)):
                if adaptive_lab_distance(centers[i], centers[j]) <= radius:
                    total = counts[i] + counts[j]
                    centers[i] = [
                        (centers[i][0] * counts[i] + centers[j][0] * counts[j]) / total,
                        (centers[i][1] * counts[i] + centers[j][1] * counts[j]) / total,
                        (centers[i][2] * counts[i] + centers[j][2] * counts[j]) / total,
                    ]
                    counts[i] = total
                    del centers[j]
                    del counts[j]
                    changed = True
                    break
    return centers, counts


def _assign_pixels(pixels_lab, centers):
    counts = [0 for _ in centers]
    labels = []
    for lab in pixels_lab:
        idx = _nearest_center(lab, centers)
        labels.append(idx)
        counts[idx] += 1
    return labels, counts


def _stats_from_counts(centers, counts, labels, total_pixels):
    if not centers or total_pixels == 0:
        return _empty_stats()
    ratios = [count / float(total_pixels) for count in counts]
    order = sorted(range(len(ratios)), key=lambda idx: ratios[idx], reverse=True)
    main_ratio = ratios[order[0]]
    second_ratio = ratios[order[1]] if len(order) > 1 else 0.0
    minor_total = max(0.0, 1.0 - main_ratio)
    n_effective = sum(1 for ratio in ratios if ratio >= EFFECTIVE_CLUSTER_MIN_RATIO)
    clusters = [
        {
            "lab": [round(centers[idx][0], 4), round(centers[idx][1], 4), round(centers[idx][2], 4)],
            "ratio": round(ratios[idx], 6),
            "chroma": round(_chroma(centers[idx]), 4),
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


def _nearest_center(lab, centers):
    if not centers:
        return None
    best_idx = 0
    best_dist = adaptive_lab_distance(lab, centers[0])
    for idx in range(1, len(centers)):
        dist = adaptive_lab_distance(lab, centers[idx])
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def _chroma(lab):
    return math.sqrt(lab[1] * lab[1] + lab[2] * lab[2])


def _neutral_weight(chroma):
    if chroma <= NEUTRAL_CHROMA_LOW:
        return 1.0
    if chroma >= NEUTRAL_CHROMA_HIGH:
        return 0.0
    return (NEUTRAL_CHROMA_HIGH - chroma) / (NEUTRAL_CHROMA_HIGH - NEUTRAL_CHROMA_LOW)


def _empty_stats():
    return {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
        "labels": [],
    }
