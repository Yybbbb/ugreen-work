import math

from config_v4 import (
    EFFECTIVE_CLUSTER_MIN_RATIO,
    SHADOW_MERGE_MAX_AB_DELTA,
    SHADOW_MERGE_MAX_HUE_DEGREES,
    SHADOW_MERGE_MIN_CHROMA,
)


def merge_shadow_clusters(stats):
    clusters = [
        {
            "lab": list(cluster["lab"]),
            "ratio": float(cluster["ratio"]),
            "chroma": float(cluster.get("chroma", _chroma(cluster["lab"]))),
        }
        for cluster in stats.get("clusters", [])
    ]
    if len(clusters) < 2:
        merged = _stats_from_clusters(clusters)
        merged["shadow_merges"] = 0
        return merged

    shadow_merges = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if changed:
                break
            for j in range(i + 1, len(clusters)):
                if _is_shadow_pair(clusters[i]["lab"], clusters[j]["lab"]):
                    clusters[i] = _merge_pair(clusters[i], clusters[j])
                    del clusters[j]
                    shadow_merges += 1
                    changed = True
                    break

    merged = _stats_from_clusters(clusters)
    merged["shadow_merges"] = shadow_merges
    return merged


def _is_shadow_pair(left, right):
    left_chroma = _chroma(left)
    right_chroma = _chroma(right)
    if left_chroma < SHADOW_MERGE_MIN_CHROMA or right_chroma < SHADOW_MERGE_MIN_CHROMA:
        return False

    if _ab_delta(left, right) > SHADOW_MERGE_MAX_AB_DELTA:
        return False

    return _hue_delta_degrees(left, right) <= SHADOW_MERGE_MAX_HUE_DEGREES


def _merge_pair(left, right):
    total = left["ratio"] + right["ratio"]
    if total <= 0:
        return {"lab": [0.0, 0.0, 0.0], "ratio": 0.0, "chroma": 0.0}
    lab = [
        (left["lab"][idx] * left["ratio"] + right["lab"][idx] * right["ratio"]) / total
        for idx in range(3)
    ]
    return {
        "lab": lab,
        "ratio": total,
        "chroma": _chroma(lab),
    }


def _stats_from_clusters(clusters):
    if not clusters:
        return {
            "main_ratio": 0.0,
            "second_ratio": 0.0,
            "minor_total_ratio": 1.0,
            "n_effective_colors": 0,
            "clusters": [],
        }

    total = sum(cluster["ratio"] for cluster in clusters)
    normalized = []
    for cluster in clusters:
        ratio = cluster["ratio"] / total if total else 0.0
        lab = cluster["lab"]
        normalized.append(
            {
                "lab": [round(lab[0], 4), round(lab[1], 4), round(lab[2], 4)],
                "ratio": round(ratio, 6),
                "chroma": round(_chroma(lab), 4),
            }
        )
    normalized.sort(key=lambda cluster: cluster["ratio"], reverse=True)

    main_ratio = normalized[0]["ratio"]
    second_ratio = normalized[1]["ratio"] if len(normalized) > 1 else 0.0
    return {
        "main_ratio": round(main_ratio, 6),
        "second_ratio": round(second_ratio, 6),
        "minor_total_ratio": round(max(0.0, 1.0 - main_ratio), 6),
        "n_effective_colors": sum(1 for cluster in normalized if cluster["ratio"] >= EFFECTIVE_CLUSTER_MIN_RATIO),
        "clusters": normalized,
    }


def _ab_delta(left, right):
    da = left[1] - right[1]
    db = left[2] - right[2]
    return math.sqrt(da * da + db * db)


def _hue_delta_degrees(left, right):
    left_angle = math.degrees(math.atan2(left[2], left[1]))
    right_angle = math.degrees(math.atan2(right[2], right[1]))
    delta = abs(left_angle - right_angle) % 360.0
    return min(delta, 360.0 - delta)


def _chroma(lab):
    return math.sqrt(lab[1] * lab[1] + lab[2] * lab[2])
