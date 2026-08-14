from color_space import chroma_ab
from config import (
    L_HIST_BINS,
    L_MINOR_TOTAL_MIN,
    L_PEAK_MIN_DELTA,
    L_PEAK_MIN_RATIO,
    NEUTRAL_CHROMA_MAX,
    NEUTRAL_REGION_RATIO_MIN,
)


def analyze_neutral_l(pixels_lab):
    if not pixels_lab:
        return _empty(skipped=True)

    chroma_values = chroma_ab(pixels_lab)
    neutral_ratio = sum(1 for value in chroma_values if value <= NEUTRAL_CHROMA_MAX) / float(len(pixels_lab))
    if neutral_ratio < NEUTRAL_REGION_RATIO_MIN:
        result = _empty(skipped=True)
        result["neutral_ratio"] = round(neutral_ratio, 6)
        return result

    l_values = [max(0.0, min(100.0, lab[0])) for lab in pixels_lab]
    hist = [0 for _ in range(L_HIST_BINS)]
    for value in l_values:
        idx = min(L_HIST_BINS - 1, int(value / 100.0 * L_HIST_BINS))
        hist[idx] += 1

    peaks = _find_hist_peaks(hist)
    if not peaks:
        max_idx = max(range(len(hist)), key=lambda idx: hist[idx])
        peaks = [max_idx]

    peak_items = []
    for idx in peaks:
        l_center = (idx + 0.5) * (100.0 / L_HIST_BINS)
        peak_items.append([l_center, hist[idx]])
    peak_items.sort(key=lambda item: item[1], reverse=True)

    total_peak_count = sum(item[1] for item in peak_items) or 1
    ratios = [[item[0], item[1] / float(total_peak_count)] for item in peak_items]

    if len(ratios) < 2:
        return {
            "neutral_l_multipeak": False,
            "skipped": False,
            "neutral_ratio": round(neutral_ratio, 6),
            "l_peaks": _round_peaks(ratios),
            "l_second_ratio": 0.0,
            "l_minor_total_ratio": 0.0,
            "l_peak_delta": 0.0,
        }

    l_delta = abs(ratios[0][0] - ratios[1][0])
    second_ratio = ratios[1][1]
    minor_total = sum(item[1] for item in ratios[1:])
    detected = (
        l_delta >= L_PEAK_MIN_DELTA
        and (second_ratio >= L_PEAK_MIN_RATIO or minor_total >= L_MINOR_TOTAL_MIN)
    )

    return {
        "neutral_l_multipeak": bool(detected),
        "skipped": False,
        "neutral_ratio": round(neutral_ratio, 6),
        "l_peaks": _round_peaks(ratios),
        "l_second_ratio": round(second_ratio, 6),
        "l_minor_total_ratio": round(minor_total, 6),
        "l_peak_delta": round(l_delta, 6),
    }


def _find_hist_peaks(hist):
    peaks = []
    for idx, value in enumerate(hist):
        if value <= 0:
            continue
        left = hist[idx - 1] if idx > 0 else -1
        right = hist[idx + 1] if idx + 1 < len(hist) else -1
        if value >= left and value >= right:
            peaks.append(idx)
    return peaks


def _round_peaks(peaks):
    return [[round(item[0], 4), round(item[1], 6)] for item in peaks]


def _empty(skipped):
    return {
        "neutral_l_multipeak": False,
        "skipped": skipped,
        "neutral_ratio": 0.0,
        "l_peaks": [],
        "l_second_ratio": 0.0,
        "l_minor_total_ratio": 0.0,
        "l_peak_delta": 0.0,
    }
