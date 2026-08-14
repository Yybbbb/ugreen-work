from config_v4 import (
    MAIN_RATIO_PURE_MIN,
    MINOR_TOTAL_PURE_MAX,
    MULTI_SMALL_COLORS_MIN_COUNT,
    MULTI_SMALL_COLORS_MINOR_TOTAL_MIN,
    SECOND_RATIO_PURE_MAX,
)


def decide_purity(stats):
    if stats["second_ratio"] >= SECOND_RATIO_PURE_MAX:
        return _result(False, "second_ratio>=0.10")
    if stats["minor_total_ratio"] >= MINOR_TOTAL_PURE_MAX:
        return _result(False, "minor_total_ratio>=0.20")
    if (
        stats["n_effective_colors"] >= MULTI_SMALL_COLORS_MIN_COUNT
        and stats["minor_total_ratio"] >= MULTI_SMALL_COLORS_MINOR_TOTAL_MIN
    ):
        return _result(False, "n_effective_colors>=3 and minor_total_ratio>=0.15")
    if stats["main_ratio"] >= MAIN_RATIO_PURE_MIN:
        return _result(True, "pure")
    return _result(False, "main_ratio<0.80")


def _result(is_pure, reason):
    return {
        "is_pure": bool(is_pure),
        "is_multicolor": not bool(is_pure),
        "decision_reason": reason,
    }
