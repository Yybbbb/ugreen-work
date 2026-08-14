#!/usr/bin/env python3
"""
杂色检测器 —— 主接口。

整合:
  1. histogram_peak  — a*b* 直方图峰值检测 + L 通道旁路
  2. texture_rle      — RLE 纹理分析（条纹/方格判定）

输入: RGB 图像 + 分割 mask + 区域 ID
输出: 杂色判定结果 + 置信度降权系数
"""

import numpy as np
from histogram_peak import analyze_region_color, N_BINS, SIGMA, PEAK_THRESHOLD, PEAK_MERGE_DIST, L_RANGE_THRESHOLD
from texture_rle import analyze_texture, STRIPE_SWITCH_THRESH, RLE_SIZE

# ── 降权系数 ──
PENALTY_PURE = 1.0          # 纯色，不降权
PENALTY_MULTI_SOLID = 0.5   # 多色块状拼接，大幅降权
PENALTY_STRIPE = 0.7        # 规则条纹/方格，适度降权（主色可能仍准）


def detect(image_rgb, mask, region_id, region_name='region',
           n_bins=N_BINS, sigma=SIGMA,
           peak_threshold=PEAK_THRESHOLD, merge_dist=PEAK_MERGE_DIST,
           l_range_threshold=L_RANGE_THRESHOLD,
           stripe_thresh=STRIPE_SWITCH_THRESH, rle_size=RLE_SIZE):
    """
    主入口: 检测图像指定区域的杂色情况。

    参数:
        image_rgb:    (H, W, 3) uint8 numpy array
        mask:         (H, W) int numpy array — 分割mask, 0=bg
        region_id:    int — 要检测的区域ID
        region_name:  str — 区域名称（用于输出标识）
        其余参数:     见各子模块

    返回:
        dict: {
            'region': str,
            'n_peaks': int,
            'purity': float,
            'is_multicolor': bool,
            'pattern': str,            # 'solid'|'vertical_stripe'|'horizontal_stripe'|'checkered'|'unknown'
            'confidence_penalty': float,
            'detail': str,             # 人类可读的判定理由
            # 子结果
            'ab_result': dict,
            'texture_result': dict,
        }
    """
    # 步骤1: 颜色多中心检测
    ab_result = analyze_region_color(
        image_rgb, mask, region_id,
        n_bins=n_bins, sigma=sigma,
        peak_threshold=peak_threshold, merge_dist=merge_dist,
        l_range_threshold=l_range_threshold,
    )

    is_multicolor = ab_result['is_multicolor']
    n_peaks = ab_result['n_peaks']
    purity = ab_result['purity']

    # 步骤2: 纹理分析（仅对判定为多色的区域，或 L 异常区域）
    texture_result = None
    pattern = 'solid'

    if is_multicolor or ab_result.get('l_anomaly'):
        texture_result = analyze_texture(
            image_rgb, mask, region_id,
            ab_result['peaks'],
            target_size=rle_size,
            stripe_thresh=stripe_thresh,
        )
        if texture_result and texture_result.get('error') is None:
            pattern = texture_result['pattern']
    elif not is_multicolor:
        # 纯色，无需纹理分析
        texture_result = {'pattern': 'solid', 'row_switch_ratio': 0.0, 'col_switch_ratio': 0.0,
                          'label_map': None, 'skipped': True}

    # 步骤3: 判定降权系数
    detail_parts = []

    if not is_multicolor:
        penalty = PENALTY_PURE
        if n_peaks == 1:
            detail_parts.append(f"单色峰, 纯度={purity:.2f}")
        else:
            detail_parts.append(f"{n_peaks}峰但纯度={purity:.2f}>0.80, 视为纯色")
    else:
        if ab_result.get('l_anomaly') and n_peaks == 1:
            detail_parts.append(f"a*b*单峰 + L通道双峰(极差={ab_result['l_info'].get('l_range',0):.0f}) → 明暗条纹")
        elif n_peaks >= 2:
            detail_parts.append(f"a*b* {n_peaks}个颜色峰, 纯度={purity:.2f}")

        if pattern in ('vertical_stripe', 'horizontal_stripe', 'checkered'):
            penalty = PENALTY_STRIPE
            detail_parts.append(f"纹理={pattern}, 降权系数={penalty}")
        else:
            penalty = PENALTY_MULTI_SOLID
            detail_parts.append(f"纹理={pattern}, 降权系数={penalty}")

    return {
        'region': region_name,
        'n_peaks': n_peaks,
        'purity': round(float(purity), 4),
        'is_multicolor': is_multicolor,
        'pattern': pattern,
        'confidence_penalty': penalty,
        'detail': '; '.join(detail_parts),
        'ab_result': {k: v for k, v in ab_result.items() if k != 'ab_histogram'},
        'texture_result': {k: v for k, v in (texture_result or {}).items() if k != 'label_map'},
    }


def detect_upper_lower(image_rgb, mask, **kwargs):
    """
    同时检测 upper (region_id=1) 和 lower (region_id=2)。

    返回:
        dict: {'upper': {...}, 'lower': {...}}
    """
    upper_result = detect(image_rgb, mask, region_id=1, region_name='upper', **kwargs)
    lower_result = detect(image_rgb, mask, region_id=2, region_name='lower', **kwargs)
    return {'upper': upper_result, 'lower': lower_result}


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == '__main__':
    import sys, json
    from PIL import Image

    if len(sys.argv) < 2:
        print("用法: python multicolor_detector.py <image_path> [mask_npy_path]")
        print("  mask 默认 region_id=1(upper), region_id=2(lower), 0=bg")
        sys.exit(1)

    img = np.array(Image.open(sys.argv[1]).convert('RGB'))

    if len(sys.argv) >= 3:
        mask = np.load(sys.argv[2])
    else:
        # 默认: 全图当作 upper
        mask = np.ones(img.shape[:2], dtype=np.int32)

    result = detect_upper_lower(img, mask)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
