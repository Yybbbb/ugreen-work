#!/usr/bin/env python3
"""
RLE 游程编码纹理分析 —— 判断多色区域的空间排列模式。

将 ROI 像素分配到最近邻的颜色峰值后，逐行逐列做游程编码，
通过颜色切换频率判断:
  - solid (纯色/块状多色)
  - vertical_stripe (竖条纹)
  - horizontal_stripe (横条纹)
  - checkered (方格)
"""

import numpy as np

# ── 参数 ──
RLE_SIZE = 64               # 纹理分析统一缩放尺寸
STRIPE_SWITCH_THRESH = 0.05  # 切换比下限: 至少5%的列/行有切换


def quantize_pixels_to_peaks(pixels_lab, peaks_ab):
    """
    将每个像素分配到最近的颜色峰值。

    参数:
        pixels_lab: (N, 3) Lab 像素
        peaks_ab:   [(a_val, b_val, ratio), ...] 颜色峰列表

    返回:
        labels: (N,) int — 每个像素所属的峰索引
    """
    if len(peaks_ab) <= 1:
        return np.zeros(len(pixels_lab), dtype=np.int32)

    a = pixels_lab[:, 1]
    b = pixels_lab[:, 2]

    peak_centers = np.array([(p[0], p[1]) for p in peaks_ab], dtype=np.float32)

    # 计算每个像素到每个峰中心的 a*b* 距离
    labels = np.zeros(len(pixels_lab), dtype=np.int32)
    min_dist = np.full(len(pixels_lab), np.inf)

    for k, (pa, pb) in enumerate(peak_centers):
        d = np.sqrt((a - pa)**2 + (b - pb)**2)
        closer = d < min_dist
        labels[closer] = k
        min_dist[closer] = d[closer]

    return labels


def build_label_map(image_rgb, mask, region_id, peaks_ab, target_size=RLE_SIZE):
    """
    构建缩放到 target_size 的颜色标签图。

    返回:
        label_map: (target_size, target_size) int — 颜色标签图
        None: 如果 ROI 太小无法分析
    """
    from histogram_peak import rgb_to_lab

    roi = mask == region_id
    if np.sum(roi) < 50:
        return None

    rgb_pixels = image_rgb[roi]
    lab_pixels = rgb_to_lab(rgb_pixels)
    labels = quantize_pixels_to_peaks(lab_pixels, peaks_ab)

    # 重建到原图尺寸
    h, w = image_rgb.shape[:2]
    label_map_full = np.full((h, w), -1, dtype=np.int32)
    ys, xs = np.where(roi)
    label_map_full[ys, xs] = labels

    # 裁剪到 ROI 包围盒，再缩放到固定尺寸
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    if y_max - y_min < 3 or x_max - x_min < 3:
        return None

    crop = label_map_full[y_min:y_max+1, x_min:x_max+1]

    # 缩放: 简单降采样
    from PIL import Image
    crop_img = Image.fromarray(crop.astype(np.int8), mode='L')
    crop_img = crop_img.resize((target_size, target_size), Image.NEAREST)

    return np.array(crop_img, dtype=np.int32)


def rle_row_scan(label_map):
    """
    逐行 RLE 扫描，统计每行颜色切换次数。

    返回:
        switches_per_row: (H,) array — 每行的颜色切换次数
        ratio: float — 平均切换次数 / 宽度
    """
    h, w = label_map.shape
    switches = np.zeros(h, dtype=np.float32)

    for y in range(h):
        row = label_map[y]
        # 跳过背景 (-1)
        valid = row[row >= 0]
        if len(valid) < 3:
            continue
        # 游程切换: 相邻像素颜色不同 → 计数+1
        switches[y] = np.sum(valid[1:] != valid[:-1])

    mean_switches = np.mean(switches)
    return switches, mean_switches / w


def rle_col_scan(label_map):
    """逐列 RLE 扫描，同 rle_row_scan。"""
    return rle_row_scan(label_map.T)


def classify_pattern(row_switch_ratio, col_switch_ratio,
                     stripe_thresh=STRIPE_SWITCH_THRESH):
    """
    根据行列切换比判定纹理类型。

    判定逻辑:
      - 双低: 纯色 / 块状多色（颜色成片，无交替）
      - 行高列低: 竖条纹（水平方向颜色频换，垂直不变）
      - 行低列高: 横条纹
      - 双高: 方格
    """
    r_high = row_switch_ratio >= stripe_thresh
    c_high = col_switch_ratio >= stripe_thresh

    if r_high and c_high:
        return 'checkered'
    elif r_high and not c_high:
        return 'vertical_stripe'
    elif not r_high and c_high:
        return 'horizontal_stripe'
    else:
        return 'solid'


def analyze_texture(image_rgb, mask, region_id, peaks_ab,
                    target_size=RLE_SIZE, stripe_thresh=STRIPE_SWITCH_THRESH):
    """
    分析区域的颜色纹理模式。

    参数:
        image_rgb:  (H, W, 3) uint8
        mask:       (H, W) int
        region_id:  int
        peaks_ab:   [(a, b, ratio), ...]
        target_size: 缩放尺寸
        stripe_thresh: 条纹判定切换比阈值

    返回:
        dict: {
            'pattern': str,             # 'solid'|'vertical_stripe'|'horizontal_stripe'|'checkered'
            'row_switch_ratio': float,
            'col_switch_ratio': float,
            'label_map': (64, 64) int or None,
        }
    """
    label_map = build_label_map(image_rgb, mask, region_id, peaks_ab, target_size)

    if label_map is None:
        return {
            'pattern': 'unknown',
            'row_switch_ratio': 0.0,
            'col_switch_ratio': 0.0,
            'label_map': None,
            'error': 'roi_too_small',
        }

    _, row_ratio = rle_row_scan(label_map)
    _, col_ratio = rle_col_scan(label_map)
    pattern = classify_pattern(row_ratio, col_ratio, stripe_thresh)

    return {
        'pattern': pattern,
        'row_switch_ratio': round(float(row_ratio), 4),
        'col_switch_ratio': round(float(col_ratio), 4),
        'label_map': label_map,
    }


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == '__main__':
    import sys, json
    from PIL import Image
    from histogram_peak import rgb_to_lab, detect_color_peaks

    if len(sys.argv) < 2:
        print("用法: python texture_rle.py <image_path> [mask_npy_path]")
        sys.exit(1)

    img = np.array(Image.open(sys.argv[1]).convert('RGB'))

    if len(sys.argv) >= 3:
        mask = np.load(sys.argv[2])
    else:
        mask = np.ones(img.shape[:2], dtype=np.int32)

    roi = mask == 1
    rgb_pixels = img[roi]
    lab_pixels = rgb_to_lab(rgb_pixels)
    ab_result = detect_color_peaks(lab_pixels)

    tex_result = analyze_texture(img, mask, 1, ab_result['peaks'])
    print(json.dumps(tex_result, indent=2, default=str))
