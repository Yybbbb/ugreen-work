#!/usr/bin/env python3
"""
a*b* 二维直方图峰值检测 —— 判断颜色空间是否存在多中心。

流程:
  1. mask 提取 ROI 像素 → RGB 数组
  2. RGB → Lab 颜色空间转换（仅保留 a*b*）
  3. 128×128 二维直方图 + 高斯平滑
  4. 局部极大值检测 → 峰值列表
  5. 低峰过滤 + 邻近峰合并 → n_peaks, purity
  6. L 通道旁路: 当 a*b* 只有一个峰但 L 异常分散时，仍可能为黑白条纹
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema


# ── 参数 ──
N_BINS = 128               # a*b* 直方图分辨率
SIGMA = 2.5                # 高斯平滑 σ (↑2.0→2.5: 更强去噪, 合并纹理小峰)
PEAK_THRESHOLD = 0.20      # 次峰高度 / 主峰高度 最低比例 (↑0.10→0.15→0.20: 减少纹理误检)
PEAK_MERGE_DIST = 15       # 峰合并距离（a*b* 空间 Δab）
L_RANGE_THRESHOLD = 100    # L 通道极差阈值 (↑70→90→100: 只有极端的黑白条纹才触发)
L_BIMODAL_THRESHOLD = 40   # L 通道双峰谷深阈值


# ═══════════════════════════════════════════
# RGB → Lab 转换（纯 numpy 实现，无 OpenCV 依赖）
# ═══════════════════════════════════════════

def rgb_to_lab(rgb):
    """
    RGB [0,255] → Lab.
    输入: (N, 3) 或 (H, W, 3) numpy uint8 array
    输出: 同形状 float32 array, L∈[0,100], a*∈[-128,127], b*∈[-128,127]
    """
    # 归一化 [0,1]
    rgb = rgb.astype(np.float32) / 255.0

    # sRGB 线性化
    mask = rgb > 0.04045
    rgb_lin = rgb / 12.92
    rgb_lin[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4

    # RGB → XYZ (D65)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    shape = rgb_lin.shape
    xyz = rgb_lin.reshape(-1, 3) @ mat.T
    xyz = xyz.reshape(shape)

    # 白点归一化 (D65)
    xyz[..., 0] /= 0.95047
    xyz[..., 1] /= 1.00000
    xyz[..., 2] /= 1.08883

    # 非线性压缩
    t = xyz
    mask_t = t > 0.008856
    f = np.zeros_like(t)
    f[mask_t] = np.cbrt(t[mask_t])
    f[~mask_t] = 7.787 * t[~mask_t] + 16.0 / 116.0

    # 组装 Lab
    lab = np.zeros_like(rgb, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0         # L
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])  # a*
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])  # b*

    return lab


# ═══════════════════════════════════════════
# 直方图峰值检测
# ═══════════════════════════════════════════

def detect_color_peaks(pixels_lab, n_bins=N_BINS, sigma=SIGMA,
                       peak_threshold=PEAK_THRESHOLD, merge_dist=PEAK_MERGE_DIST):
    """
    在 Lab 像素的 a*b* 二维直方图上检测颜色峰值。

    参数:
        pixels_lab: (N, 3) float32 array — ROI 区域的 Lab 像素
        n_bins: a*b* 直方图 bin 数
        sigma: 高斯平滑 σ
        peak_threshold: 次峰高度比例下限
        merge_dist: a*b* 空间峰合并距离

    返回:
        dict: {
            'n_peaks': int,
            'peaks': [(a_val, b_val, height_ratio), ...],  # 按高度降序
            'purity': float,  # 主峰占比 [0, 1]
            'ab_histogram': (128, 128) 平滑后的直方图
        }
    """
    if len(pixels_lab) < 50:
        # 像素太少，不可靠
        return {'n_peaks': 1, 'peaks': [(0.0, 0.0, 1.0)], 'purity': 1.0, 'ab_histogram': None}

    a_vals = pixels_lab[:, 1]
    b_vals = pixels_lab[:, 2]

    # 裁剪到有效范围 [-128, 127]
    a_vals = np.clip(a_vals, -128, 127)
    b_vals = np.clip(b_vals, -128, 127)

    # 构建二维直方图
    hist, a_edges, b_edges = np.histogram2d(
        a_vals, b_vals, bins=n_bins, range=[[-128, 127], [-128, 127]]
    )

    # 高斯平滑
    hist_smooth = gaussian_filter(hist, sigma=sigma)

    # 找局部极大值（8邻域）
    # argrelextrema 找相对最大值
    from scipy.signal import argrelextrema
    local_max_y, local_max_x = argrelextrema(hist_smooth, np.greater, order=1)

    if len(local_max_y) == 0:
        # 没有找到极值，整个直方图就是一个峰
        return {'n_peaks': 1, 'peaks': [(0.0, 0.0, 1.0)], 'purity': 1.0, 'ab_histogram': hist_smooth}

    # 组装峰值: (a_val, b_val, height)
    peaks = []
    for yi, xi in zip(local_max_y, local_max_x):
        a_center = (a_edges[yi] + a_edges[yi + 1]) / 2
        b_center = (b_edges[xi] + b_edges[xi + 1]) / 2
        height = hist_smooth[yi, xi]
        peaks.append((a_center, b_center, height))

    # 按高度降序排列
    peaks.sort(key=lambda p: p[2], reverse=True)

    # 低峰过滤
    main_height = peaks[0][2]
    if main_height == 0:
        return {'n_peaks': 1, 'peaks': [(0.0, 0.0, 1.0)], 'purity': 1.0, 'ab_histogram': hist_smooth}

    peaks = [p for p in peaks if p[2] >= peak_threshold * main_height]

    # 邻近峰合并（按高度优先，次峰与已保留的高峰逐个比对）
    merged = [peaks[0]]
    for p in peaks[1:]:
        too_close = False
        for m in merged:
            d_ab = np.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2)
            if d_ab < merge_dist:
                too_close = True
                break
        if not too_close:
            merged.append(p)

    # 计算占比
    total_height = sum(p[2] for p in merged)
    peaks_ratio = [(p[0], p[1], p[2] / total_height) for p in merged]
    purity = merged[0][2] / total_height if total_height > 0 else 1.0

    return {
        'n_peaks': len(merged),
        'peaks': peaks_ratio,
        'purity': float(purity),
        'ab_histogram': hist_smooth,
    }


# ── 空间连续性参数 ──
MIN_BLOB_AREA = 100         # 最小连通域面积（像素），低于此面积的色峰视为纹理噪声
CONNECTIVITY = 4            # 连通域连接方式: 4或8


# ═══════════════════════════════════════════
# 空间连续性过滤 —— 过滤纹理噪声产生的小峰
# ═══════════════════════════════════════════

def filter_peaks_by_spatial_contiguity(pixels_lab, roi_coords, peaks_ab,
                                        min_blob_area=100,
                                        connectivity=CONNECTIVITY):
    """
    空间连续性验证：只有形成连片区域的色峰才保留。

    原理:
      - 纯色织物纹理在 a*b* 空间也产生多个小峰（如褶皱处）
      - 但这些纹理峰对应的像素在空间上是散布的孤立点
      - 真正的多色块对应的像素在空间上是连片的

    参数:
        pixels_lab:   (N, 3) ROI内的Lab像素
        roi_coords:   (N, 2) yx坐标，对应每个像素在原图中的位置
        peaks_ab:     [(a, b, ratio), ...] 颜色峰列表
        min_blob_area: 最小连通域面积
        connectivity:  4或8连通

    返回:
        valid_peaks:  [(a, b, ratio), ...] 过滤后的有效峰列表
        peak_blobs:   {peak_idx: max_blob_area} 每个峰的最大连通域面积
    """
    from scipy.ndimage import label

    if len(peaks_ab) <= 1:
        return peaks_ab, {}

    a_vals = pixels_lab[:, 1]
    b_vals = pixels_lab[:, 2]
    peak_centers = np.array([(p[0], p[1]) for p in peaks_ab], dtype=np.float32)

    # 每个像素分配到最近峰
    pixel_labels = np.zeros(len(pixels_lab), dtype=np.int32)
    min_dist = np.full(len(pixels_lab), np.inf)
    for k, (pa, pb) in enumerate(peak_centers):
        d = np.sqrt((a_vals - pa)**2 + (b_vals - pb)**2)
        closer = d < min_dist
        pixel_labels[closer] = k
        min_dist[closer] = d[closer]

    # 重建空间标签图（仅在ROI边界框内）
    ys = roi_coords[:, 0]
    xs = roi_coords[:, 1]
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1

    peak_max_blob = {}
    valid_indices = set()

    for k in range(len(peaks_ab)):
        # 构建第k个峰的二进制空间图
        bin_map = np.zeros((h, w), dtype=np.uint8)
        k_mask = pixel_labels == k
        if not k_mask.any():
            continue
        local_ys = ys[k_mask] - y_min
        local_xs = xs[k_mask] - x_min
        bin_map[local_ys, local_xs] = 1

        # 连通域分析
        labeled, n_labels = label(bin_map, structure=np.ones((3, 3)) if connectivity == 8 else None)
        if n_labels == 0:
            continue

        # 找最大连通域面积
        blob_sizes = []
        for li in range(1, n_labels + 1):
            blob_sizes.append(int((labeled == li).sum()))
        max_area = max(blob_sizes) if blob_sizes else 0
        peak_max_blob[k] = max_area

        if max_area >= min_blob_area:
            valid_indices.add(k)

    # 主峰(peak 0)始终保留
    valid_indices.add(0)

    # 构建过滤后的峰列表
    valid_peaks = [peaks_ab[i] for i in range(len(peaks_ab)) if i in valid_indices]

    # 重新归一化ratio
    total_h = sum(p[2] for p in valid_peaks)
    valid_peaks = [(p[0], p[1], p[2] / total_h) for p in valid_peaks]

    return valid_peaks, peak_max_blob


# ═══════════════════════════════════════════
# L 通道旁路: 黑白 / 明暗条纹检测
# ═══════════════════════════════════════════

def check_l_channel_scatter(pixels_lab, l_range_threshold=L_RANGE_THRESHOLD):
    """
    检查 L 通道是否异常分散（暗示黑白/明暗交替）。

    返回:
        dict: {
            'l_anomaly': bool,       # 是否存在 L 异常
            'l_range': float,         # L 极差
            'l_variance': float,      # L 方差
            'l_is_bimodal': bool,     # L 是否呈双峰分布
        }
    """
    if len(pixels_lab) < 50:
        return {'l_anomaly': False, 'l_range': 0, 'l_variance': 0, 'l_is_bimodal': False}

    l_vals = pixels_lab[:, 0]
    l_range = float(np.max(l_vals) - np.min(l_vals))
    l_var = float(np.var(l_vals))

    # 简易双峰检测: 用 L 直方图判断
    l_hist, l_edges = np.histogram(l_vals, bins=50, range=(0, 100))
    l_hist_smooth = gaussian_filter(l_hist.astype(float), sigma=1.5)

    # 找 L 直方图的局部极小值（谷）
    local_min_indices = argrelextrema(l_hist_smooth, np.less, order=3)[0]

    is_bimodal = False
    if len(local_min_indices) >= 1:
        # 检查谷是否足够深（两侧峰高于谷）
        for mi in local_min_indices:
            left_max = np.max(l_hist_smooth[:mi]) if mi > 0 else 0
            right_max = np.max(l_hist_smooth[mi+1:]) if mi+1 < len(l_hist_smooth) else 0
            valley = l_hist_smooth[mi]
            if valley > 0 and left_max > 0 and right_max > 0:
                depth = min(left_max, right_max) / valley
                if depth > 2.0:  # 峰是谷的2倍以上
                    is_bimodal = True
                    break

    anomaly = (l_range > l_range_threshold) and is_bimodal

    return {
        'l_anomaly': anomaly,
        'l_range': l_range,
        'l_variance': l_var,
        'l_is_bimodal': is_bimodal,
    }


# ═══════════════════════════════════════════
# 联合判定
# ═══════════════════════════════════════════

def analyze_region_color(image_rgb, mask, region_id,
                         n_bins=N_BINS, sigma=SIGMA,
                         peak_threshold=PEAK_THRESHOLD,
                         merge_dist=PEAK_MERGE_DIST,
                         l_range_threshold=L_RANGE_THRESHOLD,
                         min_blob_area=MIN_BLOB_AREA,
                         connectivity=CONNECTIVITY):
    """
    分析图像中指定区域的杂色情况。

    参数:
        image_rgb:  (H, W, 3) uint8 RGB 图像
        mask:       (H, W) int 分割 mask
        region_id:  区域 ID（如 1=upper, 2=lower）
        min_blob_area: 空间连续性最小连通域面积（0=禁用）
        ... 其他参数见常量定义

    返回:
        dict: {
            'n_peaks': int, 'peaks': list, 'purity': float,
            'is_multicolor_ab': bool, 'l_anomaly': bool, 'is_multicolor': bool,
            'l_info': dict, 'n_pixels': int, 'spatial_filtered': bool,
        }
    """
    # 提取 ROI 像素
    roi = mask == region_id
    n_pixels = int(np.sum(roi))

    if n_pixels < 50:
        return {
            'n_peaks': 1, 'peaks': [(0.0, 0.0, 1.0)], 'purity': 1.0,
            'is_multicolor_ab': False,
            'l_anomaly': False, 'is_multicolor': False,
            'l_info': {}, 'n_pixels': n_pixels,
            'error': 'too_few_pixels', 'spatial_filtered': False,
        }

    rgb_pixels = image_rgb[roi]

    # 主路: a*b* 直方图峰值检测
    lab_pixels = rgb_to_lab(rgb_pixels)
    ab_result = detect_color_peaks(lab_pixels, n_bins, sigma, peak_threshold, merge_dist)

    # ★ 空间连续性过滤: 剔除纹理噪声产生的散点假峰
    spatial_filtered = False
    peak_blobs = {}
    if ab_result['n_peaks'] >= 2 and min_blob_area > 0:
        ys, xs = np.where(roi)
        roi_coords = np.column_stack([ys, xs])
        filtered_peaks, peak_blobs = filter_peaks_by_spatial_contiguity(
            lab_pixels, roi_coords, ab_result['peaks'],
            min_blob_area=min_blob_area, connectivity=connectivity
        )
        if len(filtered_peaks) < len(ab_result['peaks']):
            spatial_filtered = True
            ab_result['n_peaks'] = len(filtered_peaks)
            ab_result['peaks'] = filtered_peaks
            ab_result['purity'] = filtered_peaks[0][2] if filtered_peaks else 1.0

    # 旁路: L 通道检测
    l_info = check_l_channel_scatter(lab_pixels, l_range_threshold)

    # 判定
    is_multicolor_ab = (ab_result['n_peaks'] >= 2) and (ab_result['purity'] < 0.80)

    # 最终判定: a*b* 多中心 OR (单峰 + L 异常双峰)
    is_multicolor = is_multicolor_ab or (
        ab_result['n_peaks'] == 1 and l_info['l_anomaly']
    )

    return {
        'n_peaks': ab_result['n_peaks'],
        'peaks': ab_result['peaks'],
        'purity': ab_result['purity'],
        'is_multicolor_ab': is_multicolor_ab,
        'l_anomaly': l_info['l_anomaly'],
        'is_multicolor': is_multicolor,
        'l_info': l_info,
        'n_pixels': n_pixels,
        'spatial_filtered': spatial_filtered,
        'peak_blobs': {str(k): v for k, v in peak_blobs.items()},
    }


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python histogram_peak.py <image_path> [mask_npy_path]")
        sys.exit(1)

    from PIL import Image

    img = np.array(Image.open(sys.argv[1]).convert('RGB'))

    if len(sys.argv) >= 3:
        mask = np.load(sys.argv[2])
    else:
        # 假设整张图就是目标区域（用于快速测试）
        mask = np.ones(img.shape[:2], dtype=np.int32)

    result = analyze_region_color(img, mask, 1)
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'ab_histogram'}, indent=2, default=str))
