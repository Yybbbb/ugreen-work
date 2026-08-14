#!/usr/bin/env python3
"""
参数调优脚本 —— 在纯色集和多色集上网格搜索最优参数。

目标: 纯色准确率 > 85% AND 多色准确率 > 85%

调优参数:
  - PEAK_THRESHOLD: 次峰过滤阈值
  - SIGMA: 高斯平滑强度
  - L_RANGE_THRESHOLD: L通道旁路阈值
  - PURITY_THRESHOLD: 纯度判定线
"""

import json, sys, time, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# 直接导入 histo_peak 的低层函数（绕过 detector 的默认参数）
from histogram_peak import (
    rgb_to_lab, detect_color_peaks, check_l_channel_scatter,
    filter_peaks_by_spatial_contiguity,
    N_BINS
)
from texture_rle import analyze_texture, STRIPE_SWITCH_THRESH, RLE_SIZE

BASE = Path(__file__).parent
PURE_DIR = BASE / "val_pure"
MULTI_UPPER_DIR = BASE / "val_multicolor/upper_multicolor"


def load_image_and_upper_mask(img_path):
    """加载图像，尝试找到 mask 并提取 upper 区域。无mask则全图当作upper。"""
    try:
        img = np.array(Image.open(str(img_path)).convert('RGB'))
    except Exception:
        return None, None

    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.int32)

    # 尝试从文件名反查 manifest 中的 mask_path
    # 文件名格式: XXXX_original_stem.jpg
    stem = img_path.stem
    parts = stem.split('_', 1)
    orig_stem = parts[1] if len(parts) > 1 else stem

    # 尝试在 val_multicolor/manifest.json 和 val_pure/manifest.json 中查找
    for manifest_path in [BASE / "val_multicolor/manifest.json",
                          BASE / "val_multicolor/manifest_clean.json",
                          BASE / "val_pure/manifest.json"]:
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            if orig_stem in entry.get('id', '') or orig_stem in entry.get('img_orig', ''):
                mask_path = entry.get('mask_path', '')
                if mask_path and Path(mask_path).exists():
                    try:
                        mask = _load_mask(str(mask_path), h, w)
                        if mask is not None:
                            break
                    except Exception:
                        pass
        else:
            continue
        break

    return img, mask


def _load_mask(anno_path, h, w):
    """从标注JSON加载upper/lower mask"""
    with open(anno_path) as f:
        data = json.load(f)

    mask = np.zeros((h, w), dtype=np.int32)
    attrs = data.get('attributes', {})

    for region_id, region_name in [(1, 'upper'), (2, 'lower')]:
        region_attrs = attrs.get(region_name, {})
        if not (region_attrs.get('has_mask') and region_attrs.get('mask')):
            continue
        mask_data = region_attrs['mask']
        fmt = mask_data.get('format', '')

        if fmt == 'coco_rle' and mask_data.get('rle'):
            decoded = _decode_coco_rle(mask_data['rle'], h, w)
            if decoded is not None:
                mask[decoded > 0] = region_id

        elif fmt == 'lip_palette_ids':
            png_rel = mask_data.get('path', '')
            lip_ids = mask_data.get('lip_ids', [])
            if png_rel and lip_ids:
                png_path = Path(png_rel)
                if not png_path.exists():
                    png_path = Path('/data1/work/MichaelYu/segment-color/data/LIP_clothes_accessory_unified') / png_rel
                if png_path.exists():
                    png_mask = np.array(Image.open(str(png_path)))
                    binary = np.isin(png_mask, lip_ids).astype(np.uint8)
                    if binary.shape[:2] != (h, w):
                        png_img = Image.fromarray(binary * 255)
                        png_img = png_img.resize((w, h), Image.NEAREST)
                        binary = (np.array(png_img) > 0).astype(np.uint8)
                    mask[binary > 0] = region_id

    return mask if (mask == 1).sum() > 0 or (mask == 2).sum() > 0 else None


def _decode_coco_rle(rle_dict, h, w):
    try:
        from pycocotools import mask as cocomask
        if isinstance(rle_dict, dict) and 'counts' in rle_dict:
            decoded = cocomask.decode(rle_dict)
            if decoded.shape[:2] != (h, w):
                from PIL import Image
                decoded_img = Image.fromarray(decoded)
                decoded_img = decoded_img.resize((w, h), Image.NEAREST)
                decoded = np.array(decoded_img)
            return decoded
    except ImportError:
        pass

    try:
        counts = rle_dict.get('counts', '')
        size = rle_dict.get('size', [h, w])
        if isinstance(counts, str) and counts:
            mask = np.zeros(size[0] * size[1], dtype=np.uint8)
            pos = 0; val = 0; num = ''
            for c in counts:
                if c.isdigit(): num += c
                else:
                    run_len = int(num) if num else 0
                    if run_len > 0: mask[pos:pos+run_len] = val; pos += run_len
                    val = 1 - val; num = ''
            if num:
                run_len = int(num)
                mask[pos:pos+run_len] = val
            return mask.reshape((size[0], size[1]), order='F')
    except Exception:
        pass
    return None


def classify_upper(img, mask, peak_threshold, sigma, l_range_threshold, purity_threshold,
                   min_blob_area=100):
    """对 upper 区域做杂色判定，返回 is_multicolor + 详情"""
    if not isinstance(mask, np.ndarray) or mask.ndim < 2:
        return False, {'n_peaks': 1, 'purity': 1.0, 'n_pixels': 0, 'error': 'invalid_mask'}
    roi = mask == 1
    n_pixels = int(roi.sum())

    if n_pixels < 50:
        # 像素太少，保守返回纯色
        return False, {'n_peaks': 1, 'purity': 1.0, 'n_pixels': n_pixels, 'error': 'too_few_pixels'}

    rgb_pixels = img[roi]
    lab_pixels = rgb_to_lab(rgb_pixels)

    # a*b* 峰值检测
    ab_result = detect_color_peaks(
        lab_pixels, n_bins=N_BINS, sigma=sigma,
        peak_threshold=peak_threshold, merge_dist=15
    )

    # 空间连续性过滤：纹理噪声小峰在空间上是散布的 → 过滤掉
    roi_coords = np.column_stack(np.where(roi))
    filtered_peaks, blob_info = filter_peaks_by_spatial_contiguity(
        lab_pixels, roi_coords, ab_result['peaks'],
        min_blob_area=min_blob_area
    )
    n_peaks_f = len(filtered_peaks)
    purity_f = filtered_peaks[0][2] if filtered_peaks else 1.0

    # L 通道检查
    l_info = check_l_channel_scatter(lab_pixels, l_range_threshold)

    # 判定 (使用空间过滤后的峰)
    is_multi_ab = (n_peaks_f >= 2) and (purity_f < purity_threshold)
    is_multi = is_multi_ab or (
        n_peaks_f == 1 and l_info['l_anomaly']
    )

    detail = {
        'n_peaks_raw': ab_result['n_peaks'],
        'n_peaks_filtered': n_peaks_f,
        'purity': purity_f,
        'purity_raw': ab_result['purity'],
        'is_multi_ab': is_multi_ab,
        'l_anomaly': l_info['l_anomaly'],
        'l_range': l_info['l_range'],
        'n_pixels': n_pixels,
        'blob_info': {str(k): v for k, v in blob_info.items()},
    }
    return is_multi, detail


def evaluate_params(peak_threshold, sigma, l_range_threshold, purity_threshold,
                    min_blob_area, pure_files, multi_upper_files, verbose=False):
    """用给定参数评估纯色集和多色集的准确率"""
    pure_correct = 0
    pure_total = 0
    multi_correct = 0
    multi_total = 0

    # 纯色集: 期望 is_multicolor=False
    for img_path in pure_files:
        img, mask = load_image_and_upper_mask(img_path)
        if img is None:
            continue
        is_multi, detail = classify_upper(
            img, mask, peak_threshold, sigma, l_range_threshold, purity_threshold, min_blob_area
        )
        pure_total += 1
        if not is_multi:
            pure_correct += 1
        elif verbose:
            print(f"  [纯色FP] {img_path.name}: raw_peaks={detail['n_peaks_raw']}→{detail['n_peaks_filtered']}, "
                  f"purity={detail['purity']:.2f}, blobs={detail['blob_info']}")

    # 多色集 upper: 期望 is_multicolor=True
    for img_path in multi_upper_files:
        img, mask = load_image_and_upper_mask(img_path)
        if img is None:
            continue
        is_multi, detail = classify_upper(
            img, mask, peak_threshold, sigma, l_range_threshold, purity_threshold, min_blob_area
        )
        multi_total += 1
        if is_multi:
            multi_correct += 1
        elif verbose:
            print(f"  [多色FN] {img_path.name}: raw_peaks={detail['n_peaks_raw']}→{detail['n_peaks_filtered']}, "
                  f"purity={detail['purity']:.2f}, blobs={detail['blob_info']}")

    pure_acc = pure_correct / max(pure_total, 1)
    multi_acc = multi_correct / max(multi_total, 1)
    combined = (pure_acc + multi_acc) / 2

    return {
        'peak_threshold': peak_threshold,
        'sigma': sigma,
        'l_range_threshold': l_range_threshold,
        'purity_threshold': purity_threshold,
        'min_blob_area': min_blob_area,
        'pure_correct': pure_correct,
        'pure_total': pure_total,
        'pure_acc': round(pure_acc, 4),
        'multi_correct': multi_correct,
        'multi_total': multi_total,
        'multi_acc': round(multi_acc, 4),
        'combined_acc': round(combined, 4),
    }


def main():
    # 收集图像文件
    pure_files = []
    for sub in ['source1_qwen', 'source2_upar_rare']:
        d = PURE_DIR / sub
        if d.exists():
            pure_files.extend(list(d.glob('*.jpg')) + list(d.glob('*.png')))

    multi_upper_files = list(MULTI_UPPER_DIR.glob('*.jpg')) + list(MULTI_UPPER_DIR.glob('*.png'))

    print(f"=" * 70)
    print(f"参数调优")
    print(f"  纯色集(期望False): {len(pure_files)} 张")
    print(f"  多色集(期望True):  {len(multi_upper_files)} 张")
    print(f"  目标: 两者准确率均 >85%")
    print(f"=" * 70)

    # 参数网格 (第二轮: 降低min_blob_area和peak_threshold以提多色召回)
    param_grid = {
        'peak_threshold': [0.12, 0.15, 0.18, 0.20],
        'sigma': [2.5, 2.8, 3.0],
        'l_range_threshold': [100],
        'purity_threshold': [0.72, 0.75, 0.78, 0.80],
        'min_blob_area': [20, 30, 50, 80, 100],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n参数组合: {total_combos} 种")

    results = []
    best = None
    start = time.time()

    for i, (pt, sg, lrt, pur, mba) in enumerate(itertools.product(*param_grid.values())):
        r = evaluate_params(pt, sg, lrt, pur, mba, pure_files, multi_upper_files)
        results.append(r)

        if best is None or r['combined_acc'] > best['combined_acc']:
            best = r

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{total_combos}] {elapsed:.0f}s  best 纯色={best['pure_acc']:.1%} "
                  f"多色={best['multi_acc']:.1%} 综合={best['combined_acc']:.1%}")

    elapsed = time.time() - start

    # 排序: 综合准确率优先，然后看纯色
    results.sort(key=lambda r: (r['combined_acc'], r['pure_acc'], r['multi_acc']), reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP 20 结果 (总{len(results)}组, 耗时{elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'P_th':<8} {'σ':<6} {'L_th':<8} {'pur_th':<8} {'blob':<6} "
          f"{'纯色ACC':<10} {'多色ACC':<10} {'综合':<8}")
    print(f"{'-'*70}")

    for i, r in enumerate(results[:20]):
        star = " ★" if (r['pure_acc'] >= 0.85 and r['multi_acc'] >= 0.85) else ""
        print(f"{i+1:<5} {r['peak_threshold']:<8.2f} {r['sigma']:<6.1f} "
              f"{r['l_range_threshold']:<8} {r['purity_threshold']:<8.2f} {r['min_blob_area']:<6} "
              f"{r['pure_acc']:<10.1%} {r['multi_acc']:<10.1%} {r['combined_acc']:<8.1%}{star}")

    # 达标结果
    passed = [r for r in results if r['pure_acc'] >= 0.85 and r['multi_acc'] >= 0.85]
    print(f"\n{'='*70}")
    if passed:
        print(f"达标结果 ({len(passed)}/{len(results)}):")
        for r in passed[:10]:
            print(f"  PEAK_THRESHOLD={r['peak_threshold']}, SIGMA={r['sigma']}, "
                  f"L_RANGE_THRESHOLD={r['l_range_threshold']}, PURITY_THRESHOLD={r['purity_threshold']}")
            print(f"    纯色={r['pure_correct']}/{r['pure_total']} ({r['pure_acc']:.1%})  "
                  f"多色={r['multi_correct']}/{r['multi_total']} ({r['multi_acc']:.1%})")
    else:
        print("没有参数组合同时达到85%! 最接近的:")
        for r in results[:3]:
            print(f"  PEAK_THRESHOLD={r['peak_threshold']}, SIGMA={r['sigma']}, "
                  f"L_RANGE_THRESHOLD={r['l_range_threshold']}, PURITY_THRESHOLD={r['purity_threshold']}")
            print(f"    纯色={r['pure_correct']}/{r['pure_total']} ({r['pure_acc']:.1%})  "
                  f"多色={r['multi_correct']}/{r['multi_total']} ({r['multi_acc']:.1%})")

    # 保存
    output_path = BASE / "tune_results.json"
    with open(output_path, 'w') as f:
        json.dump({'results': results, 'best': best, 'passed': passed}, f, indent=2)
    print(f"\n结果已保存: {output_path}")

    # 更新 histogram_peak.py 的参数
    if passed:
        best_pass = passed[0]
        print(f"\n建议更新 histogram_peak.py:")
        print(f"  PEAK_THRESHOLD = {best_pass['peak_threshold']}")
        print(f"  SIGMA = {best_pass['sigma']}")
        print(f"  L_RANGE_THRESHOLD = {best_pass['l_range_threshold']}")
        print(f"  MIN_BLOB_AREA = {best_pass['min_blob_area']}")
        print(f"  # PURITY_THRESHOLD logic needs {best_pass['purity_threshold']}")
    else:
        best_r = results[0]
        print(f"  PEAK_THRESHOLD = {best_r['peak_threshold']}")
        print(f"  SIGMA = {best_r['sigma']}")
        print(f"  L_RANGE_THRESHOLD = {best_r['l_range_threshold']}")
        print(f"  MIN_BLOB_AREA = {best_r['min_blob_area']}")
        print(f"  PURITY_THRESHOLD = {best_r['purity_threshold']}")


if __name__ == '__main__':
    main()
