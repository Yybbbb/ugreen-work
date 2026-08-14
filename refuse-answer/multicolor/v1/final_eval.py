#!/usr/bin/env python3
"""用最优参数评估纯色/多色准确率和召回率"""

import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from histogram_peak import (
    rgb_to_lab, detect_color_peaks, check_l_channel_scatter,
    filter_peaks_by_spatial_contiguity, N_BINS
)

BASE = Path(__file__).parent
PURE_DIR = BASE / "val_pure"
MULTI_DIR = BASE / "val_multicolor/upper_multicolor"

# ── 最优参数 ──
# 切换评估哪个参数组合
EVAL_VERSION = sys.argv[1] if len(sys.argv) > 1 else 'balanced'

if EVAL_VERSION == 'multi_biased':
    PARAMS = {  # 偏向多色检测
        'sigma': 3.0, 'peak_threshold': 0.12, 'merge_dist': 15,
        'l_range_threshold': 100, 'purity_threshold': 0.80, 'min_blob_area': 20, 'n_bins': 128,
    }
else:
    PARAMS = {  # 最优平衡 (偏纯色)
        'sigma': 2.5, 'peak_threshold': 0.15, 'merge_dist': 15,
        'l_range_threshold': 100, 'purity_threshold': 0.80, 'min_blob_area': 50, 'n_bins': 128,
    }


def load_image_and_mask(img_path):
    try:
        img = np.array(Image.open(str(img_path)).convert('RGB'))
    except Exception:
        return None, None

    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.int32)

    stem = img_path.stem
    parts = stem.split('_', 1)
    orig_stem = parts[1] if len(parts) > 1 else stem

    for mf in [BASE / "val_multicolor/manifest.json",
               BASE / "val_multicolor/manifest_clean.json",
               BASE / "val_pure/manifest.json"]:
        if not mf.exists():
            continue
        with open(mf) as f:
            manifest = json.load(f)
        for entry in manifest:
            if orig_stem in entry.get('id', '') or orig_stem in entry.get('img_orig', ''):
                mp = entry.get('mask_path', '')
                if mp and Path(mp).exists():
                    try:
                        mask = _load_mask(mp, h, w)
                        return img, mask
                    except Exception:
                        pass
                break
        else:
            continue
        break
    return img, mask


def _load_mask(anno_path, h, w):
    with open(anno_path) as f:
        data = json.load(f)
    mask = np.zeros((h, w), dtype=np.int32)
    attrs = data.get('attributes', {})
    for region_id, region_name in [(1, 'upper'), (2, 'lower')]:
        ra = attrs.get(region_name, {})
        if not (ra.get('has_mask') and ra.get('mask')):
            continue
        md = ra['mask']
        fmt = md.get('format', '')
        if fmt == 'coco_rle' and md.get('rle'):
            d = _decode_coco_rle(md['rle'], h, w)
            if d is not None:
                mask[d > 0] = region_id
        elif fmt == 'lip_palette_ids':
            pr = md.get('path', '')
            lip_ids = md.get('lip_ids', [])
            if pr and lip_ids:
                pp = Path(pr)
                if not pp.exists():
                    pp = Path('/data1/work/MichaelYu/segment-color/data/LIP_clothes_accessory_unified') / pr
                if pp.exists():
                    pm = np.array(Image.open(str(pp)))
                    binary = np.isin(pm, lip_ids).astype(np.uint8)
                    if binary.shape[:2] != (h, w):
                        pi = Image.fromarray(binary * 255)
                        pi = pi.resize((w, h), Image.NEAREST)
                        binary = (np.array(pi) > 0).astype(np.uint8)
                    mask[binary > 0] = region_id
    return mask if (mask == 1).sum() > 0 or (mask == 2).sum() > 0 else None


def _decode_coco_rle(rle_dict, h, w):
    try:
        from pycocotools import mask as cocomask
        if isinstance(rle_dict, dict) and 'counts' in rle_dict:
            decoded = cocomask.decode(rle_dict)
            if decoded.shape[:2] != (h, w):
                from PIL import Image
                di = Image.fromarray(decoded)
                di = di.resize((w, h), Image.NEAREST)
                decoded = np.array(di)
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
                    rl = int(num) if num else 0
                    if rl > 0: mask[pos:pos+rl] = val; pos += rl
                    val = 1 - val; num = ''
            if num: rl = int(num); mask[pos:pos+rl] = val
            return mask.reshape((size[0], size[1]), order='F')
    except Exception:
        pass
    return None


def detect_upper(img, mask, params):
    """返回 (is_multicolor, detail_dict)"""
    if not isinstance(mask, np.ndarray) or mask.ndim < 2:
        return False, {'n_peaks': 1, 'purity': 1.0, 'n_pixels': 0, 'error': 'invalid_mask'}
    roi = mask == 1
    if isinstance(roi, (bool, np.bool_)):
        return False, {'n_peaks': 1, 'purity': 1.0, 'n_pixels': 0, 'error': 'invalid_roi'}
    n_pixels = int(roi.sum())
    if n_pixels < 50:
        return False, {'n_peaks': 1, 'purity': 1.0, 'n_pixels': n_pixels, 'error': 'too_few_pixels'}

    rgb_pixels = img[roi]
    lab_pixels = rgb_to_lab(rgb_pixels)

    # a*b* 峰值
    ab = detect_color_peaks(lab_pixels, n_bins=params['n_bins'],
                            sigma=params['sigma'], peak_threshold=params['peak_threshold'],
                            merge_dist=params['merge_dist'])

    # 空间连续性
    n_peaks = ab['n_peaks']
    purity = ab['purity']
    peaks = ab['peaks']
    spatial_filtered = False

    if n_peaks >= 2 and params['min_blob_area'] > 0:
        ys, xs = np.where(roi)
        roi_coords = np.column_stack([ys, xs])
        filtered_peaks, _ = filter_peaks_by_spatial_contiguity(
            lab_pixels, roi_coords, peaks,
            min_blob_area=params['min_blob_area'], connectivity=4
        )
        if len(filtered_peaks) < len(peaks):
            spatial_filtered = True
            n_peaks = len(filtered_peaks)
            peaks = filtered_peaks
            purity = filtered_peaks[0][2] if filtered_peaks else 1.0

    # L 通道
    l_info = check_l_channel_scatter(lab_pixels, params['l_range_threshold'])

    is_multi_ab = (n_peaks >= 2) and (purity < params['purity_threshold'])
    is_multi = is_multi_ab or (n_peaks == 1 and l_info['l_anomaly'])

    detail = {
        'n_peaks': n_peaks, 'purity': float(purity),
        'is_multi_ab': bool(is_multi_ab), 'l_anomaly': bool(l_info['l_anomaly']),
        'l_range': l_info['l_range'], 'n_pixels': n_pixels,
        'spatial_filtered': spatial_filtered,
    }
    return is_multi, detail


def main():
    # 收集图像
    pure_files = []
    for sub in ['source1_qwen', 'source2_upar_rare']:
        d = PURE_DIR / sub
        if d.exists():
            pure_files.extend(list(d.glob('*.jpg')) + list(d.glob('*.png')))

    multi_files = list(MULTI_DIR.glob('*.jpg')) + list(MULTI_DIR.glob('*.png'))

    print(f"纯色集: {len(pure_files)} 张")
    print(f"多色集: {len(multi_files)} 张")
    print(f"版本: {EVAL_VERSION}")
    print(f"参数: {json.dumps(PARAMS, indent=2)}")
    print()

    # ── 纯色集评估 ──
    pure_results = []
    pure_tp, pure_fp, pure_tn, pure_fn = 0, 0, 0, 0

    for img_path in pure_files:
        img, mask = load_image_and_mask(img_path)
        if img is None:
            continue
        is_multi, detail = detect_upper(img, mask, PARAMS)
        # 纯色集: ground truth = False (不是多色)
        gt = False
        if is_multi == gt:
            pure_tn += 1  # True Negative: 正确判为纯色
        else:
            pure_fp += 1  # False Positive: 误判为多色
        pure_results.append({'file': img_path.name, 'detected': bool(is_multi),
                             'gt': gt, 'correct': bool(is_multi == gt), 'detail': detail})

    # ── 多色集评估 ──
    multi_results = []
    multi_tp, multi_fn = 0, 0
    for img_path in multi_files:
        img, mask = load_image_and_mask(img_path)
        if img is None:
            continue
        is_multi, detail = detect_upper(img, mask, PARAMS)
        # 多色集: ground truth = True (是多色)
        gt = True
        if is_multi == gt:
            multi_tp += 1  # True Positive: 正确判为多色
        else:
            multi_fn += 1  # False Negative: 漏判为纯色
        multi_results.append({'file': img_path.name, 'detected': bool(is_multi),
                              'gt': gt, 'correct': bool(is_multi == gt), 'detail': detail})

    n_pure = pure_tn + pure_fp
    n_multi = multi_tp + multi_fn

    # 准确率 (Accuracy)
    pure_acc = pure_tn / n_pure if n_pure > 0 else 0
    multi_acc = multi_tp / n_multi if n_multi > 0 else 0
    overall_acc = (pure_tn + multi_tp) / (n_pure + n_multi) if (n_pure + n_multi) > 0 else 0

    # 精确率 (Precision): TP / (TP + FP)
    precision = multi_tp / (multi_tp + pure_fp) if (multi_tp + pure_fp) > 0 else 0

    # 召回率 (Recall): TP / (TP + FN)
    recall = multi_tp / (multi_tp + multi_fn) if (multi_tp + multi_fn) > 0 else 0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 特异度 (Specificity): TN / (TN + FP) — 纯色集准确率
    specificity = pure_tn / (pure_tn + pure_fp) if (pure_tn + pure_fp) > 0 else 0

    report = {
        'parameters': PARAMS,
        'pure_set': {
            'total': n_pure, 'correct': pure_tn, 'incorrect': pure_fp,
            'accuracy': round(pure_acc, 4),
        },
        'multi_set': {
            'total': n_multi, 'correct': multi_tp, 'incorrect': multi_fn,
            'accuracy': round(multi_acc, 4),
        },
        'overall': {
            'accuracy': round(overall_acc, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'specificity': round(specificity, 4),
            'f1': round(f1, 4),
        },
        'confusion_matrix': {
            'tp': multi_tp, 'fp': pure_fp,
            'tn': pure_tn, 'fn': multi_fn,
        },
    }

    # 输出
    print("=" * 60)
    print("最终评估结果")
    print("=" * 60)
    print(f"\n混淆矩阵:")
    print(f"                    预测纯色    预测多色")
    print(f"  实际纯色          TN={pure_tn:4d}    FP={pure_fp:4d}")
    print(f"  实际多色          FN={multi_fn:4d}    TP={multi_tp:4d}")
    print()
    print(f"纯色集准确率 (Specificity): {pure_acc:.1%}  ({pure_tn}/{n_pure})")
    print(f"多色集准确率 (Recall):      {multi_acc:.1%}  ({multi_tp}/{n_multi})")
    print(f"综合准确率:                 {overall_acc:.1%}")
    print(f"精确率 (Precision):         {precision:.1%}")
    print(f"召回率 (Recall):            {recall:.1%}")
    print(f"F1:                         {f1:.1%}")
    print(f"特异度 (Specificity):       {specificity:.1%}")

    # 保存
    out_path = BASE / f"final_eval_{EVAL_VERSION}.json"
    report['pure_results'] = pure_results
    report['multi_results'] = multi_results
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果: {out_path}")

    # 失败案例分析
    print(f"\n--- 纯色集误判 (FP) 前10 ---")
    fp_samples = [r for r in pure_results if not r['correct']][:10]
    for r in fp_samples:
        d = r['detail']
        print(f"  {r['file']}: n_peaks={d.get('n_peaks','?')}, purity={d.get('purity',1):.3f}, "
              f"ab={d.get('is_multi_ab','?')}, L={d.get('l_anomaly','?')}, spatial={d.get('spatial_filtered','?')}")

    print(f"\n--- 多色集漏判 (FN) 前10 ---")
    fn_samples = [r for r in multi_results if not r['correct']][:10]
    for r in fn_samples:
        d = r['detail']
        print(f"  {r['file']}: n_peaks={d.get('n_peaks','?')}, purity={d.get('purity',0):.3f}, "
              f"ab={d.get('is_multi_ab','?')}, L={d.get('l_anomaly','?')}, spatial={d.get('spatial_filtered','?')}")


if __name__ == '__main__':
    main()
