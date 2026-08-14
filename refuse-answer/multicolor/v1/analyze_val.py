#!/usr/bin/env python3
"""
在验证集上批量运行杂色检测器，输出分析报告。

用法:
  python analyze_val.py [--limit N] [--output report.json]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# 添加当前目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from multicolor_detector import detect_upper_lower

VAL_DIR = Path(__file__).parent / "val_multicolor"  # 默认，可被 --val-dir 覆盖


def load_image_and_mask(sample_entry):
    """
    从manifest中的mask_path直接加载分割mask。

    manifest中每条记录有:
      - id: val_multicolor下的相对路径
      - mask_path: 标注JSON的绝对路径 (含coco_rle或lip_palette_ids mask)

    返回: (image_rgb, mask) or (None, None)
    """
    img_path = VAL_DIR / sample_entry['id']

    if not img_path.exists():
        return None, None

    try:
        img = np.array(Image.open(str(img_path)).convert('RGB'))
    except Exception:
        return None, None

    mask_path = sample_entry.get('mask_path', '')
    if not mask_path or not Path(mask_path).exists():
        return None, None

    return img, _load_mask_from_anno(mask_path, img.shape[:2])


def _load_mask_from_anno(anno_path, img_shape):
    """
    从标注JSON中解码upper/lower的mask。

    支持格式:
      - coco_rle: COCO RLE编码 (Qwen原始标注, UPAR SAM3标注)
      - lip_palette_ids: LIP PNG调色板 (部分Qwen标注)
    """
    try:
        with open(anno_path) as f:
            data = json.load(f)
    except Exception:
        return None

    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.int32)
    has_any = False

    attrs = data.get('attributes', {})
    if not isinstance(attrs, dict):
        return None

    for region_id, region_name in [(1, 'upper'), (2, 'lower')]:
        region_attrs = attrs.get(region_name)
        if not isinstance(region_attrs, dict):
            continue
        if not region_attrs.get('has_mask'):
            continue

        mask_data = region_attrs.get('mask')
        if not isinstance(mask_data, dict):
            continue

        fmt = mask_data.get('format', '')
        decoded = None

        if fmt == 'coco_rle' and mask_data.get('rle'):
            decoded = _decode_coco_rle(mask_data['rle'], h, w)

        elif fmt == 'lip_palette_ids':
            decoded = _decode_lip_palette(mask_data, h, w)

        if decoded is not None:
            # 若mask尺寸与图像不一致，resize
            if decoded.shape[0] != h or decoded.shape[1] != w:
                from PIL import Image as PILImage
                d_img = PILImage.fromarray(decoded.astype(np.uint8) * 255)
                d_img = d_img.resize((w, h), PILImage.NEAREST)
                decoded = (np.array(d_img) > 127).astype(np.uint8)
            mask[decoded > 0] = region_id
            has_any = True

    return mask if has_any else None


def _decode_lip_palette(mask_data, h, w):
    """解码LIP palette PNG格式的mask"""
    png_rel = mask_data.get('path', '')
    lip_ids = mask_data.get('lip_ids', [])
    if not png_rel or not lip_ids:
        return None

    png_path = Path(png_rel)
    if not png_path.exists():
        alt_path = Path('/data1/work/MichaelYu/segment-color/data/LIP_clothes_accessory_unified') / png_rel
        if alt_path.exists():
            png_path = alt_path
        else:
            return None

    try:
        png_mask = np.array(Image.open(str(png_path)))
    except Exception:
        return None

    binary = np.isin(png_mask, lip_ids).astype(np.uint8)

    if binary.shape[0] != h or binary.shape[1] != w:
        png_img = Image.fromarray(binary * 255)
        png_img = png_img.resize((w, h), Image.NEAREST)
        binary = (np.array(png_img) > 0).astype(np.uint8)

    return binary


def _decode_coco_rle(rle_dict, h, w):
    """解码COCO RLE格式的mask"""
    try:
        from pycocotools import mask as cocomask
        if isinstance(rle_dict, dict) and 'counts' in rle_dict:
            decoded = cocomask.decode(rle_dict)
            if decoded.shape[0] != h or decoded.shape[1] != w:
                # 尝试resize
                from PIL import Image
                decoded_img = Image.fromarray(decoded)
                decoded_img = decoded_img.resize((w, h), Image.NEAREST)
                decoded = np.array(decoded_img)
            return decoded
    except ImportError:
        pass

    # 纯Python RLE解码 (fallback)
    try:
        counts = rle_dict.get('counts', '')
        size = rle_dict.get('size', [h, w])
        if isinstance(counts, str):
            # RLE字符串解码
            mask = np.zeros(size[0] * size[1], dtype=np.uint8)
            pos = 0
            val = 0
            num = ''
            for c in counts:
                if c.isdigit():
                    num += c
                else:
                    run_len = int(num) if num else 0
                    if run_len > 0:
                        mask[pos:pos+run_len] = val
                        pos += run_len
                    val = 1 - val
                    num = ''
            if num:
                run_len = int(num)
                mask[pos:pos+run_len] = val
            return mask.reshape((size[0], size[1]), order='F')
    except Exception:
        pass

    return None


def run_batch(limit=None, output_path=None):
    """批量运行检测器"""
    # 优先使用清洗后的manifest
    clean_manifest = VAL_DIR / "manifest_clean.json"
    if clean_manifest.exists():
        manifest_path = clean_manifest
    else:
        manifest_path = VAL_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if limit:
        manifest = manifest[:limit]

    results = []
    stats = {
        'total': 0,
        'mask_loaded': 0,
        'mask_failed': 0,
        'detected_upper_multi': 0,
        'detected_lower_multi': 0,
        'detected_any_multi': 0,
        'pattern_dist': defaultdict(int),
        'peak_dist': defaultdict(int),
        'l_anomaly_count': 0,
        'errors': 0,
    }

    t0 = time.time()
    for i, entry in enumerate(manifest):
        img, mask = load_image_and_mask(entry)

        if img is None or mask is None:
            if img is None:
                stats['errors'] += 1
            if mask is None:
                stats['mask_failed'] += 1
            results.append({
                'id': entry['id'],
                'error': 'img_missing' if img is None else 'mask_missing',
            })
            continue

        stats['total'] += 1
        stats['mask_loaded'] += 1

        try:
            detection = detect_upper_lower(img, mask)
        except Exception as e:
            stats['errors'] += 1
            results.append({
                'id': entry['id'],
                'error': str(e),
            })
            continue

        # 汇总
        entry_result = {
            'id': entry['id'],
            'source': entry['source'],
            'img_orig': entry.get('img_orig', ''),
            'expected_upper_multi': entry.get('is_upper_multi', None),
            'expected_lower_multi': entry.get('is_lower_multi', None),
            'detection': detection,
        }

        # 统计
        for region_key in ['upper', 'lower']:
            rd = detection[region_key]
            if rd['is_multicolor']:
                if region_key == 'upper':
                    stats['detected_upper_multi'] += 1
                else:
                    stats['detected_lower_multi'] += 1
                stats['detected_any_multi'] += 1

            stats['pattern_dist'][f"{region_key}:{rd['pattern']}"] += 1
            stats['peak_dist'][f"{region_key}:{rd['n_peaks']}"] += 1

            if rd.get('ab_result', {}).get('l_anomaly'):
                stats['l_anomaly_count'] += 1

        results.append(entry_result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(manifest)}] {elapsed:.1f}s, "
                  f"detected_multi={stats['detected_any_multi']}")

    elapsed = time.time() - t0

    # 构建报告
    report = {
        'config': {
            'n_bins': 128, 'sigma': 2.0, 'peak_threshold': 0.10,
            'peak_merge_dist': 15, 'l_range_threshold': 70,
            'stripe_switch_thresh': 0.05, 'rle_size': 64,
        },
        'stats': {
            'total_samples': stats['total'],
            'mask_loaded': stats['mask_loaded'],
            'mask_failed': stats['mask_failed'],
            'errors': stats['errors'],
            'elapsed_seconds': round(elapsed, 1),
            'avg_ms_per_sample': round(elapsed / max(stats['total'], 1) * 1000, 1),

            'detected_upper_multi': stats['detected_upper_multi'],
            'detected_lower_multi': stats['detected_lower_multi'],
            'detected_any_multi': stats['detected_any_multi'],

            'pattern_distribution': dict(stats['pattern_dist']),
            'peak_distribution': dict(stats['peak_dist']),
            'l_anomaly_count': stats['l_anomaly_count'],

            # 弱标签对照
            'agreement': _compute_agreement(results),
        },
        'results': results,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n报告已保存: {output_path}")

    # 打印摘要
    _print_summary(report['stats'])

    return report


def _compute_agreement(results):
    """计算检测结果与弱标签的一致性（弱标签非真值，仅供参考）"""
    agree_upper = 0
    agree_lower = 0
    total_upper_expected = 0
    total_lower_expected = 0
    disagree_upper = []
    disagree_lower = []

    for r in results:
        if 'error' in r:
            continue
        det = r.get('detection')
        if not det:
            continue

        for region_key in ['upper', 'lower']:
            expected = r.get(f'expected_{region_key}_multi')
            if expected is None:
                continue
            detected = det[region_key]['is_multicolor']

            if region_key == 'upper':
                total_upper_expected += 1
                if expected == detected:
                    agree_upper += 1
                else:
                    disagree_upper.append({
                        'id': r['id'],
                        'expected': expected,
                        'detected': detected,
                        'n_peaks': det[region_key]['n_peaks'],
                        'purity': det[region_key]['purity'],
                        'pattern': det[region_key]['pattern'],
                    })
            else:
                total_lower_expected += 1
                if expected == detected:
                    agree_lower += 1
                else:
                    disagree_lower.append({
                        'id': r['id'],
                        'expected': expected,
                        'detected': detected,
                        'n_peaks': det[region_key]['n_peaks'],
                        'purity': det[region_key]['purity'],
                        'pattern': det[region_key]['pattern'],
                    })

    return {
        'upper': {
            'total': total_upper_expected,
            'agree': agree_upper,
            'accuracy': round(agree_upper / max(total_upper_expected, 1), 3),
            'disagree_samples': disagree_upper[:20],  # Top 20 disagreements
        },
        'lower': {
            'total': total_lower_expected,
            'agree': agree_lower,
            'accuracy': round(agree_lower / max(total_lower_expected, 1), 3),
            'disagree_samples': disagree_lower[:20],
        },
    }


def _print_summary(stats):
    print(f"\n{'='*60}")
    print(f"检测结果摘要")
    print(f"{'='*60}")
    print(f"  总样本:     {stats['total_samples']}")
    print(f"  mask成功:   {stats['mask_loaded']}")
    print(f"  mask失败:   {stats['mask_failed']}")
    print(f"  错误:       {stats['errors']}")
    print(f"  耗时:       {stats['elapsed_seconds']:.1f}s ({stats['avg_ms_per_sample']:.1f}ms/样本)")
    print()
    print(f"  检测 Upper 多色: {stats['detected_upper_multi']} / {stats['total_samples']} ({stats['detected_upper_multi']/max(stats['total_samples'],1)*100:.1f}%)")
    print(f"  检测 Lower 多色: {stats['detected_lower_multi']} / {stats['total_samples']} ({stats['detected_lower_multi']/max(stats['total_samples'],1)*100:.1f}%)")
    print(f"  任意区域多色:    {stats['detected_any_multi']} / {stats['total_samples']}")
    print(f"  L通道异常检测:   {stats['l_anomaly_count']}")
    print()
    print("  峰数量分布:")
    for k, v in sorted(stats['peak_distribution'].items()):
        print(f"    {k}: {v}")
    print()
    print("  纹理分布:")
    for k, v in sorted(stats['pattern_distribution'].items()):
        print(f"    {k}: {v}")

    if stats.get('agreement'):
        for region_key in ['upper', 'lower']:
            a = stats['agreement'][region_key]
            print(f"\n  [{region_key}] 弱标签一致性: {a['accuracy']:.1%} ({a['agree']}/{a['total']})")
            if a['disagree_samples']:
                print(f"  不一致样本 (前5):")
                for d in a['disagree_samples'][:5]:
                    print(f"    {d['id']}: 期望={d['expected']}, 检测={d['detected']}, "
                          f"n_peaks={d['n_peaks']}, purity={d['purity']:.2f}, pattern={d['pattern']}")


def main():
    parser = argparse.ArgumentParser(description='在验证集上批量运行杂色检测器')
    parser.add_argument('--val-dir', type=str, default=None, help='验证集目录 (默认 val_multicolor)')
    parser.add_argument('--limit', type=int, default=None, help='限制样本数')
    parser.add_argument('--output', type=str, default=None, help='输出JSON路径')
    args = parser.parse_args()

    global VAL_DIR
    if args.val_dir:
        VAL_DIR = Path(args.val_dir)
    manifest_path = VAL_DIR / "manifest.json"

    output_path = args.output or str(VAL_DIR / 'results.json')
    run_batch(limit=args.limit, output_path=output_path)


if __name__ == '__main__':
    main()
