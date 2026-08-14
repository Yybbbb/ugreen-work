#!/usr/bin/env python3
"""
构建杂色检测验证集 v2 —— 全部带mask。

来源1 (200): Qwen标注 — labeling/annotation/ → mask在 data/LIP_clothes_accessory_unified/annotations/
来源2 (50):  UPAR rare_color — rare_color_samples.txt → mask在 data/UPAR_rare_color/annotations/

输出:
  val_multicolor/
    source1_qwen/         (200)
    source2_upar_rare/    (50)
    manifest.json
"""

import os, sys, json, random, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

BASE = Path("/data1/work/MichaelYu/segment-color")
OUT_DIR = BASE / "refuse-answer/multicolor/val_multicolor"

COLOR_NAMES = ["Black","Blue","Brown","Green","Grey","Orange","Pink","Purple","Red","White","Yellow","Other"]


# ═══════════════════════════════════════════
# 来源1: Qwen 标注 (200张)
# ═══════════════════════════════════════════

def extract_source1(n_target=200):
    """从 Qwen 标注中抽取多色+高置信度样本，验证mask存在"""
    anno_dir = BASE / "labeling/annotation"

    candidates = []
    for af in anno_dir.glob("*.json"):
        try:
            with open(af) as f:
                data = json.load(f)
        except Exception:
            continue

        # 读取 Qwen 颜色标注
        parsed = data.get('parsed', {})
        if not parsed:
            continue

        upper_high = [e for e in parsed.get('upper', [])
                      if e.get('confidence', 0) >= 0.70 and e.get('label', '') != 'unknown']
        lower_high = [e for e in parsed.get('lower', [])
                      if e.get('confidence', 0) >= 0.70 and e.get('label', '') != 'unknown']

        is_upper_multi = len(upper_high) >= 2
        is_lower_multi = len(lower_high) >= 2
        if not (is_upper_multi or is_lower_multi):
            continue

        # 验证mask路径
        mask_path = data.get('annotation_path', '')
        if not mask_path or not Path(mask_path).exists():
            continue

        # 验证image路径
        img_path = data.get('image_path', '')
        if not img_path or not Path(img_path).exists():
            continue

        candidates.append({
            'img_path': img_path,
            'mask_path': mask_path,
            'upper_colors': [e['label'] for e in upper_high],
            'lower_colors': [e['label'] for e in lower_high],
            'upper_conf': [e['confidence'] for e in upper_high],
            'lower_conf': [e['confidence'] for e in lower_high],
            'is_upper_multi': is_upper_multi,
            'is_lower_multi': is_lower_multi,
            'source': f'qwen:{af.name}',
        })

    print(f"[来源1] Qwen: {len(candidates)} 个候选 (多色+高置信+有mask)")

    # 按置信度排序，优先选置信度高的
    candidates.sort(key=lambda c: sum(c['upper_conf']) + sum(c['lower_conf']), reverse=True)

    # 按颜色组合做分层抽样
    combo_groups = defaultdict(list)
    for c in candidates:
        key = '+'.join(sorted(c['upper_colors'])) + '|' + '+'.join(sorted(c['lower_colors']))
        combo_groups[key].append(c)

    sampled = []
    # 每组合至少取1个，其余按组合大小比例分配
    rem = n_target - len(combo_groups)
    for key, group in sorted(combo_groups.items(), key=lambda x: -len(x[1])):
        n_take = 1 + max(0, int(rem * len(group) / len(candidates)))
        n_take = min(n_take, len(group))
        sampled.extend(group[:n_take])

    sampled = sampled[:n_target]
    print(f"  → 抽取 {len(sampled)} 张 ({len(combo_groups)} 种颜色组合)")
    return sampled


# ═══════════════════════════════════════════
# 来源2: UPAR rare_color (50张)
# ═══════════════════════════════════════════

def extract_source2(n_target=50):
    """从 rare_color_samples.txt 中抽取多色样本，验证SAM3 mask存在"""
    rare_file = BASE / "extradata/upar-dataset/rare_color_samples.txt"
    upar_data_root = BASE / "extradata/upar-dataset/data"
    upar_rare_anno_dir = BASE / "data/UPAR_rare_color/annotations"
    upar_rare_img_dir = BASE / "data/UPAR_rare_color/images"

    candidates = []
    with open(rare_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 40:
                continue

            # 检测格式: train有image_path前缀, val_task2没有
            if '.jpg' in parts[0] or '.png' in parts[0]:
                offset = 1
                img_rel = parts[0]
            else:
                offset = 0
                img_rel = None  # 跳过val_task2(无直接路径)

            if img_rel is None:
                continue

            upper_start = 9 + offset
            lower_start = 22 + offset

            upper_colors = [COLOR_NAMES[i] for i in range(12)
                           if upper_start + i < len(parts) and int(parts[upper_start + i]) == 1]
            lower_colors = [COLOR_NAMES[i] for i in range(12)
                           if lower_start + i < len(parts) and int(parts[lower_start + i]) == 1]

            is_upper_multi = len(upper_colors) >= 2
            is_lower_multi = len(lower_colors) >= 2
            if not (is_upper_multi or is_lower_multi):
                continue

            # 找图像路径
            img_path = upar_data_root / img_rel

            # 找SAM3标注: data/UPAR_rare_color/annotations/upar_<stem>_p000.json
            stem = Path(img_rel).stem
            anno_files = list(upar_rare_anno_dir.glob(f'*{stem}*.json'))

            mask_path = str(anno_files[0]) if anno_files else None

            # 也检查 UPAR_rare_color/images/ 下是否有对应图像
            if not img_path.exists():
                alt_img = upar_rare_img_dir / img_rel
                if alt_img.exists():
                    img_path = alt_img

            candidates.append({
                'img_path': str(img_path),
                'mask_path': mask_path,
                'upper_colors': upper_colors,
                'lower_colors': lower_colors,
                'is_upper_multi': is_upper_multi,
                'is_lower_multi': is_lower_multi,
                'source': 'upar_rare_color',
            })

    # 只保留有mask的
    has_mask = [c for c in candidates if c['mask_path']]
    print(f"[来源2] UPAR rare_color: {len(candidates)} 候选, {len(has_mask)} 有SAM3 mask")

    random.shuffle(has_mask)
    # 按颜色组合分层
    combo_groups = defaultdict(list)
    for c in has_mask:
        key = '+'.join(sorted(c['upper_colors'])) + '|' + '+'.join(sorted(c['lower_colors']))
        combo_groups[key].append(c)

    sampled = []
    for key, group in sorted(combo_groups.items(), key=lambda x: -len(x[1])):
        sampled.append(group[0])
        if len(sampled) >= n_target:
            break

    # 不够的话从剩余中补
    if len(sampled) < n_target:
        used = set(id(s) for s in sampled)
        remaining = [c for c in has_mask if id(c) not in used]
        sampled.extend(remaining[:n_target - len(sampled)])

    sampled = sampled[:n_target]
    print(f"  → 抽取 {len(sampled)} 张 ({len(combo_groups)} 种颜色组合)")
    return sampled


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════

def main():
    random.seed(42)

    print("=" * 60)
    print("构建杂色检测验证集 v2 (全部带mask)")
    print("=" * 60)

    s1 = extract_source1(n_target=200)
    s2 = extract_source2(n_target=50)

    all_sources = {
        'source1_qwen': s1,
        'source2_upar_rare': s2,
    }

    # 清理并重建
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_copied = 0
    total_missing = 0

    for subdir, samples in all_sources.items():
        sub_out = OUT_DIR / subdir
        sub_out.mkdir(exist_ok=True)

        for i, s in enumerate(samples):
            src_img = Path(s['img_path'])
            dst_name = f"{i+1:04d}_{src_img.stem}{src_img.suffix}"
            dst = sub_out / dst_name

            entry = {
                'id': f"{subdir}/{dst_name}",
                'source': s['source'],
                'img_orig': str(src_img),
                'mask_path': s['mask_path'],
                'upper_colors': s['upper_colors'],
                'lower_colors': s['lower_colors'],
                'is_upper_multi': s['is_upper_multi'],
                'is_lower_multi': s['is_lower_multi'],
            }

            if src_img.exists():
                shutil.copy2(src_img, dst)
                entry['copied'] = True
                total_copied += 1
            else:
                entry['copied'] = False
                total_missing += 1
                print(f"  ⚠ 图像缺失: {src_img}")

            # 验证mask文件存在
            if s['mask_path'] and not Path(s['mask_path']).exists():
                print(f"  ⚠ mask缺失: {s['mask_path']}")

            manifest.append(entry)

    # 保存manifest
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 统计
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"  总计: {len(manifest)}")
    print(f"  复制成功: {total_copied}")
    print(f"  缺失: {total_missing}")
    for subdir in ['source1_qwen', 'source2_upar_rare']:
        n = sum(1 for m in manifest if m['id'].startswith(subdir))
        with_mask = sum(1 for m in manifest if m['id'].startswith(subdir) and m['mask_path'])
        print(f"  {subdir}: {n} 张, {with_mask} 有mask路径")
    print(f"  manifest: {manifest_path}")

    # 颜色组合预览
    print(f"\n--- 组合分布预览 ---")
    for subdir in ['source1_qwen', 'source2_upar_rare']:
        print(f"  [{subdir}]")
        combos = defaultdict(int)
        for m in manifest:
            if m['id'].startswith(subdir):
                key = f"U:{'+'.join(sorted(m['upper_colors']))} L:{'+'.join(sorted(m['lower_colors']))}"
                combos[key] += 1
        for combo, cnt in sorted(combos.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cnt:3d}  {combo}")


if __name__ == '__main__':
    main()
