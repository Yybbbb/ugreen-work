#!/usr/bin/env python3
"""
构建纯色验证集 —— 全部带mask。

来源1 (200): Qwen标注 — 单色+高置信度(≥0.90)，有mask
来源2 (50):  UPAR rare_color — 单色标签，有SAM3 mask

输出:
  val_pure/
    source1_qwen/         (200)
    source2_upar_rare/    (50)
    manifest.json
"""

import json, random, shutil
from pathlib import Path
from collections import defaultdict

BASE = Path("/data1/work/MichaelYu/segment-color")
OUT_DIR = BASE / "refuse-answer/multicolor/val_pure"

COLOR_NAMES = ["Black","Blue","Brown","Green","Grey","Orange","Pink","Purple","Red","White","Yellow","Other"]


def extract_qwen_pure(n_target=200):
    """Qwen单色+高置信标注"""
    anno_dir = BASE / "labeling/annotation"
    candidates = []

    for af in anno_dir.glob("*.json"):
        try:
            with open(af) as f:
                data = json.load(f)
        except Exception:
            continue

        parsed = data.get('parsed', {})
        if not parsed:
            continue

        upper_entries = [e for e in parsed.get('upper', [])
                         if e.get('confidence', 0) >= 0.90 and e.get('label', '') != 'unknown']
        lower_entries = [e for e in parsed.get('lower', [])
                         if e.get('confidence', 0) >= 0.90 and e.get('label', '') != 'unknown']

        # 必须是单色（上下各一个颜色）
        is_upper_pure = len(upper_entries) == 1
        is_lower_pure = len(lower_entries) == 1

        if not (is_upper_pure and is_lower_pure):
            continue

        # 验证mask和图像路径
        mask_path = data.get('annotation_path', '')
        img_path = data.get('image_path', '')
        if not mask_path or not Path(mask_path).exists():
            continue
        if not img_path or not Path(img_path).exists():
            continue

        candidates.append({
            'img_path': img_path,
            'mask_path': mask_path,
            'upper_color': upper_entries[0]['label'],
            'lower_color': lower_entries[0]['label'],
            'upper_conf': upper_entries[0]['confidence'],
            'lower_conf': lower_entries[0]['confidence'],
            'source': f'qwen:{af.name}',
        })

    print(f"[来源1] Qwen 纯色: {len(candidates)} 候选")

    # 按颜色组合做分层抽样
    random.shuffle(candidates)
    combo_groups = defaultdict(list)
    for c in candidates:
        key = f"{c['upper_color']}|{c['lower_color']}"
        combo_groups[key].append(c)

    # 优先覆盖更多颜色组合，不够则从大组合中补
    sampled = []
    for key, group in sorted(combo_groups.items(), key=lambda x: -len(x[1])):
        sampled.append(group[0])
        if len(sampled) >= n_target:
            break

    # 不够: 从已用组合中补充第二个样本
    if len(sampled) < n_target:
        for key, group in sorted(combo_groups.items(), key=lambda x: -len(x[1])):
            if len(group) > 1:
                sampled.append(group[1])
            if len(sampled) >= n_target:
                break

    print(f"  → 抽取 {len(sampled)} 张 ({len(combo_groups)} 种颜色组合)")
    return sampled


def extract_upar_pure(n_target=50):
    """UPAR rare_color单色标签"""
    rare_file = BASE / "extradata/upar-dataset/rare_color_samples.txt"
    upar_data_root = BASE / "extradata/upar-dataset/data"
    upar_rare_anno_dir = BASE / "data/UPAR_rare_color/annotations"

    candidates = []
    with open(rare_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 40:
                continue

            if '.jpg' in parts[0] or '.png' in parts[0]:
                offset = 1
                img_rel = parts[0]
            else:
                continue

            upper_start = 9 + offset
            lower_start = 22 + offset

            upper_colors = [COLOR_NAMES[i] for i in range(12)
                           if upper_start + i < len(parts) and int(parts[upper_start + i]) == 1]
            lower_colors = [COLOR_NAMES[i] for i in range(12)
                           if lower_start + i < len(parts) and int(parts[lower_start + i]) == 1]

            # 必须都是单色
            if len(upper_colors) != 1 or len(lower_colors) != 1:
                continue

            img_path = upar_data_root / img_rel
            stem = Path(img_rel).stem
            anno_files = list(upar_rare_anno_dir.glob(f'*{stem}*.json'))
            mask_path = str(anno_files[0]) if anno_files else None

            if not mask_path:
                continue

            candidates.append({
                'img_path': str(img_path),
                'mask_path': mask_path,
                'upper_color': upper_colors[0],
                'lower_color': lower_colors[0],
                'source': 'upar_rare_color',
            })

    print(f"[来源2] UPAR rare_color 纯色: {len(candidates)} 候选")

    random.shuffle(candidates)
    # 分层
    combo_groups = defaultdict(list)
    for c in candidates:
        key = f"{c['upper_color']}|{c['lower_color']}"
        combo_groups[key].append(c)

    sampled = []
    for key, group in sorted(combo_groups.items(), key=lambda x: -len(x[1])):
        sampled.append(group[0])
        if len(sampled) >= n_target:
            break

    print(f"  → 抽取 {len(sampled)} 张 ({len(combo_groups)} 种颜色组合)")
    return sampled


def main():
    random.seed(42)

    print("=" * 60)
    print("构建纯色验证集 (全部带mask)")
    print("=" * 60)

    s1 = extract_qwen_pure(n_target=200)
    s2 = extract_upar_pure(n_target=50)

    all_sources = {'source1_qwen': s1, 'source2_upar_rare': s2}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
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
                'upper_color': s.get('upper_color', s.get('upper_colors', [''])[0]),
                'lower_color': s.get('lower_color', s.get('lower_colors', [''])[0]),
                'copied': False,
            }

            if src_img.exists():
                shutil.copy2(src_img, dst)
                entry['copied'] = True

            manifest.append(entry)

    copied = sum(1 for m in manifest if m['copied'])
    print(f"\n完成! 总计: {len(manifest)}, 复制: {copied}")

    with open(OUT_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == '__main__':
    main()
