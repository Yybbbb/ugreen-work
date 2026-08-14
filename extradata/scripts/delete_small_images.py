#!/usr/bin/env python3
"""
删除 object_detection_0309-0429 下所有短边 < 48px 的图片，
并统计删除前后的数据量变化。
"""

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

# -------- 配置 --------
ROOT_DIR = Path(__file__).resolve().parent.parent / "object_detection_0309-0429"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
THRESHOLD = 48  # 短边阈值（不包含）


def scan_images(root: Path) -> List[Path]:
    """扫描所有图片文件路径。"""
    images = []
    for filepath in root.rglob("*"):
        if filepath.suffix.lower() in IMAGE_EXTS:
            images.append(filepath)
    return images


def get_short_edge(filepath: Path) -> int:
    """获取图片短边长度。"""
    with Image.open(filepath) as img:
        w, h = img.size
        return min(w, h)


def main():
    if not ROOT_DIR.exists():
        print(f"[ERROR] 目录不存在: {ROOT_DIR}")
        sys.exit(1)

    # ---- 1. 扫描所有图片 ----
    print("正在扫描图片...")
    all_images = scan_images(ROOT_DIR)
    total_before = len(all_images)
    print(f"扫描完成，共 {total_before:,} 张图片\n")

    # ---- 2. 统计删除前的分布 ----
    print("正在读取图片尺寸（删除前统计）...")
    short_edges = []
    errors = 0
    for i, fp in enumerate(all_images):
        try:
            short_edges.append(get_short_edge(fp))
        except Exception:
            errors += 1
        if (i + 1) % 3000 == 0:
            print(f"  已处理 {i+1}/{total_before}...")

    total_valid = len(short_edges)
    print(f"  成功读取 {total_valid:,} 张，失败 {errors} 张\n")

    # 删除前：短边各区间数量
    buckets = [(0, 48), (48, 200), (200, 400), (400, 600), (600, 800)]
    before_dist = Counter()
    for v in short_edges:
        for lo, hi in buckets:
            if lo <= v < hi:
                before_dist[(lo, hi)] += 1
                break

    # ---- 3. 找出需删除的图片 ----
    print(f"正在识别短边 < {THRESHOLD} px 的图片...")
    to_delete = []
    for fp in all_images:
        try:
            se = get_short_edge(fp)
            if se < THRESHOLD:
                to_delete.append(fp)
        except Exception:
            pass

    delete_count = len(to_delete)
    print(f"  需删除: {delete_count:,} 张\n")

    if delete_count == 0:
        print("没有需要删除的图片，退出。")
        return

    # 统计各子目录删除数量
    dir_delete_count: Dict[str, int] = Counter()
    for fp in to_delete:
        dir_delete_count[fp.parent.name] += 1

    print("  各子目录将删除数量:")
    for dname in sorted(dir_delete_count.keys()):
        print(f"    {dname}: {dir_delete_count[dname]:,}")

    # ---- 4. 确认删除 ----
    print(f"\n{'='*60}")
    print(f"即将删除 {delete_count:,} 张短边 < {THRESHOLD} px 的图片")
    print(f"删除后预计剩余: {total_before - delete_count:,} 张")
    print(f"{'='*60}")

    resp = input("\n确认删除? (输入 yes 确认): ")
    if resp.strip() != "yes":
        print("已取消。")
        return

    # ---- 5. 执行删除 ----
    print("\n正在删除...")
    deleted = 0
    for i, fp in enumerate(to_delete):
        try:
            os.remove(fp)
            deleted += 1
        except Exception as e:
            print(f"  [WARN] 删除失败 {fp}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  已删除 {i+1}/{delete_count}...")

    print(f"  成功删除: {deleted:,} 张")

    # ---- 6. 删除后统计 ----
    print("\n正在扫描剩余图片...")
    remaining = scan_images(ROOT_DIR)
    total_after = len(remaining)
    print(f"剩余图片: {total_after:,} 张")

    # 读取剩余图片短边
    remaining_edges = []
    for i, fp in enumerate(remaining):
        try:
            remaining_edges.append(get_short_edge(fp))
        except Exception:
            pass
        if (i + 1) % 3000 == 0:
            print(f"  已处理 {i+1}/{total_after}...")

    after_dist = Counter()
    for v in remaining_edges:
        for lo, hi in buckets:
            if lo <= v < hi:
                after_dist[(lo, hi)] += 1
                break

    # ---- 7. 输出对比报告 ----
    print()
    print("=" * 65)
    print("                    删前 / 删后 对比报告")
    print("=" * 65)
    print(f"  删除阈值:         短边 < {THRESHOLD} px")
    print(f"  删除前图片总数:    {total_before:>10,}")
    print(f"  已删除:            {deleted:>10,}")
    print(f"  删除后剩余:        {total_after:>10,}")
    print(f"  删除比例:          {deleted/total_before*100:>9.1f}%")
    print()

    print(f"  {'区间(px)':<18}{'删前数量':>10}{'删后数量':>10}{'变化':>10}")
    print(f"  {'-' * 48}")
    for lo, hi in buckets:
        before_cnt = before_dist.get((lo, hi), 0)
        after_cnt = after_dist.get((lo, hi), 0)
        delta = after_cnt - before_cnt
        delta_str = f"{delta:+,}"
        print(f"  [{lo:>4}, {hi:>4})   {before_cnt:>10,}{after_cnt:>10,}{delta_str:>10}")

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
