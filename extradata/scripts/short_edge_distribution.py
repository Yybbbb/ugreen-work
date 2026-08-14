#!/usr/bin/env python3
"""
统计 object_detection_0309-0429 下所有照片短边长度分布。

短边 = min(width, height)，即图片的较短一边的像素数。
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

# 分桶区间（可根据需要调整）
BUCKETS = [
    (0, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 1200),
    (1200, 1400),
    (1400, 1600),
    (1600, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 4000),
    (4000, 6000),
]


def collect_short_edges(root: Path) -> List[int]:
    """遍历所有图片，返回短边长度列表。"""
    short_edges = []
    total_files = 0
    errors = 0

    for filepath in root.rglob("*"):
        # 先检查扩展名，避免对非图片文件做 stat
        if filepath.suffix.lower() not in IMAGE_EXTS:
            continue

        total_files += 1
        try:
            with Image.open(filepath) as img:
                w, h = img.size
                short_edges.append(min(w, h))
        except Exception as e:
            errors += 1
            print(f"[WARN] 无法读取 {filepath}: {e}", file=sys.stderr)

        # 进度提示
        if total_files % 2000 == 0:
            print(f"  已处理 {total_files} 个文件...")

    print(f"\n总图片数: {total_files}, 成功读取: {len(short_edges)}, 失败: {errors}")
    return short_edges


def bucketize(values: List[int]) -> Dict[Tuple[int, int], int]:
    """将值分桶统计。"""
    counts = Counter()
    for v in values:
        bucket = None
        for lo, hi in BUCKETS:
            if lo < v <= hi:
                bucket = (lo, hi)
                break
        if bucket is None:
            bucket = ("other",)  # 超出所有区间
        counts[bucket] += 1
    return counts


def compute_exact_counts(values: List[int]) -> Counter:
    """精确统计每个短边长度的数量（用于找众数等）。"""
    return Counter(values)


def main():
    if not ROOT_DIR.exists():
        print(f"[ERROR] 目录不存在: {ROOT_DIR}")
        sys.exit(1)

    print(f"扫描目录: {ROOT_DIR}")
    print(f"分桶区间: {BUCKETS}\n")

    # 1. 收集所有短边长度
    short_edges = collect_short_edges(ROOT_DIR)
    if not short_edges:
        print("未找到任何图片。")
        return

    # 2. 统计
    total = len(short_edges)
    sorted_vals = sorted(short_edges)
    mean_val = sum(sorted_vals) / total
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # 精确计数（众数等）
    exact = compute_exact_counts(sorted_vals)
    most_common = exact.most_common(10)

    # 中位数
    if total % 2 == 1:
        median_val = sorted_vals[total // 2]
    else:
        median_val = (sorted_vals[total // 2 - 1] + sorted_vals[total // 2]) / 2

    # 分桶分布
    bucket_dist = bucketize(sorted_vals)

    # 3. 输出
    print("=" * 65)
    print("                    短边长度分布报告")
    print("=" * 65)
    print(f"  图片总数:       {total:,}")
    print(f"  最小值:         {min_val} px")
    print(f"  最大值:         {max_val} px")
    print(f"  平均值:         {mean_val:.1f} px")
    print(f"  中位数:         {median_val:.0f} px")
    print()

    # 最常见短边值
    print(f"  最常见的短边值 (Top 10):")
    print(f"  {'短边(px)':<16}{'数量':>8}  {'占比'}")
    print(f"  {'-' * 40}")
    for edge, cnt in most_common:
        print(f"  {edge:<16}{cnt:>8,}  {cnt/total*100:5.1f}%")
    print()

    # 分桶分布
    print(f"  分桶分布:")
    print(f"  {'区间(px)':<20}{'数量':>8}  {'占比':>8}  {'柱状图'}")
    print(f"  {'-' * 65}")
    bar_max_width = 30
    max_count = max(bucket_dist.values())
    for (lo, hi) in BUCKETS:
        cnt = bucket_dist.get((lo, hi), 0)
        pct = cnt / total * 100
        bar_len = int(cnt / max_count * bar_max_width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {lo:>5} - {hi:<5} px   {cnt:>8,}  {pct:>7.2f}%  {bar}")

    # 如果有超出区间的
    if ("other",) in bucket_dist:
        cnt = bucket_dist[("other",)]
        pct = cnt / total * 100
        print(f"  {'超出范围':<16}     {cnt:>8,}  {pct:>7.2f}%")

    print()

    # 百分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"  百分位数:")
    print(f"  {'百分位':<12}{'短边(px)'}")
    print(f"  {'-' * 25}")
    for p in percentiles:
        idx = int(total * p / 100)
        idx = min(idx, total - 1)
        print(f"  P{p:<11}{sorted_vals[idx]}")

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
