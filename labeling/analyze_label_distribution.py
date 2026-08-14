#!/usr/bin/env python3
"""
统计标注结果中 upper 和 lower 的标签分布。
每张图（每个 JSON 文件）的 upper / lower 只取置信度最高的那一个标签。
"""

import json
import os
import glob
from collections import Counter

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "annotation")


def main():
    json_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.json"))
    total = len(json_files)
    print(f"共找到 {total} 个标注文件\n")

    upper_counter = Counter()
    lower_counter = Counter()
    skipped = 0
    multi_color_count = {"upper": 0, "lower": 0}

    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            skipped += 1
            continue

        parsed = data.get("parsed")
        if not parsed:
            skipped += 1
            continue

        # Process upper
        upper_entries = parsed.get("upper", [])
        if upper_entries:
            if len(upper_entries) > 1:
                multi_color_count["upper"] += 1
            # 取置信度最高的
            best_upper = max(upper_entries, key=lambda x: x.get("confidence", -1))
            upper_counter[best_upper.get("label", "?")] += 1

        # Process lower
        lower_entries = parsed.get("lower", [])
        if lower_entries:
            if len(lower_entries) > 1:
                multi_color_count["lower"] += 1
            # 取置信度最高的
            best_lower = max(lower_entries, key=lambda x: x.get("confidence", -1))
            lower_counter[best_lower.get("label", "?")] += 1

    # --- 输出结果 ---
    print(f"有效文件数: {total - skipped}  (跳过: {skipped})")
    print(f"多色标注 (upper): {multi_color_count['upper']} 张")
    print(f"多色标注 (lower): {multi_color_count['lower']} 张")
    print()

    # Upper 分布
    print("=" * 60)
    print(f"{'Upper 标签分布':^60}")
    print("=" * 60)
    upper_total = sum(upper_counter.values())
    print(f"{'标签':<16} {'数量':>8} {'占比':>10}")
    print("-" * 36)
    for label, count in upper_counter.most_common():
        pct = count / upper_total * 100 if upper_total > 0 else 0
        print(f"{label:<16} {count:>8} {pct:>9.2f}%")
    print("-" * 36)
    print(f"{'合计':<16} {upper_total:>8}")
    print()

    # Lower 分布
    print("=" * 60)
    print(f"{'Lower 标签分布':^60}")
    print("=" * 60)
    lower_total = sum(lower_counter.values())
    print(f"{'标签':<16} {'数量':>8} {'占比':>10}")
    print("-" * 36)
    for label, count in lower_counter.most_common():
        pct = count / lower_total * 100 if lower_total > 0 else 0
        print(f"{label:<16} {count:>8} {pct:>9.2f}%")
    print("-" * 36)
    print(f"{'合计':<16} {lower_total:>8}")
    print()

    # 合并总览
    print("=" * 60)
    print(f"{'合并总览':^60}")
    print("=" * 60)
    all_labels = sorted(set(list(upper_counter.keys()) + list(lower_counter.keys())))
    print(f"{'标签':<16} {'Upper':>8} {'Lower':>8} {'总计':>8}")
    print("-" * 44)
    grand_upper = sum(upper_counter.values())
    grand_lower = sum(lower_counter.values())
    for label in all_labels:
        u = upper_counter.get(label, 0)
        l = lower_counter.get(label, 0)
        print(f"{label:<16} {u:>8} {l:>8} {u+l:>8}")
    print("-" * 44)
    print(f"{'合计':<16} {grand_upper:>8} {grand_lower:>8} {grand_upper+grand_lower:>8}")


if __name__ == "__main__":
    main()
