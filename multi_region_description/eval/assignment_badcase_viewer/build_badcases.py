#!/usr/bin/env python3

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/eval/results/"
    "qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6_fixed.jsonl"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "badcases.json"
WORD_RE = re.compile(r"[a-z0-9]+")


def word_f1(prediction: str, target: str) -> float:
    prediction_words = WORD_RE.findall(prediction.lower())
    target_words = WORD_RE.findall(target.lower())
    if not prediction_words or not target_words:
        return 0.0
    overlap = sum(
        (Counter(prediction_words) & Counter(target_words)).values()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_words)
    recall = overlap / len(target_words)
    return 2 * precision * recall / (precision + recall)


def adjacency_details(first_bbox: list[int], second_bbox: list[int]) -> dict:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox

    overlap_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    overlap_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    overlaps = overlap_width > 0 and overlap_height > 0

    horizontal_gap = max(
        0, max(first_x1, second_x1) - min(first_x2, second_x2)
    )
    vertical_gap = max(
        0, max(first_y1, second_y1) - min(first_y2, second_y2)
    )
    edge_gap = math.hypot(horizontal_gap, vertical_gap)
    first_diagonal = math.hypot(first_x2 - first_x1, first_y2 - first_y1)
    second_diagonal = math.hypot(
        second_x2 - second_x1, second_y2 - second_y1
    )
    average_diagonal = (first_diagonal + second_diagonal) / 2
    gap_ratio = edge_gap / average_diagonal if average_diagonal else float("inf")
    adjacent = overlaps or gap_ratio <= 0.35

    return {
        "adjacent": adjacent,
        "overlaps": overlaps,
        "edge_gap": round(edge_gap, 4),
        "average_diagonal": round(average_diagonal, 4),
        "gap_ratio": round(gap_ratio, 4),
    }


def analyze_record(record: dict, assignment_gap_threshold: float) -> dict | None:
    gt_descriptions = [pair["desc"] for pair in record["gt_pairs"]]
    pred_descriptions = [pair["desc"] for pair in record["pred_pairs"]]
    gt_bboxes = [pair["bbox"] for pair in record["gt_pairs"]]
    region_count = len(gt_descriptions)

    score_matrix = [
        [
            word_f1(pred_description, gt_description)
            for gt_description in gt_descriptions
        ]
        for pred_description in pred_descriptions
    ]
    positional_f1 = sum(
        score_matrix[index][index] for index in range(region_count)
    ) / region_count

    best_permutation = tuple(range(region_count))
    best_permutation_f1 = positional_f1
    for permutation in itertools.permutations(range(region_count)):
        permutation_f1 = sum(
            score_matrix[pred_index][gt_index]
            for pred_index, gt_index in enumerate(permutation)
        ) / region_count
        if permutation_f1 > best_permutation_f1:
            best_permutation_f1 = permutation_f1
            best_permutation = permutation

    assignment_gap = best_permutation_f1 - positional_f1
    if assignment_gap < assignment_gap_threshold:
        return None

    moved_pairs = []
    has_adjacent_moved_pair = False
    for pred_index, reassigned_gt_index in enumerate(best_permutation):
        if pred_index == reassigned_gt_index:
            continue
        adjacency = adjacency_details(
            gt_bboxes[pred_index], gt_bboxes[reassigned_gt_index]
        )
        has_adjacent_moved_pair = (
            has_adjacent_moved_pair or adjacency["adjacent"]
        )
        moved_pairs.append(
            {
                "pred_index": pred_index,
                "original_gt_index": pred_index,
                "reassigned_gt_index": reassigned_gt_index,
                "original_pair_f1": round(
                    score_matrix[pred_index][pred_index], 4
                ),
                "reassigned_pair_f1": round(
                    score_matrix[pred_index][reassigned_gt_index], 4
                ),
                **adjacency,
            }
        )

    if not has_adjacent_moved_pair:
        return None

    category = (
        "adjacent_two_person_swap"
        if region_count == 2
        else "multi_person_position_interference"
    )
    return {
        **record,
        "category": category,
        "positional_f1_recomputed": round(positional_f1, 4),
        "best_permutation_f1": round(best_permutation_f1, 4),
        "assignment_gap": round(assignment_gap, 4),
        "best_permutation": list(best_permutation),
        "moved_pairs": moved_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build strict adjacent assignment badcases"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assignment-gap-threshold", type=float, default=0.10)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    badcases = []
    for record in records:
        badcase = analyze_record(record, args.assignment_gap_threshold)
        if badcase is not None:
            badcases.append(badcase)

    badcases.sort(key=lambda item: (-item["assignment_gap"], item["idx"]))
    category_counts = Counter(item["category"] for item in badcases)
    payload = {
        "method": {
            "assignment_gap_threshold": args.assignment_gap_threshold,
            "adjacency_rule": (
                "两个 bbox 有面积重叠，或 bbox 边缘欧氏距离不超过两框"
                "平均对角线长度的 35%"
            ),
            "assignment_rule": (
                "枚举同图预测描述到 GT bbox 的全部排列；若最优排列 F1 "
                "比原位置对应 F1 至少高 0.10，则认为存在明显分配错位"
            ),
        },
        "summary": {
            "total_test_samples": len(records),
            "total_badcases": len(badcases),
            "badcase_rate": round(len(badcases) / len(records), 6),
            "adjacent_two_person_swap": category_counts[
                "adjacent_two_person_swap"
            ],
            "multi_person_position_interference": category_counts[
                "multi_person_position_interference"
            ],
        },
        "badcases": badcases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
