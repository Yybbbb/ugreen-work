#!/usr/bin/env python3

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/eval/results/"
    "qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6_fixed.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/eval/results/"
    "description_matching"
)
WORD_RE = re.compile(r"[a-z0-9]+")
STATUSES = (
    "confident_match",
    "ambiguous_match",
    "wrong_bbox_assignment",
    "content_mismatch",
)


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


def classify_region(
    positive_score: float,
    best_other_score: float | None,
    content_threshold: float,
    confident_margin: float,
    wrong_bbox_margin: float,
) -> str:
    if best_other_score is None:
        return (
            "confident_match"
            if positive_score >= content_threshold
            else "content_mismatch"
        )

    if (
        best_other_score >= content_threshold
        and best_other_score - positive_score >= wrong_bbox_margin
    ):
        return "wrong_bbox_assignment"
    if positive_score < content_threshold:
        return "content_mismatch"
    if positive_score - best_other_score >= confident_margin:
        return "confident_match"
    return "ambiguous_match"


def best_permutation(score_matrix: list[list[float]]) -> tuple[float, list[int]]:
    region_count = len(score_matrix)
    identity = list(range(region_count))
    best_assignment = identity
    best_score = sum(
        score_matrix[index][index] for index in range(region_count)
    ) / region_count
    for permutation in itertools.permutations(range(region_count)):
        score = sum(
            score_matrix[pred_index][gt_index]
            for pred_index, gt_index in enumerate(permutation)
        ) / region_count
        if score > best_score + 1e-12:
            best_score = score
            best_assignment = list(permutation)
    return best_score, best_assignment


def analyze_record(record: dict, args) -> dict:
    gt_pairs = record["gt_pairs"]
    pred_pairs = record["pred_pairs"]
    if len(gt_pairs) != len(pred_pairs):
        raise ValueError(
            f"idx={record.get('idx')} has {len(gt_pairs)} GT and "
            f"{len(pred_pairs)} predictions"
        )

    score_matrix = [
        [word_f1(pred["desc"], gt["desc"]) for gt in gt_pairs]
        for pred in pred_pairs
    ]
    region_results = []
    for index, pred in enumerate(pred_pairs):
        positive_score = score_matrix[index][index]
        other_candidates = [
            (score, gt_index)
            for gt_index, score in enumerate(score_matrix[index])
            if gt_index != index
        ]
        if other_candidates:
            best_other_score, best_other_index = max(other_candidates)
        else:
            best_other_score, best_other_index = None, None
        status = classify_region(
            positive_score,
            best_other_score,
            args.content_threshold,
            args.confident_margin,
            args.wrong_bbox_margin,
        )
        region_results.append(
            {
                "region_index": index,
                "bbox": gt_pairs[index]["bbox"],
                "gt_description": gt_pairs[index]["desc"],
                "pred_description": pred["desc"],
                "positive_word_f1": round(positive_score, 6),
                "best_other_gt_index": best_other_index,
                "best_other_word_f1": (
                    round(best_other_score, 6)
                    if best_other_score is not None
                    else None
                ),
                "assignment_margin": (
                    round(positive_score - best_other_score, 6)
                    if best_other_score is not None
                    else None
                ),
                "status": status,
            }
        )

    positional_score = sum(
        score_matrix[index][index] for index in range(len(gt_pairs))
    ) / len(gt_pairs)
    best_score, assignment = best_permutation(score_matrix)
    status_counts = Counter(item["status"] for item in region_results)
    acceptable_statuses = {"confident_match", "ambiguous_match"}

    return {
        "idx": record["idx"],
        "image": record["image"],
        "n_gt": record["n_gt"],
        "n_pred": record["n_pred"],
        "raw": record["raw"],
        "score_matrix": [
            [round(score, 6) for score in row] for row in score_matrix
        ],
        "positional_word_f1": round(positional_score, 6),
        "best_permutation_word_f1": round(best_score, 6),
        "assignment_gap": round(best_score - positional_score, 6),
        "best_permutation": assignment,
        "region_results": region_results,
        "status_counts": {
            status: status_counts.get(status, 0) for status in STATUSES
        },
        "all_regions_acceptable": all(
            item["status"] in acceptable_statuses for item in region_results
        ),
        "all_regions_confident": all(
            item["status"] == "confident_match" for item in region_results
        ),
        "has_wrong_bbox_assignment": status_counts["wrong_bbox_assignment"] > 0,
        "has_content_mismatch": status_counts["content_mismatch"] > 0,
    }


def summarize(records: list[dict], args) -> dict:
    total_regions = sum(record["n_gt"] for record in records)
    all_region_results = [
        region for record in records for region in record["region_results"]
    ]
    status_counts = Counter()
    for record in records:
        status_counts.update(record["status_counts"])

    def region_rate(statuses: set[str]) -> float:
        count = sum(status_counts[status] for status in statuses)
        return count / total_regions if total_regions else 0.0

    def frame_rate(field: str) -> float:
        return (
            sum(bool(record[field]) for record in records) / len(records)
            if records
            else 0.0
        )

    grouped = {}
    for region_count in sorted({record["n_gt"] for record in records}):
        group = [record for record in records if record["n_gt"] == region_count]
        group_regions = sum(record["n_gt"] for record in group)
        group_region_results = [
            region
            for record in group
            for region in record["region_results"]
        ]
        group_status = Counter()
        for record in group:
            group_status.update(record["status_counts"])
        grouped[str(region_count)] = {
            "frames": len(group),
            "regions": group_regions,
            "positive_content_pass_rate": round(
                sum(
                    region["positive_word_f1"] >= args.content_threshold
                    for region in group_region_results
                )
                / group_regions,
                6,
            ),
            "acceptable_region_match_rate": round(
                (
                    group_status["confident_match"]
                    + group_status["ambiguous_match"]
                )
                / group_regions,
                6,
            ),
            "confident_region_match_rate": round(
                group_status["confident_match"] / group_regions, 6
            ),
            "ambiguous_region_rate": round(
                group_status["ambiguous_match"] / group_regions, 6
            ),
            "wrong_bbox_region_rate": round(
                group_status["wrong_bbox_assignment"] / group_regions, 6
            ),
            "content_mismatch_region_rate": round(
                group_status["content_mismatch"] / group_regions, 6
            ),
            "fully_acceptable_frame_rate": round(
                sum(record["all_regions_acceptable"] for record in group)
                / len(group),
                6,
            ),
            "fully_confident_frame_rate": round(
                sum(record["all_regions_confident"] for record in group)
                / len(group),
                6,
            ),
            "average_positional_word_f1": round(
                sum(record["positional_word_f1"] for record in group)
                / len(group),
                6,
            ),
        }

    return {
        "method": {
            "scorer": "lowercased alphanumeric token Word F1",
            "content_threshold": args.content_threshold,
            "confident_margin": args.confident_margin,
            "wrong_bbox_margin": args.wrong_bbox_margin,
            "status_rules": {
                "confident_match": (
                    "positive_f1 >= content_threshold and positive_f1 - "
                    "best_other_f1 >= confident_margin"
                ),
                "ambiguous_match": (
                    "positive_f1 >= content_threshold, but it is not a "
                    "confident match or a clear wrong-bbox assignment"
                ),
                "wrong_bbox_assignment": (
                    "best_other_f1 >= content_threshold and best_other_f1 - "
                    "positive_f1 >= wrong_bbox_margin"
                ),
                "content_mismatch": (
                    "positive_f1 < content_threshold and no other GT is "
                    "clearly better by wrong_bbox_margin"
                ),
            },
        },
        "overall": {
            "frames": len(records),
            "regions": total_regions,
            "status_counts": {
                status: status_counts.get(status, 0) for status in STATUSES
            },
            "positive_content_pass_rate": round(
                sum(
                    region["positive_word_f1"] >= args.content_threshold
                    for region in all_region_results
                )
                / total_regions,
                6,
            ),
            "acceptable_region_match_rate": round(
                region_rate({"confident_match", "ambiguous_match"}), 6
            ),
            "confident_region_match_rate": round(
                region_rate({"confident_match"}), 6
            ),
            "ambiguous_region_rate": round(
                region_rate({"ambiguous_match"}), 6
            ),
            "wrong_bbox_region_rate": round(
                region_rate({"wrong_bbox_assignment"}), 6
            ),
            "content_mismatch_region_rate": round(
                region_rate({"content_mismatch"}), 6
            ),
            "fully_acceptable_frame_rate": round(
                frame_rate("all_regions_acceptable"), 6
            ),
            "fully_confident_frame_rate": round(
                frame_rate("all_regions_confident"), 6
            ),
            "frames_with_wrong_bbox_rate": round(
                frame_rate("has_wrong_bbox_assignment"), 6
            ),
            "frames_with_content_mismatch_rate": round(
                frame_rate("has_content_mismatch"), 6
            ),
            "average_positional_word_f1": round(
                sum(record["positional_word_f1"] for record in records)
                / len(records),
                6,
            ),
            "average_best_permutation_word_f1": round(
                sum(record["best_permutation_word_f1"] for record in records)
                / len(records),
                6,
            ),
        },
        "by_n": grouped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether each prediction matches its corresponding GT"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--content-threshold", type=float, default=0.60)
    parser.add_argument("--confident-margin", type=float, default=0.05)
    parser.add_argument("--wrong-bbox-margin", type=float, default=0.10)
    args = parser.parse_args()

    source_records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    analyzed_records = [analyze_record(record, args) for record in source_records]
    summary = summarize(analyzed_records, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "description_matching_results.jsonl"
    summary_path = args.output_dir / "description_matching_summary.json"
    badcases_path = args.output_dir / "description_matching_badcases.jsonl"

    with details_path.open("w", encoding="utf-8") as file:
        for record in analyzed_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with badcases_path.open("w", encoding="utf-8") as file:
        for record in analyzed_records:
            if not record["all_regions_acceptable"]:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"details={details_path}")
    print(f"summary={summary_path}")
    print(f"badcases={badcases_path}")


if __name__ == "__main__":
    main()
