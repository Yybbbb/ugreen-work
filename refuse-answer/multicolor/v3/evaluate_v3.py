#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent / "v2"
for path in [HERE, V2_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detector_v3 import detect_region
from image_io import load_rgb_image
from mask_io import load_upper_mask_for_image


DEFAULT_PURE_DIR = HERE.parent / "val_pure"
DEFAULT_MULTI_DIR = HERE.parent / "val_multicolor" / "upper_multicolor"
DEFAULT_OUTPUT = HERE / "eval_val_pure_multicolor.json"


def evaluate(pure_dir=DEFAULT_PURE_DIR, multi_dir=DEFAULT_MULTI_DIR, output=DEFAULT_OUTPUT):
    sets = [
        ("multi", Path(multi_dir), False),
        ("pure", Path(pure_dir), True),
    ]
    results = []
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    errors = {"multi": 0, "pure": 0}
    processed = {"multi": 0, "pure": 0}
    total = {"multi": 0, "pure": 0}
    start = time.time()

    for label, root, gt_pure in sets:
        files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        total[label] = len(files)
        for path in files:
            item = {"id": path.name, "path": str(path), "gt_label": label, "gt_is_pure": gt_pure}
            try:
                image_rgb, width, height = load_rgb_image(path)
                mask_info = load_upper_mask_for_image(path)
                mask = mask_info["mask"]
                if mask_info["width"] != width or mask_info["height"] != height:
                    mask = _resize_mask_nearest(mask, width, height)

                detection = detect_region(image_rgb, mask, 1, "upper")
                pred_pure = bool(detection["is_pure"])
                item["prediction"] = "pure" if pred_pure else "multi"
                item["detection"] = detection
                item["annotation_path"] = mask_info["annotation_path"]
                item["label_path"] = mask_info.get("label_path")
                processed[label] += 1
                if gt_pure and pred_pure:
                    counts["tp"] += 1
                elif gt_pure and not pred_pure:
                    counts["fn"] += 1
                elif (not gt_pure) and pred_pure:
                    counts["fp"] += 1
                else:
                    counts["tn"] += 1
            except Exception as exc:
                errors[label] += 1
                item["error"] = str(exc)
            results.append(item)

    report = _build_report(total, processed, errors, counts, results, time.time() - start)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _build_report(total, processed, errors, counts, results, elapsed):
    n = sum(counts.values())
    accuracy = (counts["tp"] + counts["tn"]) / n if n else 0.0
    pure_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    multi_recall = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else 0.0
    pure_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    multi_precision = counts["tn"] / (counts["tn"] + counts["fn"]) if counts["tn"] + counts["fn"] else 0.0

    return {
        "experiment": "v3_adaptive_lab",
        "sets": {key: {"total": total[key], "processed": processed[key], "errors": errors[key]} for key in ["pure", "multi"]},
        "confusion_matrix": {
            "actual_pure_pred_pure_tp": counts["tp"],
            "actual_pure_pred_multi_fn": counts["fn"],
            "actual_multi_pred_pure_fp": counts["fp"],
            "actual_multi_pred_multi_tn": counts["tn"],
        },
        "metrics_processed_only": {
            "accuracy": round(accuracy, 6),
            "pure_recall": round(pure_recall, 6),
            "multi_recall": round(multi_recall, 6),
            "pure_precision": round(pure_precision, 6),
            "multi_precision": round(multi_precision, 6),
        },
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }


def _resize_mask_nearest(mask, target_width, target_height):
    src_height = len(mask)
    src_width = len(mask[0]) if src_height else 0
    if src_width == 0 or src_height == 0:
        return [[0 for _ in range(target_width)] for _ in range(target_height)]

    resized = []
    for y in range(target_height):
        src_y = min(src_height - 1, int(y * src_height / target_height))
        row = []
        for x in range(target_width):
            src_x = min(src_width - 1, int(x * src_width / target_width))
            row.append(mask[src_y][src_x])
        resized.append(row)
    return resized


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3 adaptive Lab experiment.")
    parser.add_argument("--pure-dir", default=str(DEFAULT_PURE_DIR))
    parser.add_argument("--multi-dir", default=str(DEFAULT_MULTI_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = evaluate(args.pure_dir, args.multi_dir, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
