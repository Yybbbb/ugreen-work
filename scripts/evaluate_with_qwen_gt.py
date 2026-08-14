#!/usr/bin/env python3
"""Recompute a run's test metrics using Qwen-extracted GT-caption attributes as the GT.

Normally ``evaluate_qwen_attribute_extraction.evaluate`` compares the Qwen
extraction of the Florence **prediction** (``attributes``) against the
hand-labelled GT (``ground_truth``). This script replaces the GT side with the
attributes Qwen extracted from the **GT caption** (``qwen_attributes`` in
``data/prepared/test_qwen_attributes.jsonl``), so both prediction and GT pass
through the same Qwen extractor — removing vocabulary/labelling bias.

The prediction side (``attributes``) and the caption stats are taken unchanged
from the run's existing ``qwen_attributes.dedup.jsonl``; only the GT is swapped.

Run:
    python scripts/evaluate_with_qwen_gt.py \
        --run-dir artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/test_inference_final/evaluation_qwen_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from evaluate_qwen_attribute_extraction import evaluate, iter_jsonl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_FILE = PROJECT_ROOT / "data" / "prepared" / "test_qwen_attributes.jsonl"


def load_qwen_gt(path: Path, field: str) -> Dict[str, Dict[str, Any]]:
    gt: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sid = str(row.get("sample_id") or row.get("id") or "")
        attrs = row.get(field)
        if sid and isinstance(attrs, dict):
            gt[sid] = attrs
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="evaluation_qwen_final dir containing qwen_attributes.dedup.jsonl")
    parser.add_argument("--gt-file", type=Path, default=DEFAULT_GT_FILE)
    parser.add_argument("--gt-field", default="qwen_attributes")
    parser.add_argument("--input", type=Path, default=None,
                        help="dedup jsonl (default <run-dir>/qwen_attributes.dedup.jsonl)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output metrics (default <run-dir>/metrics_qwen_gt.json)")
    args = parser.parse_args()

    dedup_path = args.input or (args.run_dir / "qwen_attributes.dedup.jsonl")
    out_path = args.output or (args.run_dir / "metrics_qwen_gt.json")
    qwen_gt = load_qwen_gt(args.gt_file, args.gt_field)

    source_rows = list(iter_jsonl(dedup_path))
    rows: List[Mapping[str, Any]] = []
    missing_gt = 0
    for row in source_rows:
        sid = str(row.get("id") or row.get("sample_id") or "")
        gt = qwen_gt.get(sid)
        pred = row.get("attributes")
        if not isinstance(gt, dict):
            missing_gt += 1
            continue
        merged = dict(row)
        merged["ground_truth"] = gt  # swap GT to Qwen-from-GT-caption
        # ensure pred + caption carry through
        merged["attributes"] = pred
        merged["prediction"] = row.get("prediction", "")
        rows.append(merged)

    metrics = evaluate(rows, len(source_rows))
    metrics["gt_basis"] = "qwen_extracted_from_gt_caption"
    metrics["gt_file"] = str(args.gt_file)
    metrics["gt_field"] = args.gt_field
    metrics["rows_used"] = len(rows)
    metrics["rows_missing_gt"] = missing_gt

    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "micro_f1": metrics["micro"]["f1"],
        "soft_micro_f1": metrics["soft_micro"]["f1"],
        "macro_field_f1": metrics["macro_field_f1"],
        "mean_field_exact": metrics["mean_field_exact"],
        "unknown_extra_rate": metrics["unknown_extra_rate"],
        "rows_used": metrics["rows_used"],
        "rows_missing_gt": metrics["rows_missing_gt"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
