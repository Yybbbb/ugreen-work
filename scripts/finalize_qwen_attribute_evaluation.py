#!/usr/bin/env python3
"""Retry unresolved Qwen rows, publish a complete extraction, and score it."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluate_qwen_attribute_extraction import evaluate, iter_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_success(row: Mapping[str, Any]) -> bool:
    return row.get("error") is None and isinstance(row.get("attributes"), dict)


def successful_rows_by_id(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    successful: Dict[str, Dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        row_id = str(row.get("id") or row.get("sample_id") or "")
        if row_id and _is_success(row):
            successful[row_id] = row
    return successful


def retry_input_rows(
    predictions: Sequence[Mapping[str, Any]],
    extraction_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    completed = successful_rows_by_id(extraction_rows)
    return [dict(row) for row in predictions if str(row.get("sample_id")) not in completed]


def merge_successful_rows(
    predictions: Sequence[Mapping[str, Any]],
    extraction_rows: Iterable[Mapping[str, Any]],
    retry_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    completed = successful_rows_by_id(extraction_rows)
    completed.update(successful_rows_by_id(retry_rows))
    ordered: List[Dict[str, Any]] = []
    for prediction in predictions:
        sample_id = str(prediction.get("sample_id") or "")
        if sample_id not in completed:
            raise ValueError(f"missing successful Qwen extraction for sample_id={sample_id}")
        ordered.append(completed[sample_id])
    return ordered


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    predictions = list(iter_jsonl(args.predictions))
    extraction = list(iter_jsonl(args.extraction))
    if len(predictions) != args.expected_samples:
        raise ValueError(
            f"prediction count is {len(predictions)}, expected {args.expected_samples}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retry_input = retry_input_rows(predictions, extraction)
    retry_input_path = args.output_dir / "qwen_retry_input.jsonl"
    retry_output_path = args.output_dir / "qwen_retry_output.jsonl"
    _write_jsonl_atomic(retry_input_path, retry_input)
    if retry_input:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "extract_qwen_attributes.py"),
            "--input", str(retry_input_path),
            "--output", str(retry_output_path),
            "--base-url", args.base_url,
            "--model", args.model,
            "--batch-size", "1",
            "--workers", "4",
            "--timeout", "180",
            "--retries", "2",
            "--max-tokens", "1024",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    retry_rows = list(iter_jsonl(retry_output_path)) if retry_output_path.exists() else []
    final_rows = merge_successful_rows(predictions, extraction, retry_rows)
    if len(final_rows) != args.expected_samples:
        raise ValueError(
            f"final extraction count is {len(final_rows)}, expected {args.expected_samples}"
        )
    _write_jsonl_atomic(args.output_dir / "qwen_attribute_extraction_final.jsonl", final_rows)
    _write_jsonl_atomic(args.output_dir / "qwen_attributes.dedup.jsonl", final_rows)
    metrics = evaluate(final_rows, len(extraction) + len(retry_rows))
    if metrics["samples"] != args.expected_samples or metrics["extractor_failures"] != 0:
        raise ValueError(f"invalid final metrics: {metrics}")
    metrics["initial_extraction_rows"] = len(extraction)
    metrics["retried_samples"] = len(retry_input)
    _write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:6097/v1")
    parser.add_argument("--model", default="Qwen36-35b-caption")
    parser.add_argument("--expected-samples", type=int, default=4328)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
