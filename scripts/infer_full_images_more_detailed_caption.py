#!/usr/bin/env python3
"""Run MORE_DETAILED_CAPTION inference for all full images and summarize token counts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "ugipc_1231_15words_epoch3_full_handoff" / "checkpoint"
DEFAULT_IMAGE_ROOT = REPO_ROOT.parent / "florence-data" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "ugipc_1231_15words_epoch3_full_handoff_more_detailed_caption"


def clean_text(text: str) -> str:
    cleaned = str(text or "").strip()
    for token in ("<s>", "</s>", "<pad>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip().strip('"').strip("'").rstrip(".").strip()


def iter_images(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_done_paths(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = record.get("relative_path")
            if rel_path and not record.get("error"):
                done.add(str(rel_path))
    return done


def load_model(checkpoint: Path, device: str, dtype_name: str):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    processor = AutoProcessor.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def generate_caption(
    processor,
    model,
    image: Image.Image,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[str, str, int]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = processor(
        text=TASK_PROMPT,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = clean_text(raw)
    token_count = len(processor.tokenizer(text, add_special_tokens=False).input_ids)
    return raw, text, token_count


def percentile(sorted_values: list[int], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if not record.get("error")]
    error_records = [record for record in records if record.get("error")]
    counts = sorted(int(record["token_count"]) for record in ok_records)
    histogram = Counter(counts)

    if counts:
        stats: dict[str, Any] = {
            "count": len(counts),
            "min": min(counts),
            "max": max(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "p10": percentile(counts, 0.10),
            "p25": percentile(counts, 0.25),
            "p75": percentile(counts, 0.75),
            "p90": percentile(counts, 0.90),
            "p95": percentile(counts, 0.95),
            "p99": percentile(counts, 0.99),
        }
    else:
        stats = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    return {
        "task": "more_detailed_caption",
        "prompt": TASK_PROMPT,
        "total_records": len(records),
        "successful_records": len(ok_records),
        "error_records": len(error_records),
        "token_count_stats": stats,
        "token_count_histogram": [
            {"token_count": token_count, "image_count": image_count}
            for token_count, image_count in sorted(histogram.items())
        ],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "relative_path",
        "image_path",
        "width",
        "height",
        "token_count",
        "text",
        "error",
        "seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_histogram_csv(summary: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_count", "image_count"])
        writer.writeheader()
        for row in summary["token_count_histogram"]:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests")
    parser.add_argument("--resume", action="store_true", help="Skip successful records already in outputs.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "outputs.jsonl"
    csv_path = args.output_dir / "outputs.csv"
    summary_path = args.output_dir / "token_distribution.json"
    histogram_csv_path = args.output_dir / "token_distribution_histogram.csv"

    image_paths = iter_images(args.image_root)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    done_paths = load_done_paths(jsonl_path) if args.resume else set()
    remaining_paths = [
        path for path in image_paths if str(path.relative_to(args.image_root)) not in done_paths
    ]

    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"image_root: {args.image_root}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"images_total: {len(image_paths)}", flush=True)
    print(f"images_done: {len(done_paths)}", flush=True)
    print(f"images_remaining: {len(remaining_paths)}", flush=True)

    processor, model = load_model(args.checkpoint, args.device, args.dtype)

    mode = "a" if args.resume else "w"
    started_at = time.time()
    with jsonl_path.open(mode, encoding="utf-8") as handle:
        for index, image_path in enumerate(remaining_paths, start=1):
            relative_path = str(image_path.relative_to(args.image_root))
            item_started_at = time.time()
            record: dict[str, Any] = {
                "checkpoint": str(args.checkpoint),
                "image_root": str(args.image_root),
                "image_path": str(image_path),
                "relative_path": relative_path,
                "task": "more_detailed_caption",
                "prompt": TASK_PROMPT,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            }
            try:
                image = Image.open(image_path).convert("RGB")
                record["width"], record["height"] = image.size
                raw, text, token_count = generate_caption(
                    processor,
                    model,
                    image,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                record["raw"] = raw
                record["text"] = text
                record["token_count"] = token_count
                record["error"] = None
            except Exception as exc:  # Keep long runs moving and record the failing image.
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = round(time.time() - item_started_at, 4)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if index == 1 or index % 50 == 0 or index == len(remaining_paths):
                elapsed = time.time() - started_at
                rate = index / elapsed if elapsed else 0.0
                print(
                    f"processed {index}/{len(remaining_paths)} "
                    f"({len(done_paths) + index}/{len(image_paths)} total), "
                    f"{rate:.2f} img/s, last_tokens={record.get('token_count')}, "
                    f"last={relative_path}",
                    flush=True,
                )

    records = read_jsonl(jsonl_path)
    if args.limit is not None:
        selected = {str(path.relative_to(args.image_root)) for path in image_paths}
        records = [record for record in records if record.get("relative_path") in selected]

    summary = summarize(records)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "image_root": str(args.image_root),
            "output_jsonl": str(jsonl_path),
            "output_csv": str(csv_path),
            "histogram_csv": str(histogram_csv_path),
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "dtype": args.dtype,
            "device": args.device,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(records, csv_path)
    write_histogram_csv(summary, histogram_csv_path)
    print(json.dumps(summary["token_count_stats"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
