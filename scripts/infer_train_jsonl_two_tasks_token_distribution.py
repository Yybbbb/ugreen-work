#!/usr/bin/env python3
"""Infer MORE_DETAILED_CAPTION and REGIONS_TO_DESCRIPTIONS for a JSONL image set.

Input rows are expected to contain:
    {"image": "...", "prompt": "<REGIONS_TO_DESCRIPTIONS>...", ...}

For each row/image this script runs:
  1. <MORE_DETAILED_CAPTION> with prompt + full image only.
  2. <REGIONS_TO_DESCRIPTIONS> with the row prompt + full image + region loc tokens.

It writes per-image outputs and token-count distributions for both tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "multi_region_description"

DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "multi_region_plan_b_ep2_lr1e5" / "final"
)
DEFAULT_INPUT_JSONL = (
    PROJECT_DIR
    / "data"
    / "ugipc-person-region-descriptions-yolo26m-no-name-plan-b"
    / "train.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "multi_region_plan_b_ep2_lr1e5_train_two_tasks"
)

MORE_TASK = "<MORE_DETAILED_CAPTION>"
REGIONS_TASK = "<REGIONS_TO_DESCRIPTIONS>"
LOC_VALUE_RE = re.compile(r"<loc_(\d+)>")
STRIP_TOKENS = re.compile(r"<s>|</s>|<pad>")


def patch_flash_attn_import_check() -> None:
    """Ignore optional flash-attn in Florence remote-code import checks."""
    original_get_imports = dynamic_module_utils.get_imports

    def patched_get_imports(filename):
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    dynamic_module_utils.get_imports = patched_get_imports


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("error") and "idx" in record:
                done.add(int(record["idx"]))
    return done


def load_model(checkpoint: Path, device: str, dtype_name: str):
    patch_flash_attn_import_check()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return processor, model, dtype


def clean_text(text: str) -> str:
    return STRIP_TOKENS.sub("", str(text or "")).strip()


def parse_prompt_bboxes(prompt: str) -> list[list[int]]:
    body = prompt[len(REGIONS_TASK) :] if prompt.startswith(REGIONS_TASK) else prompt
    bboxes: list[list[int]] = []
    for segment in body.split("<sep>"):
        vals = [int(v) for v in LOC_VALUE_RE.findall(segment)]
        if len(vals) == 4:
            bboxes.append(vals)
    return bboxes


def parse_plan_b_region_outputs(text: str) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    if cleaned.startswith(REGIONS_TASK):
        cleaned = cleaned[len(REGIONS_TASK) :]
    pattern = re.compile(r"(.*?)((?:<loc_\d+>\s*){4})", re.DOTALL)
    parsed: list[dict[str, Any]] = []
    for match in pattern.finditer(cleaned):
        parsed.append(
            {
                "description": match.group(1).strip(),
                "loc_0_999": [int(v) for v in LOC_VALUE_RE.findall(match.group(2))],
            }
        )
    return parsed


def count_text_tokens(processor, text: str) -> int:
    return len(processor.tokenizer(text, add_special_tokens=False).input_ids)


def generate_text(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    decode_skip_special_tokens: bool,
) -> tuple[str, str, int]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = processor(
        text=prompt,
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

    # Florence-2 remote code returns the generated sequence, not input + output.
    raw_new_text = processor.batch_decode(
        generated_ids, skip_special_tokens=decode_skip_special_tokens
    )[0]
    return raw_new_text, clean_text(raw_new_text), int(generated_ids.shape[1])


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


def summarize(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    ok_records = [record for record in records if not record.get("error")]
    error_records = [record for record in records if record.get("error")]

    def stats_for(field: str) -> dict[str, Any]:
        counts = sorted(int(record[field]) for record in ok_records if record.get(field) is not None)
        if not counts:
            stats: dict[str, Any] = {
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
        else:
            stats = {
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
        histogram = Counter(counts)
        return {
            "stats": stats,
            "histogram": [
                {"token_count": token_count, "image_count": image_count}
                for token_count, image_count in sorted(histogram.items())
            ],
        }

    return {
        "task": task,
        "total_records": len(records),
        "successful_records": len(ok_records),
        "error_records": len(error_records),
        "generated_token_count": stats_for("generated_token_count"),
        "text_token_count": stats_for("text_token_count"),
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
        "idx",
        "image",
        "width",
        "height",
        "task",
        "prompt",
        "generated_token_count",
        "text_token_count",
        "n_regions",
        "output_text",
        "raw_output",
        "error",
        "seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            if isinstance(row.get("n_regions"), list):
                row["n_regions"] = len(row["n_regions"])
            writer.writerow(row)


def write_histogram_csv(distribution: dict[str, Any], path: Path, field: str) -> None:
    rows = distribution[field]["histogram"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_count", "image_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--more-max-new-tokens", type=int, default=120)
    parser.add_argument("--regions-max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def infer_task(
    *,
    rows: list[dict[str, Any]],
    processor,
    model,
    args: argparse.Namespace,
    task: str,
    output_jsonl: Path,
    max_new_tokens: int,
    resume: bool,
) -> None:
    done = load_done_indices(output_jsonl) if resume else set()
    remaining = [(idx, row) for idx, row in enumerate(rows) if idx not in done]
    mode = "a" if resume else "w"
    started_at = time.time()

    print(
        f"{task}: rows_total={len(rows)} done={len(done)} remaining={len(remaining)} "
        f"output={output_jsonl}",
        flush=True,
    )
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        for progress_idx, (idx, row) in enumerate(remaining, start=1):
            item_started_at = time.time()
            prompt = MORE_TASK if task == MORE_TASK else row["prompt"]
            record: dict[str, Any] = {
                "idx": idx,
                "checkpoint": str(args.checkpoint),
                "input_jsonl": str(args.input_jsonl),
                "image": row["image"],
                "task": task,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "num_beams": args.num_beams,
            }
            if task == REGIONS_TASK:
                record["regions_loc_0_999"] = parse_prompt_bboxes(prompt)
                record["n_regions"] = len(record["regions_loc_0_999"])

            try:
                image = Image.open(row["image"]).convert("RGB")
                record["width"], record["height"] = image.size
                raw_output, output_text, generated_token_count = generate_text(
                    processor=processor,
                    model=model,
                    image=image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    num_beams=args.num_beams,
                    decode_skip_special_tokens=(task == MORE_TASK),
                )
                record["raw_output"] = raw_output
                record["output_text"] = output_text
                record["generated_token_count"] = generated_token_count
                record["text_token_count"] = count_text_tokens(processor, output_text)
                if task == REGIONS_TASK:
                    record["parsed_regions"] = parse_plan_b_region_outputs(raw_output)
                    record["n_parsed_regions"] = len(record["parsed_regions"])
                record["error"] = None
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"

            record["seconds"] = round(time.time() - item_started_at, 4)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if progress_idx == 1 or progress_idx % 50 == 0 or progress_idx == len(remaining):
                elapsed = time.time() - started_at
                rate = progress_idx / elapsed if elapsed else 0.0
                print(
                    f"{task}: processed {progress_idx}/{len(remaining)} "
                    f"({len(done) + progress_idx}/{len(rows)} total), "
                    f"{rate:.2f} img/s, last_generated_tokens="
                    f"{record.get('generated_token_count')}, idx={idx}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    more_jsonl = args.output_dir / "more_detailed_caption_outputs.jsonl"
    regions_jsonl = args.output_dir / "regions_to_descriptions_outputs.jsonl"
    more_csv = args.output_dir / "more_detailed_caption_outputs.csv"
    regions_csv = args.output_dir / "regions_to_descriptions_outputs.csv"
    more_distribution_json = args.output_dir / "more_detailed_caption_token_distribution.json"
    regions_distribution_json = args.output_dir / "regions_to_descriptions_token_distribution.json"

    rows = load_rows(args.input_jsonl, args.limit)
    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"input_jsonl: {args.input_jsonl}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"rows: {len(rows)}", flush=True)

    processor, model, _ = load_model(args.checkpoint, args.device, args.dtype)

    infer_task(
        rows=rows,
        processor=processor,
        model=model,
        args=args,
        task=MORE_TASK,
        output_jsonl=more_jsonl,
        max_new_tokens=args.more_max_new_tokens,
        resume=args.resume,
    )
    infer_task(
        rows=rows,
        processor=processor,
        model=model,
        args=args,
        task=REGIONS_TASK,
        output_jsonl=regions_jsonl,
        max_new_tokens=args.regions_max_new_tokens,
        resume=args.resume,
    )

    more_records = read_jsonl(more_jsonl)
    regions_records = read_jsonl(regions_jsonl)
    more_distribution = summarize(more_records, MORE_TASK)
    regions_distribution = summarize(regions_records, REGIONS_TASK)

    common_meta = {
        "checkpoint": str(args.checkpoint),
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "limit": args.limit,
    }
    more_distribution.update(common_meta)
    more_distribution["max_new_tokens"] = args.more_max_new_tokens
    more_distribution["outputs_jsonl"] = str(more_jsonl)
    more_distribution["outputs_csv"] = str(more_csv)
    regions_distribution.update(common_meta)
    regions_distribution["max_new_tokens"] = args.regions_max_new_tokens
    regions_distribution["outputs_jsonl"] = str(regions_jsonl)
    regions_distribution["outputs_csv"] = str(regions_csv)

    more_distribution_json.write_text(
        json.dumps(more_distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    regions_distribution_json.write_text(
        json.dumps(regions_distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(more_records, more_csv)
    write_csv(regions_records, regions_csv)
    write_histogram_csv(
        more_distribution,
        args.output_dir / "more_detailed_caption_generated_token_histogram.csv",
        "generated_token_count",
    )
    write_histogram_csv(
        more_distribution,
        args.output_dir / "more_detailed_caption_text_token_histogram.csv",
        "text_token_count",
    )
    write_histogram_csv(
        regions_distribution,
        args.output_dir / "regions_to_descriptions_generated_token_histogram.csv",
        "generated_token_count",
    )
    write_histogram_csv(
        regions_distribution,
        args.output_dir / "regions_to_descriptions_text_token_histogram.csv",
        "text_token_count",
    )

    print("MORE_DETAILED_CAPTION generated token stats:", flush=True)
    print(json.dumps(more_distribution["generated_token_count"]["stats"], ensure_ascii=False, indent=2), flush=True)
    print("REGIONS_TO_DESCRIPTIONS generated token stats:", flush=True)
    print(json.dumps(regions_distribution["generated_token_count"]["stats"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
