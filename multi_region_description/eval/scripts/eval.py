#!/usr/bin/env python3
"""
Evaluate the fine-tuned REGIONS_TO_DESCRIPTIONS checkpoint on test.jsonl.

For each sample we measure:
  - count_match: len(pred_descs) == len(gt_descs)  (core requirement of scheme A)
  - exact_match: every predicted description == the ground-truth description
  - token-level overlap (word F1 per region, averaged over the sample)

Reports per-sample results + aggregate summary to stdout as JSON.
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

DEFAULT_CHECKPOINT = str(PROJECT_DIR / "checkpoints/multi_region_ep2_lr1e5/final")
DEFAULT_TEST_FILE = str(
    PROJECT_DIR / "data/ugipc-person-region-descriptions-yolo26m-no-name/test.jsonl"
)
DEFAULT_OUT_FILE = str(RESULTS_DIR / "eval_results.jsonl")
TASK_TOKEN = "<REGIONS_TO_DESCRIPTIONS>"
SEP = "<sep>"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def patch_flash_attn():
    orig = dynamic_module_utils.get_imports
    def patched(fn):
        imps = orig(fn)
        if str(fn).endswith("modeling_florence2.py"):
            imps = [n for n in imps if n != "flash_attn"]
        return imps
    dynamic_module_utils.get_imports = patched


def load_model(checkpoint, device, dtype_name):
    patch_flash_attn()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def word_f1(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1
    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def run_sample(processor, model, device, dtype, sample, max_new_tokens, num_beams):
    image = Image.open(sample["image"]).convert("RGB")
    prompt = sample["prompt"]

    inputs = processor(
        text=prompt, images=image,
        return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )

    # keep special tokens so <sep> survives for the post-processor
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        raw, task=TASK_TOKEN, image_size=image.size
    )
    pred_descs = result[TASK_TOKEN]["descriptions"]
    return raw, pred_descs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUT_FILE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Only eval first N samples (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # load test data
    samples = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    log.info("evaluating %d samples", len(samples))

    # load model
    processor, model = load_model(args.checkpoint, args.device, args.dtype)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # aggregate metrics
    n_count_match = 0
    n_exact_match = 0
    total_word_f1 = 0.0
    total_regions = 0
    n_by_size = {}   # {N: {"count": int, "match": int}}

    out_records = []

    for idx, sample in enumerate(samples):
        gt_label = sample["label"]
        gt_descs = [s.strip() for s in gt_label.split(SEP) if s.strip()]
        n_gt = len(gt_descs)

        try:
            raw, pred_descs = run_sample(processor, model, device, dtype, sample, args.max_new_tokens, args.num_beams)
        except Exception as exc:
            log.warning("sample %d failed: %s", idx, exc)
            record = {"idx": idx, "image": sample["image"], "error": str(exc)}
            out_records.append(record)
            continue

        count_match = len(pred_descs) == n_gt

        # per-region metrics (aligned by position, only when count matches)
        per_region_f1 = []
        if count_match:
            for pred, ref in zip(pred_descs, gt_descs):
                per_region_f1.append(word_f1(pred, ref))

        exact_match = count_match and all(p == g for p, g in zip(pred_descs, gt_descs))
        avg_f1 = sum(per_region_f1) / len(per_region_f1) if per_region_f1 else None

        if count_match:
            n_count_match += 1
            total_word_f1 += sum(per_region_f1)
            total_regions += n_gt
        if exact_match:
            n_exact_match += 1

        bucket = n_by_size.setdefault(n_gt, {"count": 0, "count_match": 0})
        bucket["count"] += 1
        if count_match:
            bucket["count_match"] += 1

        record = {
            "idx": idx,
            "image": sample["image"],
            "n_gt": n_gt,
            "n_pred": len(pred_descs),
            "count_match": count_match,
            "exact_match": exact_match,
            "avg_word_f1": round(avg_f1, 4) if avg_f1 is not None else None,
            "gt_descriptions": gt_descs,
            "pred_descriptions": pred_descs,
            "raw": raw,
        }
        out_records.append(record)

        if (idx + 1) % 50 == 0 or idx == 0:
            log.info(
                "[%d/%d] count_match_rate=%.1f%% avg_word_f1=%.4f",
                idx + 1, len(samples),
                100 * n_count_match / (idx + 1),
                total_word_f1 / max(1, total_regions),
            )

    n_valid = sum(1 for r in out_records if "error" not in r)
    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": len(samples),
        "n_valid": n_valid,
        "count_match_rate": round(n_count_match / max(1, n_valid), 4),
        "exact_match_rate": round(n_exact_match / max(1, n_valid), 4),
        "avg_word_f1_on_matched": round(total_word_f1 / max(1, total_regions), 4),
        "count_match_by_n": {
            str(n): {
                "count": v["count"],
                "count_match_rate": round(v["count_match"] / v["count"], 4),
            }
            for n, v in sorted(n_by_size.items())
        },
    }

    # write per-sample results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("per-sample results written to %s", args.output_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
