#!/usr/bin/env python3
"""
Evaluate a Plan B fine-tuned REGIONS_TO_DESCRIPTIONS checkpoint on test.jsonl.

Plan B label format:
    description1<loc_x1><loc_y1><loc_x2><loc_y2><sep>description2<loc_x1>...

Parsing rule: the output is a strict alternation of text-segment + 4-loc-run.
One complete (text, loc×4) group = one region. An optional <sep> can delimit
groups; older no-<sep> Plan B labels are still accepted.

Metrics (superset of Plan A eval.py):
  count_match_rate      — len(pred) == len(gt)            (same as Plan A)
  exact_match_rate      — every description matches exactly (same as Plan A)
  avg_word_f1_matched   — word F1 on positionally-matched pairs (count-match only)
  loc_echo_acc          — fraction of predicted bboxes that exactly match GT input bbox
                          (Plan B specific: checks how reliably the model echoes input)
  bbox_f1               — word F1 on IoU-matched pairs (works even on count mismatch)
                          counts per region how often any pred bbox matches a GT bbox
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

DEFAULT_CHECKPOINT = str(PROJECT_DIR / "checkpoints/multi_region_ep2_lr1e5/final")
DEFAULT_TEST_FILE = str(
    PROJECT_DIR / "data/ugipc-person-region-descriptions-yolo26m-no-name-plan-b-sep/test.jsonl"
)
DEFAULT_OUT_FILE = str(RESULTS_DIR / "eval_results_plan_b.jsonl")
TASK_TOKEN = "<REGIONS_TO_DESCRIPTIONS>"

# Matches one complete Plan B region: any text (non-greedy) followed by exactly
# 4 consecutive <loc_NNN> tokens.
#
# Why non-greedy (.*?):
#   We want the SHORTEST possible text before the next loc-run. Without the `?`,
#   the greedy .* would swallow all loc tokens in the middle and only match the
#   very last loc-run in the string, collapsing all regions into one.
#
# Why (?:<loc_\d+>){4}:
#   A loc-run is exactly 4 consecutive <loc_NNN> tokens (no spaces, no other
#   tokens in between). The (?:...) non-capturing group matches one token;
#   {4} requires the group to repeat exactly 4 times consecutively.
#   Using {4} (not {1,4}) ensures partial loc-runs (model glitch: only 2 tokens)
#   are NOT treated as a region boundary — they fall through to the next full run,
#   keeping description segmentation robust.
REGION_PATTERN = re.compile(r"(.*?)((?:<loc_\d+>\s*){4})", re.DOTALL)

# Extracts the numeric value from a single <loc_NNN> token.
LOC_VALUE_RE = re.compile(r"<loc_(\d+)>")

# Tokens to strip from raw decoder output before parsing.
STRIP_TOKENS = re.compile(r"<s>|</s>|<pad>")


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def parse_regions(text: str) -> list[tuple[str, list[int]]]:
    """
    Parse a Plan B text string into a list of (description, [x1,y1,x2,y2]) pairs.

    The function scans left-to-right with re.finditer. Each match consumes:
      group(1) — the description text preceding this loc-run
      group(2) — exactly 4 <loc_NNN> tokens

    Any text after the last loc-run (there should be none in a well-formed output)
    is silently dropped — it would represent a partial/incomplete region.

    Returns [] for empty or completely unparseable input.
    """
    # Remove BOS/EOS/pad artifacts left by the decoder.
    text = STRIP_TOKENS.sub("", text).strip()
    # Also strip the task token prefix if the decoder echoed it.
    if text.startswith(TASK_TOKEN):
        text = text[len(TASK_TOKEN):]

    results = []
    for m in REGION_PATTERN.finditer(text):
        desc = m.group(1).replace("<sep>", " ").strip()
        bbox = [int(v) for v in LOC_VALUE_RE.findall(m.group(2))]
        results.append((desc, bbox))
    return results


def parse_prompt_bboxes(prompt: str) -> list[list[int]]:
    """
    Extract the ordered list of input bboxes from a Plan B prompt.

    Prompt format:
        <REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep><loc_...>...

    Strategy: strip the task token, split on <sep>, then read the 4 loc values
    from each segment. This is the ground-truth bbox order the model should echo.
    """
    body = prompt[len(TASK_TOKEN):]          # strip task token
    segments = body.split("<sep>")
    bboxes = []
    for seg in segments:
        vals = [int(v) for v in LOC_VALUE_RE.findall(seg)]
        if len(vals) == 4:
            bboxes.append(vals)
    return bboxes


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def word_f1(pred: str, ref: str) -> float:
    pred_tok = tokenize(pred)
    ref_tok = tokenize(ref)
    if not pred_tok or not ref_tok:
        return 0.0
    pred_cnt: dict[str, int] = {}
    for t in pred_tok:
        pred_cnt[t] = pred_cnt.get(t, 0) + 1
    ref_cnt: dict[str, int] = {}
    for t in ref_tok:
        ref_cnt[t] = ref_cnt.get(t, 0) + 1
    common = sum(min(pred_cnt.get(t, 0), ref_cnt.get(t, 0)) for t in ref_cnt)
    if common == 0:
        return 0.0
    precision = common / len(pred_tok)
    recall = common / len(ref_tok)
    return 2 * precision * recall / (precision + recall)


def box_iou(a: list[int], b: list[int]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes in loc-space (0–999)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_bbox_match(
    pred_pairs: list[tuple[str, list[int]]],
    gt_pairs: list[tuple[str, list[int]]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """
    Greedy IoU-based matching: for each GT region, find the best unmatched
    prediction with IoU >= threshold.

    Returns list of (pred_idx, gt_idx, iou) for matched pairs only.
    """
    used_pred = set()
    matches = []
    for gi, (_, gt_box) in enumerate(gt_pairs):
        best_iou, best_pi = 0.0, -1
        for pi, (_, pred_box) in enumerate(pred_pairs):
            if pi in used_pred:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0 and best_iou >= iou_threshold:
            used_pred.add(best_pi)
            matches.append((best_pi, gi, best_iou))
    return matches


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def patch_flash_attn():
    orig = dynamic_module_utils.get_imports
    def patched(fn):
        imps = orig(fn)
        if str(fn).endswith("modeling_florence2.py"):
            imps = [n for n in imps if n != "flash_attn"]
        return imps
    dynamic_module_utils.get_imports = patched


def load_model(checkpoint: str, device: str, dtype_name: str):
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


def run_sample(
    processor, model, device, dtype,
    sample: dict,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[str, list[tuple[str, list[int]]]]:
    """
    Run inference for one sample.

    Returns (raw_output, [(desc, bbox), ...]).

    NOTE: We intentionally bypass processor.post_process_generation() here.
    The processor's post-processor only knows Plan A's <sep>-split logic for
    the multi_region_text type. Plan B output needs the REGION_PATTERN regex
    parser instead. We decode with skip_special_tokens=False so loc tokens
    (which ARE special tokens in the Florence tokenizer) are preserved in the
    string for regex parsing.
    """
    image = Image.open(sample["image"]).convert("RGB")
    inputs = processor(
        text=sample["prompt"], images=image,
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

    # Encoder-decoder models such as Florence-2 return decoder output only.
    # Decoder-only models return input + generated output and need prompt slicing.
    if model.config.is_encoder_decoder:
        new_token_ids = generated_ids
    else:
        new_token_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False: keep <loc_NNN> tokens in the decoded string.
    raw = processor.batch_decode(new_token_ids, skip_special_tokens=False)[0]
    pred_pairs = parse_regions(raw)
    return raw, pred_pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Plan B REGIONS_TO_DESCRIPTIONS checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUT_FILE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for bbox-matched evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Only eval first N samples (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    samples = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    log.info("evaluating %d samples from %s", len(samples), args.test_file)

    processor, model = load_model(args.checkpoint, args.device, args.dtype)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # aggregate counters
    n_count_match = 0
    n_exact_match = 0
    total_positional_f1 = 0.0
    total_positional_regions = 0
    total_bbox_matched_f1 = 0.0
    total_bbox_matched_pairs = 0
    total_loc_correct = 0
    total_loc_total = 0
    n_by_size: dict[int, dict] = {}

    out_records = []

    for idx, sample in enumerate(samples):
        gt_pairs = parse_regions(sample["label"])   # [(desc, bbox), ...]
        gt_bboxes = parse_prompt_bboxes(sample["prompt"])  # ordered input bboxes
        n_gt = len(gt_pairs)

        try:
            raw, pred_pairs = run_sample(
                processor, model, device, dtype, sample,
                args.max_new_tokens, args.num_beams,
            )
        except Exception as exc:
            log.warning("sample %d failed: %s", idx, exc)
            out_records.append({"idx": idx, "image": sample["image"], "error": str(exc)})
            continue

        n_pred = len(pred_pairs)
        count_match = (n_pred == n_gt)

        # ── positional metrics (Plan A style, only when count matches) ──────
        positional_f1s: list[float] = []
        exact_match = False
        if count_match:
            positional_f1s = [word_f1(p, g) for (p, _), (g, _) in zip(pred_pairs, gt_pairs)]
            exact_match = all(p == g for (p, _), (g, _) in zip(pred_pairs, gt_pairs))

        # ── loc echo accuracy ─────────────────────────────────────────────
        # Check how many predicted bboxes exactly match the corresponding GT
        # input bbox (by position, up to min(n_pred, n_gt)).
        loc_correct = sum(
            1 for (_, pb), gb in zip(pred_pairs, gt_bboxes) if pb == gb
        )
        loc_total = min(n_pred, len(gt_bboxes))

        # ── bbox-matched metrics (Plan B advantage) ───────────────────────
        # Even when count mismatches, match pred↔gt by IoU and compute F1.
        bbox_matches = greedy_bbox_match(pred_pairs, gt_pairs, args.iou_threshold)
        bbox_matched_f1s = [
            word_f1(pred_pairs[pi][0], gt_pairs[gi][0])
            for pi, gi, _ in bbox_matches
        ]

        # ── accumulate ────────────────────────────────────────────────────
        if count_match:
            n_count_match += 1
            total_positional_f1 += sum(positional_f1s)
            total_positional_regions += n_gt
        if exact_match:
            n_exact_match += 1
        total_loc_correct += loc_correct
        total_loc_total += loc_total
        total_bbox_matched_f1 += sum(bbox_matched_f1s)
        total_bbox_matched_pairs += len(bbox_matches)

        bucket = n_by_size.setdefault(n_gt, {"count": 0, "count_match": 0})
        bucket["count"] += 1
        if count_match:
            bucket["count_match"] += 1

        record = {
            "idx": idx,
            "image": sample["image"],
            "n_gt": n_gt,
            "n_pred": n_pred,
            "count_match": count_match,
            "exact_match": exact_match,
            "positional_avg_f1": (
                round(sum(positional_f1s) / n_gt, 4) if positional_f1s else None
            ),
            "loc_echo_acc": round(loc_correct / loc_total, 4) if loc_total > 0 else None,
            "bbox_matched_pairs": len(bbox_matches),
            "bbox_matched_avg_f1": (
                round(sum(bbox_matched_f1s) / len(bbox_matched_f1s), 4)
                if bbox_matched_f1s else None
            ),
            "gt_pairs": [
                {"desc": d, "bbox": b} for d, b in gt_pairs
            ],
            "pred_pairs": [
                {"desc": d, "bbox": b} for d, b in pred_pairs
            ],
            "raw": raw,
        }
        out_records.append(record)

        if (idx + 1) % 50 == 0 or idx == 0:
            log.info(
                "[%d/%d] count_match=%.1f%%  loc_echo=%.1f%%  bbox_f1=%.4f",
                idx + 1, len(samples),
                100 * n_count_match / (idx + 1),
                100 * total_loc_correct / max(1, total_loc_total),
                total_bbox_matched_f1 / max(1, total_bbox_matched_pairs),
            )

    n_valid = sum(1 for r in out_records if "error" not in r)
    summary = {
        "checkpoint": args.checkpoint,
        "test_file": args.test_file,
        "n_samples": len(samples),
        "n_valid": n_valid,
        # ── Plan A compatible metrics ───────────────────────────────────
        "count_match_rate": round(n_count_match / max(1, n_valid), 4),
        "exact_match_rate": round(n_exact_match / max(1, n_valid), 4),
        "positional_word_f1": round(
            total_positional_f1 / max(1, total_positional_regions), 4
        ),
        # ── Plan B specific metrics ─────────────────────────────────────
        "loc_echo_accuracy": round(
            total_loc_correct / max(1, total_loc_total), 4
        ),
        "bbox_matched_word_f1": round(
            total_bbox_matched_f1 / max(1, total_bbox_matched_pairs), 4
        ),
        "count_match_by_n": {
            str(n): {
                "count": v["count"],
                "count_match_rate": round(v["count_match"] / v["count"], 4),
            }
            for n, v in sorted(n_by_size.items())
        },
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("per-sample results → %s", args.output_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
