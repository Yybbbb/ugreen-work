#!/usr/bin/env python3
"""
Benchmark: Plan A (REGIONS_TO_DESCRIPTIONS, one call/image)
       vs  Single-crop (REGION_TO_DESCRIPTION, N calls/image)

Both run with batch_size=1, num_workers=1, num_beams=1, fp16 on the same GPU.
Timing uses torch.cuda.synchronize() around each model.generate() call.

Output: JSON results file + printed report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils

# ── checkpoints ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

CKPT_PLAN_A = str(PROJECT_DIR / "checkpoints/multi_region_no_name_ep2_lr1e5/final")
CKPT_SINGLE = str(BASE_DIR / "ugipc_1231_15words_epoch3_full_handoff/checkpoint")
TEST_FILE = str(
    PROJECT_DIR / "data/ugipc-person-region-descriptions-yolo26m-no-name/test.jsonl"
)
OUT_FILE = str(RESULTS_DIR / "benchmark_plan_a_vs_single.json")
REPORT_FILE = str(RESULTS_DIR / "benchmark_report.md")

TASK_MULTI  = "<REGIONS_TO_DESCRIPTIONS>"
TASK_SINGLE = "<REGION_TO_DESCRIPTION>"
SEP = "<sep>"
MAX_NEW_TOKENS_SINGLE = 80    # matches training config of base model
MAX_NEW_TOKENS_MULTI  = 256   # matches Plan A eval config


# ── helpers ───────────────────────────────────────────────────────────────────

def patch_flash_attn():
    orig = dynamic_module_utils.get_imports
    def patched(fn):
        imps = orig(fn)
        if str(fn).endswith("modeling_florence2.py"):
            imps = [i for i in imps if i != "flash_attn"]
        return imps
    dynamic_module_utils.get_imports = patched


def load_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    patch_flash_attn()
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def timed_generate(
    processor, model, device: torch.device, dtype: torch.dtype,
    image: Image.Image, prompt: str, max_new_tokens: int,
) -> tuple[float, int, int, str]:
    """
    Run one model.generate(), return (wall_seconds, n_input_tokens, n_output_tokens, raw_text).
    Uses torch.cuda.synchronize() for accurate GPU wall time.
    """
    inputs = processor(
        text=prompt, images=image,
        return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    n_input = inputs["input_ids"].shape[1]

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" else nullcontext()
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad(), autocast_ctx:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            no_repeat_ngram_size=0,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    n_output = generated_ids.shape[1] - n_input
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return elapsed, n_input, n_output, raw


def parse_prompt_bboxes(prompt: str) -> list[list[int]]:
    """Extract ordered loc bboxes from a REGIONS_TO_DESCRIPTIONS prompt."""
    body = prompt[len(TASK_MULTI):]
    segments = body.split(SEP)
    bboxes = []
    for seg in segments:
        vals = [int(v) for v in re.findall(r"<loc_(\d+)>", seg)]
        if len(vals) == 4:
            bboxes.append(vals)
    return bboxes


@dataclass
class SampleResult:
    idx: int
    image_path: str
    n_regions: int
    # plan A
    plan_a_time_s: float = 0.0
    plan_a_input_tok: int = 0
    plan_a_output_tok: int = 0
    # single crop  (list of per-crop times)
    single_times_s: list[float] = field(default_factory=list)
    single_input_toks: list[int] = field(default_factory=list)
    single_output_toks: list[int] = field(default_factory=list)

    @property
    def single_total_time_s(self) -> float:
        return sum(self.single_times_s)

    @property
    def single_total_output_tok(self) -> int:
        return sum(self.single_output_toks)

    @property
    def single_total_input_tok(self) -> int:
        return sum(self.single_input_toks)


# ── main benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(args) -> list[SampleResult]:
    device = torch.device(args.device)
    dtype  = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Load test samples first (shared by both phases)
    samples = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Test set: {len(samples)} samples")

    # Pre-allocate result objects
    results: list[SampleResult] = []
    for i, sample in enumerate(samples):
        bboxes = parse_prompt_bboxes(sample["prompt"])
        results.append(SampleResult(
            idx=i, image_path=sample["image"], n_regions=len(bboxes)
        ))

    dummy_img = Image.new("RGB", (640, 480), color=(128, 128, 128))

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Plan A  (load → warmup → run all → unload)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 1] Loading Plan A checkpoint …")
    proc_a, model_a = load_model(args.ckpt_plan_a, device, dtype)
    print("  Warming up Plan A …")
    timed_generate(proc_a, model_a, device, dtype, dummy_img,
                   f"{TASK_MULTI}<loc_100><loc_100><loc_500><loc_500>", 20)
    print("  Running Plan A on test set …")

    for i, sample in enumerate(samples):
        image  = Image.open(sample["image"]).convert("RGB")
        prompt = sample["prompt"]
        t, ni, no, _ = timed_generate(
            proc_a, model_a, device, dtype, image, prompt, MAX_NEW_TOKENS_MULTI
        )
        results[i].plan_a_time_s     = t
        results[i].plan_a_input_tok  = ni
        results[i].plan_a_output_tok = no
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1:3d}/{len(samples)}]  N={results[i].n_regions}  t={t:.3f}s")

    # Unload Plan A to free GPU memory before loading Single
    del model_a, proc_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("  Plan A done. GPU memory released.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Single-crop  (load → warmup → run all → done)
    # ══════════════════════════════════════════════════════════════════════════
    print("[Phase 2] Loading Single-crop checkpoint …")
    proc_s, model_s = load_model(args.ckpt_single, device, dtype)
    print("  Warming up Single-crop …")
    timed_generate(proc_s, model_s, device, dtype, dummy_img,
                   f"{TASK_SINGLE}<loc_100><loc_100><loc_500><loc_500>", 20)
    print("  Running Single-crop on test set …")

    for i, sample in enumerate(samples):
        image  = Image.open(sample["image"]).convert("RGB")
        bboxes = parse_prompt_bboxes(sample["prompt"])
        for bbox in bboxes:
            loc_str = "".join(f"<loc_{v}>" for v in bbox)
            t_s, ni_s, no_s, _ = timed_generate(
                proc_s, model_s, device, dtype, image,
                f"{TASK_SINGLE}{loc_str}", MAX_NEW_TOKENS_SINGLE
            )
            results[i].single_times_s.append(t_s)
            results[i].single_input_toks.append(ni_s)
            results[i].single_output_toks.append(no_s)
        if (i + 1) % 50 == 0 or i == 0:
            r = results[i]
            print(f"    [{i+1:3d}/{len(samples)}]  N={r.n_regions}  "
                  f"total={r.single_total_time_s:.3f}s  "
                  f"speedup={r.single_total_time_s/max(r.plan_a_time_s,1e-9):.2f}×")

    print("  Single-crop done.\n")
    return results


# ── statistics ────────────────────────────────────────────────────────────────

def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "n":     len(vals),
        "total": round(sum(vals), 4),
        "mean":  round(sum(vals) / len(vals), 4),
        "min":   round(min(vals), 4),
        "p50":   round(percentile(vals, 50), 4),
        "p90":   round(percentile(vals, 90), 4),
        "p99":   round(percentile(vals, 99), 4),
        "max":   round(max(vals), 4),
    }


# ── report ────────────────────────────────────────────────────────────────────

def build_report(results: list[SampleResult], args) -> str:
    all_n = sorted({r.n_regions for r in results})

    # ── image-level times ────────────────────────────────────────────────────
    pa_img   = [r.plan_a_time_s for r in results]
    sg_img   = [r.single_total_time_s for r in results]
    speedups = [r.single_total_time_s / max(r.plan_a_time_s, 1e-9) for r in results]

    # ── crop-level times ─────────────────────────────────────────────────────
    pa_per_crop = []
    for r in results:
        t_per = r.plan_a_time_s / r.n_regions
        pa_per_crop.extend([t_per] * r.n_regions)
    sg_per_crop = [t for r in results for t in r.single_times_s]

    # ── token-level ──────────────────────────────────────────────────────────
    pa_tok_out   = sum(r.plan_a_output_tok for r in results)
    pa_tok_in    = sum(r.plan_a_input_tok  for r in results)
    sg_tok_out   = sum(r.single_total_output_tok for r in results)
    sg_tok_in    = sum(r.single_total_input_tok  for r in results)
    pa_total_t   = sum(pa_img)
    sg_total_t   = sum(sg_img)

    total_crops  = sum(r.n_regions for r in results)
    n_images     = len(results)

    lines = [
        "# Plan A vs Single-Crop 速度对比实验报告",
        "",
        "## 实验配置",
        "",
        f"| 项目 | 值 |",
        f"|------|----|",
        f"| 测试集 | `{args.test_file}` |",
        f"| 图片数 | {n_images} |",
        f"| 总 crop 数 | {total_crops} |",
        f"| GPU | {args.device} (`CUDA_VISIBLE_DEVICES={args.cuda_visible}`) |",
        f"| 精度 | {args.dtype} |",
        f"| batch_size | 1 |",
        f"| num_beams | 1 |",
        f"| Plan A checkpoint | `{args.ckpt_plan_a}` |",
        f"| Single checkpoint | `{args.ckpt_single}` |",
        f"| max_new_tokens (Plan A) | {MAX_NEW_TOKENS_MULTI} |",
        f"| max_new_tokens (Single) | {MAX_NEW_TOKENS_SINGLE} |",
        "",
        "---",
        "",
        "## 整体结果",
        "",
        f"| 指标 | Plan A | Single-Crop | 比值 (A/Single) |",
        f"|------|--------|------------|-----------------|",
        f"| **总推理时间** | {pa_total_t:.1f} s | {sg_total_t:.1f} s | {pa_total_t/sg_total_t:.3f}× |",
        f"| **单图平均时间** | {pa_total_t/n_images*1000:.1f} ms | {sg_total_t/n_images*1000:.1f} ms | {pa_total_t/sg_total_t:.3f}× |",
        f"| **单 crop 平均时间** | {pa_total_t/total_crops*1000:.1f} ms | {sg_total_t/total_crops*1000:.1f} ms | {pa_total_t/total_crops/(sg_total_t/total_crops):.3f}× |",
        f"| 总输出 token 数 | {pa_tok_out} | {sg_tok_out} | {pa_tok_out/max(sg_tok_out,1):.2f}× |",
        f"| 总输入 token 数 | {pa_tok_in} | {sg_tok_in} | {pa_tok_in/max(sg_tok_in,1):.2f}× |",
        f"| **输出 token 速率** | {pa_tok_out/pa_total_t:.1f} tok/s | {sg_tok_out/sg_total_t:.1f} tok/s | — |",
        f"| **单输出 token 平均耗时** | {pa_total_t/max(pa_tok_out,1)*1000:.2f} ms | {sg_total_t/max(sg_tok_out,1)*1000:.2f} ms | — |",
        "",
        "---",
        "",
        "## 单图时间分布（秒）",
        "",
        "| 统计量 | Plan A | Single-Crop |",
        "|--------|--------|------------|",
    ]
    for key in ("mean", "min", "p50", "p90", "p99", "max"):
        sa, ss = stats(pa_img), stats(sg_img)
        lines.append(f"| {key} | {sa[key]*1000:.1f} ms | {ss[key]*1000:.1f} ms |")
    lines += ["", "---", "", "## 单 crop 时间分布（秒）", "",
              "| 统计量 | Plan A（按 crop 均摊） | Single-Crop（每 crop） |",
              "|--------|----------------------|----------------------|"]
    for key in ("mean", "min", "p50", "p90", "p99", "max"):
        spa, ssg = stats(pa_per_crop), stats(sg_per_crop)
        lines.append(f"| {key} | {spa[key]*1000:.1f} ms | {ssg[key]*1000:.1f} ms |")

    lines += ["", "---", "", "## 按 N 分组的单图时间对比", "",
              "| N | 样本数 | Plan A 均值 | Single 均值 | 加速比 |",
              "|---|--------|------------|------------|--------|"]
    for n in all_n:
        sub = [r for r in results if r.n_regions == n]
        pa_n  = [r.plan_a_time_s for r in sub]
        sg_n  = [r.single_total_time_s for r in sub]
        sp_n  = [sg/max(pa,1e-9) for pa,sg in zip(pa_n,sg_n)]
        lines.append(
            f"| {n} | {len(sub)} | "
            f"{sum(pa_n)/len(pa_n)*1000:.1f} ms | "
            f"{sum(sg_n)/len(sg_n)*1000:.1f} ms | "
            f"**{sum(sp_n)/len(sp_n):.2f}×** |"
        )

    lines += ["", "---", "", "## 加速比分布（Single 总时 / Plan A 总时，按图）", "",
              "| 统计量 | 加速比 |",
              "|--------|--------|"]
    sp_stats = stats(speedups)
    for key in ("mean", "min", "p50", "p90", "max"):
        lines.append(f"| {key} | {sp_stats[key]:.2f}× |")

    lines += [
        "",
        "---",
        "",
        "## 结论",
        "",
        f"- 总体加速比 (Single / Plan A) = **{sg_total_t/pa_total_t:.2f}×**",
        f"  - 即 Plan A 处理全部测试集耗时 {pa_total_t:.1f}s，Single 耗时 {sg_total_t:.1f}s",
        f"  - {'Plan A 更快' if pa_total_t < sg_total_t else 'Plan A 更慢'}，差值 {abs(sg_total_t - pa_total_t):.1f}s",
        "",
        "- 注：Plan A 输出 token 数约为 Single 的"
        f" {pa_tok_out/max(sg_tok_out,1):.1f}x（因为一次输出所有描述）；",
        "  Single-crop 的 max_new_tokens=80（每条描述），Plan A 的 max_new_tokens=256（所有描述拼接）。",
    ]
    return "\n".join(lines)


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-plan-a",  default=CKPT_PLAN_A)
    parser.add_argument("--ckpt-single",  default=CKPT_SINGLE)
    parser.add_argument("--test-file",    default=TEST_FILE)
    parser.add_argument("--out-file",     default=OUT_FILE)
    parser.add_argument("--report-file",  default=REPORT_FILE)
    parser.add_argument("--device",       default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",        choices=["fp16","bf16"], default="fp16")
    parser.add_argument("--limit",        type=int, default=0, help="Only eval first N samples (0=all)")
    parser.add_argument("--cuda-visible", default="1")
    args = parser.parse_args()

    results = run_benchmark(args)

    # save raw JSON
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([{
            "idx":             r.idx,
            "image":           r.image_path,
            "n_regions":       r.n_regions,
            "plan_a_time_s":   r.plan_a_time_s,
            "plan_a_input_tok":r.plan_a_input_tok,
            "plan_a_output_tok":r.plan_a_output_tok,
            "single_times_s":  r.single_times_s,
            "single_input_toks":r.single_input_toks,
            "single_output_toks":r.single_output_toks,
        } for r in results], ensure_ascii=False, indent=2)
    )
    print(f"\nJSON saved: {args.out_file}")

    report = build_report(results, args)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report:    {args.report_file}")
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main()
