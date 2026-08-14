#!/usr/bin/env python3
"""Minimal inference utility for ugipc_1231_15words/epoch_3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoint")

CAPTION_TASKS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
}


def parse_bbox(value: str) -> list[float]:
    parts = [part for part in value.replace(",", " ").split() if part]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must contain 4 values: x1,y1,x2,y2")
    try:
        bbox = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numeric") from exc
    if not all(math.isfinite(v) for v in bbox):
        raise argparse.ArgumentTypeError("bbox values must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise argparse.ArgumentTypeError("bbox must have positive area")
    return bbox


def pixel_bbox_to_loc(bbox_xyxy: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"bbox is outside image bounds {width}x{height}: {bbox_xyxy}")
    size_per_bin_w = width / 1000.0
    size_per_bin_h = height / 1000.0
    loc = [
        math.floor(x1 / size_per_bin_w),
        math.floor(y1 / size_per_bin_h),
        math.floor(x2 / size_per_bin_w),
        math.floor(y2 / size_per_bin_h),
    ]
    return [max(0, min(999, int(v))) for v in loc]


def build_region_description_prompt(loc_box: list[int], name: str = "") -> str:
    loc_tokens = "".join(f"<loc_{int(v)}>" for v in loc_box)
    if name:
        return f"<REGION_TO_DESCRIPTION>{name}{loc_tokens}"
    return f"<REGION_TO_DESCRIPTION>{loc_tokens}"


def clean_text(text: str) -> str:
    cleaned = str(text or "").strip()
    for token in ("<s>", "</s>", "<pad>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip().strip('"').strip("'").rstrip(".").strip()


def load_model(checkpoint: str, device: str, dtype_name: str):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def generate(processor, model, image: Image.Image, prompt: str, max_new_tokens: int, num_beams: int):
    device = next(model.parameters()).device
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return raw, clean_text(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Florence-2 ugipc_1231_15words/epoch_3 inference.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--task",
        choices=["caption", "detailed_caption", "more_detailed_caption", "region_description"],
        required=True,
    )
    parser.add_argument("--bbox", type=parse_bbox, help="Pixel bbox x1,y1,x2,y2 for region_description")
    parser.add_argument("--name", default="", help="Optional plain-text region name, e.g. person")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if args.task == "region_description":
        if args.bbox is None:
            raise SystemExit("--bbox is required for region_description")
        loc_box = pixel_bbox_to_loc(args.bbox, width, height)
        prompt = build_region_description_prompt(loc_box, name=args.name)
    else:
        loc_box = None
        prompt = CAPTION_TASKS[args.task]

    processor, model = load_model(args.checkpoint, args.device, args.dtype)
    raw, clean = generate(processor, model, image, prompt, args.max_new_tokens, args.num_beams)

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "image": {"path": str(image_path), "width": width, "height": height},
                "task": args.task,
                "prompt": prompt,
                "bbox_pixel": args.bbox,
                "bbox_loc_0_999": loc_box,
                "raw": raw,
                "text": clean,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
