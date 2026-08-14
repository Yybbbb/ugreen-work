#!/usr/bin/env python3
"""Single-pass multi-region description inference for the <REGIONS_TO_DESCRIPTIONS> task.

Given one image and N pixel bboxes, build a single prompt

    <REGIONS_TO_DESCRIPTIONS><loc_..><loc_..><loc_..><loc_..><sep><loc_..>...

run ONE model.generate, and split the output on <sep> to recover one description
per input bbox — replacing N separate REGION_TO_DESCRIPTION calls with a single
forward pass (one image encode instead of N).

The bbox encoding (pixel_bbox_to_loc), model loading, and generation mirror
ugipc_1231_15words_epoch3_full_handoff/infer_ugipc.py so the prompt format is
identical to the training data built by prepare_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_CHECKPOINT = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/"
    "checkpoints/multi_region_ep2_lr1e5/final"
)
TASK_TOKEN = "<REGIONS_TO_DESCRIPTIONS>"
SEP = "<sep>"


# --------------------------------------------------------------------------- #
# bbox parsing / encoding (mirrors infer_ugipc.py)
# --------------------------------------------------------------------------- #

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
        raise ValueError(f"bbox is outside image bounds {width}x{height}: {list(bbox_xyxy)}")
    size_per_bin_w = width / 1000.0
    size_per_bin_h = height / 1000.0
    loc = [
        math.floor(x1 / size_per_bin_w),
        math.floor(y1 / size_per_bin_h),
        math.floor(x2 / size_per_bin_w),
        math.floor(y2 / size_per_bin_h),
    ]
    return [max(0, min(999, int(v))) for v in loc]


def loc_tokens(loc_box: list[int]) -> str:
    return "".join(f"<loc_{int(v)}>" for v in loc_box)


def build_multi_region_prompt(loc_boxes: list[list[int]]) -> str:
    """Join per-bbox loc-token groups with <sep>, prefixed by the task token."""
    groups = [loc_tokens(box) for box in loc_boxes]
    return f"{TASK_TOKEN}{SEP.join(groups)}"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_model(checkpoint: str, device: str, dtype_name: str):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_name]
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def generate_descriptions(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[str, list[str]]:
    """Run one generation and post-process into a list of per-region descriptions."""
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
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )
    # Keep special tokens so <sep> survives for splitting.
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        raw, task=TASK_TOKEN, image_size=(image.width, image.height)
    )
    descriptions = result[TASK_TOKEN]["descriptions"]
    return raw, descriptions


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-pass multi-region description with <REGIONS_TO_DESCRIPTIONS>."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        action="append",
        required=True,
        metavar="x1,y1,x2,y2",
        help="Pixel bbox; pass --bbox multiple times for multiple regions",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    loc_boxes = [pixel_bbox_to_loc(bbox, width, height) for bbox in args.bbox]
    prompt = build_multi_region_prompt(loc_boxes)

    processor, model = load_model(args.checkpoint, args.device, args.dtype)
    raw, descriptions = generate_descriptions(
        processor, model, image, prompt, args.max_new_tokens, args.num_beams
    )

    count_ok = len(descriptions) == len(args.bbox)
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "image": {"path": str(image_path), "width": width, "height": height},
                "task": TASK_TOKEN,
                "prompt": prompt,
                "n_bboxes": len(args.bbox),
                "bboxes_pixel": args.bbox,
                "bboxes_loc_0_999": loc_boxes,
                "raw": raw,
                "descriptions": descriptions,
                "n_descriptions": len(descriptions),
                "count_match": count_ok,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not count_ok:
        # Non-fatal: caller may choose to fall back to per-region inference.
        print(
            f"WARNING: got {len(descriptions)} descriptions for {len(args.bbox)} bboxes",
            flush=True,
        )


if __name__ == "__main__":
    main()
