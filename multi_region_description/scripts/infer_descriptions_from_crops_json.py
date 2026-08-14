#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "multi_region_description/checkpoints/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6/final"
SINGLE_TASK = "<REGION_TO_DESCRIPTION>"
MULTI_TASK = "<REGIONS_TO_DESCRIPTIONS>"
SEP_TOKEN = "<sep>"
LOC_TOKEN_PATTERN = re.compile(r"<loc_(\d+)>")
REGION_PATTERN = re.compile(r"(.*?)((?:<loc_\d+>\s*){4})", re.DOTALL)
STRIP_TOKENS_PATTERN = re.compile(r"<s>|</s>|<pad>")


@dataclass(frozen=True)
class CropInput:
    crop_key: str
    crop_index: int
    bbox_loc_0_999: tuple[int, int, int, int]

    @property
    def loc_text(self) -> str:
        return "".join(f"<loc_{value}>" for value in self.bbox_loc_0_999)


@dataclass(frozen=True)
class RegionPrediction:
    description: str
    bbox_loc_0_999: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a crop JSON to a Florence prompt and infer person descriptions.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--crop-key", action="append", dest="crop_keys")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return payload


def image_metadata(payload: dict[str, Any]) -> tuple[str, int | None, int | None]:
    image = payload.get("image")
    if isinstance(image, str):
        return image, None, None
    if isinstance(image, dict):
        path = image.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("image.path must be a non-empty string")
        width = image.get("width")
        height = image.get("height")
        return path, int(width) if width else None, int(height) if height else None
    raise ValueError("image must be a path string or an object containing image.path")


def resolve_image_path(image_text: str, input_json: Path) -> Path:
    image_path = Path(image_text)
    candidates = [image_path]
    if not image_path.is_absolute():
        candidates.extend((input_json.parent / image_path, PROJECT_ROOT.parent / image_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {image_text}")


def normalize_bbox_xyxy(bbox: Any, image_width: int | None, image_height: int | None, context: str) -> tuple[int, int, int, int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{context}: bbox must contain four numbers")
    if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
        raise ValueError(f"{context}: image width/height are required for xyxy conversion")
    values = [float(value) for value in bbox]
    normalized = (
        round(values[0] / image_width * 999),
        round(values[1] / image_height * 999),
        round(values[2] / image_width * 999),
        round(values[3] / image_height * 999),
    )
    return tuple(max(0, min(999, value)) for value in normalized)


def extract_bbox(crop: dict[str, Any], image_width: int | None, image_height: int | None, context: str) -> tuple[int, int, int, int]:
    bbox_loc = crop.get("bbox_loc_0_999")
    if isinstance(bbox_loc, (list, tuple)) and len(bbox_loc) == 4:
        bbox = tuple(int(value) for value in bbox_loc)
        if any(value < 0 or value > 999 for value in bbox):
            raise ValueError(f"{context}: bbox_loc_0_999 values must be in [0, 999]")
        return bbox
    for field in ("expanded_bbox_xyxy", "bbox_xyxy"):
        if field in crop:
            return normalize_bbox_xyxy(crop[field], image_width, image_height, context)
    raise ValueError(f"{context}: missing bbox_loc_0_999, expanded_bbox_xyxy, or bbox_xyxy")


def extract_crops(payload: dict[str, Any], crop_keys: list[str] | None = None) -> list[CropInput]:
    _, image_width, image_height = image_metadata(payload)
    raw_crops = payload.get("crops")
    if isinstance(raw_crops, dict):
        entries = [(str(key), value) for key, value in raw_crops.items() if isinstance(value, dict)]
    elif isinstance(raw_crops, list):
        entries = [
            (str(value.get("crop_key", f"crop_{index + 1:03d}")), value)
            for index, value in enumerate(raw_crops)
            if isinstance(value, dict)
        ]
    else:
        raise ValueError("crops must be an object or list")
    if crop_keys:
        requested = set(crop_keys)
        missing = requested - {key for key, _ in entries}
        if missing:
            raise ValueError(f"Unknown crop keys: {', '.join(sorted(missing))}")
        entries = [(key, crop) for key, crop in entries if key in requested]
    crops = []
    for fallback_index, (crop_key, crop) in enumerate(entries, start=1):
        crop_index = int(crop.get("crop_index", fallback_index))
        bbox = extract_bbox(crop, image_width, image_height, f"crop {crop_key}")
        crops.append(CropInput(crop_key, crop_index, bbox))
    crops.sort(key=lambda crop: (crop.crop_index, crop.crop_key))
    if not crops:
        raise ValueError("No usable crops found")
    return crops


def build_prompts(crops: list[CropInput]) -> tuple[str, str, str]:
    if len(crops) == 1:
        loc_input = crops[0].loc_text
        return SINGLE_TASK, SINGLE_TASK + loc_input, f"What does the region {loc_input} describe?"
    loc_input = SEP_TOKEN.join(crop.loc_text for crop in crops)
    return MULTI_TASK, MULTI_TASK + loc_input, f"What does each region {loc_input} describe?"


def parse_multi_region_output(text: str) -> list[RegionPrediction]:
    cleaned = STRIP_TOKENS_PATTERN.sub("", text).strip()
    if cleaned.startswith(MULTI_TASK):
        cleaned = cleaned[len(MULTI_TASK):]
    predictions = []
    for match in REGION_PATTERN.finditer(cleaned):
        description = match.group(1).replace(SEP_TOKEN, " ").strip()
        bbox_values = tuple(int(value) for value in LOC_TOKEN_PATTERN.findall(match.group(2)))
        if description and len(bbox_values) == 4:
            predictions.append(RegionPrediction(description, bbox_values))
    return predictions


def parse_single_region_output(text: str) -> str:
    cleaned = STRIP_TOKENS_PATTERN.sub("", text).strip()
    if cleaned.startswith(SINGLE_TASK):
        cleaned = cleaned[len(SINGLE_TASK):].strip()
    return cleaned


def select_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def select_dtype(dtype_text: str, device: torch.device) -> torch.dtype:
    if dtype_text == "fp16":
        return torch.float16
    if dtype_text == "bf16":
        return torch.bfloat16
    if dtype_text == "fp32":
        return torch.float32
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def patch_flash_attn_import() -> None:
    original_get_imports = dynamic_module_utils.get_imports

    def patched_get_imports(filename: str | Path) -> list[str]:
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            return [name for name in imports if name != "flash_attn"]
        return imports

    dynamic_module_utils.get_imports = patched_get_imports


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    patch_flash_attn_import()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.eval()
    return processor, model


def generate_text(processor, model, image: Image.Image, task_prompt: str, device: torch.device, dtype: torch.dtype, max_new_tokens: int, num_beams: int) -> str:
    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )
    output_ids = generated_ids if model.config.is_encoder_decoder else generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=False)[0]


def assign_multi_predictions(crops: list[CropInput], predictions: list[RegionPrediction]) -> list[dict[str, Any]]:
    unused = set(range(len(predictions)))
    assignments: list[tuple[RegionPrediction | None, str | None]] = []
    for crop in crops:
        exact_index = next(
            (index for index in sorted(unused) if predictions[index].bbox_loc_0_999 == crop.bbox_loc_0_999),
            None,
        )
        if exact_index is None:
            assignments.append((None, None))
        else:
            unused.remove(exact_index)
            assignments.append((predictions[exact_index], "bbox_exact"))
    remaining = iter(sorted(unused))
    completed = []
    for prediction, method in assignments:
        if prediction is None:
            prediction_index = next(remaining, None)
            if prediction_index is not None:
                prediction = predictions[prediction_index]
                method = "output_order_fallback"
        completed.append((prediction, method))
    return [
        {
            "crop_key": crop.crop_key,
            "crop_index": crop.crop_index,
            "bbox_loc_0_999": list(crop.bbox_loc_0_999),
            "description": prediction.description if prediction else None,
            "generated_bbox_loc_0_999": list(prediction.bbox_loc_0_999) if prediction else None,
            "match_method": method,
        }
        for crop, (prediction, method) in zip(crops, completed)
    ]


def assign_single_prediction(crop: CropInput, raw_output: str) -> tuple[dict[str, Any], list[RegionPrediction]]:
    predictions = parse_multi_region_output(raw_output)
    exact_prediction = next(
        (
            prediction
            for prediction in predictions
            if prediction.bbox_loc_0_999 == crop.bbox_loc_0_999
        ),
        None,
    )
    prediction = exact_prediction or (predictions[0] if predictions else None)
    return (
        {
            "crop_key": crop.crop_key,
            "crop_index": crop.crop_index,
            "bbox_loc_0_999": list(crop.bbox_loc_0_999),
            "description": (
                prediction.description
                if prediction
                else parse_single_region_output(raw_output)
            ),
            "generated_bbox_loc_0_999": (
                list(prediction.bbox_loc_0_999) if prediction else None
            ),
            "match_method": (
                "bbox_exact"
                if exact_prediction
                else "first_generated_region_fallback" if prediction else "pure_text"
            ),
        },
        predictions,
    )


def build_preprocessed_result(input_json: Path, image_path: Path, crops: list[CropInput], task: str, task_prompt: str, expanded_prompt: str) -> dict[str, Any]:
    return {
        "input_json": str(input_json.resolve()),
        "image": str(image_path),
        "task": task,
        "task_prompt": task_prompt,
        "expanded_prompt": expanded_prompt,
        "crop_count": len(crops),
        "crops": [
            {
                "crop_key": crop.crop_key,
                "crop_index": crop.crop_index,
                "bbox_loc_0_999": list(crop.bbox_loc_0_999),
            }
            for crop in crops
        ],
    }


def write_result(result: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if output_json is None:
        print(text)
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    print(f"Saved result to {output_json}")


def main() -> int:
    args = parse_args()
    payload = load_json(args.input_json)
    image_text, _, _ = image_metadata(payload)
    image_path = resolve_image_path(image_text, args.input_json)
    crops = extract_crops(payload, crop_keys=args.crop_keys)
    task, task_prompt, expanded_prompt = build_prompts(crops)
    result = build_preprocessed_result(args.input_json, image_path, crops, task, task_prompt, expanded_prompt)
    if args.prompt_only:
        write_result(result, args.output_json)
        return 0
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    processor, model = load_model(args.checkpoint, device, dtype)
    with Image.open(image_path) as source_image:
        raw_output = generate_text(
            processor,
            model,
            source_image.convert("RGB"),
            task_prompt,
            device,
            dtype,
            args.max_new_tokens,
            args.num_beams,
        )
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        raw_output=raw_output,
    )
    if task == SINGLE_TASK:
        crop_result, predictions = assign_single_prediction(crops[0], raw_output)
        result["crops"] = [crop_result]
        if predictions:
            result["predictions"] = [
                {
                    "description": prediction.description,
                    "bbox_loc_0_999": list(prediction.bbox_loc_0_999),
                }
                for prediction in predictions
            ]
    else:
        predictions = parse_multi_region_output(raw_output)
        result["predictions"] = [
            {"description": prediction.description, "bbox_loc_0_999": list(prediction.bbox_loc_0_999)}
            for prediction in predictions
        ]
        result["crops"] = assign_multi_predictions(crops, predictions)
    write_result(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
