#!/usr/bin/env python3
"""Ask an OpenAI-compatible Qwen VL service to label upper/lower colors.

The script reads one unified annotation JSON, extracts useful upper/lower
localization hints, converts [0, 1] xywh boxes to Qwen-style [0, 999] xyxy
boxes, renders a prompt from a separate template file, and sends the prompt
plus image to /v1/chat/completions.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Please install requests first: pip install requests")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parent / "data" / "LIP_clothes_accessory_unified"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "qwen_upper_lower_color_prompt.txt"
SERVER_IP = "192.168.111.17"
DEFAULT_BASE_URL = f"http://{SERVER_IP}:6096/v1"
DEFAULT_MODEL = "Qwen35-35b"
_SPLIT_DIR = SCRIPT_DIR.parent / "data" / "LIP_clothes_accessory_unified" / "splits" / "clothes_accessory_poseprior"
DEFAULT_SPLIT_FILES = [
    _SPLIT_DIR / "train.txt",
    _SPLIT_DIR / "val.txt",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "annotation"

COLOR_SCHEMA = [
    {"index": 0, "english": "unknown", "chinese": "未知", "rgb": None},
    {"index": 1, "english": "black", "chinese": "黑色", "rgb": [0, 0, 0]},
    {"index": 2, "english": "white", "chinese": "白色", "rgb": [255, 255, 255]},
    {"index": 3, "english": "gray", "chinese": "灰色", "rgb": [128, 128, 128]},
    {"index": 4, "english": "red", "chinese": "红色", "rgb": [255, 0, 0]},
    {"index": 5, "english": "yellow", "chinese": "黄色", "rgb": [255, 255, 0]},
    {"index": 6, "english": "green", "chinese": "绿色", "rgb": [0, 255, 0]},
    {"index": 7, "english": "blue", "chinese": "蓝色", "rgb": [0, 0, 255]},
    {"index": 8, "english": "purple", "chinese": "紫色", "rgb": [128, 0, 128]},
    {"index": 9, "english": "pink", "chinese": "粉色", "rgb": [255, 192, 203]},
    {"index": 10, "english": "orange", "chinese": "橙色", "rgb": [255, 165, 0]},
    {"index": 11, "english": "brown", "chinese": "棕色", "rgb": [137, 81, 41]},
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def xywh_norm_to_qwen_xyxy(box: list[float]) -> list[int]:
    """Convert normalized xywh to Qwen-style normalized xyxy on a 0-999 grid."""
    if len(box) != 4:
        raise ValueError(f"Expected xywh box with 4 values, got {box!r}")
    x, y, w, h = [float(v) for v in box]
    x1 = clamp(x)
    y1 = clamp(y)
    x2 = clamp(x + w)
    y2 = clamp(y + h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1 * 999), round(y1 * 999), round(x2 * 999), round(y2 * 999)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bbox(bbox: list, img_w: float, img_h: float) -> list[float]:
    """Normalize a bbox to [0,1] if it contains pixel coordinates (any value > 1)."""
    # Flatten [[x,y,w,h]] → [x,y,w,h]
    if bbox and isinstance(bbox[0], (list, tuple)):
        bbox = list(bbox[0])
    x, y, w, h = [float(v) for v in bbox]
    if max(abs(x), abs(y), abs(w), abs(h)) > 1.0:
        # Pixel coordinates – normalize by image dimensions
        if img_w > 0 and img_h > 0:
            x, w = x / img_w, w / img_w
            y, h = y / img_h, h / img_h
    return [x, y, w, h]


def mask_bbox_to_norm(mask: dict[str, Any] | None, annotation_path: Path, img_w: float, img_h: float) -> list[float] | None:
    """Derive a normalized xywh bbox from supported mask formats."""
    if not mask or mask.get("format") != "lip_palette_ids":
        return None

    mask_path_value = mask.get("path")
    lip_ids = set(mask.get("lip_ids") or [])
    if not mask_path_value or not lip_ids:
        return None

    mask_path = Path(mask_path_value)
    if not mask_path.is_absolute():
        mask_path = annotation_path.parent / mask_path
    if not mask_path.exists():
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    with Image.open(mask_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = gray.load()
        xs = []
        ys = []
        for y in range(height):
            for x in range(width):
                if pixels[x, y] in lip_ids:
                    xs.append(x)
                    ys.append(y)

    if not xs or not ys:
        return None

    x1, x2 = min(xs), max(xs) + 1
    y1, y2 = min(ys), max(ys) + 1
    denom_w = img_w or width
    denom_h = img_h or height
    if denom_w <= 0 or denom_h <= 0:
        return None
    return [x1 / denom_w, y1 / denom_h, (x2 - x1) / denom_w, (y2 - y1) / denom_h]


def extract_garment_regions(annotation_path: Path) -> list[dict[str, Any]]:
    annotation = load_json(annotation_path)
    attributes = annotation.get("attributes", {})
    img_info = annotation.get("image", {})
    img_w = float(img_info.get("width") or 0)
    img_h = float(img_info.get("height") or 0)
    regions = []

    for part in ("upper", "lower"):
        item = attributes.get(part, {})
        label = item.get("label", "unknown")
        bbox = item.get("bbox")
        if label == "unknown":
            continue
        bbox_source = "bbox"
        if bbox:
            bbox_norm = normalize_bbox(bbox, img_w, img_h)
        else:
            bbox_norm = mask_bbox_to_norm(item.get("mask"), annotation_path, img_w, img_h)
            bbox_source = "mask"
        if not bbox_norm:
            continue
        region = {
            "part": part,
            "source_label": label,
            "bbox_xywh_norm": bbox_norm,
            "bbox_qwen_xyxy_0_999": xywh_norm_to_qwen_xyxy(bbox_norm),
            "annotation_score": float(item.get("score", 0.0) or 0.0),
        }
        if bbox_source != "bbox":
            region["bbox_source"] = bbox_source
        regions.append(region)

    return regions


def resolve_image_path(annotation_path: Path, dataset_root: Path) -> Path:
    annotation = load_json(annotation_path)
    image_path = annotation.get("image", {}).get("path")
    if not image_path:
        raise ValueError(f"No image.path found in {annotation_path}")

    image_path_obj = Path(image_path)
    if image_path_obj.is_absolute():
        return image_path_obj
    return dataset_root / image_path_obj


def build_prompt(template: str, regions: list[dict[str, Any]]) -> str:
    return (
        template.replace(
            "{color_schema_json}",
            json.dumps(COLOR_SCHEMA, ensure_ascii=False, indent=2),
        ).replace(
            "{regions_json}",
            json.dumps(regions, ensure_ascii=False, indent=2),
        )
    )


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def call_qwen(
    base_url: str,
    model: str,
    prompt: str,
    image_path: Path,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Force pure JSON output, no markdown or extra text
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": "/no_think\nYou are a JSON-only clothing color labeler. Output a single valid JSON object and nothing else.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def process_entry(
    image_path: Path,
    annotation_path: Path,
    prompt_template: str,
    output_path: Path,
    base_url: str,
    model: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> bool:
    """Process one image/annotation pair and always write a result JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "annotation_path": str(annotation_path),
        "image_path": str(image_path),
        "model": model,
        "base_url": base_url,
    }

    try:
        regions = extract_garment_regions(annotation_path)
        prompt = build_prompt(prompt_template, regions)
        result["regions"] = regions
        result["prompt"] = prompt

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not dry_run:
            response_json = call_qwen(
                base_url=base_url,
                model=model,
                prompt=prompt,
                image_path=image_path,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = extract_message_text(response_json)
            result["qwen_response"] = response_json
            result["qwen_raw_text"] = raw_text
            result["parsed"] = parse_json_from_text(raw_text)
        else:
            result["qwen_raw_text"] = ""
            result["parsed"] = None
            result["note"] = "Dry run only; Qwen service was not called."

        result["status"] = "success"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return False


def iter_split_entries(split_files: list[Path], dataset_root: Path):
    """Yield (image_path, annotation_path) for every valid line across all split files in order."""
    for split_file in split_files:
        for line in split_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                yield dataset_root / parts[0], dataset_root / parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label upper/lower garment colors for all images in a split file."
    )
    parser.add_argument(
        "--split-files",
        type=Path,
        nargs="+",
        default=DEFAULT_SPLIT_FILES,
        help="One or more split txt files (default: train.txt + val.txt).",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write per-image result JSONs.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose output JSON already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts without calling the Qwen service (writes output with note).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root  = args.dataset_root.resolve()
    prompt_file   = args.prompt_file.resolve()
    output_dir    = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = [p.resolve() for p in args.split_files]
    prompt_template = prompt_file.read_text(encoding="utf-8")
    entries = list(iter_split_entries(split_files, dataset_root))
    total   = len(entries)

    print(f"Split files   : {[str(p) for p in split_files]}")
    print(f"Total entries : {total}")
    print(f"Output dir    : {output_dir}")
    print(f"Resume        : {args.resume}")
    print(f"Dry-run       : {args.dry_run}\n")

    success = skipped = errors = 0

    for i, (image_path, annotation_path) in enumerate(entries, 1):
        stem        = image_path.stem
        output_path = output_dir / f"{stem}.json"

        if args.resume and output_path.exists():
            skipped += 1
            continue

        try:
            if process_entry(
                image_path=image_path,
                annotation_path=annotation_path,
                prompt_template=prompt_template,
                output_path=output_path,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                dry_run=args.dry_run,
            ):
                success += 1
            else:
                errors += 1
                print(f"[{i}/{total}] ERROR {stem}: see {output_path}", flush=True)

            if i % 100 == 0 or i == total:
                print(f"[{i}/{total}] success={success} skipped={skipped} errors={errors}", flush=True)

        except Exception as exc:
            errors += 1
            print(f"[{i}/{total}] ERROR {stem}: {exc}", flush=True)

    print(f"\nDone — total={total}, success={success}, skipped={skipped}, errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
