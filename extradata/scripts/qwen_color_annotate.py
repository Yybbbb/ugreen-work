#!/usr/bin/env python3
"""
Batch annotate all extracted frames with Qwen VL:
  1. Check if a person is visible.
  2. If yes, classify upper-body and lower-body garment colors.

Uses the running vLLM Qwen container on port 6096.
Saves per-image JSON results with resume support.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("Please install requests: pip install requests")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES_DIR = SCRIPT_DIR / "extract_frames"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "qwen_color_annotations"
DEFAULT_BASE_URL = "http://localhost:6096/v1"
DEFAULT_MODEL = "Qwen35-35b"

# ALLOWED_LABELS is the authoritative list that the prompt enforces.
# The COLOR_SCHEMA is kept for compatibility with downstream consumers.
ALLOWED_LABELS = [
    "black", "white", "gray", "blue", "red", "green",
    "brown", "yellow", "pink", "purple", "orange", "unknown",
]

COLOR_SCHEMA = [
    {"index": 0,  "english": "black",   "chinese": "黑色", "rgb": [0, 0, 0]},
    {"index": 1,  "english": "white",   "chinese": "白色", "rgb": [255, 255, 255]},
    {"index": 2,  "english": "gray",    "chinese": "灰色", "rgb": [128, 128, 128]},
    {"index": 3,  "english": "blue",    "chinese": "蓝色", "rgb": [0, 0, 255]},
    {"index": 4,  "english": "red",     "chinese": "红色", "rgb": [255, 0, 0]},
    {"index": 5,  "english": "green",   "chinese": "绿色", "rgb": [0, 255, 0]},
    {"index": 6,  "english": "brown",   "chinese": "棕色", "rgb": [137, 81, 41]},
    {"index": 7,  "english": "yellow",  "chinese": "黄色", "rgb": [255, 255, 0]},
    {"index": 8,  "english": "pink",    "chinese": "粉色", "rgb": [255, 192, 203]},
    {"index": 9,  "english": "purple",  "chinese": "紫色", "rgb": [128, 0, 128]},
    {"index": 10, "english": "orange",  "chinese": "橙色", "rgb": [255, 165, 0]},
    {"index": 11, "english": "unknown", "chinese": "未知", "rgb": None},
]

# ---------------------------------------------------------------------------
# Prompt template (simplified – no bbox / mask regions)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = r"""/no_think

You are a clothing color annotation assistant. Output ONLY a single valid JSON object. No thinking, no explanation, no markdown.

## Task
Look at this image and do two things:
1. Determine whether at least one person is clearly visible in the image. A person counts as visible if you can see enough of their body to identify clothing colors on the upper or lower half. If only a tiny sliver, blurred blob, or extreme close-up of a non-clothing body part is visible, treat it as no person.
2. If a person IS visible, identify the dominant garment color for the upper-body garment AND the lower-body garment.

## Step 1 – Person check
- If NO person is visible → output exactly: {"has_person": false}
- If a person IS visible → proceed to Step 2 and output "has_person": true with "upper" and "lower" keys.

## Step 2 – Color classification (only when has_person is true)

### CRITICAL: You MUST output BOTH "upper" and "lower" keys. This is NON-NEGOTIABLE.
Every output with "has_person": true MUST contain both keys exactly like this:
{{
  "has_person": true,
  "upper": [{{"label": "<color>", "confidence": <float>}}],
  "lower": [{{"label": "<color>", "confidence": <float>}}]
}}

### Allowed colors (ONLY these 12 labels — NO other values):
{allowed_labels}

### Color selection rules:

**Pick ONE color when:**
- The garment is a single solid color overall (ignore shadows, wrinkles, lighting, compression artifacts).
- The garment has tiny decorative elements (small logos, buttons, zippers, trim, stitching) — ignore them and pick the main fabric color.

**Pick "unknown" (NOT multiple colors) when:**
- The garment has stripes, plaid / check / tartan patterns.
- The garment has multiple colors evenly or near-evenly distributed (e.g. floral prints with no clear dominant color, camouflage, tie-dye, color-block with equal areas).
- The lighting is too dark, overexposed, or the color is ambiguous and you genuinely cannot decide.
- The body part is fully occluded, cut off, or completely invisible.

**Output at most TWO colors ONLY when:**
- The garment clearly has one dominant background color plus a single large, distinct color-block panel (NOT stripes, NOT plaid, NOT small details).
- The two colors have clearly unequal visible areas, so you can assign highest confidence to the dominant one.

### Confidence rules:
- Confidence is a float from 0.0 to 1.0.
- For single-color output, confidence should be high (≥ 0.85) unless visibility is poor.
- For "unknown" label, set a moderate confidence (0.50–0.70).
- For two-color output, confidence MUST strictly decrease: first > second. Equal confidence is FORBIDDEN.
- If you cannot decide which color covers more area, output "unknown" instead.

## Required output format (JSON only)

Person visible, single-colored garments:
{{
  "has_person": true,
  "upper": [{{"label": "black", "confidence": 0.92}}],
  "lower": [{{"label": "blue", "confidence": 0.87}}]
}}

Person visible, one part has a color-block design (TWO colors max):
{{
  "has_person": true,
  "upper": [{{"label": "white", "confidence": 0.88}}, {{"label": "blue", "confidence": 0.71}}],
  "lower": [{{"label": "black", "confidence": 0.90}}]
}}

Person visible, one part has stripes/plaid/pattern → label it "unknown":
{{
  "has_person": true,
  "upper": [{{"label": "black", "confidence": 0.91}}],
  "lower": [{{"label": "unknown", "confidence": 0.60}}]
}}

Person visible, part is completely hidden/occluded:
{{
  "has_person": true,
  "upper": [{{"label": "white", "confidence": 0.89}}],
  "lower": [{{"label": "unknown", "confidence": 0.50}}]
}}

No person visible:
{{"has_person": false}}

Output ONLY the JSON object — no markdown fences, no commentary."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_prompt() -> str:
    labels_list = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return PROMPT_TEMPLATE.replace(
        "{allowed_labels}", labels_list,
    ).replace(
        "{color_schema_json}",
        json.dumps(COLOR_SCHEMA, ensure_ascii=False, indent=2),
    )


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_from_text(text: str):
    """Robust JSON extraction from model output."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } pair
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def call_qwen(base_url: str, model: str, prompt: str, image_path: Path,
              timeout: int, temperature: float, max_tokens: int) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "You are a JSON-only clothing color labeler. "
                    "Output a single valid JSON object and nothing else."
                ),
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


def extract_message_text(response_json: dict) -> str:
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


def collect_images(frames_dir: Path) -> list[Path]:
    """Return sorted list of all .jpg files under frames_dir."""
    return sorted(frames_dir.rglob("*.jpg"))


# ---------------------------------------------------------------------------
# Single-image processor
# ---------------------------------------------------------------------------
def process_image(image_path: Path, args) -> dict:
    """Process one image. Returns a result dict to be saved as JSON."""
    # Build output path mirroring source structure under output dir
    rel = image_path.relative_to(args.frames_dir)
    out_path = args.output_dir / rel.with_suffix(".json")

    result = {
        "image_path": str(image_path),
        "output_path": str(out_path),
        "model": args.model,
        "base_url": args.base_url,
    }

    # Resume: skip if output already exists
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                result.update(existing)
                result["_resumed"] = True
                return result
        except Exception:
            pass  # corrupt file, re-process

    try:
        prompt = build_prompt()
        result["prompt"] = prompt

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if args.dry_run:
            result["qwen_raw_text"] = ""
            result["parsed"] = None
            result["note"] = "Dry run only"
            result["status"] = "success"
        else:
            response_json = call_qwen(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                image_path=image_path,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            raw_text = extract_message_text(response_json)
            parsed = parse_json_from_text(raw_text)
            result["qwen_response"] = response_json
            result["qwen_raw_text"] = raw_text
            result["parsed"] = parsed

            if parsed is None:
                result["status"] = "parse_error"
            elif parsed.get("has_person") is False:
                result["status"] = "no_person"
            elif "upper" not in parsed or "lower" not in parsed:
                # Model forgot to include upper/lower keys
                result["status"] = "parse_error"
                result["parse_warning"] = "Missing 'upper' or 'lower' key"
            else:
                # Validate labels are in ALLOWED_LABELS
                for part in ("upper", "lower"):
                    entries = parsed.get(part, [])
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict):
                                lbl = entry.get("label", "")
                                if lbl not in ALLOWED_LABELS:
                                    entry["label"] = "unknown"
                                    entry["_corrected_from"] = lbl
                                    result.setdefault("label_corrections", []).append(
                                        f"{part}: '{lbl}' → 'unknown'"
                                    )
                result["status"] = "success"

        # Write output immediately for resume support
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Qwen color annotation on extracted frames")
    p.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                   help="Root directory containing extracted .jpg frames")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write per-image result JSONs")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=120, help="API timeout per image (seconds)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--workers", type=int, default=8,
                   help="Number of concurrent threads")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip images whose output JSON already exists (default: True)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Reprocess all images")
    p.add_argument("--dry-run", action="store_true",
                   help="Render prompts without calling the API")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.frames_dir = args.frames_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.frames_dir)
    total = len(images)

    print(f"Frames dir   : {args.frames_dir}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Total images : {total}")
    print(f"Workers      : {args.workers}")
    print(f"Model        : {args.model}")
    print(f"Base URL     : {args.base_url}")
    print(f"Resume       : {args.resume}")
    print(f"Dry-run      : {args.dry_run}")
    print()

    if total == 0:
        print("No .jpg files found.")
        return 1

    # Stats
    stats = {"success": 0, "no_person": 0, "parse_error": 0, "error": 0, "resumed": 0}
    t_start = time.time()

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, img, args): img for img in images}

        for i, future in enumerate(as_completed(futures), 1):
            img_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                stats["error"] += 1
                print(f"[{i}/{total}] ERROR {img_path.name}: {exc}", flush=True)
                continue

            status = result.get("status", "unknown")
            if result.get("_resumed"):
                stats["resumed"] += 1
                # Also count the original status
                if status == "success":
                    stats["success"] += 1
                elif status == "no_person":
                    stats["no_person"] += 1
                continue

            if status == "success":
                stats["success"] += 1
            elif status == "no_person":
                stats["no_person"] += 1
            elif status == "parse_error":
                stats["parse_error"] += 1
            else:
                stats["error"] += 1

            # Print periodic progress
            if i % 100 == 0 or i == total or status == "error":
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"[{i}/{total}] ok:{stats['success']} no_person:{stats['no_person']} "
                      f"parse_err:{stats['parse_error']} err:{stats['error']} "
                      f"resumed:{stats['resumed']} | {rate:.1f} img/s | ETA:{eta/60:.0f}min",
                      flush=True)

            # Log failures immediately
            if status in ("parse_error", "error"):
                err_detail = result.get("error", result.get("qwen_raw_text", ""))[:120]
                print(f"  {status}: {img_path.name} — {err_detail}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  success:     {stats['success']}")
    print(f"  no_person:   {stats['no_person']}")
    print(f"  parse_error: {stats['parse_error']}")
    print(f"  error:       {stats['error']}")
    print(f"  resumed:     {stats['resumed']}")
    print(f"  total:       {total}")
    print(f"Output: {args.output_dir}")

    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
