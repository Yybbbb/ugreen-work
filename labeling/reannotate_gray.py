#!/usr/bin/env python3
"""
Two-step gray-verification re-annotation for Qwen color labels.

Problem:
  Qwen systematically over-predicts "gray" when uncertain about a garment's color.
  The ANALYSIS_REPORT shows gray Precision is only 60.0% (Upper) / 43.3% (Lower),
  meaning the majority of gray predictions are wrong.  However, ~40-57% of gray
  predictions ARE correct (genuinely gray garments).

Two-step strategy (implemented here):
  Step 1 — Verification:  For each image where Qwen predicted "gray", ask Qwen
           to re-examine the image with a verification preamble that explains gray
           is often over-predicted.  Qwen keeps the full color palette (gray still
           allowed) and re-annotates the image with a more discerning eye.
           The hypothesis: when forced to "double-check", Qwen will only stand by
           gray when it's truly confident, and will switch to the real color when
           gray was just a default guess.

  Step 2 — Merge:  Compare old vs new predictions for the gray parts.
           - If Qwen STILL says "gray" → keep gray (verified as genuine)
           - If Qwen now says a different color → adopt new color (corrected)
           - If Qwen didn't return the part → keep old gray (conservative fallback)

  This is superior to the one-step "always exclude gray" approach because:
    - Genuine gray garments are preserved (Qwen will double down on gray)
    - Only uncertain-default gray predictions get corrected
    - The merge logic is a simple equality check: old_label != new_label → change

Key design decision — the verification preamble:
  The preamble biases Qwen AGAINST gray for ambiguous cases ("when in doubt
  between gray and another color, pick the other color").  This shifts the
  decision boundary: only confident-gray predictions survive verification.

Usage:
  # Dry-run – scan and print how many gray predictions exist, show sample prompts
  python reannotate_gray.py --dry-run

  # Real run, output revised files to a separate directory (originals untouched)
  python reannotate_gray.py --output-dir ./annotation_gray_fixed

  # Process only 10 images for validation
  python reannotate_gray.py --output-dir ./annotation_gray_fixed --limit 10
"""

from __future__ import annotations

import argparse
import base64
import copy
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
DEFAULT_ANNOTATION_DIR = SCRIPT_DIR / "annotation"
SERVER_IP = "192.168.111.17"
DEFAULT_BASE_URL = f"http://{SERVER_IP}:6096/v1"
DEFAULT_MODEL = "Qwen35-35b"

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


# ---------------------------------------------------------------------------
# Prompt construction (Step 1 — Verification)
# ---------------------------------------------------------------------------

def _gray_parts_to_text(gray_parts: set[str]) -> str:
    """Convert {'upper', 'lower'} → a readable English phrase for the prompt."""
    parts = sorted(gray_parts)
    if len(parts) == 2:
        return "upper and lower"
    return parts[0]  # "upper" or "lower"


VERIFICATION_PREAMBLE_TEMPLATE = (
    "/no_think\n\n"
    "## ⚠ GRAY VERIFICATION — READ CAREFULLY BEFORE ANNOTATING\n\n"
    "The previous annotation run labeled the **{gray_parts_text}** garment as \"gray\".\n"
    'However, \"gray\" is frequently over-predicted as a **safe default** when the\n'
    "true color is ambiguous.  Common failure modes:\n"
    "- Poorly-lit **black** fabric appearing grayish\n"
    "- Faded or washed-out **blue** looking gray\n"
    "- Off-**white** in shadow appearing gray\n"
    "- Muted **green** / **brown** being mistaken for gray\n\n"
    "### Your task:\n"
    "Re-examine the **{gray_parts_text}** garment with extra scrutiny.\n"
    "- If the fabric's TRUE pigment IS genuinely gray → keep \"gray\"\n"
    "- If it is actually another color that only *appears* grayish due to lighting,\n"
    "  shadows, fading, or image quality → output the TRUE color instead\n\n"
    "### Tie-breaking rule:\n"
    "When you are on the fence between \"gray\" and any other color,\n"
    "**pick the other color**.  Only output \"gray\" when you are highly confident\n"
    "the garment is truly gray pigment, not a gray-ish version of another color.\n\n"
    "### For parts NOT originally labeled gray:\n"
    "Annotate them normally — this verification only applies to the {gray_parts_text} part(s).\n\n"
    "---\n\n"
)


def build_verification_prompt(original_prompt: str, gray_parts: set[str]) -> str:
    """Prepend a gray-verification preamble to the original prompt.

    The original prompt (with full color schema including gray) is kept intact.
    We only prepend the verification preamble that:
      - Names which parts need gray re-examination
      - Explains gray over-prediction failure modes
      - Biases Qwen *against* gray in ambiguous cases
      - Instructs Qwen to annotate normally for non-gray parts

    The preamble is inserted AFTER the original /no_think directive (which we
    strip and re-add at the top of the preamble).
    """
    # Strip original /no_think — our preamble carries its own
    prompt_body = original_prompt.removeprefix("/no_think\n").removeprefix("/no_think")

    preamble = VERIFICATION_PREAMBLE_TEMPLATE.format(
        gray_parts_text=_gray_parts_to_text(gray_parts),
    )

    return preamble + prompt_body


# ---------------------------------------------------------------------------
# Qwen API call
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def has_gray_prediction(parsed: dict[str, Any] | None) -> set[str]:
    """Return set of parts ('upper', 'lower') where top-1 label is 'gray'."""
    if not parsed:
        return set()
    gray_parts = set()
    for part in ("upper", "lower"):
        colors = parsed.get(part)
        if isinstance(colors, list) and len(colors) > 0:
            if isinstance(colors[0], dict) and colors[0].get("label") == "gray":
                gray_parts.add(part)
    return gray_parts


def revise_annotation(
    annotation_path: Path,
    output_dir: Path,
    base_url: str,
    model: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Step 1 (verification) + Step 2 (merge) for one annotation file.

    Step 1:  Build a verification prompt and call Qwen.
    Step 2:  For each gray part, compare old vs new top-1 label:
               - same → keep original gray (Qwen verified it)
               - different → adopt new label (Qwen corrected itself)
               - missing → keep original gray (conservative fallback)

    Returns a dict describing what happened, for the change log.
    """
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    image_path_str = data.get("image_path", "")
    image_path = Path(image_path_str)
    if not image_path.exists():
        return {"status": "skip", "reason": f"image not found: {image_path_str}"}

    parsed = data.get("parsed")
    gray_parts = has_gray_prediction(parsed)

    if not gray_parts:
        return {"status": "skip", "reason": "no gray prediction"}

    original_prompt = data.get("prompt", "")
    if not original_prompt:
        return {"status": "skip", "reason": "no prompt in annotation"}

    # ---- Step 1: Build verification prompt ----
    verification_prompt = build_verification_prompt(original_prompt, gray_parts)

    result: dict[str, Any] = {
        "status": "pending",
        "image_path": image_path_str,
        "gray_parts": sorted(gray_parts),
    }

    if dry_run:
        result["status"] = "dry_run"
        result["verification_prompt"] = verification_prompt
        return result

    # ---- Call Qwen with verification prompt ----
    try:
        response_json = call_qwen(
            base_url=base_url,
            model=model,
            prompt=verification_prompt,
            image_path=image_path,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text = extract_message_text(response_json)
        new_parsed = parse_json_from_text(raw_text)

        result["status"] = "success"
        result["qwen_raw_text"] = raw_text
        result["new_parsed"] = new_parsed

        # ---- Step 2: Merge — only replace gray parts, keep non-gray parts intact ----
        merged = copy.deepcopy(parsed) if parsed else {"upper": [], "lower": []}
        old_labels: dict[str, list[str] | None] = {}
        new_labels: dict[str, list[str] | None] = {}
        decisions: dict[str, str] = {}  # "kept_gray" | "changed_to_<color>" | "missing_kept_gray"

        for part in gray_parts:
            old_colors = merged.get(part, [])
            old_labels[part] = (
                [c.get("label") for c in old_colors]
                if isinstance(old_colors, list)
                else None
            )

            if isinstance(new_parsed, dict) and part in new_parsed:
                new_colors = new_parsed[part]
                new_top1_label = (
                    new_colors[0].get("label")
                    if isinstance(new_colors, list) and len(new_colors) > 0 and isinstance(new_colors[0], dict)
                    else None
                )

                if new_top1_label == "gray":
                    # Qwen verified: it's truly gray → keep original
                    decisions[part] = "kept_gray"
                    new_labels[part] = old_labels[part]
                    # merged stays unchanged for this part
                elif new_top1_label is not None:
                    # Qwen corrected: it's a different color → adopt
                    decisions[part] = f"changed_to_{new_top1_label}"
                    merged[part] = new_colors
                    new_labels[part] = (
                        [c.get("label") for c in new_colors]
                        if isinstance(new_colors, list)
                        else None
                    )
                else:
                    # Parsed but empty → keep old
                    decisions[part] = "missing_kept_gray"
                    new_labels[part] = old_labels[part]
            else:
                # Qwen didn't return this part → keep old (conservative)
                decisions[part] = "missing_kept_gray"
                new_labels[part] = old_labels[part]

        # ---- Write revised annotation ----
        revised_data = copy.deepcopy(data)
        revised_data["parsed"] = merged
        revised_data["gray_revision"] = {
            "gray_parts": sorted(gray_parts),
            "decisions": decisions,
            "old_labels": old_labels,
            "new_labels": new_labels,
            "verification_prompt": verification_prompt,
            "verification_raw_text": raw_text,
        }

        # Determine output path
        if annotation_path.is_relative_to(DEFAULT_ANNOTATION_DIR):
            rel_path = annotation_path.relative_to(DEFAULT_ANNOTATION_DIR)
        else:
            rel_path = annotation_path.name
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(revised_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        result["output_path"] = str(out_path)
        result["old_labels"] = old_labels
        result["new_labels"] = new_labels
        result["decisions"] = decisions

    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def iter_annotation_files(annotation_dir: Path):
    """Yield Paths to all annotation JSON files in sorted order."""
    for p in sorted(annotation_dir.glob("*.json")):
        # Skip the change log itself if it exists in the dir
        if p.name == "gray_revision_log.json":
            continue
        yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-step gray verification: re-ask Qwen to verify gray predictions."
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing per-image annotation JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write revised JSONs here (default: overwrite in-place after confirmation).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print what would change without calling Qwen.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N gray images (0 = unlimited).",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        nargs="*",
        help="Only process specific image IDs (filename stems).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_dir = args.annotation_dir.resolve()
    output_dir = (args.output_dir or annotation_dir).resolve()

    if output_dir == annotation_dir and not args.dry_run:
        print("⚠  WARNING: Will overwrite original annotation files IN-PLACE.")
        print(f"   Target directory: {annotation_dir}")
        response = input("   Continue? [y/N] ")
        if response.strip().lower() != "y":
            print("Aborted.")
            return 0

    annotation_files = list(iter_annotation_files(annotation_dir))
    total = len(annotation_files)
    print(f"Annotation dir : {annotation_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Total files    : {total}")
    print(f"Dry-run        : {args.dry_run}")
    print(f"Limit          : {args.limit if args.limit else 'unlimited'}")
    print()

    gray_count = 0
    revised_count = 0
    skip_count = 0
    error_count = 0
    # Stats for the two-step decisions
    stats_kept = 0       # Qwen verified gray → kept
    stats_changed = 0    # Qwen corrected to different color
    stats_missing = 0    # Qwen didn't return part → kept gray (fallback)
    change_log: list[dict[str, Any]] = []

    for i, ann_path in enumerate(annotation_files, 1):
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            parsed = data.get("parsed")

            gray_parts = has_gray_prediction(parsed)
            if not gray_parts:
                continue

            gray_count += 1

            if args.sample_ids and ann_path.stem not in args.sample_ids:
                continue

            if args.limit and revised_count >= args.limit:
                continue

            stem = ann_path.stem
            print(f"[{i}/{total}] #{gray_count} VERIFY {stem}  gray_parts={sorted(gray_parts)}",
                  flush=True)

            result = revise_annotation(
                annotation_path=ann_path,
                output_dir=output_dir,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                dry_run=args.dry_run,
            )

            if result["status"] == "success":
                revised_count += 1
                change_log.append(result)
                decisions = result.get("decisions", {})
                for part in sorted(gray_parts):
                    d = decisions.get(part, "?")
                    old = (result.get("old_labels") or {}).get(part)
                    new = (result.get("new_labels") or {}).get(part)
                    print(f"  {part}: {old} → {new}  [{d}]", flush=True)

                    if d == "kept_gray":
                        stats_kept += 1
                    elif d.startswith("changed_to_"):
                        stats_changed += 1
                    elif d == "missing_kept_gray":
                        stats_missing += 1

            elif result["status"] == "dry_run":
                revised_count += 1
                change_log.append(result)
            elif result["status"] == "skip":
                skip_count += 1
            else:
                error_count += 1
                print(f"  ERROR: {result.get('error', 'unknown')}", flush=True)

        except Exception as exc:
            error_count += 1
            print(f"[{i}/{total}] ERROR {ann_path.stem}: {exc}", flush=True)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done — total={total}, gray_found={gray_count}, verified={revised_count}, "
          f"skipped={skip_count}, errors={error_count}")
    print(f"Step-2 decisions:")
    print(f"  kept_gray     = {stats_kept}   (Qwen verified: truly gray)")
    print(f"  changed       = {stats_changed}   (Qwen corrected to another color)")
    print(f"  missing_kept  = {stats_missing}   (Qwen didn't return part, kept gray)")
    print(f"  total parts   = {stats_kept + stats_changed + stats_missing}")

    # Save change log
    log_path = output_dir / "gray_revision_log.json"
    log_path.write_text(
        json.dumps(change_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Change log written to: {log_path}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
