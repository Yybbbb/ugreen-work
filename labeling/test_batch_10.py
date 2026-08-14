#!/usr/bin/env python3
"""Batch test: label 10 randomly sampled entries from train.txt and copy source images.

Outputs per entry:
  labeling/annotation/<stem>.json      – Qwen color labeling result
  labeling/images/<stem><ext>          – copy of the original image
  labeling/org_annotation/<stem>.json  – copy of the original annotation JSON
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Please install requests first: pip install requests")

# Re-use all helpers from the main labeling script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_upper_lower_color import process_entry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ROOT = SCRIPT_DIR.parent / "data" / "LIP_clothes_accessory_unified"
SPLIT_FILE   = DATASET_ROOT / "splits" / "clothes_accessory_poseprior" / "train.txt"
PROMPT_FILE  = SCRIPT_DIR / "qwen_upper_lower_color_prompt.txt"

SERVER_IP  = "192.168.111.17"
BASE_URL   = f"http://{SERVER_IP}:6096/v1"
MODEL      = "Qwen35-35b"

ANNOTATION_DIR     = SCRIPT_DIR / "annotation"
IMAGES_DIR         = SCRIPT_DIR / "images"
ORG_ANNOTATION_DIR = SCRIPT_DIR / "org_annotation"

N_LINES    = 10
TIMEOUT    = 120
TEMPERATURE = 0.0
MAX_TOKENS  = 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ORG_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = [l.strip() for l in SPLIT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    sampled   = random.sample(all_lines, min(N_LINES, len(all_lines)))
    entries   = [l.split() for l in sampled]
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    print(f"Processing {len(entries)} entries from {SPLIT_FILE.name}")
    print(f"  Annotations     → {ANNOTATION_DIR}")
    print(f"  Images          → {IMAGES_DIR}")
    print(f"  Org annotations → {ORG_ANNOTATION_DIR}\n")

    success = skipped = errors = 0

    for i, parts in enumerate(entries, 1):
        if len(parts) < 2:
            print(f"[{i:2d}/{N_LINES}] SKIP  bad line: {parts}")
            skipped += 1
            continue

        image_path      = DATASET_ROOT / parts[0]
        annotation_path = DATASET_ROOT / parts[1]
        stem            = image_path.stem

        print(f"[{i:2d}/{N_LINES}] {stem} ...", end=" ", flush=True)

        try:
            # Save annotation JSON
            out_json = ANNOTATION_DIR / f"{stem}.json"
            ok = process_entry(
                image_path=image_path,
                annotation_path=annotation_path,
                prompt_template=prompt_template,
                output_path=out_json,
                base_url=BASE_URL,
                model=MODEL,
                timeout=TIMEOUT,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                dry_run=False,
            )
            result = json.loads(out_json.read_text(encoding="utf-8"))

            # Copy source image
            if image_path.exists():
                shutil.copy2(image_path, IMAGES_DIR / image_path.name)

            # Copy original annotation JSON
            if annotation_path.exists():
                shutil.copy2(annotation_path, ORG_ANNOTATION_DIR / annotation_path.name)

            parsed = result.get("parsed")
            note   = result.get("note", "")
            if not ok:
                print(f"ERROR  {result.get('error_type')}: {result.get('error')}")
                errors += 1
            elif note:
                print(f"skipped ({note[:60]})")
                skipped += 1
            elif parsed:
                def fmt_part(entries):
                    parts = []
                    for c in entries:
                        s = c["label"]
                        if c.get("inferred"):
                            s += "(inferred)"
                        parts.append(s)
                    return parts
                upper = fmt_part(parsed.get("upper", []))
                lower = fmt_part(parsed.get("lower", []))
                print(f"OK  upper={upper}  lower={lower}")
                success += 1
            else:
                print(f"OK  (parsed=None, check json)")
                success += 1

        except Exception as exc:
            print(f"ERROR  {exc}")
            errors += 1

    print(f"\nDone — success: {success}, skipped: {skipped}, errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
