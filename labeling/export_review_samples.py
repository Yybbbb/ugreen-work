#!/usr/bin/env python3
"""Export a random JSON sample of annotation results for manual review."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_DIR = SCRIPT_DIR / "annotation"
DEFAULT_OUTPUT_IMAGE_DIR = SCRIPT_DIR / "images"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "review_samples.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_review_candidates(annotation_dir: Path) -> list[dict[str, Any]]:
    candidates = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        data = load_json(annotation_path)
        if data.get("status") != "success":
            continue

        parsed = data.get("parsed")
        image_path_value = data.get("image_path")
        if not isinstance(parsed, dict) or not image_path_value:
            continue

        image_path = Path(image_path_value)
        if not image_path.exists():
            continue

        candidates.append(
            {
                "image_path": image_path,
                "upper": parsed.get("upper", []),
                "lower": parsed.get("lower", []),
            }
        )
    return candidates


def build_review_entry(candidate: dict[str, Any], copied_image_path: Path) -> dict[str, Any]:
    return {
        "image_name": copied_image_path.name,
        "upper": candidate["upper"],
        "lower": candidate["lower"],
    }


def export_review_samples(
    annotation_dir: Path,
    output_image_dir: Path,
    json_path: Path,
    sample_size: int = 20,
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    candidates = collect_review_candidates(annotation_dir)
    if not candidates:
        raise ValueError(f"No reviewable annotations found in {annotation_dir}")

    actual_size = min(sample_size, len(candidates))
    sampled = random.Random(seed).sample(candidates, actual_size)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for candidate in sampled:
        source_image_path = candidate["image_path"]
        copied_image_path = output_image_dir / source_image_path.name
        if copied_image_path.resolve() != source_image_path.resolve():
            shutil.copy2(source_image_path, copied_image_path)
        entries.append(build_review_entry(candidate, copied_image_path))

    json_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample annotation results and export a JSON file for manual review."
    )
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = export_review_samples(
        annotation_dir=args.annotation_dir.resolve(),
        output_image_dir=args.output_image_dir.resolve(),
        json_path=args.json_path.resolve(),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Exported {len(entries)} samples to {args.json_path}")
    print(f"Copied images to {args.output_image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
