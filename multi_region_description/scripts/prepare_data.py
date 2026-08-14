#!/usr/bin/env python3
"""
Construct REGIONS_TO_DESCRIPTIONS training data from REGION_TO_DESCRIPTION inference results.

Input:  REGION_TO_DESCRIPTION inference JSON files
Output: REGIONS_TO_DESCRIPTIONS train/test JSONL files

Plan A (default) — implicit order correspondence, <sep>-delimited descriptions:
  {
    "image":  "/abs/path/to/image.jpg",
    "prompt": "<REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep><loc_...>...",
    "label":  "description1<sep>description2<sep>..."
  }

Plan B (--plan-b) — description-first, loc token appended, no <sep> in label:
  {
    "image":  "/abs/path/to/image.jpg",
    "prompt": "<REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep><loc_...>...",
    "label":  "description1<loc_x1><loc_y1><loc_x2><loc_y2>description2<loc_x1><loc_y1><loc_x2><loc_y2>..."
  }
  Aligns with Florence-2 native DENSE_REGION_CAPTION grammar (text-first, loc-appended).
  loc run of 4 tokens acts as an unambiguous delimiter — <sep> is redundant in the label.

Plan C (--label-format loc-sep) — description-first, loc token appended, <sep>-delimited:
  {
    "image":  "/abs/path/to/image.jpg",
    "prompt": "<REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep><loc_...>...",
    "label":  "description1<loc_x1><loc_y1><loc_x2><loc_y2><sep>description2<loc_x1><loc_y1><loc_x2><loc_y2>..."
  }

Rules (both plans):
  - Only regions with status == "ok" are used.
  - Images with 0 valid regions are skipped.
  - If an image has more than MAX_CROPS valid regions, keep the MAX_CROPS largest by crop_area.
  - Single-region images are kept: the model should handle N=1 gracefully too.
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_CROPS = 5
SEP = "<sep>"
DEFAULT_INPUT_DIR = "/data/work/MichaelYu/florence-data/ugipc-person-region-descriptions-yolo26m-no-name"
DEFAULT_OUTPUT_DIR_PLAN_A = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "ugipc-person-region-descriptions-yolo26m-no-name"
)
DEFAULT_OUTPUT_DIR_PLAN_B = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "ugipc-person-region-descriptions-yolo26m-no-name-plan-b"
)
LABEL_FORMATS = ("sep", "loc", "loc-sep")
DESCRIPTION_SOURCES = ("florence", "qwen")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def loc_tokens(bbox_loc: list[int]) -> str:
    """Convert a [x1, y1, x2, y2] bbox_loc_0_999 list to a <loc_XXX> token string."""
    x1, y1, x2, y2 = bbox_loc
    return f"<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"


def collect_json_files(input_dir: Path, manifest_file: str) -> list[Path]:
    """
    Collect inference JSON files.

    The no-name dataset ships a manifest that records the exact generated output
    JSON for each source image. Prefer that over a filesystem scan so split/merge
    manifests and future partial reruns stay reproducible. Passing an empty
    --manifest-file falls back to scanning input_dir/images.
    """
    if manifest_file:
        manifest_path = Path(manifest_file)
        if not manifest_path.is_absolute():
            manifest_path = input_dir / manifest_path
        if manifest_path.exists():
            json_files = []
            seen = set()
            skipped_missing_path = 0
            skipped_missing_file = 0
            with open(manifest_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        log.warning("Skipping malformed manifest line %d in %s: %s", line_no, manifest_path, exc)
                        continue
                    output_json_path = record.get("output_json_path")
                    if not output_json_path:
                        skipped_missing_path += 1
                        continue
                    json_path = Path(output_json_path)
                    if not json_path.is_absolute():
                        json_path = input_dir / json_path
                    if not json_path.exists():
                        skipped_missing_file += 1
                        continue
                    if json_path not in seen:
                        seen.add(json_path)
                        json_files.append(json_path)
            if skipped_missing_path or skipped_missing_file:
                log.warning(
                    "Manifest skipped missing_path=%d missing_file=%d",
                    skipped_missing_path,
                    skipped_missing_file,
                )
            log.info("Collected %d JSON files from manifest %s", len(json_files), manifest_path)
            return json_files
        log.warning("Manifest %s does not exist; falling back to images/ scan", manifest_path)

    scan_dir = input_dir / "images"
    if not scan_dir.is_dir():
        scan_dir = input_dir
    json_files = sorted(scan_dir.rglob("*.json"))
    log.info("Collected %d JSON files from %s", len(json_files), scan_dir)
    return json_files


def build_sample(image_path: str, regions: list[dict], label_format: str = "sep") -> dict:
    """
    Build one training sample from a list of valid region dicts.

    regions must already be filtered (status==ok) and sliced to ≤ MAX_CROPS.
    Order is preserved — caller decides the order (largest-first).

    Plan A (default): label = "desc1<sep>desc2<sep>..."
    Plan B: label = "desc1<loc><loc><loc><loc>desc2<loc><loc><loc><loc>..."
      Aligns with Florence-2 native DENSE_REGION_CAPTION grammar; loc run of 4
      tokens acts as an unambiguous delimiter — <sep> is redundant in the label.
    """
    loc_parts = [loc_tokens(r["bbox_loc_0_999"]) for r in regions]
    desc_parts = [r["text"].strip() for r in regions]

    prompt = f"<REGIONS_TO_DESCRIPTIONS>{SEP.join(loc_parts)}"
    if label_format == "loc":
        label = "".join(desc + loc for desc, loc in zip(desc_parts, loc_parts))
    elif label_format == "loc-sep":
        label = SEP.join(desc + loc for desc, loc in zip(desc_parts, loc_parts))
    else:
        label = SEP.join(desc_parts)

    return {"image": image_path, "prompt": prompt, "label": label}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def process_json(
    json_path: Path,
    label_format: str = "sep",
    allowed_crop_counts: set[int] | None = None,
    description_source: str = "florence",
) -> tuple[dict, int] | tuple[None, int]:
    """
    Process one inference JSON file and return (sample, n_regions), or (None, 0) if skipped.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    image_path = data["image"]["path"]

    if isinstance(data.get("crops"), dict):
        valid_crop_count = data.get("valid_crop_count")
        if allowed_crop_counts is not None and valid_crop_count not in allowed_crop_counts:
            return None, 0

        valid = []
        crops = sorted(
            data["crops"].values(),
            key=lambda crop: crop.get("crop_index", 0),
        )
        for crop in crops:
            if description_source == "qwen":
                qwen = crop.get("qwen", {})
                qwen_description = qwen.get("description")
                description = (
                    qwen_description.strip()
                    if qwen.get("status") == "success" and isinstance(qwen_description, str)
                    else ""
                )
            else:
                description = crop.get("florence", {}).get("description", "").strip()
            bbox_loc = crop.get("bbox_loc_0_999")
            if description and isinstance(bbox_loc, list) and len(bbox_loc) == 4:
                valid.append({"text": description, "bbox_loc_0_999": bbox_loc})

        if valid_crop_count != len(valid):
            raise ValueError(
                f"valid_crop_count={valid_crop_count} but found {len(valid)} usable crops"
            )
        return build_sample(image_path, valid, label_format=label_format), len(valid)

    valid = [r for r in data.get("regions", []) if r.get("status") == "ok"]

    # Skip only when the image has no usable crop at all.
    if len(valid) == 0:
        return None, 0

    # Deduplicate by description text: the source REGION_TO_DESCRIPTION model
    # often returns the identical caption for several boxes (same person detected
    # twice, or a generic scene-level description). Keeping duplicates would teach
    # the model to ignore the bbox and echo one caption, which defeats this task.
    # For each unique description keep only the region with the largest crop_area.
    best_by_desc: dict[str, dict] = {}
    for r in valid:
        desc = r["text"].strip()
        area = r["source_detection"]["crop_area"]
        if desc not in best_by_desc or area > best_by_desc[desc]["source_detection"]["crop_area"]:
            best_by_desc[desc] = r
    valid = list(best_by_desc.values())

    # If there are more than MAX_CROPS regions, keep the largest by crop_area.
    if len(valid) > MAX_CROPS:
        valid.sort(key=lambda r: r["source_detection"]["crop_area"], reverse=True)
        valid = valid[:MAX_CROPS]

    if allowed_crop_counts is not None and len(valid) not in allowed_crop_counts:
        return None, 0

    return build_sample(image_path, valid, label_format=label_format), len(valid)


def main():
    parser = argparse.ArgumentParser(description="Build REGIONS_TO_DESCRIPTIONS training data")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Root dir of ugipc no-name inference results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to plan-a or plan-b subdir under data/). Created if missing.",
    )
    parser.add_argument(
        "--plan-b",
        action="store_true",
        default=False,
        help="Build Plan B data: description-first label with loc tokens appended, no <sep> in label.",
    )
    parser.add_argument(
        "--label-format",
        choices=LABEL_FORMATS,
        default="sep",
        help="Label grammar: sep, loc, or loc-sep. --plan-b is an alias for loc.",
    )
    parser.add_argument(
        "--allowed-crop-counts",
        type=int,
        nargs="*",
        default=None,
        help="Keep only frames whose valid_crop_count is in this list.",
    )
    parser.add_argument(
        "--description-source",
        choices=DESCRIPTION_SOURCES,
        default="florence",
        help="Crop description field to use: florence.description or successful qwen.description.",
    )
    parser.add_argument(
        "--manifest-file",
        default="manifest.jsonl",
        help="Manifest filename/path under input-dir. Set empty to scan input-dir/images/**/*.json.",
    )
    parser.add_argument(
        "--train-file",
        default="train.jsonl",
        help="Train split JSONL filename",
    )
    parser.add_argument(
        "--test-file",
        default="test.jsonl",
        help="Test split JSONL filename (held out, not used for training)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Fraction of samples held out as the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    label_format = "loc" if args.plan_b else args.label_format
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR_PLAN_B if label_format == "loc" else DEFAULT_OUTPUT_DIR_PLAN_A)

    input_dir = Path(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Label format: %s", label_format)
    log.info("Description source: %s", args.description_source)
    log.info("Output dir: %s", output_dir)

    json_files = collect_json_files(input_dir, args.manifest_file)

    samples = []
    skipped_samples = 0
    n_regions_hist: dict[int, int] = {}

    for json_path in json_files:
        try:
            sample, n = process_json(
                json_path,
                label_format=label_format,
                description_source=args.description_source,
                allowed_crop_counts=(
                    set(args.allowed_crop_counts)
                    if args.allowed_crop_counts is not None
                    else None
                ),
            )
        except Exception as exc:
            log.warning("Failed to process %s: %s", json_path, exc)
            continue

        if sample is None:
            skipped_samples += 1
            continue

        n_regions_hist[n] = n_regions_hist.get(n, 0) + 1
        samples.append(sample)

    # Shuffle, then split off a held-out test set.
    random.seed(args.seed)
    random.shuffle(samples)

    n_test = int(round(len(samples) * args.test_ratio))
    test_samples = samples[:n_test]
    train_samples = samples[n_test:]

    train_path = output_dir / args.train_file
    test_path = output_dir / args.test_file
    for path, split in ((train_path, train_samples), (test_path, test_samples)):
        with open(path, "w", encoding="utf-8") as f:
            for sample in split:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    log.info("Total kept=%d  skipped(filtered or unusable)=%d", len(samples), skipped_samples)
    log.info("Train=%d → %s", len(train_samples), train_path)
    log.info("Test =%d → %s", len(test_samples), test_path)
    log.info("Region-count distribution over ALL kept samples:")
    for n in sorted(n_regions_hist):
        log.info("  N=%d regions: %d samples", n, n_regions_hist[n])


if __name__ == "__main__":
    main()
