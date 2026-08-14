#!/usr/bin/env python3
"""
SAM3 Annotation Script — UPAR Rare Color Samples
==================================================
使用 SAM3 Docker 服务 (server_batch.py, :8010) 对 rare_color_samples.txt
中的所有图片进行人体属性标注。

属性类别与 LIP_clothes_accessory_unified 参考格式一致:
    upper, lower, dress, hat, glasses, mask, hair

用法:
    python sam3_annotate_upar_rare_color.py

可随时中断 (Ctrl+C)，已完成的图片会自动跳过 (resume)。
"""

import json
import os
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAM3_SERVER = "http://localhost:8010"
SAMPLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extradata", "upar-dataset", "rare_color_samples.txt",
)
DATA_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extradata", "upar-dataset", "data",
)
OUTPUT_ANNOTATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "LIP_clothes_accessory_unified", "annotations", "upar_rare_color_sam3",
)
OUTPUT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "LIP_clothes_accessory_unified", "sam3_upar_rare_color_completion.jsonl",
)
OUTPUT_SUMMARY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "LIP_clothes_accessory_unified", "sam3_upar_rare_color_summary.json",
)

CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to catch more candidates
MAX_WORKERS = 4            # Concurrent requests (GPU serializes inference anyway)
REQUEST_TIMEOUT = 60       # Seconds per image
SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Prompt mapping: attribute → SAM3 text prompt
# ---------------------------------------------------------------------------
ATTRIBUTE_PROMPTS = [
    {"tag": "upper",   "prompt": "upper garment",  "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "lower",   "prompt": "lower garment",   "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "dress",   "prompt": "dress",           "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "hat",     "prompt": "hat",             "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "glasses", "prompt": "glasses",         "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "mask",    "prompt": "face mask",       "confidence_threshold": CONFIDENCE_THRESHOLD},
    {"tag": "hair",    "prompt": "hair",            "confidence_threshold": CONFIDENCE_THRESHOLD},
]

CLASSES = ["upper", "lower", "dress", "hat", "glasses", "mask", "hair"]


def parse_samples(filepath: str) -> list[dict]:
    """解析 rare_color_samples.txt，返回 [{image_relpath, dataset, filename}, ...]"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if not parts[0]:
                continue

            # Train samples: path starts with "Market1501/..." or similar
            # Val_task2 samples: no path prefix (only attributes)
            first_field = parts[0]
            if "/" in first_field:
                # Has image path prefix
                image_rel = first_field
            else:
                # Val_task2: no image path, skip (we need image paths)
                # print(f"  [warn] Line {line_num}: no image path, skipping val_task2 entry")
                continue

            # Extract dataset name (first directory component)
            dataset = image_rel.split("/")[0]
            filename = os.path.basename(image_rel)

            samples.append({
                "image_relpath": image_rel,
                "dataset": dataset,
                "filename": filename,
                "line_num": line_num,
            })

    return samples


def find_image_path(sample: dict) -> str | None:
    """Find the actual image file on disk. Returns full path or None."""
    image_rel = sample["image_relpath"]
    # Try direct path under DATA_BASE
    direct = os.path.join(DATA_BASE, image_rel)
    if os.path.isfile(direct):
        return direct

    # Try under bounding_box_train, bounding_box_test, query
    dataset = sample["dataset"]
    filename = sample["filename"]
    for subdir in ["bounding_box_train", "bounding_box_test", "query"]:
        alt = os.path.join(DATA_BASE, dataset, subdir, filename)
        if os.path.isfile(alt):
            return alt

    # Try recursive find (slow but exhaustive for any structure)
    for root, dirs, files in os.walk(os.path.join(DATA_BASE, dataset)):
        if filename in files:
            return os.path.join(root, filename)

    return None


def make_sample_id(sample: dict) -> str:
    """Generate a unique sample_id for the output JSON."""
    # Use hash of the image path for uniqueness
    identifier = sample["image_relpath"].replace("/", "_").replace(".", "_")
    # Truncate to a reasonable length
    return f"upar_{identifier}_p000"


def make_output_json_path(sample_id: str) -> str:
    """Generate the output annotation JSON path."""
    return os.path.join(OUTPUT_ANNOTATIONS_DIR, f"{sample_id}.json")


def to_attr_dict() -> dict:
    """Create an empty attribute entry (unknown / no mask)."""
    return {
        "label": "unknown",
        "has_mask": False,
        "bbox": None,
        "mask": None,
        "source_class": None,
        "score": 0.0,
        "note": None,
    }


def call_sam3_batch(image_path: str) -> dict | None:
    """Call SAM3 /annotate_batch with all 7 prompts. Returns JSON result or None on error."""
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{SAM3_SERVER}/annotate_batch",
                files={"image": (os.path.basename(image_path), f)},
                data={"prompts": json.dumps(ATTRIBUTE_PROMPTS)},
                timeout=REQUEST_TIMEOUT,
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] SAM3 call failed for {image_path}: {e}")
        return None


def convert_sam3_result_to_attributes(
    sam3_result: dict,
    img_width: int,
    img_height: int,
) -> dict:
    """
    Convert SAM3 batch result to attributes dict matching reference format.

    SAM3 returns for each tag:
        {
            "num_masks": N,
            "pred_scores": [...],   # sorted high→low
            "pred_boxes": [[x,y,w,h], ...],  # normalized 0~1
            "pred_masks": ["<RLE counts string>", ...],
        }

    We take the top-scoring mask for each attribute (if num_masks > 0).
    """
    attributes = {}
    results = sam3_result.get("results", {})

    for attr_name in CLASSES:
        attr_result = results.get(attr_name)
        if attr_result is None or attr_result.get("status") != "success":
            attributes[attr_name] = to_attr_dict()
            attributes[attr_name]["label"] = "unknown"
            continue

        num_masks = attr_result.get("num_masks", 0)
        if num_masks == 0:
            attributes[attr_name] = to_attr_dict()
            # For hat/glasses/mask: if SAM3 returns empty, label as "no"
            if attr_name in ("hat", "glasses", "mask"):
                attributes[attr_name]["label"] = "no"
            else:
                attributes[attr_name]["label"] = "unknown"
            continue

        # Take top-scoring mask
        best_score = attr_result["pred_scores"][0]
        best_box = attr_result["pred_boxes"][0]  # [x, y, w, h] normalized
        best_mask_counts = attr_result["pred_masks"][0]  # RLE counts string

        # Clip bbox to valid [0, 1] range (SAM3 sometimes produces small negatives)
        best_box = [
            max(0.0, min(1.0, best_box[0])),
            max(0.0, min(1.0, best_box[1])),
            max(0.0, min(1.0, best_box[2])),
            max(0.0, min(1.0, best_box[3])),
        ]

        # Determine label
        if attr_name in ("hat", "glasses", "mask"):
            # Binary: has the item or not
            label = attr_name if best_score >= 0.3 else "no"
        else:
            # Clothing/hair: label with the attribute name
            label = attr_name if best_score >= 0.3 else "unknown"

        has_mask = best_score >= 0.3

        attributes[attr_name] = {
            "label": label,
            "has_mask": has_mask,
            "bbox": best_box if has_mask else None,
            "mask": {
                "format": "coco_rle",
                "rle": {
                    "size": [img_height, img_width],
                    "counts": best_mask_counts,
                },
            } if has_mask else None,
            "source_class": None,  # SAM3 doesn't provide fine-grained class
            "score": round(best_score, 8),
            "note": "generated by SAM3 via sam3-annotation-local batch API",
        }

    return attributes


def build_annotation_json(sample: dict, image_path: str, attributes: dict, elapsed: float) -> dict:
    """Build the complete annotation JSON matching the reference format."""
    # Get image dimensions
    from PIL import Image
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception:
        img_width, img_height = 0, 0

    sample_id = make_sample_id(sample)
    output_json_rel = f"annotations/upar_rare_color_sam3/{sample_id}.json"
    image_rel = f"images/upar_rare_color/{sample['image_relpath']}"

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"upar_rare_color_sam3/{sample_id}",
        "source": {
            "name": "upar_rare_color",
            "image_path": image_path,
            "source_image": sample["filename"],
            "dataset": sample["dataset"],
            "original_relpath": sample["image_relpath"],
        },
        "image": {
            "path": image_rel,
            "width": img_width,
            "height": img_height,
            "is_person_crop": False,  # Market1501 images are full person crops
        },
        "annotation_path": output_json_rel,
        "classes": CLASSES,
        "attributes": attributes,
        "sam3_unified_completion": {
            "updated_at": datetime.now().isoformat(),
            "classes_requested": CLASSES,
            "classes_updated": sum(1 for a in attributes.values() if a.get("has_mask")),
            "api": "annotate_batch",
            "server": SAM3_SERVER,
            "requests": 1,
            "elapsed_seconds": round(elapsed, 3),
        },
    }


def process_single_image(sample: dict) -> dict | None:
    """Process one image: find it, call SAM3, save JSON. Returns summary dict."""
    sample_id = make_sample_id(sample)
    output_path = make_output_json_path(sample_id)

    # Skip if already processed
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
            if existing.get("status") != "error":
                return {
                    "sample_id": sample_id,
                    "image_relpath": sample["image_relpath"],
                    "status": "skipped",
                    "reason": "already_processed",
                }
        except Exception:
            pass  # Re-process if file is corrupt

    # Find image
    image_path = find_image_path(sample)
    if image_path is None:
        return {
            "sample_id": sample_id,
            "image_relpath": sample["image_relpath"],
            "status": "error",
            "reason": "image_not_found",
        }

    # Call SAM3
    t0 = time.time()
    sam3_result = call_sam3_batch(image_path)
    elapsed = time.time() - t0

    if sam3_result is None:
        return {
            "sample_id": sample_id,
            "image_relpath": sample["image_relpath"],
            "status": "error",
            "reason": "sam3_call_failed",
        }

    if sam3_result.get("status") != "success":
        return {
            "sample_id": sample_id,
            "image_relpath": sample["image_relpath"],
            "status": "error",
            "reason": f"sam3_returned_{sam3_result.get('status')}",
        }

    # Get image dimensions from SAM3 result
    img_h = sam3_result.get("orig_img_h", 0)
    img_w = sam3_result.get("orig_img_w", 0)

    # Convert to attributes
    attributes = convert_sam3_result_to_attributes(sam3_result, img_w, img_h)

    # Build annotation JSON
    annotation = build_annotation_json(sample, image_path, attributes, elapsed)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

    # Log completion
    classes_detected = [k for k, v in attributes.items() if v.get("has_mask")]
    return {
        "sample_id": sample_id,
        "image_relpath": sample["image_relpath"],
        "status": "success",
        "elapsed": round(elapsed, 3),
        "classes_detected": classes_detected,
        "num_detected": len(classes_detected),
    }


def main():
    print("=" * 70)
    print("  SAM3 UPAR Rare Color Annotation")
    print(f"  Server: {SAM3_SERVER}")
    print(f"  Samples file: {SAMPLES_FILE}")
    print(f"  Output dir: {OUTPUT_ANNOTATIONS_DIR}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 70)
    print()

    # Health check
    try:
        health = requests.get(f"{SAM3_SERVER}/health", timeout=10).json()
        print(f"[health] Service: {health}")
        if not health.get("batch_support"):
            print("[ERROR] Server doesn't support batch! Use server_batch.py.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot reach SAM3 service: {e}")
        sys.exit(1)

    # Parse samples
    print(f"\n[parse] Reading {SAMPLES_FILE} ...")
    samples = parse_samples(SAMPLES_FILE)
    print(f"[parse] {len(samples)} valid samples (train + with image paths)")

    if not samples:
        print("[ERROR] No samples found!")
        sys.exit(1)

    # Process
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

    results = []
    success_count = 0
    error_count = 0
    skip_count = 0
    total_detected = 0
    start_time = time.time()

    print(f"\n[process] Starting annotation of {len(samples)} images...")
    print(f"[process] Resume enabled — already-processed images will be skipped.")
    print(f"[process] Press Ctrl+C to interrupt safely.\n")

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_image, sample): sample
                for sample in samples
            }

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is None:
                    continue

                results.append(result)

                status = result.get("status")
                if status == "success":
                    success_count += 1
                    nd = result.get("num_detected", 0)
                    total_detected += nd
                    classes_str = ",".join(result.get("classes_detected", [])) or "none"
                elif status == "skipped":
                    skip_count += 1
                else:
                    error_count += 1

                # Progress
                pct = i / len(samples) * 100
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(samples) - i) / rate if rate > 0 else 0

                if status == "success":
                    print(
                        f"\r[{i}/{len(samples)} {pct:.1f}%] "
                        f"OK:{success_count} SKIP:{skip_count} ERR:{error_count} "
                        f"| {rate:.1f} img/s | ETA: {eta:.0f}s "
                        f"| [{classes_str}] "
                        f"| {result['image_relpath']}",
                        end="",
                    )
                elif i % 500 == 0 or status == "error":
                    print(
                        f"\r[{i}/{len(samples)} {pct:.1f}%] "
                        f"OK:{success_count} SKIP:{skip_count} ERR:{error_count} "
                        f"| {rate:.1f} img/s "
                        f"| {'ERROR' if status == 'error' else 'SKIP'}: {result.get('reason')}",
                    )

                # Periodic summary save (every 500 images)
                if i % 500 == 0:
                    _save_summary(results, start_time)

    except KeyboardInterrupt:
        print(f"\n\n[interrupt] Caught Ctrl+C. Saving progress...")

    finally:
        elapsed_total = time.time() - start_time
        print(f"\n\n[complete] Finished processing.")
        _save_summary(results, start_time)
        print(f"  Total: {len(samples)}")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total detections: {total_detected}")
        print(f"  Time: {elapsed_total:.1f}s ({elapsed_total/3600:.1f}h)")
        print(f"  Rate: {len(samples)/elapsed_total:.2f} img/s")


def _save_summary(results: list, start_time: float):
    """Save summary JSON."""
    elapsed = time.time() - start_time
    successes = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") == "skipped"]

    summary = {
        "updated_at": datetime.now().isoformat(),
        "total": len(results),
        "success": len(successes),
        "errors": len(errors),
        "skipped": len(skipped),
        "elapsed_seconds": round(elapsed, 2),
        "rate_imgs_per_sec": round(len(results) / elapsed, 3) if elapsed > 0 else 0,
        "server": SAM3_SERVER,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": CLASSES,
        "samples_file": SAMPLES_FILE,
        "output_dir": OUTPUT_ANNOTATIONS_DIR,
        "error_samples": [
            {"sample_id": r["sample_id"], "reason": r.get("reason")}
            for r in errors
        ][:50],  # Keep first 50 for brevity
    }
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_FILE), exist_ok=True)
    with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
