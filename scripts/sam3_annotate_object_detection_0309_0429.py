#!/usr/bin/env python3
"""
Annotate object_detection_0309-0429 person crops with SAM3 attributes.

Input:
    data/object_detection_0309-0429/images/<scene>/<image>

Output:
    data/object_detection_0309-0429/annotations/<scene>/<image_stem>.json

The image iterator and executor submission are streaming/bounded so the script
does not materialize the full dataset in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Please install pillow: pip install pillow")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "object_detection_0309-0429"
IMAGE_DIR = DATASET_DIR / "images"
OUTPUT_ANNOTATIONS_DIR = DATASET_DIR / "annotations"
OUTPUT_LOG_FILE = DATASET_DIR / "sam3_object_detection_0309_0429_completion.jsonl"
OUTPUT_SUMMARY_FILE = DATASET_DIR / "sam3_object_detection_0309_0429_summary.json"

SAM3_CONTAINER_ID = "c253d66a7c42"
DEFAULT_SERVER = "http://localhost:8010"
DEFAULT_GPU = "7"
SCHEMA_VERSION = "1.0"
DATASET_NAME = "object_detection_0309-0429"
SAMPLE_PREFIX = "object_detection_0309_0429"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["upper", "lower", "dress", "hat", "glasses", "mask", "hair"]


def build_attribute_prompts(confidence_threshold: float) -> list[dict]:
    return [
        {"tag": "upper", "prompt": "upper garment", "confidence_threshold": confidence_threshold},
        {"tag": "lower", "prompt": "lower garment", "confidence_threshold": confidence_threshold},
        {"tag": "dress", "prompt": "dress", "confidence_threshold": confidence_threshold},
        {"tag": "hat", "prompt": "hat", "confidence_threshold": confidence_threshold},
        {"tag": "glasses", "prompt": "glasses", "confidence_threshold": confidence_threshold},
        {"tag": "mask", "prompt": "face mask", "confidence_threshold": confidence_threshold},
        {"tag": "hair", "prompt": "hair", "confidence_threshold": confidence_threshold},
    ]


def iter_images(image_dir: Path = IMAGE_DIR) -> Iterable[Path]:
    for root, dirs, files in os.walk(image_dir):
        dirs.sort()
        for filename in sorted(files):
            path = Path(root) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def limited_iter_images(image_dir: Path, limit: int | None) -> Iterable[Path]:
    for index, image_path in enumerate(iter_images(image_dir), 1):
        if limit is not None and index > limit:
            break
        yield image_path


def output_path(
    image_path: Path,
    image_dir: Path = IMAGE_DIR,
    output_dir: Path = OUTPUT_ANNOTATIONS_DIR,
) -> Path:
    return output_dir / image_path.relative_to(image_dir).with_suffix(".json")


def empty_attr(label: str = "unknown") -> dict:
    return {
        "label": label,
        "has_mask": False,
        "bbox": None,
        "mask": None,
        "source_class": None,
        "score": 0.0,
        "note": None,
    }


def call_sam3_batch(
    image_path: Path,
    server: str,
    confidence_threshold: float,
    timeout: int,
) -> dict:
    with image_path.open("rb") as image_file:
        response = requests.post(
            f"{server}/annotate_batch",
            files={"image": (image_path.name, image_file)},
            data={"prompts": json.dumps(build_attribute_prompts(confidence_threshold))},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def convert_sam3_result_to_attributes(
    sam3_result: dict,
    img_width: int,
    img_height: int,
    confidence_threshold: float,
) -> dict:
    attributes = {}
    results = sam3_result.get("results", {})

    for attr_name in CLASSES:
        attr_result = results.get(attr_name)
        if attr_result is None or attr_result.get("status") != "success":
            attributes[attr_name] = empty_attr()
            continue

        if int(attr_result.get("num_masks", 0) or 0) <= 0:
            label = "no" if attr_name in {"hat", "glasses", "mask"} else "unknown"
            attributes[attr_name] = empty_attr(label)
            continue

        best_score = float(attr_result["pred_scores"][0])
        has_mask = best_score >= confidence_threshold
        best_box = [
            max(0.0, min(1.0, float(value)))
            for value in attr_result["pred_boxes"][0]
        ]

        if attr_name in {"hat", "glasses", "mask"}:
            label = attr_name if has_mask else "no"
        else:
            label = attr_name if has_mask else "unknown"

        attributes[attr_name] = {
            "label": label,
            "has_mask": has_mask,
            "bbox": best_box if has_mask else None,
            "mask": {
                "format": "coco_rle",
                "rle": {
                    "size": [img_height, img_width],
                    "counts": attr_result["pred_masks"][0],
                },
            } if has_mask else None,
            "source_class": None,
            "score": round(best_score, 8),
            "note": "generated by SAM3 via sam3-annotation-local batch API",
        }

    return attributes


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def build_annotation(
    image_path: Path,
    image_dir: Path,
    attributes: dict,
    server: str,
    elapsed: float,
    img_width: int,
    img_height: int,
) -> dict:
    image_rel = image_path.relative_to(image_dir).as_posix()
    annotation_rel = Path(image_rel).with_suffix(".json").as_posix()
    sample_rel = Path(image_rel).with_suffix("").as_posix()
    parts = Path(image_rel).parts
    scene = parts[0] if len(parts) > 1 else None

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{SAMPLE_PREFIX}/{sample_rel}",
        "source": {
            "name": DATASET_NAME,
            "image_path": str(image_path),
            "source_image": image_path.name,
            "scene": scene,
            "original_relpath": image_rel,
        },
        "image": {
            "path": f"images/{image_rel}",
            "width": img_width,
            "height": img_height,
            "is_person_crop": True,
        },
        "annotation_path": f"annotations/{annotation_rel}",
        "classes": CLASSES,
        "attributes": attributes,
        "sam3_unified_completion": {
            "updated_at": datetime.now().isoformat(),
            "classes_requested": CLASSES,
            "classes_updated": sum(1 for attr in attributes.values() if attr.get("has_mask")),
            "api": "annotate_batch",
            "server": server,
            "requests": 1,
            "elapsed_seconds": round(elapsed, 3),
        },
    }


def process_single_image(image_path: Path, args: argparse.Namespace) -> dict:
    result_base = {
        "image_relpath": image_path.relative_to(args.image_dir).as_posix(),
        "output_path": str(output_path(image_path, args.image_dir, args.output_dir)),
    }
    json_path = output_path(image_path, args.image_dir, args.output_dir)

    if json_path.exists() and not args.overwrite:
        return {**result_base, "status": "skipped", "reason": "already_processed"}

    try:
        start = time.time()
        sam3_result = call_sam3_batch(
            image_path=image_path,
            server=args.server,
            confidence_threshold=args.confidence_threshold,
            timeout=args.timeout,
        )
        elapsed = time.time() - start

        if sam3_result.get("status") != "success":
            return {
                **result_base,
                "status": "error",
                "reason": f"sam3_returned_{sam3_result.get('status')}",
            }

        img_width = int(sam3_result.get("orig_img_w") or 0)
        img_height = int(sam3_result.get("orig_img_h") or 0)
        if img_width <= 0 or img_height <= 0:
            img_width, img_height = read_image_size(image_path)

        attributes = convert_sam3_result_to_attributes(
            sam3_result=sam3_result,
            img_width=img_width,
            img_height=img_height,
            confidence_threshold=args.confidence_threshold,
        )
        annotation = build_annotation(
            image_path=image_path,
            image_dir=args.image_dir,
            attributes=attributes,
            server=args.server,
            elapsed=elapsed,
            img_width=img_width,
            img_height=img_height,
        )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as output_file:
            json.dump(annotation, output_file, ensure_ascii=False, indent=2)

        classes_detected = [name for name, attr in attributes.items() if attr.get("has_mask")]
        return {
            **result_base,
            "status": "success",
            "elapsed": round(elapsed, 3),
            "classes_detected": classes_detected,
            "num_detected": len(classes_detected),
        }
    except Exception as exc:
        return {
            **result_base,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def check_sam3_health(server: str) -> dict:
    response = requests.get(f"{server}/health", timeout=10)
    response.raise_for_status()
    health = response.json()
    if not health.get("batch_support"):
        raise RuntimeError(f"SAM3 service does not report batch_support: {health}")
    return health


def maybe_set_container_gpu(container_id: str, gpu: str, enabled: bool) -> None:
    if not enabled:
        return
    command = ["docker", "update", "--gpus", f"device={gpu}", container_id]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("[warn] docker command not found; skipped GPU update")
    except subprocess.CalledProcessError as exc:
        print(f"[warn] docker GPU update failed: {exc}")


def save_summary(
    summary_path: Path,
    stats: dict,
    start_time: float,
    args: argparse.Namespace,
    health: dict | None,
) -> None:
    elapsed = time.time() - start_time
    summary = {
        "updated_at": datetime.now().isoformat(),
        "total_seen": stats["seen"],
        "success": stats["success"],
        "skipped": stats["skipped"],
        "errors": stats["error"],
        "elapsed_seconds": round(elapsed, 2),
        "rate_imgs_per_sec": round(stats["seen"] / elapsed, 3) if elapsed > 0 else 0,
        "server": args.server,
        "container_id": args.container_id,
        "gpu": args.gpu,
        "confidence_threshold": args.confidence_threshold,
        "classes": CLASSES,
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "limit": args.limit,
        "health": health,
        "error_samples": stats["error_samples"][:50],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)


def handle_result(result: dict, stats: dict, log_file, total_hint: int | None) -> None:
    status = result.get("status", "error")
    stats["seen"] += 1
    stats[status] = stats.get(status, 0) + 1
    if status == "success":
        stats["detected"] += int(result.get("num_detected", 0) or 0)
    elif status == "error":
        stats["error_samples"].append(
            {
                "image_relpath": result.get("image_relpath"),
                "reason": result.get("reason"),
            }
        )

    log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    log_file.flush()

    should_print = status == "error" or stats["seen"] == 1 or stats["seen"] % 50 == 0
    if should_print:
        total_text = total_hint if total_hint is not None else "?"
        print(
            f"[{stats['seen']}/{total_text}] "
            f"OK:{stats['success']} SKIP:{stats['skipped']} ERR:{stats['error']} "
            f"| {status} | {result.get('image_relpath')}",
            flush=True,
        )


def process_streaming(args: argparse.Namespace, health: dict | None) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "seen": 0,
        "success": 0,
        "skipped": 0,
        "error": 0,
        "detected": 0,
        "error_samples": [],
    }
    pending_limit = max(args.workers * args.pending_multiplier, args.workers)
    total_hint = args.limit
    start_time = time.time()

    with args.log_file.open("a", encoding="utf-8") as log_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = set()
            image_iter = iter(limited_iter_images(args.image_dir, args.limit))

            while True:
                while len(pending) < pending_limit:
                    try:
                        image_path = next(image_iter)
                    except StopIteration:
                        break
                    pending.add(executor.submit(process_single_image, image_path, args))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    handle_result(future.result(), stats, log_file, total_hint)

                if stats["seen"] % args.summary_interval == 0:
                    save_summary(args.summary_file, stats, start_time, args, health)

    save_summary(args.summary_file, stats, start_time, args, health)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 annotate object_detection_0309-0429 person crops."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ANNOTATIONS_DIR)
    parser.add_argument("--log-file", type=Path, default=OUTPUT_LOG_FILE)
    parser.add_argument("--summary-file", type=Path, default=OUTPUT_SUMMARY_FILE)
    parser.add_argument("--container-id", default=SAM3_CONTAINER_ID)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--set-container-gpu", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pending-multiplier", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--summary-interval", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_set_container_gpu(args.container_id, args.gpu, args.set_container_gpu)

    print("=" * 72)
    print("SAM3 object_detection_0309-0429 annotation")
    print(f"server: {args.server}")
    print(f"image_dir: {args.image_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"workers: {args.workers}")
    print(f"pending_limit: {max(args.workers * args.pending_multiplier, args.workers)}")
    print(f"gpu: {args.gpu} container: {args.container_id}")
    print("=" * 72, flush=True)

    health = check_sam3_health(args.server)
    print(f"[health] {health}", flush=True)

    stats = process_streaming(args, health)
    print(
        "[complete] "
        f"seen={stats['seen']} success={stats['success']} "
        f"skipped={stats['skipped']} errors={stats['error']} "
        f"detections={stats['detected']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

