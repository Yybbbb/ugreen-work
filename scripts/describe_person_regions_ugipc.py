#!/usr/bin/env python3
"""Batch Florence person region descriptions from YOLO crop bboxes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
FLORENCE_CAPTION_DIR = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = FLORENCE_CAPTION_DIR.parent
FLORENCE_DATA_DIR = WORKSPACE_ROOT / "florence-data"
DEFAULT_INPUT_ROOT = FLORENCE_DATA_DIR / "qwen-person-json-yolo26m"
DEFAULT_INPUT_GLOB = "images/**/*.json"
DEFAULT_OUTPUT_ROOT = FLORENCE_DATA_DIR / "ugipc-person-region-descriptions-yolo26m"
DEFAULT_INFER_SCRIPT = (
    FLORENCE_CAPTION_DIR
    / "ugipc_1231_15words_epoch3_full_handoff"
    / "infer_ugipc.py"
)
DEFAULT_CHECKPOINT = DEFAULT_INFER_SCRIPT.parent / "checkpoint"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def write_json_atomic(path: Path, data: dict[str, Any], indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp_path.replace(path)


def load_infer_module(infer_script: Path):
    infer_script = infer_script.resolve()
    if not infer_script.exists():
        raise FileNotFoundError(f"infer script not found: {infer_script}")

    infer_dir = str(infer_script.parent)
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)

    spec = importlib.util.spec_from_file_location("ugipc_infer", infer_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import infer script: {infer_script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        requirement_file = infer_script.parent / "requirements.txt"
        raise ModuleNotFoundError(
            f"cannot import {infer_script}; missing Python dependency '{exc.name}'. "
            f"Run with the same environment used for infer_ugipc.py, or install "
            f"dependencies from {requirement_file}."
        ) from exc
    return module


def list_input_jsons(input_root: Path, input_glob: str) -> list[Path]:
    paths = sorted(path for path in input_root.glob(input_glob) if path.is_file())
    return paths


def apply_limit(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is not None:
        return paths[:limit]
    return paths


def apply_shard(paths: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if num_shards == 1:
        return paths
    return [path for index, path in enumerate(paths) if index % num_shards == shard_index]


def output_path_for(input_json: Path, input_root: Path, output_root: Path) -> Path:
    rel_path = input_json.resolve().relative_to(input_root.resolve())
    return output_root / rel_path


def existing_output_ok(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {}
    try:
        data = read_json(path)
    except Exception as exc:  # noqa: BLE001 - corrupt output should be regenerated.
        return False, {"existing_error": str(exc)}
    return data.get("status") == "ok", data


def resolve_image_path(image_info: dict[str, Any]) -> Path:
    path_value = image_info.get("path")
    if path_value:
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        return (FLORENCE_DATA_DIR / path).resolve()

    rel_value = image_info.get("relative_path")
    if rel_value:
        return (FLORENCE_DATA_DIR / str(rel_value)).resolve()

    raise ValueError("image.path or image.relative_path is required")


def to_bbox(value: Any) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("crop_bbox_xyxy must contain 4 values")
    bbox = [float(v) for v in value]
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError(f"bbox values must be finite: {value}")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"bbox must have positive area: {value}")
    return bbox


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[list[float], bool]:
    clamped = [
        max(0.0, min(float(width), bbox[0])),
        max(0.0, min(float(height), bbox[1])),
        max(0.0, min(float(width), bbox[2])),
        max(0.0, min(float(height), bbox[3])),
    ]
    changed = any(abs(a - b) > 1e-6 for a, b in zip(bbox, clamped, strict=True))
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"bbox is outside image bounds after clamping: {bbox}")
    return clamped, changed


def normalize_bbox_value(bbox: list[float]) -> list[int | float]:
    normalized: list[int | float] = []
    for value in bbox:
        if float(value).is_integer():
            normalized.append(int(value))
        else:
            normalized.append(round(float(value), 6))
    return normalized


def source_detection_summary(detection: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "class_id",
        "class_name",
        "confidence",
        "bbox_xyxy",
        "expanded_bbox_xyxy",
        "expanded_bbox_xyxy_norm",
        "crop_width",
        "crop_height",
        "crop_area",
        "crop_path",
    )
    return {key: detection[key] for key in keep_keys if key in detection}


def bbox_area(value: Any) -> float:
    try:
        bbox = to_bbox(value)
    except Exception:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def detection_area(detection: dict[str, Any]) -> float:
    crop_area = detection.get("crop_area")
    try:
        area = float(crop_area)
        if math.isfinite(area) and area > 0:
            return area
    except (TypeError, ValueError):
        pass

    try:
        width = float(detection.get("crop_width"))
        height = float(detection.get("crop_height"))
        area = width * height
        if math.isfinite(area) and area > 0:
            return area
    except (TypeError, ValueError):
        pass

    return bbox_area(detection.get("crop_bbox_xyxy"))


def select_largest_detections(
    detections: list[dict[str, Any]],
    max_regions_per_image: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_regions_per_image <= 0 or len(detections) <= max_regions_per_image:
        return detections, []

    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-detection_area(item[1]), item[0]),
    )
    selected = [detection for _, detection in ranked[:max_regions_per_image]]
    dropped = [detection for _, detection in ranked[max_regions_per_image:]]
    return selected, dropped


def extract_detections(source: dict[str, Any]) -> list[dict[str, Any]]:
    detections = source.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    return [d for d in detections if isinstance(d, dict) and "crop_bbox_xyxy" in d]


def extract_selected_detections(
    source: dict[str, Any],
    max_regions_per_image: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    all_detections = extract_detections(source)
    selected, dropped = select_largest_detections(all_detections, max_regions_per_image)
    return selected, dropped, len(all_detections)


def generate_batch(
    infer_module,
    processor,
    model,
    image: Image.Image,
    prompts: list[str],
    max_new_tokens: int,
    num_beams: int,
) -> list[tuple[str, str]]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = processor(
        text=prompts,
        images=[image] * len(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    with infer_module.torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=0,
        )
    raws = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [(raw, infer_module.clean_text(raw)) for raw in raws]


def describe_source_json(
    *,
    source_json_path: Path,
    output_json_path: Path,
    source: dict[str, Any],
    infer_module,
    processor,
    model,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_info = source.get("image")
    if not isinstance(image_info, dict):
        raise ValueError("image must be an object")

    image_path = resolve_image_path(image_info)
    detections, dropped_detections, source_detection_count = extract_selected_detections(
        source,
        args.max_regions_per_image,
    )

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    width, height = image.size
    regions: list[dict[str, Any]] = []
    pending: list[tuple[int, str]] = []

    for detection in detections:
        crop_index = detection.get("crop_index")
        region: dict[str, Any] = {
            "crop_index": crop_index,
            "source_detection": source_detection_summary(detection),
        }

        try:
            source_bbox = to_bbox(detection.get("crop_bbox_xyxy"))
            bbox, clipped = clamp_bbox(source_bbox, width, height)
            loc_box = infer_module.pixel_bbox_to_loc(bbox, width, height)
            prompt = infer_module.build_region_description_prompt(loc_box, name=args.name)

            region.update(
                {
                    "status": "pending",
                    "crop_bbox_xyxy": normalize_bbox_value(source_bbox),
                    "bbox_used_xyxy": normalize_bbox_value(bbox),
                    "bbox_was_clipped": clipped,
                    "bbox_loc_0_999": loc_box,
                    "prompt": prompt,
                }
            )
            pending.append((len(regions), prompt))
        except Exception as exc:  # noqa: BLE001 - continue other regions in this image.
            region.update(
                {
                    "status": "failed",
                    "crop_bbox_xyxy": detection.get("crop_bbox_xyxy"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        regions.append(region)

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [prompt for _, prompt in batch]
        try:
            outputs = generate_batch(
                infer_module,
                processor,
                model,
                image,
                prompts,
                args.max_new_tokens,
                args.num_beams,
            )
            for (region_index, _), (raw, text) in zip(batch, outputs, strict=True):
                regions[region_index].update({"status": "ok", "raw": raw, "text": text})
        except Exception as batch_exc:  # noqa: BLE001 - retry individually to isolate bad regions.
            for region_index, prompt in batch:
                try:
                    raw, text = infer_module.generate(
                        processor,
                        model,
                        image,
                        prompt,
                        args.max_new_tokens,
                        args.num_beams,
                    )
                    regions[region_index].update({"status": "ok", "raw": raw, "text": text})
                except Exception as exc:  # noqa: BLE001 - continue other regions in this image.
                    regions[region_index].update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "batch_error": f"{type(batch_exc).__name__}: {batch_exc}",
                        }
                    )

    ok_regions = sum(1 for region in regions if region.get("status") == "ok")
    failed_regions = sum(1 for region in regions if region.get("status") == "failed")
    if failed_regions == 0:
        status = "ok"
    elif ok_regions > 0:
        status = "partial"
    else:
        status = "failed"

    source_width = image_info.get("width")
    source_height = image_info.get("height")
    size_mismatch = (
        source_width is not None
        and source_height is not None
        and (int(source_width) != width or int(source_height) != height)
    )

    return {
        "schema_version": "1.0",
        "status": status,
        "generated_at": utc_now(),
        "source_json_path": str(source_json_path),
        "output_json_path": str(output_json_path),
        "image": {
            "path": str(image_path),
            "relative_path": image_info.get("relative_path"),
            "width": width,
            "height": height,
            "source_width": source_width,
            "source_height": source_height,
            "size_mismatch": size_mismatch,
        },
        "model": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "task": "region_description",
            "name": args.name,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "max_regions_per_image": args.max_regions_per_image,
        },
        "source": {
            "schema_version": source.get("schema_version"),
            "status": source.get("status"),
            "json_path": source.get("json_path"),
            "model": source.get("model"),
            "crop_rules": source.get("crop_rules"),
            "detection_stats": source.get("detection_stats"),
        },
        "person_crop_count": len(detections),
        "source_person_crop_count": source_detection_count,
        "regions_dropped_by_area_limit": len(dropped_detections),
        "dropped_regions": [
            {
                "crop_index": detection.get("crop_index"),
                "crop_area": detection_area(detection),
                "crop_bbox_xyxy": detection.get("crop_bbox_xyxy"),
            }
            for detection in dropped_detections
        ],
        "regions": regions,
        "stats": {
            "source_regions_total": source_detection_count,
            "regions_total": len(regions),
            "regions_ok": ok_regions,
            "regions_failed": failed_regions,
            "regions_dropped_by_area_limit": len(dropped_detections),
        },
    }


def manifest_record_from_output(
    source_json_path: Path,
    output_json_path: Path,
    status: str,
    output: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    stats = output.get("stats", {}) if output else {}
    image = output.get("image", {}) if output else {}
    record: dict[str, Any] = {
        "source_json_path": str(source_json_path),
        "output_json_path": str(output_json_path),
        "status": status,
        "image_path": image.get("path"),
        "source_regions_total": stats.get("source_regions_total", stats.get("regions_total", 0)),
        "regions_total": stats.get("regions_total", 0),
        "regions_ok": stats.get("regions_ok", 0),
        "regions_failed": stats.get("regions_failed", 0),
        "regions_dropped_by_area_limit": stats.get("regions_dropped_by_area_limit", 0),
    }
    if error:
        record["error"] = error
    return record


def output_exceeds_region_limit(output: dict[str, Any], max_regions_per_image: int) -> bool:
    if max_regions_per_image <= 0:
        return False
    regions = output.get("regions")
    return isinstance(regions, list) and len(regions) > max_regions_per_image


def dry_run_count_regions(path: Path, max_regions_per_image: int) -> tuple[int, int, int, str | None]:
    try:
        source = read_json(path)
        selected, dropped, source_count = extract_selected_detections(source, max_regions_per_image)
        return len(selected), source_count, len(dropped), None
    except Exception as exc:  # noqa: BLE001 - dry run should report invalid inputs.
        return 0, 0, 0, f"{type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run UGIPC Florence person region_description for every "
            "detections[].crop_bbox_xyxy in YOLO person JSON files."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--infer-script", default=str(DEFAULT_INFER_SCRIPT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--name",
        default="",
        help=(
            "Optional plain-text region name inserted between "
            "<REGION_TO_DESCRIPTION> and loc tokens. Default is empty, "
            "which uses <REGION_TO_DESCRIPTION><loc...> directly."
        ),
    )
    parser.add_argument("--device", default=None, help="Default: cuda:0 when available, else cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of regions from one image to generate together.",
    )
    parser.add_argument(
        "--max-regions-per-image",
        type=int,
        default=20,
        help="Process only the largest N crop bboxes by area per image. Use 0 to disable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sorted JSON files")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of input shards for parallel runs")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process, 0-based")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing ok outputs")
    parser.add_argument("--dry-run", action="store_true", help="Scan inputs and count regions without inference")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first image-level failure")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--indent", type=int, default=2, help="Output JSON indent")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    all_input_paths = list_input_jsons(input_root, args.input_glob)
    limited_input_paths = apply_limit(all_input_paths, args.limit)
    input_paths = apply_shard(limited_input_paths, args.num_shards, args.shard_index)
    started = time.monotonic()

    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "input_root": str(input_root),
        "input_glob": args.input_glob,
        "output_root": str(output_root),
        "infer_script": str(Path(args.infer_script).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "name": args.name,
        "device": args.device or "auto",
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "max_regions_per_image": args.max_regions_per_image,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "dry_run": args.dry_run,
        "found_json_files": len(all_input_paths),
        "limited_json_files": len(limited_input_paths),
        "selected_json_files": len(input_paths),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "partial": 0,
        "images_with_no_regions": 0,
        "source_regions_total": 0,
        "regions_total": 0,
        "regions_ok": 0,
        "regions_failed": 0,
        "regions_dropped_by_area_limit": 0,
        "elapsed_seconds": None,
    }

    if args.dry_run:
        records: list[dict[str, Any]] = []
        for source_json_path in input_paths:
            output_json_path = output_path_for(source_json_path, input_root, output_root)
            region_count, source_region_count, dropped_count, error = dry_run_count_regions(
                source_json_path,
                args.max_regions_per_image,
            )
            summary["source_regions_total"] += source_region_count
            summary["regions_total"] += region_count
            summary["regions_dropped_by_area_limit"] += dropped_count
            if region_count == 0:
                summary["images_with_no_regions"] += 1
            status = "failed" if error else "dry_run"
            if error:
                summary["failed"] += 1
            records.append(
                {
                    "source_json_path": str(source_json_path),
                    "output_json_path": str(output_json_path),
                    "status": status,
                    "source_regions_total": source_region_count,
                    "regions_total": region_count,
                    "regions_dropped_by_area_limit": dropped_count,
                    **({"error": error} if error else {}),
                }
            )
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    try:
        infer_module = load_infer_module(Path(args.infer_script))
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    if args.device is None:
        args.device = "cuda:0" if infer_module.torch.cuda.is_available() else "cpu"

    print(
        f"Loading model: checkpoint={args.checkpoint} device={args.device} dtype={args.dtype}",
        file=sys.stderr,
        flush=True,
    )
    processor, model = infer_module.load_model(args.checkpoint, args.device, args.dtype)

    records = []
    total_inputs = len(input_paths)
    for index, source_json_path in enumerate(input_paths, start=1):
        output_json_path = output_path_for(source_json_path, input_root, output_root)
        existing_ok, existing_output = existing_output_ok(output_json_path)
        if (
            existing_ok
            and not args.overwrite
            and not output_exceeds_region_limit(existing_output, args.max_regions_per_image)
        ):
            summary["skipped"] += 1
            stats = existing_output.get("stats", {})
            summary["source_regions_total"] += int(stats.get("source_regions_total", stats.get("regions_total", 0)))
            summary["regions_total"] += int(stats.get("regions_total", 0))
            summary["regions_ok"] += int(stats.get("regions_ok", 0))
            summary["regions_failed"] += int(stats.get("regions_failed", 0))
            summary["regions_dropped_by_area_limit"] += int(stats.get("regions_dropped_by_area_limit", 0))
            if int(stats.get("regions_total", 0)) == 0:
                summary["images_with_no_regions"] += 1
            records.append(
                manifest_record_from_output(
                    source_json_path,
                    output_json_path,
                    "skipped",
                    existing_output,
                )
            )
            continue

        try:
            source = read_json(source_json_path)
            output = describe_source_json(
                source_json_path=source_json_path,
                output_json_path=output_json_path,
                source=source,
                infer_module=infer_module,
                processor=processor,
                model=model,
                args=args,
            )
            write_json_atomic(output_json_path, output, indent=args.indent)

            status = str(output.get("status"))
            if status == "ok":
                summary["processed"] += 1
            elif status == "partial":
                summary["partial"] += 1
            else:
                summary["failed"] += 1

            stats = output.get("stats", {})
            summary["source_regions_total"] += int(stats.get("source_regions_total", stats.get("regions_total", 0)))
            summary["regions_total"] += int(stats.get("regions_total", 0))
            summary["regions_ok"] += int(stats.get("regions_ok", 0))
            summary["regions_failed"] += int(stats.get("regions_failed", 0))
            summary["regions_dropped_by_area_limit"] += int(stats.get("regions_dropped_by_area_limit", 0))
            if int(stats.get("regions_total", 0)) == 0:
                summary["images_with_no_regions"] += 1
            records.append(
                manifest_record_from_output(
                    source_json_path,
                    output_json_path,
                    status,
                    output,
                )
            )
        except Exception as exc:  # noqa: BLE001 - write an image-level failure record.
            error = f"{type(exc).__name__}: {exc}"
            failed_output = {
                "schema_version": "1.0",
                "status": "failed",
                "generated_at": utc_now(),
                "source_json_path": str(source_json_path),
                "output_json_path": str(output_json_path),
                "error": error,
            }
            write_json_atomic(output_json_path, failed_output, indent=args.indent)
            summary["failed"] += 1
            records.append(
                manifest_record_from_output(
                    source_json_path,
                    output_json_path,
                    "failed",
                    failed_output,
                    error=error,
                )
            )
            if args.fail_fast:
                raise

        if args.progress_every > 0 and index % args.progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"[{index}/{total_inputs}] processed={summary['processed']} "
                f"skipped={summary['skipped']} partial={summary['partial']} "
                f"failed={summary['failed']} regions_ok={summary['regions_ok']} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if args.num_shards > 1:
        shard_suffix = f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}"
        write_jsonl(output_root / f"manifest.{shard_suffix}.jsonl", records)
        write_json_atomic(output_root / f"summary.{shard_suffix}.json", summary, indent=2)
    else:
        write_jsonl(output_root / "manifest.jsonl", records)
        write_json_atomic(output_root / "summary.json", summary, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
