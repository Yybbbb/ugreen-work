#!/usr/bin/env python3
"""Prepare Florence2 single-person region-caption JSONL files."""

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from person_sft_data import (
    build_region_prompt,
    clip_bbox,
    iter_jsonl,
    load_json,
    quantize_bbox,
    sha256_file,
    validate_relative_path,
    write_json_atomic,
    write_jsonl_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "prepared"
PERSON_TASK = "<REGION_TO_CATEGORY>"
SCHEMA_VERSION = "florence_person_sft_v1"
DEFAULT_SPLITS = ("train", "dev", "test", "rl")
EXPECTED_FULL_COUNTS = {
    "train": 30000,
    "dev": 2000,
    "test": 4328,
    "rl": 5981,
}


def _require_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _load_frame_document(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], list]:
    document = load_json(path)
    image = _require_dict(document.get("image"), f"image in {path}")
    crops = _require_list(document.get("crops"), f"crops in {path}")
    image_path = Path(_require_nonempty_text(image.get("path"), f"image.path in {path}"))
    if not image_path.is_file():
        raise FileNotFoundError(f"Full frame does not exist: {image_path} (source {path})")
    with image_path.open("rb") as image_file:
        if not image_file.read(1):
            raise ValueError(f"Full frame is empty: {image_path} (source {path})")
    _require_integer(image.get("width"), f"image.width in {path}")
    _require_integer(image.get("height"), f"image.height in {path}")
    return document, image, crops


def prepare_manifest_row(
    manifest_row: Dict[str, Any],
    source_relative: Path,
    image: Dict[str, Any],
    crops: list,
    split: str,
) -> Dict[str, Any]:
    sample_id = _require_nonempty_text(manifest_row.get("sample_id"), "sample_id")
    crop_position = _require_integer(manifest_row.get("crop_position"), "crop_position")
    expected_sample_id = f"{source_relative.as_posix()}#{crop_position}"
    if sample_id != expected_sample_id:
        raise ValueError(
            f"sample_id does not match source path and crop_position: "
            f"{sample_id!r} != {expected_sample_id!r}"
        )
    manifest_crop_index = _require_integer(
        manifest_row.get("crop_index"), f"manifest crop_index for {sample_id}"
    )
    matching_crops = [
        _require_dict(candidate, f"crop for {sample_id}")
        for candidate in crops
        if isinstance(candidate, dict)
        and candidate.get("crop_index") == manifest_crop_index
    ]
    if len(matching_crops) != 1:
        raise ValueError(
            f"Expected one crop_index={manifest_crop_index} for {sample_id}, "
            f"found {len(matching_crops)}"
        )
    crop = matching_crops[0]
    annotation = _require_dict(crop.get("annotation"), f"annotation for {sample_id}")
    if annotation.get("status") != "success":
        raise ValueError(f"Crop annotation is not successful for {sample_id}")

    crop_index = _require_integer(crop.get("crop_index"), f"crop_index for {sample_id}")
    if manifest_crop_index != crop_index:
        raise ValueError(
            f"crop_index mismatch for {sample_id}: {manifest_crop_index!r} != {crop_index!r}"
        )
    caption = _require_nonempty_text(crop.get("caption"), f"caption for {sample_id}")
    manifest_caption = _require_nonempty_text(
        manifest_row.get("caption"), f"manifest caption for {sample_id}"
    )
    if caption != manifest_caption:
        raise ValueError(
            f"Caption mismatch for {sample_id}: {manifest_caption!r} != {caption!r}"
        )

    width = _require_integer(image.get("width"), f"image.width for {sample_id}")
    height = _require_integer(image.get("height"), f"image.height for {sample_id}")
    bbox = clip_bbox(crop.get("expanded_bbox_xyxy"), width, height)
    locations = quantize_bbox(bbox, width, height)
    attributes = _require_dict(crop.get("attributes"), f"attributes for {sample_id}")
    scene = _require_nonempty_text(manifest_row.get("scene"), f"scene for {sample_id}")
    session = _require_nonempty_text(
        manifest_row.get("session"), f"session for {sample_id}"
    )
    image_path = _require_nonempty_text(image.get("path"), f"image.path for {sample_id}")

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": split,
        "task": PERSON_TASK,
        "image": image_path,
        "prompt": build_region_prompt(PERSON_TASK, locations),
        "label": caption,
        "bbox_xyxy": bbox,
        "bbox_loc_0_999": locations,
        "scene": scene,
        "session": session,
        "source_json": source_relative.as_posix(),
        "crop_position": crop_position,
        "crop_index": crop_index,
        "scale": manifest_row.get("scale"),
        "attributes": attributes,
    }


def prepare_split(
    manifest_path: Path, split_root: Path, split: str
) -> Iterator[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    split_root = Path(split_root)
    cached_relative: Optional[Path] = None
    cached_image: Optional[Dict[str, Any]] = None
    cached_crops: Optional[list] = None
    for manifest_row in iter_jsonl(manifest_path):
        source_relative = validate_relative_path(
            manifest_row.get("source_json_relative_path")
        )
        if source_relative != cached_relative:
            source_path = split_root / source_relative
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Source JSON does not exist for {manifest_row.get('sample_id')}: {source_path}"
                )
            _, cached_image, cached_crops = _load_frame_document(source_path)
            cached_relative = source_relative
        assert cached_image is not None and cached_crops is not None
        yield prepare_manifest_row(
            manifest_row, source_relative, cached_image, cached_crops, split
        )


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def validate_prepared_files(
    data_root: Path, output_dir: Path, splits: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for split in splits:
        manifest_path = data_root / "manifests" / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        if not output_path.is_file():
            raise FileNotFoundError(f"Prepared split does not exist: {output_path}")
        manifest_count = _count_jsonl(manifest_path)
        prepared_count = 0
        seen = set()
        for row in iter_jsonl(output_path):
            sample_id = _require_nonempty_text(row.get("sample_id"), "sample_id")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in {output_path}: {sample_id}")
            seen.add(sample_id)
            if row.get("split") != split or row.get("task") != PERSON_TASK:
                raise ValueError(f"Invalid task or split for {sample_id} in {output_path}")
            _require_nonempty_text(row.get("image"), f"image for {sample_id}")
            _require_nonempty_text(row.get("label"), f"label for {sample_id}")
            build_region_prompt(PERSON_TASK, row.get("bbox_loc_0_999"))
            prepared_count += 1
        if prepared_count != manifest_count:
            raise ValueError(
                f"Count mismatch for {split}: prepared={prepared_count} manifest={manifest_count}"
            )
        results[split] = {
            "samples": prepared_count,
            "sha256": sha256_file(output_path),
            "path": str(output_path.resolve()),
        }
    return results


def build_prepared_data(
    data_root: Path,
    output_dir: Path,
    splits: Sequence[str],
    overwrite: bool = False,
) -> Dict[str, Any]:
    data_root = Path(data_root).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"{split}.jsonl" for split in splits]
    if not overwrite:
        existing = next((path for path in output_paths if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Refusing to overwrite prepared split: {existing}")

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        split_metadata = {}
        for split in splits:
            manifest_path = data_root / "manifests" / f"{split}.jsonl"
            split_root = data_root / split
            staged_path = staging_dir / f"{split}.jsonl"
            active_path = output_dir / f"{split}.jsonl"
            if not manifest_path.is_file() or not split_root.is_dir():
                raise FileNotFoundError(
                    f"Missing manifest or split directory for {split}: "
                    f"{manifest_path}, {split_root}"
                )
            count = write_jsonl_atomic(
                staged_path, prepare_split(manifest_path, split_root, split)
            )
            manifest_count = _count_jsonl(manifest_path)
            if count != manifest_count:
                raise ValueError(
                    f"Count mismatch for {split}: prepared={count} manifest={manifest_count}"
                )
            if data_root == DEFAULT_DATA_ROOT.resolve() and split in EXPECTED_FULL_COUNTS:
                expected = EXPECTED_FULL_COUNTS[split]
                if count != expected:
                    raise ValueError(f"Unexpected full {split} count: {count} != {expected}")
            split_metadata[split] = {
                "samples": count,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "path": str(active_path.resolve()),
                "sha256": sha256_file(staged_path),
            }
            print(f"{split}: {count} samples staged for {active_path}")

        validate_prepared_files(data_root, staging_dir, splits)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "task": PERSON_TASK,
            "splits": split_metadata,
        }
        write_json_atomic(staging_dir / "metadata.json", metadata)
        _publish_staged_prepared(staging_dir, output_dir, splits)
        return metadata
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _publish_staged_prepared(
    staging_dir: Path, output_dir: Path, splits: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{split}.jsonl" for split in splits] + ["metadata.json"]
    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.backup.", dir=output_dir.parent)
    )
    backed_up = []
    published = []
    try:
        for name in names:
            staged_path = staging_dir / name
            if not staged_path.is_file():
                raise FileNotFoundError(f"Missing staged prepared file: {staged_path}")
            active_path = output_dir / name
            backup_path = backup_dir / name
            if active_path.exists():
                os.replace(active_path, backup_path)
                backed_up.append(name)
            os.replace(staged_path, active_path)
            published.append(name)
    except BaseException:
        for name in reversed(published):
            try:
                (output_dir / name).unlink()
            except FileNotFoundError:
                pass
        for name in reversed(backed_up):
            os.replace(backup_dir / name, output_dir / name)
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Florence2 REGION_TO_CATEGORY person caption JSONL files."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS)
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_existing:
        result = validate_prepared_files(args.data_root, args.output_dir, args.splits)
    else:
        result = build_prepared_data(
            args.data_root, args.output_dir, args.splits, overwrite=args.overwrite
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
