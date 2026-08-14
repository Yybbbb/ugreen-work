#!/usr/bin/env python3
"""Build single-region Florence-2 fine-tuning data from cleaned Qwen results."""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TASK_TOKEN = "<REGION_TO_DESCRIPTION>"
DEFAULT_INPUT_DIR = Path(
    "/data/work/MichaelYu/florence-data/"
    "qwen_person_2to6_region_descriptions_qwen_cleaned"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "qwen-person-2to6-cleaned-single-region"
)


def loc_tokens(bbox_loc: list[int]) -> str:
    if (
        not isinstance(bbox_loc, list)
        or len(bbox_loc) != 4
        or any(not isinstance(value, int) or not 0 <= value <= 999 for value in bbox_loc)
    ):
        raise ValueError(f"Invalid bbox_loc_0_999: {bbox_loc!r}")
    return "".join(f"<loc_{value}>" for value in bbox_loc)


def load_frame(json_path: Path, input_dir: Path) -> dict:
    with json_path.open(encoding="utf-8") as input_file:
        document = json.load(input_file)

    image = document.get("image") or {}
    image_path = Path(image.get("path", ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    crops = document.get("crops")
    if not isinstance(crops, dict):
        raise ValueError("crops must be an object")

    relative_json = json_path.relative_to(input_dir)
    frame_id = str(relative_json.with_suffix(""))
    samples = []
    for crop_key, crop in sorted(
        crops.items(), key=lambda item: (item[1].get("crop_index", 0), item[0])
    ):
        qwen = crop.get("qwen") or {}
        description = qwen.get("description")
        if qwen.get("status") != "success" or not isinstance(description, str) or not description.strip():
            raise ValueError(f"{crop_key} has no successful Qwen description")

        bbox_loc = crop.get("bbox_loc_0_999")
        prompt = TASK_TOKEN + loc_tokens(bbox_loc)
        source_prompt = (crop.get("florence") or {}).get("prompt")
        if source_prompt and source_prompt != prompt:
            raise ValueError(f"{crop_key} prompt mismatch: {source_prompt!r} != {prompt!r}")

        samples.append(
            {
                "image": str(image_path),
                "prompt": prompt,
                "label": description.strip(),
                "sample_id": f"{frame_id}__{crop_key}",
                "frame_id": frame_id,
                "crop_key": crop_key,
                "crop_index": crop.get("crop_index"),
                "bbox_loc_0_999": bbox_loc,
                "bbox_xyxy": crop.get("bbox_xyxy"),
                "source_json": str(relative_json),
                "source_crop_path": crop.get("crop_path"),
                "description_model": qwen.get("model"),
            }
        )

    if document.get("valid_crop_count") != len(samples):
        raise ValueError(
            f"valid_crop_count={document.get('valid_crop_count')} but found {len(samples)} crops"
        )
    if not samples:
        raise ValueError("frame has no usable crops")
    return {"frame_id": frame_id, "samples": samples}


def select_test_frame_indexes(frames: list[dict], target_samples: int) -> set[int]:
    if target_samples == 0:
        return set()

    parents: dict[int, tuple[int, int] | None] = {0: None}
    for frame_index, frame in enumerate(frames):
        sample_count = len(frame["samples"])
        for current_sum in sorted(parents, reverse=True):
            next_sum = current_sum + sample_count
            if next_sum <= target_samples and next_sum not in parents:
                parents[next_sum] = (current_sum, frame_index)
        if target_samples in parents:
            break

    selected_sum = target_samples if target_samples in parents else max(parents)
    selected = set()
    while selected_sum:
        previous_sum, frame_index = parents[selected_sum]
        selected.add(frame_index)
        selected_sum = previous_sum
    return selected


def write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for sample in samples:
            output_file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict:
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1)")

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (train_path, test_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, existing))}")

    frames = [load_frame(path, input_dir) for path in sorted(input_dir.rglob("*.json"))]
    random.Random(seed).shuffle(frames)

    total_samples = sum(len(frame["samples"]) for frame in frames)
    target_test_samples = round(total_samples * test_ratio)
    test_frame_indexes = select_test_frame_indexes(frames, target_test_samples)

    train_samples = []
    test_samples = []
    train_frames = 0
    test_frames = 0
    crop_count_histogram = Counter()
    for frame_index, frame in enumerate(frames):
        crop_count_histogram[len(frame["samples"])] += 1
        if frame_index in test_frame_indexes:
            test_samples.extend(frame["samples"])
            test_frames += 1
        else:
            train_samples.extend(frame["samples"])
            train_frames += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, train_samples)
    write_jsonl(test_path, test_samples)

    metadata = {
        "schema_version": "qwen_single_region_description_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "task": "REGION_TO_DESCRIPTION",
        "task_token": TASK_TOKEN,
        "description_source": "qwen.description where qwen.status == success",
        "sample_unit": "one full image plus one person region",
        "split_unit": "source frame",
        "split_policy": "seeded frame shuffle plus exact sample-count subset selection",
        "seed": seed,
        "requested_test_ratio": test_ratio,
        "actual_test_ratio": len(test_samples) / total_samples,
        "total_frames": len(frames),
        "train_frames": train_frames,
        "test_frames": test_frames,
        "total_samples": total_samples,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "crop_count_histogram": dict(sorted(crop_count_histogram.items())),
        "train_file": str(train_path),
        "test_file": str(test_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one REGION_TO_DESCRIPTION sample for every cleaned Qwen crop."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
