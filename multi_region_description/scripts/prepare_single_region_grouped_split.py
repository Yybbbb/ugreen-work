#!/usr/bin/env python3
"""Build single-region JSONL while keeping similar-frame groups in one split."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from prepare_single_region_data import load_frame, write_jsonl


DEFAULT_INPUT_DIR = Path(
    "/data/work/MichaelYu/florence-data/"
    "qwen_person_2to6_region_descriptions_qwen_cleaned_dhash10_dedup"
)
DEFAULT_SIMILARITY_DIR = Path(
    "/data/work/MichaelYu/florence-data/outputs/similarity_analysis/"
    "qwen_cleaned_dhash10_dedup_split_groups"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "qwen-person-2to6-cleaned-dhash10-single-region"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def select_test_group_indexes(groups: list[dict], target_samples: int) -> set[int]:
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for group_index, group in enumerate(groups):
        group_samples = len(group["samples"])
        for current_sum in sorted(parents, reverse=True):
            next_sum = current_sum + group_samples
            if next_sum <= target_samples and next_sum not in parents:
                parents[next_sum] = (current_sum, group_index)
        if target_samples in parents:
            break

    selected_sum = target_samples if target_samples in parents else max(parents)
    selected = set()
    while selected_sum:
        previous_sum, group_index = parents[selected_sum]
        selected.add(group_index)
        selected_sum = previous_sum
    return selected


def build_dataset(
    input_dir: Path,
    similarity_dir: Path,
    output_dir: Path,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1)")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    similarity_summary = json.loads((similarity_dir / "summary.json").read_text(encoding="utf-8"))
    clusters = read_jsonl(similarity_dir / "similar_frame_clusters.jsonl")
    similar_pairs = read_jsonl(similarity_dir / "similar_frame_pairs.jsonl")

    frame_to_group = {}
    group_frames = {}
    for cluster_index, cluster in enumerate(clusters, start=1):
        group_id = f"similar_cluster_{cluster_index:05d}"
        group_frames[group_id] = list(cluster["frames"])
        for frame_id in cluster["frames"]:
            if frame_id in frame_to_group:
                raise ValueError(f"Frame appears in multiple similarity clusters: {frame_id}")
            frame_to_group[frame_id] = group_id

    frames = {}
    for json_path in sorted(input_dir.rglob("*.json")):
        frame = load_frame(json_path, input_dir)
        frames[frame["frame_id"]] = frame

    for frame_id in frames:
        if frame_id not in frame_to_group:
            group_id = f"singleton::{frame_id}"
            frame_to_group[frame_id] = group_id
            group_frames[group_id] = [frame_id]

    groups = []
    for group_id, frame_ids in group_frames.items():
        samples = []
        for frame_id in frame_ids:
            if frame_id not in frames:
                raise ValueError(f"Similarity group references missing frame: {frame_id}")
            for sample in frames[frame_id]["samples"]:
                samples.append({**sample, "similarity_group_id": group_id})
        groups.append(
            {
                "group_id": group_id,
                "frame_ids": sorted(frame_ids),
                "samples": samples,
                "is_similarity_cluster": not group_id.startswith("singleton::"),
            }
        )

    random.Random(seed).shuffle(groups)
    total_samples = sum(len(group["samples"]) for group in groups)
    target_test_samples = round(total_samples * test_ratio)
    test_group_indexes = select_test_group_indexes(groups, target_test_samples)

    train_samples = []
    test_samples = []
    train_frames = set()
    test_frames = set()
    group_manifest = []
    for group_index, group in enumerate(groups):
        split = "test" if group_index in test_group_indexes else "train"
        destination = test_samples if split == "test" else train_samples
        destination.extend(group["samples"])
        frame_destination = test_frames if split == "test" else train_frames
        frame_destination.update(group["frame_ids"])
        group_manifest.append(
            {
                "group_id": group["group_id"],
                "split": split,
                "is_similarity_cluster": group["is_similarity_cluster"],
                "frame_count": len(group["frame_ids"]),
                "sample_count": len(group["samples"]),
                "frame_ids": group["frame_ids"],
            }
        )

    if train_frames & test_frames:
        raise ValueError("A frame was assigned to both train and test")
    cross_split_similar_pairs = []
    for pair in similar_pairs:
        split_a = "test" if pair["frame_a"] in test_frames else "train"
        split_b = "test" if pair["frame_b"] in test_frames else "train"
        if split_a != split_b:
            cross_split_similar_pairs.append(pair)
    if cross_split_similar_pairs:
        raise ValueError(f"Found {len(cross_split_similar_pairs)} cross-split similar frame pairs")

    write_jsonl(output_dir / "train.jsonl", train_samples)
    write_jsonl(output_dir / "test.jsonl", test_samples)
    write_jsonl(output_dir / "split_groups.jsonl", group_manifest)

    metadata = {
        "schema_version": "qwen_single_region_grouped_split_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "similarity_dir": str(similarity_dir),
        "similarity_thresholds": similarity_summary["thresholds"],
        "split_unit": "connected component of relaxed similar-frame graph; otherwise singleton frame",
        "seed": seed,
        "requested_test_ratio": test_ratio,
        "actual_test_ratio": len(test_samples) / total_samples,
        "total_frames": len(frames),
        "train_frames": len(train_frames),
        "test_frames": len(test_frames),
        "total_samples": total_samples,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "group_count": len(groups),
        "similarity_cluster_count": len(clusters),
        "singleton_group_count": len(groups) - len(clusters),
        "largest_group_frames": max(len(group["frame_ids"]) for group in groups),
        "largest_group_samples": max(len(group["samples"]) for group in groups),
        "cross_split_similar_frame_pairs": 0,
        "train_file": str(output_dir / "train.jsonl"),
        "test_file": str(output_dir / "test.jsonl"),
        "group_manifest": str(output_dir / "split_groups.jsonl"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Split single-region data by similar-frame groups.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--similarity-dir", type=Path, default=DEFAULT_SIMILARITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = build_dataset(
        input_dir=args.input_dir,
        similarity_dir=args.similarity_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
