#!/usr/bin/env python3
"""Merge reviewed person crops into the current Florence dataset splits."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


STATUS_ALIASES = {
    "approved": "reviewed",
    "corrected": "reviewed",
    "needs_revision": "uncertain",
}

SPLITS = ("train", "test", "dev", "rl", "reserve")
DEFAULT_BASELINES = {
    "train": 30000,
    "test": 4328,
    "dev": 2000,
    "rl": 6000,
}
DEFAULT_DATASET_ROOT = Path("/data1/work/MichaelYu/florence-attibute/data")
DEFAULT_REVIEW_ROOT = Path(
    "/data0/work/WangHaoxiang/a35_vehicle_label/outputs/person_reviewed/frames"
)
DEFAULT_REWRITE_ROOT = DEFAULT_DATASET_ROOT / "review"

Identity = Tuple[str, tuple]


@dataclass
class SampleRecord:
    identity: Identity
    relative_json: Path
    frame_payload: dict
    crop: dict
    source_id: Optional[str]
    is_xiaohongshu: bool


@dataclass
class ReviewedCrop:
    status: str
    sample: SampleRecord


@dataclass(frozen=True)
class MergeConfig:
    dataset_root: Path
    review_root: Path
    rewrite_root: Path
    threshold: float = 0.90
    shortfall_tolerance: float = 0.04
    baselines: Optional[Dict[str, int]] = None

    def effective_baselines(self) -> Dict[str, int]:
        return dict(self.baselines or DEFAULT_BASELINES)


@dataclass
class XiaohongshuAllocation:
    assignments: Dict[str, str]
    train_crops: int
    test_crops: int

    def __getitem__(self, source_id: str) -> str:
        return self.assignments[source_id]


class UnionFind:
    def __init__(self, values: Sequence[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


class PublicationBlocked(RuntimeError):
    pass


class ReviewedLeakageConflict(PublicationBlocked):
    pass


class CaptionSimilarity:
    def __init__(self, corpus: Sequence[str]):
        if not corpus:
            raise ValueError("caption similarity corpus cannot be empty")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        self.vectorizer.fit(str(caption) for caption in corpus)

    def pairs(
        self, captions: Sequence[str], *, threshold: float
    ) -> Set[Tuple[int, int]]:
        if len(captions) < 2:
            return set()
        matrix = self.vectorizer.transform(str(caption) for caption in captions)
        similarities = (matrix @ matrix.T).tocoo()
        return {
            (row, column)
            for row, column, score in zip(
                similarities.row, similarities.col, similarities.data
            )
            if row < column and float(score) + 1e-12 >= threshold
        }

    def conflict_queries(
        self,
        query_captions: Sequence[str],
        reference_captions: Sequence[str],
        *,
        threshold: float,
    ) -> Set[int]:
        if not query_captions or not reference_captions:
            return set()
        query = self.vectorizer.transform(
            str(caption) for caption in query_captions
        )
        reference = self.vectorizer.transform(
            str(caption) for caption in reference_captions
        )
        similarities = (query @ reference.T).tocoo()
        return {
            row
            for row, score in zip(similarities.row, similarities.data)
            if float(score) + 1e-12 >= threshold
        }


def normalize_review_status(crop: dict) -> str:
    review = crop.get("human_review")
    if not isinstance(review, dict):
        legacy = crop.get("review")
        review = (
            legacy
            if isinstance(legacy, dict) and legacy.get("updated_at")
            else {}
        )
    status = str(review.get("status") or "unreviewed").strip().lower()
    return STATUS_ALIASES.get(status, status)


def iter_json_paths(root: Path) -> Iterator[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in sorted(filenames):
            if filename.endswith(".json") and not filename.startswith("."):
                yield Path(directory) / filename


def frame_key(record: dict, relative_path: Path) -> str:
    image = record.get("image")
    if isinstance(image, dict):
        scene_path = image.get("scene_relative_path")
        if isinstance(scene_path, str) and scene_path.strip():
            return Path(scene_path).with_suffix("").as_posix()
        dataset_id = image.get("dataset_id")
        artifact_key = image.get("artifact_key")
        if isinstance(dataset_id, str) and isinstance(artifact_key, str):
            return f"小红书/{dataset_id}/{artifact_key}"
    return relative_path.with_suffix("").as_posix()


def crop_key(crop: dict, position: int) -> tuple:
    source_position = crop.get("source_crop_position")
    if isinstance(source_position, int):
        return ("position", source_position)

    detection_id = crop.get("detection_id")
    if detection_id not in (None, ""):
        return ("detection", str(detection_id))

    crop_index = crop.get("crop_index", position + 1)
    bbox = crop.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return (
            "index_bbox",
            int(crop_index),
            tuple(float(value) for value in bbox),
        )
    return ("index_position", int(crop_index), position)


def _source_id(record: dict, relative_path: Path) -> Optional[str]:
    image = record.get("image")
    if isinstance(image, dict):
        source_id = image.get("source_id")
        if isinstance(source_id, str) and source_id.strip():
            return source_id.strip()
    parts = relative_path.parts
    if "videos" in parts:
        index = parts.index("videos")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _sample_from_crop(
    record: dict,
    relative_path: Path,
    crop: dict,
    position: int,
) -> SampleRecord:
    identity = (frame_key(record, relative_path), crop_key(crop, position))
    is_xiaohongshu = bool(
        relative_path.parts and relative_path.parts[0] == "小红书"
    )
    return SampleRecord(
        identity=identity,
        relative_json=relative_path,
        frame_payload=record,
        crop=crop,
        source_id=_source_id(record, relative_path),
        is_xiaohongshu=is_xiaohongshu,
    )


def load_split_samples(
    root: Path, *, split: str, preserve_review_status: bool = False
) -> Dict[Identity, SampleRecord]:
    del split
    samples: Dict[Identity, SampleRecord] = {}
    for path in iter_json_paths(root):
        relative = path.relative_to(root)
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        crops = record.get("crops")
        if not isinstance(crops, list):
            raise ValueError(f"missing crops list: {path}")
        for position, source_crop in enumerate(crops):
            if not isinstance(source_crop, dict):
                raise ValueError(f"invalid crop at {path}#{position}")
            crop = deepcopy(source_crop)
            if not preserve_review_status:
                crop.pop("review_status", None)
            sample = _sample_from_crop(record, relative, crop, position)
            if sample.identity in samples:
                raise ValueError(
                    f"duplicate crop identity in split {root}: {sample.identity}"
                )
            samples[sample.identity] = sample
    return samples


def _load_rewrite_index(root: Path) -> Dict[Identity, dict]:
    rewritten: Dict[Identity, dict] = {}
    for path in iter_json_paths(root):
        relative = path.relative_to(root)
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        crops = record.get("crops")
        if not isinstance(crops, list):
            continue
        for position, crop in enumerate(crops):
            if not isinstance(crop, dict):
                continue
            identity = (frame_key(record, relative), crop_key(crop, position))
            if identity in rewritten:
                raise ValueError(f"duplicate rewritten crop identity: {identity}")
            rewritten[identity] = crop
    return rewritten


def load_reviewed_crops(
    review_root: Path, rewrite_root: Path
) -> Tuple[List[ReviewedCrop], List[SampleRecord], Counter]:
    rewritten = _load_rewrite_index(rewrite_root)
    regular: List[ReviewedCrop] = []
    xiaohongshu: List[SampleRecord] = []
    counts: Counter = Counter()
    seen: Set[Identity] = set()
    for path in iter_json_paths(review_root):
        relative = path.relative_to(review_root)
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        crops = record.get("crops")
        if not isinstance(crops, list):
            raise ValueError(f"missing crops list: {path}")
        is_xiaohongshu = bool(
            relative.parts and relative.parts[0] == "小红书"
        )
        group = "xiaohongshu" if is_xiaohongshu else "regular"
        for position, authoritative in enumerate(crops):
            if not isinstance(authoritative, dict):
                raise ValueError(f"invalid crop at {path}#{position}")
            status = normalize_review_status(authoritative)
            identity = (
                frame_key(record, relative),
                crop_key(authoritative, position),
            )
            counts[f"{group}_{status}"] += 1
            if is_xiaohongshu and status != "reviewed":
                continue
            if identity in seen:
                raise ValueError(f"duplicate authoritative crop identity: {identity}")
            seen.add(identity)
            crop = deepcopy(authoritative)
            if status == "reviewed":
                crop = overlay_rewritten_caption(crop, rewritten.get(identity))
                if is_xiaohongshu:
                    ensure_training_bbox_fields(crop)
            else:
                crop.pop("review_status", None)
            sample = _sample_from_crop(record, relative, crop, position)
            if is_xiaohongshu:
                xiaohongshu.append(sample)
            else:
                regular.append(ReviewedCrop(status, sample))
    return regular, xiaohongshu, counts


def materialize_split(
    samples: Dict[Identity, SampleRecord], output_root: Path
) -> int:
    grouped: Dict[Path, List[SampleRecord]] = {}
    for sample in samples.values():
        grouped.setdefault(sample.relative_json, []).append(sample)
    for relative in sorted(grouped, key=lambda path: path.as_posix()):
        frame_samples = sorted(
            grouped[relative],
            key=lambda sample: (
                int(sample.crop.get("crop_index", 0)),
                repr(sample.identity),
            ),
        )
        payload = deepcopy(frame_samples[0].frame_payload)
        payload["crops"] = [deepcopy(sample.crop) for sample in frame_samples]
        payload["person_crop_count"] = len(payload["crops"])
        if "human_review_summary" in payload:
            payload["human_review_summary"] = {
                "reviewed_crop_count": sum(
                    crop.get("review_status") == "reviewed"
                    for crop in payload["crops"]
                ),
                "total_crop_count": len(payload["crops"]),
                "status_counts": dict(
                    Counter(
                        crop.get("review_status", "unreviewed")
                        for crop in payload["crops"]
                    )
                ),
                "updated_at": payload["human_review_summary"].get("updated_at"),
            }
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return len(grouped)


def overlay_rewritten_caption(
    authoritative: dict, rewritten: Optional[dict]
) -> dict:
    merged = deepcopy(authoritative)
    if isinstance(rewritten, dict):
        caption = rewritten.get("caption")
        if isinstance(caption, str) and caption.strip():
            merged["caption"] = caption.strip()
    merged["review_status"] = "reviewed"
    return merged


def ensure_training_bbox_fields(crop: dict) -> None:
    if "expanded_bbox_xyxy" not in crop and "bbox_xyxy" in crop:
        crop["expanded_bbox_xyxy"] = deepcopy(crop["bbox_xyxy"])
    if "expanded_bbox_xyxy_norm" not in crop and "bbox_xyxy_norm" in crop:
        crop["expanded_bbox_xyxy_norm"] = deepcopy(crop["bbox_xyxy_norm"])


def apply_regular_actions(
    dataset: Dict[str, Dict[Identity, SampleRecord]],
    reviewed_crops: Iterator[ReviewedCrop],
) -> Counter:
    counts: Counter = Counter()
    for reviewed in reviewed_crops:
        identity = reviewed.sample.identity
        if reviewed.status == "reviewed":
            for split_samples in dataset.values():
                split_samples.pop(identity, None)
            sample = deepcopy(reviewed.sample)
            sample.crop["review_status"] = "reviewed"
            dataset["test"][identity] = sample
            counts["regular_reviewed_to_test"] += 1
        elif reviewed.status in {"rejected", "duplicate"}:
            removed = dataset["test"].pop(identity, None)
            if removed is not None:
                counts[
                    f"regular_{reviewed.status}_removed_from_test"
                ] += 1
    return counts


def _vectorize(captions: Sequence[str]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )
    return vectorizer.fit_transform(str(caption) for caption in captions)


def high_similarity_pairs(
    captions: Sequence[str], *, threshold: float
) -> Set[Tuple[int, int]]:
    if len(captions) < 2:
        return set()
    return CaptionSimilarity(captions).pairs(captions, threshold=threshold)


def caption_conflict_queries(
    query_captions: Sequence[str],
    reference_captions: Sequence[str],
    *,
    threshold: float,
) -> Set[int]:
    if not query_captions or not reference_captions:
        return set()
    combined = list(query_captions) + list(reference_captions)
    return CaptionSimilarity(combined).conflict_queries(
        query_captions, reference_captions, threshold=threshold
    )


def allocate_xiaohongshu(
    samples: Sequence[SampleRecord],
    *,
    regular_test_captions: Sequence[str],
    threshold: float,
    similarity: Optional[CaptionSimilarity] = None,
) -> XiaohongshuAllocation:
    if not samples:
        return XiaohongshuAllocation({}, 0, 0)
    if any(not sample.source_id for sample in samples):
        raise ValueError("Xiaohongshu samples require a source_id")

    source_ids = sorted({str(sample.source_id) for sample in samples})
    union_find = UnionFind(source_ids)
    captions = [str(sample.crop.get("caption") or "") for sample in samples]
    similarity = similarity or CaptionSimilarity(
        [*captions, *regular_test_captions]
    )
    for left, right in similarity.pairs(captions, threshold=threshold):
        union_find.union(str(samples[left].source_id), str(samples[right].source_id))

    components: Dict[str, List[str]] = {}
    for source_id in source_ids:
        components.setdefault(union_find.find(source_id), []).append(source_id)

    conflicting_indices = similarity.conflict_queries(
        captions,
        regular_test_captions,
        threshold=threshold,
    )
    pinned_roots = {
        union_find.find(str(samples[index].source_id))
        for index in conflicting_indices
    }
    source_counts = Counter(str(sample.source_id) for sample in samples)
    component_counts = {
        root: sum(source_counts[source_id] for source_id in members)
        for root, members in components.items()
    }

    assignments: Dict[str, str] = {}
    train_count = 0
    test_count = 0
    for root in sorted(pinned_roots):
        for source_id in components[root]:
            assignments[source_id] = "test"
        test_count += component_counts[root]

    target = len(samples) / 2.0
    remaining = sorted(
        (root for root in components if root not in pinned_roots),
        key=lambda root: (-component_counts[root], tuple(components[root])),
    )
    for root in remaining:
        size = component_counts[root]
        train_score = abs((train_count + size) - target) + abs(test_count - target)
        test_score = abs(train_count - target) + abs((test_count + size) - target)
        destination = "train" if train_score <= test_score else "test"
        for source_id in components[root]:
            assignments[source_id] = destination
        if destination == "train":
            train_count += size
        else:
            test_count += size

    return XiaohongshuAllocation(assignments, train_count, test_count)


def _is_reviewed(sample: SampleRecord) -> bool:
    return sample.crop.get("review_status") == "reviewed"


def remove_train_test_leakage(
    train: Dict[Identity, SampleRecord],
    test: Dict[Identity, SampleRecord],
    *,
    threshold: float,
    similarity: Optional[CaptionSimilarity] = None,
) -> Set[Identity]:
    train_items = list(train.items())
    test_items = list(test.items())
    train_captions = [
        str(sample.crop.get("caption") or "") for _, sample in train_items
    ]
    test_captions = [
        str(sample.crop.get("caption") or "") for _, sample in test_items
    ]
    similarity = similarity or CaptionSimilarity(
        [*train_captions, *test_captions]
    )
    conflicting = similarity.conflict_queries(
        train_captions,
        test_captions,
        threshold=threshold,
    )
    reviewed_conflicts = [
        train_items[index][0]
        for index in sorted(conflicting)
        if _is_reviewed(train_items[index][1])
    ]
    if reviewed_conflicts:
        raise ReviewedLeakageConflict(
            f"reviewed train captions conflict with test: {reviewed_conflicts[:5]}"
        )
    removed = {train_items[index][0] for index in conflicting}
    for identity in removed:
        train.pop(identity, None)
    return removed


def _normalized_caption(caption: object) -> str:
    return " ".join(re.findall(r"\w+", str(caption).lower()))


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def reduce_to_limit(
    samples: Dict[Identity, SampleRecord],
    *,
    limit: int,
    protected: Optional[Set[Identity]] = None,
) -> Set[Identity]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    reviewed_count = sum(_is_reviewed(sample) for sample in samples.values())
    if reviewed_count > limit:
        raise PublicationBlocked(
            f"reviewed crop count {reviewed_count} exceeds split limit {limit}"
        )
    remove_count = max(0, len(samples) - limit)
    if not remove_count:
        return set()

    caption_counts = Counter(
        _normalized_caption(sample.crop.get("caption"))
        for sample in samples.values()
    )
    protected = protected or set()
    removable = [
        (identity, sample)
        for identity, sample in samples.items()
        if not _is_reviewed(sample) and identity not in protected
    ]
    removable.sort(
        key=lambda item: (
            -caption_counts[_normalized_caption(item[1].crop.get("caption"))],
            _number(item[1].crop.get("confidence")),
            _number(item[1].crop.get("crop_area")),
            repr(item[0]),
        )
    )
    if len(removable) < remove_count:
        raise PublicationBlocked("not enough non-reviewed crops to reach limit")
    removed = {identity for identity, _ in removable[:remove_count]}
    for identity in removed:
        samples.pop(identity, None)
    return removed


def validate_counts(
    actual: Dict[str, int],
    baselines: Dict[str, int],
    *,
    tolerance: float,
) -> None:
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    failures = []
    for split, baseline in baselines.items():
        count = actual.get(split, 0)
        if count > baseline:
            failures.append(f"{split} exceeds baseline: {count}>{baseline}")
        shortfall = baseline - count
        if shortfall > baseline * tolerance + 1e-9:
            failures.append(
                f"{split} shortfall exceeds {tolerance:.1%}: "
                f"{shortfall}/{baseline}"
            )
    if failures:
        raise PublicationBlocked("; ".join(failures))


def remove_test_conflicts_with_reviewed_train(
    train: Dict[Identity, SampleRecord],
    test: Dict[Identity, SampleRecord],
    *,
    threshold: float,
    similarity: Optional[CaptionSimilarity] = None,
) -> Set[Identity]:
    reviewed_train = [sample for sample in train.values() if _is_reviewed(sample)]
    non_reviewed_test = [
        (identity, sample)
        for identity, sample in test.items()
        if not _is_reviewed(sample)
    ]
    test_captions = [
        str(sample.crop.get("caption") or "") for _, sample in non_reviewed_test
    ]
    reviewed_captions = [
        str(sample.crop.get("caption") or "") for sample in reviewed_train
    ]
    if not test_captions or not reviewed_captions:
        return set()
    similarity = similarity or CaptionSimilarity(
        [*test_captions, *reviewed_captions]
    )
    conflicting = similarity.conflict_queries(
        test_captions,
        reviewed_captions,
        threshold=threshold,
    )
    removed = {non_reviewed_test[index][0] for index in conflicting}
    for identity in removed:
        test.pop(identity, None)
    return removed


def merge_samples(
    dataset: Dict[str, Dict[Identity, SampleRecord]],
    regular_reviews: Sequence[ReviewedCrop],
    xiaohongshu_reviewed: Sequence[SampleRecord],
    *,
    review_status_counts: Dict[str, int],
    baselines: Dict[str, int],
    threshold: float,
    shortfall_tolerance: float,
    similarity: Optional[CaptionSimilarity] = None,
) -> dict:
    before = {split: len(samples) for split, samples in dataset.items()}
    if similarity is None:
        corpus = [
            str(sample.crop.get("caption") or "")
            for split_samples in dataset.values()
            for sample in split_samples.values()
        ]
        corpus.extend(
            str(review.sample.crop.get("caption") or "")
            for review in regular_reviews
        )
        corpus.extend(
            str(sample.crop.get("caption") or "")
            for sample in xiaohongshu_reviewed
        )
        similarity = CaptionSimilarity(corpus)
    actions = apply_regular_actions(dataset, iter(regular_reviews))

    regular_test_captions = [
        str(sample.crop.get("caption") or "")
        for sample in dataset["test"].values()
        if _is_reviewed(sample) and not sample.is_xiaohongshu
    ]
    allocation = allocate_xiaohongshu(
        xiaohongshu_reviewed,
        regular_test_captions=regular_test_captions,
        threshold=threshold,
        similarity=similarity,
    )
    for source_sample in xiaohongshu_reviewed:
        for split_samples in dataset.values():
            split_samples.pop(source_sample.identity, None)
        sample = deepcopy(source_sample)
        sample.crop["review_status"] = "reviewed"
        destination = allocation[str(sample.source_id)]
        dataset[destination][sample.identity] = sample
        actions[f"xiaohongshu_reviewed_to_{destination}"] += 1

    removed_test_conflicts = remove_test_conflicts_with_reviewed_train(
        dataset["train"],
        dataset["test"],
        threshold=threshold,
        similarity=similarity,
    )
    removed_train_leakage = remove_train_test_leakage(
        dataset["train"],
        dataset["test"],
        threshold=threshold,
        similarity=similarity,
    )
    for identity in removed_train_leakage:
        dataset["rl"].pop(identity, None)

    reduced = {}
    reduced["train"] = reduce_to_limit(
        dataset["train"],
        limit=baselines["train"],
        protected=set(dataset["rl"]),
    )
    reduced["test"] = reduce_to_limit(
        dataset["test"], limit=baselines["test"]
    )
    for identity in reduced["train"]:
        dataset["rl"].pop(identity, None)

    dataset["rl"] = {
        identity: sample
        for identity, sample in dataset["rl"].items()
        if identity in dataset["train"]
    }
    after = {split: len(samples) for split, samples in dataset.items()}
    validate_counts(after, baselines, tolerance=shortfall_tolerance)

    remaining_conflicts = similarity.conflict_queries(
        [
            str(sample.crop.get("caption") or "")
            for sample in dataset["train"].values()
        ],
        [
            str(sample.crop.get("caption") or "")
            for sample in dataset["test"].values()
        ],
        threshold=threshold,
    )
    if remaining_conflicts:
        raise PublicationBlocked(
            f"train/test caption conflicts remain: {len(remaining_conflicts)}"
        )
    if not set(dataset["rl"]).issubset(dataset["train"]):
        raise PublicationBlocked("RL is not a subset of train")

    xhs_locations: Dict[str, Set[str]] = {}
    for split in ("train", "test"):
        for sample in dataset[split].values():
            if sample.is_xiaohongshu and _is_reviewed(sample):
                xhs_locations.setdefault(str(sample.source_id), set()).add(split)
    crossing = {
        source_id: locations
        for source_id, locations in xhs_locations.items()
        if len(locations) > 1
    }
    if crossing:
        raise PublicationBlocked(
            f"Xiaohongshu source groups cross splits: {list(crossing)[:5]}"
        )

    reviewed_locations = Counter()
    for split, split_samples in dataset.items():
        for sample in split_samples.values():
            status = sample.crop.get("review_status")
            if status is not None and status != "reviewed":
                raise PublicationBlocked(
                    f"invalid review_status on {sample.identity}: {status!r}"
                )
            if status == "reviewed":
                reviewed_locations[split] += 1

    return {
        "before": before,
        "after": after,
        "review_status_counts": dict(review_status_counts),
        "actions": dict(actions),
        "xiaohongshu_allocation": {
            "train_crops": allocation.train_crops,
            "test_crops": allocation.test_crops,
            "source_groups": len(allocation.assignments),
        },
        "removed_test_conflicts_with_reviewed_train": len(
            removed_test_conflicts
        ),
        "removed_train_test_leakage": len(removed_train_leakage),
        "reduced_to_baseline": {
            split: len(identities) for split, identities in reduced.items()
        },
        "reviewed_by_split": dict(reviewed_locations),
        "threshold": threshold,
        "shortfall_tolerance": shortfall_tolerance,
        "validation_errors": 0,
    }


def load_dataset(
    dataset_root: Path,
) -> Dict[str, Dict[Identity, SampleRecord]]:
    dataset = {}
    for split in SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_root}")
        dataset[split] = load_split_samples(split_root, split=split)
    return dataset


def build_caption_similarity(
    dataset: Dict[str, Dict[Identity, SampleRecord]],
    regular_reviews: Sequence[ReviewedCrop],
    xiaohongshu_reviewed: Sequence[SampleRecord],
) -> CaptionSimilarity:
    corpus = [
        str(sample.crop.get("caption") or "")
        for split_samples in dataset.values()
        for sample in split_samples.values()
    ]
    corpus.extend(
        str(review.sample.crop.get("caption") or "")
        for review in regular_reviews
    )
    corpus.extend(
        str(sample.crop.get("caption") or "")
        for sample in xiaohongshu_reviewed
    )
    return CaptionSimilarity(corpus)


def _session_key(relative_json: Path) -> str:
    if len(relative_json.parts) > 2:
        return "/".join(relative_json.parts[:2])
    stem = relative_json.stem
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit() and prefix:
        stem = prefix
    return f"{relative_json.parts[0]}/{stem}"


def _group_split_samples(
    samples: Dict[Identity, SampleRecord],
) -> Dict[Path, List[SampleRecord]]:
    grouped: Dict[Path, List[SampleRecord]] = {}
    for sample in samples.values():
        grouped.setdefault(sample.relative_json, []).append(sample)
    for frame_samples in grouped.values():
        frame_samples.sort(
            key=lambda sample: (
                int(sample.crop.get("crop_index", 0)),
                repr(sample.identity),
            )
        )
    return grouped


def write_manifest(
    samples: Dict[Identity, SampleRecord], path: Path, split: str
) -> int:
    grouped = _group_split_samples(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for relative in sorted(grouped, key=lambda value: value.as_posix()):
            for position, sample in enumerate(grouped[relative]):
                dimensions = sample.crop.get("dimensions")
                scale = (
                    dimensions.get("scale_bin")
                    if isinstance(dimensions, dict)
                    else None
                )
                row = {
                    "sample_id": f"{relative.as_posix()}#{position}",
                    "split": split,
                    "scene": relative.parts[0],
                    "session": _session_key(relative),
                    "source_json_relative_path": relative.as_posix(),
                    "crop_position": position,
                    "crop_index": int(sample.crop.get("crop_index", position + 1)),
                    "caption": str(sample.crop.get("caption") or ""),
                    "scale": scale,
                    "selection_score": 0.0,
                    "rl_selection_score": 0.0,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                count += 1
    return count


def _split_statistics(samples: Dict[Identity, SampleRecord]) -> dict:
    scenes = Counter(sample.relative_json.parts[0] for sample in samples.values())
    return {
        "people": len(samples),
        "frames": len({sample.relative_json for sample in samples.values()}),
        "reviewed": sum(_is_reviewed(sample) for sample in samples.values()),
        "scenes": dict(sorted(scenes.items())),
    }


def build_staging(
    dataset: Dict[str, Dict[Identity, SampleRecord]],
    staging_root: Path,
    report: dict,
) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    materialized = {}
    for split in SPLITS:
        frames = materialize_split(dataset[split], staging_root / split)
        materialized[split] = {
            "frames": frames,
            "people": len(dataset[split]),
        }
        write_manifest(
            dataset[split],
            staging_root / "manifests" / f"{split}.jsonl",
            split,
        )
    statistics = {
        "generated_at": report["generated_at"],
        "source": "reviewed-person-dataset-merge",
        "caption_similarity_threshold": report["threshold"],
        "shortfall_tolerance": report["shortfall_tolerance"],
        **{
            split: _split_statistics(dataset[split])
            for split in SPLITS
        },
        "materialized": materialized,
    }
    _write_json_atomic(staging_root / "split_statistics.json", statistics)
    report["materialized"] = materialized


def validate_staging(staging_root: Path, expected: Dict[str, int]) -> None:
    for split in SPLITS:
        split_root = staging_root / split
        crop_count = 0
        for path in iter_json_paths(split_root):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            crops = payload.get("crops")
            if not isinstance(crops, list) or not crops:
                raise PublicationBlocked(f"invalid staged crops: {path}")
            if payload.get("person_crop_count") != len(crops):
                raise PublicationBlocked(f"staged crop count mismatch: {path}")
            for crop in crops:
                status = crop.get("review_status")
                if status is not None and status != "reviewed":
                    raise PublicationBlocked(
                        f"invalid staged review_status: {path} {status!r}"
                    )
            crop_count += len(crops)
        if crop_count != expected.get(split, crop_count):
            raise PublicationBlocked(
                f"staged {split} count mismatch: {crop_count} != {expected[split]}"
            )
        manifest = staging_root / "manifests" / f"{split}.jsonl"
        manifest_count = 0
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source = split_root / row["source_json_relative_path"]
                if not source.is_file():
                    raise PublicationBlocked(
                        f"manifest source does not exist: {source}"
                    )
                manifest_count += 1
        if manifest_count != crop_count:
            raise PublicationBlocked(
                f"manifest count mismatch for {split}: "
                f"{manifest_count} != {crop_count}"
            )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_staging(staging_root: Path, dataset_root: Path) -> None:
    work_root = dataset_root / ".reviewed_merge_work"
    backup_root = work_root / "backup"
    if backup_root.exists():
        raise PublicationBlocked(
            f"backup from an earlier publication exists: {backup_root}"
        )
    backup_root.mkdir(parents=True)
    targets = [*SPLITS, "manifests", "split_statistics.json"]
    backed_up: List[str] = []
    installed: List[str] = []
    try:
        for name in targets:
            current = dataset_root / name
            if current.exists() or current.is_symlink():
                destination = backup_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                current.replace(destination)
                backed_up.append(name)
        for name in targets:
            source = staging_root / name
            if not source.exists():
                raise PublicationBlocked(f"missing staged publication target: {source}")
            source.replace(dataset_root / name)
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            _remove_path(dataset_root / name)
        for name in reversed(backed_up):
            source = backup_root / name
            source.replace(dataset_root / name)
        raise
    else:
        shutil.rmtree(backup_root)
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_merge(config: MergeConfig, *, publish: bool) -> Path:
    dataset_root = config.dataset_root.resolve(strict=True)
    review_root = config.review_root.resolve(strict=True)
    rewrite_root = config.rewrite_root.resolve(strict=True)
    dataset = load_dataset(dataset_root)
    regular, xiaohongshu, status_counts = load_reviewed_crops(
        review_root, rewrite_root
    )
    similarity = build_caption_similarity(dataset, regular, xiaohongshu)
    report = merge_samples(
        dataset,
        regular,
        xiaohongshu,
        review_status_counts=dict(status_counts),
        baselines=config.effective_baselines(),
        threshold=config.threshold,
        shortfall_tolerance=config.shortfall_tolerance,
        similarity=similarity,
    )
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "review_root": str(review_root),
            "rewrite_root": str(rewrite_root),
            "published": publish,
        }
    )
    work_root = dataset_root / ".reviewed_merge_work"
    report_path = work_root / (
        "publish_report.json" if publish else "dry_run_report.json"
    )
    if publish:
        staging_root = work_root / "staging"
        build_staging(dataset, staging_root, report)
        validate_staging(staging_root, report["after"])
        publish_staging(staging_root, dataset_root)
        model_path = work_root / "caption_similarity.joblib"
        temporary_model = model_path.with_name(f".{model_path.name}.tmp")
        joblib.dump(similarity, temporary_model)
        temporary_model.replace(model_path)
        report["caption_similarity_model"] = str(model_path)
    _write_json_atomic(report_path, report)
    return report_path


def validate_published(config: MergeConfig) -> dict:
    dataset_root = config.dataset_root.resolve(strict=True)
    dataset = {
        split: load_split_samples(
            dataset_root / split,
            split=split,
            preserve_review_status=True,
        )
        for split in SPLITS
    }
    regular, xiaohongshu, _ = load_reviewed_crops(
        config.review_root.resolve(strict=True),
        config.rewrite_root.resolve(strict=True),
    )
    model_path = (
        dataset_root / ".reviewed_merge_work" / "caption_similarity.joblib"
    )
    similarity = (
        joblib.load(model_path)
        if model_path.is_file()
        else build_caption_similarity(dataset, regular, xiaohongshu)
    )
    counts = {split: len(samples) for split, samples in dataset.items()}
    validate_counts(
        counts,
        config.effective_baselines(),
        tolerance=config.shortfall_tolerance,
    )

    regular_reviewed = {
        review.sample.identity
        for review in regular
        if review.status == "reviewed"
    }
    regular_bad = {
        review.sample.identity
        for review in regular
        if review.status in {"rejected", "duplicate"}
    }
    xhs_reviewed = {sample.identity for sample in xiaohongshu}
    expected_reviewed = regular_reviewed | xhs_reviewed
    for identity in regular_reviewed:
        locations = [
            split for split in SPLITS if identity in dataset[split]
        ]
        if locations != ["test"]:
            raise PublicationBlocked(
                f"regular reviewed crop has invalid locations: {identity} {locations}"
            )
    bad_in_test = regular_bad & set(dataset["test"])
    if bad_in_test:
        raise PublicationBlocked(
            f"regular rejected/duplicate crops remain in test: {len(bad_in_test)}"
        )
    for identity in xhs_reviewed:
        locations = [
            split for split in ("train", "test") if identity in dataset[split]
        ]
        if len(locations) != 1:
            raise PublicationBlocked(
                f"Xiaohongshu reviewed crop has invalid locations: "
                f"{identity} {locations}"
            )

    actual_reviewed: Set[Identity] = set()
    for split_samples in dataset.values():
        for identity, sample in split_samples.items():
            status = sample.crop.get("review_status")
            if status is None:
                continue
            if status != "reviewed" or identity not in expected_reviewed:
                raise PublicationBlocked(
                    f"unexpected review_status on published crop: {identity}"
                )
            actual_reviewed.add(identity)
            if sample.is_xiaohongshu and "expanded_bbox_xyxy" not in sample.crop:
                raise PublicationBlocked(
                    f"Xiaohongshu reviewed crop lacks expanded bbox: {identity}"
                )
    if actual_reviewed != expected_reviewed:
        raise PublicationBlocked(
            f"reviewed identity mismatch: actual={len(actual_reviewed)} "
            f"expected={len(expected_reviewed)}"
        )

    conflicts = similarity.conflict_queries(
        [
            str(sample.crop.get("caption") or "")
            for sample in dataset["train"].values()
        ],
        [
            str(sample.crop.get("caption") or "")
            for sample in dataset["test"].values()
        ],
        threshold=config.threshold,
    )
    if conflicts:
        raise PublicationBlocked(
            f"published train/test caption conflicts: {len(conflicts)}"
        )
    if not set(dataset["rl"]).issubset(dataset["train"]):
        raise PublicationBlocked("published RL is not a subset of train")

    source_locations: Dict[str, Set[str]] = {}
    for split in ("train", "test"):
        for sample in dataset[split].values():
            if sample.is_xiaohongshu and _is_reviewed(sample):
                source_locations.setdefault(str(sample.source_id), set()).add(split)
    if any(len(locations) > 1 for locations in source_locations.values()):
        raise PublicationBlocked("published Xiaohongshu source crosses splits")
    return {
        "counts": counts,
        "reviewed": len(actual_reviewed),
        "regular_reviewed": len(regular_reviewed),
        "xiaohongshu_reviewed": len(xhs_reviewed),
        "validation_errors": 0,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--rewrite-root", type=Path, default=DEFAULT_REWRITE_ROOT)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--shortfall-tolerance", type=float, default=0.04)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = MergeConfig(
        dataset_root=args.dataset_root,
        review_root=args.review_root,
        rewrite_root=args.rewrite_root,
        threshold=args.threshold,
        shortfall_tolerance=args.shortfall_tolerance,
    )
    if args.validate_only:
        result = validate_published(config)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        report_path = run_merge(config, publish=args.publish)
        print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
