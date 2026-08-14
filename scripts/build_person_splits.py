#!/usr/bin/env python3
"""Build deterministic A35 person train/dev/test/RL splits.

The filtered candidate and held-out test roots are read-only. Output JSON files
retain the source schema but contain only crops selected for that split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


DEFAULT_SOURCE_ROOT = Path(
    "/data1/work/MichaelYu/data/a35_person_filtered_train_v1"
)
DEFAULT_TEST_ROOT = Path(
    "/nfs/public/common/luoweixing/Person_test/pending_review"
)
DEFAULT_OUTPUT_ROOT = Path("/data1/work/MichaelYu/florence-attibute/data")
DEFAULT_SEED = 20260720

BIG_SCENES = {
    "tradeshow",
    "company_surveillance_mp4_adaptive",
}
BIG_TRAIN_QUOTA = 7500
BIG_TRAIN_FRAME_TARGET = 4000
BIG_SESSION_PERSON_CAP = 750
RL_TOTAL = 6000

# All targets are exact sums of complete sessions in the current candidate pool.
DEV_TARGETS = {
    "2026_04_03_cat_cafe": 42,
    "2026_3_09_industrial_park": 31,
    "2026_3_12_europe_city": 50,
    "2026_3_17_hardware_store": 38,
    "2026_3_20_expo": 53,
    "2026_3_25_urban_village": 26,
    "2026_3_26_cat_cafe": 13,
    "2026_3_30_shopping_mall": 7,
    "2026_4_16_transport_hub": 29,
    "2026_4_21": 6,
    "2026_4_28_park": 9,
    "2026_4_8_cat_cafe": 18,
    "2026_4_9_ikea": 49,
    "ReID_pedestrian": 48,
    "company_surveillance": 19,
    "company_surveillance_mp4_adaptive": 770,
    "fall_scene": 22,
    "tradeshow": 770,
}

UNKNOWN_VALUES = {"", "unknown", "none", "no", "n/a", "null"}


def natural_key(value: object) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def iter_json_paths(root: Path) -> Iterator[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            (name for name in dirnames if not name.startswith(".")), key=natural_key
        )
        for filename in sorted(filenames, key=natural_key):
            if filename.endswith(".json") and not filename.startswith("."):
                yield Path(directory) / filename


def session_key(relative_json: Path) -> str:
    if len(relative_json.parts) > 2:
        return "/".join(relative_json.parts[:2])
    stem = relative_json.stem
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit() and prefix:
        stem = prefix
    return f"{relative_json.parts[0]}/{stem}"


def _known_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) and str(value).strip().lower() not in UNKNOWN_VALUES


def flatten_attributes(value: object, prefix: str = "") -> List[str]:
    tokens: List[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            tokens.extend(flatten_attributes(value[key], child))
    elif isinstance(value, list):
        for item in value:
            tokens.extend(flatten_attributes(item, prefix))
    elif prefix and _known_scalar(value):
        tokens.append(f"{prefix}={str(value).strip().lower()}")
    return tokens


@dataclass
class Sample:
    identifier: object
    scene: str
    session: str
    relative_json: Path
    crop_position: int
    crop_index: int
    caption: str
    confidence: float
    crop_area: float
    scale: str
    attributes: Tuple[str, ...]
    extra_count: int
    score: float = 0.0
    rl_score: float = 0.0

    @classmethod
    def for_test(
        cls,
        *,
        identifier: object,
        scene: str,
        session: str,
        frame: str,
        crop_position: int,
        score: float,
    ) -> "Sample":
        return cls(
            identifier=identifier,
            scene=scene,
            session=session,
            relative_json=Path(frame),
            crop_position=crop_position,
            crop_index=crop_position + 1,
            caption=str(identifier),
            confidence=1.0,
            crop_area=100.0,
            scale="medium",
            attributes=(),
            extra_count=0,
            score=score,
            rl_score=score,
        )


def load_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for path in iter_json_paths(root):
        relative = path.relative_to(root)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        crops = payload.get("crops")
        if not isinstance(crops, list):
            raise ValueError(f"Missing crops list: {path}")
        scene = relative.parts[0]
        session = session_key(relative)
        for position, crop in enumerate(crops):
            caption = str(crop.get("caption", "")).strip()
            if not caption:
                raise ValueError(f"Empty caption: {path} crop={position}")
            try:
                confidence = float(crop.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                area = float(crop.get("crop_area") or 0.0)
            except (TypeError, ValueError):
                area = 0.0
            dimensions = crop.get("dimensions") or {}
            attributes = crop.get("attributes") or {}
            extra = attributes.get("extra") if isinstance(attributes, dict) else []
            tokens = tuple(sorted(set(flatten_attributes(attributes))))
            crop_index = int(crop.get("crop_index", position + 1))
            identifier = f"{relative.as_posix()}#{position}"
            samples.append(
                Sample(
                    identifier=identifier,
                    scene=scene,
                    session=session,
                    relative_json=relative,
                    crop_position=position,
                    crop_index=crop_index,
                    caption=caption,
                    confidence=max(0.0, min(1.0, confidence)),
                    crop_area=max(0.0, area),
                    scale=str(dimensions.get("scale_bin", "unknown")),
                    attributes=tokens,
                    extra_count=len(extra) if isinstance(extra, list) else 0,
                )
            )
    return samples


def _rank_map(samples: Sequence[Sample], getter) -> Dict[object, float]:
    ordered = sorted(samples, key=lambda sample: (getter(sample), str(sample.identifier)))
    denominator = max(1, len(ordered) - 1)
    return {sample.identifier: index / denominator for index, sample in enumerate(ordered)}


def assign_scores(samples: Sequence[Sample], *, seed: int) -> None:
    attribute_frequency = Counter()
    session_size = Counter(sample.session for sample in samples)
    for sample in samples:
        attribute_frequency.update(sample.attributes)

    by_scene: Dict[str, List[Sample]] = defaultdict(list)
    rarity_raw: Dict[object, float] = {}
    for sample in samples:
        by_scene[sample.scene].append(sample)
        values = [1.0 / math.sqrt(attribute_frequency[token]) for token in sample.attributes]
        rarity_raw[sample.identifier] = sum(values) / len(values) if values else 0.0

    for scene_samples in by_scene.values():
        area_rank = _rank_map(scene_samples, lambda sample: math.log1p(sample.crop_area))
        rarity_rank = _rank_map(scene_samples, lambda sample: rarity_raw[sample.identifier])
        for sample in scene_samples:
            novelty = min(1.0, math.sqrt(10.0 / max(1, session_size[sample.session])))
            sample.score = (
                0.35 * sample.confidence
                + 0.25 * area_rank[sample.identifier]
                + 0.25 * rarity_rank[sample.identifier]
                + 0.15 * novelty
                + 1e-6 * stable_fraction(str(sample.identifier), seed)
            )
            known = min(1.0, len(sample.attributes) / 14.0)
            extra = min(1.0, sample.extra_count / 3.0)
            scale_bonus = 1.0 if sample.scale in {"tiny", "large"} else 0.5
            sample.rl_score = (
                0.30 * known
                + 0.25 * rarity_rank[sample.identifier]
                + 0.20 * extra
                + 0.15 * novelty
                + 0.10 * scale_bonus
                + 1e-6 * stable_fraction(str(sample.identifier), seed + 1)
            )


def choose_sessions_exact(
    session_sizes: Dict[str, int], *, target: int, seed: int
) -> Set[str]:
    if target < 0:
        raise ValueError("target cannot be negative")
    if target == 0:
        return set()
    ordered = sorted(
        session_sizes,
        key=lambda key: (stable_fraction(key, seed), natural_key(key)),
    )
    previous: List[Optional[Tuple[int, str]]] = [None] * (target + 1)
    reachable = [False] * (target + 1)
    reachable[0] = True
    for key in ordered:
        size = session_sizes[key]
        if size <= 0 or size > target:
            continue
        for subtotal in range(target, size - 1, -1):
            if not reachable[subtotal] and reachable[subtotal - size]:
                reachable[subtotal] = True
                previous[subtotal] = (subtotal - size, key)
        if reachable[target]:
            break
    if not reachable[target]:
        raise ValueError(
            f"No exact complete-session subset for target={target}; "
            f"capacity={sum(session_sizes.values())}"
        )
    selected: Set[str] = set()
    subtotal = target
    while subtotal:
        predecessor = previous[subtotal]
        if predecessor is None:
            raise AssertionError("broken session subset predecessor chain")
        subtotal, key = predecessor
        selected.add(key)
    return selected


def select_ranked_frames(
    records: Sequence[Sample],
    *,
    quota: int,
    frame_target: int,
    session_person_cap: int,
    score_field: str = "score",
) -> List[Sample]:
    if quota < 0 or frame_target <= 0 or session_person_cap <= 0:
        raise ValueError("invalid selection limits")
    if quota > len(records):
        raise ValueError(f"quota {quota} exceeds capacity {len(records)}")
    frames: Dict[Path, List[Sample]] = defaultdict(list)
    for record in records:
        frames[record.relative_json].append(record)
    for frame_records in frames.values():
        frame_records.sort(
            key=lambda record: (getattr(record, score_field), str(record.identifier)),
            reverse=True,
        )

    ranked_frames = sorted(
        frames,
        key=lambda frame: (
            min(2, len(frames[frame])),
            getattr(frames[frame][0], score_field),
            natural_key(frame),
        ),
        reverse=True,
    )
    desired_frames = min(frame_target, len(ranked_frames), quota)
    frame_session_cap = max(1, session_person_cap // 2)
    chosen_frames: List[Path] = []
    session_frames = Counter()
    for frame in ranked_frames:
        session = frames[frame][0].session
        if session_frames[session] >= frame_session_cap:
            continue
        chosen_frames.append(frame)
        session_frames[session] += 1
        if len(chosen_frames) == desired_frames:
            break
    if len(chosen_frames) < desired_frames:
        chosen_set = set(chosen_frames)
        for frame in ranked_frames:
            if frame not in chosen_set:
                chosen_frames.append(frame)
                chosen_set.add(frame)
                if len(chosen_frames) == desired_frames:
                    break

    while sum(len(frames[frame]) for frame in chosen_frames) < quota:
        chosen_set = set(chosen_frames)
        next_frame = next((frame for frame in ranked_frames if frame not in chosen_set), None)
        if next_frame is None:
            raise ValueError("selected frames cannot satisfy person quota")
        chosen_frames.append(next_frame)

    selected: List[Sample] = []
    selected_ids: Set[object] = set()
    session_people = Counter()
    maximum_round = max(len(frames[frame]) for frame in chosen_frames)
    for round_index in range(maximum_round):
        candidates = [
            frames[frame][round_index]
            for frame in chosen_frames
            if len(frames[frame]) > round_index
        ]
        candidates.sort(
            key=lambda record: (getattr(record, score_field), str(record.identifier)),
            reverse=True,
        )
        for record in candidates:
            if session_people[record.session] >= session_person_cap:
                continue
            selected.append(record)
            selected_ids.add(record.identifier)
            session_people[record.session] += 1
            if len(selected) == quota:
                return selected

    # Relax the per-session cap only when the hard scene quota cannot otherwise be met.
    remaining = sorted(
        (record for frame in chosen_frames for record in frames[frame] if record.identifier not in selected_ids),
        key=lambda record: (getattr(record, score_field), str(record.identifier)),
        reverse=True,
    )
    selected.extend(remaining[: quota - len(selected)])
    if len(selected) != quota:
        raise ValueError(f"failed to select exact quota={quota}")
    return selected


def proportional_quotas(capacities: Dict[str, int], *, total: int) -> Dict[str, int]:
    capacity_total = sum(capacities.values())
    if total < 0 or total > capacity_total:
        raise ValueError("invalid proportional quota total")
    raw = {key: total * value / capacity_total for key, value in capacities.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        capacities,
        key=lambda key: (raw[key] - quotas[key], capacities[key], natural_key(key)),
        reverse=True,
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def select_splits(
    samples: Sequence[Sample], *, seed: int
) -> Tuple[List[Sample], List[Sample], List[Sample], List[Sample], Dict[str, int]]:
    by_scene: Dict[str, List[Sample]] = defaultdict(list)
    by_session: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_scene[sample.scene].append(sample)
        by_session[sample.session].append(sample)
    if set(by_scene) != set(DEV_TARGETS):
        raise ValueError(
            f"Scene mismatch: data={sorted(by_scene)} targets={sorted(DEV_TARGETS)}"
        )

    dev_sessions: Set[str] = set()
    for scene in sorted(by_scene, key=natural_key):
        sizes = {
            session: len(records)
            for session, records in by_session.items()
            if records[0].scene == scene
        }
        selected = choose_sessions_exact(
            sizes, target=DEV_TARGETS[scene], seed=seed
        )
        dev_sessions.update(selected)

    dev = [sample for sample in samples if sample.session in dev_sessions]
    remaining_by_scene: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.session not in dev_sessions:
            remaining_by_scene[sample.scene].append(sample)

    train: List[Sample] = []
    train_quotas: Dict[str, int] = {}
    for scene in sorted(remaining_by_scene, key=natural_key):
        records = remaining_by_scene[scene]
        if scene in BIG_SCENES:
            selected = select_ranked_frames(
                records,
                quota=BIG_TRAIN_QUOTA,
                frame_target=BIG_TRAIN_FRAME_TARGET,
                session_person_cap=BIG_SESSION_PERSON_CAP,
            )
        else:
            selected = list(records)
        train.extend(selected)
        train_quotas[scene] = len(selected)

    train_ids = {sample.identifier for sample in train}
    dev_ids = {sample.identifier for sample in dev}
    reserve = [
        sample
        for sample in samples
        if sample.identifier not in train_ids and sample.identifier not in dev_ids
    ]
    if len(dev) != 2000 or len(train) != 30000:
        raise ValueError(f"Unexpected split counts: train={len(train)} dev={len(dev)}")

    scene_train_counts = Counter(sample.scene for sample in train)
    rl_quotas = proportional_quotas(dict(scene_train_counts), total=RL_TOTAL)
    rl: List[Sample] = []
    for scene in sorted(scene_train_counts, key=natural_key):
        scene_records = [sample for sample in train if sample.scene == scene]
        quota = rl_quotas[scene]
        frame_target = min(
            len({sample.relative_json for sample in scene_records}),
            max(1, math.ceil(quota / 1.8)),
        )
        rl.extend(
            select_ranked_frames(
                scene_records,
                quota=quota,
                frame_target=frame_target,
                session_person_cap=max(2, math.ceil(quota * 0.10)),
                score_field="rl_score",
            )
        )
    if len(rl) != RL_TOTAL or not {sample.identifier for sample in rl} <= train_ids:
        raise ValueError("RL must be an exact 6k subset of train")
    return train, dev, rl, reserve, train_quotas


def materialize_split(
    records: Sequence[Sample], *, source_root: Path, output_root: Path
) -> Tuple[int, int]:
    grouped: Dict[Path, List[Sample]] = defaultdict(list)
    for record in records:
        grouped[record.relative_json].append(record)
    for relative in sorted(grouped, key=natural_key):
        source = source_root / relative
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        positions = {record.crop_position for record in grouped[relative]}
        payload["crops"] = [
            crop
            for position, crop in enumerate(payload["crops"])
            if position in positions
        ]
        payload["person_crop_count"] = len(payload["crops"])
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.chmod(destination, 0o644)
    return len(grouped), len(records)


def copy_test_split(test_root: Path, output_root: Path) -> Tuple[int, int]:
    frames = crops = 0
    for source in iter_json_paths(test_root):
        relative = source.relative_to(test_root)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        crops += len(payload.get("crops", []))
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o644)
        frames += 1
    return frames, crops


def split_statistics(records: Sequence[Sample]) -> Dict[str, object]:
    scenes = Counter(sample.scene for sample in records)
    scales = Counter(sample.scale for sample in records)
    frames = len({sample.relative_json for sample in records})
    sessions = len({sample.session for sample in records})
    ideal_length = sum(18 <= len(sample.caption.split()) <= 24 for sample in records)
    return {
        "people": len(records),
        "frames": frames,
        "sessions": sessions,
        "scenes": dict(sorted(scenes.items(), key=lambda item: natural_key(item[0]))),
        "scales": dict(sorted(scales.items())),
        "length_18_24": ideal_length,
        "length_18_24_ratio": ideal_length / len(records) if records else 0.0,
    }


def write_manifest(path: Path, records: Sequence[Sample], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in sorted(
            records,
            key=lambda sample: (natural_key(sample.relative_json), sample.crop_position),
        ):
            row = {
                "sample_id": sample.identifier,
                "split": split,
                "scene": sample.scene,
                "session": sample.session,
                "source_json_relative_path": sample.relative_json.as_posix(),
                "crop_position": sample.crop_position,
                "crop_index": sample.crop_index,
                "caption": sample.caption,
                "scale": sample.scale,
                "selection_score": round(sample.score, 8),
                "rl_selection_score": round(sample.rl_score, 8),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def publish(staged: Path, output_root: Path, *, overwrite: bool) -> None:
    backup = output_root.with_name(f".{output_root.name}.backup")
    if output_root.exists():
        if any(output_root.iterdir()) and not overwrite:
            raise FileExistsError(f"Output is not empty: {output_root}")
        remove_path(backup)
        output_root.replace(backup)
    try:
        staged.replace(output_root)
    except Exception:
        if backup.exists():
            backup.replace(output_root)
        raise
    else:
        remove_path(backup)


def build_dataset(
    *,
    source_root: Path,
    test_root: Path,
    output_root: Path,
    seed: int,
    overwrite: bool,
) -> Dict[str, object]:
    source_root = source_root.expanduser().resolve(strict=True)
    test_root = test_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().absolute()
    if output_root == source_root or output_root == test_root:
        raise ValueError("output root cannot equal an input root")
    staged = output_root.with_name(f".{output_root.name}.tmp")
    remove_path(staged)
    staged.mkdir(parents=True, mode=0o755)
    try:
        print("[load] reading filtered candidate metadata", file=sys.stderr, flush=True)
        samples = load_samples(source_root)
        assign_scores(samples, seed=seed)
        print(f"[load] candidates={len(samples)}", file=sys.stderr, flush=True)
        train, dev, rl, reserve, train_quotas = select_splits(samples, seed=seed)

        materialized = {}
        for name, records in [
            ("train", train),
            ("dev", dev),
            ("rl", rl),
            ("reserve", reserve),
        ]:
            print(f"[write] {name}: people={len(records)}", file=sys.stderr, flush=True)
            frames, people = materialize_split(
                records, source_root=source_root, output_root=staged / name
            )
            materialized[name] = {"frames": frames, "people": people}
        test_frames, test_people = copy_test_split(test_root, staged / "test")
        materialized["test"] = {"frames": test_frames, "people": test_people}

        manifests = staged / "manifests"
        for name, records in [
            ("train", train),
            ("dev", dev),
            ("rl", rl),
            ("reserve", reserve),
        ]:
            write_manifest(manifests / f"{name}.jsonl", records, name)

        test_samples = load_samples(test_root)
        write_manifest(manifests / "test.jsonl", test_samples, "test")
        statistics = {
            "seed": seed,
            "source_root": str(source_root),
            "test_root": str(test_root),
            "output_root": str(output_root),
            "selection_novelty": "session rarity; CLIP is deferred to RL grounding",
            "train_scene_quotas": train_quotas,
            "train": split_statistics(train),
            "dev": split_statistics(dev),
            "test": split_statistics(test_samples),
            "rl": split_statistics(rl),
            "reserve": split_statistics(reserve),
            "materialized": materialized,
        }
        with (staged / "split_statistics.json").open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        publish(staged, output_root, overwrite=overwrite)
    finally:
        remove_path(staged)
    return statistics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    statistics = build_dataset(
        source_root=args.source_root,
        test_root=args.test_root,
        output_root=args.output_root,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
