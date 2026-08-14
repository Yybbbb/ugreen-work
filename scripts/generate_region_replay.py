#!/usr/bin/env python3
"""Select reserve regions and generate Florence2 REGION_TO_DESCRIPTION replay."""

import argparse
import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from person_sft_data import (
    build_region_prompt,
    extract_task_text,
    iter_jsonl,
    sha256_file,
    validate_relative_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from prepare_person_sft_data import _load_frame_document, prepare_manifest_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "prepared"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pretrained" / "Florence-2-base"
DEFAULT_SEED = 20260720
REPLAY_TASK = "<REGION_TO_DESCRIPTION>"
DEFAULT_SCENE_QUOTAS = {
    "company_surveillance_mp4_adaptive": 500,
    "tradeshow": 500,
}
SCALE_ORDER = ("large", "medium", "small", "tiny", "unknown")
LOCATION_TOKEN = re.compile(r"<loc_(\d{1,3})>")


def _stable_tiebreak(seed: int, sample_id: str) -> int:
    value = f"{seed}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _quality(row: Mapping[str, Any]) -> Tuple[float, int, int]:
    score = row.get("selection_score", 0.0)
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    caption = row.get("caption") if isinstance(row.get("caption"), str) else ""
    words = len(caption.split())
    attribute_length_score = min(words, 32)
    crop_index = row.get("crop_index")
    stable_crop_index = -crop_index if isinstance(crop_index, int) else 0
    return numeric_score, attribute_length_score, stable_crop_index


def _best_per_frame(rows: Iterable[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        frame = row.get("source_json_relative_path")
        sample_id = row.get("sample_id")
        if not isinstance(frame, str) or not frame:
            raise ValueError(f"Replay candidate has no source frame: {row!r}")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Replay candidate has no sample_id: {row!r}")
        current = best.get(frame)
        priority = (_quality(row), _stable_tiebreak(seed, sample_id))
        if current is None:
            best[frame] = row
            continue
        current_priority = (
            _quality(current),
            _stable_tiebreak(seed, current["sample_id"]),
        )
        if priority > current_priority:
            best[frame] = row
    return list(best.values())


def _scale_balanced_top(
    rows: Sequence[Dict[str, Any]], quota: int, seed: int
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scale = row.get("scale")
        if scale not in SCALE_ORDER:
            scale = "unknown"
        buckets[scale].append(row)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda row: (
                _quality(row),
                _stable_tiebreak(seed, row["sample_id"]),
            ),
            reverse=True,
        )

    selected = []
    positions = Counter()
    while len(selected) < quota:
        made_progress = False
        for scale in SCALE_ORDER:
            position = positions[scale]
            bucket = buckets.get(scale, [])
            if position >= len(bucket):
                continue
            selected.append(bucket[position])
            positions[scale] += 1
            made_progress = True
            if len(selected) == quota:
                break
        if not made_progress:
            break
    return selected


def select_replay_candidates(
    rows: Iterable[Dict[str, Any]], scene_quotas: Mapping[str, int], seed: int
) -> List[Dict[str, Any]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = row.get("scene")
        if scene in scene_quotas:
            by_scene[scene].append(row)

    selected = []
    for scene in sorted(scene_quotas):
        quota = scene_quotas[scene]
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            raise ValueError(f"Replay quota for {scene} must be positive, got {quota!r}")
        unique_frames = _best_per_frame(by_scene.get(scene, []), seed)
        if len(unique_frames) < quota:
            raise ValueError(
                f"Scene {scene} has {len(unique_frames)} unique frames, needs {quota}"
            )
        scene_selected = _scale_balanced_top(unique_frames, quota, seed)
        if len(scene_selected) != quota:
            raise ValueError(f"Could not fulfill replay quota for {scene}")
        selected.extend(scene_selected)

    selected.sort(
        key=lambda row: (
            row["scene"],
            row["source_json_relative_path"],
            row["sample_id"],
        )
    )
    frames = [row["source_json_relative_path"] for row in selected]
    if len(frames) != len(set(frames)):
        raise ValueError("Replay selection contains repeated source frames")
    return selected


def hydrate_replay_selection(
    selected: Iterable[Dict[str, Any]], reserve_root: Path
) -> Iterable[Dict[str, Any]]:
    reserve_root = Path(reserve_root)
    for manifest_row in selected:
        source_relative = validate_relative_path(
            manifest_row.get("source_json_relative_path")
        )
        source_path = reserve_root / source_relative
        _, image, crops = _load_frame_document(source_path)
        row = prepare_manifest_row(
            manifest_row, source_relative, image, crops, split="replay"
        )
        row["schema_version"] = "florence_region_replay_selection_v1"
        row["source_person_caption"] = row.pop("label")
        row["task"] = REPLAY_TASK
        row["prompt"] = build_region_prompt(REPLAY_TASK, row["bbox_loc_0_999"])
        row["selection_score"] = manifest_row.get("selection_score")
        row["rl_selection_score"] = manifest_row.get("rl_selection_score")
        yield row


def validate_selection(
    selection_path: Path, scene_quotas: Mapping[str, int]
) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(selection_path))
    expected = sum(scene_quotas.values())
    if len(rows) != expected:
        raise ValueError(f"Replay selection count is {len(rows)}, expected {expected}")
    counts = Counter(row.get("scene") for row in rows)
    if dict(counts) != dict(scene_quotas):
        raise ValueError(f"Replay scene counts mismatch: {dict(counts)} != {dict(scene_quotas)}")
    frames = [row.get("source_json") for row in rows]
    if any(not isinstance(frame, str) or not frame for frame in frames):
        raise ValueError("Replay selection contains an invalid source_json")
    if len(frames) != len(set(frames)):
        raise ValueError("Replay selection contains repeated frames")
    for row in rows:
        expected_prompt = build_region_prompt(REPLAY_TASK, row.get("bbox_loc_0_999"))
        if row.get("prompt") != expected_prompt or row.get("task") != REPLAY_TASK:
            raise ValueError(f"Invalid replay prompt for {row.get('sample_id')}")
    return rows


def ensure_selection(
    data_root: Path,
    output_dir: Path,
    scene_quotas: Mapping[str, int],
    seed: int,
    overwrite: bool,
) -> Tuple[Path, List[Dict[str, Any]]]:
    selection_path = output_dir / "native_replay_selection.jsonl"
    progress_path = output_dir / "native_replay.progress.jsonl"
    if selection_path.exists() and not overwrite:
        return selection_path, validate_selection(selection_path, scene_quotas)
    if overwrite and progress_path.exists():
        raise FileExistsError(
            f"Refusing to replace selection while replay progress exists: {progress_path}"
        )
    manifest_path = data_root / "manifests" / "reserve.jsonl"
    selected = select_replay_candidates(iter_jsonl(manifest_path), scene_quotas, seed)
    count = write_jsonl_atomic(
        selection_path,
        hydrate_replay_selection(selected, data_root / "reserve"),
    )
    expected = sum(scene_quotas.values())
    if count != expected:
        raise ValueError(f"Wrote {count} replay selections, expected {expected}")
    return selection_path, validate_selection(selection_path, scene_quotas)


def load_completed_progress(progress_path: Path) -> Dict[str, Dict[str, Any]]:
    if not progress_path.exists():
        return {}
    completed = {}
    for row in iter_jsonl(progress_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Invalid sample_id in replay progress: {row!r}")
        if sample_id in completed:
            raise ValueError(f"Duplicate sample_id in replay progress: {sample_id}")
        validate_replay_label(row.get("label"), row.get("bbox_loc_0_999"))
        completed[sample_id] = row
    return completed


def validate_replay_label(
    label: Any, expected_locations: Optional[Sequence[int]] = None
) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Empty replay label")
    stripped = label.strip()
    matches = LOCATION_TOKEN.findall(stripped)
    if stripped.count("<loc_") != len(matches):
        raise ValueError(f"Malformed location token in replay label: {stripped!r}")
    locations = [int(value) for value in matches]
    if any(not 0 <= value <= 999 for value in locations):
        raise ValueError(f"Out-of-range location token in replay label: {stripped!r}")
    if locations and expected_locations is not None:
        expected = list(expected_locations)
        if locations != expected:
            raise ValueError(
                f"Replay location tokens do not match input bbox: {locations} != {expected}"
            )
    return stripped


def append_progress(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        output_file.flush()
        os.fsync(output_file.fileno())


def generate_missing(
    selection: Sequence[Dict[str, Any]],
    completed: Mapping[str, Dict[str, Any]],
    generate_one: Callable[[Dict[str, Any]], Tuple[str, str]],
    append_one: Callable[[Dict[str, Any]], None],
    max_new_samples: int = 0,
) -> int:
    generated = 0
    for row in selection:
        sample_id = row["sample_id"]
        if sample_id in completed:
            continue
        label, raw_text = generate_one(row)
        try:
            label = validate_replay_label(label, row.get("bbox_loc_0_999"))
        except ValueError as error:
            raise ValueError(f"Invalid replay label for {sample_id}: {error}") from error
        progress_row = dict(row)
        progress_row["schema_version"] = "florence_native_replay_progress_v1"
        progress_row["label"] = label
        progress_row["raw_generation"] = raw_text
        append_one(progress_row)
        generated += 1
        if max_new_samples and generated >= max_new_samples:
            break
    return generated


def finalize_replay(
    selection: Sequence[Dict[str, Any]],
    completed: Mapping[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    selection_ids = [row["sample_id"] for row in selection]
    if set(completed) != set(selection_ids):
        missing = len(set(selection_ids) - set(completed))
        extra = len(set(completed) - set(selection_ids))
        raise ValueError(f"Replay progress is incomplete: missing={missing} extra={extra}")

    def final_rows():
        for sample_id in selection_ids:
            row = dict(completed[sample_id])
            row["schema_version"] = "florence_native_replay_v1"
            yield row

    write_jsonl_atomic(output_path, final_rows())


def patch_florence_dynamic_import_check(dynamic_module_utils: Any) -> None:
    original_get_imports = dynamic_module_utils.get_imports
    if getattr(original_get_imports, "_florence_optional_import_patch", False):
        return

    def patched_get_imports(filename):
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    patched_get_imports._florence_optional_import_patch = True
    dynamic_module_utils.get_imports = patched_get_imports


def load_florence_generator(model_path: Path, device_name: Optional[str], max_new_tokens: int):
    try:
        import torch
        from PIL import Image
        from safetensors.torch import load_file as load_safetensors_file
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            dynamic_module_utils,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Replay generation requires torch, transformers, safetensors, Pillow and tqdm"
        ) from error

    patch_florence_dynamic_import_check(dynamic_module_utils)
    model_path = Path(model_path).resolve(strict=True)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager"
    )
    state_dict = load_safetensors_file(
        str(model_path / "model.safetensors"), device="cpu"
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Florence checkpoint keys: {unexpected}")
    logging.info("Loaded Florence checkpoint; missing keys: %s", list(missing))
    model.tie_weights()
    model.to(device=device, dtype=dtype)
    model.eval()

    def generate_one(row: Dict[str, Any]) -> Tuple[str, str]:
        with Image.open(row["image"]) as source_image:
            image = source_image.convert("RGB")
            image_size = image.size
            inputs = processor(
                text=row["prompt"], images=image, return_tensors="pt"
            )
        inputs = {
            key: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device=device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                early_stopping=False,
            )
        raw_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        post_processed = processor.post_process_generation(
            raw_text, task=REPLAY_TASK, image_size=image_size
        )
        return extract_task_text(post_processed, REPLAY_TASK), raw_text

    return generate_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1,000 REGION_TO_DESCRIPTION native replay samples."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-scene", type=int, default=500)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument("--max-new-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_quotas = {scene: args.per_scene for scene in DEFAULT_SCENE_QUOTAS}
    selection_path, selection = ensure_selection(
        args.data_root.resolve(strict=True),
        args.output_dir.resolve(),
        scene_quotas,
        args.seed,
        args.overwrite_selection,
    )
    print(
        json.dumps(
            {
                "selection": str(selection_path),
                "samples": len(selection),
                "scene_counts": dict(Counter(row["scene"] for row in selection)),
                "sha256": sha256_file(selection_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.selection_only or args.validate_existing:
        return

    progress_path = args.output_dir / "native_replay.progress.jsonl"
    replay_path = args.output_dir / "native_replay.jsonl"
    completed = load_completed_progress(progress_path)
    generate_one = load_florence_generator(
        args.model_path, args.device, args.max_new_tokens
    )
    generated = generate_missing(
        selection,
        completed,
        generate_one,
        lambda row: append_progress(progress_path, row),
        max_new_samples=args.max_new_samples,
    )
    completed = load_completed_progress(progress_path)
    print(f"generated={generated} completed={len(completed)}/{len(selection)}")
    if len(completed) == len(selection):
        finalize_replay(selection, completed, replay_path)
        metadata = {
            "schema_version": "florence_native_replay_v1",
            "model_path": str(args.model_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "replay_sha256": sha256_file(replay_path),
            "samples": len(selection),
            "task": REPLAY_TASK,
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            },
        }
        write_json_atomic(args.output_dir / "native_replay_metadata.json", metadata)


if __name__ == "__main__":
    main()
