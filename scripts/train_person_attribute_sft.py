#!/usr/bin/env python3
"""Full-parameter Florence2 person-attribute SFT with optional torchrun DDP."""

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import statistics
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from person_sft_data import (
    build_region_prompt,
    extract_task_text,
    iter_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pretrained" / "Florence-2-base"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "sft"
PERSON_TASK = "<REGION_TO_CATEGORY>"
REPLAY_TASK = "<REGION_TO_DESCRIPTION>"
EXPECTED_PERSON_TRAIN = 30000
EXPECTED_REPLAY = 1000
EXPECTED_DEV = 2000
DEV_ARTIFACT_SUBDIR = "dev"
SUBJECT_START = re.compile(
    r"^(?:A|An|The)\s+(?:adult|young|elderly|older|middle-aged|child|teenage|"
    r"male|female|man|woman|person|boy|girl)\b",
    re.IGNORECASE,
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def optimizer_steps(
    total_samples: int,
    world_size: int,
    per_device_batch: int,
    accumulation: int,
) -> int:
    _positive_integer(total_samples, "total_samples")
    _positive_integer(world_size, "world_size")
    _positive_integer(per_device_batch, "per_device_batch")
    _positive_integer(accumulation, "accumulation")
    per_rank_samples = math.ceil(total_samples / world_size)
    micro_batches = math.ceil(per_rank_samples / per_device_batch)
    return math.ceil(micro_batches / accumulation)


def scheduler_total_steps(
    steps_per_epoch: int, epochs: int, max_optimizer_steps: int = 0
) -> int:
    """Return the scheduler horizon, respecting an explicit capped run."""
    _positive_integer(steps_per_epoch, "steps_per_epoch")
    _positive_integer(epochs, "epochs")
    if not isinstance(max_optimizer_steps, int) or max_optimizer_steps < 0:
        raise ValueError(
            f"max_optimizer_steps must be a non-negative integer, got {max_optimizer_steps!r}"
        )
    planned_steps = steps_per_epoch * epochs
    return min(planned_steps, max_optimizer_steps) if max_optimizer_steps else planned_steps


def has_remaining_epochs(start_epoch: int, total_epochs: int) -> bool:
    if not isinstance(start_epoch, int) or start_epoch < 0:
        raise ValueError(f"start_epoch must be non-negative, got {start_epoch!r}")
    _positive_integer(total_epochs, "total_epochs")
    return start_epoch < total_epochs


def validate_world_size(world_size: int, formal: bool) -> None:
    _positive_integer(world_size, "world_size")
    if formal and world_size != 4:
        raise ValueError(
            f"Formal SFT requires WORLD_SIZE=4, got WORLD_SIZE={world_size}; "
            "launch with torchrun --nproc_per_node=4"
        )


def validate_global_batch(
    world_size: int,
    per_device_batch_size: int,
    accumulation_steps: int,
    formal: bool,
    expected_global_batch: int = 64,
) -> None:
    _positive_integer(world_size, "world_size")
    _positive_integer(per_device_batch_size, "per_device_batch_size")
    _positive_integer(accumulation_steps, "accumulation_steps")
    _positive_integer(expected_global_batch, "expected_global_batch")
    effective_batch = world_size * per_device_batch_size * accumulation_steps
    if formal and effective_batch != expected_global_batch:
        raise ValueError(
            f"Formal SFT requires global batch {expected_global_batch}, got {effective_batch}; "
            "check world size, per-device batch, and accumulation steps"
        )


def cosine_with_floor_multiplier(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return a linear-warmup/cosine multiplier with a non-zero LR floor."""
    _positive_integer(total_steps, "total_steps")
    if not isinstance(current_step, int) or current_step < 0:
        raise ValueError(f"current_step must be a non-negative integer, got {current_step!r}")
    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps!r}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio!r}")
    if warmup_steps >= total_steps:
        return 1.0 if current_step >= warmup_steps else current_step / max(1, warmup_steps)
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    if current_step >= total_steps:
        return min_lr_ratio
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def accumulation_group_size(
    micro_batch_index: int, total_micro_batches: int, accumulation: int
) -> int:
    if not isinstance(micro_batch_index, int) or not 0 <= micro_batch_index < total_micro_batches:
        raise ValueError(
            f"micro_batch_index must be in [0, {total_micro_batches}), got {micro_batch_index}"
        )
    _positive_integer(total_micro_batches, "total_micro_batches")
    _positive_integer(accumulation, "accumulation")
    group_start = (micro_batch_index // accumulation) * accumulation
    return min(accumulation, total_micro_batches - group_start)


def rng_state_for_rank(
    checkpoint_state: Mapping[str, Any], rank: int, world_size: int
) -> Dict[str, Any]:
    states = checkpoint_state.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != world_size:
        actual = len(states) if isinstance(states, list) else 0
        raise ValueError(
            f"RNG state count is {actual}, expected world_size={world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    state = states[rank]
    if not isinstance(state, dict):
        raise ValueError(f"Invalid RNG state for rank {rank}")
    return state


def parameter_role(name: str) -> str:
    normalized = name.removeprefix("module.") if hasattr(str, "removeprefix") else (
        name[7:] if name.startswith("module.") else name
    )
    if normalized.startswith("vision_tower."):
        return "vision"
    projection_prefixes = (
        "image_projection",
        "image_proj_norm",
        "image_pos_embed",
        "visual_temporal_embed",
    )
    if normalized.startswith(projection_prefixes):
        return "projection"
    return "language"


def is_no_decay_parameter(name: str) -> bool:
    lowered = name.lower()
    if lowered.endswith(".bias") or lowered == "bias":
        return True
    return any(
        marker in lowered
        for marker in ("layer_norm", "layernorm", ".norm.", "_norm.", "norm.weight")
    )


def mix_training_rows(
    person_rows: Iterable[Dict[str, Any]],
    replay_rows: Iterable[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in person_rows]
    rows.extend(dict(row) for row in replay_rows)
    rows.sort(key=lambda row: row["sample_id"])
    random.Random(seed).shuffle(rows)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Training data contains duplicate sample_id values")
    return rows


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def _single_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[-1] not in ".?!":
        return False
    return len(re.findall(r"[.?!]+", stripped)) == 1


def caption_metrics(captions: Sequence[str]) -> Dict[str, Any]:
    if not captions:
        return {
            "samples": 0,
            "empty_outputs": 0,
            "average_words": 0.0,
            "p95_words": 0,
            "length_18_24_ratio": 0.0,
            "single_sentence_ratio": 0.0,
            "subject_start_ratio": 0.0,
        }
    normalized = [caption.strip() if isinstance(caption, str) else "" for caption in captions]
    lengths = [_word_count(caption) for caption in normalized]
    sorted_lengths = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
    denominator = len(normalized)
    return {
        "samples": denominator,
        "empty_outputs": sum(not caption for caption in normalized),
        "average_words": statistics.fmean(lengths),
        "p95_words": sorted_lengths[p95_index],
        "length_18_24_ratio": sum(18 <= length <= 24 for length in lengths)
        / denominator,
        "single_sentence_ratio": sum(_single_sentence(caption) for caption in normalized)
        / denominator,
        "subject_start_ratio": sum(bool(SUBJECT_START.match(caption)) for caption in normalized)
        / denominator,
    }


def dev_artifact_dir(run_dir: Path) -> Path:
    return Path(run_dir) / DEV_ARTIFACT_SUBDIR


def dev_metrics_path(run_dir: Path, global_step: int) -> Path:
    return dev_artifact_dir(run_dir) / f"dev_metrics_step_{global_step}.json"


def dev_predictions_path(run_dir: Path, global_step: int) -> Path:
    return dev_artifact_dir(run_dir) / f"dev_predictions_step_{global_step}.jsonl"


def _validate_training_row(row: Dict[str, Any], expected_task: str, path: Path) -> None:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"Invalid sample_id in {path}: {sample_id!r}")
    if row.get("task") != expected_task:
        raise ValueError(
            f"Unexpected task for {sample_id}: {row.get('task')!r} != {expected_task!r}"
        )
    expected_prompt = build_region_prompt(expected_task, row.get("bbox_loc_0_999"))
    if row.get("prompt") != expected_prompt:
        raise ValueError(f"Prompt mismatch for {sample_id} in {path}")
    if not isinstance(row.get("label"), str) or not row["label"].strip():
        raise ValueError(f"Empty label for {sample_id} in {path}")
    if not isinstance(row.get("image"), str) or not row["image"]:
        raise ValueError(f"Invalid image path for {sample_id} in {path}")


class IndexedJsonlRows:
    """Random-access JSONL rows backed by byte offsets, not an in-memory row list."""

    def __init__(self, sources: Sequence[Sequence[Any]]):
        self.locations: List[Tuple[Path, int, str]] = []
        self.source_counts: Dict[Path, int] = {}
        seen_ids = set()
        for source in sources:
            if len(source) not in (2, 3):
                raise ValueError(
                    "Each JSONL source must be (path, expected_task[, limit])"
                )
            path = Path(source[0]).resolve(strict=True)
            expected_task = source[1]
            limit = source[2] if len(source) == 3 else None
            if limit is not None:
                _positive_integer(limit, f"row limit for {path}")
            accepted = 0
            with path.open("rb") as input_file:
                while True:
                    offset = input_file.tell()
                    line = input_file.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(f"Invalid JSONL row at {path}:{offset}") from error
                    _validate_training_row(row, expected_task, path)
                    sample_id = row["sample_id"]
                    identity = (expected_task, sample_id)
                    if identity in seen_ids:
                        raise ValueError(f"Duplicate sample_id: {sample_id}")
                    seen_ids.add(identity)
                    self.locations.append((path, offset, expected_task))
                    accepted += 1
                    if limit is not None and accepted >= limit:
                        break
            self.source_counts[path] = accepted
        self._handles: Dict[Path, Any] = {}
        self._handle_pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self.locations)

    def _handle_for(self, path: Path):
        pid = os.getpid()
        if self._handle_pid != pid:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
            self._handle_pid = pid
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: Any):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError(f"JSONL row index must be int or slice, got {type(index)!r}")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        path, offset, expected_task = self.locations[index]
        handle = self._handle_for(path)
        handle.seek(offset)
        row = json.loads(handle.readline().decode("utf-8"))
        _validate_training_row(row, expected_task, path)
        return row

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def load_training_data(
    args: argparse.Namespace,
) -> Tuple[IndexedJsonlRows, IndexedJsonlRows, Dict[str, Any]]:
    person_limit = args.max_train_samples or None
    replay_limit = None
    dev_limit = args.max_dev_samples or None
    if person_limit and not args.person_only:
        replay_limit = max(1, round(person_limit * EXPECTED_REPLAY / EXPECTED_PERSON_TRAIN))
    train_sources: List[Sequence[Any]] = [(args.train_data, PERSON_TASK, person_limit)]
    if not args.person_only:
        if not args.replay_data.is_file():
            raise FileNotFoundError(
                f"Formal training requires the 1,000-row replay file: {args.replay_data}"
            )
        train_sources.append((args.replay_data, REPLAY_TASK, replay_limit))
    train_index = IndexedJsonlRows(train_sources)
    dev_index = IndexedJsonlRows([(args.dev_data, PERSON_TASK, dev_limit)])
    person_samples = train_index.source_counts[args.train_data.resolve()]
    replay_samples = (
        train_index.source_counts[args.replay_data.resolve()] if not args.person_only else 0
    )

    if not args.allow_nonstandard_counts:
        if person_samples != EXPECTED_PERSON_TRAIN:
            raise ValueError(
                f"Person train count is {person_samples}, expected {EXPECTED_PERSON_TRAIN}"
            )
        if len(dev_index) != EXPECTED_DEV:
            raise ValueError(f"Dev count is {len(dev_index)}, expected {EXPECTED_DEV}")
        if not args.person_only and replay_samples != EXPECTED_REPLAY:
            raise ValueError(
                f"Replay count is {replay_samples}, expected {EXPECTED_REPLAY}"
            )

    metadata = {
        "person_samples": person_samples,
        "replay_samples": replay_samples,
        "train_samples": len(train_index),
        "dev_samples": len(dev_index),
        "person_only": args.person_only,
        "data_loading": {
            "format": "prepared_jsonl_byte_offsets",
            "rows_materialized": False,
            "shuffle": "DistributedSampler per epoch",
        },
        "files": {
            "train": {
                "path": str(args.train_data.resolve()),
                "sha256": sha256_file(args.train_data),
            },
            "dev": {
                "path": str(args.dev_data.resolve()),
                "sha256": sha256_file(args.dev_data),
            },
        },
    }
    if replay_samples:
        metadata["files"]["replay"] = {
            "path": str(args.replay_data.resolve()),
            "sha256": sha256_file(args.replay_data),
        }
    return train_index, dev_index, metadata


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


def load_training_runtime() -> Dict[str, Any]:
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as functional
        from PIL import Image
        from safetensors.torch import load_file as load_safetensors_file
        from torch.nn.parallel import DistributedDataParallel
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        from torch.utils.data import DataLoader, Dataset
        from torch.utils.data.distributed import DistributedSampler
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            dynamic_module_utils,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Training requires torch, transformers, safetensors, Pillow and tqdm"
        ) from error
    return locals()


def _resolve_precision(torch: Any, precision: str, device: Any) -> Dict[str, Any]:
    if device.type != "cuda" or precision == "fp32":
        return {
            "name": "fp32",
            "model_dtype": torch.float32,
            "autocast_dtype": None,
            "use_scaler": False,
        }
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but the selected GPU does not support it")
        return {
            "name": "bf16_amp",
            "model_dtype": torch.float32,
            "autocast_dtype": torch.bfloat16,
            "use_scaler": False,
        }
    if precision == "fp16":
        return {
            "name": "fp16_amp",
            "model_dtype": torch.float32,
            "autocast_dtype": torch.float16,
            "use_scaler": True,
        }
    raise ValueError(f"Unsupported precision: {precision}")


def _load_model_and_processor(runtime: Dict[str, Any], model_path: Path, dtype: Any):
    AutoConfig = runtime["AutoConfig"]
    AutoModelForCausalLM = runtime["AutoModelForCausalLM"]
    AutoProcessor = runtime["AutoProcessor"]
    dynamic_module_utils = runtime["dynamic_module_utils"]
    load_safetensors_file = runtime["load_safetensors_file"]
    patch_florence_dynamic_import_check(dynamic_module_utils)
    model_path = Path(model_path).resolve(strict=True)
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
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")
    state_dict = load_safetensors_file(str(safetensors_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {list(unexpected)}")
    model.tie_weights()
    if missing:
        logging.warning("Missing checkpoint keys after tied-weight load: %s", list(missing))
    model.to(dtype=dtype)
    return model, processor


def _build_optimizer_groups(
    model: Any,
    learning_rates: Mapping[str, float],
    weight_decay: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    buckets: Dict[Tuple[str, bool], List[Any]] = {}
    summaries: Dict[str, Dict[str, int]] = {
        role: {"tensors": 0, "parameters": 0} for role in learning_rates
    }
    total_trainable = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        role = parameter_role(name)
        no_decay = is_no_decay_parameter(name)
        buckets.setdefault((role, no_decay), []).append(parameter)
        summaries[role]["tensors"] += 1
        summaries[role]["parameters"] += parameter.numel()
        total_trainable += parameter.numel()
    if total_trainable != sum(value["parameters"] for value in summaries.values()):
        raise AssertionError("A trainable parameter was not assigned to exactly one role")
    groups = []
    for (role, no_decay), parameters in sorted(buckets.items()):
        groups.append(
            {
                "params": parameters,
                "lr": learning_rates[role],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": f"{role}_{'no_decay' if no_decay else 'decay'}",
            }
        )
    return groups, summaries


def _make_dataset(runtime: Dict[str, Any], rows: Sequence[Dict[str, Any]]):
    Dataset = runtime["Dataset"]
    Image = runtime["Image"]

    class FlorenceRows(Dataset):
        def __init__(self, values):
            self.rows = values

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["image"]) as source_image:
                image = source_image.convert("RGB")
            return row, image

    return FlorenceRows(rows)


def _make_collator(processor: Any, torch: Any, prompt_max_length: int, target_max_length: int):
    def collate(batch):
        rows, images = zip(*batch)
        inputs = processor(
            text=[row["prompt"] for row in rows],
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
        labels = processor.tokenizer(
            text=[row["label"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=target_max_length,
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return list(rows), dict(inputs), labels

    return collate


def _cross_entropy(functional: Any, logits: Any, labels: Any, smoothing: float, reduction: str):
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=smoothing,
        reduction=reduction,
    )


def _move_batch(inputs: Dict[str, Any], labels: Any, device: Any):
    moved = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    return moved, labels.to(device, non_blocking=True)


def _autocast_context(torch: Any, device: Any, autocast_dtype: Any):
    if device.type == "cuda" and autocast_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def _evaluate_loss(
    runtime: Dict[str, Any],
    model: Any,
    loader: Any,
    device: Any,
    autocast_dtype: Any,
    smoothing: float,
    distributed: bool,
) -> float:
    torch = runtime["torch"]
    dist = runtime["dist"]
    functional = runtime["functional"]
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)
    with torch.inference_mode():
        for _, inputs, labels in loader:
            inputs, labels = _move_batch(inputs, labels, device)
            with _autocast_context(torch, device, autocast_dtype):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    labels=labels,
                )
                batch_loss = _cross_entropy(
                    functional, outputs.logits, labels, smoothing, "sum"
                )
            loss_sum += batch_loss.double()
            token_count += (labels != -100).sum().double()
    if distributed:
        dist.all_reduce(loss_sum)
        dist.all_reduce(token_count)
    model.train()
    return (loss_sum / token_count.clamp_min(1)).item()


def _generate_dev_predictions(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    rows: Sequence[Dict[str, Any]],
    device: Any,
    autocast_dtype: Any,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    torch = runtime["torch"]
    Image = runtime["Image"]
    model.eval()
    predictions = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image"]) as source_image:
                image = source_image.convert("RGB")
                image_size = image.size
                inputs = processor(text=row["prompt"], images=image, return_tensors="pt")
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }
            with _autocast_context(torch, device, autocast_dtype):
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                )
            raw_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            post_processed = processor.post_process_generation(
                raw_text, task=PERSON_TASK, image_size=image_size
            )
            prediction = extract_task_text(post_processed, PERSON_TASK)
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "scene": row["scene"],
                    "scale": row.get("scale"),
                    "label": row["label"],
                    "prediction": prediction.strip(),
                    "attributes": row.get("attributes", {}),
                }
            )
    model.train()
    return predictions, caption_metrics([row["prediction"] for row in predictions])


def _capture_rng_state(torch: Any) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _gather_rng_states(
    runtime: Dict[str, Any], distributed: bool, rank: int, world_size: int
) -> List[Dict[str, Any]]:
    torch = runtime["torch"]
    local_state = _capture_rng_state(torch)
    if not distributed:
        return [local_state]
    dist = runtime["dist"]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered if rank == 0 else []


def _save_checkpoint(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    checkpoint_dir: Path,
    state: Dict[str, Any],
) -> None:
    torch = runtime["torch"]
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    if "rng_by_rank" not in state:
        raise ValueError("Checkpoint state is missing rng_by_rank")
    training_state = dict(state)
    training_state.update(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
    )
    torch.save(training_state, checkpoint_dir / "training_state.pt")
    write_json_atomic(checkpoint_dir / "run_state.json", state["invariants"])


def _load_resume_state(
    runtime: Dict[str, Any],
    checkpoint_dir: Path,
    invariants: Dict[str, Any],
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    torch = runtime["torch"]
    saved_invariants = json.loads(
        (checkpoint_dir / "run_state.json").read_text(encoding="utf-8")
    )
    if saved_invariants != invariants:
        raise ValueError("Resume checkpoint invariants do not match the current run")
    state = torch.load(
        checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    _restore_rng_state(torch, rng_state_for_rank(state, rank, world_size))
    return state


def _setup_logging(run_dir: Path, rank: int) -> None:
    handlers: List[logging.Handler] = []
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        ]
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers or [logging.NullHandler()],
        force=True,
    )


def train(args: argparse.Namespace) -> None:
    train_rows, dev_rows, data_metadata = load_training_data(args)
    if args.validate_data_only:
        print(json.dumps(data_metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return

    runtime = load_training_runtime()
    torch = runtime["torch"]
    dist = runtime["dist"]
    DataLoader = runtime["DataLoader"]
    DistributedSampler = runtime["DistributedSampler"]
    DistributedDataParallel = runtime["DistributedDataParallel"]
    AdamW = runtime["AdamW"]
    LambdaLR = runtime["LambdaLR"]
    functional = runtime["functional"]

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    validate_world_size(
        world_size, formal=not args.allow_nonstandard_counts
    )
    validate_global_batch(
        world_size,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        formal=not args.allow_nonstandard_counts,
        expected_global_batch=args.formal_global_batch_size,
    )
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Florence training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    run_name = args.run_name or datetime.now().strftime("person_sft_%Y%m%d_%H%M%S")
    if distributed:
        names = [run_name if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        run_name = names[0]
    run_dir = args.output_root / run_name
    _setup_logging(run_dir, rank)

    try:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
        precision = _resolve_precision(torch, args.precision, device)
        model_source = args.resume_from if args.resume_from else args.model_path
        model, processor = _load_model_and_processor(
            runtime, model_source, precision["model_dtype"]
        )
        model.to(device)
        learning_rates = {
            "vision": args.vision_lr,
            "projection": args.projection_lr,
            "language": args.language_lr,
        }
        optimizer_groups, group_summary = _build_optimizer_groups(
            model, learning_rates, args.weight_decay
        )
        if rank == 0:
            logging.info("parameter groups: %s", group_summary)
        optimizer = AdamW(
            optimizer_groups,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

        train_dataset = _make_dataset(runtime, train_rows)
        dev_dataset = _make_dataset(runtime, dev_rows)
        collator = _make_collator(
            processor, torch, args.prompt_max_length, args.target_max_length
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        dev_sampler = (
            DistributedSampler(
                dev_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if distributed
            else None
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            generator=generator,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.per_device_batch_size,
            sampler=dev_sampler,
            shuffle=False,
            num_workers=args.dev_num_workers,
            prefetch_factor=(
                args.prefetch_factor if args.dev_num_workers > 0 else None
            ),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
            persistent_workers=args.dev_num_workers > 0,
            drop_last=False,
        )
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_steps = scheduler_total_steps(
            steps_per_epoch, args.epochs, args.max_optimizer_steps
        )
        warmup_steps = math.ceil(total_steps * args.warmup_ratio)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: cosine_with_floor_multiplier(
                current_step,
                warmup_steps,
                total_steps,
                args.min_lr_ratio,
            ),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=precision["use_scaler"]
        )
        invariants = {
            "model_source": str(args.model_path.resolve()),
            "data": data_metadata,
            "world_size": world_size,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": (
                world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
            ),
            "formal_global_batch_size": args.formal_global_batch_size,
            "num_workers_per_rank": args.num_workers,
            "dev_num_workers_per_rank": args.dev_num_workers,
            "prefetch_factor": args.prefetch_factor,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "learning_rates": learning_rates,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "adam_eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "person_only": args.person_only,
        }
        if rank == 0:
            write_json_atomic(run_dir / "run_config.json", invariants)

        start_epoch = 0
        start_micro_batch = 0
        global_step = 0
        best_checkpoints: List[Tuple[float, str]] = []
        if args.resume_from:
            resume_state = _load_resume_state(
                runtime,
                args.resume_from,
                invariants,
                optimizer,
                scheduler,
                scaler,
                rank,
                world_size,
            )
            start_epoch = resume_state["epoch"]
            start_micro_batch = resume_state["next_micro_batch"]
            global_step = resume_state["global_step"]
            best_checkpoints = resume_state.get("best_checkpoints", [])
            if not has_remaining_epochs(start_epoch, args.epochs):
                logging.info(
                    "checkpoint already completed all %d configured epochs; nothing to resume",
                    args.epochs,
                )
                return

        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        unwrapped_model = model.module if distributed else model
        optimizer.zero_grad(set_to_none=True)
        stop_training = False
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for micro_batch_index, (_, inputs, labels) in enumerate(train_loader):
                if epoch == start_epoch and micro_batch_index < start_micro_batch:
                    continue
                inputs, labels = _move_batch(inputs, labels, device)
                group_size = accumulation_group_size(
                    micro_batch_index,
                    len(train_loader),
                    args.gradient_accumulation_steps,
                )
                is_group_end = (
                    (micro_batch_index + 1) % args.gradient_accumulation_steps == 0
                    or micro_batch_index + 1 == len(train_loader)
                )
                sync_context = (
                    nullcontext()
                    if is_group_end or not distributed
                    else model.no_sync()
                )
                with sync_context:
                    with _autocast_context(
                        torch, device, precision["autocast_dtype"]
                    ):
                        outputs = model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            pixel_values=inputs["pixel_values"],
                            labels=labels,
                        )
                        loss = _cross_entropy(
                            functional,
                            outputs.logits,
                            labels,
                            args.label_smoothing,
                            "mean",
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at optimizer step {global_step + 1}"
                        )
                    scaled_loss = loss / group_size
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                if not is_group_end:
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (
                    global_step == 1 or global_step % args.log_steps == 0
                ):
                    logging.info(
                        "epoch=%d step=%d/%d loss=%.6f grad_norm=%.4f lr=%s",
                        epoch + 1,
                        global_step,
                        total_steps,
                        float(loss.detach().cpu()),
                        float(grad_norm),
                        [group["lr"] for group in optimizer.param_groups],
                    )

                should_eval = (
                    args.eval_steps > 0 and global_step % args.eval_steps == 0
                )
                if should_eval:
                    dev_loss = _evaluate_loss(
                        runtime,
                        model,
                        dev_loader,
                        device,
                        precision["autocast_dtype"],
                        args.label_smoothing,
                        distributed,
                    )
                    if rank == 0:
                        dev_output_dir = dev_artifact_dir(run_dir)
                        dev_output_dir.mkdir(parents=True, exist_ok=True)
                        sample_rows = dev_rows[: args.eval_generation_samples]
                        predictions, metrics = _generate_dev_predictions(
                            runtime,
                            unwrapped_model,
                            processor,
                            sample_rows,
                            device,
                            precision["autocast_dtype"],
                            args.generation_max_new_tokens,
                        )
                        metrics["dev_loss"] = dev_loss
                        metrics["global_step"] = global_step
                        write_jsonl_atomic(
                            dev_predictions_path(run_dir, global_step), predictions
                        )
                        write_json_atomic(
                            dev_metrics_path(run_dir, global_step), metrics
                        )
                        logging.info("dev metrics: %s", metrics)
                    rng_by_rank = (
                        _gather_rng_states(runtime, distributed, rank, world_size)
                        if not args.no_save
                        else []
                    )
                    if rank == 0 and not args.no_save:
                        checkpoint_name = f"checkpoint-{global_step}"
                        checkpoint_dir = run_dir / checkpoint_name
                        candidates = best_checkpoints + [(dev_loss, checkpoint_name)]
                        candidates.sort()
                        kept = candidates[:3]
                        state = {
                            "epoch": epoch,
                            "next_micro_batch": micro_batch_index + 1,
                            "global_step": global_step,
                            "best_checkpoints": kept,
                            "rng_by_rank": rng_by_rank,
                            "invariants": invariants,
                        }
                        _save_checkpoint(
                            runtime,
                            unwrapped_model,
                            processor,
                            optimizer,
                            scheduler,
                            scaler,
                            checkpoint_dir,
                            state,
                        )
                        for _, removed_name in candidates[3:]:
                            removed_path = run_dir / removed_name
                            if removed_path.exists():
                                shutil.rmtree(removed_path)
                        best_checkpoints = kept
                    if distributed:
                        dist.barrier()

                if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                    stop_training = True
                    break
            start_micro_batch = 0
            if stop_training:
                break

        final_rng_by_rank = (
            _gather_rng_states(runtime, distributed, rank, world_size)
            if not args.no_save
            else []
        )
        if rank == 0 and not args.no_save:
            final_dir = run_dir / "final"
            state = {
                "epoch": min(args.epochs, epoch + 1),
                "next_micro_batch": 0,
                "global_step": global_step,
                "best_checkpoints": best_checkpoints,
                "rng_by_rank": final_rng_by_rank,
                "invariants": invariants,
            }
            _save_checkpoint(
                runtime,
                unwrapped_model,
                processor,
                optimizer,
                scheduler,
                scaler,
                final_dir,
                state,
            )
        if distributed:
            dist.barrier()
        if rank == 0:
            logging.info("training stopped at optimizer step %d", global_step)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-parameter Florence2 REGION_TO_CATEGORY person SFT."
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_PREPARED_DIR / "train.jsonl")
    parser.add_argument("--dev-data", type=Path, default=DEFAULT_PREPARED_DIR / "dev.jsonl")
    parser.add_argument(
        "--replay-data", type=Path, default=DEFAULT_PREPARED_DIR / "native_replay.jsonl"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--person-only", action="store_true")
    parser.add_argument("--allow-nonstandard-counts", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--formal-global-batch-size",
        type=int,
        default=64,
        help="Expected global batch for formal runs; V1=64 and V2/V3=32.",
    )
    parser.add_argument("--vision-lr", type=float, default=1e-7)
    parser.add_argument("--projection-lr", type=float, default=5e-7)
    parser.add_argument("--language-lr", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prompt-max-length", type=int, default=1024)
    parser.add_argument("--target-max-length", type=int, default=64)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-generation-samples", type=int, default=128)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--dev-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0 or args.dev_num_workers < 0:
        parser.error("DataLoader worker counts must be non-negative")
    for name in (
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "formal_global_batch_size",
        "prompt_max_length",
        "target_max_length",
        "generation_max_new_tokens",
        "prefetch_factor",
    ):
        _positive_integer(getattr(args, name), name)
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if not 0 <= args.min_lr_ratio <= 1:
        parser.error("--min-lr-ratio must be in [0, 1]")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        parser.error("Adam betas must be in [0, 1)")
    if args.adam_eps <= 0:
        parser.error("--adam-eps must be positive")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
