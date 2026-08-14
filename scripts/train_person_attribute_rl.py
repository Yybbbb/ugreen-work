#!/usr/bin/env python3
"""Full-parameter Florence2 person-attribute SCST (Self-Critical Sequence Training).

Reuses the SFT trainer's model loading, parameter grouping, DDP, scheduler and
checkpoint machinery (see train_person_attribute_sft.py). The SCST-specific path
samples a caption and a greedy baseline per sample, scores both with the Qwen
reward pipeline (rl_clients + rl_reward), and optimizes a mixed
`(1-alpha)*L_CE + alpha*L_SCST` loss with REINFORCE advantages. See
EXPERIMENT_PLAN.md section 10-11.
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import statistics
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from train_person_attribute_sft import (
    DEFAULT_PREPARED_DIR,
    PERSON_TASK,
    PROJECT_ROOT,
    IndexedJsonlRows,
    _autocast_context,
    _build_optimizer_groups,
    _cross_entropy,
    _evaluate_loss,
    _gather_rng_states,
    _load_resume_state,
    _load_model_and_processor,
    _make_collator,
    _make_dataset,
    _move_batch,
    _positive_integer,
    _resolve_precision,
    _save_checkpoint,
    _setup_logging,
    accumulation_group_size,
    caption_metrics,
    cosine_with_floor_multiplier,
    extract_task_text,
    has_remaining_epochs,
    load_training_runtime,
    optimizer_steps,
    scheduler_total_steps,
    dev_metrics_path,
    dev_predictions_path,
    validate_global_batch,
    validate_world_size,
)
from person_sft_data import sha256_file, write_json_atomic, write_jsonl_atomic
from rl_reward import (
    SCALAR_FIELDS,
    W_BACKGROUND,
    W_F1,
    W_FABRICATION,
    W_LENGTH,
    W_STRUCTURE,
    background_penalty,
    classify_sample,
    compute_reward,
    lexical_similarity,
    normalize_extra,
    normalize_value,
    normalize_value_set,
)
from rl_clients import extract_attributes, make_judge_fn, preflight_service
from rl_service_config import EXTRACTOR_SERVICE, JUDGE_SERVICE, service_metadata


DEFAULT_MODEL_PATH = PROJECT_ROOT / "pretrained" / "Florence-2-base"
DEFAULT_SFT_FINAL = PROJECT_ROOT / "artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final"
DEFAULT_RL_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "rl"

ALPHA_START = 0.10
ALPHA_END = 0.50
RECALL_THRESHOLD = 0.80
FORMAL_RL_SAMPLES = 5981


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, import-safe)
# --------------------------------------------------------------------------- #

def alpha_at_step(
    step: int,
    total_steps: int,
    start: float = ALPHA_START,
    end: float = ALPHA_END,
) -> float:
    """Linear SCST weight ramp from ALPHA_START to ALPHA_END over training."""
    if total_steps <= 1:
        return end
    frac = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + (end - start) * frac


def advantage(reward_sampled: float, reward_greedy: float) -> float:
    return reward_sampled - reward_greedy


def scst_loss(advantages: Sequence[float], logprobs: Sequence[float]) -> float:
    """REINFORCE loss L = -mean(advantage * logprob). Pure (list) version."""
    n = len(advantages)
    if n == 0:
        return 0.0
    total = sum(a * lp for a, lp in zip(advantages, logprobs))
    return -total / n


def mixed_loss(ce_loss: float, scst: float, alpha: float) -> float:
    return (1.0 - alpha) * ce_loss + alpha * scst


def rl_optimizer_steps(n_samples: int, world_size: int, per_device_batch: int,
                       accumulation: int, epochs: int) -> int:
    global_batch = max(world_size * per_device_batch * accumulation, 1)
    return math.ceil(n_samples * epochs / global_batch)


def _rl_data_metadata(train_rows: IndexedJsonlRows) -> Dict[str, Any]:
    return {
        "train_samples": len(train_rows),
        "task": PERSON_TASK,
        "schema": "florence_person_rl_v1",
    }


def _soft_f1(counts: Mapping[str, Any]) -> float:
    soft_tp = float(counts["soft_tp"])
    denominator = 2.0 * soft_tp + float(counts["soft_fp"]) + float(counts["soft_fn"])
    return 2.0 * soft_tp / denominator if denominator > 0 else 0.0


def summarize_dev_records(
    records: Sequence[Mapping[str, Any]], dev_loss: float
) -> Dict[str, Any]:
    captions = [str(record.get("caption", "")) for record in records]
    rewards = [float(record["reward"]) for record in records]
    soft_f1_values = [_soft_f1(record["counts"]) for record in records]
    fabrication_ratios = [
        float(record["counts"]["n_fab"])
        / max(int(record["counts"]["n_gen_assert"]), 1)
        for record in records
    ]
    metrics = caption_metrics(captions)
    sample_count = len(records)
    metrics.update(
        {
            "dev_loss": float(dev_loss),
            "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "mean_soft_f1": statistics.fmean(soft_f1_values) if soft_f1_values else 0.0,
            "mean_fabrication_ratio": (
                statistics.fmean(fabrication_ratios) if fabrication_ratios else 0.0
            ),
            "background_keyword_ratio": (
                statistics.fmean(background_penalty(caption) for caption in captions)
                if captions
                else 0.0
            ),
            "empty_ratio": metrics["empty_outputs"] / sample_count if sample_count else 0.0,
            "extractor_failures": sum(
                bool(record.get("extractor_failed")) for record in records
            ),
        }
    )
    return metrics


def select_top_checkpoints(
    candidates: Sequence[Mapping[str, Any]], limit: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (
            -float(item["mean_reward"]),
            float(item["dev_loss"]),
            int(item["global_step"]),
        ),
    )
    return ranked[:limit], ranked[limit:]


def build_dev_reward_records(
    rows: Sequence[Mapping[str, Any]],
    captions: Sequence[str],
    extracted_attributes: Sequence[Optional[Mapping[str, Any]]],
    similarity_fn,
) -> List[Dict[str, Any]]:
    if not (len(rows) == len(captions) == len(extracted_attributes)):
        raise ValueError("dev rows, captions, and extractions must have equal lengths")
    records: List[Dict[str, Any]] = []
    for row, caption, extracted in zip(rows, captions, extracted_attributes):
        extractor_failed = not isinstance(extracted, Mapping)
        generated = dict(extracted) if isinstance(extracted, Mapping) else {}
        gt_attributes = row.get("attributes", {})
        counts = classify_sample(gt_attributes, generated, similarity_fn)
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "scene": row.get("scene"),
                "scale": row.get("scale"),
                "label": row.get("label"),
                "caption": caption,
                "attributes": gt_attributes,
                "extracted_attributes": generated,
                "extractor_failed": extractor_failed,
                "counts": counts,
                "reward": compute_reward(caption, counts),
            }
        )
    return records


def _nested_attribute(attributes: Mapping[str, Any], path: str) -> Any:
    value: Any = attributes
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def similarity_requests(
    gt_attrs: Mapping[str, Any], gen_attrs: Mapping[str, Any]
) -> List[Tuple[str, str, str]]:
    requests: List[Tuple[str, str, str]] = []
    for field in SCALAR_FIELDS:
        gen_value = normalize_value(field, _nested_attribute(gen_attrs, field))
        for gt_value in normalize_value_set(field, _nested_attribute(gt_attrs, field)):
            if gen_value is not None:
                requests.append((field, gt_value, gen_value))
    for gt_value in normalize_extra(gt_attrs.get("extra")):
        for gen_value in normalize_extra(gen_attrs.get("extra")):
            requests.append(("extra", gt_value, gen_value))
    return requests


def prefetch_similarity_requests(
    gt_attributes: Sequence[Mapping[str, Any]],
    generated_attributes: Sequence[Mapping[str, Any]],
    similarity_fn,
) -> None:
    prefetch = getattr(similarity_fn, "prefetch", None)
    if prefetch is None:
        return
    requests = [
        request
        for gt_attrs, gen_attrs in zip(gt_attributes, generated_attributes)
        for request in similarity_requests(gt_attrs, gen_attrs)
    ]
    prefetch(requests)


def indexed_dev_shard(
    rows: Sequence[Mapping[str, Any]], rank: int, world_size: int
) -> List[Tuple[int, Mapping[str, Any]]]:
    if world_size <= 0 or not (0 <= rank < world_size):
        raise ValueError("invalid rank/world_size for dev reward sharding")
    return [(index, rows[index]) for index in range(rank, len(rows), world_size)]


def merge_dev_reward_payloads(
    payloads: Sequence[Mapping[str, Any]], expected_count: int
) -> List[Dict[str, Any]]:
    indexed_records: List[Tuple[int, Dict[str, Any]]] = []
    for payload in payloads:
        if not payload.get("ok"):
            raise RuntimeError(
                f"rank {payload.get('rank', '?')} dev reward failed: "
                f"{payload.get('error', 'unknown error')}"
            )
        indexed_records.extend(
            (int(index), dict(record)) for index, record in payload.get("items", [])
        )
    indices = [index for index, unused in indexed_records]
    if len(indices) != expected_count or sorted(indices) != list(range(expected_count)):
        raise RuntimeError(
            f"dev reward indices must cover 0..{expected_count - 1} exactly once; "
            f"got {sorted(indices)!r}"
        )
    return [record for unused, record in sorted(indexed_records)]


def extract_attributes_batched(
    captions: Sequence[str],
    *,
    base_url: str,
    model: str,
    batch_size: int,
    extract_fn=extract_attributes,
) -> List[Optional[Dict[str, Any]]]:
    _positive_integer(batch_size, "eval_reward_batch_size")
    extracted: List[Optional[Dict[str, Any]]] = []
    for start in range(0, len(captions), batch_size):
        extracted.extend(
            extract_fn(
                captions[start : start + batch_size],
                base_url=base_url,
                model=model,
            )
        )
    return extracted


def run_reward_service_preflight(
    args: argparse.Namespace,
    *,
    preflight_fn=preflight_service,
) -> Dict[str, Any]:
    """Probe every HTTP service required by the selected reward matcher.

    Errors are returned as data so rank 0 can broadcast a coherent outcome to
    every DDP worker before any large model allocation begins.
    """
    if args.skip_reward_service_preflight:
        return {"ok": True, "skipped": True, "checks": []}
    services = [
        ("extractor", args.extractor_base_url, args.extractor_model),
    ]
    if args.reward_matcher == "qwen":
        services.append(("judge", args.judge_base_url, args.judge_model))
    checks: List[Dict[str, Any]] = []
    try:
        for role, base_url, model in services:
            probe = preflight_fn(base_url, model)
            checks.append(
                {"role": role, "base_url": base_url, "model": model, "probe": probe}
            )
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "checks": checks,
            "error": f"{role} reward service preflight failed: {exc}",
        }
    return {"ok": True, "skipped": False, "checks": checks}


def resolve_run_dir(
    output_root: Path,
    run_name: Optional[str],
    resume_from: Optional[Path],
    generated_name: str,
) -> Path:
    if resume_from is None:
        return Path(output_root) / (run_name or generated_name)
    original_run_dir = Path(resume_from).resolve().parent
    if run_name is not None:
        requested_run_dir = (Path(output_root) / run_name).resolve()
        if requested_run_dir != original_run_dir:
            raise ValueError(
                f"--run-name must identify the original run directory {original_run_dir}"
            )
    return original_run_dir


def final_status(
    global_step: int, total_steps: int, max_optimizer_steps: int
) -> Dict[str, bool]:
    stopped_by_max_steps = bool(
        max_optimizer_steps and global_step >= max_optimizer_steps
    )
    return {
        "completed": global_step >= total_steps and not stopped_by_max_steps,
        "stopped_by_max_steps": stopped_by_max_steps,
    }


def load_rl_training_data(args: argparse.Namespace) -> Tuple[IndexedJsonlRows, IndexedJsonlRows, Dict[str, Any]]:
    train_sources = [(str(args.train_data), PERSON_TASK)]
    if args.max_train_samples:
        train_sources[0] = (str(args.train_data), PERSON_TASK, args.max_train_samples)
    train_rows = IndexedJsonlRows(train_sources)
    dev_sources = [(str(args.dev_data), PERSON_TASK)]
    if args.max_dev_samples:
        dev_sources[0] = (str(args.dev_data), PERSON_TASK, args.max_dev_samples)
    dev_rows = IndexedJsonlRows(dev_sources)
    if not args.allow_nonstandard_counts and len(train_rows) != FORMAL_RL_SAMPLES:
        raise ValueError(
            f"expected {FORMAL_RL_SAMPLES} RL train rows, got {len(train_rows)}"
        )
    return train_rows, dev_rows, _rl_data_metadata(train_rows)


# --------------------------------------------------------------------------- #
# SCST torch helpers
# --------------------------------------------------------------------------- #

def _generate_captions(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    images: Sequence[Any],
    device: Any,
    autocast_dtype: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    """Per-row generation; returns post-processed caption strings."""
    torch = runtime["torch"]
    captions: List[str] = []
    # no_grad (not inference_mode): the sampled captions are re-tokenized and
    # fed back into a differentiable forward; inference tensors cannot take
    # part in autograd, but no_grad tensors can.
    with torch.no_grad():
        for row, image in zip(rows, images):
            inputs = processor(text=row["prompt"], images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            gen_kwargs: Dict[str, Any] = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                do_sample=do_sample,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with _autocast_context(torch, device, autocast_dtype):
                generated_ids = model.generate(**gen_kwargs)
            raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            caption = extract_task_text(
                processor.post_process_generation(raw_text, task=PERSON_TASK, image_size=image.size),
                PERSON_TASK,
            ).strip()
            captions.append(caption)
    return captions


def _per_sample_logprob(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    prompt_inputs: Dict[str, Any],
    caption: str,
    target_max_length: int,
    device: Any,
    autocast_dtype: Any,
) -> Any:
    """Differentiable summed log-prob of one sampled caption.

    Re-tokenizes the caption text and passes it as `labels` through the SAME
    forward path the SFT trainer uses (the model builds decoder_input_ids from
    labels), so the logits/labels alignment is identical to SFT. Per-element CE
    with reduction="none" is masked and summed per sample to give logprob = -CE.
    `model` is the DDP-wrapped model so gradients sync at the group boundary.
    """
    torch = runtime["torch"]
    functional = runtime["functional"]
    labels = processor.tokenizer(
        text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=target_max_length,
    ).input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    with _autocast_context(torch, device, autocast_dtype):
        outputs = model(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs.get("attention_mask"),
            pixel_values=prompt_inputs["pixel_values"],
            labels=labels,
        )
    per_token = _cross_entropy(functional, outputs.logits, labels, 0.0, "none")
    per_token = per_token.reshape(labels.shape)  # [1, L]
    mask = (labels != -100).float()
    return -(per_token * mask).sum()  # scalar logprob


def _compute_rewards(
    captions: Sequence[str],
    gt_attrs: Sequence[Mapping[str, Any]],
    extractor_base_url: str,
    extractor_model: str,
    similarity_fn,
) -> List[float]:
    """Extract attributes for a batch of captions and score each against its GT."""
    extracted = extract_attributes(captions, base_url=extractor_base_url, model=extractor_model)
    prefetch_similarity_requests(
        gt_attrs,
        [item if isinstance(item, Mapping) else {} for item in extracted],
        similarity_fn,
    )
    rewards: List[float] = []
    for caption, gt, gen in zip(captions, gt_attrs, extracted):
        gen_attrs = gen if isinstance(gen, dict) else {}
        counts = classify_sample(gt, gen_attrs, similarity_fn)
        rewards.append(compute_reward(caption, counts))
    return rewards


def _evaluate_dev_rewards(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    autocast_dtype: Any,
    max_new_tokens: int,
    extractor_base_url: str,
    extractor_model: str,
    reward_batch_size: int,
    similarity_fn,
) -> List[Dict[str, Any]]:
    Image = runtime["Image"]
    was_training = model.training
    model.eval()
    try:
        images = []
        for row in rows:
            with Image.open(row["image"]) as source:
                images.append(source.convert("RGB"))
        captions = _generate_captions(
            runtime,
            model,
            processor,
            rows,
            images,
            device,
            autocast_dtype,
            max_new_tokens,
            False,
            1.0,
            1.0,
        )
        extracted = extract_attributes_batched(
            captions,
            base_url=extractor_base_url,
            model=extractor_model,
            batch_size=reward_batch_size,
        )
        prefetch_similarity_requests(
            [row.get("attributes", {}) for row in rows],
            [item if isinstance(item, Mapping) else {} for item in extracted],
            similarity_fn,
        )
        return build_dev_reward_records(rows, captions, extracted, similarity_fn)
    finally:
        if was_training:
            model.train()


def evaluate_dev_rewards_distributed(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    autocast_dtype: Any,
    max_new_tokens: int,
    extractor_base_url: str,
    extractor_model: str,
    reward_batch_size: int,
    similarity_fn,
    distributed: bool,
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    """Evaluate dev reward on every rank and return ordered records on rank 0."""
    dist = runtime["dist"]
    indexed_rows = indexed_dev_shard(rows, rank if distributed else 0, world_size if distributed else 1)
    try:
        local_records = _evaluate_dev_rewards(
            runtime,
            model,
            processor,
            [row for unused, row in indexed_rows],
            device,
            autocast_dtype,
            max_new_tokens,
            extractor_base_url,
            extractor_model,
            reward_batch_size,
            similarity_fn,
        )
        payload: Dict[str, Any] = {
            "ok": True,
            "rank": rank,
            "items": [
                (index, record)
                for (index, unused), record in zip(indexed_rows, local_records)
            ],
        }
    except Exception as exc:
        payload = {"ok": False, "rank": rank, "error": f"{type(exc).__name__}: {exc}"}

    if not distributed:
        return merge_dev_reward_payloads([payload], expected_count=len(rows))

    gathered = [None for unused in range(world_size)] if rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    status: List[Optional[Dict[str, Any]]] = [None]
    if rank == 0:
        try:
            records = merge_dev_reward_payloads(gathered, expected_count=len(rows))
            status[0] = {"ok": True, "count": len(records)}
        except Exception as exc:
            status[0] = {"ok": False, "error": str(exc)}
    dist.broadcast_object_list(status, src=0)
    if not status[0].get("ok"):
        raise RuntimeError(status[0]["error"])
    if rank == 0:
        return records
    return []


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    train_rows, dev_rows, data_metadata = load_rl_training_data(args)
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
    Image = runtime["Image"]

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    validate_world_size(world_size, formal=not args.allow_nonstandard_counts)
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
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=args.ddp_timeout_minutes)
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    generated_name = datetime.now().strftime("person_rl_%Y%m%d_%H%M%S")
    if distributed and args.resume_from is None and args.run_name is None:
        names = [generated_name if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        generated_name = names[0]
    run_dir = resolve_run_dir(
        args.output_root, args.run_name, args.resume_from, generated_name
    )
    _setup_logging(run_dir, rank)

    try:
        preflight_result = (
            run_reward_service_preflight(args) if rank == 0 else None
        )
        if distributed:
            preflight_holder = [preflight_result]
            dist.broadcast_object_list(preflight_holder, src=0)
            preflight_result = preflight_holder[0]
        if not isinstance(preflight_result, Mapping) or not preflight_result.get("ok"):
            error = (
                preflight_result.get("error", "unknown preflight error")
                if isinstance(preflight_result, Mapping)
                else "rank 0 returned no preflight result"
            )
            raise RuntimeError(str(error))
        if rank == 0:
            if preflight_result.get("skipped"):
                logging.warning("reward service preflight explicitly skipped")
            else:
                logging.info("reward service preflight passed: %s", preflight_result["checks"])

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
            logging.info("RL parameter groups: %s", group_summary)
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
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            drop_last=False,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.per_device_batch_size,
            sampler=dev_sampler,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.dev_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.dev_num_workers > 0,
            prefetch_factor=(
                args.prefetch_factor if args.dev_num_workers > 0 else None
            ),
            drop_last=False,
        )

        steps_per_epoch = math.ceil(
            len(train_loader) / args.gradient_accumulation_steps
        )
        planned_total_steps = steps_per_epoch * args.epochs
        total_steps = scheduler_total_steps(
            steps_per_epoch, args.epochs, args.max_optimizer_steps
        )
        warmup_steps = math.ceil(total_steps * args.warmup_ratio)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: cosine_with_floor_multiplier(
                current_step, warmup_steps, total_steps, args.min_lr_ratio
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=precision["use_scaler"])
        invariants = _invariants(
            args, data_metadata, world_size, steps_per_epoch, total_steps
        )
        if rank == 0 and args.resume_from is None:
            write_json_atomic(run_dir / "run_config.json", invariants)

        similarity_fn = (
            make_judge_fn(
                args.judge_base_url,
                args.judge_model,
                concurrency=args.judge_request_concurrency,
            )
            if args.reward_matcher == "qwen"
            else lexical_similarity
        )
        if rank == 0:
            logging.info(
                "RL config: train=%d dev=%d global_batch=%d steps=%d "
                "alpha=%.2f->%.2f matcher=%s extractor=%s judge=%s",
                len(train_rows),
                len(dev_rows),
                invariants["global_batch_size"],
                total_steps,
                args.alpha_start,
                args.alpha_end,
                args.reward_matcher,
                args.extractor_model,
                args.judge_model,
            )

        start_epoch = 0
        start_micro_batch = 0
        global_step = 0
        best_checkpoints: List[Dict[str, Any]] = []
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
            start_epoch = int(resume_state["epoch"])
            start_micro_batch = int(resume_state["next_micro_batch"])
            global_step = int(resume_state["global_step"])
            best_checkpoints = list(resume_state.get("best_checkpoints", []))
            if global_step >= total_steps:
                logging.info(
                    "checkpoint already reached all %d configured optimizer steps; nothing to resume",
                    total_steps,
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
        final_epoch = start_epoch
        final_next_micro_batch = start_micro_batch
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for micro_batch_index, (batch_rows, inputs, labels) in enumerate(
                train_loader
            ):
                if epoch == start_epoch and micro_batch_index < start_micro_batch:
                    continue
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

                inputs, labels = _move_batch(inputs, labels, device)
                images = []
                for row in batch_rows:
                    with Image.open(row["image"]) as source:
                        images.append(source.convert("RGB"))
                gt_attrs = [row.get("attributes", {}) for row in batch_rows]

                with sync_context:
                    with _autocast_context(
                        torch, device, precision["autocast_dtype"]
                    ):
                        ce_outputs = model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            pixel_values=inputs["pixel_values"],
                            labels=labels,
                        )
                        ce_loss = _cross_entropy(
                            functional,
                            ce_outputs.logits,
                            labels,
                            args.label_smoothing,
                            "mean",
                        )
                    sampled = _generate_captions(
                        runtime,
                        unwrapped_model,
                        processor,
                        batch_rows,
                        images,
                        device,
                        precision["autocast_dtype"],
                        args.generation_max_new_tokens,
                        True,
                        args.temperature,
                        args.top_p,
                    )
                    greedy = _generate_captions(
                        runtime,
                        unwrapped_model,
                        processor,
                        batch_rows,
                        images,
                        device,
                        precision["autocast_dtype"],
                        args.generation_max_new_tokens,
                        False,
                        args.temperature,
                        args.top_p,
                    )
                    r_sampled = _compute_rewards(
                        sampled,
                        gt_attrs,
                        args.extractor_base_url,
                        args.extractor_model,
                        similarity_fn,
                    )
                    r_greedy = _compute_rewards(
                        greedy,
                        gt_attrs,
                        args.extractor_base_url,
                        args.extractor_model,
                        similarity_fn,
                    )
                    advantages_t = torch.tensor(
                        [advantage(s, g) for s, g in zip(r_sampled, r_greedy)],
                        dtype=torch.float32,
                        device=device,
                    )
                    prompt_inputs_per_row = [
                        {
                            key: value.to(device)
                            for key, value in processor(
                                text=row["prompt"],
                                images=image,
                                return_tensors="pt",
                            ).items()
                        }
                        for row, image in zip(batch_rows, images)
                    ]
                    logprobs = torch.stack(
                        [
                            _per_sample_logprob(
                                runtime,
                                model,
                                processor,
                                prompt_inputs,
                                caption,
                                args.target_max_length,
                                device,
                                precision["autocast_dtype"],
                            )
                            for prompt_inputs, caption in zip(
                                prompt_inputs_per_row, sampled
                            )
                        ]
                    )
                    alpha = alpha_at_step(
                        global_step,
                        total_steps,
                        args.alpha_start,
                        args.alpha_end,
                    )
                    scst = -(advantages_t * logprobs).mean()
                    loss = (1.0 - alpha) * ce_loss + alpha * scst
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {global_step + 1}"
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
                final_epoch = epoch
                final_next_micro_batch = micro_batch_index + 1

                if rank == 0 and (
                    global_step == 1 or global_step % args.log_steps == 0
                ):
                    logging.info(
                        "epoch=%d step=%d/%d ce=%.4f scst=%.4f alpha=%.3f "
                        "r_s=%.3f r_g=%.3f advantage=%.3f grad=%.4f",
                        epoch + 1,
                        global_step,
                        total_steps,
                        float(ce_loss.detach()),
                        float(scst.detach()),
                        alpha,
                        statistics.fmean(r_sampled),
                        statistics.fmean(r_greedy),
                        float(advantages_t.mean()),
                        float(grad_norm),
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
                    kept = best_checkpoints
                    removed: List[Dict[str, Any]] = []
                    sample_rows = dev_rows[: args.eval_generation_samples]
                    records = evaluate_dev_rewards_distributed(
                        runtime,
                        unwrapped_model,
                        processor,
                        sample_rows,
                        device,
                        precision["autocast_dtype"],
                        args.generation_max_new_tokens,
                        args.extractor_base_url,
                        args.extractor_model,
                        args.eval_reward_batch_size,
                        similarity_fn,
                        distributed,
                        rank,
                        world_size,
                    )
                    if rank == 0:
                        metrics = summarize_dev_records(records, dev_loss)
                        metrics["global_step"] = global_step
                        logging.info("dev metrics: %s", metrics)
                        if not args.no_save:
                            dev_predictions_path(run_dir, global_step).parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            write_jsonl_atomic(
                                dev_predictions_path(run_dir, global_step), records
                            )
                            write_json_atomic(
                                dev_metrics_path(run_dir, global_step), metrics
                            )
                            candidate = {
                                "name": f"checkpoint-{global_step}",
                                "mean_reward": metrics["mean_reward"],
                                "dev_loss": dev_loss,
                                "global_step": global_step,
                            }
                            kept, removed = select_top_checkpoints(
                                best_checkpoints + [candidate]
                            )
                    rng_by_rank = (
                        _gather_rng_states(runtime, distributed, rank, world_size)
                        if not args.no_save
                        else []
                    )
                    if rank == 0 and not args.no_save:
                        checkpoint_dir = run_dir / f"checkpoint-{global_step}"
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
                        for removed_checkpoint in removed:
                            removed_path = run_dir / removed_checkpoint["name"]
                            if removed_path.exists():
                                shutil.rmtree(removed_path)
                    if distributed:
                        shared_best = [kept if rank == 0 else None]
                        dist.broadcast_object_list(shared_best, src=0)
                        best_checkpoints = shared_best[0]
                        dist.barrier()
                    else:
                        best_checkpoints = kept

                if global_step >= total_steps:
                    stop_training = True
                    break
            if stop_training:
                break
            final_epoch = epoch + 1
            final_next_micro_batch = 0
            start_micro_batch = 0

        status = final_status(
            global_step, planned_total_steps, args.max_optimizer_steps
        )
        final_rng_by_rank = (
            _gather_rng_states(runtime, distributed, rank, world_size)
            if not args.no_save
            else []
        )
        if rank == 0 and not args.no_save:
            final_dir = run_dir / "final"
            state = {
                "epoch": final_epoch,
                "next_micro_batch": final_next_micro_batch,
                "global_step": global_step,
                "best_checkpoints": best_checkpoints,
                "rng_by_rank": final_rng_by_rank,
                "invariants": invariants,
                **status,
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
            logging.info(
                "RL training stopped at step %d; saved final to %s status=%s",
                global_step,
                final_dir,
                status,
            )
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _invariants(
    args: argparse.Namespace,
    data_metadata: Mapping[str, Any],
    world_size: int,
    steps_per_epoch: int,
    total_steps: int,
) -> Dict[str, Any]:
    return {
        "model_source": str(args.model_path.resolve()),
        "data": {
            **dict(data_metadata),
            "train_path": str(args.train_data.resolve()),
            "train_sha256": sha256_file(args.train_data),
            "dev_path": str(args.dev_data.resolve()),
            "dev_sha256": sha256_file(args.dev_data),
        },
        "world_size": world_size,
        "distributed": {"timeout_minutes": args.ddp_timeout_minutes},
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_size": (
            world_size
            * args.per_device_batch_size
            * args.gradient_accumulation_steps
        ),
        "formal_global_batch_size": args.formal_global_batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "max_optimizer_steps": args.max_optimizer_steps,
        "learning_rates": {
            "vision": args.vision_lr,
            "projection": args.projection_lr,
            "language": args.language_lr,
        },
        "optimizer": {
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "adam_eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
        },
        "scheduler": {
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "precision": args.precision,
        "label_smoothing": args.label_smoothing,
        "alpha": {"start": args.alpha_start, "end": args.alpha_end},
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt_max_length": args.prompt_max_length,
            "target_max_length": args.target_max_length,
            "max_new_tokens": args.generation_max_new_tokens,
        },
        "reward": {
            "matcher": args.reward_matcher,
            "extractor_base_url": args.extractor_base_url,
            "extractor_model": args.extractor_model,
            "judge_base_url": args.judge_base_url,
            "judge_model": args.judge_model,
            "judge_request_concurrency": args.judge_request_concurrency,
            "service_preflight_skipped": args.skip_reward_service_preflight,
            "services": {
                "extractor": service_metadata(EXTRACTOR_SERVICE),
                "judge": service_metadata(JUDGE_SERVICE),
            },
            "weights": {
                "f1": W_F1,
                "fabrication": W_FABRICATION,
                "structure": W_STRUCTURE,
                "length": W_LENGTH,
                "background": W_BACKGROUND,
            },
            "reward_code_sha256": sha256_file(PROJECT_ROOT / "scripts/rl_reward.py"),
            "client_code_sha256": sha256_file(PROJECT_ROOT / "scripts/rl_clients.py"),
            "service_config_code_sha256": sha256_file(
                PROJECT_ROOT / "scripts/rl_service_config.py"
            ),
            "extractor_prompt_code_sha256": sha256_file(
                PROJECT_ROOT / "scripts/extract_qwen_attributes.py"
            ),
        },
        "evaluation": {
            "eval_steps": args.eval_steps,
            "generation_samples": args.eval_generation_samples,
            "reward_batch_size": args.eval_reward_batch_size,
            "dev_num_workers": args.dev_num_workers,
        },
        "data_loading": {
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "seed": args.seed,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_PREPARED_DIR / "rl.jsonl")
    parser.add_argument("--dev-data", type=Path, default=DEFAULT_PREPARED_DIR / "dev.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SFT_FINAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RL_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--allow-nonstandard-counts", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--formal-global-batch-size", type=int, default=8)
    parser.add_argument("--vision-lr", type=float, default=5e-8)
    parser.add_argument("--projection-lr", type=float, default=2e-7)
    parser.add_argument("--language-lr", type=float, default=2e-7)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt-max-length", type=int, default=1024)
    parser.add_argument("--target-max-length", type=int, default=64)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-generation-samples", type=int, default=128)
    parser.add_argument("--eval-reward-batch-size", type=int, default=4)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dev-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--alpha-start", type=float, default=ALPHA_START)
    parser.add_argument("--alpha-end", type=float, default=ALPHA_END)
    parser.add_argument("--reward-matcher", choices=["qwen", "lexical"], default="qwen",
                        help="similarity function for the soft-F1 reward: qwen (5-level judge) or lexical (evaluator-style)")
    parser.add_argument("--extractor-base-url", default=os.environ.get("QWEN_EXTRACTOR_BASE_URL", "http://127.0.0.1:6097/v1"))
    parser.add_argument("--extractor-model", default=os.environ.get("QWEN_EXTRACTOR_MODEL", "Qwen3.6-27B-FP8"))
    parser.add_argument("--judge-base-url", default=os.environ.get("QWEN_JUDGE_BASE_URL", "http://127.0.0.1:6098/v1"))
    parser.add_argument("--judge-model", default=os.environ.get("QWEN_JUDGE_MODEL", "Qwen3.5-4B"))
    parser.add_argument("--judge-request-concurrency", type=int, default=16)
    parser.add_argument("--ddp-timeout-minutes", type=int, default=60)
    parser.add_argument(
        "--skip-reward-service-preflight",
        action="store_true",
        help="skip model identity and no-thinking HTTP probes (recorded in run invariants)",
    )
    args = parser.parse_args()
    for name in (
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "formal_global_batch_size",
        "eval_generation_samples",
        "eval_reward_batch_size",
        "judge_request_concurrency",
        "ddp_timeout_minutes",
    ):
        _positive_integer(getattr(args, name), name)
    if not (0.0 <= args.label_smoothing < 1.0):
        parser.error("--label-smoothing must be in [0, 1)")
    if args.warmup_ratio >= 1.0:
        parser.error("--warmup-ratio must be < 1")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        parser.error("--min-lr-ratio must be in [0, 1]")
    if not (0.0 <= args.alpha_start <= args.alpha_end <= 1.0):
        parser.error("require 0 <= --alpha-start <= --alpha-end <= 1")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
