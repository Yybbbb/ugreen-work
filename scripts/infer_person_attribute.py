#!/usr/bin/env python3
"""Streaming, resumable full-test inference for Florence person attributes."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "sft"
    / "region_category_person_sft_30k_replay1k_b64"
    / "final"
)
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "prepared" / "test.jsonl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "sft"
    / "region_category_person_sft_30k_replay1k_b64"
    / "test_inference_final"
)
PERSON_TASK = "<REGION_TO_CATEGORY>"


def clean_prediction(value: str) -> str:
    """Remove tokenizer markers that can be emitted after EOS in a batch."""
    cleaned = value
    for token in ("<pad>", "</s>", "<s>"):
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split()).strip()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def iter_limited_jsonl(path: Path, max_rows: int = 0) -> Iterable[Dict[str, Any]]:
    for index, row in enumerate(iter_jsonl(path)):
        if max_rows and index >= max_rows:
            break
        yield row


def rank_indices(total_rows: int, rank: int, world_size: int) -> Iterable[int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    return range(rank, total_rows, world_size)


def load_completed_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not Path(path).exists():
        return completed
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Prediction row missing sample_id in {path}")
        if sample_id in completed:
            raise ValueError(f"Duplicate sample_id in {path}: {sample_id}")
        completed[sample_id] = row
    return completed


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)


def merge_predictions(
    test_path: Path,
    rank_paths: Sequence[Path],
    output_path: Path,
    max_rows: int = 0,
) -> int:
    predictions: Dict[str, Dict[str, Any]] = {}
    for rank_path in rank_paths:
        for row in iter_jsonl(rank_path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Prediction row missing sample_id in {rank_path}")
            if sample_id in predictions:
                raise ValueError(f"Duplicate sample_id across predictions: {sample_id}")
            predictions[sample_id] = row

    ordered: List[Dict[str, Any]] = []
    input_ids = set()
    for row in iter_limited_jsonl(test_path, max_rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Test row missing sample_id")
        if sample_id in input_ids:
            raise ValueError(f"Duplicate sample_id in test data: {sample_id}")
        input_ids.add(sample_id)
        if sample_id not in predictions:
            raise ValueError(f"missing sample_id in predictions: {sample_id}")
        ordered.append(predictions[sample_id])
    extra = set(predictions).difference(input_ids)
    if extra and not max_rows:
        raise ValueError(f"Predictions contain {len(extra)} sample_id values absent from test")
    _write_jsonl_atomic(output_path, ordered)
    return len(ordered)


def _runtime_and_model(checkpoint: Path, device: Any):
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_person_attribute_sft import (
        _load_model_and_processor,
        load_training_runtime,
    )

    runtime = load_training_runtime()
    torch = runtime["torch"]
    model, processor = _load_model_and_processor(runtime, checkpoint, torch.float32)
    model.to(device=device)
    model.eval()
    return runtime, model, processor


def _generate_batch(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    torch = runtime["torch"]
    Image = runtime["Image"]
    images = []
    image_sizes = []
    for row in rows:
        with Image.open(row["image"]) as source_image:
            image = source_image.convert("RGB")
        image_sizes.append(image.size)
        images.append(image)
    inputs = processor(
        text=[str(row["prompt"]) for row in rows],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device=device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                early_stopping=False,
            )
    raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    results = []
    for row, image_size, raw_text in zip(rows, image_sizes, raw_texts):
        post_processed = processor.post_process_generation(
            raw_text, task=PERSON_TASK, image_size=image_size
        )
        from person_sft_data import extract_task_text

        prediction = clean_prediction(extract_task_text(post_processed, PERSON_TASK))
        result = dict(row)
        result.update(
            {
                "prediction": prediction.strip(),
                "raw_generation": raw_text,
            }
        )
        results.append(result)
    for image in images:
        image.close()
    return results


def _distributed_context(torch: Any):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun inference requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, distributed, device


def run_inference(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.test_data.is_file():
        raise FileNotFoundError(f"Test JSONL does not exist: {args.test_data}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the runtime before distributed initialization so each rank owns one model.
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_person_attribute_sft import load_training_runtime

    runtime = load_training_runtime()
    torch = runtime["torch"]
    rank, world_size, distributed, device = _distributed_context(torch)
    rank_path = args.output_dir / f"predictions.rank{rank}.jsonl"
    completed = load_completed_predictions(rank_path)
    model_runtime, model, processor = _runtime_and_model(args.checkpoint, device)
    total_rows = sum(1 for _ in iter_limited_jsonl(args.test_data, args.max_samples))
    pending: List[Dict[str, Any]] = []
    pending_indices: List[int] = []
    processed = 0
    with rank_path.open("a", encoding="utf-8") as output_file:
        for row_index, row in enumerate(iter_limited_jsonl(args.test_data, args.max_samples)):
            if row_index % world_size != rank or row.get("sample_id") in completed:
                continue
            pending.append(row)
            pending_indices.append(row_index)
            if len(pending) < args.batch_size:
                continue
            for row_index, result in zip(pending_indices, _generate_batch(
                model_runtime, model, processor, pending, device, args.max_new_tokens
            )):
                result["row_index"] = row_index
                result["checkpoint"] = str(args.checkpoint.resolve())
                result["generation"] = {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": args.max_new_tokens,
                }
                output_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                output_file.write("\n")
                processed += 1
            output_file.flush()
            os.fsync(output_file.fileno())
            pending.clear()
            pending_indices.clear()
            if processed and processed % (args.batch_size * 10) == 0:
                logging.info("rank=%d generated=%d", rank, processed)
        if pending:
            for row_index, result in zip(
                pending_indices,
                _generate_batch(
                    model_runtime, model, processor, pending, device, args.max_new_tokens
                ),
            ):
                result["row_index"] = row_index
                result["checkpoint"] = str(args.checkpoint.resolve())
                result["generation"] = {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": args.max_new_tokens,
                }
                output_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        rank_paths = [args.output_dir / f"predictions.rank{i}.jsonl" for i in range(world_size)]
        count = merge_predictions(
            args.test_data,
            rank_paths,
            args.output_dir / "predictions.jsonl",
            max_rows=args.max_samples,
        )
        manifest = {
            "checkpoint": str(args.checkpoint.resolve()),
            "test_data": str(args.test_data.resolve()),
            "output": str((args.output_dir / "predictions.jsonl").resolve()),
            "samples": count,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logging.info("merged %d test predictions into %s", count, args.output_dir / "predictions.jsonl")
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Run only the first N rows for smoke tests; 0 means the complete test set.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.max_samples < 0:
        parser.error("batch size and max new tokens must be positive; max samples cannot be negative")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run_inference(parse_args()))


if __name__ == "__main__":
    main()
