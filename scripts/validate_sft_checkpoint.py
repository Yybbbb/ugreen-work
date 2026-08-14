#!/usr/bin/env python3
"""Reload a Florence SFT checkpoint and validate its native task contract."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from person_sft_data import build_region_prompt, extract_task_text, write_json_atomic
from train_person_attribute_sft import (
    PERSON_TASK,
    _load_model_and_processor,
    load_training_runtime,
    patch_florence_dynamic_import_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "pretrained" / "Florence-2-base"
CONTRACT_RAW_PROMPT = "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>"
CONTRACT_EXPANDED_PROMPT = "What is the region <loc_1><loc_2><loc_3><loc_4>?"


def validate_processor_contract(
    processor: Any, expected_vocab_size: Optional[int] = None
) -> Dict[str, Any]:
    expanded = processor._construct_prompts([CONTRACT_RAW_PROMPT])[0]
    if expanded != CONTRACT_EXPANDED_PROMPT:
        raise ValueError(
            f"Florence prompt contract changed: {expanded!r} != {CONTRACT_EXPANDED_PROMPT!r}"
        )
    post_processing = processor.tasks_answer_post_processing_type.get(PERSON_TASK)
    if post_processing != "pure_text":
        raise ValueError(
            f"Florence post-processing contract changed: {post_processing!r} != 'pure_text'"
        )
    tokenizer_size = len(processor.tokenizer)
    if expected_vocab_size is not None and tokenizer_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer size changed: {tokenizer_size} != {expected_vocab_size}"
        )
    return {
        "raw_prompt": CONTRACT_RAW_PROMPT,
        "expanded_prompt": expanded,
        "post_processing": post_processing,
        "tokenizer_size": tokenizer_size,
    }


def _load_processor_only(checkpoint: Path):
    try:
        from transformers import AutoConfig, AutoProcessor, dynamic_module_utils
    except ModuleNotFoundError as error:
        raise RuntimeError("Checkpoint validation requires transformers") from error
    patch_florence_dynamic_import_check(dynamic_module_utils)
    config = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    return processor, config


def _generate_one(
    runtime: Dict[str, Any],
    model: Any,
    processor: Any,
    image_path: Path,
    bbox_loc: Sequence[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    torch = runtime["torch"]
    Image = runtime["Image"]
    device = next(model.parameters()).device
    prompt = build_region_prompt(PERSON_TASK, bbox_loc)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        image_size = image.size
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    post_processed = processor.post_process_generation(
        raw_text, task=PERSON_TASK, image_size=image_size
    )
    prediction = extract_task_text(post_processed, PERSON_TASK)
    return {
        "image": str(image_path.resolve()),
        "prompt": prompt,
        "raw_generation": raw_text,
        "prediction": prediction,
    }


def validate_checkpoint(
    checkpoint: Path,
    processor_only: bool,
    image: Optional[Path],
    bbox_loc: Sequence[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint).resolve(strict=True)
    if processor_only:
        processor, config = _load_processor_only(checkpoint)
        expected_vocab_size = int(config.vocab_size)
        result = {
            "checkpoint": str(checkpoint),
            "processor": validate_processor_contract(processor, expected_vocab_size),
            "model_loaded": False,
        }
        if image is not None:
            raise ValueError("--image requires full model validation")
        return result

    runtime = load_training_runtime()
    torch = runtime["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(
        runtime, checkpoint, torch.float32
    )
    model.to(device)
    result = {
        "checkpoint": str(checkpoint),
        "processor": validate_processor_contract(
            processor, int(model.config.vocab_size)
        ),
        "model_loaded": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    if image is not None:
        result["generation"] = _generate_one(
            runtime, model, processor, image, bbox_loc, max_new_tokens
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload and validate a Florence person-attribute checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor-only", action="store_true")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--bbox-loc", type=int, nargs=4, default=[1, 2, 3, 4], metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_checkpoint(
        args.checkpoint,
        args.processor_only,
        args.image,
        args.bbox_loc,
        args.max_new_tokens,
    )
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
