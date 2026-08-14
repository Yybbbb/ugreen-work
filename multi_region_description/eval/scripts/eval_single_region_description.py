#!/usr/bin/env python3
"""Batch inference and text evaluation for single REGION_TO_DESCRIPTION data."""

import argparse
import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils


ATTRIBUTE_PATTERNS = {
    "hair": re.compile(r"\b(?:hair|bald|shaved head|ponytail|bun)\b", re.I),
    "eyewear": re.compile(r"\b(?:glasses|sunglasses|goggles)\b", re.I),
    "upper_clothing": re.compile(
        r"\b(?:shirt|t-shirt|top|jacket|coat|hoodie|sweater|blouse|vest|dress|uniform)\b",
        re.I,
    ),
    "lower_clothing": re.compile(r"\b(?:pants|trousers|jeans|shorts|skirt|leggings)\b", re.I),
    "footwear": re.compile(r"\b(?:shoes|sneakers|boots|sandals|slippers|socks)\b", re.I),
    "color": re.compile(
        r"\b(?:black|white|gray|grey|red|yellow|green|blue|purple|pink|orange|brown|beige|dark|light-colored)\b",
        re.I,
    ),
    "carried_object": re.compile(
        r"\b(?:holding|carrying|phone|smartphone|bag|backpack|bottle|cup|umbrella|document|laptop)\b",
        re.I,
    ),
    "pose_action": re.compile(
        r"\b(?:standing|stands|sitting|seated|walking|walks|running|crouching|kneeling|looking|using)\b",
        re.I,
    ),
}


def patch_florence_dynamic_import_check() -> None:
    original_get_imports = dynamic_module_utils.get_imports

    def patched_get_imports(filename):
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    dynamic_module_utils.get_imports = patched_get_imports


def normalize_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for index_b, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def score_prediction(prediction: str, reference: str) -> dict:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((prediction_counts & reference_counts).values())
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lcs = lcs_length(prediction_tokens, reference_tokens)
    rouge_precision = lcs / len(prediction_tokens) if prediction_tokens else 0.0
    rouge_recall = lcs / len(reference_tokens) if reference_tokens else 0.0
    rouge_l_f1 = (
        2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
        if rouge_precision + rouge_recall
        else 0.0
    )
    return {
        "normalized_prediction": normalized_prediction,
        "normalized_reference": normalized_reference,
        "exact_match": normalized_prediction == normalized_reference,
        "character_similarity": SequenceMatcher(
            None, normalized_prediction, normalized_reference, autojunk=False
        ).ratio(),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": token_f1,
        "rouge_l_f1": rouge_l_f1,
        "prediction_tokens": len(prediction_tokens),
        "reference_tokens": len(reference_tokens),
    }


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    patch_florence_dynamic_import_check()
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return processor, model


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-region descriptions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    samples = []
    with args.test_file.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                samples.append(json.loads(line))
                if args.limit and len(samples) >= args.limit:
                    break

    processor, model = load_model(args.checkpoint, device, dtype)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    metric_sums = Counter()
    attribute_counts = {
        name: {"prediction": 0, "reference": 0} for name in ATTRIBUTE_PATTERNS
    }
    prediction_token_lengths = []
    reference_token_lengths = []

    with args.output_file.open("w", encoding="utf-8") as output_file:
        processed = 0
        for batch in batched(samples, args.batch_size):
            images = [Image.open(sample["image"]).convert("RGB") for sample in batch]
            inputs = processor(
                text=[sample["prompt"] for sample in batch],
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=640,
            )
            for image in images:
                image.close()
            inputs = {
                key: value.to(
                    device=device,
                    dtype=dtype if torch.is_floating_point(value) else value.dtype,
                    non_blocking=True,
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
            ):
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    no_repeat_ngram_size=0,
                )
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for sample, prediction in zip(batch, predictions):
                prediction = prediction.strip()
                scores = score_prediction(prediction, sample["label"])
                for key in (
                    "exact_match",
                    "character_similarity",
                    "token_precision",
                    "token_recall",
                    "token_f1",
                    "rouge_l_f1",
                ):
                    metric_sums[key] += float(scores[key])
                prediction_token_lengths.append(scores["prediction_tokens"])
                reference_token_lengths.append(scores["reference_tokens"])
                for name, pattern in ATTRIBUTE_PATTERNS.items():
                    attribute_counts[name]["prediction"] += int(bool(pattern.search(prediction)))
                    attribute_counts[name]["reference"] += int(bool(pattern.search(sample["label"])))
                output_file.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "frame_id": sample["frame_id"],
                            "image": sample["image"],
                            "prompt": sample["prompt"],
                            "reference": sample["label"],
                            "prediction": prediction,
                            **scores,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            processed += len(batch)
            logging.info("processed=%d/%d", processed, len(samples))

    sample_count = len(samples)
    summary = {
        "checkpoint": str(args.checkpoint),
        "test_file": str(args.test_file),
        "sample_count": sample_count,
        "generation": {
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
        },
        "metrics": {
            key: metric_sums[key] / sample_count for key in metric_sums
        },
        "lengths": {
            "prediction_mean_tokens": sum(prediction_token_lengths) / sample_count,
            "reference_mean_tokens": sum(reference_token_lengths) / sample_count,
            "prediction_max_tokens": max(prediction_token_lengths),
            "reference_max_tokens": max(reference_token_lengths),
        },
        "attribute_mention_rates": {
            name: {
                "prediction": counts["prediction"] / sample_count,
                "reference": counts["reference"] / sample_count,
            }
            for name, counts in attribute_counts.items()
        },
        "output_file": str(args.output_file),
    }
    args.summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
