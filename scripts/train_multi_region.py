#!/usr/bin/env python3
"""
Fine-tune Florence-2 on the new <REGIONS_TO_DESCRIPTIONS> task.

Single-GPU training. The dataset is the pre-built no-name JSONL from
multi_region_description/scripts/prepare_data.py, where each row already holds:
    {"image": "<abs path>", "prompt": "<REGIONS_TO_DESCRIPTIONS>...", "label": "d1<sep>d2..."}

The processor's _construct_prompts() expands the task token into the text prompt
and keeps every <loc_XXX>/<sep> special token intact, so we pass `prompt` straight
through. All required tokens already exist in the base tokenizer -> no embedding resize.

Goal: light-touch fine-tune (2 epochs, small lr) so the model learns the
multi-region description *pattern* without drifting from the base checkpoint.
"""

import argparse
import json
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    dynamic_module_utils,
    get_scheduler,
)

try:
    from safetensors.torch import load_file as load_safetensors_file
except ModuleNotFoundError:
    load_safetensors_file = None


DEFAULT_BASE_MODEL = "/data/work/MichaelYu/florence-caption/ugipc_1231_15words_epoch3_full_handoff/checkpoint"
DEFAULT_DATA = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "ugipc-person-region-descriptions-yolo26m-no-name/train.jsonl"
)
DEFAULT_OUTPUT_ROOT = "/data/work/MichaelYu/florence-caption/multi_region_description/checkpoints"
DEFAULT_LOG_ROOT = "/data/work/MichaelYu/florence-caption/multi_region_description/logs"


# ---------------------------------------------------------------------------
# precision (mirrors the person-attribute trainer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    model_dtype: torch.dtype
    input_dtype: torch.dtype
    autocast_dtype: Optional[torch.dtype]
    use_grad_scaler: bool


def resolve_precision(precision):
    cuda = torch.cuda.is_available()
    if precision == "fp32" or not cuda:
        return PrecisionConfig("fp32", torch.float32, torch.float32, None, False)
    if precision == "fp16":
        return PrecisionConfig("fp16_amp", torch.float32, torch.float32, torch.float16, True)
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but device does not support it")
        return PrecisionConfig("bf16_amp", torch.float32, torch.float32, torch.bfloat16, False)
    raise ValueError(f"unsupported precision: {precision}")


# ---------------------------------------------------------------------------
# model loading (mirrors the person-attribute trainer's robust path)
# ---------------------------------------------------------------------------

def patch_florence_dynamic_import_check():
    """Ignore optional flash-attn during Transformers dynamic-module import checks."""
    original_get_imports = dynamic_module_utils.get_imports

    def patched_get_imports(filename):
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    dynamic_module_utils.get_imports = patched_get_imports


def load_florence_model(model_path, torch_dtype):
    patch_florence_dynamic_import_check()
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager"
    )
    safetensors_path = Path(model_path) / "model.safetensors"
    if safetensors_path.exists():
        if load_safetensors_file is None:
            raise ModuleNotFoundError("safetensors is required to load model.safetensors")
        state_dict = load_safetensors_file(str(safetensors_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        logging.info("loaded safetensors; missing=%s unexpected=%s", list(missing), list(unexpected))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="eager",
            local_files_only=True, torch_dtype=torch_dtype,
        )
    return model.to(dtype=torch_dtype)


# ---------------------------------------------------------------------------
# dataset: rows are pre-built (image/prompt/label)
# ---------------------------------------------------------------------------

class MultiRegionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        return row["prompt"], row["label"], image


def collate_fn(batch, processor, input_dtype, prompt_max_length, target_max_length):
    prompts, answers, images = zip(*batch)
    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_max_length,
    )
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.to(dtype=input_dtype)
    labels = processor.tokenizer(
        text=list(answers),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=target_max_length,
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return inputs, labels


def save_checkpoint(model, processor, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logging.info("saved checkpoint to %s", output_dir)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(args):
    run_name = args.run_name or datetime.now().strftime("multi_region_%Y%m%d_%H%M%S")
    log_dir = Path(args.log_root) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "train.log", encoding="utf-8")],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision)
    logging.info("run_name=%s device=%s precision=%s", run_name, device, precision.name)
    logging.info(
        "epochs=%s batch_size=%s lr=%s warmup=%s seed=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.warmup_steps,
        args.seed,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    patch_florence_dynamic_import_check()
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = load_florence_model(args.model_path, precision.model_dtype)
    model.to(device)

    dataset = MultiRegionDataset(args.data)
    logging.info("train samples=%d", len(dataset))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, processor, precision.input_dtype, args.prompt_max_length, args.target_max_length),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.use_grad_scaler)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        seen = 0
        iterator = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for inputs, labels in iterator:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=precision.autocast_dtype)
                if precision.autocast_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels,
                )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            seen += 1
            running_loss += float(loss.detach().cpu())
            if global_step % args.log_steps == 0 or seen == 1:
                avg = running_loss / max(1, seen)
                logging.info(
                    "epoch=%s step=%s/%s global_step=%s loss=%.6f avg_loss=%.6f lr=%.8g",
                    epoch + 1, seen, steps_per_epoch, global_step,
                    float(loss.detach().cpu()), avg, scheduler.get_last_lr()[0],
                )
                iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", avg=f"{avg:.4f}")

            if args.max_steps and global_step >= args.max_steps:
                logging.info("reached --max-steps=%s, stopping dry-run without saving", args.max_steps)
                return

        logging.info("epoch=%s completed avg_loss=%.6f", epoch + 1, running_loss / max(1, seen))
        save_checkpoint(model, processor, Path(args.output_root) / run_name / f"epoch_{epoch + 1}")

    save_checkpoint(model, processor, Path(args.output_root) / run_name / "final")
    logging.info("training done; final checkpoint at %s", Path(args.output_root) / run_name / "final")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Florence-2 for <REGIONS_TO_DESCRIPTIONS>")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="slightly higher lr so the model actually learns the multi-region pattern")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prompt-max-length",
        type=int,
        default=1024,
        help=(
            "Passed to the processor as max_length. NOTE: the Florence-2 processor "
            "subtracts image_seq_length (577) from this internally, so it must stay "
            "well above 577 + prompt token count (~35) or the prompt gets truncated away."
        ),
    )
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If > 0, stop after this many optimizer steps and skip saving. Use for dry-run smoke tests.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
