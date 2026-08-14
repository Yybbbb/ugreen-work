#!/usr/bin/env python3
"""Benchmark joint multi-region generation against sequential single-region calls."""

from __future__ import annotations

import os

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints/qwen_person_2to5_cleaned_loc_order_gpu1_ep2_bs16_lr8e6_center_left_to_right/final"
)
DEFAULT_DATA_DIR = (
    PROJECT_DIR
    / "data/qwen-person-2to5-cleaned-description-loc-order-center-left-to-right"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "eval/results/crop_count_sequential_benchmark"

MULTI_TASK = "<REGIONS_TO_DESCRIPTIONS>"
SINGLE_TASK = "<REGION_TO_DESCRIPTION>"
SEP_TOKEN = "<sep>"
LOC_BBOX_PATTERN = re.compile(
    r"<loc_\d+><loc_\d+><loc_\d+><loc_\d+>"
)
EXPECTED_FULL_COUNTS = {2: 4219, 3: 2159, 4: 1218, 5: 728}


@dataclass(frozen=True)
class Sample:
    split: str
    split_index: int
    image_path: str
    multi_prompt: str
    bbox_tokens: tuple[str, ...]

    @property
    def crop_count(self) -> int:
        return len(self.bbox_tokens)


@dataclass(frozen=True)
class GenerationResult:
    generate_time_s: float
    end_to_end_time_s: float
    input_tokens: int
    output_tokens: int
    ended_with_eos: bool
    hit_max_new_tokens: bool
    peak_allocated_mib: float
    peak_reserved_mib: float
    raw_output: str | None


def parse_bbox_tokens(prompt: str) -> tuple[str, ...]:
    if not prompt.startswith(MULTI_TASK):
        raise ValueError(f"Expected prompt to start with {MULTI_TASK}: {prompt[:80]!r}")
    body = prompt[len(MULTI_TASK):]
    segments = body.split(SEP_TOKEN)
    bbox_tokens = tuple(segment.strip() for segment in segments)
    if not bbox_tokens or any(LOC_BBOX_PATTERN.fullmatch(item) is None for item in bbox_tokens):
        raise ValueError(f"Malformed multi-region prompt: {prompt!r}")
    return bbox_tokens


def build_single_prompts(bbox_tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{SINGLE_TASK}{bbox}" for bbox in bbox_tokens)


def load_samples(
    data_dir: Path,
    splits: Sequence[str],
    limit_per_crop_count: int,
) -> list[Sample]:
    samples: list[Sample] = []
    selected_counts = {crop_count: 0 for crop_count in EXPECTED_FULL_COUNTS}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as input_file:
            for split_index, line in enumerate(input_file):
                if not line.strip():
                    continue
                record = json.loads(line)
                bbox_tokens = parse_bbox_tokens(record["prompt"])
                crop_count = len(bbox_tokens)
                if crop_count not in EXPECTED_FULL_COUNTS:
                    raise ValueError(
                        f"Unexpected crop count {crop_count} in {path}:{split_index + 1}"
                    )
                if limit_per_crop_count and selected_counts[crop_count] >= limit_per_crop_count:
                    continue
                samples.append(
                    Sample(
                        split=split,
                        split_index=split_index,
                        image_path=record["image"],
                        multi_prompt=record["prompt"],
                        bbox_tokens=bbox_tokens,
                    )
                )
                selected_counts[crop_count] += 1
    return samples


def count_samples(samples: Iterable[Sample]) -> dict[int, int]:
    counts = {crop_count: 0 for crop_count in EXPECTED_FULL_COUNTS}
    for sample in samples:
        counts[sample.crop_count] += 1
    return counts


def validate_full_dataset_counts(
    samples: Sequence[Sample],
    splits: Sequence[str],
    limit_per_crop_count: int,
) -> None:
    if tuple(splits) != ("train", "test") or limit_per_crop_count:
        return
    actual = count_samples(samples)
    if actual != EXPECTED_FULL_COUNTS:
        raise ValueError(f"Dataset count mismatch: expected={EXPECTED_FULL_COUNTS}, actual={actual}")


def select_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but the selected CUDA device does not support it")
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(dtype_name)


def patch_flash_attn_import() -> None:
    original_get_imports = dynamic_module_utils.get_imports
    if getattr(original_get_imports, "_florence_benchmark_patched", False):
        return

    def patched_get_imports(filename: str | Path) -> list[str]:
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            return [name for name in imports if name != "flash_attn"]
        return imports

    patched_get_imports._florence_benchmark_patched = True  # type: ignore[attr-defined]
    dynamic_module_utils.get_imports = patched_get_imports


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    patch_flash_attn_import()
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generated_token_count(model, generated_ids: torch.Tensor, input_tokens: int) -> int:
    sequence_length = int(generated_ids.shape[1])
    if model.config.is_encoder_decoder:
        return max(sequence_length - 1, 0)
    return max(sequence_length - input_tokens, 0)


def sequence_ended_with_eos(processor, generated_ids: torch.Tensor) -> bool:
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None or generated_ids.shape[1] == 0:
        return False
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = set(eos_token_id)
    return int(generated_ids[0, -1].item()) in eos_token_ids


def timed_generate(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    num_beams: int,
    include_raw_output: bool,
) -> GenerationResult:
    synchronize(device)
    end_to_end_start = time.perf_counter()

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)
    if int(inputs["input_ids"].shape[0]) != 1:
        raise RuntimeError("Benchmark requires batch size exactly 1")

    input_tokens = int(inputs["input_ids"].shape[1])
    synchronize(device)
    generate_start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=False,
            no_repeat_ngram_size=0,
        )
    synchronize(device)
    generate_time_s = time.perf_counter() - generate_start

    output_tokens = generated_token_count(model, generated_ids, input_tokens)
    ended_with_eos = sequence_ended_with_eos(processor, generated_ids)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    end_to_end_time_s = time.perf_counter() - end_to_end_start

    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

    return GenerationResult(
        generate_time_s=generate_time_s,
        end_to_end_time_s=end_to_end_time_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ended_with_eos=ended_with_eos,
        hit_max_new_tokens=output_tokens >= max_new_tokens,
        peak_allocated_mib=peak_allocated_mib,
        peak_reserved_mib=peak_reserved_mib,
        raw_output=decoded if include_raw_output else None,
    )


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def aggregate_single_results(results: Sequence[GenerationResult]) -> dict[str, Any]:
    return {
        "call_count": len(results),
        "generate_time_s": sum(item.generate_time_s for item in results),
        "end_to_end_time_s": sum(item.end_to_end_time_s for item in results),
        "input_tokens": sum(item.input_tokens for item in results),
        "output_tokens": sum(item.output_tokens for item in results),
        "ended_with_eos_count": sum(item.ended_with_eos for item in results),
        "hit_max_new_tokens_count": sum(item.hit_max_new_tokens for item in results),
        "peak_allocated_mib": max((item.peak_allocated_mib for item in results), default=0.0),
        "peak_reserved_mib": max((item.peak_reserved_mib for item in results), default=0.0),
        "calls": [asdict(item) for item in results],
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p99", "min", "max")}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def summarize_subset(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        image_count = len(subset)
        crop_count = sum(record["crop_count"] for record in subset)
        output: dict[str, Any] = {
            "image_count": image_count,
            "crop_count": crop_count,
        }
        for timing_key in ("generate_time_s", "end_to_end_time_s"):
            multi_times = [record["multi"][timing_key] for record in subset]
            single_times = [record["single"][timing_key] for record in subset]
            multi_total = sum(multi_times)
            single_total = sum(single_times)
            speedups = [
                single_time / multi_time
                for single_time, multi_time in zip(single_times, multi_times)
                if multi_time > 0
            ]
            multi_output_tokens = sum(record["multi"]["output_tokens"] for record in subset)
            single_output_tokens = sum(record["single"]["output_tokens"] for record in subset)
            output[timing_key.removesuffix("_time_s")] = {
                "multi_total_s": multi_total,
                "single_total_s": single_total,
                "aggregate_speedup_single_over_multi": (
                    single_total / multi_total if multi_total else 0.0
                ),
                "per_image_speedup": distribution(speedups),
                "multi_time_distribution_s": distribution(multi_times),
                "single_time_distribution_s": distribution(single_times),
                "multi_images_per_s": image_count / multi_total if multi_total else 0.0,
                "single_images_per_s": image_count / single_total if single_total else 0.0,
                "multi_crops_per_s": crop_count / multi_total if multi_total else 0.0,
                "single_crops_per_s": crop_count / single_total if single_total else 0.0,
                "multi_output_tokens_per_s": (
                    multi_output_tokens / multi_total if multi_total else 0.0
                ),
                "single_output_tokens_per_s": (
                    single_output_tokens / single_total if single_total else 0.0
                ),
            }
        output["multi_hit_max_new_tokens"] = sum(
            record["multi"]["hit_max_new_tokens"] for record in subset
        )
        output["single_hit_max_new_tokens"] = sum(
            record["single"]["hit_max_new_tokens_count"] for record in subset
        )
        output["multi_peak_allocated_mib"] = max(
            (record["multi"]["peak_allocated_mib"] for record in subset), default=0.0
        )
        output["multi_peak_reserved_mib"] = max(
            (record["multi"]["peak_reserved_mib"] for record in subset), default=0.0
        )
        output["single_peak_allocated_mib"] = max(
            (record["single"]["peak_allocated_mib"] for record in subset), default=0.0
        )
        output["single_peak_reserved_mib"] = max(
            (record["single"]["peak_reserved_mib"] for record in subset), default=0.0
        )
        return output

    return {
        "overall": summarize_subset(records),
        "by_crop_count": {
            str(crop_count): summarize_subset(
                [record for record in records if record["crop_count"] == crop_count]
            )
            for crop_count in EXPECTED_FULL_COUNTS
        },
    }


def query_gpu_state(physical_gpu: str) -> dict[str, Any]:
    gpu_id = physical_gpu.split(",")[0]
    command = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        state: dict[str, Any] = {
            "physical_gpu": gpu_id,
            "name": values[1],
            "memory_total_mib": int(values[2]),
            "memory_used_mib": int(values[3]),
            "memory_free_mib": int(values[4]),
            "utilization_percent": int(values[5]),
        }
        process_command = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        process_result = subprocess.run(
            process_command, check=True, capture_output=True, text=True
        )
        processes = []
        for line in process_result.stdout.splitlines():
            if not line.strip():
                continue
            process_values = [value.strip() for value in line.split(",")]
            if len(process_values) == 3:
                processes.append(
                    {
                        "pid": int(process_values[0]),
                        "process_name": process_values[1],
                        "used_memory_mib": int(process_values[2]),
                    }
                )
        state["compute_processes"] = processes
        return state
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError) as error:
        return {"physical_gpu": gpu_id, "query_error": str(error)}


def build_report(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# 多 Crop 联合推理与单框顺序推理速度测试报告",
        "",
        "## 实验配置",
        "",
        f"- 模型 checkpoint：`{metadata['checkpoint']}`",
        f"- 数据集：`{metadata['data_dir']}`，使用 split：{', '.join(metadata['splits'])}",
        f"- 设备：`{metadata['device']}`，CUDA_VISIBLE_DEVICES=`{metadata['cuda_visible_devices']}`",
        f"- dtype / num_beams / max_new_tokens：{metadata['dtype']} / {metadata['num_beams']} / {metadata['max_new_tokens']}",
        "- 联合模式：每张整图只调用一次 `model.generate()`，prompt 中同时包含该图全部 loc bbox。",
        "- 单框顺序模式：同一张整图按 loc bbox 顺序调用 N 次 `model.generate()`，每次只输入一个 bbox。",
        "- 所有输入均为 batch size 1；不使用 DataLoader worker、多线程、多进程、异步队列或并行生成。",
        "- 模型只加载一次，两种模式使用同一 checkpoint、GPU、dtype 和生成参数。",
        "",
        "## 测量方法与指标定义",
        "",
        "### 计时范围",
        "",
        "- **Generate-only（纯生成耗时）**：在输入已经完成 processor 处理并传到 GPU 后，先执行 `torch.cuda.synchronize()`，随后从调用 `model.generate()` 前开始计时；生成结束后再次执行 `torch.cuda.synchronize()` 再停止计时。该指标只反映模型生成阶段的 GPU wall-clock 时间，不包含图像预处理、文本 tokenization、CPU→GPU 传输和文本 decode。",
        "- **End-to-end（单次推理端到端耗时）**：从调用 processor 之前开始，到生成结果完成 `batch_decode()` 后停止；包含图像预处理、prompt tokenization、张量传输、`model.generate()` 和输出 decode。",
        "- 磁盘读取图片、模型加载、warmup 和最终结果写盘均不计入上述两种耗时。每张图片只从磁盘读取一次；单框顺序模式会对同一张 PIL 图片重复执行 N 次 processor 和推理。",
        "- 为降低固定执行顺序带来的漂移，偶数序号图片先运行联合模式，奇数序号图片先运行单框顺序模式；任意时刻仍只有一个推理调用在执行。",
        "",
        "### 统计单位与计算公式",
        "",
        "- **Image（图片）**：数据集中的一条完整样本，即一张整图以及该图的 2–5 个 loc bbox。单框顺序模式处理完该图全部 bbox 后，才算完成一张 image。",
        "- **Crop（区域）**：prompt 中的一个 loc bbox。联合模式一次调用处理 N 个 crop；单框顺序模式使用 N 次调用处理相同的 N 个 crop。",
        "- **Mean**：该 crop 数分组内，每张整图耗时的算术平均值。单框顺序模式的一张图耗时为该图 N 次串行调用耗时之和。",
        "- **P50 / P90 / P99**：按每张整图耗时计算的第 50、90、99 百分位数。例如 P90 表示约 90% 的图片耗时不超过该值。",
        "- **Images/s**：`完成的整图数量 ÷ 该模式总耗时`，表示每秒能够完整处理多少张图片。",
        "- **Crops/s**：`全部 bbox 数量 ÷ 该模式总耗时`，表示每秒能够处理多少个区域。两种模式的分子完全相同，区别仅在处理这些区域所需的总时间。",
        "- **Tokens/s**：`实际生成的输出 token 总数 ÷ generate-only 或 end-to-end 总耗时`。Florence-2 为 encoder-decoder 模型，统计时排除 decoder start token，保留实际生成的文本、loc token、EOS 等 token。由于两种模式可能生成不同数量的 token，该指标用于描述生成吞吐量，不代表描述质量。",
        "- **加速比**：`单框顺序模式总耗时 ÷ 联合模式总耗时`。结果大于 1 表示联合模式更快，例如 2.0× 表示单框顺序模式耗时约为联合模式的两倍。",
        "- **总体加权加速比**：直接使用全量样本的总耗时计算，而不是对 2/3/4/5-crop 四组加速比做简单平均，因此样本更多的分组权重更高。",
        "- **达到 max_new_tokens**：实际生成 token 数达到 320，视为疑似被最大生成长度截断；正常遇到 EOS 提前结束的输出不计为截断。",
        "- **Peak allocated / reserved**：PyTorch CUDA allocator 记录的峰值已分配显存和峰值保留显存，不包含同一 GPU 上其他进程占用的显存。",
        "",
        "## GPU 运行状态",
        "",
        "```json",
        json.dumps(metadata["gpu_state"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for timing_name, title in (
        ("generate", "Generate-only 结果（纯模型生成）"),
        ("end_to_end", "End-to-end 结果（processor 到 decode）"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Crop 数 | 模式 | 图片数 | 平均耗时 | P50 | P90 | P99 | Images/s | Crops/s | Tokens/s |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for crop_count, group in summary["by_crop_count"].items():
            timing = group[timing_name]
            for mode, label in (("multi", "联合"), ("single", "单框顺序")):
                stats = timing[f"{mode}_time_distribution_s"]
                lines.append(
                    f"| {crop_count} | {label} | {group['image_count']} | "
                    f"{stats['mean'] * 1000:.2f} ms | {stats['p50'] * 1000:.2f} ms | "
                    f"{stats['p90'] * 1000:.2f} ms | {stats['p99'] * 1000:.2f} ms | "
                    f"{timing[f'{mode}_images_per_s']:.3f} | "
                    f"{timing[f'{mode}_crops_per_s']:.3f} | "
                    f"{timing[f'{mode}_output_tokens_per_s']:.3f} |"
                )
            lines.append(
                f"| {crop_count} | **联合加速比** | {group['image_count']} | "
                f"**{timing['aggregate_speedup_single_over_multi']:.3f}×** | — | — | — | — | — | — |"
            )
        overall = summary["overall"][timing_name]
        lines.extend(
            [
                "",
                f"全量样本加权加速比：**{overall['aggregate_speedup_single_over_multi']:.3f}×**",
                "",
                "| 模式 | P50 | P90 | P99 | Crops/s | 输出 Tokens/s |",
                "|---|---:|---:|---:|---:|---:|",
                f"| 联合 | {overall['multi_time_distribution_s']['p50'] * 1000:.2f} ms | "
                f"{overall['multi_time_distribution_s']['p90'] * 1000:.2f} ms | "
                f"{overall['multi_time_distribution_s']['p99'] * 1000:.2f} ms | "
                f"{overall['multi_crops_per_s']:.3f} | {overall['multi_output_tokens_per_s']:.3f} |",
                f"| 单框顺序 | {overall['single_time_distribution_s']['p50'] * 1000:.2f} ms | "
                f"{overall['single_time_distribution_s']['p90'] * 1000:.2f} ms | "
                f"{overall['single_time_distribution_s']['p99'] * 1000:.2f} ms | "
                f"{overall['single_crops_per_s']:.3f} | {overall['single_output_tokens_per_s']:.3f} |",
                "",
            ]
        )
    overall = summary["overall"]
    lines.extend(
        [
            "## 截断与显存",
            "",
            f"- 联合模式达到 max_new_tokens 的输出数：{overall['multi_hit_max_new_tokens']}",
            f"- 单框顺序模式达到 max_new_tokens 的输出数：{overall['single_hit_max_new_tokens']}",
            f"- 联合模式 peak allocated / reserved：{overall['multi_peak_allocated_mib']:.1f} / {overall['multi_peak_reserved_mib']:.1f} MiB",
            f"- 单框顺序模式 peak allocated / reserved：{overall['single_peak_allocated_mib']:.1f} / {overall['single_peak_reserved_mib']:.1f} MiB",
            "",
            "## 结果解读",
            "",
            f"- Generate-only 全量加权加速比为 **{summary['overall']['generate']['aggregate_speedup_single_over_multi']:.3f}×**，说明只比较模型生成阶段时，联合模式处理全量数据所需时间更少。",
            f"- End-to-end 全量加权加速比为 **{summary['overall']['end_to_end']['aggregate_speedup_single_over_multi']:.3f}×**；该收益高于纯生成阶段，因为联合模式每张图只执行一次 processor、张量传输和 decode，而单框顺序模式需要重复 N 次。",
            "- 随 crop 数从 2 增加到 5，联合模式的加速收益整体增大，说明一次共享整图视觉编码和推理开销比重复处理同一整图更高效。",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = select_dtype(args.dtype, device)
    samples = load_samples(args.data_dir, args.splits, args.limit_per_crop_count)
    validate_full_dataset_counts(samples, args.splits, args.limit_per_crop_count)
    print(f"Loaded {len(samples)} images: {count_samples(samples)}", flush=True)

    processor, model = load_model(args.checkpoint, device, dtype)
    print(f"Loaded checkpoint on {device} with dtype={dtype}", flush=True)

    if samples and args.warmup_iterations:
        warmup_sample = max(samples, key=lambda sample: sample.crop_count)
        with Image.open(warmup_sample.image_path) as opened_image:
            warmup_image = opened_image.convert("RGB")
        single_prompt = build_single_prompts(warmup_sample.bbox_tokens)[0]
        for _ in range(args.warmup_iterations):
            timed_generate(
                processor,
                model,
                warmup_image,
                warmup_sample.multi_prompt,
                device,
                dtype,
                args.max_new_tokens,
                args.num_beams,
                False,
            )
            timed_generate(
                processor,
                model,
                warmup_image,
                single_prompt,
                device,
                dtype,
                args.max_new_tokens,
                args.num_beams,
                False,
            )
        print(f"Completed {args.warmup_iterations} warmup iterations per mode", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.jsonl"
    records: list[dict[str, Any]] = []
    with details_path.open("w", encoding="utf-8") as details_file:
        for sample_index, sample in enumerate(samples):
            with Image.open(sample.image_path) as opened_image:
                image = opened_image.convert("RGB")
            single_prompts = build_single_prompts(sample.bbox_tokens)

            if sample_index % 2 == 0:
                execution_order = ("multi", "single")
            else:
                execution_order = ("single", "multi")
            multi_result: GenerationResult | None = None
            single_results: list[GenerationResult] = []
            for mode in execution_order:
                reset_peak_memory(device)
                if mode == "multi":
                    multi_result = timed_generate(
                        processor,
                        model,
                        image,
                        sample.multi_prompt,
                        device,
                        dtype,
                        args.max_new_tokens,
                        args.num_beams,
                        not args.omit_raw_output,
                    )
                else:
                    for single_prompt in single_prompts:
                        single_results.append(
                            timed_generate(
                                processor,
                                model,
                                image,
                                single_prompt,
                                device,
                                dtype,
                                args.max_new_tokens,
                                args.num_beams,
                                not args.omit_raw_output,
                            )
                        )
            if multi_result is None or len(single_results) != sample.crop_count:
                raise RuntimeError("Sequential benchmark call count invariant failed")

            record = {
                "sample_index": sample_index,
                "split": sample.split,
                "split_index": sample.split_index,
                "image": sample.image_path,
                "crop_count": sample.crop_count,
                "bbox_tokens": list(sample.bbox_tokens),
                "execution_order": list(execution_order),
                "multi": asdict(multi_result),
                "single": aggregate_single_results(single_results),
            }
            records.append(record)
            details_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            details_file.flush()
            if sample_index == 0 or (sample_index + 1) % args.log_every == 0:
                print(
                    f"[{sample_index + 1}/{len(samples)}] crops={sample.crop_count} "
                    f"generate multi={multi_result.generate_time_s:.3f}s "
                    f"single={record['single']['generate_time_s']:.3f}s",
                    flush=True,
                )
    return records, {"details_path": str(details_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one joint multi-region call with N sequential single-region calls."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument(
        "--limit-per-crop-count",
        type=int,
        default=0,
        help="Select at most N images for each crop count; 0 runs the full dataset.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--omit-raw-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.warmup_iterations < 0 or args.log_every <= 0:
        raise ValueError("Invalid non-positive benchmark argument")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)

    output_files = [
        args.output_dir / "details.jsonl",
        args.output_dir / "summary.json",
        args.output_dir / "report.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing benchmark outputs: "
            + ", ".join(str(path) for path in existing)
        )

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_state_before = query_gpu_state(cuda_visible_devices or "2")
    records, paths = run_benchmark(args)
    summary = summarize_records(records)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "splits": list(args.splits),
        "device": args.device,
        "cuda_visible_devices": cuda_visible_devices,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "warmup_iterations": args.warmup_iterations,
        "limit_per_crop_count": args.limit_per_crop_count,
        "gpu_state": {
            "before": gpu_state_before,
            "after": query_gpu_state(cuda_visible_devices or "2"),
        },
        **paths,
    }
    summary_document = {"metadata": metadata, "results": summary}
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(build_report(summary, metadata), encoding="utf-8")
    print(f"Details: {paths['details_path']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
