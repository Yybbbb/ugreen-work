#!/usr/bin/env python3
"""Render V4 serial experiment status and metrics into the training record."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from v4_serial_config import ExperimentSpec, experiment_specs


START_MARKER = "<!-- V4_RESULTS_START -->"
END_MARKER = "<!-- V4_RESULTS_END -->"


def _metrics_path(spec: ExperimentSpec) -> Path:
    return spec.inference_dir / "evaluation_qwen_final" / "metrics.json"


def _load_metrics(spec: ExperimentSpec) -> Optional[Dict[str, Any]]:
    path = _metrics_path(spec)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _status(spec: ExperimentSpec) -> str:
    if _metrics_path(spec).exists():
        return "评估完成"
    if (spec.inference_dir / "qwen_attribute_extraction.manifest.json").exists():
        return "Qwen收尾与指标计算中"
    if (spec.inference_dir / "manifest.json").exists():
        return "Qwen抽取中"
    if (spec.run_dir / "final" / "run_state.json").exists():
        return "测试集推理中"
    if spec.run_dir.exists():
        return "训练中"
    return "等待串行执行"


def _result_row(spec: ExperimentSpec, metrics: Optional[Dict[str, Any]]) -> str:
    replay = "1,000" if not spec.person_only else "0"
    if metrics is None:
        values = ["-", "-", "-", "-", "-", "-", "-", "-"]
    else:
        caption = metrics["caption"]
        values = [
            f"{metrics['micro']['precision']:.4f}/{metrics['micro']['recall']:.4f}/{metrics['micro']['f1']:.4f}",
            f"{metrics['soft_micro']['f1']:.4f}",
            f"{metrics['macro_field_f1']:.4f}",
            f"{metrics['mean_field_exact']:.4f}",
            f"{metrics['unknown_extra_rate']:.2%}",
            f"{caption['average_words']:.2f}",
            f"{caption['length_18_24_ratio']:.2%}",
            f"{caption['background_keyword_ratio']:.3%}",
        ]
    return "| " + " | ".join([spec.label, replay, _status(spec), *values]) + " |"


def render_v4_block(project_root: Path) -> str:
    replay, person_only = experiment_specs(project_root)
    rows = [
        _result_row(replay, _load_metrics(replay)),
        _result_row(person_only, _load_metrics(person_only)),
    ]
    comparison = "两组结果尚未全部完成，流水线将在每个评估阶段结束后自动刷新本节。"
    replay_metrics = _load_metrics(replay)
    person_metrics = _load_metrics(person_only)
    if replay_metrics and person_metrics:
        replay_f1 = replay_metrics["micro"]["f1"]
        person_f1 = person_metrics["micro"]["f1"]
        winner = replay.label if replay_f1 >= person_f1 else person_only.label
        comparison = (
            f"两组均已完成。Micro F1更高的是{winner}；含replay与不含replay的差值为"
            f"{replay_f1 - person_f1:+.4f}。最终选择仍需同时检查Macro-field F1、"
            "unknown-extra、长度以及后续通用能力保持指标。"
        )
    return "\n".join(
        [
            "## V4方案C：串行全参训练",
            "",
            "两个run均从`pretrained/Florence-2-base`独立初始化，使用同一30k人物训练集；先运行含1k原生`<REGION_TO_DESCRIPTION>` replay版本，完整评估成功后再运行不含replay版本。任一阶段失败会终止流水线，不会跳过失败继续下一个run。",
            "",
            "| 配置 | V4 replay1k | V4 no-replay |",
            "|---|---:|---:|",
            "| 人物/replay样本 | 30,000 / 1,000 | 30,000 / 0 |",
            "| Global batch | 16 | 16 |",
            "| 每卡batch / 梯度累积 | 4 / 1 | 4 / 1 |",
            "| Epoch | 3 | 3 |",
            "| Optimizer steps | 5,814 | 5,625 |",
            "| Vision / Projection / Language LR | `2e-7` / `7.5e-7` / `1.25e-6` | `2e-7` / `7.5e-7` / `1.25e-6` |",
            "| Weight decay / Label smoothing | `0.015` / `0.05` | `0.015` / `0.05` |",
            "| Warmup / Cosine floor | 5% / 3% | 5% / 3% |",
            "| Eval interval | 200 steps | 200 steps |",
            "",
            "每个run的自动阶段固定为：`训练 → final测试集推理 → Qwen抽取 → 失败单条重试 → 指标计算 → 本文档回填`。测试集只用于训练完成后的冻结final报告，不参与checkpoint选择。",
            "",
            "| 实验 | Replay | 状态 | Micro P/R/F1 | Soft Micro F1 | Macro-field F1 | Mean exact | Unknown-extra | 平均词数 | 18-24词 | 背景关键词 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            comparison,
            "",
            f"- V4 replay1k指标：[`metrics.json`](../artifacts/sft/{replay.run_name}/test_inference_final/evaluation_qwen_final/metrics.json)",
            f"- V4 no-replay指标：[`metrics.json`](../artifacts/sft/{person_only.run_name}/test_inference_final/evaluation_qwen_final/metrics.json)",
            "- 串行状态：[`state.json`](../artifacts/v4_serial_pipeline/state.json)",
        ]
    )


def update_marker_block(path: Path, block: str) -> None:
    content = path.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        raise ValueError(f"V4 result markers are missing from {path}")
    before, remainder = content.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    updated = before + START_MARKER + "\n" + block.strip() + "\n" + END_MARKER + after
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(updated)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    document = root / "docs" / "FLORENCE_TRAINING_RUNS.md"
    update_marker_block(document, render_v4_block(root))
    print(document, flush=True)


if __name__ == "__main__":
    main()
