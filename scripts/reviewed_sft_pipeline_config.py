#!/usr/bin/env python3
"""Immutable contracts for the reviewed-data V1 through V4B reruns."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/work/MichaelYu/miniconda3/envs/florence/bin/python")
TORCHRUN = Path("/data1/work/MichaelYu/miniconda3/envs/florence/bin/torchrun")
QWEN_BASE_URL = "http://127.0.0.1:6097/v1"
QWEN_MODEL = "Qwen36-35b-caption"


@dataclass(frozen=True)
class ExperimentSpec:
    project_root: Path
    label: str
    run_name: str
    global_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    optimizer_steps: int
    person_only: bool
    vision_lr: str
    projection_lr: str
    language_lr: str
    weight_decay: str
    min_lr_ratio: str
    eval_steps: int
    train_port: int
    infer_port: int
    inference_name: str = "test_inference_final"

    @property
    def run_dir(self) -> Path:
        return self.project_root / "artifacts" / "sft" / self.run_name

    @property
    def inference_dir(self) -> Path:
        return self.run_dir / self.inference_name

    @property
    def evaluation_dir(self) -> Path:
        return self.inference_dir / "evaluation_qwen_final"


@dataclass(frozen=True)
class StageSpec:
    experiment: str
    name: str
    command: Tuple[str, ...]
    environment: Dict[str, str]
    completion_path: Optional[Path]
    log_path: Path


def experiment_specs(project_root: Path = DEFAULT_PROJECT_ROOT) -> Tuple[ExperimentSpec, ...]:
    root = Path(project_root).resolve()
    common = {
        "project_root": root,
        "person_only": False,
        "vision_lr": "1e-07",
        "projection_lr": "5e-07",
        "language_lr": "1e-06",
        "weight_decay": "0.01",
        "min_lr_ratio": "0.1",
    }
    return (
        ExperimentSpec(**common, label="V1", run_name="region_category_person_sft_30k_replay1k_b64", global_batch_size=64, gradient_accumulation_steps=4, epochs=1, optimizer_steps=485, eval_steps=50, train_port=29810, infer_port=29811, inference_name="test_inference_final_v2"),
        ExperimentSpec(**common, label="V2", run_name="region_category_person_sft_30k_replay1k_b32", global_batch_size=32, gradient_accumulation_steps=2, epochs=1, optimizer_steps=969, eval_steps=100, train_port=29820, infer_port=29821),
        ExperimentSpec(**common, label="V3", run_name="region_category_person_sft_30k_replay1k_b32_e1p5", global_batch_size=32, gradient_accumulation_steps=2, epochs=2, optimizer_steps=1454, eval_steps=100, train_port=29830, infer_port=29831),
        ExperimentSpec(project_root=root, label="V4A", run_name="region_category_person_sft_30k_replay1k_b16_e3_lr125", global_batch_size=16, gradient_accumulation_steps=1, epochs=3, optimizer_steps=5814, person_only=False, vision_lr="2e-07", projection_lr="7.5e-07", language_lr="1.25e-06", weight_decay="0.015", min_lr_ratio="0.03", eval_steps=200, train_port=29840, infer_port=29841),
        ExperimentSpec(project_root=root, label="V4B", run_name="region_category_person_sft_30k_person_only_b16_e3_lr125", global_batch_size=16, gradient_accumulation_steps=1, epochs=3, optimizer_steps=5625, person_only=True, vision_lr="2e-07", projection_lr="7.5e-07", language_lr="1.25e-06", weight_decay="0.015", min_lr_ratio="0.03", eval_steps=200, train_port=29850, infer_port=29851),
    )


def training_environment() -> Dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "3,4,5,6",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }


def training_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    command: List[str] = [
        str(TORCHRUN), "--standalone", "--nproc_per_node=4", f"--master_port={spec.train_port}",
        str(root / "scripts" / "train_person_attribute_sft.py"),
        "--train-data", str(root / "data" / "prepared" / "train.jsonl"),
        "--dev-data", str(root / "data" / "prepared" / "dev.jsonl"),
        "--replay-data", str(root / "data" / "prepared" / "native_replay.jsonl"),
        "--model-path", str(root / "pretrained" / "Florence-2-base"),
        "--output-root", str(root / "artifacts" / "sft"),
        "--run-name", spec.run_name,
        "--formal-global-batch-size", str(spec.global_batch_size),
        "--per-device-batch-size", "4",
        "--gradient-accumulation-steps", str(spec.gradient_accumulation_steps),
        "--vision-lr", spec.vision_lr,
        "--projection-lr", spec.projection_lr,
        "--language-lr", spec.language_lr,
        "--weight-decay", spec.weight_decay,
        "--label-smoothing", "0.05",
        "--warmup-ratio", "0.05",
        "--min-lr-ratio", spec.min_lr_ratio,
        "--eval-steps", str(spec.eval_steps),
        "--log-steps", "10",
        "--num-workers", "6",
        "--dev-num-workers", "0",
        "--prefetch-factor", "1",
        "--epochs", str(spec.epochs),
        "--max-optimizer-steps", str(spec.optimizer_steps),
    ]
    if spec.person_only:
        command.append("--person-only")
    return tuple(command)


def inference_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    return (
        str(TORCHRUN), "--standalone", "--nproc_per_node=4", f"--master_port={spec.infer_port}",
        str(root / "scripts" / "infer_person_attribute.py"),
        "--checkpoint", str(spec.run_dir / "final"),
        "--test-data", str(root / "data" / "prepared" / "test.jsonl"),
        "--output-dir", str(spec.inference_dir),
        "--batch-size", "4", "--max-new-tokens", "64",
    )


def gt_stage(project_root: Path = DEFAULT_PROJECT_ROOT) -> StageSpec:
    root = Path(project_root).resolve()
    output = root / "data" / "prepared" / "test_qwen_attributes.jsonl"
    return StageSpec(
        experiment="qwen_gt",
        name="extract",
        command=(
            str(PYTHON), str(root / "scripts" / "extract_qwen_gt_attributes.py"),
            "--input", str(root / "data" / "prepared" / "test.jsonl"),
            "--output", str(output), "--base-url", QWEN_BASE_URL,
            "--model", QWEN_MODEL, "--batch-size", "4", "--workers", "16",
            "--timeout", "180", "--retries", "2", "--max-tokens", "2048", "--fresh",
        ),
        environment={"PYTHONUNBUFFERED": "1"},
        completion_path=output.with_suffix(".manifest.json"),
        log_path=root / "artifacts" / "reviewed_five_run_pipeline" / "qwen_gt.log",
    )


def florence_stage_commands(spec: ExperimentSpec) -> Tuple[StageSpec, StageSpec]:
    log_root = spec.project_root / "artifacts" / "reviewed_five_run_pipeline" / spec.run_name
    environment = training_environment()
    return (
        StageSpec(spec.run_name, "train", training_command(spec), environment, spec.run_dir / "final" / "run_state.json", log_root / "train.log"),
        StageSpec(spec.run_name, "infer", inference_command(spec), environment, spec.inference_dir / "manifest.json", log_root / "infer.log"),
    )


def qwen_stage_commands(spec: ExperimentSpec, expected_samples: int) -> Tuple[StageSpec, ...]:
    root = spec.project_root
    log_root = root / "artifacts" / "reviewed_five_run_pipeline" / spec.run_name
    extraction = spec.inference_dir / "qwen_attribute_extraction.jsonl"
    return (
        StageSpec(
            spec.run_name, "extract",
            (str(PYTHON), str(root / "scripts" / "extract_qwen_attributes.py"),
             "--input", str(spec.inference_dir / "predictions.jsonl"),
             "--output", str(extraction), "--base-url", QWEN_BASE_URL,
             "--model", QWEN_MODEL, "--batch-size", "4", "--workers", "16",
             "--timeout", "180", "--retries", "2", "--max-tokens", "2048"),
            {"PYTHONUNBUFFERED": "1"}, extraction.with_suffix(".manifest.json"), log_root / "extract.log",
        ),
        StageSpec(
            spec.run_name, "finalize",
            (str(PYTHON), str(root / "scripts" / "finalize_qwen_attribute_evaluation.py"),
             "--predictions", str(spec.inference_dir / "predictions.jsonl"),
             "--extraction", str(extraction), "--output-dir", str(spec.evaluation_dir),
             "--base-url", QWEN_BASE_URL, "--model", QWEN_MODEL,
             "--expected-samples", str(expected_samples)),
            {"PYTHONUNBUFFERED": "1"}, spec.evaluation_dir / "metrics.json", log_root / "finalize.log",
        ),
        StageSpec(
            spec.run_name, "qwen_gt",
            (str(PYTHON), str(root / "scripts" / "evaluate_with_qwen_gt.py"),
             "--run-dir", str(spec.evaluation_dir),
             "--gt-file", str(root / "data" / "prepared" / "test_qwen_attributes.jsonl")),
            {"PYTHONUNBUFFERED": "1"}, spec.evaluation_dir / "metrics_qwen_gt.json", log_root / "qwen_gt.log",
        ),
    )
