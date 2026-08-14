#!/usr/bin/env python3
"""Immutable V4 experiment specifications and subprocess commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/work/MichaelYu/miniconda3/envs/florence/bin/python")
TORCHRUN = Path("/data1/work/MichaelYu/miniconda3/envs/florence/bin/torchrun")


@dataclass(frozen=True)
class ExperimentSpec:
    project_root: Path
    run_name: str
    label: str
    person_only: bool
    total_samples: int
    optimizer_steps: int
    train_port: int
    infer_port: int

    @property
    def run_dir(self) -> Path:
        return self.project_root / "artifacts" / "sft" / self.run_name

    @property
    def inference_dir(self) -> Path:
        return self.run_dir / "test_inference_final"


@dataclass(frozen=True)
class StageSpec:
    experiment: str
    name: str
    command: Tuple[str, ...]
    environment: Dict[str, str]
    completion_path: Optional[Path]
    log_path: Path


def experiment_specs(project_root: Path = DEFAULT_PROJECT_ROOT) -> Tuple[ExperimentSpec, ExperimentSpec]:
    root = Path(project_root).resolve()
    return (
        ExperimentSpec(
            project_root=root,
            run_name="region_category_person_sft_30k_replay1k_b16_e3_lr125",
            label="V4 replay1k",
            person_only=False,
            total_samples=31000,
            optimizer_steps=5814,
            train_port=29710,
            infer_port=29711,
        ),
        ExperimentSpec(
            project_root=root,
            run_name="region_category_person_sft_30k_person_only_b16_e3_lr125",
            label="V4 no-replay",
            person_only=True,
            total_samples=30000,
            optimizer_steps=5625,
            train_port=29720,
            infer_port=29721,
        ),
    )


def training_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    command: List[str] = [
        str(TORCHRUN),
        "--standalone",
        "--nproc_per_node=4",
        f"--master_port={spec.train_port}",
        str(root / "scripts" / "train_person_attribute_sft.py"),
        "--train-data", str(root / "data" / "prepared" / "train.jsonl"),
        "--dev-data", str(root / "data" / "prepared" / "dev.jsonl"),
        "--replay-data", str(root / "data" / "prepared" / "native_replay.jsonl"),
        "--model-path", str(root / "pretrained" / "Florence-2-base"),
        "--output-root", str(root / "artifacts" / "sft"),
        "--run-name", spec.run_name,
        "--formal-global-batch-size", "16",
        "--per-device-batch-size", "4",
        "--gradient-accumulation-steps", "1",
        "--vision-lr", "2e-07",
        "--projection-lr", "7.5e-07",
        "--language-lr", "1.25e-06",
        "--weight-decay", "0.015",
        "--label-smoothing", "0.05",
        "--warmup-ratio", "0.05",
        "--min-lr-ratio", "0.03",
        "--eval-steps", "200",
        "--log-steps", "10",
        "--num-workers", "6",
        "--dev-num-workers", "0",
        "--prefetch-factor", "1",
        "--epochs", "3",
        "--max-optimizer-steps", str(spec.optimizer_steps),
    ]
    if spec.person_only:
        command.append("--person-only")
    return tuple(command)


def inference_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    return (
        str(TORCHRUN), "--standalone", "--nproc_per_node=4",
        f"--master_port={spec.infer_port}",
        str(root / "scripts" / "infer_person_attribute.py"),
        "--checkpoint", str(spec.run_dir / "final"),
        "--test-data", str(root / "data" / "prepared" / "test.jsonl"),
        "--output-dir", str(spec.inference_dir),
        "--batch-size", "4", "--max-new-tokens", "64",
    )


def extraction_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    return (
        str(PYTHON), str(root / "scripts" / "extract_qwen_attributes.py"),
        "--input", str(spec.inference_dir / "predictions.jsonl"),
        "--output", str(spec.inference_dir / "qwen_attribute_extraction.jsonl"),
        "--base-url", "http://127.0.0.1:6097/v1",
        "--model", "Qwen36-35b-caption",
        "--batch-size", "4", "--workers", "16",
        "--timeout", "180", "--retries", "2", "--max-tokens", "2048",
    )


def finalization_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    root = spec.project_root
    return (
        str(PYTHON), str(root / "scripts" / "finalize_qwen_attribute_evaluation.py"),
        "--predictions", str(spec.inference_dir / "predictions.jsonl"),
        "--extraction", str(spec.inference_dir / "qwen_attribute_extraction.jsonl"),
        "--output-dir", str(spec.inference_dir / "evaluation_qwen_final"),
        "--base-url", "http://127.0.0.1:6097/v1",
        "--model", "Qwen36-35b-caption",
    )


def document_command(spec: ExperimentSpec) -> Tuple[str, ...]:
    return (
        str(PYTHON),
        str(spec.project_root / "scripts" / "update_v4_training_runs.py"),
        "--project-root", str(spec.project_root),
    )


def stage_specs(spec: ExperimentSpec) -> Tuple[StageSpec, ...]:
    common_gpu_env = {
        "CUDA_VISIBLE_DEVICES": "3,4,5,6",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    log_root = spec.project_root / "artifacts" / "v4_serial_pipeline" / spec.run_name
    return (
        StageSpec(spec.run_name, "train", training_command(spec), common_gpu_env, spec.run_dir / "final" / "run_state.json", log_root / "train.log"),
        StageSpec(spec.run_name, "infer", inference_command(spec), common_gpu_env, spec.inference_dir / "manifest.json", log_root / "infer.log"),
        StageSpec(spec.run_name, "extract", extraction_command(spec), {"PYTHONUNBUFFERED": "1"}, spec.inference_dir / "qwen_attribute_extraction.manifest.json", log_root / "extract.log"),
        StageSpec(spec.run_name, "finalize", finalization_command(spec), {}, spec.inference_dir / "evaluation_qwen_final" / "metrics.json", log_root / "finalize.log"),
        StageSpec(spec.run_name, "document", document_command(spec), {}, None, log_root / "document.log"),
    )


def serial_stage_specs(project_root: Path = DEFAULT_PROJECT_ROOT) -> Tuple[StageSpec, ...]:
    stages: List[StageSpec] = []
    for experiment in experiment_specs(project_root):
        stages.extend(stage_specs(experiment))
    return tuple(stages)

