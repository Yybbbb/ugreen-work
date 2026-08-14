#!/usr/bin/env python3
"""Immutable vLLM service contracts for Florence RL rewards."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ServiceSpec:
    role: str
    model: str
    model_path: Path
    gpu: int
    host_port: int
    image: str
    container_name: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    vllm_version: Optional[str] = None
    swap_space_gib: Optional[int] = None
    environment: Tuple[Tuple[str, str], ...] = ()
    container_port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}/v1"


EXTRACTOR_SERVICE = ServiceSpec(
    role="extractor",
    model="Qwen3.6-27B-FP8",
    model_path=Path("/data1/work/MichaelYu/models/Qwen3.6-27B-FP8"),
    gpu=1,
    host_port=6097,
    image="docker.m.daocloud.io/vllm/vllm-openai:v0.19.1",
    container_name="vllm-qwen36-27b-fp8",
    vllm_version="0.19.1",
    max_model_len=65536,
    max_num_seqs=64,
    max_num_batched_tokens=8192,
    environment=(("VLLM_TEST_FORCE_FP8_MARLIN", "1"),),
)

JUDGE_SERVICE = ServiceSpec(
    role="judge",
    model="Qwen3.5-4B",
    model_path=Path("/data1/work/MichaelYu/models/Qwen3.5-4B"),
    gpu=2,
    host_port=6098,
    image="docker.m.daocloud.io/vllm/vllm-openai:qwen3_5",
    container_name="vllm-qwen35-4b",
    max_model_len=32768,
    max_num_seqs=128,
    max_num_batched_tokens=65536,
    swap_space_gib=8,
)

SERVICES = (EXTRACTOR_SERVICE, JUDGE_SERVICE)


def service_for_role(role: str) -> ServiceSpec:
    for spec in SERVICES:
        if spec.role == role:
            return spec
    raise ValueError(f"unknown reward service role: {role!r}")


def docker_run_argv(spec: ServiceSpec) -> list[str]:
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        spec.container_name,
        "--gpus",
        f"device={spec.gpu}",
        "--publish",
        f"{spec.host_port}:{spec.container_port}",
        "--volume",
        f"{spec.model_path}:/models:ro",
    ]
    for key, value in spec.environment:
        argv.extend(["--env", f"{key}={value}"])
    argv.extend(
        [
            spec.image,
            "/models",
            "--served-model-name",
            spec.model,
            "--port",
            str(spec.container_port),
            "--tensor-parallel-size",
            str(spec.tensor_parallel_size),
            "--language-model-only",
            "--max-model-len",
            str(spec.max_model_len),
            "--max-num-seqs",
            str(spec.max_num_seqs),
            "--max-num-batched-tokens",
            str(spec.max_num_batched_tokens),
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--reasoning-parser",
            "qwen3",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
        ]
    )
    if spec.swap_space_gib is not None:
        argv.extend(["--swap-space", str(spec.swap_space_gib)])
    argv.extend(["--gpu-memory-utilization", str(spec.gpu_memory_utilization)])
    return argv


def service_metadata(spec: ServiceSpec) -> Dict[str, Any]:
    return {
        "role": spec.role,
        "model": spec.model,
        "model_path": str(spec.model_path),
        "gpu": spec.gpu,
        "base_url": spec.base_url,
        "host_port": spec.host_port,
        "container_port": spec.container_port,
        "image": spec.image,
        "container_name": spec.container_name,
        "vllm_version": spec.vllm_version,
        "max_model_len": spec.max_model_len,
        "max_num_seqs": spec.max_num_seqs,
        "max_num_batched_tokens": spec.max_num_batched_tokens,
        "tensor_parallel_size": spec.tensor_parallel_size,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "swap_space_gib": spec.swap_space_gib,
        "enable_thinking": False,
        "environment": dict(spec.environment),
    }
