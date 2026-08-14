# RL Reward Services Design

## Goal

Make the two RL reward services reproducible and prevent formal Florence RL
training from connecting to missing or incorrectly served models.

This change adds service configuration, command generation, status/start
commands, and training preflight checks. Implementation verification must not
restart, remove, or replace an existing container.

## Service Contracts

### Attribute Extractor

- role: caption-to-fixed-schema attribute extraction;
- model: `Qwen3.6-27B-FP8`;
- local model path: `/data1/work/MichaelYu/models/Qwen3.6-27B-FP8`;
- GPU: physical GPU 1;
- host endpoint: `http://127.0.0.1:6097/v1`;
- container port: 8000;
- image: `docker.m.daocloud.io/vllm/vllm-openai:v0.19.1`;
- vLLM version: `0.19.1`;
- language-model-only mode;
- default chat template argument: `{"enable_thinking": false}`;
- expected probe response: `content == "OK"` and no reasoning content;
- maximum model length: 65,536;
- maximum concurrent sequences: 64;
- maximum batched tokens: 8,192;
- chunked prefill and prefix caching enabled;
- dtype: bfloat16;
- FP8 kernel override: `VLLM_TEST_FORCE_FP8_MARLIN=1`, forcing FP8 Marlin on
  Ada instead of block-FP8 Triton;
- tensor parallel size: 1;
- GPU memory utilization: 0.90.

### Value Judge

- role: five-level semantic attribute-value similarity judging;
- model: `Qwen3.5-4B`;
- local model path: `/data1/work/MichaelYu/models/Qwen3.5-4B`;
- GPU: physical GPU 2;
- host endpoint: `http://127.0.0.1:6098/v1`;
- container port: 8000;
- image: `docker.m.daocloud.io/vllm/vllm-openai:qwen3_5`;
- language-model-only mode;
- default chat template argument: `{"enable_thinking": false}`;
- expected probe response: `content == "OK"` and no reasoning content;
- maximum model length: 32,768;
- maximum concurrent sequences: 128;
- maximum batched tokens: 65,536, matching the current verified service;
- chunked prefill and prefix caching enabled;
- dtype: bfloat16;
- tensor parallel size: 1;
- swap space: 8 GiB;
- GPU memory utilization: 0.90.

## Configuration Module

`scripts/rl_service_config.py` defines an immutable service specification for
each role and generates Docker argv as a list. It does not construct a shell
command string for execution.

Generated Docker commands:

- use stable container names `vllm-qwen36-27b-fp8` and `vllm-qwen35-4b`;
- expose only the specified host port;
- mount the local model directory read-only at `/models`;
- request exactly the configured physical GPU;
- set service-specific environment variables;
- pass the complete vLLM arguments from the service contract.

The same objects provide endpoint, served model name, GPU, image, vLLM version,
and capacity metadata for training provenance.

## Service CLI

`scripts/run_rl_reward_services.py` provides:

- `print-command [extractor|judge|all]`: print shell-escaped commands for
  inspection without executing them;
- `status [extractor|judge|all]`: query Docker container state, `/v1/models`,
  and the `OK` chat probe, returning nonzero if any selected check fails;
- `start [extractor|judge|all]`: start missing containers with generated argv,
  then wait until model and probe checks pass.

`start` is non-destructive. If a configured container name already exists, it
does not stop, remove, recreate, or mutate it. It validates the existing
container and fails with a diagnostic if the service is unhealthy or serves a
different model. There is no stop, remove, or replace command.

The readiness wait is condition-based with a configurable timeout. Docker
process failures, timeout, wrong served model, non-OK content, or nonempty
reasoning produce a nonzero exit and a concrete diagnostic.

## Client Preflight

`scripts/rl_clients.py` adds import-safe HTTP helpers to:

1. query `/v1/models` and require the expected model id;
2. send a deterministic chat request asking for exactly `OK` with thinking
   disabled and temperature zero;
3. require stripped content `OK`;
4. require `reasoning`, `reasoning_content`, or equivalent returned reasoning
   fields to be absent or null.

The helpers reuse the existing retry/timeout behavior and do not import torch,
Docker, or vLLM.

## Training Integration

Before loading Florence weights, formal training performs reward-service
preflight on rank 0:

- lexical route: extractor only;
- qwen route: extractor and judge.

Rank 0 broadcasts success or the error text. Every rank aborts coherently on a
failure, before model allocation and DDP training. Non-formal smoke runs use the
same check unless `--skip-reward-service-preflight` is explicitly supplied.

The immutable run configuration records both complete service specifications,
the selected route, and whether preflight was skipped. Endpoint/model CLI
overrides remain supported, but preflight validates the overridden values.

## Tests And Verification

Tests cover:

- exact extractor and judge service constants;
- Docker argv, GPU assignment, read-only mounts, ports, model names, capacity
  flags, thinking defaults, and extractor Marlin environment;
- no destructive Docker subcommands;
- refusal to replace an existing named container;
- `/v1/models` parsing and wrong-model rejection;
- `OK` content with null reasoning acceptance;
- non-OK content and nonempty reasoning rejection;
- lexical versus qwen preflight routing;
- service specifications in training invariants.

Verification runs unit tests, Python compile checks, command printing, and
status against currently running services. It does not execute `start` while a
configured container already exists and does not start Florence RL training.
