# RL Reward Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Qwen reward-service launch configuration, non-destructive service management, HTTP readiness probes, and coherent Florence RL training preflight.

**Architecture:** Keep immutable service metadata and Docker argv generation in an import-safe configuration module. Keep HTTP model/probe validation in the existing client module, use a small CLI for Docker orchestration, and call the same preflight from rank 0 before Florence model allocation.

**Tech Stack:** Python 3.10 standard library, Docker CLI, OpenAI-compatible vLLM HTTP API, PyTorch DDP integration, pytest/unittest.

---

### Task 1: Define Immutable Service Specifications

**Files:**
- Create: `scripts/rl_service_config.py`
- Create: `tests/test_rl_service_config.py`

- [x] Add failing tests for exact models, paths, GPUs, ports, images, context, concurrency, batch tokens, thinking defaults, and extractor Marlin environment.
- [x] Add failing tests for generated Docker argv: read-only model mount, exact GPU, port mapping, container name, vLLM flags, and absence of stop/remove/rm flags.
- [x] Run the focused test and confirm failure because the module is missing.
- [x] Implement frozen `ServiceSpec` objects and Docker argv generation without shell string execution.
- [x] Run focused tests and compile the module.

### Task 2: Add HTTP Model And OK Probes

**Files:**
- Modify: `scripts/rl_clients.py`
- Modify: `tests/test_rl_clients.py`

- [x] Add failing tests for model-list parsing, wrong-model rejection, OK/null-reasoning acceptance, and content/reasoning rejection.
- [x] Run focused tests and confirm the new helpers are missing.
- [x] Implement import-safe model-list and deterministic chat-probe helpers with existing timeout/retry conventions.
- [x] Run focused tests and compile the client.

### Task 3: Add Non-Destructive Service CLI

**Files:**
- Create: `scripts/run_rl_reward_services.py`
- Create: `tests/test_run_rl_reward_services.py`

- [x] Add failing tests for role selection, print-command escaping, existing-container refusal, missing-container start argv, status failure propagation, and lack of destructive commands.
- [x] Run focused tests and confirm failure because the CLI module is missing.
- [x] Implement `print-command`, `status`, and `start` with injected subprocess/probe functions for unit testing and condition-based readiness timeout.
- [x] Run focused tests, compile, and print both commands without executing them.

### Task 4: Integrate Training Preflight And Provenance

**Files:**
- Modify: `scripts/train_person_attribute_rl.py`
- Modify: `tests/test_train_person_attribute_rl.py`

- [x] Add failing tests that lexical checks only extractor, qwen checks both services, failures are returned for rank broadcast, skip mode is explicit, and service specs appear in invariants.
- [x] Run focused tests and confirm failures against the current trainer.
- [x] Implement pure route selection, rank-0 preflight result, DDP broadcast/abort before Florence loading, skip CLI validation, and immutable service metadata.
- [x] Run focused RL and SFT-helper tests and compile the trainer.

### Task 5: Document And Verify The Full RL Chain

**Files:**
- Modify: `EXPERIMENT_PLAN.md`
- Modify: `docs/FLORENCE_TRAINING_RUNS.md`

- [x] Document exact service configurations, preflight behavior, launch/status commands, and skip semantics.
- [x] Run the full Python 3.13 test suite and the Florence RL/SFT critical suite (the workspace exposes Python 3.13; all import-safe and trainer helper tests pass there).
- [x] Run compile checks, command printing, current service status, formal data-only validation, CLI help inspection, and process checks.
- [x] Confirm 6097/6098 status precisely and confirm no Florence RL training or service replacement was started.

The workspace is not a Git repository, so commit steps are omitted.
