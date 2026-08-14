# RL Distributed Dev Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent long Qwen dev reward evaluation from timing out DDP and reduce evaluation latency by using all four Florence ranks and concurrent judge requests.

**Architecture:** Add pure indexed sharding/merge helpers to the RL trainer, gather structured rank results before rank 0 writes artifacts, and add bounded cache-prefetch concurrency to the Qwen judge callable. Configure a 60-minute process-group timeout as defense in depth.

**Tech Stack:** Python 3.10, PyTorch DDP/NCCL, standard-library `ThreadPoolExecutor`, OpenAI-compatible vLLM HTTP, pytest.

---

### Task 1: Concurrent Judge Prefetch

**Files:**
- Modify: `scripts/rl_clients.py`
- Modify: `scripts/train_person_attribute_rl.py`
- Test: `tests/test_rl_clients.py`
- Test: `tests/test_train_person_attribute_rl.py`

- [x] Add failing tests proving comparison pairs are deduplicated and Qwen prefetch executes more than one request concurrently while populating the normal similarity cache.
- [x] Run focused tests and confirm failure because prefetch is absent.
- [x] Implement bounded `ThreadPoolExecutor` prefetch and pair enumeration for fixed fields and `extra` values.
- [x] Run focused reward/client tests and compile both scripts.

### Task 2: Distributed Ordered Dev Reward

**Files:**
- Modify: `scripts/train_person_attribute_rl.py`
- Test: `tests/test_train_person_attribute_rl.py`

- [x] Add failing tests for round-robin indexed shards, ordered merge, missing/duplicate rejection, and rank error propagation.
- [x] Run focused tests and confirm the helpers are absent.
- [x] Implement local evaluation payloads, `gather_object`, rank-0 validation/order restoration, and coherent broadcast failure.
- [x] Replace rank-0-only reward evaluation with the distributed helper and keep rank 0 as the only artifact writer.
- [x] Run focused trainer tests.

### Task 3: Timeout, Provenance, And Restart Verification

**Files:**
- Modify: `scripts/train_person_attribute_rl.py`
- Modify: `scripts/run_person_attribute_rl_serial.sh`
- Test: `tests/test_train_person_attribute_rl.py`

- [x] Add failing tests for 60-minute timeout and judge concurrency CLI/invariants.
- [x] Initialize NCCL with the configured `datetime.timedelta` and record both settings in run invariants.
- [x] Run the full test suite, compile checks, and service preflight.
- [x] Run a four-GPU one-step real dev-evaluation smoke using reduced sample counts.
- [x] Archive the failed formal run without deletion and restart Qwen then lexical from V4B on GPUs 3-6.
