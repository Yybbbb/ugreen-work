# Reviewed Five-Run SFT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild reviewed-test Qwen attributes and run the reviewed V1-V4B Florence training, inference, and evaluation pipeline with the requested GPU overlap and historical output paths.

**Architecture:** A declarative five-run configuration preserves every historical hyperparameter. One durable orchestrator owns preflight, recoverable archive, a serial Florence worker, and an ordered Qwen worker joined through inference-completion events; atomic state and output publication make restart behavior explicit.

**Tech Stack:** Python 3.10, unittest/pytest, PyTorch torchrun, existing Florence SFT/inference scripts, OpenAI-compatible vLLM HTTP API.

---

### Task 1: Lock the five experiment contracts

**Files:**
- Create: `scripts/reviewed_sft_pipeline_config.py`
- Create: `tests/test_reviewed_sft_pipeline.py`

- [ ] Write tests asserting the exact run names, V1 legacy inference directory, optimizer steps, batch/accumulation, epochs, learning rates, decay, eval interval, replay mode, ports, prepared data paths, and GPU environment.
- [ ] Run `miniconda3/envs/florence/bin/python -m pytest tests/test_reviewed_sft_pipeline.py -q` and verify the missing module fails.
- [ ] Implement immutable experiment and command specifications for V1, V2, V3, V4A, and V4B.
- [ ] Run the focused tests and verify all command-contract assertions pass.

### Task 2: Make GT-caption extraction publish a fresh complete dataset

**Files:**
- Modify: `scripts/extract_qwen_gt_attributes.py`
- Create: `tests/test_extract_qwen_gt_attributes.py`

- [ ] Write tests proving blank/whitespace captions are rejected, successful output is ordered by the reviewed test input, stale output IDs are not reused in fresh mode, and publication plus manifest replacement are atomic.
- [ ] Run the focused test and verify failures expose the current append/resume behavior.
- [ ] Add input validation, `--fresh`, staged output, complete-ID validation, and atomic JSONL/manifest publication while retaining explicit resume support for pipeline-owned staging files.
- [ ] Run the focused test and verify all extraction publication tests pass.

### Task 3: Implement reviewed-data preflight and recoverable archive

**Files:**
- Create: `scripts/run_reviewed_sft_pipeline.py`
- Modify: `tests/test_reviewed_sft_pipeline.py`

- [ ] Write tests for metadata/count/hash verification, empty-caption refusal before mutation, exact old-run archive targets, Qwen GT backup, and archive idempotence on restart.
- [ ] Run the focused tests and verify the orchestration helpers are initially missing.
- [ ] Implement read-only preflight first, then same-filesystem timestamped moves recorded atomically in pipeline state.
- [ ] Run the focused tests and verify a simulated restart never archives newly created reviewed outputs.

### Task 4: Implement overlapping Florence and Qwen workers

**Files:**
- Modify: `scripts/run_reviewed_sft_pipeline.py`
- Modify: `tests/test_reviewed_sft_pipeline.py`

- [ ] Write deterministic fake-runner tests proving Qwen GT and V1 start concurrently, Florence order is train/infer per run, the next train starts without waiting for the previous Qwen evaluation, and dependent stages stop on failure.
- [ ] Run the focused tests and verify the scheduling tests fail before implementation.
- [ ] Implement thread workers, inference completion events, Qwen-ready health wait, subprocess logging, checkpoint resume discovery, completion-path validation, and atomic stage state transitions.
- [ ] Run the focused tests and verify all scheduling and failure-path assertions pass.

### Task 5: Add Qwen-GT evaluation and final documentation stage

**Files:**
- Modify: `scripts/reviewed_sft_pipeline_config.py`
- Modify: `scripts/run_reviewed_sft_pipeline.py`
- Modify: `scripts/update_v4_training_runs.py`
- Modify: `tests/test_reviewed_sft_pipeline.py`
- Modify: `tests/test_update_v4_training_runs.py`

- [ ] Write tests asserting each run executes prediction extraction, finalization, and `evaluate_with_qwen_gt.py`, with expected sample count taken from sanitized test preflight rather than a hard-coded stale count.
- [ ] Write document tests that reject stale dataset hashes and render refreshed metrics only after all five reviewed runs complete.
- [ ] Implement the Qwen-GT command and generalize document refresh to the five active run paths while preserving surrounding prose.
- [ ] Run focused tests for pipeline and documentation updates.

### Task 6: Verify and launch

**Files:**
- Runtime output: `artifacts/reviewed_five_run_pipeline/`

- [ ] Run all directly affected unit tests with the Florence Python environment.
- [ ] Run the existing full unit suite in its compatible environments and record any pre-existing failures separately.
- [ ] Run `python scripts/run_reviewed_sft_pipeline.py --dry-run` and inspect all five commands, exact active/backup paths, dataset hashes, and stage graph.
- [ ] Verify `http://127.0.0.1:6097/v1/models`, GPUs 1-6, available disk, and absence of competing Florence torchrun processes.
- [ ] Start the launcher detached, recording PID and start command under `artifacts/reviewed_five_run_pipeline/`.
- [ ] Poll state, logs, processes, and GPU memory until both `qwen_gt:extract` and `V1:train` are confirmed running.

The workspace is not a Git repository, so this plan intentionally omits commit steps; verification artifacts and the recoverable old-run backup provide the operational audit trail.
