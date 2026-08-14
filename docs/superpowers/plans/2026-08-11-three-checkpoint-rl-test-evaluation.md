# Three-Checkpoint RL Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a resumable, frozen-test comparison of V4B SFT, Qwen RL, and lexical RL using the approved eight metric groups.

**Architecture:** A pure Python metrics module owns reproducible definitions, paired session bootstrap, and recommendation logic. A second CLI owns artifact validation and service-backed extraction/judging, while a shell runner serializes inference/evaluation and waits for the still-running lexical checkpoint without competing for GPUs.

**Tech Stack:** Python 3.10 standard library, existing Florence/PyTorch inference, existing Qwen OpenAI-compatible clients, unittest/pytest, Bash.

---

### Task 1: Pure eight-metric scorer

**Files:**
- Create: `scripts/rl_test_metrics.py`
- Create: `tests/test_rl_test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Cover lexical micro P/R/F1, macro-field F1, fabrication denominator, structure,
length, background, invalid format, semantic similarity injection, and empty
denominators with small real rows.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rl_test_metrics.py -q`

Expected: FAIL because `scripts/rl_test_metrics.py` does not exist.

- [ ] **Step 3: Implement minimal pure scorer**

Reuse canonical normalization and frozen regular expressions from
`evaluate_qwen_attribute_extraction.py`, and matching classification semantics
from `rl_reward.py`. Return per-sample sufficient statistics plus aggregate
eight-metric groups without network or torch imports.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_rl_test_metrics.py -q`

Expected: PASS.

### Task 2: Bootstrap and recommendation

**Files:**
- Modify: `scripts/rl_test_metrics.py`
- Modify: `tests/test_rl_test_metrics.py`

- [ ] **Step 1: Write failing comparison tests**

Assert deterministic session bootstrap, paired delta intervals, V4B-relative
gates, joint-F1 ranking, tie behavior, and diagnostic win/tie/loss ordering.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rl_test_metrics.py -q`

Expected: FAIL because comparison/report APIs are absent.

- [ ] **Step 3: Implement comparison and rendering**

Add seeded 2,000-replicate paired bootstrap, selection gates, field deltas,
machine-readable JSON payload, and a concise Markdown table showing exactly
the three checkpoints and eight approved metric groups.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_rl_test_metrics.py -q`

Expected: PASS.

### Task 3: Resumable evaluation CLI

**Files:**
- Create: `scripts/evaluate_person_attribute_rl.py`
- Create: `tests/test_evaluate_person_attribute_rl.py`

- [ ] **Step 1: Write failing orchestration tests**

Test ordered-ID validation, checkpoint/decoding validation for prediction
reuse, extraction resume, judge-cache persistence, atomic outputs, and refusal
to publish with service failures.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_evaluate_person_attribute_rl.py -q`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the CLI**

Use `extract_qwen_attributes.request_batch` for bounded concurrent extraction
with the locked 27B endpoint/model. Deduplicate semantic comparisons, evaluate
them concurrently through `rl_clients.make_judge_fn`, preserve source order,
persist caches, and write per-candidate metrics/details atomically.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_evaluate_person_attribute_rl.py -q`

Expected: PASS.

### Task 4: Serial end-to-end runner

**Files:**
- Create: `scripts/run_person_attribute_rl_evaluation.sh`
- Create: `tests/test_run_person_attribute_rl_evaluation.py`
- Modify: `EXPERIMENT_PLAN.md`
- Modify: `docs/FLORENCE_TRAINING_RUNS.md`

- [ ] **Step 1: Write failing runner-contract tests**

Assert fixed paths, endpoints/models, GPU 3-6 inference, wait-for-final behavior,
V4B reuse checks, three serial candidates, and final report paths.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_run_person_attribute_rl_evaluation.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement runner and documentation**

Make each stage idempotent and log to `artifacts/rl/evaluation/runner.log`.
Wait for lexical `final/model.safetensors`, then run distributed inference and
candidate evaluation serially. Record the approved eight-metric protocol and
artifact locations in both project documents.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_run_person_attribute_rl_evaluation.py -q`

Expected: PASS.

### Task 5: Verification and launch

**Files:**
- Modify only if a failing verification exposes a scoped defect.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_rl_test_metrics.py tests/test_evaluate_person_attribute_rl.py tests/test_run_person_attribute_rl_evaluation.py -q`

Expected: PASS.

- [ ] **Step 2: Run full regression suite**

Run: `/data1/work/MichaelYu/miniconda3/envs/florence/bin/python -m pytest tests -q`

Expected: all tests PASS.

- [ ] **Step 3: Run a service-backed smoke evaluation**

Use existing V4B predictions restricted to a few test rows and both locked
services; verify zero unresolved requests, stable input order, and complete JSON
and Markdown output.

- [ ] **Step 4: Launch the resumable formal runner**

Start `scripts/run_person_attribute_rl_evaluation.sh` in the background. It must
wait while lexical RL owns GPU 3-6, then continue automatically and write its
PID and live log under `artifacts/rl/evaluation/`.

- [ ] **Step 5: Report live state**

Report whether the runner is waiting, inferring, extracting, judging, or
complete, along with exact artifact and log paths.
