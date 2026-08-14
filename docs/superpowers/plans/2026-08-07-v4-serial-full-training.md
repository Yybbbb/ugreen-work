# V4 Serial Full-Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two independent V4 full-parameter Florence2 trainings serially, first with 1,000 native replay samples and then without replay, automatically continuing through test inference, Qwen attribute extraction, metric computation, and `FLORENCE_TRAINING_RUNS.md` publication.

**Architecture:** A Python orchestrator owns the ordered stages and records atomic state after each subprocess succeeds. Existing training, inference, Qwen extraction, and evaluation scripts remain the stage implementations; a focused finalizer retries failed Qwen rows and a focused document updater renders deterministic V4 result tables between Markdown markers.

**Tech Stack:** Python 3.10 standard library, PyTorch `torchrun`, Florence2 training/inference scripts, OpenAI-compatible vLLM Qwen service, `unittest`, Markdown.

---

### Task 1: Freeze the V4 experiment contracts

**Files:**
- Create: `scripts/v4_serial_config.py`
- Test: `tests/test_v4_serial_pipeline.py`

- [ ] Write tests asserting the replay run uses 31,000 samples, global batch 16, 3 epochs, 5,814 optimizer steps, and no `--person-only` flag.
- [ ] Write tests asserting the person-only run uses 30,000 samples, global batch 16, 3 epochs, 5,625 optimizer steps, and includes `--person-only`.
- [ ] Write tests asserting both runs start from `pretrained/Florence-2-base` and use vision/projection/language learning rates `2e-7/7.5e-7/1.25e-6`, weight decay `0.015`, label smoothing `0.05`, warmup `0.05`, cosine floor `0.03`, and evaluation every 200 optimizer steps.
- [ ] Run `python -m unittest tests/test_v4_serial_pipeline.py -q` and verify the tests fail because the config module is absent.
- [ ] Implement immutable run specifications and command builders in `scripts/v4_serial_config.py`.
- [ ] Run the targeted test and verify it passes.

### Task 2: Automate Qwen failure retry and final evaluation

**Files:**
- Create: `scripts/finalize_qwen_attribute_evaluation.py`
- Test: `tests/test_finalize_qwen_attribute_evaluation.py`
- Reuse: `scripts/extract_qwen_attributes.py`
- Reuse: `scripts/evaluate_qwen_attribute_extraction.py`

- [ ] Write tests showing successful extraction rows win over failed duplicates, only unresolved IDs enter `qwen_retry_input.jsonl`, and original plus successful retries produce one final row per test sample.
- [ ] Run the targeted test and verify it fails because the finalizer is absent.
- [ ] Implement streaming source lookup, retry-input generation, single-row Qwen retry, merged extraction publication, and invocation of the existing evaluator.
- [ ] Require `samples=4328` and `extractor_failures=0` before returning success.
- [ ] Run the targeted test and verify it passes.

### Task 3: Render V4 results into the training record

**Files:**
- Create: `scripts/update_v4_training_runs.py`
- Test: `tests/test_update_v4_training_runs.py`
- Modify: `docs/FLORENCE_TRAINING_RUNS.md`

- [ ] Write tests for rendering pending, partial, and complete two-run result tables from pipeline state and `metrics.json` files.
- [ ] Run the targeted test and verify it fails because the updater is absent.
- [ ] Add stable `<!-- V4_RESULTS_START -->` and `<!-- V4_RESULTS_END -->` markers to the document.
- [ ] Implement atomic marker-block replacement containing configuration, stage status, strict/soft Micro F1, Macro-field F1, exact rate, unknown-extra, length, and background metrics.
- [ ] Run the targeted test and verify it passes.

### Task 4: Implement the durable serial pipeline

**Files:**
- Create: `scripts/run_v4_serial_pipeline.py`
- Test: `tests/test_v4_serial_pipeline.py`

- [ ] Write tests asserting the stage order is `train → infer → extract → finalize → document` for the replay run, followed by the same order for the person-only run.
- [ ] Write tests asserting a failed stage stops the pipeline and a completed artifact is skipped on restart.
- [ ] Implement subprocess execution with explicit `CUDA_VISIBLE_DEVICES=3,4,5,6` for training/inference, Qwen endpoint health validation for extraction, per-stage logs, atomic pipeline state, and `check=True` failure propagation.
- [ ] Use unique ports for each four-GPU stage and preserve the parent process under `nohup setsid`.
- [ ] Run the targeted test and verify it passes.

### Task 5: Verify and launch

**Files:**
- Verify: `scripts/run_v4_serial_pipeline.py`
- Verify: `docs/FLORENCE_TRAINING_RUNS.md`

- [ ] Run all V4 targeted tests plus existing trainer, inference, extractor-evaluation tests.
- [ ] Run `python -m py_compile` for all new scripts.
- [ ] Run the orchestrator with `--dry-run` and confirm both command sequences, step caps, paths, and GPU assignments.
- [ ] Confirm GPUs 3-6 are idle, Qwen `/v1/models` responds, and neither V4 output directory exists.
- [ ] Launch with `nohup setsid`, redirecting to `artifacts/v4_serial_pipeline/launcher.log`.
- [ ] Monitor through model load, step 1, step 10, and the first evaluation; report PID, observed throughput, and estimated finish time.

The workspace is not a Git repository, so commit steps are intentionally omitted; all changes remain directly in the shared workspace.
