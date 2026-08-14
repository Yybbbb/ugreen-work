# Florence RL Training Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Florence SCST trainer use V4B, write under `artifacts/rl`, resume complete training state, evaluate dev reward during training, retain top-3 checkpoints, and always retain `final/`.

**Architecture:** Extend the RL trainer with small pure helpers for configuration, checkpoint ranking, dev aggregation, and run-directory resolution, while reusing the SFT trainer's checkpoint/RNG/loss utilities. Keep service-dependent reward evaluation on rank 0 and full CE evaluation distributed across all ranks.

**Tech Stack:** Python 3.10, PyTorch DDP, Florence2/Transformers, JSON/JSONL, unittest/pytest, OpenAI-compatible vLLM reward services.

---

### Task 1: Correct Formal Defaults And Alpha Schedule

**Files:**
- Modify: `tests/test_train_person_attribute_rl.py`
- Modify: `scripts/train_person_attribute_rl.py`

- [ ] Add failing tests that assert V4B is `DEFAULT_SFT_FINAL`, `DEFAULT_OUTPUT_ROOT` is `artifacts/rl`, `ALPHA_START=0.10`, `ALPHA_END=0.50`, and CLI defaults expose both alpha values.
- [ ] Run `miniconda3/envs/florence/bin/python -m pytest tests/test_train_person_attribute_rl.py -q` and confirm failures are caused by the old defaults.
- [ ] Define the RL-specific output root and V4B checkpoint, add `--alpha-start` and `--alpha-end`, validate `0 <= start <= end <= 1`, and pass these values to `alpha_at_step`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Add Deterministic Dev Reward Artifacts And Ranking

**Files:**
- Modify: `tests/test_train_person_attribute_rl.py`
- Modify: `scripts/train_person_attribute_rl.py`

- [ ] Add failing pure-helper tests for step-specific dev paths, reward/count aggregation, extractor failures, and ranking by descending reward then ascending dev loss and step.
- [ ] Run the focused tests and verify they fail because the helpers do not exist.
- [ ] Implement dev artifact paths, per-caption reward records, aggregate metrics, and top-3 ranking helpers.
- [ ] Add full-dev distributed CE evaluation by reusing `_evaluate_loss`, a dev DataLoader/Sampler, and rank-0 greedy generation for `--eval-generation-samples` rows.
- [ ] Save prediction JSONL and summary JSON under `<run>/dev`, then select and prune top-3 evaluation checkpoints.
- [ ] Run focused tests and compile the trainer.

### Task 3: Implement Complete Resume And Final State

**Files:**
- Modify: `tests/test_train_person_attribute_rl.py`
- Modify: `scripts/train_person_attribute_rl.py`

- [ ] Add failing tests for resume run-directory inference, explicit-name mismatch, completed/capped final metadata, and expanded immutable configuration.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Build complete invariants, write `run_config.json`, and reuse `_load_resume_state`, `_gather_rng_states`, and `_save_checkpoint` for optimizer/scheduler/scaler/progress/top-3/RNG restoration.
- [ ] Resume from saved epoch and micro-batch, preserve scheduler/alpha global step, coordinate checkpoint barriers, and reject mismatched resume configuration.
- [ ] Always save `final/` separately with `completed` and `stopped_by_max_steps` metadata; never include it in top-3 pruning.
- [ ] Run focused tests and compile checks.

### Task 4: Align Documentation And Verify

**Files:**
- Modify: `EXPERIMENT_PLAN.md`
- Modify: `docs/FLORENCE_TRAINING_RUNS.md`

- [ ] Update alpha to `0.10 -> 0.50`, V4B initialization, RL output root, full resume semantics, dev evaluation, top-3 selection, and retained final checkpoint.
- [ ] Run all RL tests plus the SFT helper tests used by the trainer.
- [ ] Run Python compile checks and `--validate-data-only` in the Florence environment.
- [ ] Inspect CLI help/defaults and confirm no formal RL process was started.

The workspace is not a Git repository, so commit steps are omitted.
