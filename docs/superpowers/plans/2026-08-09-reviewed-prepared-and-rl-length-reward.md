# Reviewed Prepared Data And RL Length Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild prepared data from the reviewed manifests without topping up RL, and change SCST length scoring to reward 18-24 word captions.

**Architecture:** Generate requested prepared splits in a sibling staging directory, validate the complete staged set, and only then atomically replace active files and metadata. Keep reward math pure and unit-tested; derive the formal RL row contract from the reviewed 5,981-row dataset.

**Tech Stack:** Python 3.10 standard library, JSONL, `unittest`, Florence2 training scripts.

---

### Task 1: Make prepared rebuild safe for reviewed counts

**Files:**
- Modify: `scripts/prepare_person_sft_data.py`
- Modify: `tests/test_prepare_person_sft_data.py`

- [ ] Add tests asserting the reviewed RL expected count is 5,981 and a failed multi-split rebuild leaves existing output files unchanged.
- [ ] Run `python -m unittest tests.test_prepare_person_sft_data -v` and confirm the new tests fail for the current direct-publication implementation.
- [ ] Change `EXPECTED_FULL_COUNTS["rl"]` to `5981`; generate requested files and metadata in a temporary sibling staging directory, validate them, then publish each file with `os.replace`; always clean staging in `finally`.
- [ ] Run the focused preparation tests and confirm they pass.

### Task 2: Implement signed length reward and reviewed RL contract

**Files:**
- Modify: `tests/test_rl_reward.py`
- Modify: `tests/test_train_person_attribute_rl.py`
- Modify: `scripts/rl_reward.py`
- Modify: `scripts/train_person_attribute_rl.py`

- [ ] Replace length tests with boundary assertions for score `+1.0` at 18-24, `-0.5` at 12-17 and 25-28, and `-1.0` outside; add reward-delta tests and a 5,981-row/8-global-batch step test expecting 748.
- [ ] Run the focused RL tests and confirm failures occur because `length_score` and the 5,981-row formal contract do not exist.
- [ ] Implement `length_score`, use `+ W_LENGTH * length_score`, require 5,981 formal RL rows, and leave unknown-to-concrete classification as fabrication.
- [ ] Run all RL tests and confirm they pass.

### Task 3: Align experiment documentation

**Files:**
- Modify: `EXPERIMENT_PLAN.md`

- [ ] Update RL data count to 5,981, optimizer steps to 748, unknown semantics to mean absent, and the signed three-band length reward formula.
- [ ] Search the RL sections for contradictory 6,000/750/unknown/length statements and resolve them without rewriting historical SFT records.

### Task 4: Rebuild and verify reviewed prepared data

**Files:**
- Replace through script: `data/prepared/{train,dev,test,rl}.jsonl`
- Replace through script: `data/prepared/metadata.json`

- [ ] Run the full preparation unit tests and Python compile checks in `/data1/work/MichaelYu/miniconda3/envs/florence`.
- [ ] Run `scripts/prepare_person_sft_data.py --overwrite` to build and publish all four reviewed splits.
- [ ] Validate prepared files against manifests, compare sample-ID sets, assert RL is a subset of train, and verify metadata hashes.
- [ ] Run the complete relevant unit suite for preparation, reward, clients, and RL trainer.
- [ ] Confirm no SFT or RL training process was started.

The workspace is not a Git repository, so commit steps are omitted.
