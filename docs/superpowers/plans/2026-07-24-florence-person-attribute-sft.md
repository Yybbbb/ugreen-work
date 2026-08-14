# Florence2 Person Attribute SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic person/replay JSONL preprocessing and a 1-GPU/4-GPU full-parameter Florence2 SFT trainer without running the full 31,000-sample training job.

**Architecture:** Convert the existing split manifests and per-frame JSON files into self-contained JSONL rows that reference full images and native Florence region prompts. Generate a fixed 1,000-row reserve replay selection separately from model inference, then train through a torchrun-compatible DDP entry point with three learning-rate groups and resumable checkpoints.

**Tech Stack:** Python 3.8+, unittest, Pillow, PyTorch, Transformers 4.41.x, safetensors, tqdm.

---

### Task 1: Shared person SFT data primitives

**Files:**
- Create: `scripts/person_sft_data.py`
- Create: `tests/test_person_sft_data.py`

- [ ] **Step 1: Write failing bbox and JSONL tests**

```python
class BboxTests(unittest.TestCase):
    def test_quantize_bbox_uses_floor_and_clamp(self):
        self.assertEqual(
            quantize_bbox([-4.0, 50.9, 640.9, 900.0], 1280, 720),
            [0, 70, 500, 999],
        )

    def test_build_prompt_uses_native_task_and_four_locations(self):
        self.assertEqual(
            build_region_prompt("<REGION_TO_CATEGORY>", [1, 2, 3, 4]),
            "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>",
        )
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_person_sft_data -v`

Expected: import failure because `scripts/person_sft_data.py` does not exist.

- [ ] **Step 3: Implement the shared functions**

```python
def quantize_bbox(bbox, width, height):
    clipped = clip_bbox(bbox, width, height)
    return [
        quantize_coordinate(clipped[0], width),
        quantize_coordinate(clipped[1], height),
        quantize_coordinate(clipped[2], width),
        quantize_coordinate(clipped[3], height),
    ]

def build_region_prompt(task, bbox_loc):
    if task not in {"<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"}:
        raise ValueError(f"Unsupported task: {task}")
    return task + "".join(f"<loc_{value}>" for value in validate_loc(bbox_loc))

def write_jsonl_atomic(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
```

Also implement `iter_jsonl`, `sha256_file`, finite-number validation, positive-area clipping, duplicate-ID rejection, and stable compact JSON serialization.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m unittest tests.test_person_sft_data -v`

Expected: all shared data tests pass.

### Task 2: Prepare person train/dev/test/RL JSONL

**Files:**
- Create: `scripts/prepare_person_sft_data.py`
- Create: `tests/test_prepare_person_sft_data.py`

- [ ] **Step 1: Write a failing fixture-based mapping test**

Create a temporary manifest and frame JSON with two crops. Assert that `prepare_split()` uses stable `crop_index` (because split materialization compacts the crop list), preserves original `crop_position` in the sample ID, uses `expanded_bbox_xyxy`, recomputes loc tokens, preserves attributes, and emits `<REGION_TO_CATEGORY>`.

```python
rows = prepare_split(manifest, split_root, "train")
self.assertEqual(rows[0]["sample_id"], "scene/frame.json#1")
self.assertEqual(rows[0]["bbox_loc_0_999"], [100, 200, 600, 800])
self.assertEqual(rows[0]["label"], "An adult male wears a blue jacket.")
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_prepare_person_sft_data -v`

Expected: import failure because the preparation module is absent.

- [ ] **Step 3: Implement strict frame-to-crop conversion**

```python
def prepare_manifest_row(manifest_row, split_root, split):
    source_relative = validate_relative_path(manifest_row["source_json_relative_path"])
    document = load_json(split_root / source_relative)
    crop = document["crops"][manifest_row["crop_position"]]
    if crop["caption"].strip() != manifest_row["caption"].strip():
        raise ValueError(f"Caption mismatch for {manifest_row['sample_id']}")
    bbox = clip_bbox(crop["expanded_bbox_xyxy"], image["width"], image["height"])
    loc = quantize_bbox(bbox, image["width"], image["height"])
    return build_person_row(manifest_row, document, crop, bbox, loc, split)
```

The CLI defaults to the repository `data` directory, processes `train dev test rl`, and writes `data/prepared/<split>.jsonl` plus `metadata.json`. It must reject count mismatches against 30,000/2,000/4,328/6,000 when using default full inputs.

- [ ] **Step 4: Verify GREEN and run full preprocessing**

Run:

```bash
python3 -m unittest tests.test_prepare_person_sft_data -v
python3 scripts/prepare_person_sft_data.py
wc -l data/prepared/{train,dev,test,rl}.jsonl
```

Expected counts: 30,000; 2,000; 4,328; 6,000.

### Task 3: Deterministic reserve replay selection and generation

**Files:**
- Create: `scripts/generate_region_replay.py`
- Create: `tests/test_generate_region_replay.py`

- [ ] **Step 1: Write failing selection and resume tests**

Use temporary reserve manifests with repeated frame paths and two scenes. Assert exact per-scene quota, no repeated frame, stable IDs across reruns, task replacement with `<REGION_TO_DESCRIPTION>`, and progress resume that calls the generator only for missing IDs.

```python
selected = select_replay_candidates(rows, {"scene_a": 2, "scene_b": 2}, seed=7)
self.assertEqual(Counter(row["scene"] for row in selected), {"scene_a": 2, "scene_b": 2})
self.assertEqual(len({row["source_json_relative_path"] for row in selected}), 4)
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_generate_region_replay -v`

Expected: import failure because replay generation is absent.

- [ ] **Step 3: Implement selection, model loading, and resumable inference**

Selection hydrates reserve manifest rows through the same strict conversion as Task 2, changes only the task/prompt, and defaults to 500 rows per reserve scene. Model imports are lazy so `--selection-only` works without torch.

```python
def generate_missing(selection, completed, generate_one, append_progress):
    seen = set(completed)
    for row in selection:
        if row["sample_id"] in seen:
            continue
        label, raw_text = generate_one(row)
        if not label.strip():
            raise ValueError(f"Empty replay label: {row['sample_id']}")
        append_progress({**row, "label": label.strip(), "raw_generation": raw_text})
```

The real generator loads local Florence2-base with `trust_remote_code=True`, patches only the optional `flash_attn` import check, uses eager attention and greedy generation, then calls `post_process_generation(task="<REGION_TO_DESCRIPTION>")`.

- [ ] **Step 4: Verify GREEN and materialize selection only**

Run:

```bash
python3 -m unittest tests.test_generate_region_replay -v
python3 scripts/generate_region_replay.py --selection-only
wc -l data/prepared/native_replay_selection.jsonl
```

Expected: tests pass and selection contains 1,000 rows, 500 per scene, with unique frames.

- [ ] **Step 5: Run one real replay inference if a compatible environment exists**

Run: `python scripts/generate_region_replay.py --max-new-samples 1`

Expected: one non-empty progress row. If no torch/transformers environment exists, record that dependency limitation and retain fake-generator test coverage.

### Task 4: Full-parameter DDP trainer

**Files:**
- Create: `scripts/train_person_attribute_sft.py`
- Create: `tests/test_train_person_attribute_sft.py`

- [ ] **Step 1: Write failing pure-logic trainer tests**

Test optimizer-step rounding, tail accumulation size, parameter role classification, no-decay classification, generation text metrics, and replay/person deterministic mixing without importing torch.

```python
self.assertEqual(optimizer_steps(31000, world_size=4, per_device_batch=4, accumulation=4), 485)
self.assertEqual(accumulation_group_size(3874, 3875, 4), 3)
self.assertEqual(parameter_role("vision_tower.blocks.0.weight"), "vision")
self.assertEqual(parameter_role("image_projection"), "projection")
self.assertEqual(parameter_role("language_model.encoder.layers.0.weight"), "language")
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_train_person_attribute_sft -v`

Expected: import failure because the trainer is absent.

- [ ] **Step 3: Implement import-safe helpers and CLI validation**

The module imports torch/transformers only inside `load_training_runtime()` and `train()`. Standard-library helpers implement exact step math, metric calculation, JSONL hashing, run configuration, and parameter-name routing. Formal runs require exactly 1,000 replay rows; `--person-only` is an explicit smoke-test-only switch and is recorded in run metadata.

- [ ] **Step 4: Implement Florence loading and data collation**

Load processor with the fast tokenizer from the local checkpoint. Load `model.safetensors` through `AutoModelForCausalLM.from_config()` plus safetensors state dict, then assert no missing/unexpected keys except documented tied weights. Dataset rows open full images and collate prompts, pixels and tokenized labels with pad IDs replaced by `-100`.

- [ ] **Step 5: Implement DDP training, dev, and checkpoints**

Initialize NCCL when launched by torchrun, use `DistributedSampler`, three role-specific learning rates with decay/no-decay subgroups, BF16 autocast, label-smoothed cross entropy, `no_sync()` during accumulation, cosine scheduling and global-norm clipping.

At eval intervals, all ranks reduce dev loss; rank 0 generates the fixed dev subset and saves metrics/predictions. Save top three dev-loss checkpoints and final, including optimizer/scheduler/scaler/RNG/config/data hashes. `--resume-from` validates the saved invariant fields before restoring.

- [ ] **Step 6: Verify GREEN and static CLI behavior**

Run:

```bash
python3 -m unittest tests.test_train_person_attribute_sft -v
python3 scripts/train_person_attribute_sft.py --help
python3 scripts/train_person_attribute_sft.py --validate-data-only --person-only
```

Expected: pure tests pass; help works without model load; person-only validation reports 30,000 person rows. A formal validation without `--person-only` must reject a missing or incomplete 1,000-row replay file.

- [ ] **Step 7: Run a one-step GPU smoke test if dependencies and memory are available**

Run:

```bash
python3 scripts/train_person_attribute_sft.py \
  --max-train-samples 4 --max-dev-samples 2 \
  --max-optimizer-steps 1 --num-workers 0 --no-save --person-only
```

Expected: one finite optimizer step. Do not launch the complete 31,000-sample run.

### Task 5: Checkpoint validation and final verification

**Files:**
- Create: `scripts/validate_sft_checkpoint.py`
- Create or modify: `tests/test_train_person_attribute_sft.py`

- [ ] **Step 1: Write a failing processor contract test**

```python
expanded = construct_prompt(processor, "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>")
self.assertEqual(expanded, "What is the region <loc_1><loc_2><loc_3><loc_4>?")
self.assertEqual(processor.tasks_answer_post_processing_type["<REGION_TO_CATEGORY>"], "pure_text")
```

- [ ] **Step 2: Implement checkpoint validator**

The script loads processor and model in a fresh process, validates the prompt contract, checks that no tokenizer tokens were added, optionally performs one greedy generation, and emits machine-readable JSON.

- [ ] **Step 3: Run all available verification**

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
python3 scripts/prepare_person_sft_data.py --validate-existing
python3 scripts/generate_region_replay.py --selection-only --validate-existing
```

Expected: all tests pass, all Python files compile, prepared counts match manifests, replay selection is exactly 1,000 unique-frame rows with a 500/500 scene split, and no full training process is running.

Because `/data1/work/MichaelYu/florence-attibute` is not a Git repository, commit steps are intentionally omitted. Verification output and generated metadata provide the implementation audit trail.
