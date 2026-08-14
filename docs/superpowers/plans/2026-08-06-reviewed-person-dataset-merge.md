# Reviewed Person Dataset Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deterministically merge regular and Xiaohongshu reviewed person crops into the current train/test/dev/rl dataset while enforcing review provenance, caption leakage limits, and split-size constraints.

**Architecture:** Add one self-contained dataset migration script with pure functions for identity, status normalization, text similarity, allocation, crop movement, validation, and staged publication. Unit tests exercise those functions with temporary JSON trees; the production run first generates a dry-run report and only publishes when every invariant passes.

**Tech Stack:** Python 3.13, standard library, NumPy/SciPy, scikit-learn TF-IDF, unittest/pytest-compatible tests.

---

### Task 1: Review normalization and crop identity

**Files:**
- Create: `scripts/merge_reviewed_person_dataset.py`
- Create: `tests/test_merge_reviewed_person_dataset.py`

- [x] **Step 1: Write failing tests for normalized review state and stable identity**

```python
def test_normalize_review_status_supports_current_legacy_and_aliases():
    assert mod.normalize_review_status({"human_review": {"status": "approved"}}) == "reviewed"
    assert mod.normalize_review_status({"review": {"status": "duplicate", "updated_at": "now"}}) == "duplicate"
    assert mod.normalize_review_status({"review": {"status": "reviewed"}}) == "unreviewed"

def test_crop_key_prefers_source_position_and_falls_back_to_index_bbox():
    assert mod.crop_key({"source_crop_position": 3}, 7) == ("position", 3)
    key = mod.crop_key({"crop_index": 2, "bbox_xyxy": [1, 2, 3, 4]}, 7)
    assert key == ("index_bbox", 2, (1.0, 2.0, 3.0, 4.0))
```

- [x] **Step 2: Run the focused tests and verify they fail**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: import failure because `scripts/merge_reviewed_person_dataset.py` does not exist.

- [x] **Step 3: Implement status normalization, JSON traversal, frame identity, and crop-key functions**

Implement these public functions with explicit validation:

```python
normalize_review_status(crop: dict) -> str
iter_json_paths(root: Path) -> Iterator[Path]
frame_key(record: dict, relative_path: Path) -> str
crop_key(crop: dict, position: int) -> tuple
```

Use aliases `approved/corrected -> reviewed` and `needs_revision -> uncertain`.
Only use legacy `review` when `updated_at` is present. Reject duplicate crop keys
within one frame rather than guessing.

- [x] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: all Task 1 tests pass.

### Task 2: Authoritative/rewrite overlay and regular review actions

**Files:**
- Modify: `scripts/merge_reviewed_person_dataset.py`
- Modify: `tests/test_merge_reviewed_person_dataset.py`

- [x] **Step 1: Add failing tests for caption overlay and regular actions**

```python
def test_reviewed_crop_uses_authoritative_fields_and_rewritten_caption():
    merged = mod.overlay_rewritten_caption(authoritative, rewritten)
    assert merged["caption"] == rewritten["caption"]
    assert merged["attributes"] == authoritative["attributes"]
    assert merged["dimensions"] == authoritative["dimensions"]
    assert merged["review_status"] == "reviewed"

def test_regular_actions_move_reviewed_and_delete_test_rejected_duplicate():
    result = mod.apply_regular_actions(dataset, reviewed_records)
    assert reviewed_key not in result["train"]
    assert reviewed_key in result["test"]
    assert rejected_key not in result["test"]
    assert duplicate_key not in result["test"]
```

- [x] **Step 2: Run tests and verify the new cases fail**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: missing overlay/action functions.

- [x] **Step 3: Implement review indexes and crop-level actions**

Load authoritative records into regular and Xiaohongshu indexes. Load
`data/review` into a rewritten-caption index using the same frame/crop keys.
For regular reviewed crops, remove all old split copies and create one test
copy. For regular rejected or duplicate crops, delete only matching test crops.
Strip `review_status` from every non-reviewed crop.

- [x] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: all Task 1-2 tests pass.

### Task 3: Caption similarity and Xiaohongshu allocation

**Files:**
- Modify: `scripts/merge_reviewed_person_dataset.py`
- Modify: `tests/test_merge_reviewed_person_dataset.py`

- [x] **Step 1: Add failing tests for normalization, similarity components, and grouped allocation**

```python
def test_caption_similarity_ignores_case_punctuation_and_uses_threshold():
    pairs = mod.high_similarity_pairs(captions, threshold=0.90)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs

def test_xiaohongshu_allocator_never_splits_video_or_similarity_component():
    allocation = mod.allocate_xiaohongshu(samples, regular_test, threshold=0.90)
    assert allocation["video_a"] == allocation["video_b"]
    assert abs(allocation.train_crops - allocation.test_crops) <= largest_component
```

- [x] **Step 2: Run tests and verify the new cases fail**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: missing similarity/allocation functions.

- [x] **Step 3: Implement sparse TF-IDF comparison and deterministic component allocation**

Use `TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), token_pattern=r"(?u)\\b\\w+\\b")`. Build sparse cosine blocks so production comparison does not materialize a dense 30k-by-test matrix. Use union-find to combine Xiaohongshu videos connected by a similarity score at least 0.90. Pin components conflicting with regular reviewed test captions to test, then deterministically assign remaining components to minimize distance from a 50/50 crop target.

- [x] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: all Task 1-3 tests pass.

### Task 4: Leakage removal, split sizing, and invariants

**Files:**
- Modify: `scripts/merge_reviewed_person_dataset.py`
- Modify: `tests/test_merge_reviewed_person_dataset.py`

- [x] **Step 1: Add failing tests for leakage priority, size caps, and 4 percent aborts**

```python
def test_train_test_leakage_removes_only_non_reviewed_train_crop():
    result = mod.remove_train_test_leakage(train, test, threshold=0.90)
    assert non_reviewed_conflict not in result
    assert reviewed_sample in result

def test_reduce_split_never_removes_reviewed_crop():
    reduced = mod.reduce_to_limit(samples, limit=2)
    assert reviewed_sample in reduced
    assert len(reduced) == 2

def test_validate_counts_rejects_more_than_four_percent_shortfall():
    with self.assertRaises(mod.PublicationBlocked):
        mod.validate_counts({"train": 28799}, {"train": 30000}, tolerance=0.04)
```

- [x] **Step 2: Run tests and verify the new cases fail**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: missing leakage, reduction, and validation functions.

- [x] **Step 3: Implement deterministic reduction and complete validation**

Remove non-reviewed train crops that match any finalized test caption at 0.90.
When over a split limit, rank removable crops by within-split duplicate score,
confidence ascending, crop area ascending, then stable identity. Validate JSON
counts, reviewed provenance, rejected/duplicate absence from test, no cross-split
similarity, no Xiaohongshu group leakage, RL subset membership, upper bounds,
and the 4 percent shortfall threshold.

- [x] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: all Task 1-4 tests pass.

### Task 5: Staging, manifests, dry-run, and atomic publication

**Files:**
- Modify: `scripts/merge_reviewed_person_dataset.py`
- Modify: `tests/test_merge_reviewed_person_dataset.py`

- [x] **Step 1: Add failing integration test with temporary split trees**

```python
def test_build_staging_preserves_paths_filters_crops_and_does_not_publish_on_failure():
    report = mod.build_staging(config)
    assert (staging / "test" / "scene" / "frame.json").exists()
    assert original_train.read_bytes() == original_bytes
    mod.validate_staging(staging, report)
```

- [x] **Step 2: Run tests and verify the integration case fails**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: missing staging functions.

- [x] **Step 3: Implement CLI and staged materialization**

Support `--dry-run` and `--publish`. Dry-run must write only a JSON report under
`data/.reviewed_merge_work`. Publish must build complete staged split trees,
regenerate `data/manifests/{train,test,dev,rl,reserve}.jsonl` and
`data/split_statistics.json`, validate, then swap active directories. If a swap
fails, restore all original directories before raising.

- [x] **Step 4: Run the complete automated test suite**

Run: `python -m unittest tests.test_merge_reviewed_person_dataset -v`

Expected: all tests pass.

Run: `python -m unittest discover -s tests -v`

Expected: existing and new tests pass, except any explicitly documented
environment-only tests unrelated to this data migration.

### Task 6: Production dry-run and publication

**Files:**
- Generated: `data/.reviewed_merge_work/dry_run_report.json`
- Modify through validated publication: `data/train`, `data/test`, `data/dev`, `data/rl`, `data/reserve`, `data/manifests`, `data/split_statistics.json`

- [x] **Step 1: Run production dry-run**

Run:

```bash
python scripts/merge_reviewed_person_dataset.py \
  --dataset-root /data1/work/MichaelYu/florence-attibute/data \
  --review-root /data0/work/WangHaoxiang/a35_vehicle_label/outputs/person_reviewed/frames \
  --rewrite-root /data1/work/MichaelYu/florence-attibute/data/review \
  --threshold 0.90 --shortfall-tolerance 0.04 --dry-run
```

Expected: report includes exact before/after split counts, status action counts,
Xiaohongshu allocation counts, leakage removals, and zero validation errors.

- [x] **Step 2: Stop for user guidance if any split shortfall exceeds 4 percent**

Do not run `--publish` when the dry-run exits with `PublicationBlocked`. Report
the exact split deficits and the dominant deletion causes.

- [x] **Step 3: Publish only after a clean dry-run**

Run the same command with `--publish` instead of `--dry-run`.

Expected: validated staged directories replace active split directories and the
script prints the final report path.

- [x] **Step 4: Independently recount and validate the published dataset**

Run the script with `--validate-only`, then independently count JSON files,
crops, `review_status`, regular rejected/duplicate test matches, cross-split
caption similarity, and RL subset membership.

Expected: all invariants pass and counts match the published report.

## Repository Constraint

`/data1/work/MichaelYu/florence-attibute` is not a Git repository. Commit steps
are intentionally omitted; no commit can be created without changing repository
ownership or initializing new version-control metadata.

