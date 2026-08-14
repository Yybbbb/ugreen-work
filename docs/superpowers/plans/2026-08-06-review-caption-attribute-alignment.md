# Review Caption Attribute Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, resumable script that conservatively aligns review captions with meaningful crop attributes through the local Qwen vLLM service and atomically updates captions in place.

**Architecture:** Pure functions filter attributes, build prompts, parse and validate model responses, and validate cache records. A bounded batch client stores successful sidecars before an in-place materializer updates only `crops[*].caption` using same-directory atomic replacement.

**Tech Stack:** Python 3 standard library, `unittest`, OpenAI-compatible vLLM HTTP API

---

### Task 1: Attribute Policy And Protocol

**Files:**
- Create: `data/scripts/rewrite_review_captions.py`
- Create: `tests/test_rewrite_review_captions.py`

- [ ] Write failing tests asserting that `extra`, empty values, and placeholder
  scalars disappear while meaningful nested values remain; also assert that
  prompts contain original captions and filtered attributes but no image data.
- [ ] Run `python3 -m unittest tests.test_rewrite_review_captions -v` and confirm
  the import fails because the script is absent.
- [ ] Implement `filter_meaningful_attributes`, `build_chat_payload`,
  `parse_rewrites`, and `validate_rewrite`. The validator rejects empty text,
  Markdown, non-English output, implausible length, and negative phrasing.
- [ ] Run the focused tests and confirm the policy/protocol cases pass.

### Task 2: Cache And Atomic In-Place Updates

**Files:**
- Modify: `data/scripts/rewrite_review_captions.py`
- Modify: `tests/test_rewrite_review_captions.py`

- [ ] Add failing temporary-directory tests for cache reuse after an applied
  caption, invalidation after attribute changes, preservation after service
  failure, and equality of every non-caption JSON value.
- [ ] Run the focused tests and confirm the new cache/materialization cases fail.
- [ ] Implement stable crop IDs, atomic JSON sidecars, cache validation,
  same-directory atomic writes, and optimistic caption conflict checks.
- [ ] Run the focused tests and confirm all filesystem cases pass.

### Task 3: Bounded Client And CLI

**Files:**
- Modify: `data/scripts/rewrite_review_captions.py`
- Modify: `tests/test_rewrite_review_captions.py`

- [ ] Add failing fake-requester tests for batch retries, individual fallback,
  partial-success preservation, and nonzero status when tasks remain failed.
- [ ] Implement bounded concurrent batches, retry backoff, individual fallback,
  progress output, `--audit-only`, and `--limit-crops`.
- [ ] Use defaults `http://127.0.0.1:6097/v1`, `Qwen36-35b-caption`, batch size
  16, four workers, low temperature, and disabled thinking.
- [ ] Run `python3 -m unittest tests.test_rewrite_review_captions -v` and
  `python3 -m py_compile data/scripts/rewrite_review_captions.py`.

### Task 4: Dataset And Live-Service Verification

**Files:**
- Verify: `data/scripts/rewrite_review_captions.py`

- [ ] Run `python3 data/scripts/rewrite_review_captions.py --audit-only` and
  confirm the source JSON/Crop/caption counts.
- [ ] Run a controlled live request without publication through the script's
  request helper or a temporary fixture, then validate response parsing and
  attribute alignment.
- [ ] Re-run the complete unit suite and compile check, inspect `--help`, and
  verify that no review JSON changed during tests or the smoke test.
