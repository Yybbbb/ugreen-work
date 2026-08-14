# SFT Dev Artifact Archive Design

## Goal

Keep development-set evaluation artifacts for every SFT run under one stable
directory:

```text
artifacts/sft/<run-name>/dev/
  dev_metrics_step_<step>.json
  dev_predictions_step_<step>.jsonl
```

Training checkpoints, `final/`, test inference, Qwen extraction, and all other
run artifacts remain in their current locations.

## Runtime Output

`train_person_attribute_sft.py` will create `<run-dir>/dev` before its first
development evaluation and write both metric and prediction files there. The
filenames and JSON schemas remain unchanged, so the only contract change is
their parent directory.

## Historical Archive

A dedicated script will scan direct children of `artifacts/sft`. For each run,
it moves only root-level files matching these patterns into `dev/`:

- `dev_metrics_step_*.json`
- `dev_predictions_step_*.jsonl`

The script supports a dry-run mode and prints a per-run summary. A real run
uses same-filesystem atomic renames. It does not recursively scan or modify
checkpoint directories and it does not touch `test_inference*` directories.

## Collision Rules

If a destination path already exists, the script compares file sizes and a
streaming SHA-256 digest. Identical files are treated as already archived and
the redundant root file is removed. Different files cause the script to fail
without overwriting either file. This keeps the operation repeatable and
prevents accidental data loss.

## Verification

Automated tests will cover the training output path and archive behavior:

1. development output helpers resolve inside `run/dev`;
2. dry-run does not modify files;
3. a real archive moves matching root files only;
4. a repeated archive is a no-op; and
5. a conflicting destination is rejected without overwrite.

The completed historical archive will be verified by checking that no matching
development artifact remains directly under a run root and that the expected
178 files exist under their corresponding `dev/` directories.
