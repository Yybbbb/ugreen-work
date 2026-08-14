# Reviewed Five-Run SFT Pipeline Design

## Goal

Rebuild the Qwen-derived test-caption attributes from the reviewed test split, then rerun V1, V2, V3, V4A, and V4B with their original training contracts on GPUs 3-6. Training and inference remain serial, while Qwen work on the service hosted by GPUs 1-2 overlaps the next Florence training run.

## Data Contract

The pipeline accepts only the reviewed prepared files under `data/prepared`. Before any destructive artifact operation it verifies the current train, dev, test, and RL counts and SHA-256 values recorded in `data/prepared/metadata.json`.

The test preflight checks `label.strip()` for every row. An empty caption is not sent to Qwen and is not allowed to remain in the evaluation set. If an empty row is found, the pipeline stops before archiving or training and reports the sample IDs; the reviewed source split must then be sanitized and prepared data rebuilt as one consistent transaction. The currently rebuilt test set has 4,328 rows and zero empty captions, so this guard does not change this run.

`data/prepared/test_qwen_attributes.jsonl` and its manifest are rebuilt from a fresh staging path. Existing records are never reused because the old and reviewed test sets overlap only partially. Publication uses atomic replacement after all test rows have successful Qwen results.

## Experiment Contracts

The five experiment names and active output paths stay unchanged. V1 retains its historical `test_inference_final_v2` directory; V2 through V4B retain `test_inference_final`.

| Run | Global batch | Epoch exposure | Max steps | Replay | LR vision/projector/language | Weight decay | Min LR | Eval interval |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| V1 | 64 | 1.0 | 485 | 1,000 | 1e-7 / 5e-7 / 1e-6 | 0.01 | 0.10 | 50 |
| V2 | 32 | 1.0 | 969 | 1,000 | 1e-7 / 5e-7 / 1e-6 | 0.01 | 0.10 | 100 |
| V3 | 32 | 1.5 | 1,454 | 1,000 | 1e-7 / 5e-7 / 1e-6 | 0.01 | 0.10 | 100 |
| V4A | 16 | 3.0 | 5,814 | 1,000 | 2e-7 / 7.5e-7 / 1.25e-6 | 0.015 | 0.03 | 200 |
| V4B | 16 | 3.0 | 5,625 | 0 | 2e-7 / 7.5e-7 / 1.25e-6 | 0.015 | 0.03 | 200 |

All runs use four processes on `CUDA_VISIBLE_DEVICES=3,4,5,6`, per-device batch 4, BF16, label smoothing 0.05, warmup ratio 0.05, six train workers per rank, zero dev workers, prefetch factor 1, max target length 64, and seed 20260720.

## Artifact Replacement

Before launching, the five exact old run directories are atomically renamed under `artifacts/sft/reviewed_retrain_backup_<timestamp>/`. The backup operation is same-filesystem and does not copy model files. The old Qwen GT JSONL and manifest are moved into the same backup root. New artifacts are then written at the historical active paths. Backups remain until the user explicitly removes them.

The pipeline refuses to archive a second time. A restart reads its state file and resumes only the new run checkpoints. This prevents a resumed launcher from mistaking backed-up old completion markers for reviewed results.

## Scheduling

The Florence worker runs `train -> final inference` for each version, in V1 through V4B order. After an inference manifest is complete, it places that run on the Qwen queue and immediately starts the next training run.

The Qwen worker first rebuilds the GT-caption extraction while V1 trains. It then consumes completed inference runs in order: prediction extraction, retry/finalization against manual attributes, and Qwen-GT evaluation. It uses only the existing service at `127.0.0.1:6097`; no additional GPU process is launched.

The final document update runs only after both workers finish all five versions. Each stage writes start, completion, failure, command, timestamps, and log path into an atomically replaced state JSON. A failed stage records the error and stops dependent work.

## Verification And Launch

Unit tests cover exact commands, run order, empty-caption rejection, fresh GT publication, backup idempotence, stage dependency, and Qwen/Florence overlap. A dry run validates all paths and commands without moving artifacts. Immediately before launch the pipeline rechecks Qwen health, GPU allocation, prepared hashes, sample counts, and that no competing Florence job is running.

The real launcher runs detached with a stable PID, launcher log, and state file. Launch is considered successful only after the state shows both Qwen GT extraction and V1 training running and system process/GPU inspection confirms the two resource groups.
