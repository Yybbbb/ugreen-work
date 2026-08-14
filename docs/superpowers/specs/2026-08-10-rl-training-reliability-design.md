# Florence RL Training Reliability Design

## Goal

Make the Florence person-attribute SCST trainer ready for the first formal RL
run by using the selected V4B SFT checkpoint, storing RL artifacts separately,
raising the SCST schedule to `alpha=0.10 -> 0.50`, supporting complete outage
recovery, and selecting checkpoints with an in-training dev evaluation.

This change prepares and verifies the trainer. It does not start formal RL
training or start, stop, or replace either reward service.

## Fixed Training Contract

The first formal run uses:

- initialization checkpoint:
  `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final`;
- output root: `artifacts/rl`;
- reviewed RL train data: `data/prepared/rl.jsonl`, 5,981 rows;
- reviewed dev data: `data/prepared/dev.jsonl`, 2,000 rows;
- four ranks, per-device batch 2, accumulation 1, global batch 8;
- one data pass and 748 optimizer steps;
- `alpha_start=0.10` and `alpha_end=0.50`, linearly interpolated over the
  configured optimizer-step horizon;
- the existing optimizer learning rates, sampling temperature, top-p, and
  merged soft-F1 reward unless explicitly overridden by CLI arguments;
- either `qwen` or `lexical` reward matching through the existing
  `--reward-matcher` switch.

The CE term remains the language anchor. The larger final alpha gives SCST
more influence than the previous `0.05 -> 0.30` schedule without removing the
teacher-forced caption objective.

## Configuration And Provenance

The trainer writes `run_config.json` before training. The configuration and
checkpoint invariants include:

- resolved initialization checkpoint and train/dev paths with SHA-256 hashes;
- world size, batch sizes, accumulation, epochs, step counts, and seed;
- optimizer, scheduler, precision, and label-smoothing settings;
- `alpha_start`, `alpha_end`, generation temperature, top-p, and token limits;
- reward matcher, extractor endpoint/model, and judge endpoint/model;
- reward weights and the reward/judge/extractor prompt identifiers available
  from the local code.

Resume rejects a checkpoint when these invariants differ. Service credentials
are not written. Endpoint URLs are recorded because changing the endpoint can
change the served reward model.

## Complete Resume Semantics

Every saved checkpoint contains model and processor files plus
`training_state.pt` with:

- optimizer, scheduler, and GradScaler states;
- completed optimizer step;
- current epoch and next micro-batch index;
- retained top-3 checkpoint metadata;
- Python, CPU torch, and CUDA RNG state for every rank;
- the immutable run configuration.

`--resume-from PATH` loads model and processor weights from `PATH`, restores all
training state, sets the distributed sampler to the saved epoch, skips already
consumed micro-batches, and resumes alpha and scheduler progression from the
saved global step.

When `--run-name` is omitted, the trainer infers the original run directory
from the checkpoint parent. When it is supplied, it must resolve to that same
run directory. This prevents a resumed run from silently writing into a new
experiment. All ranks participate in RNG collection and checkpoint barriers.

An evaluation checkpoint is written every `eval_steps`. A power interruption
can therefore lose at most the work since the latest successful evaluation
checkpoint. Existing checkpoint directories are never overwritten.

## Dev Evaluation

At each evaluation step:

1. All ranks compute teacher-forced CE over all 2,000 dev rows and reduce the
   numerator and token/sample denominator to one global dev loss.
2. Rank 0 greedily generates captions for a deterministic 128-row prefix by
   default. `--eval-generation-samples` controls this count without changing
   the formal training data contract.
3. The captions pass through the configured attribute extractor and the same
   matcher used by training. The lexical route does not contact the judge; the
   qwen route uses the configured five-level judge. Extractor calls use batches
   of four captions by default so fixed-schema JSON fits the response budget.
4. The evaluator records each caption, extracted attributes, reward, reward
   counts, ground-truth attributes, and identifying metadata.
5. It writes step-specific JSONL predictions and JSON summary metrics under
   `<run>/dev/`.

Summary metrics include global dev CE, mean/min/max greedy reward, mean soft-F1,
mean fabrication ratio, average word count, 18-24 word ratio, P95 length,
single-sentence ratio, subject-start ratio, background-keyword ratio, empty
ratio, and extractor failure count.

An extractor request failure after configured retries fails the evaluation and
does not publish a checkpoint. Missing or malformed individual extraction
items are counted as failures and conservatively scored as empty extracted
attributes.

## Checkpoint Selection And Final Model

Evaluation checkpoints are ranked by:

1. higher mean greedy dev reward;
2. lower full-dev CE loss as the tie breaker;
3. earlier global step as the deterministic final tie breaker.

Only the best three evaluation checkpoints are retained. Their ranking is
stored in every subsequent checkpoint so it survives resume.

The trainer always writes a separate `<run>/final/` checkpoint after a clean
training-loop exit. `final/` is retained in addition to the top three and is
never removed by checkpoint pruning. It contains the same complete recovery
state and records whether the run reached the configured total step count or
stopped at `--max-optimizer-steps`. A fully completed `final/` is a terminal
checkpoint; attempting to resume it reports that no training remains.

## Distributed Coordination

All ranks participate in full-dev loss evaluation, RNG-state gathering, and
barriers. Rank 0 alone performs greedy reward evaluation and filesystem writes
while other ranks wait at a barrier. Training resumes only after evaluation
and checkpoint publication complete, preventing ranks from entering the next
DDP forward pass while rank 0 is still saving.

## Tests And Verification

Unit tests cover:

- V4B and `artifacts/rl` defaults;
- alpha endpoints and interpolation for `0.10 -> 0.50`;
- immutable configuration contents;
- top-3 ordering and deterministic tie breaking;
- dev reward aggregation, caption metrics, and extractor-failure handling;
- resume run-directory inference and mismatch rejection;
- restoration of epoch, batch offset, global step, optimizer, scheduler,
  scaler, top-3 state, and per-rank RNG state;
- final checkpoint metadata for complete and step-capped runs;
- the existing 5,981-row/748-step formal contract.

Verification uses the Florence Python 3.10 environment and includes focused RL
tests, the related SFT helper tests, compile checks, data-only validation, and a
non-formal short smoke run only when the required reward service is available.
No formal RL run is part of implementation verification.
